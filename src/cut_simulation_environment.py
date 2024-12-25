import taichi as ti
import numpy as np
import os
import math
import trimesh
from typing import List, Tuple, Callable, Dict
import logging
import time

from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

import yourdfpy

from sksparse.cholmod import cholesky
from scipy.sparse import coo_matrix, csc_matrix

from src.cloth import Cloth
from src.collision import ClothScissorCollision, ClothEndeffectorCollision
from src.cutting import Cutting
from src.robot import Robot
from src.endeffector import EndEffector

from src.mpcg import *
from src.scissor import *
from src.maths import *
from src.utils import *

CG_SOLVER = 1
CHOLMOD_SOLVER = 2

CONSTRAINT_RELATIVE_MASS = 1e6
CG_RELATIVE_A_EPS = 1e-7


@ti.dataclass
class I32F32:
    i: ti.i32
    f: ti.f32


@ti.data_oriented
class CutSimulationEnvironment:
    def __init__(self, cfg: DictConfig, log: logging.Logger) -> None:
        # init cut_sim
        self.batch_size: int = cfg.env.batch_size

        self.use_ascii = cfg.output.use_ascii
        self.print_info = cfg.output.print_info
        self.print_stuck = cfg.output.print_stuck

        self.dim = 3
        self.gravity = ti.field(dtype=ti.f32, shape=3)
        self.gravity.from_numpy(np.array(cfg.env.cut_sim.get(
            'gravity', [0.0, 0.0, -9.8])).astype(np.float32))

        self.frame_dt = cfg.env.cut_sim.get('frame_dt', 5e-2)
        self.dt = cfg.env.cut_sim.get('dt', 1e-2)
        self.substep_n = 0

        self.vel_damping1 = cfg.env.cut_sim.get(
            'vel_damping1', 0.0)  # dv=-k*v*dt
        self.vel_damping2 = cfg.env.cut_sim.get(
            'vel_damping2', 0.0)  # dv=-k*v*v*dt

        self.position_bounds = ti.Vector.field(n=3, dtype=ti.f32, shape=(2))
        self.position_bounds.from_numpy(
            np.array(cfg.env.cut_sim.position_bounds).astype(np.float32))

        self.max_velocity = cfg.env.cut_sim.max_velocity

        self.use_implicit_integral = cfg.env.cut_sim.get(
            'use_implicit_integral', True)

        self.topo_check = cfg.env.cut_sim.get('topo_check', True)
        self.log = log
        self.log_detail = cfg.output.log.detail
        self.log_time = cfg.output.log.time

        ######################################################################
        ########                        Objects                       ########
        # init cloth
        self.cloth = Cloth(self.batch_size, cfg.cloth, cfg.output)

        # init scissor
        self.scissors = Scissor(self.batch_size, cfg.scissor, cfg.output)
        self.scissors_old_pose = [copy.deepcopy(self.scissors.get_init_pose()) for _ in range(self.batch_size)]  # around joint_0

        # init robot
        self.use_robot = cfg.robot.use_robot
        if self.use_robot:
            self.robot = Robot(cfg.robot)
            self.ee_joint = cfg.robot.ee_joint
        
        self.use_endeffector = cfg.endeffector.use_endeffector
        if self.use_endeffector:
            self.endeffector = EndEffector(self.batch_size, cfg.endeffector, cfg.output)

        self.ti_objects:Dict[str, TiObject] = {"Cloth": self.cloth}
        if self.use_robot:
            self.ti_objects["Robot"] = self.robot
        if self.use_endeffector:
            self.ti_objects["EndEffector"] = self.endeffector
        self.ti_objects[f"Scissor"] = self.scissors

        ########                        Objects                       ########
        ######################################################################

        ######################################################################
        ########                      Interactions                    ########

        # for collision handling
        self.cs_collision = ClothScissorCollision(
            self.batch_size, cfg.env.cloth_scissor_collision, cfg.scissor, cfg.output, self.cloth, self.scissors, log)
        self.ce_collision = ClothEndeffectorCollision(
            self.batch_size, cfg.env.cloth_endeffector_collision, cfg.endeffector, cfg.output, self.cloth, self.endeffector, log)

        # for cutting
        self.cutting = Cutting(self.batch_size, cfg.env.cut_sim, cfg.env.cloth_scissor_collision, cfg.scissor,
                               self.cloth, self.scissors)
        self.allow_rotation_when_stuck = cfg.env.cut_sim.allow_rotation_when_stuck

        self.ti_interactions:Dict[str, TiInteraction] = {
            "cs_collision": self.cs_collision,
            "ce_collision": self.ce_collision,
            "cutting": self.cutting,
        }

        ########                      Interactions                    ########
        ######################################################################

        # set constrain
        self.constraint = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.batch_size, self.cloth.mesh.nmax_vertices))
        """[B, V][3]"""
        self.constraint.fill(1.0)

        # init clock
        self.clock = Clock()

        # total force and hessian
        self.total_force = ti.field(dtype=ti.f32, shape=(self.batch_size, self.cloth.mesh.nmax_vertices * self.dim))
        """[B, V * 3]"""

        self.total_hessian = SparseMatrix(
            batch_size=self.batch_size,
            nmax_row=self.cloth.mesh.nmax_vertices * self.dim,
            nmax_column=self.cloth.mesh.nmax_vertices * self.dim,
            nmax_triplet=self.cloth.hessian_sparse.nmax_triplet + 
                         self.cs_collision.penalty_hessian_sparse.nmax_triplet + 
                         self.ce_collision.penalty_hessian_sparse.nmax_triplet, store_dense=False)
        
        self.nvert_times3 = ti.field(dtype=ti.i32, shape=(self.batch_size, ))
        """[B, ]"""

        if self.use_implicit_integral:
            # for linearsystem solver
            self.hess_coeff = cfg.env.cut_sim.hess_coeff
            if cfg.env.cut_sim.solver_type == "CG":
                self.solver_type = CG_SOLVER
                self.CG_max_iter = cfg.env.cut_sim.CG_max_iter
                self.CG_relative_tol = cfg.env.cut_sim.CG_relative_tol
                self.CG_dx_tol = cfg.env.cut_sim.CG_dx_tol
                self.CG_init_zero = cfg.env.cut_sim.CG_init_zero
                self.CG_judge_plateau_avg_num = cfg.env.cut_sim.CG_judge_plateau_avg_num
                self.CG_restart_threshold = cfg.env.cut_sim.CG_restart_threshold
                self.total_iter_cnt = 0

                self.incomplete_cholesky_initial_alpha = None

                if cfg.env.cut_sim.CG_precond == "Jacobi":
                    self.CG_precond = JACOBI_PRECOND
                elif cfg.env.cut_sim.CG_precond == "IncompleteCholesky":
                    self.CG_precond = INCOMPLETE_CHOLESKY_PRECOND
                    self.incomplete_cholesky_initial_alpha = cfg.env.cut_sim.CG_incomplete_cholesky_initial_alpha
                elif cfg.env.cut_sim.CG_precond == "BlockJacobi":
                    self.CG_precond = BLOCK_JACOBI_PRECOND
                else:
                    raise NotImplementedError

                self.dv0 = ti.field(dtype=ti.f32, shape=(self.batch_size, self.cloth.mesh.nmax_vertices * self.dim))
                """[B, V * 3]"""
                
                self.update_flag = ti.field(dtype=bool, shape=(self.batch_size, ))

                # A's triplet = hess's triplet + diag(mass)'s triplet
                self.solver = Modified_PCG_sparse_solver(
                    self.batch_size,
                    nmax=self.cloth.mesh.nmax_vertices * self.dim,
                    nmax_triplet=self.total_hessian.nmax_triplet + self.cloth.mesh.nmax_vertices * self.dim, 
                    precond=self.CG_precond, dx_eps=self.cloth.dx_eps / self.dt, 
                    dA_eps=self.cloth.E * self.dt ** 2 * self.cloth.h * CG_RELATIVE_A_EPS, 
                    max_iter=self.CG_max_iter, 
                    judge_plateau_avg_num=self.CG_judge_plateau_avg_num,
                    print_warning=False)

                # tmp variables
                self.tmpsca1_f = ti.field(ti.f32, shape=(self.batch_size, ))
                self.tmpsca2_f = ti.field(ti.f32, shape=(self.batch_size, ))
                self.tmpvec1_f = ti.field(ti.f32, shape=(self.batch_size, self.solver.nmax))
                self.tmpvec2_f = ti.field(ti.f32, shape=(self.batch_size, self.solver.nmax))

            elif cfg.env.cut_sim.solver_type == "cholmod":
                raise NotImplementedError
                self.solver_type = CHOLMOD_SOLVER
                nmax_triplet = self.total_hessian.nmax_triplet + \
                    self.cloth.mesh.nmax_vertices * self.dim
                nmax = self.cloth.mesh.nmax_vertices * self.dim

                self.cholmod_A = SparseMatrix(
                    nmax, nmax, nmax_triplet, store_dense=False)
                self.cholmod_b_f = ti.field(dtype=ti.f32, shape=(nmax))
                self.cholmod_helper_1_f = ti.field(dtype=ti.f32, shape=(nmax))
                self.cholmod_helper_2_f = ti.field(dtype=ti.f32, shape=(nmax))

                self.A_csc = None
                self.b_arr = None

            else:
                raise NotImplementedError

    @ti.kernel
    def apply_elastic_and_external_force_explicit_kernel(self):
        dv_max = ti.abs(self.max_velocity)

        nv_max = vec_max_func(self.cloth.mesh.n_vertices, self.batch_size)
        for _b, vid, i in ti.ndrange(self.batch_size, nv_max, 3):
            if vid < self.cloth.mesh.n_vertices[_b]:
                dv = self.total_force[_b, vid * 3 + i] * self.dt / ti.max(
                    self.cloth.vertices_mass[_b, vid], self.cloth.dx_eps ** 2 * self.cloth.h * self.cloth.rho)

                self.cloth.vertices_vel[_b, vid][i] += self.constraint[_b, vid][i] * \
                    ti.max(ti.min(dv, dv_max), -dv_max)

    def explicit_time_integral(self):
        """
        Update velocity using elastic force and external force, and explicit time integral.

        Please call 'self.assemble_total_force_kernel()' first.
        """
        self.apply_elastic_and_external_force_explicit_kernel()

    @ti.kernel
    def update_velocity_with_constraint_use_flattened_numpy_kernel(self, dv_arr: ti.types.ndarray()):
        dv_max = ti.abs(self.max_velocity)
        nv_max = vec_max_func(self.cloth.mesh.n_vertices, self.batch_size)
        for _b, vid, i in ti.ndrange(self.batch_size, nv_max, 3):
            if vid < self.cloth.mesh.n_vertices[_b]:
                dv = dv_arr[_b, 3 * vid + i]
                self.cloth.vertices_vel[_b, vid][i] += self.constraint[_b, vid][i] * \
                    ti.max(ti.min(dv, dv_max), -dv_max)

    @ti.kernel
    def assemble_total_hessian_kernel(self):
        self.total_hessian.sparse_add_sparse_func(
            self.cloth.hessian_sparse, self.cs_collision.penalty_hessian_sparse)
        self.total_hessian.add_sparse_func(self.ce_collision.penalty_hessian_sparse)

    @ti.kernel
    def assemble_total_force_kernel(self):
        nv_max = vec_max_func(self.cloth.mesh.n_vertices, self.batch_size)
        for _b, vid, j in ti.ndrange(self.batch_size, nv_max, 3):
            if vid < self.cloth.mesh.n_vertices[_b]:
                elastic = self.cloth.elastic_force[_b, vid][j]
                penalty = self.cs_collision.penalty_force[_b, vid][j] + self.ce_collision.penalty_force[_b, vid][j]
                gravity = self.cloth.vertices_mass[_b, vid] * self.gravity[j]
                self.total_force[_b, vid * 3 + j] = elastic + penalty + gravity

    @ti.kernel
    def init_CG_solver_A_b_cons_eps_kernel(self, dt: ti.f32, hess_coeff: ti.f32, dx_tol: ti.f32):
        # (M + dt^2 * hess_coeff * hess) @ dv = (f - hess_coeff * hess @ v * dt) * dt

        # n = self.cloth.mesh.n_vertices[None] * 3
        for _b in ti.static(range(self.batch_size)):
            self.nvert_times3[_b] = self.cloth.mesh.n_vertices[_b] * 3
            self.update_flag[_b] = (not self.cs_collision.error_flag[_b]) and \
                                   (not self.ce_collision.error_flag[_b])

        # init A & eps
        self.solver.eps_f.fill(0.0)
        nv_max = vec_max_func(self.cloth.mesh.n_vertices, self.batch_size)
        for _b, vid, j in ti.ndrange(self.batch_size, nv_max, 3):
            if vid < self.cloth.mesh.n_vertices[_b]:
                self.tmpvec1_f[_b, 3 * vid + j] = self.cloth.vertices_mass[_b, vid]
                self.solver.eps_f[_b] += (dx_tol / dt) ** 2 * self.cloth.vertices_mass[_b, vid]
        for _b in ti.static(range(self.batch_size)):
            self.tmpsca1_f[_b] = dt ** 2 * hess_coeff
            self.tmpsca2_f[_b] = dt
        self.solver.A.sparse_add_diag_func(
            self.tmpvec1_f, self.total_hessian, self.solver.pos_ones_batch, self.tmpsca1_f, self.nvert_times3)

        # init eps
        '''self.solver.eps_f[None] = (dx_tol / dt) ** 2 * \
            vec_sum_func(self.solver.tmp1_f, 0, n)'''

        # init constraint
        for _b, vid, j in ti.ndrange(self.batch_size, nv_max, 3):
            if vid < self.cloth.mesh.n_vertices[_b]:
                if not(self.constraint[_b, vid][j] == 1.0 or self.constraint[_b, vid][j] == 0.0):
                    print("[ERROR] Constraint[{}][{}] should be either 1.0 or 0.0, but {} got.".format(
                        vid, j, self.constraint[_b, vid][j]))
                self.solver.con_f[_b, 3 * vid + j] = self.constraint[_b, vid][j]

        # init b
        for _b, vid, j in ti.ndrange(self.batch_size, nv_max, 3):
            if vid < self.cloth.mesh.n_vertices[_b]:
                self.tmpvec1_f[_b, 3 * vid + j] = self.cloth.vertices_vel[_b, vid][j]
        self.total_hessian.mul_vec_func(
            self.solver.pos_ones_batch, self.tmpvec2_f, self.tmpvec1_f, self.nvert_times3)  # hess @ v
        for _b in ti.static(range(self.batch_size)):
            self.tmpsca1_f[_b] = -dt ** 2 * hess_coeff
            self.tmpsca2_f[_b] = dt
        vec_add_vec_batch_func(self.batch_size, self.solver.pos_ones_batch,
            self.solver.b, self.tmpsca2_f, self.total_force, 
            self.tmpsca1_f, self.tmpvec2_f, self.nvert_times3)

        # init dv0
        '''for vid, j in ti.ndrange(self.cloth.mesh.n_vertices[None], 3):
            self.dv0[3 * vid + j] = self.cloth.vertices_vel[vid][j] * \
                (self.CG_init_remain_velocity - 1.0)'''
        self.cloth.estimate_delta_velocity_func(self.dv0)

    def init_CG_solver(self):
        self.clock.start_clock("init_env")
        self.init_CG_solver_A_b_cons_eps_kernel(
            self.dt, self.hess_coeff, self.CG_dx_tol)
        self.clock.end_clock("init_env")
        self.clock.start_clock("init_solver")
        self.solver.init(
            self.nvert_times3, self.incomplete_cholesky_initial_alpha)
        self.clock.end_clock("init_solver")

    @ti.kernel
    def init_cholmod_A_b_kernel(self, dt: ti.f32, hess_coeff: ti.f32):
        raise NotImplementedError
        # (M + dt^2 * hess_coeff * hess) @ dv = (f - hess_coeff * hess @ v * dt) * dt
        nvert = self.cloth.mesh.n_vertices[None]
        n = nvert * 3

        constraint_mass = CONSTRAINT_RELATIVE_MASS * \
            vec_sum_func(self.cloth.vertices_mass, 0, nvert)

        # init A
        for vid, j in ti.ndrange(self.cloth.mesh.n_vertices[None], 3):
            assert self.constraint[vid][j] == 1.0 or self.constraint[vid][j] == 0.0, "[ERROR] Constraint[{}][{}] should be either 1.0 or 0.0, but {} got.".format(
                vid, j, self.constraint[vid][j])
            if self.constraint[vid][j] == 1.0:
                self.cholmod_helper_1_f[3 * vid + j] = \
                    self.cloth.vertices_mass[vid]
            else:
                self.cholmod_helper_1_f[3 * vid + j] = constraint_mass

        self.cholmod_A.sparse_add_diag_func(
            self.cholmod_helper_1_f, self.total_hessian, 1.0, dt ** 2 * hess_coeff, n)

        # init b
        for vid, j in ti.ndrange(self.cloth.mesh.n_vertices[None], 3):
            self.cholmod_helper_1_f[3 * vid + j] = \
                self.cloth.vertices_vel[vid][j]
        self.total_hessian.mul_vec_func(
            self.cholmod_helper_2_f, self.cholmod_helper_1_f, n)  # hess @ v
        vec_add_vec_func(self.cholmod_b_f, dt, self.total_force, -
                         hess_coeff * dt**2, self.cholmod_helper_2_f, n)

    def init_cholmod_solver(self):
        raise NotImplementedError
        self.init_cholmod_A_b_kernel(self.dt, self.hess_coeff)
        n = self.cloth.mesh.n_vertices[None] * 3
        n_triplet = self.cholmod_A.n_triplet[None]

        A_triplet_data = self.cholmod_A.triplet.to_numpy()
        A_val = A_triplet_data["value"][:n_triplet]
        A_row = A_triplet_data["row"][:n_triplet]
        A_col = A_triplet_data["column"][:n_triplet]
        A_coo = coo_matrix((A_val, (A_row, A_col)),
                           shape=(n, n), dtype=np.float32)

        self.A_csc = csc_matrix(A_coo, dtype=np.float32)
        self.b_arr = self.cholmod_b_f.to_numpy()[:n]

    def implicit_time_integral(self):
        """
        Update velocity using elastic force and external force, and implicit time integral.

        Please call:
            - self.assemble_total_force_kernel()
            - self.assemble_total_hessian_kernel()

        first.
        """
        # (M + h^2 * hess) @ dv = (f - hess @ v * h) * h
        # A @ dv = b
        dv = None
        if self.solver_type == CG_SOLVER:
            self.init_CG_solver()

            self.clock.start_clock("solve")
            dv = self.solver.solve(
                self.CG_max_iter, self.CG_init_zero, self.dv0, self.update_flag, self.CG_relative_tol, self.CG_restart_threshold)
            self.total_iter_cnt += self.solver.iter_cnt
            if self.log_detail:
                self.log.info(
                    "substep {}: nvert={}, nface={}, nedge={}, total_iter={}, ".format(self.substep_n, self.cloth.mesh.n_vertices, self.cloth.mesh.n_faces, self.cloth.mesh.n_edges, self.total_iter_cnt) +
                    "CG solver converge? {} {}, delta_0={} delta_n={}".format(self.solver.iter_cnt, self.solver.is_converge, self.solver.delta_0, self.solver.delta_new))
            else:
                self.log.info("substep {}: iter={} total_iter={}".format(self.substep_n, self.solver.iter_cnt, self.total_iter_cnt))
            self.clock.end_clock("solve")
        elif self.solver_type == CHOLMOD_SOLVER:
            raise NotImplementedError
            self.clock.start_clock("init")
            self.init_cholmod_solver()
            self.clock.end_clock("init")

            self.clock.start_clock("solve")
            factor = cholesky(self.A_csc)
            dv = factor(self.b_arr).astype(np.float32)
            self.clock.end_clock("solve")
        else:
            raise NotImplementedError

        self.clock.start_clock("update_vel")
        self.update_velocity_with_constraint_use_flattened_numpy_kernel(dv)
        self.clock.end_clock("update_vel")

    @ti.kernel
    def apply_material_damping_kernel(self):
        """
        update velocity using damping force inside material.
        Please call self.cloth.calculate_damping_force() first
        """
        dv_max = ti.abs(self.max_velocity)

        nv_max = vec_max_func(self.cloth.mesh.n_vertices, self.batch_size)
        for _b, vid in ti.ndrange(self.batch_size, nv_max):
            if vid < self.cloth.mesh.n_vertices[_b]:
                force_i = self.cloth.damping_force[_b, vid]
                dv = force_i * self.dt / ti.max(
                    self.cloth.vertices_mass[_b, vid], self.cloth.dx_eps ** 2 * self.cloth.h * self.cloth.rho)

                self.cloth.vertices_vel[_b, vid] += self.constraint[_b, vid] * \
                    ti.max(ti.min(dv, dv_max), -dv_max)

    @ti.kernel
    def apply_air_damping_kernel(self):
        """update velocity using air damping."""
        nv_max = vec_max_func(self.cloth.mesh.n_vertices, self.batch_size)
        for _b, vid in ti.ndrange(self.batch_size, nv_max):
            if vid < self.cloth.mesh.n_vertices[_b]:
                self.cloth.vertices_vel[_b, vid] -= self.cloth.vertices_vel[_b, vid] * \
                    self.vel_damping1 * self.dt
                self.cloth.vertices_vel[_b, vid] -= self.cloth.vertices_vel[_b, vid] * \
                    ti.min(1.0, self.cloth.vertices_vel[_b, vid].norm(
                    ) * self.vel_damping2 * self.dt)

    @ti.kernel
    def update_position_use_velocity_kernel(self):
        nv_max = vec_max_func(self.cloth.mesh.n_vertices, self.batch_size)
        for _b, vid in ti.ndrange(self.batch_size, nv_max):
            if vid < self.cloth.mesh.n_vertices[_b]:
                dx = self.dt * self.cloth.vertices_vel[_b, vid]
                self.cloth.vertices_pos[_b, vid] += dx

    @ti.kernel
    def fix_outlier_vertices_kernel(self, position_bounds: ti.template()):
        nv_max = vec_max_func(self.cloth.mesh.n_vertices, self.batch_size)
        for _b, vid, j in ti.ndrange(self.batch_size, nv_max, 3):
            if vid < self.cloth.mesh.n_vertices[_b] and \
                not (position_bounds[0][j] <= self.cloth.vertices_pos[_b, vid][j] and
                    self.cloth.vertices_pos[_b, vid][j] <= position_bounds[1][j]):
                self.cloth.vertices_vel[_b, vid][j] = 0.0
                self.cloth.vertices_pos[_b, vid][j] = ti.min(
                    self.cloth.vertices_pos[_b, vid][j], position_bounds[1][j])
                self.cloth.vertices_pos[_b, vid][j] = ti.max(
                    self.cloth.vertices_pos[_b, vid][j], position_bounds[0][j])

    @ti.kernel
    def update_constraints_kernel(self, batch_idx: ti.i32, vid_arr: ti.types.ndarray(), pos_arr: ti.types.ndarray()):
        # self.constraint.fill(1.0)
        for vid in range(self.cloth.mesh.n_vertices[batch_idx]):
            self.constraint[batch_idx, vid] = 1.0
        for idx in range(vid_arr.shape[0]):
            vid = vid_arr[idx]
            x = pos_arr[idx, 0]
            y = pos_arr[idx, 1]
            z = pos_arr[idx, 2]
            self.constraint[batch_idx, vid] = ti.Vector([0.0, 0.0, 0.0], ti.f32)
            self.cloth.vertices_pos[batch_idx, vid] = ti.Vector([x, y, z], ti.f32)

    def update_scissors_pose(self, new_pose: List[dict], compute_ee_pose: Callable):
        """
        new_pose is a list contains 7D pose for each pair of scissor.

        Each 7D pose is a dict.

        Perform a forward kinematic and calculate inverse kinematic.
        """
        assert len(new_pose) == self.batch_size
        self.clock.start_clock("update_scissors_pose")
        self.scissors.update_pose_quick(new_pose, "joint_0")
        self.clock.end_clock("update_scissors_pose")
        if self.use_robot:
            raise NotImplementedError
            self.robot.inverse_kinematics(
                self.ee_joint, compute_ee_pose(self.scissors[0])["robot_ee"])
        if self.use_endeffector:
            self.clock.start_clock("update_endeffector_pose_calculate")
            new_pose = compute_ee_pose(self.scissors)["endeffector"]
            self.clock.end_clock("update_endeffector_pose_calculate")
            self.clock.start_clock("update_endeffector_pose_update")
            self.endeffector.update_pose_quick(new_pose)
            self.clock.end_clock("update_endeffector_pose_update")

    def update_scissors_old_pose(self, new_pose: List[dict]):
        """
        new_pose is a list contains 7D pose for each pair of scissor.

        Each 7D pose is a dict.

        Only perform a deepcopy.
        """
        assert len(new_pose) == self.batch_size
        for batch_idx in range(self.batch_size):
            self.scissors_old_pose[batch_idx] = copy.deepcopy(new_pose[batch_idx])

    def calculate_updated_scissors_old_pose(self, scissors_new_pose: List[dict], is_stuck: List[bool]):
        assert len(is_stuck) == self.batch_size, \
            "len(is_stuck) should be {}, but get {}".format(
            self.batch_size, len(is_stuck))
        assert len(scissors_new_pose) == self.batch_size, \
            "len(scissors_new_pose) should be {}, but get {}".format(
            self.batch_size, len(scissors_new_pose))

        for batch_idx in range(self.batch_size):
            if not is_stuck[batch_idx]:
                self.scissors_old_pose[batch_idx]["joint_0"] = scissors_new_pose[batch_idx]["joint_0"]
                self.scissors_old_pose[batch_idx]["joint_1"][:] = scissors_new_pose[batch_idx]["joint_1"][:]

        old_direct_cfg = self.scissors.get_direct_cfg(True)

        scissors_new_pose_batch = []
        scissors_old_pose_batch = []
        old_direct_cfg_batch = []
        for batch_idx in range(self.batch_size):
            if not is_stuck[batch_idx]:
                scissors_new_pose_batch.append(old_direct_cfg[batch_idx])
                scissors_old_pose_batch.append(old_direct_cfg[batch_idx])
                old_direct_cfg_batch.append(old_direct_cfg[batch_idx])
            else:
                scissors_new_pose_batch.append(scissors_new_pose[batch_idx])
                scissors_old_pose_batch.append(self.scissors_old_pose[batch_idx])
                old_direct_cfg_batch.append(old_direct_cfg[batch_idx])

        self.scissors.update_pose_quick(scissors_new_pose_batch, "joint_0")
        pos_new_batch = self.scissors.get_cut_front_point_quick("world")

        self.scissors.update_pose_quick(scissors_old_pose_batch, "joint_0")
        pos_old_batch = self.scissors.get_cut_front_point_quick("world")

        self.scissors.update_pose_quick(old_direct_cfg_batch, "direct")
        pos_batch, vec_batch, axs_batch = self.scissors.get_cut_direction_quick()
        
        for batch_idx in range(self.batch_size):
            if is_stuck[batch_idx]:
                pos_new, pos_old, vec, axs = pos_new_batch[batch_idx], pos_old_batch[batch_idx], vec_batch[batch_idx], axs_batch[batch_idx]

                front_dr = pos_new - pos_old
                dx = front_dr.dot(vec)

                # allow pull back
                if dx < 0.0:
                    front_dr -= vec * dx

                for i in range(3):
                    self.scissors_old_pose[batch_idx]["joint_1"][i] = scissors_new_pose[batch_idx]["joint_1"][i] - front_dr[i]

                # allow open and close
                dangle = scissors_new_pose[batch_idx]["joint_0"] - \
                    self.scissors_old_pose[batch_idx]["joint_0"]
                self.scissors_old_pose[batch_idx]["joint_0"] += dangle

                if self.allow_rotation_when_stuck:
                    for i in [ANGLE_IDX, THETA_IDX, PHI_IDX]:
                        self.scissors_old_pose[batch_idx]["joint_1"][i] = scissors_new_pose[batch_idx]["joint_1"][i]

                else:
                    old_mat = axis_angle_to_matrix(
                        *self.scissors_old_pose[batch_idx]["joint_1"][3:])
                    fwd_mat = tra.rotation_matrix(-dangle / 2, axs)
                    new_mat = fwd_mat @ old_mat
                    angle, direc, point = tra.rotation_from_matrix(new_mat)
                    theta, phi = direc_to_theta_phi(direc)

                    self.scissors_old_pose[batch_idx]["joint_1"][ANGLE_IDX] = angle
                    self.scissors_old_pose[batch_idx]["joint_1"][THETA_IDX] = theta
                    self.scissors_old_pose[batch_idx]["joint_1"][PHI_IDX] = phi

    def get_scissors_pose(self, batch_idx: Union[None, int] = None) -> Union[List[dict], dict]:
        """
        Return a deep copy of scissors pose.
        """
        if batch_idx is None:
            return copy.deepcopy(self.scissors_old_pose)
        elif isinstance(batch_idx, int):
            return copy.deepcopy(self.scissors_old_pose[batch_idx])
        else:
            raise NotImplementedError(type(batch_idx))

    def substep(self, scissors_new_pose: List[dict], scissors_velocity: List[dict], compute_ee_pose: Callable, constraints: List[Dict[int, np.ndarray]], now_time: float):
        # update constrains
        self.clock.start_clock("update_constraint")
        assert isinstance(constraints, list)
        assert len(constraints) == self.batch_size
        for batch_idx, constraint in enumerate(constraints):
            vid_arr = np.array(list(constraint.keys()), np.int32)
            pos_arr = np.array(list(constraint.values()), np.float32)
            self.update_constraints_kernel(batch_idx, vid_arr, pos_arr)
        if self.log_detail:
            self.log.info("substep {}: constraints:{}".format(
                self.substep_n, constraints))
        self.clock.end_clock("update_constraint")

        # update scissors pose
        self.clock.start_clock("update_pose_before_cut")
        last_front_points = self.scissors.get_cut_front_point(["world"] * self.batch_size)
        self.update_scissors_pose(scissors_new_pose, compute_ee_pose)
        self.clock.end_clock("update_pose_before_cut")

        # cut cloth
        self.clock.start_clock("cut")
        is_stuck = self.cutting.cut_cloth(
            last_front_points, scissors_new_pose, scissors_velocity, now_time)
        if self.log_detail:
            self.log.info("substep {}: is_stuck:{}".format(
                self.substep_n, is_stuck))
        if self.print_stuck and True in is_stuck:
            print(f"Scissor {np.where(np.array(is_stuck)==True)[0].tolist()} stuck!")
        self.clock.end_clock("cut")

        self.clock.start_clock("update_pose_after_cut")
        self.calculate_updated_scissors_old_pose(scissors_new_pose, is_stuck)
        self.update_scissors_pose(self.scissors_old_pose, compute_ee_pose)
        self.clock.end_clock("update_pose_after_cut")

        if self.log_detail:
            self.log.info(
                "substep {}: modified_scissors_pose:{} front_point:{} cut_vec:{} robot_pose:{}".format(
                    self.substep_n,
                    self.scissors_old_pose,
                    self.scissors.get_cut_front_point(["world"] * self.batch_size),
                    self.scissors.get_cut_direction(),
                    self.robot.yrdf.cfg if self.use_robot else None
                ))
        

        # handle collision with scissor
        self.clock.start_clock("penalty")
        self.cs_collision.calculate_penalty_force(
            self.position_bounds, self.substep_n, self.dt)
        self.ce_collision.calculate_penalty_force(
            self.position_bounds, self.substep_n)
        self.clock.end_clock("penalty")

        # calculate force inside cloth
        self.clock.start_clock("force")
        self.cloth.calculate_elastic_force()
        if self.use_implicit_integral:
            self.cloth.assemble_hessian_kernel()
        self.cloth.update_plasticity(self.dt)
        self.clock.end_clock("force")

        # time integral
        self.assemble_total_force_kernel()
        if self.use_implicit_integral:
            self.clock.start_clock("assemble_hessian")
            self.assemble_total_hessian_kernel()
            self.clock.end_clock("assemble_hessian")
            self.implicit_time_integral()
        else:
            self.explicit_time_integral()

        # add damping force and update velocity
        self.clock.start_clock("damp")
        self.cloth.calculate_damping_force()
        self.apply_material_damping_kernel()
        self.apply_air_damping_kernel()
        self.clock.end_clock("damp")

        # update position
        self.clock.start_clock("update_pos")
        self.update_position_use_velocity_kernel()
        for batch_idx, constraint in enumerate(constraints):
            vid_arr = np.array(list(constraint.keys()), np.int32)
            pos_arr = np.array(list(constraint.values()), np.float32)
            self.update_constraints_kernel(batch_idx, vid_arr, pos_arr)
        self.clock.end_clock("update_pos")

        # apply constraints
        self.clock.start_clock("fix_vertices")
        self.fix_outlier_vertices_kernel(self.position_bounds)

        if self.topo_check:
            self.cloth.topological_check()

        '''total_energy = self.cloth.get_total_energy_kernel(self.gravity)
        self.log.info("substep {}: cloth energy:{:.6e} stretch:{:.6e} bend:{:.6e} gravity:{:.6e} kinetic:{:.6e}".format(
            self.substep_n,
            total_energy[0] + total_energy[1] +
            total_energy[2] + total_energy[3],
            total_energy[0],
            total_energy[1],
            total_energy[2],
            total_energy[3]))'''
        self.clock.end_clock("fix_vertices")

        self.substep_n += 1

    def get_scissors(self) -> Scissor:
        """
        Return a reference of self.scissors.
        """
        return self.scissors

    @ti.kernel
    def get_current_pos_given_rest_kernel(self, batch_idx: ti.i32, rest_pos: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
        dx_eps = self.cloth.dx_eps

        min_dist = ti.math.inf
        for fid in range(self.cloth.mesh.n_faces[batch_idx]):
            v1id, v2id, v3id = self.cloth.mesh.faces_vid[batch_idx, fid]

            v1pos_rest = self.cloth.vertices_rest_pos[batch_idx, v1id]
            v2pos_rest = self.cloth.vertices_rest_pos[batch_idx, v2id]
            v3pos_rest = self.cloth.vertices_rest_pos[batch_idx, v3id]

            dist, u, v, w = get_distance_to_triangle_func(
                rest_pos, v1pos_rest, v2pos_rest, v3pos_rest, dx_eps)
            ti.atomic_min(min_dist, dist)

        min_fid = -1
        bc = ti.Vector.zero(ti.f32, 3)
        for fid in range(self.cloth.mesh.n_faces[batch_idx]):
            v1id, v2id, v3id = self.cloth.mesh.faces_vid[batch_idx, fid]

            v1pos_rest = self.cloth.vertices_rest_pos[batch_idx, v1id]
            v2pos_rest = self.cloth.vertices_rest_pos[batch_idx, v2id]
            v3pos_rest = self.cloth.vertices_rest_pos[batch_idx, v3id]

            dist, u, v, w = get_distance_to_triangle_func(
                rest_pos, v1pos_rest, v2pos_rest, v3pos_rest, dx_eps)
            if dist == min_dist:
                min_fid = fid
                bc = u, v, w

        u, v, w = bc
        v1id, v2id, v3id = self.cloth.mesh.faces_vid[batch_idx, min_fid]
        a = self.cloth.vertices_pos[batch_idx, v1id]
        b = self.cloth.vertices_pos[batch_idx, v2id]
        c = self.cloth.vertices_pos[batch_idx, v3id]

        return u * a + v * b + w * c
    
    def get_cloth_mesh(self) -> List[trimesh.Trimesh]:
        return self.cloth.get_mesh()

    def get_scissors_mesh(self) -> List[trimesh.Trimesh]:
        return self.scissors.get_mesh()

    def get_robot_mesh(self) -> List[trimesh.Trimesh]:
        raise NotImplementedError
        assert self.use_robot
        return self.robot.get_mesh()

    def get_endeffector_mesh(self) -> List[trimesh.Trimesh]:
        assert self.use_endeffector
        return self.endeffector.get_mesh()

    def export_all_ply(self, frame_i, series_prefix_list: List[str], series_suffix="", export_cloth=True, export_scissor=True, export_robot=True, export_endeffector=True):
        """export all cloth and scissor"""

        meshes_batch = [[] for _ in range(self.batch_size)]
        if export_cloth:
            for batch_idx, mesh in enumerate(self.get_cloth_mesh()):
                meshes_batch[batch_idx].append(mesh)

        if export_scissor:
            for batch_idx, mesh in enumerate(self.get_scissors_mesh()):
                meshes_batch[batch_idx].append(mesh)

        if self.use_robot and export_robot:
            for batch_idx, mesh in enumerate(self.get_robot_mesh()):
                meshes_batch[batch_idx].append(mesh)

        if self.use_endeffector and export_endeffector:
            for batch_idx, mesh in enumerate(self.get_endeffector_mesh()):
                meshes_batch[batch_idx].append(mesh)
        
        assert isinstance(series_prefix_list, list)
        assert len(series_prefix_list) == self.batch_size
        for batch_idx, meshes in enumerate(meshes_batch):
            if len(meshes) >= 1:
                filename = to_absolute_path(series_prefix_list[batch_idx] + "_" + str(frame_i).zfill(6) + series_suffix + ".ply")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                mesh_merge: trimesh.Trimesh = trimesh.util.concatenate(meshes)
                if self.use_ascii:
                    with open(filename, "bw") as f_obj:
                        f_obj.write(trimesh.exchange.ply.export_ply(
                            mesh_merge, encoding="ascii"))
                else:
                    mesh_merge.export(filename)

    def export_numpy_arr(self, numpy_arr: np.ndarray, frame_i: int, series_prefix: str, series_suffix=""):
        """export arbitrary numpy array"""
        filename = to_absolute_path(series_prefix + "_" + str(frame_i).zfill(6) + series_suffix + ".npy")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.save(filename, numpy_arr)


    def reset(self) -> None:
        """
        Reset all configuration to initial state.
        """
        # reset object
        for ti_object in self.ti_objects.values():
            ti_object.reset()

        # reset interaction
        for ti_interaction in self.ti_interactions.values():
            ti_interaction.reset()

        # reset other variables
        for batch_idx in range(self.batch_size):
            self.scissors_old_pose[batch_idx] = copy.deepcopy(self.scissors.get_init_pose())

        self.constraint.fill(1.0)

        # reset clock
        self.clock.reset()

    def get_state(self) -> List[dict]:
        ti_objects_state = [{} for _ in range(self.batch_size)]
        for ti_object_name, ti_object in self.ti_objects.items():
            self.clock.start_clock(f"{ti_object_name} state")
            for batch_idx, state in enumerate(ti_object.get_state()):
                ti_objects_state[batch_idx][ti_object_name] = state
            self.clock.end_clock(f"{ti_object_name} state")

        ti_interactions_state = [{} for _ in range(self.batch_size)]
        for ti_interaction_name, ti_interaction in self.ti_interactions.items():
            self.clock.start_clock(f"{ti_interaction_name} state")
            for batch_idx, state in enumerate(ti_interaction.get_state()):
                ti_interactions_state[batch_idx][ti_interaction_name] = state
            self.clock.end_clock(f"{ti_interaction_name} state")

        self.clock.start_clock("constraint_state")
        constraint_state = self.constraint.to_numpy()
        self.clock.end_clock("constraint_state")

        return [{
            "ti_objects": copy.deepcopy(ti_obj),
            "ti_interactions": copy.deepcopy(ti_int),
            "scissors_old_pose": copy.deepcopy(old_pose),
            "constraint": copy.deepcopy(constraint),
        } for ti_obj, ti_int, old_pose, constraint in 
        zip(ti_objects_state, ti_interactions_state, self.scissors_old_pose, constraint_state)]
    
    def set_state(self, states: List[dict]) -> None:
        assert isinstance(states, list)
        assert isinstance(states[0], dict)
        assert len(states) == self.batch_size

        for k in self.ti_objects.keys():
            self.ti_objects[k].set_state([state["ti_objects"][k] for state in states])

        for k in self.ti_interactions.keys():
            self.ti_interactions[k].set_state([state["ti_interactions"][k] for state in states])

        for batch_idx in range(self.batch_size):
            self.scissors_old_pose[batch_idx] = copy.deepcopy(states[batch_idx]["scissors_old_pose"])
        
        self.constraint.from_numpy(np.array([state["constraint"] for state in states]))
