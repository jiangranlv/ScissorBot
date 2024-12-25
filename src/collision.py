import taichi as ti

from typing import List
import logging

from omegaconf import DictConfig

from src.cloth import Cloth
from src.scissor import *
from src.endeffector import EndEffector
from src.maths import *
from src.utils import *
from src.cbase import *


@ti.dataclass
class FaceSampleInfo:
    face_id: ti.i32
    bary_coor: ti.types.vector(2, ti.f32)


@ti.data_oriented
class ClothScissorCollision(TiInteraction):
    def __init__(self, batch_size: int, collision_cfg: DictConfig, scissor_cfg: DictConfig, output_cfg: DictConfig, cloth: Cloth, scissor: Scissor, log: logging.Logger) -> None:
        self.batch_size = batch_size
        
        # Miscellaneous
        self.dim = 3
        self.log = log
        self.log_detail = output_cfg.log.detail

        # Objects
        self.cloth = cloth
        assert isinstance(scissor, Scissor)
        self.scissor = scissor

        # Dynamics
        self.vert_sdf_info: ti.MatrixField = \
            ti.Matrix.field(n=2, m=SDF_SHAPE, dtype=ti.f32, shape=(self.batch_size, self.cloth.mesh.nmax_vertices))
        """[B, V][2, 4]"""
        self.penalty_force: ti.MatrixField = \
            ti.Vector.field(n=3, dtype=ti.f32, shape=(self.batch_size, self.cloth.mesh.nmax_vertices))
        """[B, V][3]"""
        self.penalty_hessian = ti.Matrix.field(
            n=3, m=3, dtype=ti.f32, shape=(self.batch_size, self.cloth.mesh.nmax_vertices))
        """[B, V][3, 3]"""

        self.penalty_hessian_sparse = SparseMatrix(
            batch_size=self.batch_size,
            nmax_row=self.cloth.mesh.nmax_vertices * self.dim,
            nmax_column=self.cloth.mesh.nmax_vertices * self.dim,
            nmax_triplet=self.cloth.mesh.nmax_vertices * (self.dim ** 2) * 2, store_dense=False)

        # Kinematics
        self.nmax_query = scissor_cfg.nmax_query
        self.n_query = ti.field(dtype=ti.i32, shape=(self.batch_size, ))
        """[B, ]"""
        self.n_query_acc = ti.field(dtype=ti.i32, shape=(self.batch_size, ))
        """[B, ]"""
        self.query_pos = ti.Vector.field(
            n=3, dtype=ti.f32, shape=(self.batch_size, self.nmax_query))
        """[B, Q][3]"""

        self.face_sample_info = FaceSampleInfo.field(shape=(self.batch_size, self.nmax_query))
        """[B, Q]"""
        self.face_sample_cnt = ti.field(
            dtype=ti.i32, shape=(self.batch_size, self.cloth.mesh.nmax_faces))
        """[B, F]"""
        self.sdf_info = ti.Matrix.field(
            n=2, m=SDF_SHAPE, dtype=ti.f32, shape=(self.batch_size, self.nmax_query))
        """[B, Q][2, 4]"""

        '''self.scissor_penalty_force = ti.Vector.field(
            n=6, dtype=ti.f32, shape=(self.scissor_num, self.nmax_query * 2))
        self.scissor_penalty_force_cnt = ti.field(
            dtype=ti.i32, shape=(self.scissor_num))

        self.model_penetration = ti.Vector.field(
            n=5, dtype=ti.f32, shape=self.scissor_num * self.nmax_query * 2)  # [x, y, z, sdf, volume]
        self.model_penetration_cnt = ti.field(dtype=ti.i32, shape=())'''

        # Parameters
        self.barrier_width = collision_cfg.barrier_width
        self.barrier_sdf_offset = collision_cfg.barrier_sdf_offset
        self.barrier_strength = collision_cfg.barrier_strength
        self.barrier_param1 = collision_cfg.barrier_param1
        self.collision_sample_dx = collision_cfg.collision_sample_dx
        self.friction_mu = float(collision_cfg.friction_mu)
        self.collision_sdf_offset = float(collision_cfg.collision_sdf_offset)

        # error flag
        self.error_flag = ti.field(bool, shape=(self.batch_size, ))
        self.error_flag.fill(False)

    @ti.kernel
    def generate_sample_on_vertices_kernel(self):
        self.n_query.fill(0)
        nv_max = vec_max_func(self.cloth.mesh.n_vertices, self.batch_size)
        for _b, vid in ti.ndrange(self.batch_size, nv_max):
            if vid < self.cloth.mesh.n_vertices[_b]:
                if self.cloth.mesh.n_vertices[_b] < self.nmax_query:
                    self.query_pos[_b, vid] = self.cloth.vertices_pos[_b, vid]
                    self.n_query[_b] = self.cloth.mesh.n_vertices[_b]
                else:
                    old_flag = ti.atomic_or(self.error_flag[_b], True)
                    if not old_flag:
                        print(f"[ERROR] batch_idx {_b} cs-collision query number reaches maximum.")

    def query_all_vertices_sdf(self):
        self.generate_sample_on_vertices_kernel()
        self.vert_sdf_info.fill(
            ti.Matrix([[self.scissor.sdf_inf, 0.0, 0.0, 0.0], [self.scissor.sdf_inf, 0.0, 0.0, 0.0]], ti.f32))
        self.scissor.query_sdf(self.query_pos, self.vert_sdf_info, self.n_query, False)

    @ti.func
    def generate_sample_on_single_face_func(self, batch_idx, fid: ti.i32, sample_dx: ti.f32, barrier_width: ti.f32, v0sdf: ti.types.matrix(2, SDF_SHAPE, ti.f32), v1sdf: ti.types.matrix(2, SDF_SHAPE, ti.f32), v2sdf: ti.types.matrix(2, SDF_SHAPE, ti.f32), position_bounds: ti.template()):
        assert sample_dx >= 0.0

        vids = self.cloth.mesh.faces_vid[batch_idx, fid]
        pos0 = self.cloth.vertices_pos[batch_idx, vids[0]]
        pos1 = self.cloth.vertices_pos[batch_idx, vids[1]]
        pos2 = self.cloth.vertices_pos[batch_idx, vids[2]]

        self.face_sample_cnt[batch_idx, fid] = 0

        if sample_dx == 0.0 or not (in_simulation_region_func(pos0, position_bounds) and
                                    in_simulation_region_func(pos1, position_bounds) and
                                    in_simulation_region_func(pos2, position_bounds)):
            sample_id = ti.atomic_add(self.n_query[batch_idx], 1)
            if sample_id < self.nmax_query:
                bc = ti.Vector([1 / 3, 1 / 3], dt=ti.f32)
                self.face_sample_info[batch_idx, sample_id] = FaceSampleInfo(
                    fid, bc)
                self.query_pos[batch_idx, sample_id] = (pos0 + pos1 + pos2) / 3
                self.face_sample_cnt[batch_idx, fid] += 1
            else:
                ti.atomic_add(self.n_query[batch_idx], -1)
                old_flag = ti.atomic_or(self.error_flag[batch_idx], True)
                if not old_flag:
                    print(f"[ERROR] batch_idx {batch_idx} cs-collision query number reaches maximum.")

        else:
            max_edge_length = self.cloth.get_face_max_length_func(batch_idx, fid)
            maximum_sdf = max_edge_length + barrier_width

            if v0sdf[0, 0] < maximum_sdf or v0sdf[1, 0] < maximum_sdf or \
                    v1sdf[0, 0] < maximum_sdf or v1sdf[1, 0] < maximum_sdf or \
                    v2sdf[0, 0] < maximum_sdf or v2sdf[1, 0] < maximum_sdf:
                sample_res = ti.ceil(max_edge_length / sample_dx, dtype=ti.i32)
                for a in range(sample_res):
                    for b in range(sample_res - a):
                        c = sample_res - 1 - a - b
                        u, v, w = (a + 1 / 3) / sample_res, \
                            (b + 1 / 3) / sample_res, (c + 1 / 3) / sample_res
                        assert ti.abs(u + v + w - 1.0) < 1e-6

                        sample_id = ti.atomic_add(self.n_query[batch_idx], 1)
                        if sample_id < self.nmax_query:
                            bc = ti.Vector([u, v], dt=ti.f32)
                            self.face_sample_info[batch_idx, sample_id] = FaceSampleInfo(
                                fid, bc)
                            self.query_pos[batch_idx, sample_id] = pos0 * u + \
                                + pos1 * v + pos2 * w
                            self.face_sample_cnt[batch_idx, fid] += 1
                        else:
                            ti.atomic_add(self.n_query[batch_idx], -1)
                            old_flag = ti.atomic_or(self.error_flag[batch_idx], True)
                            if not old_flag:
                                print(f"[ERROR] batch_idx {batch_idx} cs-collision query number reaches maximum.")

    @ti.kernel
    def generate_sample_on_faces_kernel(self, sample_dx: ti.f32, barrier_width: ti.f32, vert_sdf_info: ti.template(), position_bounds: ti.template()):
        """
        Generate samples inside each face with face_need_sample[fid] != 0.
        These samples can be used to calculate penalty force.
        """
        self.n_query.fill(0)
        nf_max = vec_max_func(self.cloth.mesh.n_faces, self.batch_size)
        for _b, fid in ti.ndrange(self.batch_size, nf_max):
            if fid < self.cloth.mesh.n_faces[_b]:
                v0id, v1id, v2id = self.cloth.mesh.faces_vid[_b, fid]
                v0sdf = vert_sdf_info[_b, v0id]
                v1sdf = vert_sdf_info[_b, v1id]
                v2sdf = vert_sdf_info[_b, v2id]

                self.generate_sample_on_single_face_func(
                    _b, fid, sample_dx, barrier_width, v0sdf, v1sdf, v2sdf, position_bounds)

    def generate_collision_sample(self, vert_sdf_info: ti.Field, position_bounds: ti.Field):
        """
        Generate collision samples.

        Results are stored in 'self.face_sample_info', 'self.face_sample_cnt', 'self.n_query', 'self.query_pos'.
        """
        self.generate_sample_on_faces_kernel(
            self.collision_sample_dx, self.barrier_width, vert_sdf_info, position_bounds)

    @ti.func
    def get_face_weight_on_vert_func(self, batch_idx, fid: int) -> ti.types.vector(3, ti.f32):
        weight = ti.Vector.zero(dt=ti.f32, n=3)
        vids = self.cloth.mesh.faces_vid[batch_idx, fid]
        for i in range(3):
            weight[i] = 1.0 / self.cloth.mesh.vertices_fid_cnt[batch_idx, vids[i]]
        return weight

    @ti.kernel
    def add_penalty_force_kernel(self, sdf_info_field: ti.template(), sample_info_field: ti.template(), dt: float):
        """
        Add penalty force and penalty hessian.

        Please call:
            - self.penalty_force.fill(0.0)
            - self.penalty_hessian.fill(0.0)
            - self.model_penetration_cnt[None] = 0
        first.
        """

        '''self.scissor_penalty_force_cnt[sid] = 0'''
        get_accumulate_func(self.n_query, self.n_query_acc, self.batch_size)
        for ib, l in ti.ndrange(self.n_query_acc[self.batch_size - 1], 2):
            _b, s = get_batch_and_idx(ib, self.batch_size, self.n_query_acc)
            sdf_info = sdf_info_field[_b, s][l, :]
            sdf = sdf_info[0]
            sdf_grad = sdf_info[1:4]

            sample_info = sample_info_field[_b, s]
            fid = sample_info.face_id
            vids = self.cloth.mesh.faces_vid[_b, fid]
            bc = ti.Vector([sample_info.bary_coor[0], sample_info.bary_coor[1],
                            1.0 - sample_info.bary_coor[0] - sample_info.bary_coor[1]], ti.f32)
            sample_volume = self.cloth.get_face_area_func(_b, fid) * self.cloth.h / \
                self.face_sample_cnt[_b, fid]
            e0, e1, e2 = self.barrier_strength * barrier_function_func(
                sdf + self.barrier_sdf_offset + self.collision_sdf_offset, self.barrier_width, self.barrier_param1) * sample_volume
            assert e2 >= 0.0, "[WARNING] in collision e2 < 0.0"

            # for cloth collision force
            op = sdf_grad.outer_product(sdf_grad)
            for v in range(3):
                vid = vids[v]
                self.penalty_force[_b, vid] += -e1 * sdf_grad * bc[v]
                self.penalty_hessian[_b, vid] += e2 * bc[v] * op
                assert bc[v] >= 0.0, "[WARNING] in collision bc < 0.0"

            penalty_force = -e1 * sdf_grad
            penalty_hessian = e2 * op
            
            # friction
            mass = sample_volume * self.cloth.rho
            sample_vel = (bc[0] * self.cloth.vertices_vel[_b, vids[0]] +
                          bc[1] * self.cloth.vertices_vel[_b, vids[1]] +
                          bc[2] * self.cloth.vertices_vel[_b, vids[2]])
            dv_guess = (
                ti.Matrix.identity(float, 3) * mass +
                dt * dt * penalty_hessian
            ).inverse() @ (
                penalty_force - penalty_hessian @ sample_vel * dt
            ) * dt
            rel_vel = sample_vel + dv_guess
            rel_vel_proj = rel_vel - rel_vel.dot(sdf_grad) * sdf_grad / ti.max(sdf_grad.norm_sqr(), self.cloth.dx_eps)
            rel_vel_proj_abs = ti.math.length(rel_vel_proj)
            friction_force = ti.min(
                self.friction_mu * ti.math.length(dv_guess * mass / dt),
                rel_vel_proj_abs * mass / dt,
            )
            for v in range(3):
                vid = vids[v]
                self.penalty_force[_b, vid] += friction_force * bc[v]

            '''# for scissors collision force
            oldcnt = ti.atomic_add(self.scissor_penalty_force_cnt[sid], 1)
            force_pos = bc[0] * self.cloth.vertices_pos[0] + \
                bc[1] * self.cloth.vertices_pos[1] + \
                bc[2] * self.cloth.vertices_pos[2]
            self.scissor_penalty_force[sid, oldcnt][0:3] = force_pos
            self.scissor_penalty_force[sid, oldcnt][3:6] = e1 * sdf_grad

            # for model penetration
            if sdf < 0.0:
                old_cnt = ti.atomic_add(self.model_penetration_cnt[None], 1)
                self.model_penetration[old_cnt][0:3] = force_pos
                self.model_penetration[old_cnt][3] = sdf
                self.model_penetration[old_cnt][4] = sample_volume'''

    @ti.kernel
    def assemble_penalty_hessian_kernel(self):
        self.penalty_hessian_sparse.set_zero_func()
        nv_max = vec_max_func(self.cloth.mesh.n_vertices, self.batch_size)
        for _b, vid, i, j in ti.ndrange(self.batch_size, nv_max, 3, 3):
            if vid < self.cloth.mesh.n_vertices[_b]:
                self.penalty_hessian_sparse.add_value_func(
                    _b, 3 * vid + i, 3 * vid + j, self.penalty_hessian[_b, vid][i, j])

    def calculate_penalty_force(self, position_bounds: ti.Field, substep_n: int, dt: float):
        """
        Calculate penalty force and penalty hessian as well. 
        """
        self.penalty_force.fill(0.0)
        self.penalty_hessian.fill(0.0)
        '''self.model_penetration_cnt[None] = 0'''

        # we add this pair's penalty force to the cloth
        self.query_all_vertices_sdf()

        # self.clock.start_clock("generate collision sample")
        self.generate_collision_sample(
            self.vert_sdf_info, position_bounds)
        if self.log_detail:
            self.log.info("substep {}: cs-collision sample number:{}".format(
                substep_n, self.n_query))
        # self.clock.end_clock("generate collision sample")

        # self.clock.start_clock("query collision sdf")
        self.scissor.query_sdf(self.query_pos, self.sdf_info, self.n_query, False)
        # self.clock.end_clock("query collision sdf")

        # self.clock.start_clock("add penalty force")
        self.add_penalty_force_kernel(self.sdf_info, self.face_sample_info, dt)
        # self.clock.end_clock("add penalty force")

        self.assemble_penalty_hessian_kernel()

    def get_scissor_penalty_force(self):
        raise NotImplementedError
        '''forces = []
        for sid in range(self.scissor_num):
            forces.append(self.scissor_penalty_force.to_numpy()[
                          sid, 0:self.scissor_penalty_force_cnt[sid], :])
        return forces'''

    def reset(self):
        self.error_flag.fill(False)
    
    def get_state(self) -> List[dict]:
        error_flag = self.error_flag.to_numpy()
        return [{"class": "ClothScissorCollision",
                 "error_flag": error_flag[_]} for _ in range(self.batch_size)]
    
    def set_state(self, states: List[dict]) -> None:
        assert isinstance(states, list)
        assert isinstance(states[0], dict)
        assert len(states) == self.batch_size
        
        error_flag = np.array([state["error_flag"] for state in states])
        self.error_flag.from_numpy(error_flag)
    

@ti.data_oriented
class ClothEndeffectorCollision(TiInteraction):
    def __init__(self, batch_size: int, collision_cfg: DictConfig, endeffector_cfg: DictConfig, output_cfg: DictConfig, cloth: Cloth, endeffector: EndEffector, log: logging.Logger) -> None:
        self.batch_size = batch_size
        
        # Miscellaneous
        self.dim = 3
        self.log = log
        self.log_detail = output_cfg.log.detail

        # Objects
        self.cloth = cloth
        self.endeffector = endeffector
        self.link_n = endeffector.link_n

        # Dynamics
        self.vert_sdf_info_link:ti.MatrixField = \
            ti.Vector.field(n=4, dtype=ti.f32, shape=(self.batch_size, self.cloth.mesh.nmax_vertices, endeffector.link_n))
        """[B, V, L][4]"""
        self.vert_sdf_info_merge:ti.MatrixField = \
            ti.Vector.field(n=4, dtype=ti.f32, shape=(self.batch_size, self.cloth.mesh.nmax_vertices))
        """[B, V][4]"""
        self.penalty_force:ti.MatrixField = ti.Vector.field(
            n=3, dtype=ti.f32, shape=(self.batch_size, self.cloth.mesh.nmax_vertices))
        """[B, V][3]"""
        self.penalty_hessian = ti.Matrix.field(
            n=3, m=3, dtype=ti.f32, shape=(self.batch_size, self.cloth.mesh.nmax_vertices))
        """[B, V][3, 3]"""
        self.penalty_hessian_sparse = SparseMatrix(
            batch_size=self.batch_size,
            nmax_row=self.cloth.mesh.nmax_vertices * self.dim,
            nmax_column=self.cloth.mesh.nmax_vertices * self.dim,
            nmax_triplet=self.cloth.mesh.nmax_vertices * (self.dim ** 2) * self.link_n, store_dense=False)

        # Kinematics
        self.nmax_query = endeffector_cfg.nmax_query
        self.n_query = ti.field(dtype=ti.i32, shape=(self.batch_size, ))
        """[B, ]"""
        self.n_query_acc = ti.field(dtype=ti.i32, shape=(self.batch_size, ))
        """[B, ]"""
        self.query_pos = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.batch_size, self.nmax_query))
        """[B, Q]"""

        self.face_sample_info = FaceSampleInfo.field(shape=(self.batch_size, self.nmax_query))
        """[B, Q]"""
        self.face_sample_cnt = ti.field(dtype=ti.i32, shape=(self.batch_size, self.cloth.mesh.nmax_faces))
        """[B, F]"""
        self.sdf_info_link:ti.MatrixField = ti.Vector.field(n=4, dtype=ti.f32, shape=(self.batch_size, self.nmax_query, endeffector.link_n))
        """[B, Q, L][4]"""
        self.sdf_info_merge:ti.MatrixField = ti.Vector.field(n=4, dtype=ti.f32, shape=(self.batch_size, self.nmax_query))
        """[B, Q][4]"""

        # Parameters
        self.barrier_width = collision_cfg.barrier_width
        self.barrier_sdf_offset = collision_cfg.barrier_sdf_offset
        self.barrier_strength = collision_cfg.barrier_strength
        self.barrier_param1 = collision_cfg.barrier_param1
        self.collision_sample_dx = collision_cfg.collision_sample_dx

        # error flag
        self.error_flag = ti.field(bool, shape=(self.batch_size, ))
        self.error_flag.fill(False)

    @ti.kernel
    def generate_sample_on_vertices_kernel(self):
        self.n_query.fill(0)
        nv_max = vec_max_func(self.cloth.mesh.n_vertices, self.batch_size)
        for _b, vid in ti.ndrange(self.batch_size, nv_max):
            if vid < self.cloth.mesh.n_vertices[_b]:
                if self.cloth.mesh.n_vertices[_b] < self.nmax_query:
                    self.query_pos[_b, vid] = self.cloth.vertices_pos[_b, vid]
                    self.n_query[_b] = self.cloth.mesh.n_vertices[_b]
                else:
                    old_flag = ti.atomic_or(self.error_flag[_b], True)
                    if not old_flag:
                        print(f"[ERROR] batch_idx {_b} ce-collision query number reaches maximum.")

    def query_all_vertices_sdf(self):
        self.generate_sample_on_vertices_kernel()
        self.vert_sdf_info_link.fill(ti.Vector([self.endeffector.sdf_inf, 0.0, 0.0, 0.0]))
        self.vert_sdf_info_merge.fill(ti.Vector([self.endeffector.sdf_inf, 0.0, 0.0, 0.0]))
        self.endeffector.query_sdf(self.query_pos, self.vert_sdf_info_link, self.vert_sdf_info_merge, 
                                   self.n_query)

    @ti.func
    def generate_sample_on_single_face_func(self, batch_idx, fid: ti.i32, sample_dx: ti.f32, barrier_width: ti.f32,
                                            vert_sdf_info_merge: ti.template(), position_bounds: ti.template()):
        assert sample_dx >= 0.0

        vids = self.cloth.mesh.faces_vid[batch_idx, fid]
        pos0 = self.cloth.vertices_pos[batch_idx, vids[0]]
        pos1 = self.cloth.vertices_pos[batch_idx, vids[1]]
        pos2 = self.cloth.vertices_pos[batch_idx, vids[2]]

        self.face_sample_cnt[batch_idx, fid] = 0

        if sample_dx == 0.0 or not (in_simulation_region_func(pos0, position_bounds) and
                                    in_simulation_region_func(pos1, position_bounds) and
                                    in_simulation_region_func(pos2, position_bounds)):
            sample_id = ti.atomic_add(self.n_query[batch_idx], 1)
            if sample_id < self.nmax_query:
                bc = ti.Vector([1 / 3, 1 / 3], dt=ti.f32)
                self.face_sample_info[batch_idx, sample_id] = FaceSampleInfo(
                    fid, bc)
                self.query_pos[batch_idx, sample_id] = (pos0 + pos1 + pos2) / 3
                self.face_sample_cnt[batch_idx, fid] += 1
            else:
                ti.atomic_add(self.n_query[batch_idx], -1)
                old_flag = ti.atomic_or(self.error_flag[batch_idx], True)
                if not old_flag:
                    print(f"[ERROR] batch_idx {batch_idx} ce-collision query number reaches maximum.")

        else:
            max_edge_length = self.cloth.get_face_max_length_func(batch_idx, fid)
            maximum_sdf = max_edge_length + barrier_width

            if vert_sdf_info_merge[batch_idx, vids[0]][0] < maximum_sdf or \
                    vert_sdf_info_merge[batch_idx, vids[1]][0] < maximum_sdf or \
                    vert_sdf_info_merge[batch_idx, vids[2]][0] < maximum_sdf:
                sample_res = ti.ceil(max_edge_length / sample_dx, dtype=ti.i32)
                for a in range(sample_res):
                    for b in range(sample_res - a):
                        c = sample_res - 1 - a - b
                        u, v, w = (a + 1 / 3) / sample_res, \
                            (b + 1 / 3) / sample_res, (c + 1 / 3) / sample_res
                        assert ti.abs(u + v + w - 1.0) < 1e-6

                        sample_id = ti.atomic_add(self.n_query[batch_idx], 1)
                        if sample_id < self.nmax_query:
                            bc = ti.Vector([u, v], dt=ti.f32)
                            self.face_sample_info[batch_idx, sample_id] = FaceSampleInfo(
                                fid, bc)
                            self.query_pos[batch_idx, sample_id] = pos0 * u + \
                                + pos1 * v + pos2 * w
                            self.face_sample_cnt[batch_idx, fid] += 1
                        else:
                            ti.atomic_add(self.n_query[batch_idx], -1)
                            old_flag = ti.atomic_or(self.error_flag[batch_idx], True)
                            if not old_flag:
                                print(f"[ERROR] batch_idx {batch_idx} ce-collision query number reaches maximum.")

    @ti.kernel
    def generate_sample_on_faces_kernel(self, sample_dx: ti.f32, barrier_width: ti.f32, vert_sdf_info_merge: ti.template(), position_bounds: ti.template()):
        """
        Generate samples inside each face with face_need_sample[fid] != 0.
        These samples can be used to calculate penalty force.
        """
        self.n_query.fill(0)
        nf_max = vec_max_func(self.cloth.mesh.n_faces, self.batch_size)
        for _b, fid in ti.ndrange(self.batch_size, nf_max):
            if fid < self.cloth.mesh.n_faces[_b]:
                self.generate_sample_on_single_face_func(
                    _b, fid, sample_dx, barrier_width, vert_sdf_info_merge, position_bounds)

    def generate_collision_sample(self, vert_sdf_info_merge: ti.Field, position_bounds: ti.Field):
        """
        Generate collision samples.

        Results are stored in 'self.face_sample_info', 'self.face_sample_cnt', 'self.n_query', 'self.query_pos'.
        """
        self.generate_sample_on_faces_kernel(
            self.collision_sample_dx, self.barrier_width, vert_sdf_info_merge, position_bounds)

    @ti.func
    def get_face_weight_on_vert_func(self, batch_idx, fid: int) -> ti.types.vector(3, ti.f32):
        weight = ti.Vector.zero(dt=ti.f32, n=3)
        vids = self.cloth.mesh.faces_vid[batch_idx, fid]
        for i in range(3):
            weight[i] = 1.0 / self.cloth.mesh.vertices_fid_cnt[batch_idx, vids[i]]
        return weight

    @ti.kernel
    def add_penalty_force_kernel(self, sdf_info_link_field: ti.template(), sample_info_field: ti.template()):
        """
        Add penalty force and penalty hessian.

        Please call:
            - self.penalty_force.fill(0.0)
            - self.penalty_hessian.fill(0.0)
        first.
        """
        get_accumulate_func(self.n_query, self.n_query_acc, self.batch_size)
        for ib, l in ti.ndrange(self.n_query_acc[self.batch_size - 1], 2):
            _b, s = get_batch_and_idx(ib, self.batch_size, self.n_query_acc)
            sdf_info = sdf_info_link_field[_b, s, l]
            sdf = sdf_info[0]
            sdf_grad = sdf_info[1:4]

            sample_info = sample_info_field[_b, s]
            fid = sample_info.face_id
            vids = self.cloth.mesh.faces_vid[_b, fid]
            bc = ti.Vector([sample_info.bary_coor[0], sample_info.bary_coor[1],
                            1.0 - sample_info.bary_coor[0] - sample_info.bary_coor[1]], ti.f32)
            sample_volume = self.cloth.get_face_area_func(_b, fid) * self.cloth.h / \
                self.face_sample_cnt[_b, fid]
            e0, e1, e2 = self.barrier_strength * barrier_function_func(
                sdf + self.barrier_sdf_offset, self.barrier_width, self.barrier_param1) * sample_volume
            assert e2 >= 0.0, "[WARNING] in collision e2 < 0.0"

            # for cloth collision force
            op = sdf_grad.outer_product(sdf_grad)
            for v in range(3):
                vid = vids[v]
                self.penalty_force[_b, vid] += -e1 * sdf_grad * bc[v]
                self.penalty_hessian[_b, vid] += e2 * bc[v] * op
                assert bc[v] >= 0.0, "[WARNING] in collision bc < 0.0"

    @ti.kernel
    def assemble_penalty_hessian_kernel(self):
        self.penalty_hessian_sparse.set_zero_func()
        nv_max = vec_max_func(self.cloth.mesh.n_vertices, self.batch_size)
        for _b, vid, i, j in ti.ndrange(self.batch_size, nv_max, 3, 3):
            if vid < self.cloth.mesh.n_vertices[_b]:
                self.penalty_hessian_sparse.add_value_func(
                    _b, 3 * vid + i, 3 * vid + j, self.penalty_hessian[_b, vid][i, j])

    def calculate_penalty_force(self, position_bounds: ti.Field, substep_n: int):
        """
        Calculate penalty force and penalty hessian as well. 
        """
        self.penalty_force.fill(0.0)
        self.penalty_hessian.fill(0.0)

        self.query_all_vertices_sdf()

        self.generate_collision_sample(
            self.vert_sdf_info_merge, position_bounds)
        if self.log_detail:
            self.log.info("substep {}: ce-collision sample number:{}".format(
                substep_n, self.n_query))

        self.endeffector.query_sdf(self.query_pos, self.sdf_info_link, 
                                   self.sdf_info_merge, self.n_query)
        
        self.add_penalty_force_kernel(
            self.sdf_info_link, self.face_sample_info)

        self.assemble_penalty_hessian_kernel()

    def reset(self):
        self.error_flag.fill(False)
    
    def get_state(self) -> List[dict]:
        error_flag = self.error_flag.to_numpy()
        return [{"class": "ClothEndeffectorCollision",
                 "error_flag": error_flag[_]} for _ in range(self.batch_size)]
    
    def set_state(self, states: List[dict]) -> None:
        assert isinstance(states, list)
        assert isinstance(states[0], dict)
        assert len(states) == self.batch_size
        
        error_flag = np.array([state["error_flag"] for state in states])
        self.error_flag.from_numpy(error_flag)
