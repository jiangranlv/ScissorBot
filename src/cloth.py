import taichi as ti
import numpy as np

import pywavefront
import trimesh

from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from src.maths import *
from src.sparse import *
from src.mesh import *
from src.cbase import *


_MINIMUM_CUT_RATIO_FOR_CLOTH = 0.01  # wider range for 'Cloth'
MINIMUM_CUT_RATIO = 0.02  # more strict range for caller

DEFORM_GRAD_DSVD_TOL = 1e-5
LDLT_MAX_LOOP_CNT = 1024


@ti.data_oriented
class Cloth(TiObject):
    def __init__(self, batch_size: int, cloth_cfg: DictConfig, output_cfg: DictConfig) -> None:
        self.batch_size = batch_size

        self.dim = 3
        self.print_info = output_cfg.print_info

        self.h = cloth_cfg.h  # thickness
        self.rho = cloth_cfg.rho  # density

        # initialize elastic properties
        self.E = cloth_cfg.E  # Young's modulus
        self.nu = cloth_cfg.nu  # Poisson's ratio
        self.alpha = cloth_cfg.alpha  # bending param

        # relaxation time
        self.stretch_relax_t = cloth_cfg.stretch_relax_t
        self.bending_relax_t = cloth_cfg.bending_relax_t

        # Lame parameters
        self.mu = self.E / 2 / (1 + self.nu)
        self.lda = self.E * self.nu / \
            (1 + self.nu) / (1 - 2 * self.nu)

        # stretch coefficient
        self.ks = self.E * self.h / (1 - self.nu ** 2)

        # bending coefficient
        self.kb = self.ks * self.h ** 2 * self.alpha / 12

        # bending plasticity
        self.bending_yield_criterion = cloth_cfg.bending_yield_criterion
        self.bending_yield_rate = cloth_cfg.bending_yield_rate

        # hessian fix initial displacement
        self.ldlt_relative_err = cloth_cfg.ldlt_relative_err
        self.hess_fix_init_disp = cloth_cfg.hess_fix_init_disp

        # for safety
        self.min_vertex_mass = cloth_cfg.min_vertex_mass
        self.dx_eps = float(cloth_cfg.dx_eps)

        # load mesh
        file_path = to_absolute_path(cloth_cfg.cloth_file)

        n_vertices_multiplier = cloth_cfg.n_vertices_multiplier
        n_edges_multiplier = cloth_cfg.n_edges_multiplier
        n_faces_multiplier = cloth_cfg.n_faces_multiplier
        nmax_edges_connect_to_vertex = cloth_cfg.nmax_edges_connect_to_vertex
        nmax_faces_connect_to_vertex = cloth_cfg.nmax_faces_connect_to_vertex

        self.mesh = DynamicTriangularMesh(self.batch_size, file_path, n_vertices_multiplier, n_edges_multiplier,
                                          n_faces_multiplier, nmax_edges_connect_to_vertex, nmax_faces_connect_to_vertex)

        # initialize vertices position and velocity

        # current position of vertices
        self.vertices_pos = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.batch_size, self.mesh.nmax_vertices))
        """[B, V][3]"""
        self.vertices_pos.from_numpy(
            np.tile(
                np.pad(self.mesh.origin_mesh.vertices.astype(np.float32),
                   ((0, self.mesh.nmax_vertices - int(self.mesh.origin_mesh.vertices.shape[0])), (0, 0)), 
                   'constant', constant_values=0)[None, ...],
                reps=(self.batch_size, 1, 1)
        ))

        # initial position of vertices
        self.vertices_rest_pos = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.batch_size, self.mesh.nmax_vertices))
        """[B, V][3]"""
        self.vertices_rest_pos.from_numpy(
            np.tile(
                np.pad(self.mesh.origin_mesh.vertices.astype(np.float32),
                   ((0, self.mesh.nmax_vertices - int(self.mesh.origin_mesh.vertices.shape[0])), (0, 0)), 
                   'constant', constant_values=0)[None, ...],
                reps=(self.batch_size, 1, 1)
        ))

        # current velocity of vertices
        self.vertices_vel = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.batch_size, self.mesh.nmax_vertices))
        """[B, V][3]"""
        self.vertices_vel.fill(0)

        # current mass of vertices, this mass may vary due to topological changes
        self.vertices_mass = ti.field(dtype=ti.f32, shape=(self.batch_size, self.mesh.nmax_vertices))
        """[B, V]"""
        self.vertices_mass.fill(0)

        # initialize faces

        # initialize TTT43 matrix, Deformation Gradient F=(x0 x1 x2 n)@TTT43
        self.TTT43 = ti.Matrix.field(n=4, m=3, dtype=ti.f32, shape=(self.batch_size, self.mesh.nmax_faces))
        """[B, F][4, 3]"""
        self.TTT43.fill(0)

        self.rest_area = ti.field(dtype=ti.f32, shape=(self.batch_size, self.mesh.nmax_faces))
        """[B, F]"""
        self.rest_area.fill(0)

        self._init_face_property_kernel()
        self._init_mass_kernel()

        # initialize edges

        pass

        # elastic force
        self.deform_grad = ti.Matrix.field(3, 3, ti.f32, (self.batch_size, self.mesh.nmax_faces))
        """[B, F][3, 3]"""
        self.ddpsi_dFdF = ti.field(ti.f32, (self.batch_size, self.mesh.nmax_faces, 9, 9))
        """[B, F, 9, 9]"""
        self.elastic_force = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.batch_size, self.mesh.nmax_vertices))
        """[B, V][3]"""
        self.E_hessian_stretch = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=(self.batch_size, self.mesh.nmax_faces, 3, 3))
        """[B, F, 3, 3][3, 3]"""
        self.E_hessian_bending = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=(self.batch_size, self.mesh.nmax_edges, 4, 4))
        """[B, E, 4, 4][3, 3]"""
        self.stretch_energy = ti.field(ti.f32, shape=(self.batch_size, self.mesh.nmax_faces))
        """[B, F]"""
        self.bending_energy = ti.field(ti.f32, shape=(self.batch_size, self.mesh.nmax_edges))
        """[B, E]"""

        # plastic
        self.rest_angle = ti.field(ti.f32, shape=(self.batch_size, self.mesh.nmax_edges))
        """[B, E]"""

        # fix hessian
        self.is_fixed_flag = ti.field(bool, shape=(self.batch_size, ))
        """[B, ]"""
        self.is_all_success = ti.field(bool, shape=(self.batch_size, ))
        """[B, ]"""

        # damping force
        self.damping_force = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.batch_size, self.mesh.nmax_vertices))
        """[B, V][3]"""

        # derivative of deformation gradient wrt coordinates dF / dx
        self.dF_dx = tiTensor3333.field(shape=(self.batch_size, self.mesh.nmax_faces))
        """[B, F][3, 3, 3, 3]"""
        self.U_f = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=(self.batch_size, self.mesh.nmax_faces))
        """[B, F][3, 3]"""
        self.S_f = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=(self.batch_size, self.mesh.nmax_faces))
        """[B, F][3, 3]"""
        self.V_f = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=(self.batch_size, self.mesh.nmax_faces))
        """[B, F][3, 3]"""
        self.dU_f = tiTensor3333.field(shape=(self.batch_size, self.mesh.nmax_faces))
        """[B, F][3, 3, 3, 3]"""
        self.dS_f = tiTensor333.field(shape=(self.batch_size, self.mesh.nmax_faces))
        """[B, F][3, 3, 3]"""
        self.dV_f = tiTensor3333.field(shape=(self.batch_size, self.mesh.nmax_faces))
        """[B, F][3, 3, 3, 3]"""

        # sparse matrix
        self.hessian_sparse = SparseMatrix(batch_size=self.batch_size,
                                           nmax_row=self.mesh.nmax_vertices * self.dim,
                                           nmax_column=self.mesh.nmax_vertices * self.dim,
                                           nmax_triplet=self.mesh.nmax_faces * (3 * self.dim) ** 2 + self.mesh.nmax_edges * (4 * self.dim) ** 2, store_dense=False)

    #######################################################
    # Section: Helper functions
    #######################################################

    @ti.func
    def get_face_cross_product_func(self, batch_idx, fid: ti.i32) -> ti.Vector:
        ia, ib, ic = self.mesh.faces_vid[batch_idx, fid]
        a, b, c = self.vertices_pos[batch_idx, ia], self.vertices_pos[batch_idx, ib], self.vertices_pos[batch_idx, ic]
        return (a - c).cross(b - c)

    @ti.func
    def get_face_rest_cross_product_func(self, batch_idx, fid: ti.i32) -> ti.Vector:
        ia, ib, ic = self.mesh.faces_vid[batch_idx, fid]
        a, b, c = self.vertices_rest_pos[batch_idx, ia], self.vertices_rest_pos[batch_idx, ib], self.vertices_rest_pos[batch_idx, ic]
        return (a - c).cross(b - c)

    @ti.func
    def get_face_normalized_func(self, batch_idx, fid: ti.i32) -> ti.Vector:
        cp = self.get_face_cross_product_func(batch_idx, fid)
        return safe_normalized(cp, self.dx_eps ** 2)

    @ti.func
    def get_face_area_func(self, batch_idx, fid: ti.i32) -> ti.f32:
        return (self.get_face_cross_product_func(batch_idx, fid)).norm() / 2

    @ti.func
    def get_face_max_length_func(self, batch_idx, fid: ti.i32) -> ti.f32:
        ia, ib, ic = self.mesh.faces_vid[batch_idx, fid]
        a, b, c = self.vertices_pos[batch_idx, ia], self.vertices_pos[batch_idx, ib], self.vertices_pos[batch_idx, ic]
        return ti.max(ti.math.length(a - b), ti.math.length(b - c), ti.math.length(c - a))

    @ti.func
    def get_edge_norm_func(self, batch_idx, eid: ti.i32):
        """Return normalized edge norm. """
        v1id, v2id, f1id, f2id = self.mesh.edges_vid_fid[batch_idx, eid]
        cp = self.get_face_cross_product_func(batch_idx, f1id)
        if f2id != -1:
            cp += self.get_face_cross_product_func(batch_idx, f2id)
        return safe_normalized(cp, self.dx_eps)

    @ti.kernel
    def get_edge_norm_kernel(self, batch_idx: ti.i32, eid: ti.i32) -> ti.types.vector(3, ti.f32):
        return self.get_edge_norm_func(batch_idx, eid)

    @ti.func
    def get_edge_length_func(self, batch_idx, eid: ti.i32) -> ti.f32:
        v0id, v1id, f0id, f1id = self.mesh.edges_vid_fid[batch_idx, eid]
        return ti.math.length(self.vertices_pos[batch_idx, v0id] - self.vertices_pos[batch_idx, v1id])

    @ti.func
    def get_edge_rest_length_func(self, batch_idx, eid: ti.i32) -> ti.f32:
        v0id, v1id, f0id, f1id = self.mesh.edges_vid_fid[batch_idx, eid]
        return ti.math.length(self.vertices_rest_pos[batch_idx, v0id] - self.vertices_rest_pos[batch_idx, v1id])

    @ti.func
    def get_vertex_pos_func(self, batch_idx, eid: ti.i32, beta: ti.f32) -> ti.Vector:
        """Return x1 * beta + x2 * (1 - beta)"""
        v1id, v2id, f1id, f2id = self.mesh.edges_vid_fid[batch_idx, eid]
        v1pos = self.vertices_pos[batch_idx, v1id]
        v2pos = self.vertices_pos[batch_idx, v2id]
        return v1pos * beta + v2pos * (1 - beta)

    @ti.func
    def get_vertex_norm_func(self, batch_idx, vid: ti.i32) -> ti.Vector:
        """Return normalized vertex norm. """
        ret_val = ti.Vector.zero(ti.f32, 3)
        for i in range(self.mesh.vertices_fid_cnt[batch_idx, vid]):
            fid = self.mesh.vertices_fid[batch_idx, vid][i]
            ret_val += self.get_face_cross_product_func(batch_idx, fid)
        return safe_normalized(ret_val, self.dx_eps)

    @ti.func
    def get_rest_dist_to_oppo_vert_func(self, batch_idx, eid: ti.i32, beta: ti.f32) -> ti.types.vector(2, ti.f32):
        """
        Args:
            - eid, v1id, v2id: int
            - beta: float

        vpos = v1pos * beta + v2pos * (1 - beta)

        Return: 2D Vector (d3, d4)
            - d3 = |vpos - v3pos|: float, where 3 vertices on f1id is v1id, v2id, v3id
            - d4 = |vpos - v4pos|: float, where 3 vertices on f2id is v1id, v2id, v4id
            - d4 = -1.0 if not exist
        """
        v1id, v2id, f1id, f2id = self.mesh.edges_vid_fid[batch_idx, eid]
        v3id = self.mesh.find_opp_vert_on_face_func(batch_idx, f1id, v1id, v2id)
        v4id = self.mesh.find_opp_vert_on_face_func(batch_idx, f2id, v1id, v2id)

        v1pos = self.vertices_rest_pos[batch_idx, v1id]
        v2pos = self.vertices_rest_pos[batch_idx, v2id]
        vpos = v1pos * beta + v2pos * (1 - beta)

        ret_val = -ti.Vector.one(ti.f32, 2)

        v3pos = self.vertices_rest_pos[batch_idx, v3id]
        ret_val[0] = ti.math.length(vpos - v3pos)
        if f2id != -1:
            v4pos = self.vertices_rest_pos[batch_idx, v4id]
            ret_val[1] = ti.math.length(vpos - v4pos)

        return ret_val

    #######################################################
    # Section: Physics
    #######################################################

    @ti.func
    def update_vertex_mass_func(self, batch_idx, vid: ti.i32):
        """calculate vertex mass and update"""
        new_mass = 0.0
        ti.loop_config(serialize=True)
        for i in range(self.mesh.vertices_fid_cnt[batch_idx, vid]):
            fid = self.mesh.vertices_fid[batch_idx, vid][i]
            cp = self.get_face_rest_cross_product_func(batch_idx, fid)
            new_mass += cp.norm() / 6 * self.rho * self.h
        self.vertices_mass[batch_idx, vid] = ti.max(new_mass, self.min_vertex_mass)

    @ti.kernel
    def _init_mass_kernel(self):
        nv_max = vec_max_func(self.mesh.n_vertices, self.batch_size)
        for batch_idx, vid in ti.ndrange(self.batch_size, nv_max):
            if vid < self.mesh.n_vertices[batch_idx]:
                self.update_vertex_mass_func(batch_idx, vid)

    @ti.func
    def update_face_property_func(self, batch_idx, fid: ti.i32):
        """update face's TTT43-matrix and rest area"""
        cp = self.get_face_rest_cross_product_func(batch_idx, fid)
        self.rest_area[batch_idx, fid] = cp.norm() / 2

        ia, ib, ic = self.mesh.faces_vid[batch_idx, fid]
        a, b, c = self.vertices_rest_pos[batch_idx, ia], self.vertices_rest_pos[batch_idx, ib], self.vertices_rest_pos[batch_idx, ic]

        tmp_T = ti.Matrix.cols([b - a, c - a])
        tmp_TT = tmp_T.transpose()

        tmp_TTT = tmp_TT @ tmp_T # [2, 2]
        if ti.abs(tmp_TTT.determinant()) > tmp_TTT.norm_sqr() * 1e-7:
            TTT_inv = tmp_TTT.inverse()
            tmp_N = safe_normalized(cp, self.dx_eps ** 2)

            tmp_TTT43_0 = -ti.Matrix([[1, 1]]) @ TTT_inv @ tmp_TT
            tmp_TTT43_12 = TTT_inv @ tmp_TT
            tmp_TTT43_3 = tmp_N

            self.TTT43[batch_idx, fid][0, :] = tmp_TTT43_0[0, :]
            self.TTT43[batch_idx, fid][1:3, :] = tmp_TTT43_12
            self.TTT43[batch_idx, fid][3, :] = tmp_TTT43_3
        else:
            self.TTT43[batch_idx, fid] = ti.Matrix.zero(ti.f32, 4, 3)

    @ti.kernel
    def _init_face_property_kernel(self):
        nf_max = vec_max_func(self.mesh.n_faces, self.batch_size)
        for batch_idx, fid in ti.ndrange(self.batch_size, nf_max):
            if fid < self.mesh.n_faces[batch_idx]:
                self.update_face_property_func(batch_idx, fid)

    @ti.func
    def get_dF_dx_func(self, batch_idx, fid, w, i, j, k) -> ti.f32:
        """Return dF_{fid}^{j, k} / dx_{vid}^{i} with vid=f2v[fid][w]"""
        # for implicit integral
        return self.dF_dx[batch_idx, fid].get(w, i, j, k)

    @ti.func
    def get_deformation_gradient_func(self, batch_idx, fid: ti.i32) -> ti.Matrix:
        v0id, v1id, v2id = self.mesh.faces_vid[batch_idx, fid]
        x0 = self.vertices_pos[batch_idx, v0id]
        x1 = self.vertices_pos[batch_idx, v1id]
        x2 = self.vertices_pos[batch_idx, v2id]
        normalized = self.get_face_normalized_func(batch_idx, fid)
        tmp_X = ti.Matrix.cols([x0, x1, x2, normalized])
        ret_val = ti.Matrix.identity(ti.f32, 3)
        if self.TTT43[batch_idx, fid].any():
            ret_val = tmp_X @ self.TTT43[batch_idx, fid]
        return ret_val

    @ti.func
    def get_dihedral_func(self, batch_idx, eid: ti.i32) -> ti.f32:
        """
        calculate theta = n1 x n2 along (x2 - x1)
        IMPORTANT: consider chirality.
        Example:
        self.edges_vid_fid[1] = v1id, v2id, f1id, f2id = 1, 2, 100, 101
        self.faces_vid[100] = 1, 2, 3 # chirality = +1
        self.faces_vid[100] = 2, 1, 3 # chirality = -1
        """
        v1id, v2id, f1id, f2id = self.mesh.edges_vid_fid[batch_idx, eid]
        ret_val = 0.0
        chirality = 1.0
        if f2id != -1:
            cp1 = self.get_face_cross_product_func(batch_idx, f1id)
            cp2 = self.get_face_cross_product_func(batch_idx, f2id)
            n1 = safe_normalized(cp1, self.dx_eps ** 2)
            n2 = safe_normalized(cp2, self.dx_eps ** 2)
            x12 = safe_normalized(self.vertices_pos[batch_idx, v2id] -
                                  self.vertices_pos[batch_idx, v1id], self.dx_eps)
            sine = n2.cross(n1).dot(x12)
            cosine = n2.dot(n1)
            theta = ti.atan2(sine, cosine)

            f1_v1id, f1_v2id, f1_v3id = self.mesh.faces_vid[batch_idx, f1id]
            if (v2id == f1_v1id and v1id == f1_v2id) or \
               (v2id == f1_v2id and v1id == f1_v3id) or \
               (v2id == f1_v3id and v1id == f1_v1id):
                chirality = -1.0

            ret_val = theta * chirality
        return ret_val

    @ti.func
    def get_dtheta_dX_func(self, batch_idx, eid: ti.i32) -> ti.Matrix:
        """return a 4x3 matrix """
        ret_val = ti.Matrix.zero(dt=ti.f32, n=4, m=3)
        v1id, v2id, f1id, f2id = self.mesh.edges_vid_fid[batch_idx, eid]
        if f2id != -1:
            v3id = self.mesh.find_opp_vert_on_face_func(batch_idx, f1id, v1id, v2id)
            v4id = self.mesh.find_opp_vert_on_face_func(batch_idx, f2id, v1id, v2id)

            x1 = self.vertices_pos[batch_idx, v1id]
            x2 = self.vertices_pos[batch_idx, v2id]
            x3 = self.vertices_pos[batch_idx, v3id]
            x4 = self.vertices_pos[batch_idx, v4id]

            h1 = get_distance_func(x3, x1, x2, False, self.dx_eps)
            h2 = get_distance_func(x4, x1, x2, False, self.dx_eps)

            n1 = self.get_face_normalized_func(batch_idx, f1id)
            n2 = self.get_face_normalized_func(batch_idx, f2id)

            w_f1 = get_2D_barycentric_weights_func(x3, x1, x2, self.dx_eps)
            w_f2 = get_2D_barycentric_weights_func(x4, x1, x2, self.dx_eps)

            dtheta_dX1 = - w_f1[0] * n1 / ti.max(h1, self.dx_eps) - \
                w_f2[0] * n2 / ti.max(h2, self.dx_eps)
            dtheta_dX2 = - w_f1[1] * n1 / ti.max(h1, self.dx_eps) - \
                w_f2[1] * n2 / ti.max(h2, self.dx_eps)
            dtheta_dX3 = n1 / ti.max(h1, self.dx_eps)
            dtheta_dX4 = n2 / ti.max(h2, self.dx_eps)

            ret_val[0, :] = dtheta_dX1
            ret_val[1, :] = dtheta_dX2
            ret_val[2, :] = dtheta_dX3
            ret_val[3, :] = dtheta_dX4

        return ret_val

    @ti.func
    def get_edge_bending_coeff_func(self, batch_idx, eid: ti.i32) -> ti.f32:
        """E_b = 0.5 * coeff * (theta - rest_angle) ^ 2"""
        ret_val = 0.0
        v1id, v2id, f1id, f2id = self.mesh.edges_vid_fid[batch_idx, eid]
        if f2id != -1:
            area12 = self.rest_area[batch_idx, f1id] + self.rest_area[batch_idx, f2id]
            edge_length = (
                self.vertices_rest_pos[batch_idx, v1id] - self.vertices_rest_pos[batch_idx, v2id]).norm()

            ret_val = self.kb * edge_length ** 2 / \
                (4.0 * ti.max(area12, self.dx_eps ** 2))
        return ret_val

    @ti.kernel
    def calculate_derivative_kernel(self):
        # calculate dF / dx
        nf_max = vec_max_func(self.mesh.n_faces, self.batch_size)
        for batch_idx, fid in ti.ndrange(self.batch_size, nf_max):
            if fid < self.mesh.n_faces[batch_idx]:
                n = self.get_face_normalized_func(batch_idx, fid)
                x0 = self.vertices_pos[batch_idx, self.mesh.faces_vid[batch_idx, fid][0]]
                x1 = self.vertices_pos[batch_idx, self.mesh.faces_vid[batch_idx, fid][1]]
                x2 = self.vertices_pos[batch_idx, self.mesh.faces_vid[batch_idx, fid][2]]

                h0 = get_distance_vec_func(x0, x1, x2, False, self.dx_eps)
                h1 = get_distance_vec_func(x1, x2, x0, False, self.dx_eps)
                h2 = get_distance_vec_func(x2, x0, x1, False, self.dx_eps)

                for j, k in ti.ndrange(3, 3):
                    for i in range(3):
                        v0 = self.TTT43[batch_idx, fid][3, k] * h0[j] * \
                            n[i] / ti.max(h0.norm_sqr(), self.dx_eps ** 2)
                        v1 = self.TTT43[batch_idx, fid][3, k] * h1[j] * \
                            n[i] / ti.max(h1.norm_sqr(), self.dx_eps ** 2)
                        v2 = self.TTT43[batch_idx, fid][3, k] * h2[j] * \
                            n[i] / ti.max(h2.norm_sqr(), self.dx_eps ** 2)

                        if i == j:
                            v0 += self.TTT43[batch_idx, fid][0, k]
                            v1 += self.TTT43[batch_idx, fid][1, k]
                            v2 += self.TTT43[batch_idx, fid][2, k]

                        self.dF_dx[batch_idx, fid].set(0, i, j, k, v0)
                        self.dF_dx[batch_idx, fid].set(1, i, j, k, v1)
                        self.dF_dx[batch_idx, fid].set(2, i, j, k, v2)

                self.deform_grad[batch_idx, fid] = self.get_deformation_gradient_func(batch_idx, fid)
                self.U_f[batch_idx, fid], self.S_f[batch_idx, fid], self.V_f[batch_idx, fid], self.dU_f[batch_idx, fid], \
                    self.dS_f[batch_idx, fid], self.dV_f[batch_idx, fid] = dsvd_func(
                        self.deform_grad[batch_idx, fid], DEFORM_GRAD_DSVD_TOL)
                self.stretch_energy[batch_idx, fid] = self.lda / 2 * \
                    (self.S_f[batch_idx, fid][0, 0] + self.S_f[batch_idx, fid]
                    [1, 1] + self.S_f[batch_idx, fid][2, 2] - 3.0) ** 2
                for i in range(3):
                    self.stretch_energy[batch_idx, fid] += (self.S_f[batch_idx, fid]
                                                [i, i] - 1.0) ** 2 * self.mu
                self.stretch_energy[batch_idx, fid] *= self.h * self.rest_area[batch_idx, fid]

    @ti.kernel
    def calculate_ddpsi_dFdF_kernel(self):
        nf_max = vec_max_func(self.mesh.n_faces, self.batch_size)
        for batch_idx, fid in ti.ndrange(self.batch_size, nf_max):
            if fid < self.mesh.n_faces[batch_idx]:
                # Deformation Gradient F_f
                # F_f = self.get_deformation_gradient_func(fid)

                U_f, S_f, V_f, dU_f, dS_f, dV_f = self.U_f[batch_idx, fid], self.S_f[batch_idx, fid], self.V_f[batch_idx, fid], self.dU_f[batch_idx, fid], \
                    self.dS_f[batch_idx, fid], self.dV_f[batch_idx, fid]

                # dSi = d psi / d Si
                dS = ti.Matrix.zero(ti.f32, n=3, m=3)
                for s in range(3):
                    dS[s, s] = 2 * self.mu * (S_f[s, s] - 1.0) + \
                        self.lda * (S_f[0, 0] + S_f[1, 1] + S_f[2, 2] - 3.0)

                U_dS_V = U_f @ dS @ V_f.transpose()  # d psi / d F
                for k, l in ti.ndrange(3, 3):
                    for w, i in ti.ndrange(3, 3):
                        coeff = U_dS_V[k, l] * self.h * self.rest_area[batch_idx, fid]
                        self.elastic_force[batch_idx, self.mesh.faces_vid[batch_idx, fid][w]][i] -= coeff * \
                            self.get_dF_dx_func(batch_idx, fid, w, i, k, l)

                for m, n in ti.ndrange(3, 3):
                    # calculate d / d Fmn (d psi / d F)
                    dU_dF_mn = ti.Matrix.zero(ti.f32, n=3, m=3)
                    dV_dF_mn = ti.Matrix.zero(ti.f32, n=3, m=3)

                    for k, l in ti.ndrange(3, 3):
                        dU_dF_mn[k, l] = dU_f.get(k, l, m, n)
                        dV_dF_mn[k, l] = dV_f.get(k, l, m, n)

                    ds_mn = ti.Matrix.zero(ti.f32, n=3, m=3)

                    for j in range(3):
                        ds_mn[j, j] = dS_f.get(j, m, n) * 2.0 * self.mu + \
                            (dS_f.get(0, m, n) + dS_f.get(1, m, n) +
                            dS_f.get(2, m, n)) * self.lda

                    ddpsi_dFdF_mn = U_f @ ds_mn @ V_f.transpose() + \
                        U_f @ dS @ dV_dF_mn.transpose() + \
                        dU_dF_mn @ dS @ V_f.transpose()

                    for k, l in ti.ndrange(3, 3):
                        self.ddpsi_dFdF[batch_idx, fid, 3 * m + n, 3 * k + l] = \
                            ddpsi_dFdF_mn[k, l]
                        
        nf_max = vec_max_func(self.mesh.n_faces, self.batch_size)
        for batch_idx, fid, i, j in ti.ndrange(self.batch_size, nf_max, 9, 9):
            if fid < self.mesh.n_faces[batch_idx] and i <= j:
                self.ddpsi_dFdF[batch_idx, fid, i, j] = (self.ddpsi_dFdF[batch_idx, fid, i, j] +
                                              self.ddpsi_dFdF[batch_idx, fid, j, i]) / 2
                self.ddpsi_dFdF[batch_idx, fid, j, i] = self.ddpsi_dFdF[batch_idx, fid, i, j]

    @ti.kernel
    def fix_ddpsi_dFdF_kernel(self, alpha: ti.f32) -> bool:
        assert alpha >= 0.0, "[ERROR] in fix ddpsi_dFdF, displacement alpha:{} should not be negative.".format(
            alpha)
        self.is_all_success.fill(True)

        nf_max = vec_max_func(self.mesh.n_faces, self.batch_size)
        for batch_idx, fid in ti.ndrange(self.batch_size, nf_max):
            if fid < self.mesh.n_faces[batch_idx] and not self.is_fixed_flag[batch_idx]:
                A_mat = tiMatrix9x9()
                for i, j in ti.ndrange(9, 9):
                    A_mat.set(i, j, self.ddpsi_dFdF[batch_idx, fid, i, j])
                    assert ti.abs(
                        self.ddpsi_dFdF[batch_idx, fid, i, j] - self.ddpsi_dFdF[batch_idx, fid, j, i]) < 1e-6 * self.E
                is_success, L_mat, D_vec = ldlt_decompose_9x9_func(
                    A_mat, self.E * self.ldlt_relative_err)
                if not is_success:
                    self.is_all_success[batch_idx] = False
                    for i in range(9):
                        self.ddpsi_dFdF[batch_idx, fid, i, i] += alpha

        # update is fixed flag:
        is_all_batch_fixed = True
        for batch_idx in range(self.batch_size):
            if self.is_all_success[batch_idx]:
                self.is_fixed_flag[batch_idx] = True
            if not self.is_fixed_flag[batch_idx]:
                is_all_batch_fixed = False
        return is_all_batch_fixed
            

    @ti.kernel
    def add_stretch_with_hessian_fixed_kernel(self):
        nf_max = vec_max_func(self.mesh.n_faces, self.batch_size)
        for batch_idx, fid, m, n, k, l in ti.ndrange(self.batch_size, nf_max, 3, 3, 3, 3):
            if fid < self.mesh.n_faces[batch_idx]:
                coeff = self.ddpsi_dFdF[batch_idx, fid, 3 * m + n, 3 * k + l] * \
                    self.rest_area[batch_idx, fid] * self.h
                for a, i in ti.ndrange(3, 3):
                    for b, j in ti.ndrange(3, 3):
                        self.E_hessian_stretch[batch_idx, fid, a, b][i, j] += coeff * \
                            self.get_dF_dx_func(batch_idx, fid, a, i, m, n) * \
                            self.get_dF_dx_func(batch_idx, fid, b, j, k, l)

    def add_stretch_with_hessian(self):
        self.is_fixed_flag.fill(False)
        self.calculate_ddpsi_dFdF_kernel()
        alpha0 = self.hess_fix_init_disp
        for i in range(LDLT_MAX_LOOP_CNT):
            is_all_batch_fixed = self.fix_ddpsi_dFdF_kernel(float(alpha0 * (2 ** i)))
            if is_all_batch_fixed:
                break
            if i == LDLT_MAX_LOOP_CNT - 1:
                print("[ERROR] ldlt fix failed, there may exist some bugs.")
        self.add_stretch_with_hessian_fixed_kernel()

    @ti.kernel
    def add_bending_with_hessian_kernel(self):
        ne_max = vec_max_func(self.mesh.n_edges, self.batch_size)
        for batch_idx, eid in ti.ndrange(self.batch_size, ne_max):
            if eid < self.mesh.n_edges[batch_idx]:

                v1id, v2id, f1id, f2id = self.mesh.edges_vid_fid[batch_idx, eid]
                if f2id != -1:
                    v3id = self.mesh.find_opp_vert_on_face_func(batch_idx, f1id, v1id, v2id)
                    v4id = self.mesh.find_opp_vert_on_face_func(batch_idx, f2id, v1id, v2id)
                    vert = ti.Vector([v1id, v2id, v3id, v4id])

                    theta = self.get_dihedral_func(batch_idx, eid)
                    rest_angle = self.rest_angle[batch_idx, eid]
                    dtheta_dX = self.get_dtheta_dX_func(batch_idx, eid)
                    coeff = self.get_edge_bending_coeff_func(batch_idx, eid)
                    self.bending_energy[batch_idx, eid] = 0.5 * coeff * (theta - rest_angle) ** 2
                    assert coeff >= 0.0, "[ERROR] in add bending hessian, coeff={} < 0.0".format(
                        coeff)
                    for i, j in ti.ndrange(3, 4):
                        self.elastic_force[batch_idx, vert[j]][i] -= coeff * \
                            (theta - rest_angle) * dtheta_dX[j, i]
                        for k, l in ti.ndrange(3, 4):
                            self.E_hessian_bending[batch_idx, eid, j, l][i, k] += coeff * \
                                dtheta_dX[j, i] * dtheta_dX[l, k]

    @ti.kernel
    def assemble_hessian_kernel(self):
        # set zero
        self.hessian_sparse.set_zero_func()

        # assemble hessian
        nf_max = vec_max_func(self.mesh.n_faces, self.batch_size)
        for batch_idx, fid, i, j in ti.ndrange(self.batch_size, nf_max, 3, 3):
            if fid < self.mesh.n_faces[batch_idx]:
                vi, vj = self.mesh.faces_vid[batch_idx, fid][i], self.mesh.faces_vid[batch_idx, fid][j]
                for k, l in ti.ndrange(3, 3):
                    self.hessian_sparse.add_value_func(
                        batch_idx, 3 * vi + k, 3 * vj + l, self.E_hessian_stretch[batch_idx, fid, i, j][k, l])

        ne_max = vec_max_func(self.mesh.n_edges, self.batch_size)
        for batch_idx, eid in ti.ndrange(self.batch_size, ne_max):
            if eid < self.mesh.n_edges[batch_idx]:
                v1id, v2id, f1id, f2id = self.mesh.edges_vid_fid[batch_idx, eid]
                if f2id != -1:
                    v3id = self.mesh.find_opp_vert_on_face_func(batch_idx, f1id, v1id, v2id)
                    v4id = self.mesh.find_opp_vert_on_face_func(batch_idx, f2id, v1id, v2id)
                    vert = ti.Vector([v1id, v2id, v3id, v4id])
                    for i, j in ti.ndrange(4, 4):
                        vi, vj = vert[i], vert[j]
                        for k, l in ti.ndrange(3, 3):
                            self.hessian_sparse.add_value_func(
                                batch_idx, 3 * vi + k, 3 * vj + l, self.E_hessian_bending[batch_idx, eid, i, j][k, l])

    def calculate_elastic_force(self):
        self.elastic_force.fill(0.0)
        self.E_hessian_stretch.fill(0.0)
        self.E_hessian_bending.fill(0.0)
        self.calculate_derivative_kernel()
        self.add_stretch_with_hessian()
        self.add_bending_with_hessian_kernel()

    @ti.kernel
    def clear_damping_force_kernel(self):
        self.damping_force.fill(0.0)

    @ti.kernel
    def add_stretch_damping_force_kernel(self):
        ne_max = vec_max_func(self.mesh.n_edges, self.batch_size)
        for batch_idx, eid in ti.ndrange(self.batch_size, ne_max):
            if eid < self.mesh.n_edges[batch_idx]:
                v1id, v2id, f1id, f2id = self.mesh.edges_vid_fid[batch_idx, eid]
                r1 = self.vertices_pos[batch_idx, v1id]
                r2 = self.vertices_pos[batch_idx, v2id]
                r12_normalized = safe_normalized(r2 - r1, self.dx_eps)

                v1 = self.vertices_vel[batch_idx, v1id]
                v2 = self.vertices_vel[batch_idx, v2id]
                v1p = v1.dot(r12_normalized) * r12_normalized
                v2p = v2.dot(r12_normalized) * r12_normalized
                vp = v1p - v2p

                m_reduced = 1.0 / (1.0 / self.vertices_mass[batch_idx, v1id] +
                                1.0 / self.vertices_mass[batch_idx, v2id])
                f0 = m_reduced / self.stretch_relax_t

                self.damping_force[batch_idx, v1id] = -f0 * vp
                self.damping_force[batch_idx, v2id] = +f0 * vp

    @ti.kernel
    def add_bending_damping_force_kernel(self):
        ne_max = vec_max_func(self.mesh.n_edges, self.batch_size)
        for batch_idx, eid in ti.ndrange(self.batch_size, ne_max):
            if eid < self.mesh.n_edges[batch_idx]:
                v1id, v2id, f1id, f2id = self.mesh.edges_vid_fid[batch_idx, eid]
                if f2id != -1:
                    v3id = self.mesh.find_opp_vert_on_face_func(batch_idx, f1id, v1id, v2id)
                    v4id = self.mesh.find_opp_vert_on_face_func(batch_idx, f2id, v1id, v2id)

                    x1 = self.vertices_pos[batch_idx, v1id]
                    x2 = self.vertices_pos[batch_idx, v2id]
                    x3 = self.vertices_pos[batch_idx, v3id]
                    x4 = self.vertices_pos[batch_idx, v4id]

                    h1 = get_distance_func(x3, x1, x2, False, self.dx_eps)
                    h2 = get_distance_func(x4, x1, x2, False, self.dx_eps)

                    n1 = self.get_face_normalized_func(batch_idx, f1id)
                    n2 = self.get_face_normalized_func(batch_idx, f2id)

                    w_f1 = get_2D_barycentric_weights_func(x3, x1, x2, self.dx_eps)
                    w_f2 = get_2D_barycentric_weights_func(x4, x1, x2, self.dx_eps)

                    dtheta_dX1 = - w_f1[0] * n1 / ti.max(h1, self.dx_eps) - \
                        w_f2[0] * n2 / ti.max(h2, self.dx_eps)
                    dtheta_dX2 = - w_f1[1] * n1 / ti.max(h1, self.dx_eps) - \
                        w_f2[1] * n2 / ti.max(h2, self.dx_eps)
                    dtheta_dX3 = n1 / ti.max(h1, self.dx_eps)
                    dtheta_dX4 = n2 / ti.max(h2, self.dx_eps)

                    omega = \
                        dtheta_dX1.dot(self.vertices_vel[batch_idx, v1id]) + \
                        dtheta_dX2.dot(self.vertices_vel[batch_idx, v2id]) + \
                        dtheta_dX3.dot(self.vertices_vel[batch_idx, v3id]) + \
                        dtheta_dX4.dot(self.vertices_vel[batch_idx, v4id])

                    m_reduced = 1.0 / (1.0 / self.vertices_mass[batch_idx, v1id] +
                                    1.0 / self.vertices_mass[batch_idx, v2id] +
                                    1.0 / self.vertices_mass[batch_idx, v3id] +
                                    1.0 / self.vertices_mass[batch_idx, v4id])
                    h_reduced = 1.0 / (1.0 / h1 + 1.0 / h2)
                    edge_inertia = h_reduced ** 2 * m_reduced
                    torq = -omega * edge_inertia / self.bending_relax_t

                    self.damping_force[batch_idx, v3id] += torq * n1 / ti.max(h1, self.dx_eps)
                    self.damping_force[batch_idx, v4id] += torq * n2 / ti.max(h2, self.dx_eps)
                    self.damping_force[batch_idx, v1id] += torq * (-w_f1[0] * n1 / ti.max(h1, self.dx_eps) +
                                                        -w_f2[0] * n2 / ti.max(h2, self.dx_eps))
                    self.damping_force[batch_idx, v2id] += torq * (-w_f1[1] * n1 / ti.max(h1, self.dx_eps) +
                                                        -w_f2[1] * n2 / ti.max(h2, self.dx_eps))

    def calculate_damping_force(self):
        self.clear_damping_force_kernel()
        self.add_stretch_damping_force_kernel()
        self.add_bending_damping_force_kernel()

    @ti.kernel
    def update_bending_plasticity_kernel(self, dt: float):
        ne_max = vec_max_func(self.mesh.n_edges, self.batch_size)
        for batch_idx, eid in ti.ndrange(self.batch_size, ne_max):
            if eid < self.mesh.n_edges[batch_idx]:
                v1id, v2id, f1id, f2id = self.mesh.edges_vid_fid[batch_idx, eid]
                if f2id != -1:
                    theta = self.get_dihedral_func(batch_idx, eid)
                    if theta > self.bending_yield_criterion:
                        self.rest_angle[batch_idx, eid] = ti.min(
                            self.rest_angle[batch_idx, eid] + dt * self.bending_yield_rate,
                            theta,
                        )
                    elif theta < -self.bending_yield_criterion:
                        self.rest_angle[batch_idx, eid] = ti.max(
                            self.rest_angle[batch_idx, eid] - dt * self.bending_yield_rate,
                            theta,
                        )

    def update_plasticity(self, dt: float):
        self.update_bending_plasticity_kernel(dt)

    @ti.kernel
    def get_total_energy_kernel(self, gravity: ti.template(), total_energy: ti.template()):
        """
        Args:
            gravity: [3, ]
            total_energy: [B, 4]
        """
        total_energy.fill(0.0)
        nf_max = vec_max_func(self.mesh.n_faces, self.batch_size)
        for batch_idx, fid in ti.ndrange(self.batch_size, nf_max):
            if fid < self.mesh.n_faces[batch_idx]:
                total_energy[batch_idx, 0] += self.stretch_energy[batch_idx, fid]

        ne_max = vec_max_func(self.mesh.n_edges, self.batch_size)
        for batch_idx, eid in ti.ndrange(self.batch_size, ne_max):
            if eid < self.mesh.n_edges[batch_idx]:
                total_energy[batch_idx, 1] += self.bending_energy[batch_idx, eid]

        nv_max = vec_max_func(self.mesh.n_vertices, self.batch_size)
        for batch_idx, vid in ti.ndrange(self.batch_size, nv_max):
            if vid < self.mesh.n_vertices[batch_idx]:
                for i in range(3):
                    total_energy[batch_idx, 2] += -self.vertices_mass[batch_idx, vid] * \
                        self.vertices_pos[batch_idx, vid][i] * gravity[i]
                total_energy[batch_idx, 3] += 0.5 * self.vertices_mass[batch_idx, vid] * \
                    self.vertices_vel[batch_idx, vid].norm_sqr()

    @ti.func
    def estimate_delta_velocity_func(self, dv):
        """
        Estimate delta velocity and stored in dv.

        Args:
            - dv: ti.Field, shape=(B, nv * 3)
        """
        dv.fill(0.0)

        nv_max = vec_max_func(self.mesh.n_vertices, self.batch_size)
        for batch_idx, vid in ti.ndrange(self.batch_size, nv_max):
            if vid < self.mesh.n_vertices[batch_idx]:
                n = self.get_vertex_norm_func(batch_idx, vid)
                vertex_dv = n.dot(self.vertices_vel[batch_idx, vid]) * n - \
                    self.vertices_vel[batch_idx, vid]

                for i in ti.static(range(3)):
                    dv[batch_idx, vid * 3 + i] = vertex_dv[i]

    #######################################################
    # Section: Topological Change
    #######################################################

    @ti.func
    def add_cloth_vertex_func(self, batch_idx, new_pos, new_rest_pos, new_vel) -> ti.i32:
        """
        Add a new vertex and return new vertex id.
        set: pos, rest_pos, vel, n_vertices.
        not set: mass, vertices_eid, vertices_fid.
        """
        old_n = self.mesh.add_vertex_func(batch_idx)
        self.vertices_pos[batch_idx, old_n] = new_pos
        self.vertices_rest_pos[batch_idx, old_n] = new_rest_pos
        self.vertices_vel[batch_idx, old_n] = new_vel

        return old_n

    @ti.func
    def add_cloth_face_func(self, batch_idx, new_vid) -> ti.i32:
        """
        Add a new face and return new face id.
        set: faces_vid, n_faces.
        not set: TTT43-matrix, rest_area
        """
        return self.mesh.add_face_func(batch_idx, new_vid)

    @ti.func
    def add_cloth_edge_func(self, batch_idx, v1id, v2id, f1id, f2id) -> ti.i32:
        """
        add a new edge and return new edge id.
        set: edges_vid_fid, n_edges
        (automatically exchange v1,v2 if v1id > v2id)
        """
        return self.mesh.add_edge_func(batch_idx, v1id, v2id, f1id, f2id)

    @ti.func
    def insert_vertices_eid_func(self, batch_idx, eid: ti.i32, beta: ti.f32) -> ti.i32:
        """
        Add a new vertex on edge(eid) at 'beta*v1id + (1-beta)*v2id'.

        Return:
            New vertex id.
        """
        v1id, v2id, f1id, f2id = self.mesh.edges_vid_fid[batch_idx, eid]
        ret_val = -1

        if beta < _MINIMUM_CUT_RATIO_FOR_CLOTH or beta > 1 - _MINIMUM_CUT_RATIO_FOR_CLOTH:
            assert False, "[ERROR] insert_eid:{} beta {} not in (0,1). ".format(
                eid, beta)
        elif (f2id == -1) and (self.mesh.n_edges[batch_idx] + 2 > self.mesh.nmax_edges or
                               self.mesh.n_vertices[batch_idx] + 1 > self.mesh.nmax_vertices or
                               self.mesh.n_faces[batch_idx] + 1 > self.mesh.nmax_faces):
            assert False, "[ERROR] insert_eid:{} (f2id==-1) Reach maximum edges/vertices/faces.".format(
                eid)
        elif (f2id != -1) and (self.mesh.n_edges[batch_idx] + 3 > self.mesh.nmax_edges or
                               self.mesh.n_vertices[batch_idx] + 1 > self.mesh.nmax_vertices or
                               self.mesh.n_faces[batch_idx] + 2 > self.mesh.nmax_faces):
            assert False, "[ERROR] insert_eid:{} (f2id!=-1) Reach maximum edges/vertices/faces.".format(
                eid)
        else:
            self.mesh.topo_modified[batch_idx] = True

            # add vertex
            v3id = -1
            v5id = -1
            new_pos = self.vertices_pos[batch_idx, v1id] * beta + \
                self.vertices_pos[batch_idx, v2id] * (1 - beta)
            new_rest_pos = self.vertices_rest_pos[batch_idx, v1id] * beta + \
                self.vertices_rest_pos[batch_idx, v2id] * (1 - beta)
            new_vel = self.vertices_vel[batch_idx, v1id] * beta + \
                self.vertices_vel[batch_idx, v2id] * (1 - beta)
            v4id = self.add_cloth_vertex_func(batch_idx, new_pos, new_rest_pos, new_vel)
            ret_val = v4id

            # add faces
            old_face_vid = self.mesh.faces_vid[batch_idx, f1id]
            for i in ti.static(range(3)):
                if self.mesh.faces_vid[batch_idx, f1id][i] == v1id:
                    self.mesh.faces_vid[batch_idx, f1id][i] = v4id
                if old_face_vid[i] == v2id:
                    old_face_vid[i] = v4id
                elif old_face_vid[i] != v1id:
                    v3id = old_face_vid[i]
            if v3id == -1:
                assert False, "[ERROR] insert_eid:{} Not found vertex 3!".format(
                    eid)

            f3id = self.add_cloth_face_func(batch_idx, new_vid=old_face_vid)
            f4id = -1
            self.update_face_property_func(batch_idx, f1id)
            self.update_face_property_func(batch_idx, f3id)

            if f2id != -1:
                old_face_vid = self.mesh.faces_vid[batch_idx, f2id]
                for i in ti.static(range(3)):
                    if self.mesh.faces_vid[batch_idx, f2id][i] == v1id:
                        self.mesh.faces_vid[batch_idx, f2id][i] = v4id
                    if old_face_vid[i] == v2id:
                        old_face_vid[i] = v4id
                    elif old_face_vid[i] != v1id:
                        v5id = old_face_vid[i]
                if v5id == -1:
                    assert False, "[ERROR] insert_eid:{} Not found vertex 5!".format(
                        eid)
                f4id = self.add_cloth_face_func(batch_idx, new_vid=old_face_vid)

            # add edges
            self.mesh.edges_vid_fid[batch_idx, eid][0] = v1id
            self.mesh.edges_vid_fid[batch_idx, eid][1] = v4id
            self.mesh.edges_vid_fid[batch_idx, eid][2] = f3id
            self.mesh.edges_vid_fid[batch_idx, eid][3] = f4id

            e2id = self.add_cloth_edge_func(batch_idx, v2id, v4id, f1id, f2id)
            e3id = self.add_cloth_edge_func(batch_idx, v3id, v4id, f1id, f3id)
            e4id = -1
            if f2id != -1:
                e4id = self.add_cloth_edge_func(batch_idx, v5id, v4id, f2id, f4id)

            # update vertices
            self.mesh.replace_vertices_fid_func(batch_idx, v1id, f1id, f3id)
            if f2id != -1:
                self.mesh.replace_vertices_fid_func(batch_idx, v1id, f2id, f4id)

            self.mesh.replace_vertices_eid_func(batch_idx, v2id, eid, e2id)

            self.mesh.append_vertices_eid_func(batch_idx, v3id, e3id)
            self.mesh.append_vertices_fid_func(batch_idx, v3id, f3id)

            if f2id != -1:
                self.mesh.append_vertices_eid_func(batch_idx, v5id, e4id)
                self.mesh.append_vertices_fid_func(batch_idx, v5id, f4id)

            self.mesh.vertices_eid_cnt[batch_idx, v4id] = 3
            self.mesh.vertices_eid[batch_idx, v4id][0] = eid
            self.mesh.vertices_eid[batch_idx, v4id][1] = e2id
            self.mesh.vertices_eid[batch_idx, v4id][2] = e3id
            self.mesh.vertices_fid_cnt[batch_idx, v4id] = 2
            self.mesh.vertices_fid[batch_idx, v4id][0] = f1id
            self.mesh.vertices_fid[batch_idx, v4id][1] = f3id
            if f2id != -1:
                self.mesh.vertices_eid_cnt[batch_idx, v4id] = 4
                self.mesh.vertices_eid[batch_idx, v4id][3] = e4id
                self.mesh.vertices_fid_cnt[batch_idx, v4id] = 4
                self.mesh.vertices_fid[batch_idx, v4id][2] = f2id
                self.mesh.vertices_fid[batch_idx, v4id][3] = f4id

            self.update_vertex_mass_func(batch_idx, v1id)
            self.update_vertex_mass_func(batch_idx, v2id)
            self.update_vertex_mass_func(batch_idx, v4id)

            # update faces
            if f2id != -1:
                self.update_face_property_func(batch_idx, f2id)
                self.update_face_property_func(batch_idx, f4id)

            # update edges
            edge_v1_v5_changed = False
            edge_v1_v3_changed = False
            ti.loop_config(serialize=True)
            for i in range(self.mesh.vertices_eid_cnt[batch_idx, v1id]):
                tmp_eid = self.mesh.vertices_eid[batch_idx, v1id][i]
                tmp_v1id, tmp_v2id, tmp_f1id, tmp_f2id = self.mesh.edges_vid_fid[batch_idx, tmp_eid]

                if not edge_v1_v5_changed and f2id != -1 and \
                    ((tmp_v1id == v1id and tmp_v2id == v5id) or
                     (tmp_v2id == v1id and tmp_v1id == v5id)) and \
                        (self.mesh.edges_vid_fid[batch_idx, tmp_eid][2] == f2id or self.mesh.edges_vid_fid[batch_idx, tmp_eid][3] == f2id):
                    if self.mesh.edges_vid_fid[batch_idx, tmp_eid][2] == f2id:
                        self.mesh.edges_vid_fid[batch_idx, tmp_eid][2] = f4id
                    elif self.mesh.edges_vid_fid[batch_idx, tmp_eid][3] == f2id:
                        self.mesh.edges_vid_fid[batch_idx, tmp_eid][3] = f4id
                    else:
                        assert False, "[ERROR] insert_eid:{} Update edges error1. Not found v{} on e{}".format(
                            eid, tmp_eid, f2id)
                    edge_v1_v5_changed = True

                elif not edge_v1_v3_changed and \
                    ((tmp_v1id == v1id and tmp_v2id == v3id) or
                     (tmp_v2id == v1id and tmp_v1id == v3id)) and \
                        (self.mesh.edges_vid_fid[batch_idx, tmp_eid][2] == f1id or self.mesh.edges_vid_fid[batch_idx, tmp_eid][3] == f1id):
                    if self.mesh.edges_vid_fid[batch_idx, tmp_eid][2] == f1id:
                        self.mesh.edges_vid_fid[batch_idx, tmp_eid][2] = f3id
                    elif self.mesh.edges_vid_fid[batch_idx, tmp_eid][3] == f1id:
                        self.mesh.edges_vid_fid[batch_idx, tmp_eid][3] = f3id
                    else:
                        assert False, "[ERROR] insert_eid:{} Update edges error2. Not found v{} on e{}".format(
                            eid, tmp_eid, f1id)
                    edge_v1_v3_changed = True

            assert edge_v1_v3_changed and (edge_v1_v5_changed or f2id == -1)

        if self.print_info:
            print("[INFO] insert between e{} v{} v{} beta:{} success.".format(
                eid, v1id, v2id, beta))

        return ret_val

    @ti.kernel
    def insert_vertices_eid_kernel(self, batch_idx: ti.i32, eid: ti.i32, beta: ti.f32) -> ti.i32:
        """
        Add a new vertex on edge(eid) at 'beta*v1id + (1-beta)*v2id'.

        Return:
            New vertex id.
        """
        return self.insert_vertices_eid_func(batch_idx, eid, beta)

    @ti.func
    def insert_vertex_between_v1_v2_func(self, batch_idx, v1id: ti.i32, v2id: ti.i32, beta: ti.f32):
        """
        A simple wrapper function. Add a new vertex on edge at beta*v1id + (1-beta)*v2id. Return the new vertex id. 
        """
        new_beta = beta
        new_v1id = v1id
        new_v2id = v2id
        if v1id > v2id:
            new_beta = 1 - beta
            new_v1id = v2id
            new_v2id = v1id
        elif v1id == v2id:
            assert False, "[ERROR] insert between v1:{} v2:{} v1id=v2id.".format(
                v1id, v2id)

        ret_val = -1
        is_inserted = False
        ti.loop_config(serialize=True)
        for i in range(self.mesh.vertices_eid_cnt[batch_idx, new_v1id]):
            eid = self.mesh.vertices_eid[batch_idx, new_v1id][i]
            if not is_inserted and self.mesh.edges_vid_fid[batch_idx, eid][1] == new_v2id:
                ret_val = self.insert_vertices_eid_func(batch_idx, eid, new_beta)
                is_inserted = True
                break

        assert is_inserted, "[ERROR] insert between v1:{} v2:{} failed.".format(
            v1id, v2id)
        return ret_val

    @ti.kernel
    def insert_vertex_between_v1_v2_kernel(self, batch_idx: ti.i32, v1id: ti.i32, v2id: ti.i32, beta: ti.f32):
        """
        A simple wrapper function.
        Add a new vertex on edge at beta*v1id + (1-beta)*v2id.
        """
        self.insert_vertex_between_v1_v2_func(batch_idx, v1id, v2id, beta)

    @ti.func
    def split_from_v1_to_v3_func(self, batch_idx, v1id: ti.i32, v3id: ti.i32):
        """assume v1 is an outside vertex, v3 is an inside vertex"""

        if not (self.mesh.n_edges[batch_idx] + 1 <= self.mesh.nmax_edges and
                self.mesh.n_vertices[batch_idx] + 1 <= self.mesh.nmax_vertices):
            assert False, "[ERROR] cannot split v1={} to v3={}, edges or vertices reach maximum.".format(
                v1id, v3id)
        else:
            assert self.mesh.is_outside_vertex_func(batch_idx, v1id), \
                "[ERROR] cannot split v1={} to v3={}, v1 is not an outside vertex.".format(
                v1id, v3id)
            assert self.mesh.is_inside_vertex_func(batch_idx, v3id), \
                "[ERROR] cannot split v1={} to v3={}, v3 is not an inside vertex.".format(
                v1id, v3id)
            assert self.mesh.is_connect_func(batch_idx, v1id, v3id)[0] == 1, \
                "[ERROR] connectivity detection failed. v1={} to v3={}".format(
                v1id, v3id)

            self.mesh.topo_modified[batch_idx] = True

            # add new vertex
            new_pos = self.vertices_pos[batch_idx, v1id]
            new_rest_pos = self.vertices_rest_pos[batch_idx, v1id]
            new_vel = self.vertices_vel[batch_idx, v1id]
            v2id = self.add_cloth_vertex_func(batch_idx, new_pos, new_rest_pos, new_vel)

            # add new edge
            e1id = -1
            ti.loop_config(serialize=True)
            for i in range(self.mesh.vertices_eid_cnt[batch_idx, v1id]):
                tmp_eid = self.mesh.vertices_eid[batch_idx, v1id][i]
                if ti.min(v1id, v3id) == self.mesh.edges_vid_fid[batch_idx, tmp_eid][0] and \
                        ti.max(v1id, v3id) == self.mesh.edges_vid_fid[batch_idx, tmp_eid][1]:
                    e1id = tmp_eid
                    break

            if e1id == -1:
                assert False, "[ERROR] split v1={} to v3={} e1id not found".format(
                    v1id, v3id)

            f1id = self.mesh.edges_vid_fid[batch_idx, e1id][2]
            f2id = self.mesh.edges_vid_fid[batch_idx, e1id][3]

            self.mesh.edges_vid_fid[batch_idx, e1id][3] = -1
            e2id = self.add_cloth_edge_func(batch_idx, v3id, v2id, f2id, -1)

            # add new face
            pass

            # update vertex
            u1id = self.mesh.find_opp_vert_on_face_func(batch_idx, f1id, v1id, v3id)
            u2id = self.mesh.find_opp_vert_on_face_func(batch_idx, f2id, v1id, v3id)

            self.mesh.remove_vertices_fid_func(batch_idx, v1id, f2id)
            self.mesh.append_vertices_fid_func(batch_idx, v2id, f2id)

            self.mesh.append_vertices_eid_func(batch_idx, v2id, e2id)
            self.mesh.append_vertices_eid_func(batch_idx, v3id, e2id)

            prev_uid = u2id
            prev_faceid = f2id

            # update vertex, edge, face in a while loop

            # update around v1
            loop_cnt = ti.max(
                self.mesh.vertices_fid_cnt[batch_idx, v1id], self.mesh.vertices_eid_cnt[batch_idx, v1id]) + 10
            ti.loop_config(serialize=True)
            for _ in range(loop_cnt):
                # update prev_faces_vid
                for i in ti.static(range(3)):
                    if self.mesh.faces_vid[batch_idx, prev_faceid][i] == v1id:
                        self.mesh.faces_vid[batch_idx, prev_faceid][i] = v2id

                '''if prev_uid == u1id:
                    break'''

                # calculate now_eid
                now_eid = -1
                ti.loop_config(serialize=True)
                for i in range(self.mesh.vertices_eid_cnt[batch_idx, v1id]):
                    tmp_eid = self.mesh.vertices_eid[batch_idx, v1id][i]
                    tmp_u1 = self.mesh.edges_vid_fid[batch_idx, tmp_eid][0]
                    tmp_u2 = self.mesh.edges_vid_fid[batch_idx, tmp_eid][1]
                    tmp_f1 = self.mesh.edges_vid_fid[batch_idx, tmp_eid][2]
                    tmp_f2 = self.mesh.edges_vid_fid[batch_idx, tmp_eid][3]
                    if ((tmp_u1 == v1id and tmp_u2 == prev_uid) or
                            (tmp_u2 == v1id and tmp_u1 == prev_uid)) and \
                            (tmp_f1 == prev_faceid or tmp_f2 == prev_faceid):
                        now_eid = tmp_eid
                        self.mesh.edges_vid_fid[batch_idx, tmp_eid][0] = ti.min(
                            prev_uid, v2id)
                        self.mesh.edges_vid_fid[batch_idx, tmp_eid][1] = ti.max(
                            prev_uid, v2id)
                        break
                self.mesh.append_vertices_eid_func(batch_idx, v2id, now_eid)
                self.mesh.remove_vertices_eid_func(batch_idx, v1id, now_eid)

                # update now_eid

                # find next face
                if self.mesh.edges_vid_fid[batch_idx, now_eid][3] == -1:
                    break
                else:
                    now_faceid = -1
                    if self.mesh.edges_vid_fid[batch_idx, now_eid][2] == prev_faceid:
                        now_faceid = self.mesh.edges_vid_fid[batch_idx, now_eid][3]
                    else:
                        now_faceid = self.mesh.edges_vid_fid[batch_idx, now_eid][2]
                    self.mesh.append_vertices_fid_func(batch_idx, v2id, now_faceid)
                    self.mesh.remove_vertices_fid_func(batch_idx, v1id, now_faceid)

                    now_uid = self.mesh.find_opp_vert_on_face_func(
                        batch_idx, now_faceid, v1id, prev_uid)

                    prev_uid = now_uid
                    prev_faceid = now_faceid

                if _ == loop_cnt - 1:
                    print(
                        "[ERROR] split from v1 to v3, vertices v1 loop failed, there may exist some bugs.")

            self.update_vertex_mass_func(batch_idx, v1id)
            self.update_vertex_mass_func(batch_idx, v2id)

            if self.print_info:
                print("[INFO] split from v1:{} to v3:{} success.".format(v1id, v3id))

    @ti.kernel
    def split_from_v1_to_v3_kernel(self, batch_idx: ti.i32, v1id: ti.i32, v3id: ti.i32):
        self.split_from_v1_to_v3_func(batch_idx, v1id, v3id)

    @ti.func
    def split_both_v1_and_v3_func(self, batch_idx, v1id: ti.i32, v3id: ti.i32):
        """assume both v1 & v3 are outside vertices"""

        if not (self.mesh.n_edges[batch_idx] + 1 <= self.mesh.nmax_edges and
                self.mesh.n_vertices[batch_idx] + 2 <= self.mesh.nmax_vertices):
            assert False, "[ERROR] cannot split v1={} to v3={}, edges or vertices reach maximum.".format(
                v1id, v3id)

        else:
            assert self.mesh.is_outside_vertex_func(batch_idx, v1id), \
                "[ERROR] cannot split v1={} to v3={}, v1 is not an outside vertex.".format(
                    v1id, v3id)
            assert self.mesh.is_outside_vertex_func(batch_idx, v3id), \
                "[ERROR] cannot split v1={} to v3={}, v3 is not an outside vertex.".format(
                    v1id, v3id)
            assert self.mesh.is_connect_func(batch_idx, v1id, v3id)[0] == 1, \
                "[ERROR] connectivity detection failed. v1={} to v3={}".format(
                    v1id, v3id)

            self.mesh.topo_modified[batch_idx] = True

            # add new vertex
            new_pos = self.vertices_pos[batch_idx, v1id]
            new_rest_pos = self.vertices_rest_pos[batch_idx, v1id]
            new_vel = self.vertices_vel[batch_idx, v1id]
            v2id = self.add_cloth_vertex_func(batch_idx, new_pos, new_rest_pos, new_vel)

            new_pos = self.vertices_pos[batch_idx, v3id]
            new_rest_pos = self.vertices_rest_pos[batch_idx, v3id]
            new_vel = self.vertices_vel[batch_idx, v3id]
            v4id = self.add_cloth_vertex_func(batch_idx, new_pos, new_rest_pos, new_vel)

            # add new edge
            e1id = -1
            ti.loop_config(serialize=True)
            for i in range(self.mesh.vertices_eid_cnt[batch_idx, v1id]):
                tmp_eid = self.mesh.vertices_eid[batch_idx, v1id][i]
                if ti.min(v1id, v3id) == self.mesh.edges_vid_fid[batch_idx, tmp_eid][0] and \
                        ti.max(v1id, v3id) == self.mesh.edges_vid_fid[batch_idx, tmp_eid][1]:
                    e1id = tmp_eid
                    break

            if e1id == -1:
                assert False, "[ERROR] split v1={} and v3={} e1id not found".format(
                    v1id, v3id)

            f1id = self.mesh.edges_vid_fid[batch_idx, e1id][2]
            f2id = self.mesh.edges_vid_fid[batch_idx, e1id][3]

            assert f2id != -1, "[ERROR] split v1={} and v3={} f2id=-1! e1id={}".format(
                v1id, v3id, e1id)

            self.mesh.edges_vid_fid[batch_idx, e1id][3] = -1
            e2id = self.add_cloth_edge_func(batch_idx, v2id, v4id, f2id, -1)

            # add new face
            pass

            # update vertex
            u1id = self.mesh.find_opp_vert_on_face_func(batch_idx, f1id, v1id, v3id)
            u2id = self.mesh.find_opp_vert_on_face_func(batch_idx, f2id, v1id, v3id)

            self.mesh.remove_vertices_fid_func(batch_idx, v1id, f2id)
            self.mesh.append_vertices_fid_func(batch_idx, v2id, f2id)

            self.mesh.remove_vertices_fid_func(batch_idx, v3id, f2id)
            self.mesh.append_vertices_fid_func(batch_idx, v4id, f2id)

            self.mesh.append_vertices_eid_func(batch_idx, v2id, e2id)
            self.mesh.append_vertices_eid_func(batch_idx, v4id, e2id)

            # update vertex, edge, face in a while loop

            # update around v1
            prev_uid = u2id
            prev_faceid = f2id

            loop_cnt = ti.max(
                self.mesh.vertices_fid_cnt[batch_idx, v1id], self.mesh.vertices_eid_cnt[batch_idx, v1id]) + 10
            ti.loop_config(serialize=True)
            for _ in range(loop_cnt):
                # update prev_faces_vid
                for i in ti.static(range(3)):
                    if self.mesh.faces_vid[batch_idx, prev_faceid][i] == v1id:
                        self.mesh.faces_vid[batch_idx, prev_faceid][i] = v2id

                '''if prev_uid == u1id:
                    break'''

                # calculate now_eid
                now_eid = -1
                ti.loop_config(serialize=True)
                for i in range(self.mesh.vertices_eid_cnt[batch_idx, v1id]):
                    tmp_eid = self.mesh.vertices_eid[batch_idx, v1id][i]

                    tmp_u1 = self.mesh.edges_vid_fid[batch_idx, tmp_eid][0]
                    tmp_u2 = self.mesh.edges_vid_fid[batch_idx, tmp_eid][1]
                    tmp_f1 = self.mesh.edges_vid_fid[batch_idx, tmp_eid][2]
                    tmp_f2 = self.mesh.edges_vid_fid[batch_idx, tmp_eid][3]

                    if ((tmp_u1 == v1id and tmp_u2 == prev_uid) or
                            (tmp_u2 == v1id and tmp_u1 == prev_uid)) and \
                            (tmp_f1 == prev_faceid or tmp_f2 == prev_faceid):
                        now_eid = tmp_eid
                        self.mesh.edges_vid_fid[batch_idx, tmp_eid][0] = ti.min(
                            prev_uid, v2id)
                        self.mesh.edges_vid_fid[batch_idx, tmp_eid][1] = ti.max(
                            prev_uid, v2id)
                        break

                self.mesh.append_vertices_eid_func(batch_idx, v2id, now_eid)
                self.mesh.remove_vertices_eid_func(batch_idx, v1id, now_eid)

                # find next face
                if self.mesh.edges_vid_fid[batch_idx, now_eid][3] == -1:
                    break
                else:
                    now_faceid = -1
                    if self.mesh.edges_vid_fid[batch_idx, now_eid][2] == prev_faceid:
                        now_faceid = self.mesh.edges_vid_fid[batch_idx, now_eid][3]
                    else:
                        now_faceid = self.mesh.edges_vid_fid[batch_idx, now_eid][2]
                    self.mesh.append_vertices_fid_func(batch_idx, v2id, now_faceid)
                    self.mesh.remove_vertices_fid_func(batch_idx, v1id, now_faceid)

                    now_uid = self.mesh.find_opp_vert_on_face_func(
                        batch_idx, now_faceid, v1id, prev_uid)

                    prev_uid = now_uid
                    prev_faceid = now_faceid

                if _ == loop_cnt - 1:
                    print(
                        "[ERROR] split both vertices v1 loop failed, there may exist some bugs.")

            # update around v3
            prev_uid = u2id
            prev_faceid = f2id

            loop_cnt = ti.max(
                self.mesh.vertices_fid_cnt[batch_idx, v3id], self.mesh.vertices_eid_cnt[batch_idx, v3id]) + 10
            ti.loop_config(serialize=True)
            for _ in range(loop_cnt):
                # update prev_faces_vid
                for i in ti.static(range(3)):
                    if self.mesh.faces_vid[batch_idx, prev_faceid][i] == v3id:
                        self.mesh.faces_vid[batch_idx, prev_faceid][i] = v4id

                '''if prev_uid == u1id:
                    break'''

                # calculate now_eid
                now_eid = -1
                ti.loop_config(serialize=True)
                for i in range(self.mesh.vertices_eid_cnt[batch_idx, v3id]):
                    tmp_eid = self.mesh.vertices_eid[batch_idx, v3id][i]
                    tmp_u1 = self.mesh.edges_vid_fid[batch_idx, tmp_eid][0]
                    tmp_u2 = self.mesh.edges_vid_fid[batch_idx, tmp_eid][1]
                    tmp_f1 = self.mesh.edges_vid_fid[batch_idx, tmp_eid][2]
                    tmp_f2 = self.mesh.edges_vid_fid[batch_idx, tmp_eid][3]
                    if ((tmp_u1 == v3id and tmp_u2 == prev_uid) or
                            (tmp_u2 == v3id and tmp_u1 == prev_uid)) and \
                            (tmp_f1 == prev_faceid or tmp_f2 == prev_faceid):
                        now_eid = tmp_eid
                        self.mesh.edges_vid_fid[batch_idx, tmp_eid][0] = ti.min(
                            prev_uid, v4id)
                        self.mesh.edges_vid_fid[batch_idx, tmp_eid][1] = ti.max(
                            prev_uid, v4id)
                        break
                self.mesh.append_vertices_eid_func(batch_idx, v4id, now_eid)
                self.mesh.remove_vertices_eid_func(batch_idx, v3id, now_eid)

                # find next face
                if self.mesh.edges_vid_fid[batch_idx, now_eid][3] == -1:
                    break
                else:
                    now_faceid = -1
                    if self.mesh.edges_vid_fid[batch_idx, now_eid][2] == prev_faceid:
                        now_faceid = self.mesh.edges_vid_fid[batch_idx, now_eid][3]
                    else:
                        now_faceid = self.mesh.edges_vid_fid[batch_idx, now_eid][2]
                    self.mesh.append_vertices_fid_func(batch_idx, v4id, now_faceid)
                    self.mesh.remove_vertices_fid_func(batch_idx, v3id, now_faceid)

                    now_uid = self.mesh.find_opp_vert_on_face_func(
                        batch_idx, now_faceid, v3id, prev_uid)

                    prev_uid = now_uid
                    prev_faceid = now_faceid

                if _ == loop_cnt - 1:
                    print(
                        "[ERROR] split both vertices v3 loop failed, there may exist some bugs.")

            self.update_vertex_mass_func(batch_idx, v1id)
            self.update_vertex_mass_func(batch_idx, v2id)
            self.update_vertex_mass_func(batch_idx, v3id)
            self.update_vertex_mass_func(batch_idx, v4id)

            if self.print_info:
                print("[INFO] split both v1:{} and v3:{} success.".format(v1id, v3id))

    @ti.kernel
    def split_both_v1_and_v3_kernel(self, batch_idx: ti.i32, v1id: ti.i32, v3id: ti.i32):
        self.split_both_v1_and_v3_func(batch_idx, v1id, v3id)

    @ti.func
    def split_inside_v1_and_v3_func(self, batch_idx, v1id: ti.i32, v3id: ti.i32):
        """assume both v1 & v3 are inside vertices"""

        if not (self.mesh.n_edges[batch_idx] + 1 <= self.mesh.nmax_edges):
            assert False, "[ERROR] cannot split inside v1={} and v3={}, edges reach maximum.".format(
                v1id, v3id)

        else:
            assert self.mesh.is_inside_vertex_func(batch_idx, v1id), \
                "[ERROR] cannot split v1={} to v3={}, v1 is not an inside vertex.".format(
                    v1id, v3id)
            assert self.mesh.is_inside_vertex_func(batch_idx, v3id), \
                "[ERROR] cannot split v1={} to v3={}, v3 is not an inside vertex.".format(
                    v1id, v3id)
            assert self.mesh.is_connect_func(batch_idx, v1id, v3id)[0] == 1, \
                "[ERROR] connectivity detection failed. v1={} to v3={}".format(
                    v1id, v3id)

            self.mesh.topo_modified[batch_idx] = True

            eid_is_found = False
            ti.loop_config(serialize=True)
            for i in range(self.mesh.vertices_eid_cnt[batch_idx, v1id]):
                eid = self.mesh.vertices_eid[batch_idx, v1id][i]
                tmp_v1id, tmp_v2id, tmp_f1id, tmp_f2id = self.mesh.edges_vid_fid[batch_idx, eid]

                if (tmp_v1id == v1id and tmp_v2id == v3id) or \
                        (tmp_v2id == v1id and tmp_v1id == v3id):
                    assert tmp_f2id != - \
                        1, "[ERROR] split inside v1={} and v3={}, eid={} f2id={}, which shouldn't be -1".format(
                            v1id, v3id, eid, tmp_f2id)
                    eid_is_found += 1

                    # add new edge
                    e2id = self.mesh.add_edge_func(
                        batch_idx, tmp_v1id, tmp_v2id, tmp_f2id, -1)

                    # update edge
                    self.mesh.edges_vid_fid[batch_idx, eid][3] = -1

                    # update vertex
                    self.mesh.append_vertices_eid_func(batch_idx, tmp_v1id, e2id)
                    self.mesh.append_vertices_eid_func(batch_idx, tmp_v2id, e2id)

                    eid_is_found = True
                    break

            assert eid_is_found, "[ERROR] split inside v1={} and v3={} no edge is found.".format(
                v1id, v3id)

            if self.print_info:
                print(
                    "[INFO] split inside v1:{} and v3:{} success.".format(v1id, v3id))

    @ti.kernel
    def split_inside_v1_and_v3_kernel(self, batch_idx: ti.i32, v1id: ti.i32, v3id: ti.i32):
        self.split_inside_v1_and_v3_func(batch_idx, v1id, v3id)

    @ti.func
    def split_edge_func(self, batch_idx, eid: ti.i32):
        """
        Safely break an edge.
        """
        v1id, v2id, f1id, f2id = self.mesh.edges_vid_fid[batch_idx, eid]
        connect = self.mesh.is_connect_func(batch_idx, v1id, v2id)
        # if there is no artifect on mesh, we do split.
        if connect[0] == 1 and self.mesh.edges_vid_fid[batch_idx, connect[1]][2] != -1 \
                and self.mesh.edges_vid_fid[batch_idx, connect[1]][3] != -1:
            if self.mesh.is_inside_vertex_func(batch_idx, v1id) and self.mesh.is_outside_vertex_func(batch_idx, v2id):
                self.split_from_v1_to_v3_func(batch_idx, v2id, v1id)
            elif self.mesh.is_outside_vertex_func(batch_idx, v1id) and self.mesh.is_inside_vertex_func(batch_idx, v2id):
                self.split_from_v1_to_v3_func(batch_idx, v1id, v2id)
            elif self.mesh.is_outside_vertex_func(batch_idx, v1id) and self.mesh.is_outside_vertex_func(batch_idx, v2id):
                self.split_both_v1_and_v3_func(batch_idx, v1id, v2id)
            else:
                self.split_inside_v1_and_v3_func(batch_idx, v1id, v2id)
        elif self.print_info:
            print(
                "[INFO] Split e{} failed. Please check your code or mesh.".format(eid))

    @ti.kernel
    def split_edge_kernel(self, batch_idx: ti.i32, eid: ti.i32):
        """
        Safely break an edge.
        """
        self.split_edge_func(batch_idx, eid)

    '''def cut_vertices_pos_set_zero(self):
        self.cut_vertices_pos_cnt.fill(0)

    @ti.func
    def append_cut_vertices_pos_func(self, vid: ti.i32, rest_pos: ti.types.vector(3, ti.f32), pos: ti.types.vector(3, ti.f32)):
        old_cnt = ti.atomic_add(self.cut_vertices_pos_cnt[vid], 1)
        assert old_cnt < self.cut_vertices_pos_nmax, "[ERROR] number of cut vertices position reaches maximum."

        self.cut_vertices_rest_pos[vid, old_cnt] = rest_pos
        self.cut_vertices_pos[vid, old_cnt] = pos

    @ti.func
    def rest_pos_too_close_to_adjacent_vert_func(self, vid: ti.i32, rest_pos: ti.types.vector(3, ti.f32), distance_threshold: ti.f32) -> bool:
        ret_val = False
        for i in range(self.mesh.vertices_eid_cnt[vid]):
            eid = self.mesh.vertices_eid[vid][i]

            adja_vert = self.mesh.find_opp_vert_on_edge_func(eid, vid)
            adja_vert_rest_pos = self.vertices_rest_pos[adja_vert]
            dist = ti.math.length(adja_vert_rest_pos - rest_pos)

            if dist < distance_threshold:
                ret_val = True
        return ret_val

    @ti.func
    def update_vertex_adjacent_faces_func(self, vid: ti.f32):
        for i in range(self.mesh.vertices_fid_cnt[vid]):
            fid = self.mesh.vertices_fid[vid][i]
            self.update_face_property_func(fid)

    @ti.kernel
    def move_vertices_to_cut_position_kernel(self, cut_dx_tolerance: ti.f32):
        for vid in range(self.mesh.n_vertices[None]):
            if self.cut_vertices_pos_cnt[vid] >= 1 and self.mesh.is_inside_vertex_func(vid):
                rest_pos_avg = ti.Vector.zero(dt=ti.f32, n=3)
                pos_avg = ti.Vector.zero(dt=ti.f32, n=3)

                for i in range(self.cut_vertices_pos_cnt[vid]):
                    rest_pos_avg += self.cut_vertices_rest_pos[vid, i]
                    pos_avg += self.cut_vertices_pos[vid, i]

                rest_pos_avg /= self.cut_vertices_pos_cnt[vid]
                pos_avg /= self.cut_vertices_pos_cnt[vid]

                if not self.rest_pos_too_close_to_adjacent_vert_func(vid, rest_pos_avg, cut_dx_tolerance):
                    self.vertices_rest_pos[vid] = rest_pos_avg
                    self.vertices_pos[vid] = pos_avg

                    self.update_vertex_adjacent_faces_func(vid)

    @ti.func
    def reconnect_edge_func(self, eid: ti.i32):
        """Reconnect eid: [v1id, v2id, f1id, f2id] to eid: [v3id, v4id, f1id, f2id]"""
        v1id, v2id, f1id, f2id = self.mesh.edges_vid_fid[eid]
        if f2id != -1:
            mininum_new_face_area = self.min_reconnect_area

            v3id = self.mesh.find_opp_vert_on_face_func(f1id, v1id, v2id)
            v4id = self.mesh.find_opp_vert_on_face_func(f2id, v1id, v2id)

            v1_rest_pos = self.vertices_pos[v1id]
            v2_rest_pos = self.vertices_pos[v2id]
            v3_rest_pos = self.vertices_pos[v3id]
            v4_rest_pos = self.vertices_pos[v4id]

            n134 = (v3_rest_pos - v1_rest_pos).cross(v4_rest_pos - v1_rest_pos)
            n243 = (v4_rest_pos - v2_rest_pos).cross(v3_rest_pos - v2_rest_pos)

            if n134.dot(n243) > 0.0 and ti.math.length(n134) / 2 > mininum_new_face_area and \
                    ti.math.length(n243) / 2 > mininum_new_face_area:
                # v2f
                self.mesh.remove_vertices_fid_func(v1id, f2id)
                self.mesh.remove_vertices_fid_func(v2id, f1id)
                self.mesh.append_vertices_fid_func(v3id, f2id)
                self.mesh.append_vertices_fid_func(v4id, f1id)

                # v2e
                self.mesh.remove_vertices_eid_func(v1id, eid)
                self.mesh.remove_vertices_eid_func(v2id, eid)
                self.mesh.append_vertices_eid_func(v3id, eid)
                self.mesh.append_vertices_eid_func(v4id, eid)

                # f2v
                for i in ti.static(range(3)):
                    if self.mesh.faces_vid[f1id][i] == v2id:
                        self.mesh.faces_vid[f1id][i] = v4id

                    if self.mesh.faces_vid[f2id][i] == v1id:
                        self.mesh.faces_vid[f2id][i] = v3id

                # e2f & e2v
                ti.loop_config(serialize=True)
                for i in range(self.mesh.vertices_eid_cnt[v1id]):
                    tmp_eid = self.mesh.vertices_eid[v1id][i]
                    tmp_v1id, tmp_v2id, tmp_f1id, tmp_f2id = self.mesh.edges_vid_fid[tmp_eid]
                    if v4id == tmp_v1id or v4id == tmp_v2id:
                        if tmp_f1id == f2id:
                            self.mesh.edges_vid_fid[tmp_eid][2] = f1id
                            break
                        elif tmp_f2id == f2id:
                            self.mesh.edges_vid_fid[tmp_eid][3] = f1id
                            break

                ti.loop_config(serialize=True)
                for i in range(self.mesh.vertices_eid_cnt[v2id]):
                    tmp_eid = self.mesh.vertices_eid[v2id][i]
                    tmp_v1id, tmp_v2id, tmp_f1id, tmp_f2id = self.mesh.edges_vid_fid[tmp_eid]
                    if v3id == tmp_v1id or v3id == tmp_v2id:
                        if tmp_f1id == f1id:
                            self.mesh.edges_vid_fid[tmp_eid][2] = f2id
                            break
                        elif tmp_f2id == f1id:
                            self.mesh.edges_vid_fid[tmp_eid][3] = f2id
                            break

                self.mesh.edges_vid_fid[eid][0] = ti.min(v3id, v4id)
                self.mesh.edges_vid_fid[eid][1] = ti.max(v3id, v4id)

                # update

                self.update_face_property_func(f1id)
                self.update_face_property_func(f2id)

                self.update_vertex_mass_func(v1id)
                self.update_vertex_mass_func(v2id)
                self.update_vertex_mass_func(v3id)
                self.update_vertex_mass_func(v4id)

                self.mesh.topo_modified[None] = True

                if self.print_info:
                    print("[INFO] Reconnect edge:{} v1:{} v2:{} v3:{} v4:{} f1:{} f2:{}".format(
                        eid, v1id, v2id, v3id, v4id, f1id, f2id))

    @ti.kernel
    def reconnect_edge_kernel(self, eid: ti.i32):
        """Reconnect eid: [v1id, v2id, f1id, f2id] to eid: [v3id, v4id, f1id, f2id]"""
        self.reconnect_edge_func(eid)'''

    def topological_check(self):
        raise NotImplementedError
        self.mesh.topological_check()
        if self.print_info:
            print("[INFO] Topological check is completed. OK. ")

    def get_topo_info_str(self, batch_idx: int):
        topo_info_str = ""

        def get_v2f_str(batch_idx, vid):
            ret_str = "["
            for i in range(self.mesh.vertices_fid_cnt[batch_idx, vid]):
                ret_str += str(self.mesh.vertices_fid[batch_idx, vid][i]) + ","
            return ret_str + "]"

        def get_v2e_str(batch_idx, vid):
            ret_str = "["
            for i in range(self.mesh.vertices_eid_cnt[batch_idx, vid]):
                ret_str += str(self.mesh.vertices_eid[batch_idx, vid][i]) + ","
            return ret_str + "]"

        for vid in range(self.mesh.n_vertices[batch_idx]):
            topo_info_str += "vert{}:{} v2f:{} v2e:{}\n".format(vid,
                                                                self.vertices_pos[batch_idx, vid], get_v2f_str(batch_idx, vid), get_v2e_str(batch_idx, vid))

        for fid in range(self.mesh.n_faces[batch_idx]):
            topo_info_str += "face{}:{} ".format(fid, self.mesh.faces_vid[batch_idx, fid])
            if fid % 5 == 4 or fid == self.mesh.n_faces[batch_idx] - 1:
                topo_info_str += "\n"

        for eid in range(self.mesh.n_edges[batch_idx]):
            topo_info_str += "edge{}:{} ".format(eid,
                                                 self.mesh.edges_vid_fid[batch_idx, eid])
            if eid % 5 == 4 or eid == self.mesh.n_faces[batch_idx] - 1:
                topo_info_str += "\n"

        return topo_info_str

    def print_topo_info(self, batch_idx):
        print(self.get_topo_info_str(batch_idx))

    #######################################################
    # Section: Miscellaneous
    #######################################################

    def get_mesh(self) -> List[trimesh.Trimesh]:
        cloth_vert_np = self.vertices_pos.to_numpy()
        cloth_face_np = self.mesh.faces_vid.to_numpy()
        nv_np = self.mesh.n_vertices.to_numpy()
        nf_np = self.mesh.n_faces.to_numpy()
        return [trimesh.Trimesh(vertices=cloth_vert_np[batch_idx, :nv_np[batch_idx]], 
                                faces=cloth_face_np[batch_idx, :nf_np[batch_idx]],
                                process=True, validate=True)
                                for batch_idx in range(self.batch_size)]

    def vertices_pos_taichi_to_python(self):
        self.vertices_pos_python: np.ndarray = self.vertices_pos.to_numpy()
        self.vertices_rest_pos_python: np.ndarray = self.vertices_rest_pos.to_numpy()

    def reset(self):
        """
        Reset all configuration to initial state.
        - vert position & velocity
        - face properties
        """
        self.mesh.reset()

        # reset vertices position and velocity

        # current position of vertices
        self.vertices_pos.from_numpy(
            np.tile(
                np.pad(self.mesh.origin_mesh.vertices.astype(np.float32),
                   ((0, self.mesh.nmax_vertices - int(self.mesh.origin_mesh.vertices.shape[0])), (0, 0)), 
                   'constant', constant_values=0)[None, ...],
                reps=(self.batch_size, 1, 1)
        ))
        
        # initial position of vertices
        self.vertices_rest_pos.from_numpy(
            np.tile(
                np.pad(self.mesh.origin_mesh.vertices.astype(np.float32),
                   ((0, self.mesh.nmax_vertices - int(self.mesh.origin_mesh.vertices.shape[0])), (0, 0)), 
                   'constant', constant_values=0)[None, ...],
                reps=(self.batch_size, 1, 1)
        ))

        # current velocity of vertices
        self.vertices_vel.fill(0)

        # rest angle
        self.rest_angle.fill(0.)

        # reset face properties
        self._init_face_property_kernel()
        self._init_mass_kernel()

    def get_state(self) -> List[dict]:
        nv = self.mesh.n_vertices.to_numpy()
        ne = self.mesh.n_vertices.to_numpy()
        mesh = self.mesh.get_state()

        vp_np = self.vertices_pos.to_numpy()
        vrp_np = self.vertices_rest_pos.to_numpy()
        vv_np = self.vertices_vel.to_numpy()
        ra = self.rest_angle.to_numpy()

        return [{
            "class": "Cloth",
            "mesh": mesh[batch_size],
            "vertices_pos": slice_array(vp_np[batch_size, ...], nv[batch_size]),
            "vertices_rest_pos": slice_array(vrp_np[batch_size, ...], nv[batch_size]),
            "vertices_vel": slice_array(vv_np[batch_size, ...], nv[batch_size]),
            "rest_angle": slice_array(ra[batch_size, ...], ne[batch_size]),
        } for batch_size in range(self.batch_size)]

    def set_state(self, states: List[dict]) -> None:
        assert isinstance(states, list)
        assert isinstance(states[0], dict)
        assert len(states) == self.batch_size

        self.mesh.set_state([state["mesh"] for state in states])
        self.vertices_pos.from_numpy(np.array([pad_array(state["vertices_pos"], 
                                               self.mesh.nmax_vertices) for state in states]))
        self.vertices_rest_pos.from_numpy(np.array([pad_array(state["vertices_rest_pos"], 
                                                    self.mesh.nmax_vertices) for state in states]))
        self.vertices_vel.from_numpy(np.array([pad_array(state["vertices_vel"], 
                                               self.mesh.nmax_vertices) for state in states]))
        self.rest_angle.from_numpy(np.array([pad_array(state["rest_angle"], 
                                                       self.mesh.nmax_edges) for state in states]))

        self._init_face_property_kernel()
        self._init_mass_kernel()
