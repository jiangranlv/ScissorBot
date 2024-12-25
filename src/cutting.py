import taichi as ti

from typing import List
from omegaconf import DictConfig

from src.cloth import Cloth, MINIMUM_CUT_RATIO
from src.scissor import *
from src.maths import *
from src.utils import *
from src.cbase import *

MID_POINT_BC = 0.5


@ti.dataclass
class EdgeIntersect:
    edge_id: ti.i32
    bary_coor_edge: ti.f32


@ti.dataclass
class FaceIntersect:
    face_id: ti.i32
    bary_coor: ti.types.vector(3, ti.f32)


@ti.dataclass
class MinimalDist:
    face_id: ti.i32
    bary_coor: ti.types.vector(3, ti.f32)
    min_dist: ti.f32


class SplitVertex:
    def __init__(self, vert_id: int, cut_pos: float, splitted: bool) -> None:
        self.vert_id = int(vert_id)
        self.cut_pos = float(cut_pos)
        self.splitted = bool(splitted)

    def __repr__(self) -> str:
        return "[v:{} p:{:.6e} s:{}]".format(self.vert_id, self.cut_pos, self.splitted)

    def set_splitted(self):
        self.splitted = True


@ti.data_oriented
class Cutting(TiInteraction):
    def __init__(self, batch_size: int, cut_sim_cfg: DictConfig, collision_cfg: DictConfig, scissor_cfg: DictConfig, cloth: Cloth, scissor: Scissor) -> None:
        self.batch_size = batch_size
        
        # Miscellaneous
        self.dim = 3
        self.barrier_width = collision_cfg.barrier_width
        self.barrier_sdf_offset = collision_cfg.barrier_sdf_offset

        # Objects
        self.cloth = cloth
        assert isinstance(scissor, Scissor)
        self.scissor = scissor

        self.first_cut_backward_tol = cut_sim_cfg.first_cut_backward_tol
        self.use_last_split_vertex_id = cut_sim_cfg.use_last_split_vertex_id

        self.insert_in_plane_tol = cut_sim_cfg.insert_in_plane_tol
        self.insert_out_plane_tol = cut_sim_cfg.insert_out_plane_tol

        self.cut_min_length = cut_sim_cfg.cut_min_length

        self.stuck_in_plane_tol = cut_sim_cfg.stuck_in_plane_tol
        self.stuck_out_plane_tol = cut_sim_cfg.stuck_out_plane_tol

        self.disable_stuck = bool(cut_sim_cfg.get("disable_stuck", False))

        self.max_scissor_cloth_cut_angle = cut_sim_cfg.max_scissor_cloth_cut_angle

        self.min_cut_angle = scissor_cfg.min_cut_angle

        self.edge_intersect = EdgeIntersect.field(shape=self.cloth.mesh.nmax_edges)
        """[E, ]"""
        self.edge_intersect_cnt = ti.field(dtype=ti.i32, shape=())
        """[,]"""
        self.best_edge_intersect = EdgeIntersect.field(shape=())
        """[,]"""

        self.face_intersect = FaceIntersect.field(shape=self.cloth.mesh.nmax_faces)
        """[F, ]"""
        self.face_intersect_cnt = ti.field(dtype=ti.i32, shape=())
        """[,]"""
        self.face_min_dist = MinimalDist.field(shape=())
        """[,]"""

        self.edge_is_dirty = ti.field(dtype=bool, shape=self.cloth.mesh.nmax_edges)
        """[E, ]"""
        self.face_is_dirty = ti.field(dtype=bool, shape=self.cloth.mesh.nmax_faces)
        """[F, ]"""

        '''self.curr_split_vertex_list = [
            None for _ in range(self.scissor_num)]  # only for debug'''
        '''self.save_cutting_info = False''' # When false, disable self.split_edge_origin_coordinates and self.front_point_projection

        self.last_split_vertex_id:List[int] = [-1 for _ in range(self.batch_size)]

        '''self.split_edge_origin_coordinates = []
        self.front_point_projection = []'''

    def is_cut_state(self, scissor_new_pose: dict, scissor_velocity: dict) -> bool:
        return scissor_velocity["joint_0"] < 0.0 and scissor_new_pose["joint_0"] > self.min_cut_angle

    @ti.kernel
    def find_all_edge_intersection_kernel(self, batch_idx: ti.i32, x0_raw: ti.types.vector(3, ti.f32), x1_raw: ti.types.vector(3, ti.f32), last_split_vertex: ti.i32, insert_in_plane_tol: ti.f32, insert_out_plane_tol: ti.f32):
        """
        Detect intersection from x0 to x1.

        If last_split_vertex != -1, x0 = last_split_vertex's position

        Results are stored in self.edge_intersect, self.edge_intersect_cnt.
        """
        self.edge_intersect_cnt[None] = 0
        self.edge_is_dirty.fill(False)
        self.face_is_dirty.fill(False)

        if last_split_vertex != -1:
            x0_raw[:] = self.cloth.vertices_pos[batch_idx, last_split_vertex][:]
        dx_eps = self.cloth.dx_eps

        for eid in range(self.cloth.mesh.n_edges[batch_idx]):
            v1id, v2id, f1id, f2id = self.cloth.mesh.edges_vid_fid[batch_idx, eid]
            y0_raw = self.cloth.vertices_pos[batch_idx, v1id]
            y1_raw = self.cloth.vertices_pos[batch_idx, v2id]

            vec = self.cloth.get_edge_norm_func(batch_idx, eid)
            # vec = (y0_raw - y1_raw).cross(x0_raw - x1_raw)
            scaling = get_scaling_matrix_func(
                vec, insert_in_plane_tol / insert_out_plane_tol, dx_eps)

            x0 = scaling @ x0_raw
            x1 = scaling @ x1_raw
            y0 = scaling @ y0_raw
            y1 = scaling @ y1_raw

            bcy = get_intersect_func(x0, x1, y0, y1, dx_eps)
            bcx = get_intersect_func(y0, y1, x0, x1, dx_eps)
            x2 = bcx * x0 + (1.0 - bcx) * x1
            y2 = bcy * y0 + (1.0 - bcy) * y1
            dist = ti.math.length(x2 - y2)

            dist_y0 = get_distance_func(y0, x0, x1, True, dx_eps)
            dist_y1 = get_distance_func(y1, x0, x1, True, dx_eps)
            dist_x0 = get_distance_func(x0, y0, y1, True, dx_eps)
            dist_x1 = get_distance_func(x1, y0, y1, True, dx_eps)

            min_dist_01 = ti.min(dist_y0, dist_y1, dist_x0, dist_x1)
            if dist < min_dist_01 and \
                    0.0 <= bcx and bcx <= 1.0 and \
                    0.0 <= bcy and bcy <= 1.0:
                min_dist_01 = dist

            if min_dist_01 < insert_in_plane_tol:
                if dist == min_dist_01 and \
                        0.0 <= bcx and bcx <= 1.0 and \
                        0.0 <= bcy and bcy <= 1.0:
                    old_cnt = ti.atomic_add(self.edge_intersect_cnt[None], 1)
                    self.edge_intersect[old_cnt] = EdgeIntersect(
                        edge_id=eid, bary_coor_edge=bcy)
                elif dist_y0 == min_dist_01:
                    old_cnt = ti.atomic_add(self.edge_intersect_cnt[None], 1)
                    self.edge_intersect[old_cnt] = EdgeIntersect(
                        edge_id=eid, bary_coor_edge=1.0)
                elif dist_y1 == min_dist_01:
                    old_cnt = ti.atomic_add(self.edge_intersect_cnt[None], 1)
                    self.edge_intersect[old_cnt] = EdgeIntersect(
                        edge_id=eid, bary_coor_edge=0.0)
                elif dist_x0 == min_dist_01:
                    old_cnt = ti.atomic_add(self.edge_intersect_cnt[None], 1)
                    u, v = ti.math.clamp(
                        get_2D_barycentric_weights_func(x0, y0, y1, dx_eps), 0.0, 1.0)
                    self.edge_intersect[old_cnt] = EdgeIntersect(
                        edge_id=eid, bary_coor_edge=u)
                elif dist_x1 == min_dist_01:
                    old_cnt = ti.atomic_add(self.edge_intersect_cnt[None], 1)
                    u, v = ti.math.clamp(
                        get_2D_barycentric_weights_func(x1, y0, y1, dx_eps), 0.0, 1.0)
                    self.edge_intersect[old_cnt] = EdgeIntersect(
                        edge_id=eid, bary_coor_edge=u)

    @ti.func
    def safe_insert_func(self, batch_idx, eid, bc, min_rest_length) -> ti.i32:
        v1id, v2id, f1id, f2id = self.cloth.mesh.edges_vid_fid[batch_idx, eid]
        length = self.cloth.get_edge_rest_length_func(batch_idx, eid)
        too_close_to_v1 = (length * (1.0 - bc) < min_rest_length) or \
            (1.0 - bc < MINIMUM_CUT_RATIO)
        too_close_to_v2 = (length * bc < min_rest_length) or \
            (bc < MINIMUM_CUT_RATIO)

        ret_val = -1
        if not too_close_to_v1 and not too_close_to_v2:
            self.edge_is_dirty[eid] = True
            self.face_is_dirty[f1id] = True
            if f2id != -1:
                self.face_is_dirty[f2id] = True
            ret_val = self.cloth.insert_vertices_eid_func(batch_idx, eid, bc)

        elif too_close_to_v1 and too_close_to_v2:
            if bc < MID_POINT_BC:
                ret_val = v2id
            else:
                ret_val = v1id

        elif too_close_to_v1:
            if length * (1.0 - bc) < min_rest_length / 2:
                ret_val = v1id
            elif length > min_rest_length * 2:
                new_bc = 1.0 - min_rest_length / length
                self.edge_is_dirty[eid] = True
                self.face_is_dirty[f1id] = True
                if f2id != -1:
                    self.face_is_dirty[f2id] = True
                ret_val = self.cloth.insert_vertices_eid_func(batch_idx, eid, new_bc)
            else:
                ret_val = v1id

        else:
            if length * bc < min_rest_length / 2:
                ret_val = v2id
            elif length > min_rest_length * 2:
                new_bc = min_rest_length / length
                self.edge_is_dirty[eid] = True
                self.face_is_dirty[f1id] = True
                if f2id != -1:
                    self.face_is_dirty[f2id] = True
                ret_val = self.cloth.insert_vertices_eid_func(batch_idx, eid, new_bc)
            else:
                ret_val = v2id

        return ret_val

    @ti.kernel
    def insert_edge_intersection_kernel(self, batch_idx: ti.i32, min_rest_length: ti.f32, intersect_id: ti.i32, cut_plane_norm: ti.types.vector(3, ti.f32)) -> ti.i32:
        """
        Use self.edge_intersect[intersect_id] to insert.

        Return: 
            - If this edge is dirty (may have inserted other vertices before), return -1.
            - Else return new vertex id.
        """
        bc = self.edge_intersect[intersect_id].bary_coor_edge
        eid = self.edge_intersect[intersect_id].edge_id
        ret_val = -1
        edge_norm = self.cloth.get_edge_norm_func(batch_idx, eid)
        if (not self.edge_is_dirty[eid]) and \
                ti.math.acos(ti.abs(cut_plane_norm.dot(edge_norm))) < self.max_scissor_cloth_cut_angle:
            ret_val = self.safe_insert_func(batch_idx, eid, bc, min_rest_length)
        return ret_val

    @ti.kernel
    def find_adj_face_intersection_kernel(self, batch_idx: ti.i32, vid: ti.i32, x1_raw: ti.types.vector(3, ti.f32), insert_in_plane_tol: ti.f32, insert_out_plane_tol: ti.f32):
        """
        Detect intersection from x1 to fid which is adjacent to vid.

        Results are stored in self.face_intersect.
        """
        self.face_intersect_cnt[None] = 0
        self.edge_is_dirty.fill(False)
        self.face_is_dirty.fill(False)

        dx_eps = self.cloth.dx_eps
        for i in range(self.cloth.mesh.vertices_fid_cnt[batch_idx, vid]):
            fid = self.cloth.mesh.vertices_fid[batch_idx, vid][i]
            vec_cp = self.cloth.get_face_cross_product_func(batch_idx, fid)
            v1id, v2id, v3id = self.cloth.mesh.faces_vid[batch_idx, fid]

            v1pos_raw = self.cloth.vertices_pos[batch_idx, v1id]
            v2pos_raw = self.cloth.vertices_pos[batch_idx, v2id]
            v3pos_raw = self.cloth.vertices_pos[batch_idx, v3id]

            scaling = get_scaling_matrix_func(
                vec_cp, insert_in_plane_tol / insert_out_plane_tol, dx_eps)
            x1 = scaling @ x1_raw
            v1pos = scaling @ v1pos_raw
            v2pos = scaling @ v2pos_raw
            v3pos = scaling @ v3pos_raw

            u, v, w = get_3D_barycentric_weights_func(
                x1, v1pos, v2pos, v3pos, dx_eps)

            proj = u * v1pos + v * v2pos + w * v3pos
            dist = ti.math.length(proj - x1)
            dist12 = get_distance_func(x1, v1pos, v2pos, True, dx_eps)
            dist23 = get_distance_func(x1, v2pos, v3pos, True, dx_eps)
            dist31 = get_distance_func(x1, v3pos, v1pos, True, dx_eps)
            dist1 = ti.math.length(x1 - v1pos)
            dist2 = ti.math.length(x1 - v2pos)
            dist3 = ti.math.length(x1 - v3pos)

            min_dist_123 = ti.min(dist1, dist2, dist3, dist12, dist23, dist31)
            if dist < min_dist_123 and \
                    0.0 <= u and u <= 1.0 and \
                    0.0 <= v and v <= 1.0 and \
                    0.0 <= w and w <= 1.0:
                min_dist_123 = dist

            if min_dist_123 < insert_in_plane_tol:
                if dist == min_dist_123 and \
                        0.0 <= u and u <= 1.0 and \
                        0.0 <= v and v <= 1.0 and \
                        0.0 <= w and w <= 1.0:
                    old_cnt = ti.atomic_add(self.face_intersect_cnt[None], 1)
                    self.face_intersect[old_cnt] = FaceIntersect(
                        face_id=fid, bary_coor=ti.Vector([u, v, w], ti.f32))
                elif dist1 == min_dist_123:
                    old_cnt = ti.atomic_add(self.face_intersect_cnt[None], 1)
                    self.face_intersect[old_cnt] = FaceIntersect(
                        face_id=fid, bary_coor=ti.Vector([1.0, 0.0, 0.0], ti.f32))
                elif dist2 == min_dist_123:
                    old_cnt = ti.atomic_add(self.face_intersect_cnt[None], 1)
                    self.face_intersect[old_cnt] = FaceIntersect(
                        face_id=fid, bary_coor=ti.Vector([0.0, 1.0, 0.0], ti.f32))
                elif dist3 == min_dist_123:
                    old_cnt = ti.atomic_add(self.face_intersect_cnt[None], 1)
                    self.face_intersect[old_cnt] = FaceIntersect(
                        face_id=fid, bary_coor=ti.Vector([0.0, 0.0, 1.0], ti.f32))
                elif dist12 == min_dist_123:
                    uu, vv = ti.math.clamp(
                        get_2D_barycentric_weights_func(x1, v1pos, v2pos, dx_eps), 0.0, 1.0)
                    old_cnt = ti.atomic_add(self.face_intersect_cnt[None], 1)
                    self.face_intersect[old_cnt] = FaceIntersect(
                        face_id=fid, bary_coor=ti.Vector([uu, vv, 0.0], ti.f32))
                elif dist23 == min_dist_123:
                    vv, ww = ti.math.clamp(
                        get_2D_barycentric_weights_func(x1, v2pos, v3pos, dx_eps), 0.0, 1.0)
                    old_cnt = ti.atomic_add(self.face_intersect_cnt[None], 1)
                    self.face_intersect[old_cnt] = FaceIntersect(
                        face_id=fid, bary_coor=ti.Vector([0.0, vv, ww], ti.f32))
                elif dist31 == min_dist_123:
                    ww, uu = ti.math.clamp(
                        get_2D_barycentric_weights_func(x1, v3pos, v1pos, dx_eps), 0.0, 1.0)
                    old_cnt = ti.atomic_add(self.face_intersect_cnt[None], 1)
                    self.face_intersect[old_cnt] = FaceIntersect(
                        face_id=fid, bary_coor=ti.Vector([uu, 0.0, ww], ti.f32))

    @ti.kernel
    def insert_adj_face_intersection_kernel(self, batch_idx: ti.i32, vid: ti.i32, min_rest_length: ti.f32, min_move_dist: ti.f32, intersect_id: ti.i32, cut_plane_norm: ti.types.vector(3, ti.f32)) -> ti.i32:
        """
        Use self.face_intersect to insert.

        Return
            - If the face is dirty, return vid. 
            - Else, return new vertex id.
        """
        ret_val = vid

        fid = self.face_intersect[intersect_id].face_id
        face_norm = self.cloth.get_face_normalized_func(batch_idx, fid)

        if (not self.face_is_dirty[fid]) and \
                ti.math.acos(ti.abs(cut_plane_norm.dot(face_norm))) < self.max_scissor_cloth_cut_angle:
            bc = self.face_intersect[intersect_id].bary_coor
            u, v, w = self.face_intersect[intersect_id].bary_coor

            vids = self.cloth.mesh.faces_vid[batch_idx, fid]
            v1id, v2id, v3id = self.cloth.mesh.faces_vid[batch_idx, fid]
            a = -1
            b = -1
            c = -1
            if v1id == vid:
                a = 0
                b = 1
                c = 2
            elif v2id == vid:
                a = 1
                b = 2
                c = 0
            elif v3id == vid:
                a = 2
                b = 0
                c = 1
            else:
                print(
                    "[ERROR] in cut_simulation_environment.insert_adj_face_intersection_kernel() error.")

            v1pos = self.cloth.vertices_pos[batch_idx, v1id]
            v2pos = self.cloth.vertices_pos[batch_idx, v2id]
            v3pos = self.cloth.vertices_pos[batch_idx, v3id]
            newpos = u * v1pos + v * v2pos + w * v3pos
            dista = ti.math.length(newpos - self.cloth.vertices_pos[batch_idx, vids[a]])
            distb = ti.math.length(newpos - self.cloth.vertices_pos[batch_idx, vids[b]])
            distc = ti.math.length(newpos - self.cloth.vertices_pos[batch_idx, vids[c]])
            dist = ti.min(dista, distb, distc)

            tmp_v1id = -1
            if dist > min_move_dist and bc[b] + bc[c] > MINIMUM_CUT_RATIO:
                connect = self.cloth.mesh.is_connect_func(batch_idx, vids[b], vids[c])
                if connect[0] == 1:
                    eid_bc = connect[1]
                    beta_b_c = bc[b] / (bc[b] + bc[c])
                    if self.cloth.mesh.edges_vid_fid[batch_idx, eid_bc][0] == vids[c]:
                        beta_b_c = bc[c] / (bc[b] + bc[c])
                    tmp_v1id = self.safe_insert_func(
                        batch_idx, eid_bc, beta_b_c, min_rest_length)
                elif bc[b] > bc[c]:
                    tmp_v1id = vids[b]
                else:
                    tmp_v1id = vids[c]

                connect = self.cloth.mesh.is_connect_func(batch_idx, vids[a], tmp_v1id)
                if connect[0] == 1:
                    eid_bc_a = connect[1]
                    beta_bc_a = bc[b] + bc[c]
                    if self.cloth.mesh.edges_vid_fid[batch_idx, eid_bc_a][0] == vids[a]:
                        beta_bc_a = bc[a]
                    ret_val = self.safe_insert_func(
                        batch_idx, eid_bc_a, beta_bc_a, min_rest_length)
                else:
                    ret_val = vids[a]

            else:
                if dist == distb:
                    ret_val = vids[b]
                elif dist == distc:
                    ret_val = vids[c]

        return ret_val

    def get_split_vertex_list(self, batch_idx: int, last_split_vertex: int, last_front_point: np.ndarray, curr_front_point: np.ndarray, cut_vec: np.ndarray, cut_pos: np.ndarray, cut_plane_norm: np.ndarray) -> List[SplitVertex]:
        split_vertex_list: List[SplitVertex] = []
        # split_vertex_id = []
        # split_vertex_bc = []

        def in_split_vertex_list(_new_vert_id: int, _split_vertex_list: List[SplitVertex]):
            for _split_vertex in _split_vertex_list:
                if _split_vertex.vert_id == _new_vert_id:
                    return True
            return False

        if last_split_vertex == -1 or not self.use_last_split_vertex_id:
            self.find_all_edge_intersection_kernel(
                batch_idx, last_front_point - cut_vec * self.first_cut_backward_tol, curr_front_point, -1, self.insert_in_plane_tol, self.insert_out_plane_tol)
        else:
            self.find_all_edge_intersection_kernel(
                batch_idx, last_front_point, curr_front_point, last_split_vertex, self.insert_in_plane_tol, self.insert_out_plane_tol)
        '''else:
            self.find_mininal_dist_kernel(
                curr_front_point + cut_vec * self.insert_tol / 2)
            self.find_all_edge_intersection_use_mindist_lastvert_kernel(
                last_split_vertex)'''

        '''print("[DCUT] edge_intersect_cnt:{}".format(
            self.edge_intersect_cnt[None]))'''
        for i in range(self.edge_intersect_cnt[None]):
            new_vert_id = self.insert_edge_intersection_kernel(
                batch_idx, self.cut_min_length, i, cut_plane_norm)

            if new_vert_id != -1 and not in_split_vertex_list(new_vert_id, split_vertex_list):
                new_vert_pos = self.cloth.vertices_pos[batch_idx, new_vert_id].to_numpy()
                split_vertex_list.append(SplitVertex(
                    new_vert_id, np.linalg.norm(curr_front_point - new_vert_pos), False))

        if len(split_vertex_list) >= 1:
            split_vertex_list_copy = copy.deepcopy(split_vertex_list)

            split_vertex_id_list = [
                split_vertex.vert_id for split_vertex in split_vertex_list]

            # It is hard too judge which vertex is the most front 'big' vertex.
            # This for loop is to avoid misjudge flying small triangles.
            for split_vertex in split_vertex_list_copy:
                if split_vertex.cut_pos > max(self.insert_in_plane_tol, self.insert_out_plane_tol) * 2:
                    continue

                self.find_adj_face_intersection_kernel(
                    batch_idx, split_vertex.vert_id, curr_front_point, self.insert_in_plane_tol, self.insert_out_plane_tol)

                '''print("[DCUT] face_intersect_cnt:{} split_vertex.vert_id:{}".format(
                    self.face_intersect_cnt[None], split_vertex.vert_id))'''
                for i in range(self.face_intersect_cnt[None]):
                    old_face_vids = self.cloth.mesh.faces_vid[
                        batch_idx, self.face_intersect[i].face_id].to_numpy()
                    new_vert_id = self.insert_adj_face_intersection_kernel(
                        batch_idx, split_vertex.vert_id, self.cut_min_length, self.cut_min_length, i, cut_plane_norm)
                    '''print("[DCUT] face_intersect[{}]: vids:{} bc:{} new_vert_id:{}".format(
                        i, old_face_vids, self.face_intersect[i].bary_coor, new_vert_id))'''

                    if not in_split_vertex_list(new_vert_id, split_vertex_list):
                        new_vert_pos = \
                            self.cloth.vertices_pos[batch_idx, new_vert_id].to_numpy()
                        split_vertex_list.append(SplitVertex(
                            new_vert_id, np.linalg.norm(curr_front_point - new_vert_pos), False))

        return split_vertex_list

    def split_edge(self, batch_idx: int, split_vertex_list: List[SplitVertex], cut_plane_norm: np.ndarray, now_time: float):
        n = len(split_vertex_list)
        for i in range(n):
            for j in range(i):
                vid_from = split_vertex_list[i].vert_id
                vid_to = split_vertex_list[j].vert_id

                connect_num, connect_eid = self.cloth.mesh.is_connect_kernel(
                    batch_idx, vid_from, vid_to)

                if connect_num == 1 and not self.cloth.mesh.is_outside_edge_kernel(batch_idx, connect_eid):
                    edge_norm = (self.cloth.get_edge_norm_kernel(
                        batch_idx, connect_eid)).to_numpy()

                    if math.acos(min(1.0, abs(cut_plane_norm.dot(edge_norm)))) < self.max_scissor_cloth_cut_angle:
                        self.cloth.split_edge_kernel(batch_idx, connect_eid)
                        '''if self.save_cutting_info:
                            self.split_edge_origin_coordinates.append((
                                self.cloth.vertices_rest_pos[vid_from].to_numpy(),
                                self.cloth.vertices_rest_pos[vid_to].to_numpy(),
                                now_time
                            ))'''
                        split_vertex_list[i].set_splitted()
                        split_vertex_list[j].set_splitted()

    @ti.kernel
    def judge_stuck_check_all_face_kernel(self, batch_idx: ti.i32, x0: ti.types.vector(3, ti.f32), x1: ti.types.vector(3, ti.f32), stuck_in_plane_tol: ti.f32, stuck_out_plane_tol: ti.f32) -> bool:
        is_found = False
        dx_eps = self.cloth.dx_eps

        x1_is_in_mesh = False
        for fid in range(self.cloth.mesh.n_faces[batch_idx]):
            v1id, v2id, v3id = self.cloth.mesh.faces_vid[batch_idx, fid]

            v1pos = self.cloth.vertices_pos[batch_idx, v1id]
            v2pos = self.cloth.vertices_pos[batch_idx, v2id]
            v3pos = self.cloth.vertices_pos[batch_idx, v3id]

            dist, _u, _v, _w = get_distance_to_triangle_func(
                x1, v1pos, v2pos, v3pos, dx_eps)

            if dist < stuck_out_plane_tol:
                x1_is_in_mesh = True

        if x1_is_in_mesh:
            is_found = True

        for eid in range(self.cloth.mesh.n_edges[batch_idx]):
            if x1_is_in_mesh:
                v1id, v2id, f1id, f2id = self.cloth.mesh.edges_vid_fid[batch_idx, eid]
                if f2id == -1:
                    # v1id, v2id, v3id may be different with faces_vid[f1id]
                    # difference is a permutation
                    v3id = self.cloth.mesh.find_opp_vert_on_face_func(
                        batch_idx, f1id, v1id, v2id)

                    v1pos = self.cloth.vertices_pos[batch_idx, v1id]
                    v2pos = self.cloth.vertices_pos[batch_idx, v2id]
                    v3pos = self.cloth.vertices_pos[batch_idx, v3id]

                    v4id, v5id, v6id = self.cloth.mesh.faces_vid[batch_idx, f1id]

                    v4pos = self.cloth.vertices_pos[batch_idx, v4id]
                    v5pos = self.cloth.vertices_pos[batch_idx, v5id]
                    v6pos = self.cloth.vertices_pos[batch_idx, v6id]

                    dist_to_face, _u, _v, _w = get_distance_to_triangle_func(
                        x1, v4pos, v5pos, v6pos, dx_eps)

                    h_vec_norm = safe_normalized(get_distance_vec_func(
                        v3pos, v1pos, v2pos, False, dx_eps), dx_eps)
                    proj_signed_dist = h_vec_norm.dot(
                        x1 - v1pos)  # outside is positive

                    u1, v1, w1 = get_3D_barycentric_weights_func(
                        x1, v1pos, v2pos, v3pos, dx_eps)
                    x1p = u1 * v1pos + v1 * v2pos + w1 * v3pos

                    bc_edge, _ = get_2D_barycentric_weights_func(
                        x1, v1pos, v2pos, dx_eps)
                    edge_length = ti.math.length(v1pos - v2pos)
                    bc_edge_min = -stuck_in_plane_tol / \
                        ti.max(edge_length, dx_eps)
                    bc_edge_max = 1.0 + stuck_in_plane_tol / \
                        ti.max(edge_length, dx_eps)

                    proj_dist_to_edge = get_distance_func(
                        x1p, v1pos, v2pos, True, dx_eps)

                    if dist_to_face < stuck_out_plane_tol and \
                        proj_signed_dist > -stuck_in_plane_tol and \
                        (bc_edge > bc_edge_min and bc_edge < bc_edge_max) and \
                            (proj_signed_dist > 0.0 or proj_dist_to_edge < stuck_in_plane_tol):
                        is_found = False

        return is_found

    def judge_stuck(self, batch_idx: int, last_front_point: np.ndarray, curr_front_point: np.ndarray, cut_vec: np.ndarray) -> bool:
        if self.disable_stuck:
            return False
        return bool(self.judge_stuck_check_all_face_kernel(batch_idx, last_front_point, curr_front_point + cut_vec * (self.stuck_in_plane_tol + self.barrier_width + self.barrier_sdf_offset), self.stuck_in_plane_tol, self.stuck_out_plane_tol))

    @ti.kernel
    def find_mininal_dist_kernel(self, batch_idx: ti.i32, x1: ti.types.vector(3, ti.f32)) -> ti.types.vector(7, ti.f32):
        """
        Find mininal dist to x1. Position with minimal dist is stored in self.face_min_dist[None].

        Return:
            - 1D vector v:
                - v[0] the distance
                - v[1:4] the origin position with minimal distance. 
                - v[4:7] the current position with minimal distance. 
        """
        min_dist = ti.math.inf

        dx_eps = self.cloth.dx_eps
        for fid in range(self.cloth.mesh.n_faces[batch_idx]):
            v1id, v2id, v3id = self.cloth.mesh.faces_vid[batch_idx, fid]

            v1pos = self.cloth.vertices_pos[batch_idx, v1id]
            v2pos = self.cloth.vertices_pos[batch_idx, v2id]
            v3pos = self.cloth.vertices_pos[batch_idx, v3id]

            dist_to_triangle, _u, _v, _w = get_distance_to_triangle_func(
                x1, v1pos, v2pos, v3pos, dx_eps)
            ti.atomic_min(min_dist, dist_to_triangle)

        for fid in range(self.cloth.mesh.n_faces[batch_idx]):
            v1id, v2id, v3id = self.cloth.mesh.faces_vid[batch_idx, fid]

            v1pos = self.cloth.vertices_pos[batch_idx, v1id]
            v2pos = self.cloth.vertices_pos[batch_idx, v2id]
            v3pos = self.cloth.vertices_pos[batch_idx, v3id]

            dist_to_triangle, u, v, w = get_distance_to_triangle_func(
                x1, v1pos, v2pos, v3pos, dx_eps)
            if dist_to_triangle == min_dist:
                self.face_min_dist[None] = MinimalDist(
                    face_id=fid, bary_coor=ti.Vector([u, v, w], ti.f32), min_dist=min_dist)

        fid = self.face_min_dist[None].face_id
        u, v, w = self.face_min_dist[None].bary_coor
        dist = self.face_min_dist[None].min_dist

        v1id, v2id, v3id = self.cloth.mesh.faces_vid[batch_idx, fid]
        v1_rest_pos = self.cloth.vertices_rest_pos[batch_idx, v1id]
        v2_rest_pos = self.cloth.vertices_rest_pos[batch_idx, v2id]
        v3_rest_pos = self.cloth.vertices_rest_pos[batch_idx, v3id]

        v1_pos = self.cloth.vertices_pos[batch_idx, v1id]
        v2_pos = self.cloth.vertices_pos[batch_idx, v2id]
        v3_pos = self.cloth.vertices_pos[batch_idx, v3id]

        ret_val = ti.Vector.zero(ti.f32, 7)
        ret_val[0] = dist
        ret_val[1: 4] = u * v1_rest_pos + v * v2_rest_pos + w * v3_rest_pos
        ret_val[4: 7] = u * v1_pos + v * v2_pos + w * v3_pos
        return ret_val

    def update_last_split_vertex_id(self, batch_idx: int, split_vertex_list: List[SplitVertex]):
        if len(split_vertex_list) == 0:
            self.last_split_vertex_id[batch_idx] = -1
        else:
            split_vertex_list_sorted = sorted(
                split_vertex_list, key=lambda x: x.cut_pos)
            find_splitted = False

            n = len(split_vertex_list_sorted)
            # for i in range(n - 1, -1, -1):
            for i in range(n):
                split_vertex = split_vertex_list_sorted[i]
                if split_vertex.splitted:
                    self.last_split_vertex_id[batch_idx] = split_vertex.vert_id
                    find_splitted = True
                    break

            if not find_splitted:
                self.last_split_vertex_id[batch_idx] = -1

    def cut_cloth(self, last_front_points: List[np.ndarray], scissors_new_pose: List[dict], scissors_velocity: List[dict], now_time: float) -> List[bool]:
        scissors_is_stuck = [None for _ in range(self.batch_size)]

        '''front_point_proj = [None for _ in range(self.batch_size)]'''

        for batch_idx in range(self.batch_size):
            # self.reset_last_split_vertex(sid)
            is_cut_state = self.is_cut_state(
                scissors_new_pose[batch_idx], scissors_velocity[batch_idx])
            '''print("[DCUT] is_cut_state:{}".format(is_cut_state))'''

            pos, vec, axs = self.scissor.get_cut_direction_given_batch(batch_idx)

            curr_front_point = self.scissor.get_cut_front_point_given_batch(batch_idx, "world")
            last_front_point = last_front_points[batch_idx]
            '''print("[DCUT] curr_front_point:{} last_front_point:{} last_split_vert {}: {}".format(
                curr_front_point, last_front_point, self.last_split_vertex_id[sid], 0.0 if self.last_split_vertex_id[sid] == -1 else self.cloth.vertices_pos[self.last_split_vertex_id[sid]]))'''

            if is_cut_state:
                cut_plane_norm = np.cross(vec, axs)
                cut_plane_norm /= max(np.linalg.norm(cut_plane_norm),
                                      self.cloth.dx_eps)

                split_vertex_list = self.get_split_vertex_list(
                    batch_idx, self.last_split_vertex_id[batch_idx], last_front_point, curr_front_point, vec, pos, cut_plane_norm)

                self.split_edge(batch_idx, split_vertex_list, cut_plane_norm, now_time)

                '''for split_vertex in split_vertex_list:
                    print("[DCUT] split_vertex:{} pos:{}".format(
                        split_vertex, self.cloth.vertices_pos[split_vertex.vert_id]))'''

                '''self.curr_split_vertex_list[sid] = split_vertex_list'''
                scissors_is_stuck[batch_idx] = False
                self.update_last_split_vertex_id(batch_idx, split_vertex_list)

            else:
                '''self.curr_split_vertex_list[sid] = []'''
                scissors_is_stuck[batch_idx] = self.judge_stuck(
                    batch_idx, last_front_point, curr_front_point, vec)
                self.update_last_split_vertex_id(batch_idx, [])

            '''front_point_proj[batch_idx] = (bool(is_cut_state), float(now_time),
                                     self.find_mininal_dist_kernel(batch_idx, curr_front_point).to_numpy())'''

        '''if self.save_cutting_info:
            self.front_point_projection.append(front_point_proj)'''

        assert None not in scissors_is_stuck
        return scissors_is_stuck

    def reset(self):
        '''self.split_edge_origin_coordinates = []
        self.front_point_projection = []'''
        self.last_split_vertex_id = [-1 for _ in range(self.batch_size)]

    def get_state(self) -> List[dict]:
        '''state = {"class": "Cutting",
                 "split_edge_origin_coordinates": copy.deepcopy(self.split_edge_origin_coordinates),
                 "front_point_projection": copy.deepcopy(self.front_point_projection),
                 "last_split_vertex_id": copy.deepcopy(self.last_split_vertex_id)}'''
        return [{
            "class": "Cutting",
            "last_split_vertex_id": self.last_split_vertex_id[batch_idx]
        } for batch_idx in range(self.batch_size)]
    
    def set_state(self, states: List[dict]) -> None:
        assert isinstance(states, list)
        assert isinstance(states[0], dict)
        assert len(states) == self.batch_size
        '''self.split_edge_origin_coordinates = copy.deepcopy(state["split_edge_origin_coordinates"])
        self.front_point_projection = copy.deepcopy(state["front_point_projection"])'''
        self.last_split_vertex_id = [state["last_split_vertex_id"] for state in states]
