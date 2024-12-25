import taichi as ti

import trimesh
import pywavefront
import numpy as np

from typing import Dict, List

from src.cbase import *
from src.utils import *

# IMPORTANT Conventions:
# 1)
# edges_vid_fid[k] = v1id, v2id, f1id, f2id
# v1id < v2id
# only f2id could be -1.
# 2)
# faces_vid[k] = v1id, v2id, v3id
# These 3 vertices must in counter clock wise.


@ti.data_oriented
class DynamicTriangularMesh(TiObject):
    def __init__(self, batch_size: int, file_path: str, n_vertices_multiplier: float, n_edges_multiplier: float, n_faces_multiplier: float, nmax_edges_connect_to_vertex: int, nmax_faces_connect_to_vertex: int) -> None:
        self.batch_size = batch_size
        self.dim = 3

        # init mesh
        self.origin_mesh = self.load_mesh(file_path=file_path)

        self.n_vertices_multiplier = n_vertices_multiplier
        self.n_edges_multiplier = n_edges_multiplier
        self.n_faces_multiplier = n_faces_multiplier
        self.nmax_edges_connect_to_vertex = nmax_edges_connect_to_vertex
        self.nmax_faces_connect_to_vertex = nmax_faces_connect_to_vertex

        self.topo_modified = ti.field(dtype=bool, shape=(self.batch_size, ))
        """[B, ]"""
        self.topo_modified.fill(False)

        # initialize vertices

        # number of total vertices
        self.n_vertices = ti.field(dtype=ti.i32, shape=(self.batch_size, ))
        """[B, ]"""
        self.n_vertices.fill(int(self.origin_mesh.vertices.shape[0]))

        self.nmax_vertices = int(self.n_vertices_multiplier * int(self.origin_mesh.vertices.shape[0]))

        # v2e, v2f
        self.vertices_eid = ti.Vector.field(n=self.nmax_edges_connect_to_vertex, dtype=ti.i32, shape=(self.batch_size, self.nmax_vertices))
        """[B, V][NVE]"""

        self.vertices_eid_cnt = ti.field(dtype=ti.i32, shape=(self.batch_size, self.nmax_vertices))
        """[B, V]"""
        self.vertices_eid_cnt.fill(0)

        self.vertices_fid = ti.Vector.field(n=self.nmax_faces_connect_to_vertex, dtype=ti.i32, shape=(self.batch_size, self.nmax_vertices))
        """[B, V][NVF]"""

        self.vertices_fid_cnt = ti.field(dtype=ti.i32, shape=(self.batch_size, self.nmax_vertices))
        """[B, V]"""
        self.vertices_fid_cnt.fill(0)

        # initialize faces

        # number of total faces
        self.n_faces = ti.field(dtype=ti.i32, shape=(self.batch_size, ))
        """[B, ]"""
        self.n_faces.fill(int(self.origin_mesh.faces.shape[0]))

        self.nmax_faces = int(self.n_faces_multiplier *int(self.origin_mesh.faces.shape[0]))

        # faces' adjacent vertices id
        self.faces_vid = ti.Vector.field(n=3, dtype=ti.i32, shape=(self.batch_size, self.nmax_faces))
        """[B, F][3]"""
        self.faces_vid.from_numpy(
            np.tile(
                np.pad(self.origin_mesh.faces.astype(np.int32),
                       ((0, self.nmax_faces - int(self.origin_mesh.faces.shape[0])), (0, 0)), 
                       'constant', constant_values=0)[None, ...],
                reps=(self.batch_size, 1, 1)
        ))

        for fid in range(self.n_faces[0]):
            v1id, v2id, v3id = self.faces_vid[0, fid]
            self.append_vertices_fid_allbatch_kernel(v1id, fid)
            self.append_vertices_fid_allbatch_kernel(v2id, fid)
            self.append_vertices_fid_allbatch_kernel(v3id, fid)

        # initialize edges

        # edges_dict[(vid1,vid2)] = [fid1, fid2]
        edges_dict: Dict[tuple:list] = dict({})
        for fid in range(self.n_faces[0]):
            face = self.faces_vid[0, fid]
            dict_keys = [(min(face[0], face[1]), max(face[0], face[1])),
                         (min(face[0], face[2]), max(face[0], face[2])),
                         (min(face[1], face[2]), max(face[1], face[2]))]
            for dict_key in dict_keys:
                if dict_key in edges_dict.keys():
                    edges_dict[dict_key].append(fid)
                else:
                    edges_dict[dict_key] = [fid]

        self.edges_list = []
        for key in edges_dict:
            if len(edges_dict[key]) == 1:
                edges_dict[key].append(-1)
            self.edges_list.append(
                [key[0], key[1], edges_dict[key][0], edges_dict[key][1]])
            
        self.nmax_edges = int(self.n_edges_multiplier * len(self.edges_list))

        # total number of edges
        self.n_edges = ti.field(dtype=ti.i32, shape=(self.batch_size, ))
        """[B, ]"""
        self.n_edges.fill(len(self.edges_list))

        # edges' adjacent vertices id and faces id
        self.edges_vid_fid = ti.Vector.field(n=4, dtype=ti.i32, shape=(self.batch_size, self.nmax_edges))
        """[B, E][4]"""
        self.edges_vid_fid.from_numpy(
            np.tile(
                np.pad(np.array(self.edges_list, dtype=np.int32),
                       ((0, self.nmax_edges - len(self.edges_list)), (0, 0)), 
                       'constant', constant_values=0)[None, ...],
                reps=(self.batch_size, 1, 1)
        ))

        for eid in range(len(self.edges_list)):
            v1id, v2id, f1id, f2id = self.edges_list[eid]
            self.append_vertices_eid_allbatch_kernel(v1id, eid)
            self.append_vertices_eid_allbatch_kernel(v2id, eid)

        # for topo check
        self.topo_check_cnt_face_v2f = ti.field(
            dtype=ti.i32, shape=(self.batch_size, self.nmax_faces))
        self.topo_check_cnt_face_e2f = ti.field(
            dtype=ti.i32, shape=(self.batch_size, self.nmax_faces))
        self.topo_check_cnt_vertex_on_face_e2v = ti.Vector.field(
            n=3, dtype=ti.i32, shape=(self.batch_size, self.nmax_faces))
        self.topo_check_cnt_edge_remain = ti.field(
            dtype=ti.i32, shape=(self.batch_size, self.nmax_edges))
        
        # for data transfer
        self.vertices_eid_python = self.vertices_eid.to_numpy()
        """[B, V][NVE]"""

        self.vertices_eid_cnt_python = self.vertices_eid_cnt.to_numpy()
        """[B, V]"""

        self.vertices_fid_python = self.vertices_fid.to_numpy()
        """[B, V][NVF]"""

        self.vertices_fid_cnt_python = self.vertices_fid_cnt.to_numpy()
        """[B, V]"""

        self.faces_vid_python = self.faces_vid.to_numpy()
        """[B, F][3]"""

        self.edges_vid_fid_python = self.edges_vid_fid.to_numpy()
        """[B, E][4]"""

    def load_mesh(self, file_path: str) -> trimesh.Trimesh:
        if file_path.endswith(".obj"):
            mesh_obj = pywavefront.Wavefront(
                file_name=file_path, collect_faces=True)
            mesh = trimesh.Trimesh(
                vertices=np.array(mesh_obj.vertices, dtype=np.float32),
                faces=np.array(mesh_obj.mesh_list[0].faces, dtype=np.int32)
            )

        elif file_path.endswith(".ply"):
            mesh = trimesh.load(file_path)

        else:
            raise ValueError("Unrecognize file format: {}".format(file_path))

        return mesh

    @ti.func
    def find_opp_vert_on_face_func(self, batch_idx: ti.i32, fid: ti.i32, v1id: ti.i32, v2id: ti.i32) -> ti.i32:
        """
        Args:
            - fid, v1id, v2id: int

        Return:
            - v3id: int, where 3 vertices on fid is v1id, v2id, v3id
        """
        assert v1id != v2id, "[ERROR] in find_vertex_3_on_face fid:{} v1id:{} v2id:{} v1id=v2id.".format(
            fid, v1id, v2id)
        v3id = -1
        if fid != -1:
            for i in ti.static(range(3)):
                if v1id != self.faces_vid[batch_idx, fid][i] and v2id != self.faces_vid[batch_idx, fid][i]:
                    assert v3id == -1, "[ERROR] find multiple v3id on fid:{} [{},{},{}] with v1id:{} v2id:{}".format(
                        fid, self.faces_vid[batch_idx, fid][0], self.faces_vid[batch_idx, fid][1], self.faces_vid[batch_idx, fid][2], v1id, v2id)
                    v3id = self.faces_vid[batch_idx, fid][i]
        return v3id

    @ti.func
    def find_opp_vert_on_edge_func(self, batch_idx: ti.i32, eid: ti.i32, vid: ti.i32) -> ti.i32:
        """
        Args:
            - eid, vid: int

        Return:
            - uid: int, where 2 vertices on eid is vid, uid
        """
        v1id, v2id, f1id, f2id = self.edges_vid_fid[batch_idx, eid]
        assert v1id == vid or v2id == vid, "[ERROR] vid:{} is not on eid:{}".format(
            vid, eid)

        opp_vert = v1id
        if opp_vert == vid:
            opp_vert = v2id

        return opp_vert

    @ti.func
    def add_vertex_func(self, batch_idx: ti.i32) -> ti.i32:
        """
        Add a new vertex and return new vertex id.

        Args:
            - None

        Return:
            - int (new vertex index)
        """
        self.topo_modified[batch_idx] = True
        return ti.atomic_add(self.n_vertices[batch_idx], 1)

    @ti.func
    def add_face_func(self, batch_idx, new_vid) -> ti.i32:
        """
        Add a new face and return new face id.

        Args:
            - new_vid: 3D taichi vector

        Return:
            - int (new face index)
        """
        self.topo_modified[batch_idx] = True
        old_n = ti.atomic_add(self.n_faces[batch_idx], 1)
        self.faces_vid[batch_idx, old_n] = new_vid
        return old_n

    @ti.func
    def add_edge_func(self, batch_idx, v1id, v2id, f1id, f2id) -> ti.i32:
        """
        Add a new edge and return new edge id, automatically exchange v1,v2 if v1id > v2id.

        Args:
            - v1id, v2id, f1id, f2id: int, edge's adjacent vertices and faces.

        Return:
            - int (new edge index) 
        """
        assert v1id != v2id, "[ERROR] v1id({}) = v2id({})!".format(v1id, v2id)
        assert f1id != f2id, "[ERROR] f1id({}) = f2id({})!".format(f1id, f2id)

        self.topo_modified[batch_idx] = True

        if v1id > v2id:
            v1id, v2id = v2id, v1id

        old_n = ti.atomic_add(self.n_edges[batch_idx], 1)

        self.edges_vid_fid[batch_idx, old_n][0] = v1id
        self.edges_vid_fid[batch_idx, old_n][1] = v2id
        self.edges_vid_fid[batch_idx, old_n][2] = f1id
        self.edges_vid_fid[batch_idx, old_n][3] = f2id

        return old_n

    @ti.func
    def replace_vertices_fid_func(self, batch_idx, vid, old_fid, new_fid):
        is_replaced = False
        ti.loop_config(serialize=True)
        for i in range(self.vertices_fid_cnt[batch_idx, vid]):
            if self.vertices_fid[batch_idx, vid][i] == old_fid:
                self.vertices_fid[batch_idx, vid][i] = new_fid
                is_replaced = True
                self.topo_modified[batch_idx] = True
                break
        assert is_replaced, "[ERROR] vid:{} cannot replace old_fid:{} with new_fid:{}".format(
            vid, old_fid, new_fid)

    @ti.func
    def append_vertices_fid_func(self, batch_idx, vid, new_fid):
        if self.vertices_fid_cnt[batch_idx, vid] < self.nmax_faces_connect_to_vertex:
            old_n = ti.atomic_add(self.vertices_fid_cnt[batch_idx, vid], 1)
            self.vertices_fid[batch_idx, vid][old_n] = new_fid
            self.topo_modified[batch_idx] = True
        else:
            assert False, "[ERROR] vid:{} failed in append new_fid:{} to a vertex".format(
                vid, new_fid)

    @ti.kernel
    def append_vertices_fid_kernel(self, batch_idx: ti.i32, vid: ti.i32, new_fid: ti.i32):
        self.append_vertices_fid_func(batch_idx, vid, new_fid)

    @ti.kernel
    def append_vertices_fid_allbatch_kernel(self, vid: ti.i32, new_fid: ti.i32):
        for batch_idx in range(self.batch_size):
            self.append_vertices_fid_func(batch_idx, vid, new_fid)

    @ti.func
    def remove_vertices_fid_func(self, batch_idx, vid, old_fid):
        is_found = False
        ti.loop_config(serialize=True)
        for i in range(self.vertices_fid_cnt[batch_idx, vid]):
            if self.vertices_fid[batch_idx, vid][i] == old_fid:
                is_found = True
            if is_found:
                if i < self.nmax_faces_connect_to_vertex:
                    self.vertices_fid[batch_idx, vid][i] = self.vertices_fid[batch_idx, vid][i + 1]
                else:
                    self.vertices_fid[batch_idx, vid][i] = 0
        if is_found:
            self.vertices_fid_cnt[batch_idx, vid] -= 1
            self.topo_modified[batch_idx] = True
        else:
            assert False, "[ERROR] vid:{} fail to remove fid:{}".format(
                vid, old_fid)

    @ti.func
    def replace_vertices_eid_func(self, batch_idx, vid, old_eid, new_eid):
        is_replaced = False
        ti.loop_config(serialize=True)
        for i in range(self.vertices_eid_cnt[batch_idx, vid]):
            if self.vertices_eid[batch_idx, vid][i] == old_eid:
                self.vertices_eid[batch_idx, vid][i] = new_eid
                is_replaced = True
                self.topo_modified[batch_idx] = True
                break
        assert is_replaced, "[ERROR] vid:{} cannot replace old_eid:{} with new_eid:{}".format(
            vid, old_eid, new_eid)

    @ti.func
    def append_vertices_eid_func(self, batch_idx, vid, new_eid):
        if self.vertices_eid_cnt[batch_idx, vid] < self.nmax_edges_connect_to_vertex:
            old_n = ti.atomic_add(self.vertices_eid_cnt[batch_idx, vid], 1)
            self.vertices_eid[batch_idx, vid][old_n] = new_eid
            self.topo_modified[batch_idx] = True
        else:
            assert False, "[ERROR] vid:{} failed in append new_eid:{} to a vertex".format(
                vid, new_eid)

    @ti.kernel
    def append_vertices_eid_kernel(self, batch_idx: ti.i32, vid: ti.i32, new_eid: ti.i32):
        self.append_vertices_eid_func(batch_idx, vid, new_eid)

    @ti.kernel
    def append_vertices_eid_allbatch_kernel(self, vid: ti.i32, new_eid: ti.i32):
        for batch_idx in range(self.batch_size):
            self.append_vertices_eid_func(batch_idx, vid, new_eid)

    @ti.func
    def remove_vertices_eid_func(self, batch_idx, vid, old_eid):
        is_found = False
        ti.loop_config(serialize=True)
        for i in range(self.vertices_eid_cnt[batch_idx, vid]):
            if self.vertices_eid[batch_idx, vid][i] == old_eid:
                is_found = True
            if is_found:
                if i < self.nmax_edges_connect_to_vertex:
                    self.vertices_eid[batch_idx, vid][i] = self.vertices_eid[batch_idx, vid][i + 1]
                else:
                    self.vertices_eid[batch_idx, vid][i] = 0
        if is_found:
            self.vertices_eid_cnt[batch_idx, vid] -= 1
            self.topo_modified[batch_idx] = True
        else:
            assert False, "[ERROR] vid:{} fail to remove eid:{}".format(
                vid, old_eid)

    def is_inside_vertex_python(self, batch_idx, vid: int) -> bool:
        """
        Please call self.topo_info_taichi_to_python() first.

        Return if vid is an outside vertex. 
        """
        return self.vertices_fid_cnt_python[batch_idx, vid] == self.vertices_eid_cnt_python[batch_idx, vid]

    @ti.func
    def is_inside_vertex_func(self, batch_idx, vid: ti.i32) -> bool:
        return self.vertices_fid_cnt[batch_idx, vid] == self.vertices_eid_cnt[batch_idx, vid]

    @ti.func
    def is_outside_vertex_func(self, batch_idx, vid: ti.i32) -> bool:
        assert self.vertices_fid_cnt[batch_idx, vid] + 1 >= self.vertices_eid_cnt[batch_idx, vid], "[ERROR] v{}: fid_cnt:{} eid_cnt:{}".format(
            vid, self.vertices_fid_cnt[batch_idx, vid], self.vertices_eid_cnt[batch_idx, vid])
        return self.vertices_fid_cnt[batch_idx, vid] < self.vertices_eid_cnt[batch_idx, vid]

    def is_outside_edge_python(self, batch_idx, eid: int) -> bool:
        """
        Please call self.topo_info_taichi_to_python() first.

        Return if eid is an outside edge. 
        """
        return self.edges_vid_fid_python[batch_idx, eid][3] == -1

    @ti.func
    def is_outside_edge_func(self, batch_idx, eid: ti.i32) -> bool:
        return self.edges_vid_fid[batch_idx, eid][3] == -1

    @ti.kernel
    def is_outside_edge_kernel(self, batch_idx:ti.i32, eid: ti.i32) -> bool:
        return self.is_outside_edge_func(batch_idx, eid)

    def is_outside_face_python(self, batch_idx, fid: int) -> bool:
        vids = self.faces_vid_python[batch_idx, fid]
        return self.vertices_fid_cnt_python[batch_idx, vids[0]] == 1 or \
            self.vertices_fid_cnt_python[batch_idx, vids[1]] == 1 or \
            self.vertices_fid_cnt_python[batch_idx, vids[2]] == 1

    @ti.func
    def is_outside_face_func(self, batch_idx, fid: ti.i32) -> bool:
        vids = self.faces_vid[batch_idx, fid]
        return self.vertices_fid_cnt[batch_idx, vids[0]] == 1 or \
            self.vertices_fid_cnt[batch_idx, vids[1]] == 1 or \
            self.vertices_fid_cnt[batch_idx, vids[2]] == 1

    @ti.kernel
    def is_outside_face_kernel(self, batch_idx: ti.i32, fid: ti.i32) -> bool:
        return self.is_outside_face_func(batch_idx, fid)

    def is_connect_python(self, batch_idx, v1id: int, v2id: int) -> list:
        """
        Please call self.topo_info_taichi_to_python() first.

        Return: [x, y]
            - x: how many edges bewteen v1 and v2
            - y: arbitrary edge id bewteen v1 and v2
        """
        ret_val = [0, 0]
        for i in range(self.vertices_eid_cnt_python[batch_idx, v1id]):
            eid = self.vertices_eid_python[batch_idx, v1id][i]
            edge_v1, edge_v2, f1id, f2id = self.edges_vid_fid_python[batch_idx, eid]
            if (edge_v1 == v1id and edge_v2 == v2id) or \
                    (edge_v1 == v2id and edge_v2 == v1id):
                ret_val[0] += 1
                ret_val[1] = eid
        return ret_val

    @ti.func
    def is_connect_func(self, batch_idx, v1id: ti.i32, v2id: ti.i32) -> ti.types.vector(2, ti.i32):
        """
        Return: 2D vector(x,y)
            - x: how many edges bewteen v1 and v2
            - y: arbitrary edge id bewteen v1 and v2
        """
        ret_val = ti.Vector.zero(dt=ti.i32, n=2)
        for i in range(self.vertices_eid_cnt[batch_idx, v1id]):
            eid = self.vertices_eid[batch_idx, v1id][i]
            edge_v1, edge_v2, f1id, f2id = self.edges_vid_fid[batch_idx, eid]
            if (edge_v1 == v1id and edge_v2 == v2id) or \
                    (edge_v1 == v2id and edge_v2 == v1id):
                ret_val[0] += 1
                ret_val[1] = eid
        return ret_val

    @ti.kernel
    def is_connect_kernel(self, batch_idx: ti.i32, v1id: ti.i32, v2id: ti.i32) -> ti.types.vector(2, ti.i32):
        """
        Return: 2D vector(x,y)
            - x: how many edges bewteen v1 and v2
            - y: arbitrary edge id bewteen v1 and v2
        """
        return self.is_connect_func(batch_idx, v1id, v2id)

    @ti.func
    def check_vertex_in_face_func(self, batch_idx, vid: ti.i32, fid: ti.i32):
        ret_val = 0
        for j in ti.static(range(3)):
            if self.faces_vid[batch_idx, fid][j] == vid:
                ret_val += 1
        return ret_val

    @ti.func
    def edge_vertex_check_func(self, fid, v1id, v2id, eid):
        raise NotImplementedError
    
        fid_vids = self.faces_vid[fid]

        assert self.check_vertex_in_face_func(
            v1id, fid) == 1, "[ERROR] topological_check e{} v{} not in f{}".format(eid, v1id, fid)
        assert self.check_vertex_in_face_func(
            v2id, fid) == 1, "[ERROR] topological_check e{} v{} not in f{}".format(eid, v2id, fid)

        if v1id == fid_vids[0] and v2id == fid_vids[1] or \
                v2id == fid_vids[0] and v1id == fid_vids[1]:
            self.topo_check_cnt_vertex_on_face_e2v[fid][0] += 1
        elif v1id == fid_vids[0] and v2id == fid_vids[2] or \
                v2id == fid_vids[0] and v1id == fid_vids[2]:
            self.topo_check_cnt_vertex_on_face_e2v[fid][1] += 1
        elif v1id == fid_vids[1] and v2id == fid_vids[2] or \
                v2id == fid_vids[1] and v1id == fid_vids[2]:
            self.topo_check_cnt_vertex_on_face_e2v[fid][2] += 1
        else:
            assert False, "[ERROR] topological_check v1{} v2{} is not an edge of f{}".format(
                v1id, v2id, fid)

    @ti.kernel
    def topological_check_kernel(self):
        """check v2f f2v v2e e2v e2f"""
        raise NotImplementedError

        # first simply check f2v
        ti.loop_config(serialize=True)
        for fid in range(self.n_faces[None]):
            vid_list = ti.Vector([0, 0, 0])
            for i in range(3):
                vid = self.faces_vid[fid][i]
                assert vid >= 0 and vid <= self.n_vertices[None] - \
                    1, "[ERROR] topological_check vid out of range"
                vid_list[i] = vid

            assert vid_list[0] != vid_list[1] and vid_list[1] != vid_list[2] and vid_list[
                0] != vid_list[2], "[ERROR] topological_check duplicate vid"

        # then we assume f2v is correct, we check the remain variables.

        # check v2f
        self.topo_check_cnt_face_v2f.fill(0)

        ti.loop_config(serialize=True)
        for vid in range(self.n_vertices[None]):
            for i in range(self.vertices_fid_cnt[vid]):
                fid = self.vertices_fid[vid][i]
                self.topo_check_cnt_face_v2f[fid] += 1
                assert self.check_vertex_in_face_func(
                    vid, fid) == 1, "[ERROR] topological_check v{} not in f{}".format(vid, fid)

        ti.loop_config(serialize=True)
        for i in range(self.n_faces[None]):
            assert self.topo_check_cnt_face_v2f[i] == 3, "[ERROR] topological_check v2f fid{} appear {}!=3 times".format(
                i, self.topo_check_cnt_face_v2f[i])

        # check e2f and e2v
        self.topo_check_cnt_face_e2f.fill(0)
        self.topo_check_cnt_vertex_on_face_e2v.fill(0)

        ti.loop_config(serialize=True)
        for eid in range(self.n_edges[None]):
            v1id, v2id, f1id, f2id = self.edges_vid_fid[eid]
            assert v1id < v2id, "[ERROR] topological_check edges{} v1id{} >= v2id{}".format(
                eid, v1id, v2id)

            fid = f1id

            self.topo_check_cnt_face_e2f[fid] += 1
            self.edge_vertex_check_func(fid, v1id, v2id, eid)

            if f2id != -1:
                self.topo_check_cnt_face_e2f[f2id] += 1
                self.edge_vertex_check_func(f2id, v1id, v2id, eid)

        ti.loop_config(serialize=True)
        for fid in range(self.n_faces[None]):
            for i in range(3):
                assert self.topo_check_cnt_vertex_on_face_e2v[fid][i] == 1, "[ERROR] topological_check f{} edge error".format(
                    fid)
            assert self.topo_check_cnt_face_e2f[fid] == 3, "[ERROR] topological_check e2f fid{} appear {}!=3 times".format(
                fid, self.topo_check_cnt_face_e2f[fid])

        # check v2e
        self.topo_check_cnt_edge_remain.fill(0)

        ti.loop_config(serialize=True)
        for vid in range(self.n_vertices[None]):
            for i in range(self.vertices_eid_cnt[vid]):
                eid = self.vertices_eid[vid][i]
                if self.edges_vid_fid[eid][0] == vid or \
                        self.edges_vid_fid[eid][1] == vid:
                    self.topo_check_cnt_edge_remain[eid] += 1
                else:
                    assert False, "[ERROR] topological_check v2e vid{} not on eid{}.".format(
                        vid, eid)

        ti.loop_config(serialize=True)
        for eid in range(self.n_edges[None]):
            assert self.topo_check_cnt_edge_remain[eid] == 2, "[ERROR] topological_check v2e eid{} error.".format(
                eid)

    def topological_check(self):
        raise NotImplementedError
        self.topological_check_kernel()

    def topo_info_taichi_to_python(self):
        """
        Transfer data from taichi scope to python scope.
        """
        if True in self.topo_modified.to_numpy():
            # vertex v2e, v2f
            self.vertices_eid_python = self.vertices_eid.to_numpy()
            self.vertices_eid_cnt_python = self.vertices_eid_cnt.to_numpy()

            self.vertices_fid_python = self.vertices_fid.to_numpy()
            self.vertices_fid_cnt_python = self.vertices_fid_cnt.to_numpy()

            # face f2v
            self.faces_vid_python = self.faces_vid.to_numpy()

            # edge e2v, e2f
            self.edges_vid_fid_python = self.edges_vid_fid.to_numpy()

            self.topo_modified.fill(False)

    def reset(self):
        """
        Reset all configuration to initial state.
        - number of vert, face, edge
        - v2f, v2e, f2v, e2v, e2f
        """
        self.topo_modified.fill(True)

        # number of total vertices
        self.n_vertices.fill(int(self.origin_mesh.vertices.shape[0]))

        # v2e, v2f
        self.vertices_eid_cnt.fill(0)
        self.vertices_fid_cnt.fill(0)

        # number of total faces
        self.n_faces.fill(int(self.origin_mesh.faces.shape[0]))

        # faces' adjacent vertices id
        self.faces_vid.from_numpy(
            np.tile(
                np.pad(self.origin_mesh.faces.astype(np.int32),
                       ((0, self.nmax_faces - int(self.origin_mesh.faces.shape[0])), (0, 0)), 
                       'constant', constant_values=0)[None, ...],
                reps=(self.batch_size, 1, 1)
        ))

        for fid in range(self.n_faces[0]):
            v1id, v2id, v3id = self.faces_vid[0, fid]
            self.append_vertices_fid_allbatch_kernel(v1id, fid)
            self.append_vertices_fid_allbatch_kernel(v2id, fid)
            self.append_vertices_fid_allbatch_kernel(v3id, fid)

        # total number of edges
        self.n_edges.fill(len(self.edges_list))

        # edges' adjacent vertices id and faces id"
        self.edges_vid_fid.from_numpy(
            np.tile(
                np.pad(np.array(self.edges_list, dtype=np.int32),
                       ((0, self.nmax_edges - len(self.edges_list)), (0, 0)), 
                       'constant', constant_values=0)[None, ...],
                reps=(self.batch_size, 1, 1)
        ))

        for eid in range(len(self.edges_list)):
            v1id, v2id, f1id, f2id = self.edges_list[eid]
            self.append_vertices_eid_allbatch_kernel(v1id, eid)
            self.append_vertices_eid_allbatch_kernel(v2id, eid)

    def get_state(self) -> List[dict]:
        nv_batch = self.n_vertices.to_numpy()
        nf_batch = self.n_faces.to_numpy()
        ne_batch = self.n_edges.to_numpy()
        ve_cnt_batch = self.vertices_eid_cnt.to_numpy()
        ve_cnt_max_batch = ve_cnt_batch.max(axis=1)
        vf_cnt_batch = self.vertices_fid_cnt.to_numpy()
        vf_cnt_max_batch = vf_cnt_batch.max(axis=1)

        vertices_eid_batch = self.vertices_eid.to_numpy()
        vertices_fid_batch = self.vertices_fid.to_numpy()
        faces_vid_batch = self.faces_vid.to_numpy()
        edges_vid_fid_batch = self.edges_vid_fid.to_numpy()

        return [{
            "class":"DynamicTriangularMesh",
            "n_vertices":nv_batch[batch_idx],
            "vertices_eid":slice2D_array(vertices_eid_batch[batch_idx, ...], nv_batch[batch_idx], ve_cnt_max_batch[batch_idx]),
            "vertices_fid":slice2D_array(vertices_fid_batch[batch_idx, ...], nv_batch[batch_idx], vf_cnt_max_batch[batch_idx]),
            "vertices_eid_cnt":slice_array(ve_cnt_batch[batch_idx, ...], nv_batch[batch_idx]),
            "vertices_fid_cnt":slice_array(vf_cnt_batch[batch_idx, ...], nv_batch[batch_idx]),
            "n_faces":nf_batch[batch_idx],
            "faces_vid":slice_array(faces_vid_batch[batch_idx, ...], nf_batch[batch_idx]),
            "n_edges":ne_batch[batch_idx],
            "edges_vid_fid":slice_array(edges_vid_fid_batch[batch_idx, ...], ne_batch[batch_idx])
        } for batch_idx in range(self.batch_size)]
    
    def set_state(self, states: List[dict]) -> None:
        assert isinstance(states, list)
        assert isinstance(states[0], dict)
        assert len(states) == self.batch_size

        self.topo_modified.fill(True)

        self.n_vertices.from_numpy(np.array([state["n_vertices"] for state in states]))
        self.vertices_eid.from_numpy(np.array([pad2D_array(state["vertices_eid"], 
                                               self.nmax_vertices, self.nmax_edges_connect_to_vertex) for state in states]))
        self.vertices_fid.from_numpy(np.array([pad2D_array(state["vertices_fid"],
                                               self.nmax_vertices, self.nmax_faces_connect_to_vertex) for state in states]))
        self.vertices_eid_cnt.from_numpy(np.array([pad_array(state["vertices_eid_cnt"], self.nmax_vertices) for state in states]))
        self.vertices_fid_cnt.from_numpy(np.array([pad_array(state["vertices_fid_cnt"], self.nmax_vertices) for state in states]))
        self.n_faces.from_numpy(np.array([state["n_faces"] for state in states]))
        self.faces_vid.from_numpy(np.array([pad_array(state["faces_vid"], self.nmax_faces) for state in states]))
        self.n_edges.from_numpy(np.array([state["n_edges"] for state in states]))
        self.edges_vid_fid.from_numpy(np.array([pad_array(state["edges_vid_fid"], self.nmax_edges) for state in states]))
