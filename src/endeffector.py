import taichi as ti

import os
import copy
from typing import Dict, List, Tuple

import trimesh
import numpy as np
import torch

import batch_urdf
import mesh_to_sdf

from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

import src.cbase
import src.utils
import src.maths


@ti.data_oriented
class TiSdf:
    @staticmethod
    def _get_sdf_parameters(sdf_cfg: DictConfig) -> list:
        """return scan_count, scan_resolution, sdf_diff, sdf_beta, sdf_gamma"""
        return [sdf_cfg.scan_count,
                sdf_cfg.scan_resolution,
                sdf_cfg.sdf_diff,
                sdf_cfg.sdf_beta,
                sdf_cfg.sdf_gamma]
    
    def __init__(self, batch_size: int, sdf_cfg: DictConfig, output_cfg: DictConfig, 
                 link_scenes: Dict[str, trimesh.Scene], link_strs: List[str]) -> None:
        """
        Class used to manage sdf of articulated object.
        """
        self.batch_size = batch_size
        self.print_info = output_cfg.print_info

        self.use_sdf_cache: bool = sdf_cfg.use_sdf_cache
        self.sdf_file = sdf_cfg.sdf_file
        self.sdf_path = to_absolute_path(
            os.path.join(sdf_cfg.directory, self.sdf_file))
        
        self.link_scenes = link_scenes
        self.link_strs = link_strs
        self.link_n = len(self.link_strs)

        if self.use_sdf_cache and os.path.exists(self.sdf_path):
            sdf_data = np.load(self.sdf_path, allow_pickle=True)
            sdf_params = self._get_sdf_parameters(sdf_cfg)
            cache_params = sdf_data[0]
            if cache_params != sdf_params:
                self._calculate_and_save_sdf(sdf_cfg)
                sdf_data = np.load(self.sdf_path, allow_pickle=True)
                if self.print_info:
                    print("[INFO] cache file parameters:{} doesn't match sdf parameters:{}. Recalculate sdf.".format(
                        cache_params, sdf_params))
            else:
                if self.print_info:
                    print("[INFO] successfully loaded endeffector sdf file.")

        else:
            self._calculate_and_save_sdf(sdf_cfg)
            sdf_data = np.load(self.sdf_path, allow_pickle=True)
            if self.print_info:
                print("[INFO] Recalculate endeffector sdf.")

        self.voxels = [None for _ in range(self.link_n)]
        self.grads = [None for _ in range(self.link_n)]
        self.bounds = np.zeros(shape=(self.link_n, 2, 3), dtype=np.float32)

        self.res = np.zeros(shape=(self.link_n, 3), dtype=np.int32)
        self.diff = np.zeros(shape=(self.link_n, 3), dtype=np.float32)
        for l in range(self.link_n):
            self.voxels[l], self.grads[l], self.bounds[l] = sdf_data[l + 1]
            self.res[l] = self.voxels[l].shape
            for i in range(3):
                self.diff[l, i] = (self.bounds[l, 1, i] - self.bounds[l, 0, i]) / \
                    (self.res[l, i] - 1)
        
        self.sdf_cat = []
        for l in range(self.link_n):
            self.sdf_cat.append(ti.Vector.field(
                n=4, dtype=ti.f32, shape=self.voxels[l].shape))
            self.sdf_cat[l].from_numpy(
                np.concatenate((self.voxels[l][:, :, :, None], self.grads[l]), axis=3).astype(np.float32))

        self.bounds_field = ti.field(dtype=ti.f32, shape=self.bounds.shape)
        self.bounds_field.from_numpy(np.array(self.bounds, dtype=np.float32))

        self.res_field = ti.field(dtype=ti.i32, shape=self.res.shape)
        self.res_field.from_numpy(np.array(self.res, dtype=np.int32))

        self.diff_field = ti.field(dtype=ti.f32, shape=self.diff.shape)
        self.diff_field.from_numpy(np.array(self.diff, dtype=np.float32))

        self.trans_mat_fwd: ti.MatrixField = ti.Matrix.field(n=4, m=4, dtype=ti.f32, shape=(self.batch_size, self.link_n))
        self.trans_mat_inv: ti.MatrixField = ti.Matrix.field(n=4, m=4, dtype=ti.f32, shape=(self.batch_size, self.link_n))

        self.n_accmulate = ti.field(dtype=ti.i32, shape=(self.batch_size, ))

        self.sdf_inf: float = sdf_cfg.sdf_inf

    def _calculate_and_save_sdf(self, sdf_cfg: DictConfig) -> None:
        sdf_params = self._get_sdf_parameters(sdf_cfg)
        scan_count, scan_resolution, sdf_diff, sdf_beta, sdf_gamma = sdf_params
        data = [sdf_params]

        for link_str in self.link_strs:
            mesh = self.link_scenes[link_str].dump(concatenate=True)
            bounds = np.zeros_like(self.link_scenes[link_str].bounds).tolist()

            for i in range(2):
                bounds[i][:] = (1 + sdf_beta) * self.link_scenes[link_str].bounds[i][:] - \
                    sdf_beta * self.link_scenes[link_str].bounds[1 - i][:] + \
                    (2 * i - 1) * sdf_gamma

            res = [0 for i in range(3)]
            for i in range(3):
                res[i] = int((bounds[1][i] - bounds[0][i]) / sdf_diff[i]) + 1

            print(f"Scan link {link_str} 's point cloud. Res:{res}")
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            cloud = mesh_to_sdf.get_surface_point_cloud(
                mesh, surface_point_method='scan', scan_count=scan_count, scan_resolution=scan_resolution)

            print(f"Use link {link_str} 's point cloud to get sdf.")
            sdf, grads = cloud.get_sdf_in_batches(query_points=src.utils.get_raster_points(
                res, bounds), use_depth_buffer=True, return_gradients=True)

            voxels = sdf.reshape(tuple(res))
            grads = grads.reshape(tuple(res + [3]))
            data.append([voxels, grads, bounds])

        obj_arr = np.array(data, dtype=object)
        os.makedirs(os.path.split(self.sdf_path)[0], exist_ok=True)
        np.save(self.sdf_path, obj_arr)

    @ti.func
    def query_single_sdf_func(self, batch_idx: ti.i32, pos: ti.types.vector(3, ti.f32), l: ti.i32,
                              sdf_cat) -> ti.types.vector(4, ti.f32):
        """
        Return sdf of link `l`
        """
        ret_val = ti.Vector([self.sdf_inf, 0., 0., 0.], dt=ti.f32)

        # coordinate transform
        pos_homo = src.maths.point_to_homo_func(pos)
        pos_link_frame_homo = self.trans_mat_inv[batch_idx, l] @ pos_homo

        # to calculate which cube should use in trilinear interpolation
        cube_id = ti.Vector.zero(dt=ti.i32, n=3)
        offset = ti.Vector.zero(dt=ti.f32, n=3)
        for i in ti.static(range(3)):
            offset[i] = pos_link_frame_homo[i] - self.bounds_field[l, 0, i]
            cube_id[i] = ti.floor(offset[i] / self.diff_field[l, i], dtype=ti.i32)

        if 0 <= cube_id[0] and cube_id[0] < self.res_field[l, 0] - 1 \
                and 0 <= cube_id[1] and cube_id[1] < self.res_field[l, 1] - 1 \
                and 0 <= cube_id[2] and cube_id[2] < self.res_field[l, 2] - 1:

            # coordinate inside the cube
            cube_coord = ti.Vector.zero(dt=ti.f32, n=3)
            for i in range(3):
                cube_coord[i] = (offset[i] - cube_id[i] * self.diff_field[l, i]) / self.diff_field[l, i]
                assert cube_coord[i] >= -src.utils.CUBE_COORDINATE_EPS and cube_coord[i] <= 1.0 + src.utils.CUBE_COORDINATE_EPS, "[ERROR] cube_coord[{}]:{} out of range.".format(
                    i, cube_coord[i])

            # read sdf data
            local_sdf_and_grad = ti.Vector.zero(dt=ti.f32, n=4)

            sdf_val = ti.Vector.zero(dt=ti.f32, n=8, m=4)
            for dx, dy, dz in ti.static(ti.ndrange(2, 2, 2)):
                i = dx * 4 + dy * 2 + dz
                sdf_val[i, :] = sdf_cat[cube_id[0] + dx, cube_id[1] + dy, cube_id[2] + dz]
            # trilinear interpolation
            local_sdf_and_grad = src.maths.trilinear_4D_func(cube_coord, sdf_val)

            local_sdf = local_sdf_and_grad[0]
            local_grad = local_sdf_and_grad[1:4]

            # rotate grad back to world frame
            ret_val[0] = local_sdf
            local_grad_homo = src.maths.vector_to_homo_func(local_grad)
            world_grad_homo = self.trans_mat_fwd[batch_idx, l] @ local_grad_homo
            ret_val[1:4] = world_grad_homo[0:3]

        return ret_val
    
    @ti.kernel
    def query_batch_sdf_kernel(self, pos: ti.template(), ans_each_link: ti.template(), 
                               sdf_cat: ti.template(), l: int,
                               n_field: ti.template(), n_accumulate: ti.template()):
        """
        Args:
            pos: b x n x 3 3D Vector field
            ans_each_link: b x n x l x 4 4D Vector field
        """
        src.maths.get_accumulate_func(n_field, n_accumulate, self.batch_size)
        for ib in range(n_accumulate[self.batch_size - 1]):
            b, i = src.maths.get_batch_and_idx(ib, self.batch_size, n_accumulate)
            tmp_ans = self.query_single_sdf_func(b, pos[b, i], l, sdf_cat)
            ans_each_link[b, i, l] = tmp_ans[0:4]

    @ti.kernel
    def sdf_merge_kernel(self, ans_each_link: ti.template(),
                         ans_merge: ti.template(),
                         n_field: ti.template(), n_accumulate: ti.template()):
        src.maths.get_accumulate_func(n_field, n_accumulate, self.batch_size)
        for ib in range(n_accumulate[self.batch_size - 1]):
            b, i = src.maths.get_batch_and_idx(ib, self.batch_size, n_accumulate)
            min_sdf = ti.math.inf
            ans_merge[b, i] = ti.Vector([self.sdf_inf, 0., 0., 0.], dt=ti.f32)
            for l in range(self.link_n):
                if ans_each_link[b, i, l][0] < min_sdf:
                    ans_merge[b, i] = ans_each_link[b, i, l]
                    min_sdf = ans_each_link[b, i, l][0]

    def query_sdf(self, pos: ti.Field, ans_each_link: ti.Field, ans_merge: ti.Field, n_field: ti.Field,
                  fwd_matrices: np.ndarray, inv_matrices: np.ndarray):
        """
        query sdf and store in ans.

        Args:
            pos: b x n x 3 3D Vector field
            ans_each_link: b x n x l x 4 4D Vector field
            ans_merge: b x n x 4 4D Vector field
        """
        self.trans_mat_fwd.from_numpy(fwd_matrices.astype(np.float32))
        self.trans_mat_inv.from_numpy(inv_matrices.astype(np.float32))
        for l in range(self.link_n):
            self.query_batch_sdf_kernel(pos, ans_each_link, self.sdf_cat[l], l, n_field, self.n_accmulate)
        self.sdf_merge_kernel(ans_each_link, ans_merge, n_field, self.n_accmulate)


@ti.data_oriented
class EndEffector(src.cbase.TiObject):
    def __init__(self, batch_size: int, endeffector_cfg: DictConfig, output_cfg: DictConfig) -> None:
        assert isinstance(batch_size, int)
        assert isinstance(endeffector_cfg, DictConfig)
        assert isinstance(output_cfg, DictConfig)
        self.batch_size = batch_size
        self.endeffector_cfg = endeffector_cfg

        self.directory: str = endeffector_cfg.directory
        self.urdf_file: str = endeffector_cfg.urdf_file
        self.urdf_path: str = to_absolute_path(
            os.path.join(self.directory, self.urdf_file))
        
        self.init_pose: dict = OmegaConf.to_container(endeffector_cfg.init_pose)

        self.torch_dtype = getattr(torch, endeffector_cfg.torch_dtype)
        self.torch_device = endeffector_cfg.torch_device
        self.yrdf = batch_urdf.URDF(self.batch_size, self.urdf_path, dtype=self.torch_dtype, device=self.torch_device)

        self.link_strs = []
        for link_name in self.yrdf.link_map.keys():
            if len(self.yrdf.get_link_scene(link_name, True).geometry) > 0:
                self.link_strs.append(link_name)
        self.link_scenes = {link_name: self.yrdf.get_link_scene(link_name, True) for link_name in self.link_strs}
        self.link_n = len(self.link_scenes)

        self.yrdf_cfg: List[dict] = [copy.deepcopy(self.init_pose) for b in range(self.batch_size)]
        self.fwd_matrices = np.zeros((self.batch_size, self.link_n, 4, 4), np.float32)
        """[b, n, 4, 4]"""
        self.inv_matrices = np.zeros((self.batch_size, self.link_n, 4, 4), np.float32)
        """[b, n, 4, 4]"""
        self.update_pose([self.init_pose] * self.batch_size)

        self.sdf = TiSdf(self.batch_size, endeffector_cfg.sdf_cfg, output_cfg, self.link_scenes, self.link_strs)

        self.dx_eps: float = endeffector_cfg.dx_eps
        self.sdf_inf: float = endeffector_cfg.sdf_cfg.sdf_inf

    def _wrap_cfg(self, yrdf_cfg: List[dict]) -> Dict[str, torch.Tensor]:
        all_pose = {}
        for joint_name in self.yrdf.actuated_joint_map.keys():
            all_pose[joint_name] = torch.tensor([yrdf_cfg[b][joint_name] for b in range(self.batch_size)], dtype=self.torch_dtype, device=self.torch_device)
        return src.utils.floating_pose6D_to_pose7D(all_pose)

    def _update_pose(self, batch_idx: int, new_pose: dict):
        self.yrdf_cfg[batch_idx] = copy.deepcopy(new_pose)
        self.yrdf.update_cfg(self._wrap_cfg(self.yrdf_cfg))
        self.fwd_matrices[...] = np.concatenate([src.utils.torch_to_numpy(self.yrdf.link_transform_map[ls])[:, None, ...] for ls in self.link_strs], axis=1)
        self.inv_matrices[...] = np.linalg.pinv(self.fwd_matrices)

    def _update_pose_quick(self, new_pose: List[dict]):
        self.yrdf_cfg = copy.deepcopy(new_pose)
        self.yrdf.update_cfg(self._wrap_cfg(self.yrdf_cfg))
        self.fwd_matrices[...] = np.concatenate([src.utils.torch_to_numpy(self.yrdf.link_transform_map[ls])[:, None, ...] for ls in self.link_strs], axis=1)
        self.inv_matrices[...] = np.linalg.pinv(self.fwd_matrices)

    def update_pose(self, new_pose: List[dict]):
        assert isinstance(new_pose, list)
        assert len(new_pose) == self.batch_size
        for b, newp in enumerate(new_pose):
            self._update_pose(b, newp)

    def update_pose_quick(self, new_pose: List[dict]):
        assert isinstance(new_pose, list)
        assert len(new_pose) == self.batch_size
        self._update_pose_quick(new_pose)

    def get_mesh(self) -> List[trimesh.Trimesh]:
        return [self.yrdf.get_scene(b).dump(concatenate=True) for b in range(self.batch_size)]
    
    def reset(self):
        self.update_pose([self.init_pose] * self.batch_size)

    def set_state(self, states: List[dict]) -> None:
        assert isinstance(states, list)
        assert isinstance(states[0], dict)
        assert len(states) == self.batch_size
        self.update_pose([state["yrdf_cfg"] for state in states])

    def get_state(self) -> dict:
        return [{
            "class": "EndEffector",
            "yrdf_cfg": yrdf_cfg,
        } for yrdf_cfg in copy.deepcopy(self.yrdf_cfg)]
    
    def query_sdf(self, pos: ti.MatrixField, ans_each_link: ti.MatrixField, 
                  ans_merge: ti.MatrixField, n_field: ti.Field):
        """
        query sdf and store in ans.

        Args:
            pos: b x n x 3 3D Vector field
            ans_each_link: b x n x l x 4 4D Vector field
            ans_merge: b x n x 4 4D Vector field
        """
        assert isinstance(pos, ti.MatrixField) and isinstance(ans_each_link, ti.MatrixField) \
            and isinstance(ans_merge, ti.MatrixField) and isinstance(n_field, ti.Field)
        assert pos.n == 3 and ans_each_link.n == 4 and ans_merge.n == 4
        assert len(pos.shape) == 2 and pos.shape[0] == self.batch_size
        assert len(ans_each_link.shape) == 3 and ans_each_link.shape[0] == self.batch_size \
            and ans_each_link.shape[2] == self.link_n
        assert len(ans_merge.shape) == 2 and ans_merge.shape[0] == self.batch_size

        self.sdf.query_sdf(pos, ans_each_link, ans_merge, n_field, self.fwd_matrices, self.inv_matrices)

