import taichi as ti
import numpy as np
import torch
import os
import copy
import time
from typing import Union, List, Literal, Dict
from math import ceil

import batch_urdf
import trimesh
import mesh_to_sdf

from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from src.utils import *
from src.maths import *
from src.cbase import *

SCISSOR_ZERO_ACTION = {
    "joint_0": 0.0,
    "joint_1": [0.0] * 6
}


@ti.data_oriented
class ScissorUrdf:
    def __init__(self, batch_size: int, scissor_cfg: DictConfig, output_cfg: DictConfig) -> None:
        self.batch_size = batch_size
        self.print_info = output_cfg.print_info

        self.directory: str = scissor_cfg.directory
        self.urdf_file: str = scissor_cfg.urdf_file
        self.urdf_path: str = to_absolute_path(
            os.path.join(self.directory, self.urdf_file))
        
        self.torch_dtype = getattr(torch, scissor_cfg.torch_dtype)
        self.torch_device = scissor_cfg.torch_device
        self.yrdf = batch_urdf.URDF(self.batch_size, self.urdf_path, dtype=self.torch_dtype, device=self.torch_device)

        self.blade_name: List[str] = copy.deepcopy(scissor_cfg.blade_name)
        
        self.fwd_matrices = np.zeros((self.batch_size, 2, 4, 4), dtype=np.float32)
        """[b, 2, 4, 4]"""

        self.inv_matrices = np.zeros((self.batch_size, 2, 4, 4), dtype=np.float32)
        """[b, 2, 4, 4]"""

        self.init_pose: Dict[str, list] = OmegaConf.to_container(scissor_cfg.init_pose) 

        self.joint_0_origin: np.ndarray = torch_to_numpy(self.yrdf.joint_map["joint_0"].origin)[0, ...]
        """[4, 4]"""
        self.joint_0_axs: np.ndarray = torch_to_numpy(self.yrdf.joint_map["joint_0"].axis)[0, ...]
        """[3, ]"""

        self.cut_direction_origin = np.array(OmegaConf.to_container(
            scissor_cfg.cut_direction))
        self.cut_direction_origin /= np.linalg.norm(self.cut_direction_origin)

        self.direct_cfg: List[dict] = [self.init_pose] * self.batch_size
        self.update_cfg([self.init_pose] * self.batch_size, ["joint_0"] * self.batch_size, scissor_cfg.dx_eps)

        self.use_front_cache: bool = scissor_cfg.use_front_cache
        self.front_file = scissor_cfg.front_file
        self.front_path = to_absolute_path(
            os.path.join(self.directory, self.front_file))

        self.front_theta_arr: np.ndarray = None
        self.front_dist_arr: np.ndarray = None
        self.neg_front_dist_arr: np.ndarray = None

    def get_front_parameters(self, front_cfg: DictConfig) -> list:
        """return [sample_dtheta, dx_tol, sdf_tol, x_lower, x_upper, x_shift, poly_deg]"""
        return [front_cfg.sample_dtheta,
                front_cfg.dx_tol,
                front_cfg.sdf_tol,
                front_cfg.x_lower,
                front_cfg.x_upper,
                front_cfg.x_shift,
                front_cfg.poly_deg]

    def _update_cfg_direct(self, batch_idx: int, new_pose: dict):
        self.direct_cfg[batch_idx] = copy.deepcopy(new_pose)

        all_pose = {}
        for joint_name in new_pose.keys():
            all_pose[joint_name] = torch.tensor([self.direct_cfg[b][joint_name] for b in range(self.batch_size)], dtype=self.torch_dtype, device=self.torch_device)
        self.yrdf.update_cfg(floating_pose6D_to_pose7D(all_pose))

        for l in range(2):
            self.fwd_matrices[batch_idx, l] = torch_to_numpy(self.yrdf.link_transform_map[self.blade_name[l]])[batch_idx, ...]
            self.inv_matrices[batch_idx, l] = np.linalg.inv(self.fwd_matrices[batch_idx, l])

    def _update_cfg_direct_quick(self, new_pose: Dict[str, torch.Tensor]):
        for batch_idx in range(self.batch_size):
            self.direct_cfg[batch_idx]["joint_0"] = float(new_pose["joint_0"][batch_idx])
            self.direct_cfg[batch_idx]["joint_1"] = new_pose["joint_1"][batch_idx].tolist()

        self.yrdf.update_cfg(floating_pose6D_to_pose7D(new_pose))

        for l in range(2):
            self.fwd_matrices[:, l] = torch_to_numpy(self.yrdf.link_transform_map[self.blade_name[l]])[:, ...]
            self.inv_matrices[:, l] = np.linalg.inv(self.fwd_matrices[:, l])
    
    def update_cfg_direct(self, new_pose: List[dict]):
        """Update self.direct_cfg, self.fwd_matrix, self.inv_matrix, calling self.yrdf.update_cfg()."""
        assert isinstance(new_pose, list)
        for b, p in zip(range(self.batch_size), new_pose):
            self._update_cfg_direct(b, p)

    def _update_cfg(self, batch_idx: int, new_pose: dict, rotation_center: Literal['joint_0', 'front', 'direct'], dx_eps: float):
        assert rotation_center in ["joint_0", "front", "direct"]

        if rotation_center == "direct":
            self._update_cfg_direct(batch_idx, new_pose)

        else:
            new_pose_modified = copy.deepcopy(new_pose)
            self._update_cfg_direct(batch_idx, new_pose)

            pos, vec, axs = self._get_cut_direction(batch_idx, dx_eps)
            if rotation_center == "joint_0":
                for i in range(3):
                    new_pose_modified["joint_1"][i] += new_pose["joint_1"][i] - pos[i]

            elif rotation_center == "front":
                front_pos = pos + vec * \
                    self.get_cut_front_dist(new_pose["joint_0"])
                for i in range(3):
                    new_pose_modified["joint_1"][i] += new_pose["joint_1"][i] - front_pos[i]

            else:
                raise NotImplementedError

            self._update_cfg_direct(batch_idx, new_pose_modified)

    def _update_cfg_quick(self, new_poses: List[dict], rotation_center: Literal['joint_0', 'front', 'direct'], dx_eps: float):
        assert rotation_center in ["joint_0", "front", "direct"]
        assert isinstance(new_poses, list)
        assert len(new_poses) == self.batch_size

        assert rotation_center in ["joint_0", "front", "direct"]

        new_pose_tensor = {}
        for k in ["joint_0", "joint_1"]:
            new_pose_tensor[k] = torch.tensor([new_pose[k] for new_pose in new_poses], dtype=self.torch_dtype, device=self.torch_device)
            
        if rotation_center == "direct":
            self._update_cfg_direct_quick(new_pose_tensor)

        else:
            new_pose_tensor_modified = copy.deepcopy(new_pose_tensor)
            self._update_cfg_direct_quick(new_pose_tensor)

            pos, vec, axs = self._get_cut_direction_quick(dx_eps)
            if rotation_center == "joint_0":
                new_pose_tensor_modified["joint_1"][:, :3] += new_pose_tensor["joint_1"][:, :3] - torch.tensor(pos, dtype=self.torch_dtype, device=self.torch_device)

            elif rotation_center == "front":
                front_pos = pos + vec * \
                    self.get_cut_front_dist(new_pose_tensor["joint_0"])[:, None]
                new_pose_tensor_modified["joint_1"][:, :3] += new_pose_tensor["joint_1"][:, :3] - torch.tensor(front_pos, dtype=self.torch_dtype, device=self.torch_device)

            else:
                raise NotImplementedError

            self._update_cfg_direct_quick(new_pose_tensor_modified)


    def update_cfg(self, new_poses: List[dict], rotation_centers: List[Literal['joint_0', 'front', 'direct']], dx_eps: float):
        """
        Update self.direct_cfg, self.fwd_matrix, self.inv_matrix, calling self.yrdf.update_cfg().

        keys in new_pose:
            - joint_0: float, open angle
            - joint_1: 6D list, translation and rotation (around rotation center)

        rotation_center:
            - "joint_0": rotation around joint_0
            - "front": rotation around (old) front point
            - "direct": directly modify origin configuration
        """
        assert isinstance(new_poses, list)
        assert isinstance(rotation_centers, list)
        assert len(rotation_centers) == self.batch_size and len(new_poses) == self.batch_size

        for batch_idx, new_pose, rotation_center in zip(range(self.batch_size), new_poses, rotation_centers):
            self._update_cfg(batch_idx, new_pose, rotation_center, dx_eps)

    def update_pose_quick(self, new_poses: List[dict], rotation_center: Literal['joint_0', 'front', 'direct'], dx_eps: float) -> None:
        """
        Update self.direct_cfg, self.fwd_matrix, self.inv_matrix, calling self.yrdf.update_cfg().

        keys in new_pose:
            - joint_0: float, open angle
            - joint_1: 6D list, translation and rotation (around rotation center)

        rotation_center:
            - "joint_0": rotation around joint_0
            - "front": rotation around (old) front point
            - "direct": directly modify origin configuration
        """
        self._update_cfg_quick(new_poses, rotation_center, dx_eps)

    def update_cfg_given_batch(self, batch_idx, new_pose: dict, rotation_center: Literal['joint_0', 'front', 'direct'], dx_eps: float):
        self._update_cfg(batch_idx, new_pose, rotation_center, dx_eps)

    def _get_direct_cfg(self, batch_idx: int, need_deepcopy: bool) -> dict:
        if need_deepcopy:
            return copy.deepcopy(self.direct_cfg[batch_idx])
        else:
            return self.direct_cfg[batch_idx]

    def get_direct_cfg(self, need_deepcopy: bool) -> List[dict]:
        """
        Return a copy of self.direct_cfg
        """
        return [self._get_direct_cfg(b, need_deepcopy) for b in range(self.batch_size)]

    def get_mesh(self) -> List[trimesh.Trimesh]:
        return [self.yrdf.get_scene(b).dump(concatenate=True) for b in range(self.batch_size)]

    def get_limit(self, joint_name: str) -> Union[batch_urdf.Limit, None]:
        return self.yrdf.joint_map[joint_name].limit

    def _get_cut_direction(self, batch_idx: int, dx_eps: float) -> List[np.ndarray]:
        fwd_matrix_0 = self.fwd_matrices[batch_idx, 0]
        fwd_matrix_1 = self.fwd_matrices[batch_idx, 1]

        pos = (fwd_matrix_1 @ self.joint_0_origin @ point_to_homo(np.zeros((3, ), float)))[:3]

        vec0 = (fwd_matrix_0 @ vector_to_homo(self.cut_direction_origin))[:3]
        vec1 = (fwd_matrix_1 @ vector_to_homo(self.cut_direction_origin))[:3]

        vec = vec0 + vec1
        if np.linalg.norm(vec) > dx_eps:
            vec /= np.linalg.norm(vec)

        axs = (fwd_matrix_1 @ vector_to_homo(self.joint_0_axs))[:3]
        if np.linalg.norm(axs) > dx_eps:
            axs /= np.linalg.norm(axs)

        return [pos, vec, axs]
    
    def _get_cut_direction_quick(self, dx_eps: float) -> List[np.ndarray]:
        """
        Return:
            pos, vec, axs: [b, 3]
        """
        fwd_matrix_0 = self.fwd_matrices[:, 0] # [b, 4, 4]
        fwd_matrix_1 = self.fwd_matrices[:, 1] # [b, 4, 4]

        pos = (fwd_matrix_1 @ self.joint_0_origin[None, ...] @ point_to_homo(np.zeros((3, ), float))[None, :, None])[:, :3, 0]
        """[b, 3]"""

        vec0 = (fwd_matrix_0 @ vector_to_homo(self.cut_direction_origin)[None, :, None])[:, :3, 0]
        vec1 = (fwd_matrix_1 @ vector_to_homo(self.cut_direction_origin)[None, :, None])[:, :3, 0]

        vec = vec0 + vec1
        vec /= np.clip(np.linalg.norm(vec, axis=1, keepdims=True), dx_eps, np.inf)

        axs = (fwd_matrix_1 @ vector_to_homo(self.joint_0_axs)[None, :, None])[:, :3, 0]
        axs /= np.clip(np.linalg.norm(axs, axis=1, keepdims=True), dx_eps, np.inf)

        return [pos, vec, axs]
    
    def get_cut_direction(self, dx_eps: float) -> List[List[np.ndarray]]:
        """
        Need correct fwd_matrix.

        Return:
            - pos: joint_0 current position
            - vec: normalized cut direction
            - axs: normalized joint_0 axis direction
        """
        return [self._get_cut_direction(b, dx_eps) for b in range(self.batch_size)]

    def get_cut_front_dist(self, theta: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return np.interp(theta, self.front_theta_arr, self.front_dist_arr)
    
    def _get_cut_front_point(self, batch_idx: int, frame_name: Literal['world', 'joint_0'], dx_eps: float, theta: Union[float, None]) -> np.ndarray:
        assert frame_name in ["world", "joint_0"]
        assert (self.front_theta_arr is not None) and (self.front_dist_arr is not None) and (self.neg_front_dist_arr is not None), \
            "Please call self.initialize_front_point_data() first. "

        if theta is None:
            theta = self.direct_cfg[batch_idx]["joint_0"]

        pos, vec, axs = self._get_cut_direction(batch_idx, dx_eps)
        dist = self.get_cut_front_dist(theta) * vec
        if frame_name == 'world':
            dist += pos
        return dist
    
    def get_cut_front_point(self, frame_name: List[Literal['world', 'joint_0']], dx_eps: float, theta: List[Union[float, None]]) -> List[np.ndarray]:
        """
        Get the position of the intersection point between two blades. 
        """
        assert isinstance(frame_name, list)
        assert isinstance(theta, list)
        assert len(frame_name) == self.batch_size and len(theta) == self.batch_size
        return [self._get_cut_front_point(b, f, dx_eps, t) for b, f, t in zip(range(self.batch_size), frame_name, theta)]
    
    def get_cut_front_point_quick(self, frame_name: Literal['world', 'joint_0'], dx_eps: float, theta: Union[List[float], None]) -> np.ndarray:
        """
        Get the position of the intersection point between two blades.

        Return:
            [b, 3]
        """
        assert frame_name in ["world", "joint_0"]


        if theta is None:
            theta = [self.direct_cfg[batch_idx]["joint_0"] for batch_idx in range(self.batch_size)]

        pos, vec, axs = self._get_cut_direction_quick(dx_eps)
        dist = self.get_cut_front_dist(theta)[:, None] * vec
        if frame_name == 'world':
            dist += pos
        return dist

    def compute_theta_given_front_point_dist(self, front_point_dist: float) -> float:
        """
        Get the value of theta given distance between front point and joint_0. 
        """
        assert (self.front_theta_arr is not None) and (self.front_dist_arr is not None) and (self.neg_front_dist_arr is not None), \
            "Please call self.initialize_front_point_data() first. "

        theta = np.interp(-front_point_dist, self.neg_front_dist_arr,
                          self.front_theta_arr)
        return theta

    def reset(self, dx_eps):
        self.update_cfg([self.init_pose] * self.batch_size, ["joint_0"] * self.batch_size, dx_eps)


@ti.data_oriented
class ScissorSdf:
    def __init__(self, batch_size: int, scissor_cfg: DictConfig, output_cfg: DictConfig, scissor_urdf: ScissorUrdf) -> None:
        """
        Class used to calculate signed distance function. 
        """
        self.batch_size = batch_size
        self.print_info = output_cfg.print_info

        self.use_sdf_cache: bool = scissor_cfg.use_sdf_cache
        self.sdf_file = scissor_cfg.sdf_file
        self.sdf_path = to_absolute_path(
            os.path.join(scissor_urdf.directory, self.sdf_file))

        self.blade_scene = [scissor_urdf.yrdf.get_link_scene(
            scissor_urdf.blade_name[l], True) for l in range(2)]

        if self.use_sdf_cache and os.path.exists(self.sdf_path):
            sdf_data = np.load(self.sdf_path, allow_pickle=True)
            sdf_params = self.get_sdf_parameters(scissor_cfg.sdf_cfg)
            cache_params = sdf_data[0]
            if cache_params != sdf_params:
                self.calculate_and_save_sdf(scissor_cfg.sdf_cfg)
                sdf_data = np.load(self.sdf_path, allow_pickle=True)
                if self.print_info:
                    print("[INFO] cache file parameters:{} doesn't match sdf parameters:{}. Recalculate sdf.".format(
                        cache_params, sdf_params))
            else:
                if self.print_info:
                    print("[INFO] successfully loaded scissor sdf file.")

        else:
            self.calculate_and_save_sdf(scissor_cfg.sdf_cfg)
            sdf_data = np.load(self.sdf_path, allow_pickle=True)
            if self.print_info:
                print("[INFO] Recalculate scissor sdf.")

        self.voxels = [None, None]
        self.grads = [None, None]
        self.bounds = np.zeros(shape=(2, 2, 3), dtype=np.float32)

        self.res = np.zeros(shape=(2, 3), dtype=np.int32)
        self.diff = np.zeros(shape=(2, 3), dtype=np.float32)
        for l in range(2):
            self.voxels[l], self.grads[l], self.bounds[l] = sdf_data[l + 1]
            self.res[l] = self.voxels[l].shape
            for i in range(3):
                self.diff[l, i] = (self.bounds[l, 1, i] - self.bounds[l, 0, i]) / \
                    (self.res[l, i] - 1)

        self.sdf_cat_b0 = ti.Vector.field(
            n=4, dtype=ti.f32, shape=self.voxels[0].shape)
        self.sdf_cat_b0.from_numpy(
            np.concatenate((self.voxels[0][:, :, :, None], self.grads[0]), axis=3).astype(np.float32))

        self.sdf_cat_b1 = ti.Vector.field(
            n=4, dtype=ti.f32, shape=self.voxels[1].shape)
        self.sdf_cat_b1.from_numpy(
            np.concatenate((self.voxels[1][:, :, :, None], self.grads[1]), axis=3).astype(np.float32))

        self.bounds_field = ti.field(dtype=ti.f32, shape=self.bounds.shape)
        self.bounds_field.from_numpy(np.array(self.bounds, dtype=np.float32))

        self.res_field = ti.field(dtype=ti.i32, shape=self.res.shape)
        self.res_field.from_numpy(np.array(self.res, dtype=np.int32))

        self.diff_field = ti.field(dtype=ti.f32, shape=self.diff.shape)
        self.diff_field.from_numpy(np.array(self.diff, dtype=np.float32))

        self.trans_mat_fwd: ti.MatrixField = ti.Matrix.field(n=4, m=4, dtype=ti.f32, shape=(self.batch_size, 2))
        self.trans_mat_inv: ti.MatrixField = ti.Matrix.field(n=4, m=4, dtype=ti.f32, shape=(self.batch_size, 2))

        self.n_accmulate = ti.field(dtype=ti.i32, shape=(self.batch_size, ))

        self.sdf_inf = scissor_cfg.sdf_inf

    def get_sdf_parameters(self, sdf_cfg: DictConfig) -> list:
        """return scan_count, scan_resolution, sdf_diff, sdf_beta, sdf_gamma"""
        return [sdf_cfg.scan_count,
                sdf_cfg.scan_resolution,
                sdf_cfg.sdf_diff,
                sdf_cfg.sdf_beta,
                sdf_cfg.sdf_gamma]

    def calculate_and_save_sdf(self, sdf_cfg: DictConfig):
        sdf_params = self.get_sdf_parameters(sdf_cfg)
        scan_count, scan_resolution, sdf_diff, sdf_beta, sdf_gamma = sdf_params
        data = [sdf_params]

        for l in range(2):
            mesh = self.blade_scene[l].dump(concatenate=True)
            bounds = np.zeros_like(self.blade_scene[l].bounds).tolist()

            for i in range(2):
                bounds[i][:] = (1 + sdf_beta) * self.blade_scene[l].bounds[i][:] - \
                    sdf_beta * self.blade_scene[l].bounds[1 - i][:] + \
                    (2 * i - 1) * sdf_gamma

            res = [0 for i in range(3)]
            for i in range(3):
                res[i] = int((bounds[1][i] - bounds[0][i]) / sdf_diff[i]) + 1

            print(f"Scan blade{l} 's point cloud.")
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            cloud = mesh_to_sdf.get_surface_point_cloud(
                mesh, surface_point_method='scan', scan_count=scan_count, scan_resolution=scan_resolution)

            print(f"Use blade{l} 's point cloud to get sdf.")
            sdf, grads = cloud.get_sdf_in_batches(query_points=get_raster_points(
                res, bounds), use_depth_buffer=True, return_gradients=True)

            voxels = sdf.reshape(tuple(res))
            grads = grads.reshape(tuple(res + [3]))
            data.append([voxels, grads, bounds])

        obj_arr = np.array(data, dtype=object)
        os.makedirs(os.path.split(self.sdf_path)[0], exist_ok=True)
        np.save(self.sdf_path, obj_arr)

    @ti.func
    def query_single_sdf_func(self, batch_idx: ti.i32, pos: ti.types.vector(3, ti.f32)) -> ti.types.matrix(2, SDF_SHAPE, ti.f32):
        """return a 2 x SDF_SHAPE matrix, corresponding to 2 blade's sdf and grad"""
        ret_val = ti.Matrix.zero(dt=ti.f32, n=2, m=SDF_SHAPE)
        ret_val[0, 0] = ret_val[1, 0] = self.sdf_inf

        # coordinate transform
        pos_homo = point_to_homo_func(pos)
        pos_blade_frame_homo = ti.Matrix.zero(dt=ti.f32, n=2, m=4)
        for l in range(2):
            pos_blade_frame_homo[l, :] = self.trans_mat_inv[batch_idx, l] @ pos_homo

        # to calculate which cube should use in trilinear interpolation
        cube_id = ti.Matrix.zero(dt=ti.i32, n=2, m=3)
        offset = ti.Matrix.zero(dt=ti.f32, n=2, m=3)
        for l, i in ti.ndrange(2, 3):
            offset[l, i] = pos_blade_frame_homo[l, i] - \
                self.bounds_field[l, 0, i]
            cube_id[l, i] = ti.floor(
                offset[l, i] / self.diff_field[l, i], dtype=ti.i32)

        for l in range(2):
            if 0 <= cube_id[l, 0] and cube_id[l, 0] < self.res_field[l, 0] - 1 \
                    and 0 <= cube_id[l, 1] and cube_id[l, 1] < self.res_field[l, 1] - 1 \
                    and 0 <= cube_id[l, 2] and cube_id[l, 2] < self.res_field[l, 2] - 1:

                # coordinate inside the cube
                cube_coord = ti.Vector.zero(dt=ti.f32, n=3)
                for i in range(3):
                    cube_coord[i] = (offset[l, i] -
                                     cube_id[l, i] * self.diff_field[l, i]) / self.diff_field[l, i]
                    assert cube_coord[i] >= -CUBE_COORDINATE_EPS and cube_coord[i] <= 1.0 + CUBE_COORDINATE_EPS, "[ERROR] cube_coord[{}]:{} out of range.".format(
                        i, cube_coord[i])

                # read sdf data
                local_sdf_and_grad = ti.Vector.zero(dt=ti.f32, n=4)

                sdf_val = ti.Vector.zero(dt=ti.f32, n=8, m=4)
                for dx, dy, dz in ti.static(ti.ndrange(2, 2, 2)):
                    i = dx * 4 + dy * 2 + dz
                    if l == 0:
                        sdf_val[i, :] = self.sdf_cat_b0[cube_id[l, 0] +
                                                        dx, cube_id[l, 1] + dy, cube_id[l, 2] + dz]
                    else:
                        sdf_val[i, :] = self.sdf_cat_b1[cube_id[l, 0] +
                                                        dx, cube_id[l, 1] + dy, cube_id[l, 2] + dz]
                # trilinear interpolation
                local_sdf_and_grad = trilinear_4D_func(cube_coord, sdf_val)

                local_sdf = local_sdf_and_grad[0]
                local_grad = local_sdf_and_grad[1:4]

                # rotate grad back to world frame
                ret_val[l, 0] = local_sdf
                local_grad_homo = vector_to_homo_func(local_grad)
                world_grad_homo = self.trans_mat_fwd[batch_idx, l] @ local_grad_homo
                ret_val[l, 1:4] = world_grad_homo[0:3]

        return ret_val

    @ti.kernel
    def query_single_sdf_kernel(self, batch_idx: ti.f32, pos: ti.types.vector(3, ti.f32)) -> ti.types.matrix(2, SDF_SHAPE, ti.f32):
        """return a 2xSDF_SHAPE matrix, corresponding to 2 blade, sdf and grad and grad^2"""
        return self.query_single_sdf_func(batch_idx, pos)

    @ti.kernel
    def query_batch_sdf_merge_kernel(self, pos: ti.template(), ans: ti.template(), n_field: ti.template(), n_accumulate: ti.template()):
        get_accumulate_func(n_field, n_accumulate, self.batch_size)
        for ib in range(n_accumulate[self.batch_size - 1]):
            b, i = get_batch_and_idx(ib, self.batch_size, n_accumulate)
            tmp_ans = self.query_single_sdf_func(b, pos[b, i])

            l = -1
            if tmp_ans[0, 0] <= 0.0 and tmp_ans[1, 0] <= 0.0:
                if tmp_ans[1, 0] > tmp_ans[0, 0]:
                    l = 1
                else:
                    l = 0
            else:
                if tmp_ans[1, 0] > tmp_ans[0, 0]:
                    l = 0
                else:
                    l = 1

            ans[b, i] = tmp_ans[l, :]

    @ti.kernel
    def query_batch_sdf_kernel(self, pos: ti.template(), ans: ti.template(), n_field: ti.template(), n_accumulate: ti.template()):
        get_accumulate_func(n_field, n_accumulate, self.batch_size)
        for ib in range(n_accumulate[self.batch_size - 1]):
            b, i = get_batch_and_idx(ib, self.batch_size, n_accumulate)
            ans[b, i] = self.query_single_sdf_func(b, pos[b, i])

    def query_sdf(self, pos: ti.MatrixField, ans: ti.MatrixField, n_field: ti.Field, urdf: ScissorUrdf, need_merge: bool):
        """
        query sdf and store in ans.

        Args:
            - pos: (B x N) x (3) 3D (Vector) field
            - ans: (B x N) x (SDF_SHAPE) SDF_SHAPED (Vector) field (need_merge=True)
            - ans: (B x N) x (2 x SDF_SHAPE) (Matrix) field (need_merge=False)
        """
        self.trans_mat_fwd.from_numpy(urdf.fwd_matrices)
        self.trans_mat_inv.from_numpy(urdf.inv_matrices)
        if need_merge:
            self.query_batch_sdf_merge_kernel(pos, ans, n_field, self.n_accmulate)
        else:
            self.query_batch_sdf_kernel(pos, ans, n_field, self.n_accmulate)

    def query_single_sdf_python(self, batch_idx: int, pos: np.ndarray, urdf: ScissorUrdf) -> np.ndarray:
        """
        Query single sdf using python. 

        Args:
            - pos: np.ndarray((3,), float32)
            - urdf: ScissorUrdf

        Return:
            - sdf: np.ndarray((2,), float32)
        """
        ret_val = np.ones((2, ), dtype=np.float32) * self.sdf_inf

        pos_homo = point_to_homo(pos)
        pos_blade_frame_homo = [None] * 2
        for l in range(2):
            pos_blade_frame_homo[l] = urdf.inv_matrices[batch_idx, l] @ pos_homo
        pos_blade_frame_homo = np.array(pos_blade_frame_homo, np.float32)

        offset = pos_blade_frame_homo[:, :3] - self.bounds[:, 0, :]
        cube_id = np.floor(offset / self.diff).astype(np.int32)

        cube_coord = (offset - cube_id * self.diff) / self.diff
        for l in range(2):
            if 0 <= cube_id[l, 0] and cube_id[l, 0] < self.res[l, 0] - 1 and \
                    0 <= cube_id[l, 1] and cube_id[l, 1] < self.res[l, 1] - 1 and \
                    0 <= cube_id[l, 2] and cube_id[l, 2] < self.res[l, 2] - 1:
                sdf_val = np.zeros((8,), np.float32)
                for dx in range(2):
                    for dy in range(2):
                        for dz in range(2):
                            i = dx * 4 + dy * 2 + dz
                            xx = cube_id[l, 0] + dx
                            yy = cube_id[l, 1] + dy
                            zz = cube_id[l, 2] + dz
                            sdf_val[i] = self.voxels[l][xx, yy, zz]
                ret_val[l] = trilinear(cube_coord[l, :], sdf_val)

        return ret_val


@ti.data_oriented
class Scissor(TiObject):
    def __init__(self, batch_size: int, scissor_cfg: DictConfig, output_cfg: DictConfig) -> None:
        """Create a scissor object.  

        One can reuse this 'ScissorSdf' object while creating 'Scissor' object to save memory cost, as long as the scissor's geometry remains the same. """
        assert isinstance(batch_size, int)
        assert isinstance(scissor_cfg, DictConfig)
        assert isinstance(output_cfg, DictConfig)

        self.batch_size = batch_size
        self.print_info = output_cfg.print_info

        self.scissor_cfg = scissor_cfg
        self.urdf = ScissorUrdf(batch_size, scissor_cfg, output_cfg)
        self.sdf = ScissorSdf(batch_size, scissor_cfg, output_cfg, self.urdf)

        self.dx_eps: float = scissor_cfg.dx_eps
        self.sdf_inf: float = scissor_cfg.sdf_inf

        self.initialize_front_point_data()

    def update_pose(self, new_pose: List[dict], rotation_center: List[Literal['joint_0', 'front', 'direct']]) -> None:
        """
        new_pose:
            - joint_0: float, open angle
            - joint_1: 6D list, translation and rotation
        """
        assert isinstance(new_pose, list)
        assert isinstance(rotation_center, list)
        self.urdf.update_cfg(new_pose, rotation_center, self.dx_eps)

    def update_pose_quick(self, new_pose: List[dict], rotation_center: Literal['joint_0', 'front', 'direct']) -> None:
        self.urdf.update_pose_quick(new_pose, rotation_center, self.dx_eps)

    def update_pose_given_batch(self, batch_idx: int, new_pose: dict, rotation_center: Literal['joint_0', 'front', 'direct']) -> None:
        """
        new_pose:
            - joint_0: float, open angle
            - joint_1: 6D list, translation and rotation
        """
        self.urdf.update_cfg_given_batch(batch_idx, new_pose, rotation_center, self.dx_eps)

    def _get_direct_cfg(self, batch_idx: int, need_deepcopy: bool) -> dict:
        return self.urdf._get_direct_cfg(batch_idx, need_deepcopy)
    
    def get_direct_cfg(self, need_deepcopy: bool) -> List[dict]:
        """
        Return a copy of direct configuration
        """
        return [self._get_direct_cfg(b, need_deepcopy) for b in range(self.batch_size)]
    
    def get_direct_cfg_given_batch(self, batch_idx: int, need_deepcopy: bool) -> dict:
        """
        Return a copy of direct configuration
        """
        return self._get_direct_cfg(batch_idx, need_deepcopy)

    def get_init_pose(self) -> dict:
        return self.urdf.init_pose

    def get_mesh(self) -> List[trimesh.Trimesh]:
        return self.urdf.get_mesh()

    def get_limit(self, joint_name: str) -> Union[batch_urdf.Limit, None]:
        return self.urdf.get_limit(joint_name)

    def query_sdf(self, pos: ti.MatrixField, ans: ti.MatrixField, n_field: ti.Field, need_merge: bool):
        """
        query sdf and store in ans.

        Args:
            - pos: (B x N) x (3) 3D (Vector) field
            - ans: (B x N) x (SDF_SHAPE) SDF_SHAPED (Vector) field (need_merge=True)
            - ans: (B x N) x (2 x SDF_SHAPE) (Matrix) field (need_merge=False)
        """
        assert isinstance(pos, ti.MatrixField)
        assert isinstance(ans, ti.MatrixField)
        assert isinstance(n_field, ti.Field)
        assert isinstance(need_merge, bool)

        assert (pos.n, pos.m, pos.ndim) == (3, 1, 1)
        if need_merge:
            assert (ans.n, ans.m, ans.ndim) == (SDF_SHAPE, 1, 1)
        else:
            assert (ans.n, ans.m, ans.ndim) == (2, SDF_SHAPE, 2)

        self.sdf.query_sdf(pos, ans, n_field, self.urdf, need_merge)

    def initialize_front_point_data(self):
        front_params = self.urdf.get_front_parameters(
            self.scissor_cfg.front_cfg)
        if self.urdf.use_front_cache and os.path.exists(self.urdf.front_path):
            front_data = np.load(self.urdf.front_path, allow_pickle=True)
            cache_params = front_data[0]
            if cache_params != front_params:
                self.calculate_and_save_front(front_params)
                front_data = np.load(self.urdf.front_path, allow_pickle=True)
                if self.print_info:
                    print("[INFO] cache file parameters:{} doesn't match front parameters:{}. Recalculate sdf.".format(
                        cache_params, front_params))
            else:
                if self.print_info:
                    print("[INFO] successfully loaded scissor front file.")

        else:
            self.calculate_and_save_front(front_params)
            front_data = np.load(self.urdf.front_path, allow_pickle=True)
            if self.print_info:
                print("[INFO] Recalculate scissor front.")

        self.urdf.front_theta_arr = front_data[1]
        self.urdf.front_dist_arr = front_data[2]
        self.urdf.neg_front_dist_arr = -front_data[2]

        assert np.all(np.diff(self.urdf.front_theta_arr) >= 0.0)
        assert np.all(np.diff(self.urdf.neg_front_dist_arr) >= 0.0)

    def calculate_and_save_front(self, front_params: list):
        data = [front_params]
        sample_dtheta, dx_tol, sdf_tol, x_lower, x_upper, x_shift, poly_deg = front_params

        limit: batch_urdf.Limit = self.get_limit("joint_0")
        front_sample_num = int(ceil(
            (limit.upper - limit.lower) / sample_dtheta))
        theta_arr = np.linspace(
            limit.lower, limit.upper, front_sample_num)
        dist_arr = np.zeros((front_sample_num,), dtype=np.float32)

        scissor_old_direct_cfg = self.urdf._get_direct_cfg(0, True)

        self.urdf._update_cfg(0, SCISSOR_ZERO_ACTION, "joint_0", self.dx_eps)
        pos, vec, axs = self.urdf._get_cut_direction(0, self.dx_eps)
        joint_0_theta, joint_0_phi = direc_to_theta_phi(axs)

        for i in range(front_sample_num):
            theta = theta_arr[i]

            # do transform
            new_pose = copy.deepcopy(SCISSOR_ZERO_ACTION)
            new_pose["joint_0"] = theta
            new_pose["joint_1"][THETA_IDX] = joint_0_theta
            new_pose["joint_1"][PHI_IDX] = joint_0_phi
            new_pose["joint_1"][ANGLE_IDX] = - theta / 2
            self.urdf._update_cfg(0, new_pose, "joint_0", self.dx_eps)

            pos, vec, axs = self.urdf._get_cut_direction(0, self.dx_eps)
            # sample_vertical_pos = np.array(sample_vertical) * axs

            # do a bi-search
            left = x_lower
            right = x_upper

            while right - left > dx_tol:
                m = (left + right) / 2

                sample_pos = pos + vec * m
                m_sdf = self.sdf.query_single_sdf_python(0, sample_pos, self.urdf)

                if m_sdf.sum() < sdf_tol * 2:
                    left = m
                else:
                    right = m

            if m_sdf.any() == self.sdf_inf:
                print(
                    "[WARNING] In calculate front point position, get sdf_inf:{} in Bi-search".format(self.sdf_inf))
            dist_arr[i] = (left + right) / 2

        dist_arr += x_shift
        self.urdf._update_cfg(0, scissor_old_direct_cfg, "direct", self.dx_eps)
        if poly_deg >= 0:
            poly_coeff = np.polyfit(theta_arr, dist_arr, poly_deg)
            dist_fit_arr = np.polyval(poly_coeff, theta_arr)
        else:
            dist_fit_arr = dist_arr

        # force dist_fit_arr to be monotonously decreasing
        for i in range(front_sample_num - 1):
            if dist_fit_arr[i + 1] > dist_fit_arr[i]:
                dist_fit_arr[i + 1] = dist_fit_arr[i]

        data.append(theta_arr)
        data.append(dist_fit_arr)
        data.append(dist_arr)

        obj_arr = np.array(data, dtype=object)
        os.makedirs(os.path.split(self.urdf.front_path)[0], exist_ok=True)
        np.save(self.urdf.front_path, obj_arr)

    def get_scissor_frame_coordinates(self, pos: List[np.ndarray]) -> List[np.ndarray]:
        """
        Returns: np.array([dx, dy, dz])
            - dx: coordinate along cutting direction
            - dy: coordinate along rotation axis
            - dz: remaining coordinate
        """
        assert isinstance(pos, list)
        assert len(pos) == self.batch_size
        return [self._get_scissor_frame_coordinates(b, p) for b, p in enumerate(pos)]

    def _get_scissor_frame_coordinates(self, batch_idx: int, pos: np.ndarray) -> np.ndarray:
        cut_pos, cut_vec, cut_axs = self.urdf._get_cut_direction(batch_idx, self.dx_eps)
        dr = pos - cut_pos
        dx = dr.dot(cut_vec)
        dy = dr.dot(cut_axs)
        dz = np.linalg.norm(dr - dx * cut_vec - dy * cut_axs)
        return np.array([dx, dy, dz])
    
    def _get_cut_direction(self, batch_idx: int) -> List[np.ndarray]:
        return self.urdf._get_cut_direction(batch_idx, self.dx_eps)
    
    def get_cut_direction_given_batch(self, batch_idx: int) -> List[np.ndarray]:
        """
        Return:
            - pos: joint_0 current position
            - vec: normalized cut direction
            - axs: normalized joint_0 axis direction
        """
        return self._get_cut_direction(batch_idx)

    def get_cut_direction(self) -> List[List[np.ndarray]]:
        """
        Return:
            - pos: joint_0 current position
            - vec: normalized cut direction
            - axs: normalized joint_0 axis direction
        """
        return [self._get_cut_direction(b) for b in range(self.batch_size)]
    
    def get_cut_direction_quick(self) -> List[np.ndarray]:
        """
        Return:
            - pos: joint_0 current position
            - vec: normalized cut direction
            - axs: normalized joint_0 axis direction
        """
        return self.urdf._get_cut_direction_quick(self.dx_eps)

    def get_cut_front_point(self, frame_name: List[Literal['world', 'joint_0']]) -> List[np.ndarray]:
        return self.urdf.get_cut_front_point(frame_name, self.dx_eps, [None] * self.batch_size)
    
    def get_cut_front_point_quick(self, frame_name: Literal['world', 'joint_0']) -> np.ndarray:
        return self.urdf.get_cut_front_point_quick(frame_name, self.dx_eps, None)
    
    def get_cut_front_point_given_batch(self, batch_idx: int, frame_name: Literal['world', 'joint_0']) -> np.ndarray:
        return self.urdf._get_cut_front_point(batch_idx, frame_name, self.dx_eps, None)

    def compute_cut_front_point_given_theta(self, frame_name: List[Literal['world', 'joint_0']], theta: List[float]) -> List[np.ndarray]:
        return self.urdf.get_cut_front_point(frame_name, self.dx_eps, theta)
    
    def compute_cut_front_point_given_theta_given_batch(self, batch_idx: int, frame_name: Literal['world', 'joint_0'], theta: float) -> np.ndarray:
        return self.urdf._get_cut_front_point(batch_idx, frame_name, self.dx_eps, theta)

    def compute_cut_front_point_dist_given_theta(self, theta: float) -> float:
        return self.urdf.get_cut_front_dist(theta)

    def compute_theta_given_front_point_dist(self, front_point_dist: float) -> float:
        """
        compute the value of theta given distance between front point and joint_0. 
        """
        return self.urdf.compute_theta_given_front_point_dist(front_point_dist)

    def reset(self):
        self.urdf.reset(self.dx_eps)

    def get_state(self) -> List[dict]:
        return [{
            "class": "Scissor",
            "direct_cfg": direct_cfg
        } for direct_cfg in self.get_direct_cfg(True)]
    
    def set_state(self, states: List[dict]) -> None:
        assert isinstance(states, list)
        assert isinstance(states[0], dict)
        assert len(states) == self.batch_size
        self.urdf.update_cfg([copy.deepcopy(state["direct_cfg"]) for state in states],
                             ["direct"] * self.batch_size, self.dx_eps)


def default_compute_ee_pose(scissor: Scissor) -> dict:
    raise DeprecationWarning
    # linear interpolation
    d = 0.2 + 0.02 * (0.0 - scissor.get_direct_cfg(True)["joint_0"])
    mat = np.eye(4, dtype=float)
    pos, vec, axs = scissor.get_cut_direction()
    y = axs[:]
    z = vec[:]
    x = np.cross(y, z)
    mat[:3, :3] = np.column_stack([x, y, z])
    mat[:3, 3] = pos - vec * d

    # linear interpolation
    fj = -0.2 + 0.7 * (0.0 - scissor.get_direct_cfg(True)["joint_0"])
    angle, direc, point = tra.rotation_from_matrix(mat)
    theta, phi = direc_to_theta_phi(direc)
    ee_pose = [0.0] * 6
    ee_pose[:3] = pos - vec * d
    ee_pose[ANGLE_IDX] = angle
    ee_pose[THETA_IDX] = theta
    ee_pose[PHI_IDX] = phi

    ret = {
        "robot_ee": mat,
        "endeffector": {
            "finger_joint":fj,
            "base_joint":ee_pose
        }
    }
    return ret

def realworld_compute_ee_pose(scissor: Scissor) -> Dict[str, list]:
    # linear interpolation
    joint_0_arr = np.array([0.0000, 0.0132, 0.0265, 0.0398, 0.0531, 0.0664, 0.0798, 0.0932, 0.1067, 0.1202, 0.1337, 0.1473, 0.1610, 0.1747, 0.1885, 0.2023, 0.2163, 0.2303, 0.2444, 0.2586, 0.2729, 0.2874, 0.3019, 0.3166, 0.3314, 0.3463, 0.3614, 0.3767, 0.3921, 0.4078, 0.4236, 0.4396, 0.4559, 0.4724, 0.4892, 0.5062, 0.5236, 0.5413, 0.5594, 0.5778, 0.5967])
    joint_l_1_arr = np.array([0.0000, -0.0050, -0.0100, -0.0150, -0.0200, -0.0250, -0.0300, -0.0350, -0.0400, -0.0450, -0.0500, -0.0550, -0.0600, -0.0650, -0.0700, -0.0750, -0.0800, -0.0850, -0.0900, -0.0950, -0.1000, -0.1050, -0.1100, -0.1150, -0.1200, -0.1250, -0.1300, -0.1350, -0.1400, -0.1450, -0.1500, -0.1550, -0.1600, -0.1650, -0.1700, -0.1750, -0.1800, -0.1850, -0.1900, -0.1950, -0.2000])
    joint_l_4_arr = np.array([0.0000, 0.0065, 0.0131, 0.0199, 0.0268, 0.0339, 0.0411, 0.0484, 0.0559, 0.0635, 0.0713, 0.0792, 0.0872, 0.0954, 0.1038, 0.1123, 0.1210, 0.1299, 0.1389, 0.1481, 0.1574, 0.1669, 0.1767, 0.1866, 0.1967, 0.2069, 0.2174, 0.2282, 0.2391, 0.2503, 0.2617, 0.2733, 0.2853, 0.2975, 0.3100, 0.3227, 0.3359, 0.3493, 0.3632, 0.3774, 0.3921])

    ret = {"robot_ee": [], "endeffector": []}
    for b in range(scissor.batch_size):
        mat = np.eye(4, dtype=np.float64)
        pos, vec, axs = scissor._get_cut_direction(b)
        # Theoretically, y.T @ z = 0, but because of floating number error, it may change
        y = -axs[:].astype(np.float64)
        y /= np.linalg.norm(y)
        z = vec[:].astype(np.float64)
        z -= z.dot(y) * y
        z /= np.linalg.norm(z)
        x = np.cross(y, z)
        mat[:3, :3] = np.column_stack([x, y, z])
        mat[:3, 3] = pos
        
        joint_0 = scissor._get_direct_cfg(b, True)["joint_0"]
        joint_l_1 = np.interp(joint_0, joint_0_arr, joint_l_1_arr)
        joint_l_4 = np.interp(joint_0, joint_0_arr, joint_l_4_arr)

        angle, direc, point = tra.rotation_from_matrix(mat)
        theta, phi = direc_to_theta_phi(direc)
        ee_pose = [0.0] * 6
        ee_pose[:3] = pos - vec * 0.176
        ee_pose[ANGLE_IDX] = angle
        ee_pose[THETA_IDX] = theta
        ee_pose[PHI_IDX] = phi

        ret["robot_ee"].append(mat) # to calculate IK
        ret["endeffector"].append({
            "joint_l_1":joint_l_1,
            "joint_l_4":joint_l_4,
            "base_joint":ee_pose
        })
    return ret


def compute_ee_pose_clipper(scissor: Scissor) -> Dict[str, list]:
    # linear interpolation
    ret = {"robot_ee": [], "endeffector": []}
    for b in range(scissor.batch_size):
        mat = np.eye(4, dtype=np.float64)
        pos, vec, axs = scissor._get_cut_direction(b)
        # Theoretically, y.T @ z = 0, but because of floating number error, it may change
        y = -axs[:].astype(np.float64)
        y /= np.linalg.norm(y)
        z = vec[:].astype(np.float64)
        z -= z.dot(y) * y
        z /= np.linalg.norm(z)
        x = np.cross(y, z)
        mat[:3, :3] = np.column_stack([x, y, z])
        mat[:3, 3] = pos

        angle, direc, point = tra.rotation_from_matrix(mat)
        theta, phi = direc_to_theta_phi(direc)
        ee_pose = [0.0] * 6
        ee_pose[:3] = pos - vec * .131
        ee_pose[ANGLE_IDX] = angle
        ee_pose[THETA_IDX] = theta
        ee_pose[PHI_IDX] = phi

        ret["robot_ee"].append(mat) # to calculate IK
        ret["endeffector"].append({
            "base_joint":ee_pose
        })
    return ret