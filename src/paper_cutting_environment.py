import taichi as ti

import logging
import os
import sys
import shutil
import numpy as np
import math
import datetime
from typing import List, Dict, Tuple, Union, Callable
import copy
import time

import tqdm
from omegaconf import DictConfig, OmegaConf
import trimesh
import trimesh.transformations as tra

from src.cut_simulation_environment import CutSimulationEnvironment
from src.scissor import *
from src.maths import *
from src.utils import *


PAPER_CUTTING_ENVIRONMENT_VERSION = "3.0"


class PaperCuttingEnvironment:
    def __init__(self, cfg: DictConfig) -> None:
        self.batch_size: int = cfg.env.batch_size
        assert isinstance(self.batch_size, int)
        assert self.batch_size >= 1

        self._cfg = cfg
        self.log = logging.getLogger(__name__)
        self.log.info(" ".join(sys.argv))
        self.log_detail = self._cfg.output.log.detail
        self.log_time = self._cfg.output.log.time
        assert cfg.config_version == PAPER_CUTTING_ENVIRONMENT_VERSION, \
            "config_version:{} in the yaml does not match current version:{}, please update your config file. ".format(
                cfg.config_version, PAPER_CUTTING_ENVIRONMENT_VERSION)

        # initialize taichi
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.setup.cuda)
        except Exception as e:
            print(f"cannot set 'CUDA_VISIBLE_DEVICES' because of exception {e}")

        if cfg.setup.arch == 'cuda':
            arch = ti.cuda
        else:
            arch = ti.cpu
        ti.init(arch=arch, debug=cfg.setup.debug,
                kernel_profiler=cfg.setup.kernel_profiler, offline_cache=cfg.setup.offline_cache, random_seed=0, device_memory_GB=cfg.setup.device_memory_GB, default_gpu_block_dim=cfg.setup.default_gpu_block_dim, fast_math=False, advanced_optimization=cfg.setup.advanced_optimization)

        # output file path setup
        self._series_prefix = f"{os.getcwd()}/"

        self._output_models: dict = OmegaConf.to_container(cfg.output.models)
        self._is_write_uv = cfg.output.is_write_uv
        self._uv_prefix = cfg.output.uv_prefix

        self._save_misc_state = cfg.output.save_misc_state

        # create sim env
        self._sim_env = CutSimulationEnvironment(cfg, self.log)
        self._open_close_max_velocity = cfg.env.cut_limit.open_close_max_velocity
        self._translation_max_velocity = cfg.env.cut_limit.translation_max_velocity
        self._rotation_max_velocity = cfg.env.cut_limit.rotation_max_velocity

        self._constraints_queue = [Queue() for b in range(self.batch_size)]
        self._last_constraints = [{} for b in range(self.batch_size)]

        # scissors info
        self._scissors_action_queue = [Queue() for b in range(self.batch_size)]

        # time tracking
        self._now_time = 0.0
        self._end_time = 0.0

        self._now_substep = 0
        self._now_frame_i = 0

        self._dt = float(self._sim_env.dt)
        self._frame_dt = float(self._sim_env.frame_dt)
        assert self._dt <= self._frame_dt

    def _get_scissor_action_for_next_substep(self, scissors_old_pose: List[dict]) -> Tuple[List[dict], List[dict]]:
        """
        Get scissor action for next substep from action queues.

        Return:
            - new_pose
            - velocity
        """
        new_pose = []
        velocity = []
        for batch_idx in range(self.batch_size):
            q = self._scissors_action_queue[batch_idx]
            scissors = self._sim_env.get_scissors()

            if q.empty():
                dpose = copy.deepcopy(SCISSOR_ZERO_ACTION)
                dpose["rotation_center"] = "joint_0"
            else:
                dpose = q.get()

            assert dpose["rotation_center"] in ["joint_0", "front"]
            tmp_new_pose = copy.deepcopy(SCISSOR_ZERO_ACTION)
            tmp_velocity = copy.deepcopy(SCISSOR_ZERO_ACTION)

            # open and close
            limit: batch_urdf.Limit = scissors.get_limit("joint_0")
            tmp_new_pose["joint_0"] = scissors_old_pose[batch_idx]["joint_0"] + \
                dpose["joint_0"]
            tmp_new_pose["joint_0"] = max(limit.lower,
                                          min(limit.upper, tmp_new_pose["joint_0"]))
            tmp_velocity["joint_0"] = (tmp_new_pose["joint_0"] -
                                       scissors_old_pose[batch_idx]["joint_0"]) / self._dt

            # rotation
            old_direc = theta_phi_to_direc(
                scissors_old_pose[batch_idx]["joint_1"][THETA_IDX], scissors_old_pose[batch_idx]["joint_1"][PHI_IDX])
            old_angle = scissors_old_pose[batch_idx]["joint_1"][ANGLE_IDX]
            old_mat = tra.rotation_matrix(old_angle, old_direc)

            fwd_direc = theta_phi_to_direc(
                dpose["joint_1"][THETA_IDX], dpose["joint_1"][PHI_IDX])
            fwd_angle = dpose["joint_1"][ANGLE_IDX]
            fwd_mat = tra.rotation_matrix(fwd_angle, fwd_direc)

            new_mat = fwd_mat @ old_mat

            angle, direc, point = tra.rotation_from_matrix(new_mat)
            theta, phi = direc_to_theta_phi(direc)

            tmp_new_pose["joint_1"][ANGLE_IDX] = angle
            tmp_new_pose["joint_1"][THETA_IDX] = theta
            tmp_new_pose["joint_1"][PHI_IDX] = phi

            tmp_velocity["joint_1"][ANGLE_IDX] = dpose["joint_1"][ANGLE_IDX] / self._dt
            tmp_velocity["joint_1"][THETA_IDX] = dpose["joint_1"][THETA_IDX]
            tmp_velocity["joint_1"][PHI_IDX] = dpose["joint_1"][PHI_IDX]

            # translation
            for i in range(3):
                tmp_new_pose["joint_1"][i] = scissors_old_pose[batch_idx]["joint_1"][i] + \
                    dpose["joint_1"][i]
                tmp_velocity["joint_1"][i] = dpose["joint_1"][i] / self._dt

            if dpose["rotation_center"] == "front":
                dist_old = scissors.get_cut_front_point_given_batch(batch_idx, "joint_0")
                dist_new = scissors.compute_cut_front_point_given_theta_given_batch(batch_idx, 
                    "joint_0", tmp_new_pose["joint_0"])

                delta_r = -(dist_new - dist_old)
                # actually dist_old == dist_new (Move) or no rotation (open and close),
                # so the formula below is always precise.
                delta_r -= ((fwd_mat - np.eye(4, dtype=float)) @
                            vector_to_homo(dist_new))[:3]

                for i in range(3):
                    tmp_new_pose["joint_1"][i] += delta_r[i]
                    tmp_velocity["joint_1"][i] += delta_r[i] / self._dt

            new_pose.append(tmp_new_pose)
            velocity.append(tmp_velocity)

        return (new_pose, velocity)

    def append_scissors_action(self, action_list: List[dict]) -> None:
        """
        Append scissors action.

        Example: An environment with batch_size=5

            action_list = [
                {"Action":"None"}, \n
                {"Action":"Open", "Angle":0.1, "Time":0.1}, \n
                {"Action":"Close", "Angle":0.1, "Time":0.1}, \n
                {"Action":"Move", "Displacement": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], "Time":0.1}, \n
                {"Action":"Stay", "Time":0.1}
            ]

        Arguments in joint_1's 6D list: [x, y, z, angle, theta, phi]
            - x, y, z: translation
            - theta, phi: rotation axis direction in spherical coordinate (around front point)
            - angle: rotation angle
        """
        assert isinstance(action_list, list), "action_list should be a list, but get {}.".format(
            type(action_list))
        assert len(action_list) == self.batch_size, "len(action_list) should be {}, but get {}".format(
            self.batch_size, len(action_list))

        for batch_idx in range(self.batch_size):
            action = action_list[batch_idx]
            assert isinstance(action, dict), "each element in action_list should be dict, but get {}".format(
                type(action))
            assert "Action" in action.keys(
            ), "each element in action_list should contain a key called 'Action'."

            action_type = action["Action"]

            if action_type == "None":
                pass

            elif action_type == "Open" or action_type == "Close":
                assert action["Angle"] > 0.0

                angle = action["Angle"]
                time = action["Time"]

                n_step = math.ceil(time / self._dt)
                dphi = angle / n_step

                # apply limit
                if dphi / self._dt > self._open_close_max_velocity:
                    if self.log_detail:
                        self.log.info("Action:{}\ndphi/dt={:.4f} > {}, use {}".format(
                            action, dphi / self._dt, self._open_close_max_velocity, self._open_close_max_velocity))
                    dphi = self._dt * self._open_close_max_velocity

                # change sign
                dphi *= (1.0 if action_type == "Open" else -1.0)

                scissors = self._sim_env.get_scissors()
                pos, vec, axs = scissors.get_cut_direction_given_batch(batch_idx)
                theta, phi = direc_to_theta_phi(axs)

                joint_1_list = [0.0] * 6
                joint_1_list[ANGLE_IDX] = -dphi / 2
                joint_1_list[THETA_IDX] = theta
                joint_1_list[PHI_IDX] = phi

                for i in range(n_step):
                    self._scissors_action_queue[batch_idx].put({
                        "rotation_center": "joint_0",
                        "joint_0": dphi,
                        "joint_1": copy.deepcopy(joint_1_list)
                    })

            elif action_type == "Move":
                action_disp = action["Displacement"]
                time = action["Time"]

                n_step = math.ceil(time / self._dt)

                daction = [0.0] * 6
                for i in range(6):
                    # theta and phi should not divided by n_step
                    if i in [THETA_IDX, PHI_IDX]:
                        daction[i] = action_disp[i]

                    # rotation limit
                    elif i == ANGLE_IDX:
                        drotate = action_disp[i] / n_step
                        if abs(drotate / self._dt) > self._rotation_max_velocity:
                            if self.log_detail:
                                self.log.info("Action:{}\ndrotate/dt={:.4f} > {}, use {}".format(
                                    action, drotate / self._dt, self._rotation_max_velocity, self._rotation_max_velocity))
                            if drotate / self._dt > 0.0:
                                drotate = +self._rotation_max_velocity * self._dt
                            else:
                                drotate = -self._rotation_max_velocity * self._dt
                        daction[i] = drotate

                    # translation limit
                    else:
                        dx = action_disp[i] / n_step
                        if abs(dx / self._dt) > self._translation_max_velocity:
                            if self.log_detail:
                                self.log.info("Action:{}\ndx/dt={:.4f} > {}, use {}".format(
                                    action, dx / self._dt, self._translation_max_velocity, self._translation_max_velocity))
                            if dx / self._dt > 0.0:
                                dx = +self._translation_max_velocity * self._dt
                            else:
                                dx = -self._translation_max_velocity * self._dt
                        daction[i] = dx

                for i in range(n_step):
                    self._scissors_action_queue[batch_idx].put({
                        "rotation_center": "front",
                        "joint_0": 0.0,
                        "joint_1": copy.deepcopy(daction)
                    })

            elif action_type == "Stay":
                time = action["Time"]
                n_step = math.ceil(time / self._dt)

                tmp_action = copy.deepcopy(SCISSOR_ZERO_ACTION)
                tmp_action["rotation_center"] = "joint_0"
                for i in range(n_step):
                    self._scissors_action_queue[batch_idx].put(
                        copy.deepcopy(tmp_action))

            else:
                raise NotImplementedError(action_type)

        if self.log_detail:
            self.log.info("Append scissors action:{}".format(action_list))

    def _get_cloth_constraint(self, batch_idx: int) -> Dict[int, np.ndarray]:
        if not self._constraints_queue[batch_idx].empty():
            self._last_constraints[batch_idx] = self._constraints_queue[batch_idx].get()
        return self._last_constraints[batch_idx]

    def append_constraints(self, batch_idx: int, new_constraints: Dict[int, np.ndarray], constrain_time: float, old_constraints: Dict[int, np.ndarray] = {}, n_step:int=None) -> None:
        """
        Append some new constraints on cloth to a constraint queue. 

        If `n_step` is `None`, then `n_step=max(1, ceil(time / dt))`

        --- 

        Example1:
            - new_constraints = {1: np.array([0.0, 0.0, 0.0]),
                                 2: np.array([0.3, 0.0, 0.0])}
            - constrain_time = 1.5

        Explanation1:
            - Cost 1.5 seconds move vertex 1 from current position (see notes) to [0.0, 0.0, 0.0]
            - Cost 1.5 seconds move vertex 2 from current position (see notes) to [0.3, 0.0, 0.0]
            - 1.5 seconds later, vertex 1 and vertex 2 will stay at [0.0, 0.0, 0.0] and [0.3, 0.0, 0.0] if there is no more actions in the constraint queue. 

        ---

        Example2:
            - new_constraints = {}
            - constrain_time = 1.5

        Explanation2:
            - No constraint on all vertices. 

        ---

        Example3:
            - new_constraints = {1: np.array([0.0, 0.0, 0.0]),
                                 2: np.array([0.3, 0.0, 0.0])}
            - constrain_time = 0.0

        Explanation3:
            - Move vertex 1 to [0.0, 0.0, 0.0] at once
            - Move vertex 2 to [0.3, 0.0, 0.0] at once
            - As long as the queue is not empty, cloth will use previous constraints in the queue.

        ---

        Example4:
            - new_constraints = {1: np.array([0.0, 0.0, 0.0]),
                                 2: np.array([0.3, 0.0, 0.0])}
            - old_constraints = {1: np.array([0.0, 0.0, 0.1]),
                                 2: np.array([0.3, 0.0, 0.1])}
            - constrain_time = 1.5

        Explanation4:
            - Cost 1.5 seconds move vertex 1 from [0.0, 0.0, 0.1] to [0.0, 0.0, 0.0]
            - Cost 1.5 seconds move vertex 2 from [0.3, 0.0, 0.1] to [0.3, 0.0, 0.0]
            - the vertex index in 2 dicts must match, otherwise it will use current position (see notes).

        ---

        Notes:
            - 'Current' in 'current position' means when this function is called. 
        """
        if n_step is None:
            n_step = max(1, math.ceil(constrain_time / self._dt))
        assert isinstance(n_step, int) and n_step >= 1
        
        old_constraints = copy.deepcopy(old_constraints)

        nv = self._sim_env.cloth.mesh.n_vertices[batch_idx]
        for vid, new_pos in new_constraints.items():
            assert 0 <= vid and vid < nv, f"vertex index:{vid} output of range:[{0}, {nv})"
            if not (vid in old_constraints.keys()):
                old_constraints[vid] = self._sim_env.cloth.vertices_pos[batch_idx, vid].to_numpy()

        for i in range(n_step):
            actions = {}
            for vid in new_constraints.keys():
                old_pos = old_constraints[vid]
                new_pos = new_constraints[vid]
                actions[vid] = old_pos + (i + 1) * (new_pos - old_pos) / n_step
            self._constraints_queue[batch_idx].put(copy.deepcopy(actions))

    def set_scissors_pose(self, scissors_new_pose: List[dict], compute_ee_pose: Callable = None) -> None:
        """
        Set scissors pose.

        Each new pose is dict:
            - joint_0: float, open angle
            - joint_1: 6D list, translation and rotation

        Arguments in joint_1's 6D list: [x, y, z, angle, theta, phi]
            - x, y, z: translation
            - theta, phi: rotation axis direction in spherical coordinate (around joint_0)
            - angle: rotation angle
        """
        _l, _u = self.get_scissor_open_range()
        assert False not in [_l <= scissors_new_pose[batch_idx]["joint_0"] and
                             scissors_new_pose[batch_idx]["joint_0"] <= _u for batch_idx in range(self.batch_size)], f"[ERROR] In set_scissors_pose, joint_0 of scissors_new_pose{scissors_new_pose} out of range. "
        assert not self._cfg.robot.use_robot or callable(compute_ee_pose), \
            "Please pass a python function (or any callable object) as a parameter: compute_ee_pose=func, if you want to use robot arm. "

        self._sim_env.update_scissors_old_pose(scissors_new_pose)
        self._sim_env.update_scissors_pose(scissors_new_pose, compute_ee_pose)
        if self.log_detail:
            self.log.info("Set scissors pose:{}".format(scissors_new_pose))

    def get_scissors_pose(self, batch_idx: Union[None, int] = None) -> Union[List[dict], dict]:
        """
        Return a deep copy of scissors pose.

        Each pose is a dict:
            - joint_0: float, open angle
            - joint_1: 6D list, translation and rotation

        Arguments in joint_1's 6D list: [x, y, z, angle, theta, phi]
            - x, y, z: translation
            - theta, phi: rotation axis direction in spherical coordinate (around joint_0)
            - angle: rotation angle
        """
        return self._sim_env.get_scissors_pose(batch_idx)

    def get_scissors_mesh(self) -> List[trimesh.Trimesh]:
        """
        Return a list of all scissors mesh. 
        """
        return self._sim_env.get_scissors_mesh()

    def get_scissors_cut_direction(self, batch_idx: Union[None, int] = None) -> Union[List[List[np.ndarray]], List[np.ndarray]]:
        """
        Return all scissors cut direction. 

        Each item in the return list is a list consist of 3 arrays:
            - pos: joint_0 current position
            - vec: normalized cut direction
            - axs: normalized joint_0 axis direction
        """
        if batch_idx is None:
            return self._sim_env.get_scissors().get_cut_direction()
        elif isinstance(batch_idx, int):
            return self._sim_env.get_scissors().get_cut_direction_given_batch(batch_idx)
        else:
            raise NotImplementedError(type(batch_idx))

    def _get_scissors_front_point(self, frame_name: List[Literal["world", "joint_0"]]) -> List[np.ndarray]:
        """
        Return the position of the intersection point between two blades of all scissors. 
        """
        assert isinstance(frame_name, list), type(frame_name)
        assert len(frame_name) == self.batch_size, len(frame_name)
        return self._sim_env.get_scissors().get_cut_front_point(frame_name)
    
    def get_scissor_open_range(self) -> Tuple[float, float]:
        """
        Return the open range of a single pair of scissors. 

        Return: (l, u)
            - l: lower bound
            - u: upper bound
        """
        scissors = self._sim_env.get_scissors()
        limit: batch_urdf.Limit = scissors.get_limit("joint_0")
        return (limit.lower, limit.upper)

    def compute_scissors_front_point_given_theta(self, 
                                                 frame_name: Union[List[Literal["world", "joint_0"]], Literal["world", "joint_0"]], 
                                                 theta: Union[List[float], float],
                                                 batch_idx: Union[None, int] = None) -> Union[List[np.ndarray], np.ndarray]:
        """
        Return the position of the intersection point between two blades of all scissors given theta. 
        """
        if batch_idx is None:
            return self._sim_env.get_scissors().compute_cut_front_point_given_theta(frame_name, theta)
        elif isinstance(batch_idx, int):
            return self._sim_env.get_scissors().compute_cut_front_point_given_theta_given_batch(batch_idx, frame_name, theta)
        else:
            raise NotImplementedError

    def compute_front_point_dist_given_theta(self, theta: float) -> float:
        """
        Compute the value of distance of front point movement given theta. 
        """
        return self._sim_env.get_scissors().compute_cut_front_point_dist_given_theta(theta)

    def compute_theta_given_front_point_dist(self, front_point_dist: float) -> float:
        """
        Compute the value of theta given distance between front point and joint_0. 
        """
        return self._sim_env.get_scissors().compute_theta_given_front_point_dist(front_point_dist)

    '''def get_split_edge_rest_position(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Return a deepcopy of all splitted edges in history.

        Each item in the return list is a tuple containing 2 arrays:
            - v1pos: vertex 1 rest position
            - v2pos: vertex 2 rest position
            - time: float
        """
        return copy.deepcopy(self._sim_env.cutting.split_edge_origin_coordinates)'''

    def get_current_cloth_state(self, batch_idx: int) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return current cloth's state.

        Return:
            - number of vertices: int
            - number of faces: int
            - vertex current position: np.ndarray, shape=(nv, 3)
            - vertex rest position: np.ndarray, shape=(nv, 3)
            - face to vert: np.ndarray, shape=(nf, 3)
        """
        nv = self._sim_env.cloth.mesh.n_vertices[batch_idx]
        nf = self._sim_env.cloth.mesh.n_faces[batch_idx]
        vert_curr_pos = self._sim_env.cloth.vertices_pos.to_numpy()[batch_idx, :nv, :]
        vert_rest_pos = self._sim_env.cloth.vertices_rest_pos.to_numpy()[batch_idx, :nv, :]
        face_to_vert = self._sim_env.cloth.mesh.faces_vid.to_numpy()[batch_idx, :nf, :]
        assert vert_curr_pos.shape == (nv, 3)
        assert vert_rest_pos.shape == (nv, 3)
        assert face_to_vert.shape == (nf, 3)
        return (nv, nf, vert_curr_pos, vert_rest_pos, face_to_vert)
    
    def get_current_pos_given_rest(self, batch_idx: int, rest_pos: np.ndarray):
        """
        Calculate current position of a point given rest position.

        Args:
            - rest_pos: np.ndarray, shape=(3, )

        Return:
            - curr_pos: np.ndarray, shape=(3, )
        """
        return self._sim_env.get_current_pos_given_rest_kernel(batch_idx, ti.Vector(rest_pos, ti.f32)).to_numpy()

    def get_cloth_mesh(self) -> List[trimesh.Trimesh]:
        """
        Return cloth mesh. 
        """
        return self._sim_env.get_cloth_mesh()
    
    def compute_vertex_uv(self, batch_idx: int, func: Callable) -> np.ndarray:
        """
        Compute uv coordinate on each vertex.

        Args:
            - func: function
                - A user-defined mapping from rest position to uv coordinate. 
                - y = func(x) 
                    - x: np.ndarray(shape=(?, 3), float)
                    - y: np.ndarray(shape=(?, 2), float)
                - Example:
                    def func(x):
                        return x[:, 0:2] / 0.3

        Return:
            - uv: np.ndarray, shape=(nv, 3)
        """
        nv = self._sim_env.cloth.mesh.n_vertices[batch_idx]
        vert_rest_pos = self._sim_env.cloth.vertices_rest_pos.to_numpy()[batch_idx, :nv, :]
        assert vert_rest_pos.shape == (nv, 3)

        uv = func(vert_rest_pos)
        assert type(uv) == np.ndarray and uv.shape == (nv, 2)
        return uv
    
    '''def get_front_point_projection(self) -> List[List[Tuple[bool, float, np.ndarray]]]:
        """
        Return a deepcopy of all front point's projection in history.

        First list is all substep, second list is all scissors (usually 1).

        Each item in the second list is a tuple containing 3 elements:
            - is_cut_state: bool, whether this pair of scissors is in cut state.
            - time: float
            - proj: 7D vector, 
                - proj[0]: Minimal distance from front point to a point on cloth manifold. 
                - proj[1:4]: Rest position of the projection point. 
                - proj[4:7]: Current position of the projection point. 
        """
        return copy.deepcopy(self._sim_env.cutting.front_point_projection)'''

    '''def get_current_scissors_model_penetration(self) -> Tuple[int, np.ndarray]:
        """
        Return current scissors' model penetration information.

        Return:
            - number of model penetration: int
            - penetration information: np.ndarray, shape=(nmp, 3+1+1)
                - x, y, z: penetration position in world frame
                - sdf: (negative) signed distance function
                - volume: penetration volume
        """
        nmp = self._sim_env.collision.model_penetration_cnt[None]
        mp = self._sim_env.collision.model_penetration.to_numpy()[:nmp, :]
        assert mp.shape == (nmp, 5)
        return (nmp, mp)'''
    
    def get_robot_base_cfg(self) -> np.ndarray:
        """Get a deepcopy of robot base configuration. Return a 4x4 matrix."""
        return self._sim_env.robot.get_base_transform_matrix()
    
    def set_robot_base_cfg(self, mat: np.ndarray):
        """Set robot base configuration. Argument `mat` is a 4x4 matrix."""
        return self._sim_env.robot.set_base_transform_matrix(mat)
    
    def _export(self, uv_func: Callable=None, frame_i: int=None, prefix=""):
        if frame_i is None:
            frame_i = self._now_frame_i

        for k, v in self._output_models.items():
            self._sim_env.export_all_ply(
                frame_i=frame_i, 
                series_prefix_list=[self._series_prefix + f"batch_{str(batch_idx).zfill(len(str(self.batch_size - 1)))}/" + k + prefix for batch_idx in range(self.batch_size)],
                export_cloth=v["cloth"],
                export_scissor=v["scissor"],
                export_robot=v["robot"],
                export_endeffector=v["endeffector"]
            )

        if self._is_write_uv:
            for batch_idx in range(self.batch_size):
                self._sim_env.export_numpy_arr(numpy_arr=self.compute_vertex_uv(
                    batch_idx, uv_func), frame_i=frame_i, series_prefix=self._series_prefix + f"batch_{str(batch_idx).zfill(len(str(self.batch_size - 1)))}/" + prefix + self._uv_prefix + "_vertuv")

    def get_all_mesh(self) -> List[trimesh.Trimesh]:
        """
        Please call set_state(state) first
        """
        meshes_batch = [[] for _ in range(self.batch_size)]
        for batch_idx, mesh in enumerate(self.get_cloth_mesh()):
            meshes_batch[batch_idx].append(mesh)
        for batch_idx, mesh in enumerate(self.get_scissors_mesh()):
            meshes_batch[batch_idx].append(mesh)
        if self._sim_env.use_robot:
            for batch_idx, mesh in enumerate(self._sim_env.get_robot_mesh()):
                meshes_batch[batch_idx].append(mesh)
        if self._sim_env.use_endeffector:
            for batch_idx, mesh in enumerate(self._sim_env.get_endeffector_mesh()):
                meshes_batch[batch_idx].append(mesh)
        return [trimesh.util.concatenate(meshes) for meshes in meshes_batch]

    def simulate(self, simulation_time: float, save_state_freq=-1, compute_ee_pose: Callable = None,
                  uv_func: Callable = None, print_progress = True) -> List[dict]:
        """
        Simulate for a given time. Return a list of states.

        Args:
            - simulation_time: float,
            - save_state_freq: int, save state frequency, if -1 then no state will be saved.
            - compute_ee_pose: callable, how to calculate robot end effector pose
            - uv_func: callable, how to calculate uv according to rest position

        Return:
            - states: list, saved states
        """
        assert simulation_time > 0.0
        assert not self._is_write_uv or callable(uv_func), \
            "Please pass a python function (or any callable object) as a parameter: uv_func=func, if you want to export uv."
        assert not ((self._sim_env.use_robot or self._sim_env.use_endeffector) and 
                    not callable(compute_ee_pose)), "compute_ee_pose is not callable."

        self._end_time += simulation_time

        end_substep = int(self._end_time / self._dt)
        start_substep = int(self._now_substep)
        states = []

        # while self._dt * self._now_substep < self._end_time:
        if print_progress:
            print("Current time in simulation: [{:.3f} -> {:.3f}], Current time in real world: {}. ".format(
            self._now_time, self._end_time, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            substeps = tqdm.tqdm(range(start_substep, end_substep))
        else:
            substeps = range(start_substep, end_substep)
        for _ in substeps:
            self._sim_env.clock.start_clock("inner_loop")
            self._sim_env.clock.start_clock("get_action")
            scissors_new_pose, scissors_velocity = \
                self._get_scissor_action_for_next_substep(
                    self._sim_env.get_scissors_pose())
            constraints = [self._get_cloth_constraint(batch_idx) for batch_idx in range(self.batch_size)]
            self._sim_env.clock.end_clock("get_action")

            self._sim_env.substep(
                scissors_new_pose, scissors_velocity, compute_ee_pose, constraints, self._now_time)

            self._sim_env.clock.start_clock("get_output")

            if save_state_freq != -1 and (self._now_substep + 1) % save_state_freq == 0:
                states.append(self.get_state())

            # we do things below once every self._frame_dt
            if self._now_substep * self._dt < (self._now_frame_i + 1) * self._frame_dt and \
                    (self._now_substep + 1) * self._dt >= (self._now_frame_i + 1) * self._frame_dt:

                self._export(uv_func=uv_func)

                if self.log_time:
                    self.log.info("\n" + self._sim_env.clock.get_info_str(drop=(5, 5)))

                if self._cfg.output.print_info:
                    if self._now_frame_i == self._cfg.output.print_kernel_info_frame and self._cfg.setup.kernel_profiler:
                        ti.profiler.print_kernel_profiler_info()

                self._now_frame_i += 1

            self._sim_env.clock.end_clock("get_output")

            self._now_substep += 1
            self._now_time = self._dt * self._now_substep
            self._sim_env.clock.end_clock("inner_loop")

        return states
        
    def replay(self, states: List[List[dict]], replay_frequency=1, uv_func: Callable=None, prefix="_replay"):
        """
        Replay states in `states`. 
        """
        replay_idx = 0
        for replay_idx, state in tqdm.tqdm(enumerate(states)):
            if (replay_idx + 1) % replay_frequency == 0:
                self.set_state(state)
                self._export(uv_func=uv_func, frame_i=replay_idx // replay_frequency, prefix=prefix)

    def reset(self) -> None:
        """
        - Reset all configuration to initial state. (Keep time unchanged.)
        - Clear all action.
        """
        self._sim_env.reset()
        self._constraints_queue = [Queue() for b in range(self.batch_size)]
        self._last_constraints = [{} for b in range(self.batch_size)]
        self._scissors_action_queue = [Queue() for b in range(self.batch_size)]

    def get_state(self) -> List[dict]:
        """
        A large dict stores all necessary information during simulation. 
        """
        self._sim_env.clock.start_clock("sim_env_state")
        sim_env_state = self._sim_env.get_state()
        self._sim_env.clock.end_clock("sim_env_state")

        self._sim_env.clock.start_clock("misc_state")
        scissors_action_queue_state = copy.deepcopy(self._scissors_action_queue) if self._save_misc_state else [Queue() for b in range(self.batch_size)]
        constraints_queue_state = copy.deepcopy(self._constraints_queue) if self._save_misc_state else [{} for b in range(self.batch_size)]
        last_constraints_state = copy.deepcopy(self._last_constraints) if self._save_misc_state else [Queue() for b in range(self.batch_size)]
        scissor_front_point_state = copy.deepcopy(self._get_scissors_front_point(["world"] * self.batch_size))
        self._sim_env.clock.end_clock("misc_state")

        return [{
            "_sim_env": sim_env,
            "_scissors_action_queue": scissors_action_queue,
            "_constraints_queue": constraints_queue,
            "_last_constraints": last_constraints,
            "scissor_front_point": scissor_front_point,
        } for sim_env, scissors_action_queue, constraints_queue, last_constraints, scissor_front_point
        in zip(sim_env_state, scissors_action_queue_state, constraints_queue_state, last_constraints_state, scissor_front_point_state)]
    
    def set_state(self, states: List[dict]) -> None:
        assert isinstance(states, list)
        assert isinstance(states[0], dict)
        assert len(states) == self.batch_size

        self._sim_env.set_state([state["_sim_env"] for state in states])

        for batch_idx in range(self.batch_size):
            self._scissors_action_queue[batch_idx] = copy.deepcopy(states[batch_idx]["_scissors_action_queue"])
            self._constraints_queue[batch_idx] = copy.deepcopy(states[batch_idx]["_constraints_queue"])
            self._last_constraints[batch_idx] = copy.deepcopy(states[batch_idx]["_last_constraints"])
