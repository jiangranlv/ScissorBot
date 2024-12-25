import taichi as ti

import os
from typing import Tuple, Union, List
import copy

import numpy as np

import trimesh
import trimesh.transformations as tra

import pinocchio
import yourdfpy
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from src.cbase import *


def set_yrdf_joint_origin_cfg(yrdf: yourdfpy.URDF, joint: str, mat: np.ndarray):
    assert joint in yrdf.joint_map.keys(), f"joint {joint} is not in urdf."
    assert mat.shape == (4, 4), f"mat.shape should be (4, 4), but got {mat.shape}"
    # modify origin configuration
    j = yrdf.joint_map[joint]
    j.origin = mat.copy()

    # re-perform a forward kinematics
    matrix, q = yrdf._forward_kinematics_joint(j)
    if yrdf._scene is not None:
        yrdf._scene.graph.update(
            frame_from=j.parent, frame_to=j.child, matrix=matrix
        )
    if yrdf._scene_collision is not None:
        yrdf._scene_collision.graph.update(
            frame_from=j.parent, frame_to=j.child, matrix=matrix
        )

def get_yrdf_joint_origin_cfg(yrdf: yourdfpy.URDF, joint: str) -> np.ndarray:
    assert joint in yrdf.joint_map.keys(), f"joint {joint} is not in urdf."
    j = yrdf.joint_map[joint]
    return j.origin.copy()

def set_pin_jointplacement_rotation_translation(model, joint: str, rotation: np.ndarray, translation: np.ndarray):
    assert rotation.shape == (3, 3) and translation.shape == (3, )
    frameid = model.getFrameId(joint)
    jointid = model.frames[frameid].parent
    model.jointPlacements[jointid].rotation = rotation.copy()
    model.jointPlacements[jointid].translation = translation.copy()

def get_pin_jointplacement_rotation_translation(model, joint: str) -> Tuple[np.ndarray, np.ndarray]:
    frameid = model.getFrameId(joint)
    jointid = model.frames[frameid].parent
    return model.jointPlacements[jointid].rotation.copy(), \
        model.jointPlacements[jointid].translation.copy()

def in_actuated_joint_limit_range(yrdf: yourdfpy.URDF, new_pose: Union[List, np.ndarray]) -> Tuple[bool, List[Union[None, Tuple[int, int]]]]:
    assert yrdf.num_actuated_joints == len(new_pose)
    in_limit_range = True
    all_limits = []
    for joint, value in zip(yrdf.actuated_joints, new_pose):
        joint: yourdfpy.Joint
        all_limits.append(joint.limit)
        if joint.limit is not None:
            if value < joint.limit.lower or value > joint.limit.upper:
                in_limit_range = False
    return in_limit_range, all_limits

@ti.data_oriented
class RigidBody(TiObject):
    def __init__(self) -> None:
        pass


@ti.data_oriented
class Robot(TiObject):
    def __init__(self, robot_cfg: DictConfig) -> None:
        raise NotImplementedError
        self.robot_cfg = robot_cfg
        self.ik_cfg = robot_cfg.inverse_kinematics

        self.directory: str = robot_cfg.directory
        self.urdf_file: str = robot_cfg.urdf_file
        self.urdf_path: str = to_absolute_path(
            os.path.join(self.directory, self.urdf_file))

        self.init_pose = OmegaConf.to_container(robot_cfg.init_pose)
        self.default_check_limit = robot_cfg.check_limit
        self.yrdf = yourdfpy.URDF.load(self.urdf_path)
        self.update_pose(self.init_pose)

        self.model = pinocchio.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        self.joint_names = list(self.model.names)
        self.base_joint = self.robot_cfg.base_joint
        self.first_joint = self.robot_cfg.first_joint
        
        self.base_current_matrix = get_yrdf_joint_origin_cfg(self.yrdf, self.base_joint)
        self.base_origin_matrix = get_yrdf_joint_origin_cfg(self.yrdf, self.base_joint)
        self.first_origin_rotation, self.first_origin_translation = \
            get_pin_jointplacement_rotation_translation(self.model, self.first_joint)
        
    def update_pose(self, new_pose, overwrite_check_limit: Union[None, bool]=None):
        check_limit = overwrite_check_limit if overwrite_check_limit is not None \
            else self.default_check_limit
        if check_limit:
            in_limit, all_limit = in_actuated_joint_limit_range(self.yrdf, new_pose)
        if not check_limit or in_limit:
            self.yrdf_cfg = copy.deepcopy(new_pose)
            self.yrdf.update_cfg(self.yrdf_cfg)
        else:
            print(f"[WARNING] Robot is not updated. new_pose out of range. newpose:[{new_pose}] limit:[{all_limit}] ")

    def inverse_kinematics(self, joint_name: str, pose: np.ndarray, 
                           overwrite_check_limit: Union[None, bool]=None) -> Tuple[int, np.ndarray]:
        assert joint_name in self.joint_names, f"joint_name:{joint_name} not in {self.joint_names}"
        joint_id = self.joint_names.index(joint_name)

        oMdes = pinocchio.SE3(pose[:3, :3], pose[:3, 3])
        q = self.yrdf.cfg
        # q = pinocchio.neutral(self.model)

        best_err = float("inf")
        best_q = copy.deepcopy(q)

        i = 0
        while True:
            pinocchio.forwardKinematics(self.model, self.data, q)
            dMi = oMdes.actInv(self.data.oMi[joint_id])
            err_vec = pinocchio.log(dMi).vector
            err = np.linalg.norm(err_vec)
            if err < best_err:
                best_err = err
                best_q = copy.deepcopy(q)

            if err < self.ik_cfg.eps:
                success = True
                break
            if i >= self.ik_cfg.max_iter:
                success = False
                break

            j = pinocchio.computeJointJacobian(
                self.model, self.data, q, joint_id)
            v = - j.T.dot(np.linalg.solve(j.dot(j.T) +
                          self.ik_cfg.damp * np.eye(6, dtype=float), err_vec))
            q = pinocchio.integrate(self.model, q, v * self.ik_cfg.dt)
            i += 1

        best_q = best_q.flatten()
        self.update_pose(best_q, overwrite_check_limit)

        return success, i, best_q.flatten()

    def get_mesh(self) -> trimesh.Trimesh:
        return self.yrdf.scene.dump(concatenate=True)

    def set_base_transform_matrix(self, mat:np.ndarray):
        mat = mat.copy()
        self.base_current_matrix = mat.copy()
        set_yrdf_joint_origin_cfg(self.yrdf, self.base_joint, self.base_current_matrix)

        first_origin_mat = np.eye(4, dtype=float)
        first_origin_mat[:3, :3] = self.first_origin_rotation
        first_origin_mat[:3, 3] = self.first_origin_translation
        current_mat = mat @ np.linalg.inv(self.base_origin_matrix) \
            @ first_origin_mat
        r = current_mat[:3, :3]
        t = current_mat[:3, 3]
        set_pin_jointplacement_rotation_translation(self.model, self.first_joint, r, t)

    def get_base_transform_matrix(self) -> np.ndarray:
        return self.base_current_matrix.copy()
    
    def reset(self):
        self.update_pose(self.init_pose)
        self.set_base_transform_matrix(self.base_origin_matrix)

    def get_state(self) -> dict:
        state = {"class": "Robot",
                 "yrdf_cfg": copy.deepcopy(self.yrdf_cfg),
                 "base_transform_matrix": self.base_current_matrix.copy(),
                 "base_origin_matrix": self.base_origin_matrix.copy()}
        return state
    
    def set_state(self, state: dict):
        self.update_pose(copy.deepcopy(state["yrdf_cfg"]))
        self.set_base_transform_matrix(copy.deepcopy(state["base_transform_matrix"]))
