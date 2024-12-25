import math
import sys 
import os
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__) ))
sys.path.append(proj_root)
sys.path.append(os.path.join(proj_root, 'policy'))

from typing import List
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import copy
import numpy as np

from src.maths import compute_relative_rotation

from policy.cutgym import gym

class PoseGym():
    def __init__(self, simulation_cfg: DictConfig, gym_cfg) -> None:
        self.gym = gym(simulation_cfg, gym_cfg, debug = 'False', output_mode = None)
        self.state_num = 0
        assert self.gym.batch_size == 1
    
    def reset(self, init = False):
        self.gym.reset(init)
        self.state_num = 0

    def save_state_action(self, state_batch = None, action_batch = None):
        assert action_batch is not None
        assert self.gym.batch_dirs is not None, f"please first call self.set_batch_dirs()"
        assert isinstance(action_batch, list)
        
        state_action_batch = self.gym.get_state() if state_batch is None else state_batch
        for batch_idx, (action, state_action) in enumerate(zip(action_batch, state_action_batch)):
            assert isinstance(action, dict)
            state_action['action'] = action
            state_num = '{:06d}'.format(self.state_num)
            pickle.dump(state_action, open(f"{self.gym.workspace}/{self.gym.batch_dirs[batch_idx]}/state{state_num}.pkl", "bw"))
        self.state_num += 1 

    def pose2action(self, pose_batch):
        pose_idx = dict(angle = 0, front_point = [1, 2, 3], direction = [4, 5, 6])
        pose_batch = copy.deepcopy(pose_batch)
        action_batch = []
        action_batch_2 = []

        # Only works for batch_size = 1
        batch_idx = 0
        pos, current_direction, current_axs = self.gym.get_scissors_cut_direction(batch_idx)
        current_front_point = self.gym.get_scissors_front_point()[batch_idx]
        current_angle = self.gym.get_scissors_pose(batch_idx)['joint_0']

        angle = pose_batch[batch_idx][pose_idx['angle']]
        direction = pose_batch[batch_idx][pose_idx['direction']]
        front_point = pose_batch[batch_idx][pose_idx['front_point']]

        delta_rot= list(compute_relative_rotation(current_direction, direction))
        delta_trans = front_point - current_front_point
        delta_angle = angle - current_angle
        movement = delta_trans.tolist() + delta_rot

        angle_ratio = delta_angle / self.gym.max_close_per_step
        rot_ratio = delta_rot[0] / self.gym.max_angle_per_step
        trans_ratio = np.linalg.norm(delta_trans) / self.gym.max_move_per_step
        action_move = action_angle = None
        
        if rot_ratio > 0.1 or trans_ratio > 0.1:
            action_move = dict(Action = 'Move', Displacement = movement)
        if angle_ratio > 0.1:
            action_angle = dict(Action = 'Open', Angle = delta_angle)
        elif angle_ratio < -0.1:
            action_angle = dict(Action = 'Close', Angle = -delta_angle)
        
        if action_move is None and action_angle is None:
            action_batch.append(dict(Action = 'Stay'))
            action_batch_2.append(None)
        elif action_move is not None and action_angle is None:
            action_batch.append(action_move)
            action_batch_2.append(None)
        elif action_move is None and action_angle is not None:
            action_batch.append(action_angle)
            action_batch_2.append(None)
        elif action_move is not None and action_angle is not None:
            action_angle['Time'] = angle_ratio / (angle_ratio + rot_ratio + trans_ratio) * self.gym.gym_time_step
            action_move['Time'] = (rot_ratio + trans_ratio) / (angle_ratio + rot_ratio + trans_ratio) * self.gym.gym_time_step
            action_batch.append(action_move)
            action_batch_2.append(action_angle)
            
        return action_batch, action_batch_2 

    
    def step(self, pose_batch):
        action_batch_1, action_batch_2 = self.pose2action(pose_batch)
        self.gym.step(action_batch_1)
        if action_batch_2[0] is not None:
            self.gym.step(action_batch_2)
        return self.gym.get_state()


        

    

if __name__ == "__main__":
    main()