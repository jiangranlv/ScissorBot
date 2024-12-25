import pathlib
import math
from typing import List, Dict, Tuple, Union
import hydra
import pickle
import copy
from omegaconf import DictConfig, OmegaConf
from rl.prepare_dataset import rotate2quat_numpy
from src.maths import *
import numpy as np
import yaml
from pprint import pprint
from src.paper_cutting_environment import PaperCuttingEnvironment
from src.scissor import compute_ee_pose_clipper

# from debug_tool import timing_decorator
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__) ))


def uv_func(x):
    return (x / np.array([0.210, 1.0, 0.297]))[:, [0, 2]]

class changable_int():
    def __init__(self, value) -> None:
        self.value = value
    
    def __iadd__(self, other):
        assert isinstance(other, int)
        self.value += other
        return self
    
    def set(self, other):
        assert isinstance(other, int)
        self.value = other
        return self
    
    def get(self):
        return self.value
    
class gym(PaperCuttingEnvironment):
    def __init__(self, simulation_cfg: DictConfig, gym_cfg, debug =False, output_mode = 'demos') -> None:
        super().__init__(simulation_cfg)
        
        self.gym_time_step = gym_cfg.gym_time_step
        goal = get_config(gym_cfg.goal)
        self.goal_edge_set_batch = [np.array(goal.goal_edge_set) for _ in range(self.batch_size)]

        self.scissor_initial_pose_batch: List[dict] = \
            [OmegaConf.to_container(gym_cfg.scissor.init_pose) for _ in range(self.batch_size)]
        
        # safe angle range
        self.MINIMUM_ANGLE, self.MAXIMUM_ANGLE = super().get_scissor_open_range()
        safe_angle_slot = 0.1 * (self.MAXIMUM_ANGLE - self.MINIMUM_ANGLE)
        self.MINIMUM_ANGLE += safe_angle_slot #0.052
        self.MAXIMUM_ANGLE = 0.2 #0.468

        self.MAX_CUT_LENGTH = abs(self.compute_front_point_movement_given_theta(self.MAXIMUM_ANGLE,self.MINIMUM_ANGLE))#0.042
        print('Simultation Scissor Open Range: ', self.MINIMUM_ANGLE, self.MAXIMUM_ANGLE)
        print('Simultation Scissor Max Cut Length: ', self.MAX_CUT_LENGTH)
        
        self.max_move_per_step = self._translation_max_velocity * self.gym_time_step
        self.max_close_per_step = self._open_close_max_velocity * self.gym_time_step
        self.max_angle_per_step = self._rotation_max_velocity * self.gym_time_step
        print('Simulation Scissor Max Mover Per Step: ', self.max_move_per_step)
        print('Simulation Scissor Max Rotate Per Step: ', self.max_angle_per_step)
        
        self.debug = debug
        if self.debug:
            self.point_pos = []
        self.add_noise = gym_cfg.add_noise
        
        self.state_num = 0
        self.print_progress = gym_cfg.print_progress
        self.output_mode = output_mode
        self.completed_batch = [changable_int(-1) for _ in range(self.batch_size)]

        self.workspace = '.'

        self.front_point_theta = gym_cfg.scissor.front_point_theta

    def set_batch_dirs(self, batch_dirs: List[str]):
        self.batch_dirs = copy.deepcopy(batch_dirs)
        
    def save_state_action(self, action_batch):
        assert self.batch_dirs is not None, f"please first call self.set_batch_dirs()"
        state_action_batch = self.get_state()
        assert isinstance(action_batch, list)
        for batch_idx, (action, state_action) in enumerate(zip(action_batch, state_action_batch)):
            assert isinstance(action, dict)
            state_action['action'] = action
            state_id = '{:06d}'.format(self.state_num)
            pickle.dump(state_action, open(f"{self.workspace}/{self.batch_dirs[batch_idx]}/state{state_id}.pkl", "bw"))
        self.state_num += 1   

    def seperate_move_multi_step(self, action_batch: List[dict]):
        assert isinstance(action_batch, list)
        assert isinstance(action_batch[0], dict)
        assert len(action_batch) == self.batch_size
        assert action_batch[0]["Action"] == "Translate" or action_batch[0]["Action"] == "Rotate"
        _action_batch=copy.deepcopy(action_batch)

        _action_batch_detail = []
        for _action in _action_batch:
            _action_batch_idx_detail = []
            if _action['Action'] == 'Translate':
                trans_dis = np.linalg.norm(np.array(_action['Displacement']))
                if trans_dis > self.max_move_per_step:
                    n_step_move = int(trans_dis // self.max_move_per_step)
                    ratio = self.max_move_per_step / trans_dis
                    trans_one_step = [ratio * x for x in _action["Displacement"]]
                    for i_step in range(n_step_move):
                        _action_batch_idx_detail.append(dict(Action = 'Translate', Displacement = trans_one_step, Time = self.gym_time_step))
                    
                last_dis = [(trans_dis % self.max_move_per_step) / trans_dis * x for x in _action["Displacement"] ]
                _action_batch_idx_detail.append(dict(Action = 'Translate', Displacement = last_dis, Time = self.gym_time_step))
            if _action['Action'] == 'Rotate': 
                goal_angle = _action['Displacement'][0]
                if goal_angle > self.max_angle_per_step:
                    n_step_move = int(goal_angle // self.max_angle_per_step)
                    rotate_one_step = [self.max_angle_per_step, _action['Displacement'][1], _action['Displacement'][2]]
                    for i_step in range(n_step_move):
                        _action_batch_idx_detail.append(dict(Action = 'Rotate', Displacement = rotate_one_step, Time = self.gym_time_step))

                last_dis = [goal_angle % self.max_angle_per_step, _action['Displacement'][1], _action['Displacement'][2]]
                _action_batch_idx_detail.append(dict(Action = 'Rotate', Displacement = last_dis, Time = self.gym_time_step))
            
            _action_batch_detail.append(_action_batch_idx_detail)

        max_n_step = 0
        for _action_batch_idx_detail in _action_batch_detail:
            max_n_step = max(max_n_step, len(_action_batch_idx_detail))
        
        batch_action_padding = []
        for i_step in range(max_n_step):
            batch_action_padding_idx = []
            for _action_batch_idx_detail in _action_batch_detail:
                if i_step >=len(_action_batch_idx_detail):
                    batch_action_padding_idx.append(dict(Action = "None"))
                else:
                    batch_action_padding_idx.append(_action_batch_idx_detail[i_step])
            batch_action_padding.append(batch_action_padding_idx)

        return batch_action_padding, max_n_step

    # @timing_decorator
    def step(self, action_batch: List[dict], warn_info = True):
        assert isinstance(action_batch, list)
        assert isinstance(action_batch[0], dict)
        assert len(action_batch) == self.batch_size

        _action_batch=copy.deepcopy(action_batch)
        states = []
        # if self.output_mode == 'demos':
        #     pos_start = self.get_scissors_front_point()

        # before simulation save
        if self.output_mode == 'demos':
            action_batch_save = copy.deepcopy(action_batch)
            # if action_save['Action'] == 'Translate':
            #     pos_end = self.get_scissors_front_point()
            #     action_save['Displacement'] = (pos_end- pos_start).tolist()
            self.save_state_action(action_batch_save)
        elif self.output_mode == 'None':
            self.state_num += 1

        save_freq = self._frame_dt // self._dt if self.output_mode == 'render' or self.output_mode == 'video' else -1
        
        for _action in _action_batch:
            if _action['Action'] == 'Open' or _action['Action'] == 'Close':
                displacement = "{:.2f}".format(_action['Angle'])
            elif _action['Action'] == 'Rotate' or _action['Action'] == 'Translate':
                displacement = '[' + ','.join(["{:.2f}".format(num) for num in _action['Displacement']]) + ']'
            else: 
                displacement = 'None'
            action_info = (_action['Action'] + ':' +  displacement).ljust(40, ' ')
            
            if self.print_progress:
                print(action_info, end = ' ')
        
            _action['Time'] = self.gym_time_step if not 'Time' in _action.keys() else _action['Time']

        if _action_batch[0]['Action'] == 'Rotate' or _action_batch[0]['Action'] == 'Translate':
            _action_batch_seperate, max_n_step = self.seperate_move_multi_step(_action_batch)

            if max_n_step > 1 and warn_info:
                print(f'[WARNING] Multi Action Step Occurs, {max_n_step} steps append')
            for i_step in range(max_n_step):
                self.append_scissors_action(_action_batch_seperate[i_step])
                states += self.simulate(self.gym_time_step, save_freq = save_freq)
        
        else:
            self.append_scissors_action(_action_batch)
            states += self.simulate(self.gym_time_step, save_freq = save_freq)
        
        # after simulation save
        if self.output_mode == 'video':
            raise NotImplementedError
            for idx, state in enumerate(states):
                pickle.dump(state, open(f'{self.state_num}.pkl','bw'))
                self.state_num += 1
            
        return states

    def reset_scissor(self):
        self.set_scissors_pose(self.scissor_initial_pose_batch, compute_ee_pose_clipper)  

    def reset(self, init = False):
        if not init:
            super().reset()
            
        # self.append_constraints({
        # 0: np.array([0.0, 0.0, 0.0]),
        # 1: np.array([0.3, 0.0, 0.0]),
        # 2: np.array([0.3, 0.3, 0.0]),
        # 3: np.array([0.0, 0.3, 0.0]),
        #     }, 1.0)
        self.reset_scissor()
        self.state_num = 0
        for completed in self.completed_batch:
            completed.set(-1)
        
    def get_scissors_front_point(self) -> List[np.ndarray]:
        if self.front_point_theta is None:
            return super()._get_scissors_front_point(frame_name = ['world'] * self.batch_size)
        else:
            return super().compute_scissors_front_point_given_theta(frame_name = ['world'] * self.batch_size, 
                                            theta = [float(self.front_point_theta)] * self.batch_size)
    
    def append_scissors_action(self, action_batch) -> None:
        for action in action_batch:
            suffix = [0.0, 0.0, 0.0]
            if action['Action'] == 'Rotate':
                action["Action"] = 'Move'
                action['Displacement'] = suffix + action['Displacement']

            elif action['Action'] == 'Translate':
                action["Action"] = 'Move'
                action['Displacement'] = action['Displacement'] + suffix

        return super().append_scissors_action(action_batch)
    
    def debug_pos_movement(self):
        self.point_pos = np.array(self.point_pos).transpose(0,1) #N_points, times ,3
        
        
        
    def compute_theta_given_front_point_dist(self, front_point_dist:float) -> float:
        return super().compute_theta_given_front_point_dist(front_point_dist)
    
    def compute_front_point_dist_given_theta(self, theta:float) -> float:
        return super().compute_front_point_dist_given_theta(theta)
    
    def compute_front_point_movement_given_theta(self, start_theta:float, end_theta:float) -> float:
        return self.compute_front_point_dist_given_theta(end_theta) \
                        - self.compute_front_point_dist_given_theta(start_theta)
                        
    def compute_angle_given_front_point_movement(self, start_front_point_dist:float, end_front_point_dist:float) -> float:
        return self.compute_theta_given_front_point_dist(end_front_point_dist) \
                        - self.compute_theta_given_front_point_dist(start_front_point_dist)
                        
    def compute_front_point_movement_from_current(self, batch_idx: int, delta_theta:float):
        start_theta = super().get_scissors_pose()[batch_idx]['joint_0']
        end_theta = start_theta + delta_theta
        return self.compute_front_point_movement_given_theta(start_theta, end_theta)
    
    def compute_angle_from_current(self, batch_idx: int, delta_dist):
        start_theta = super().get_scissors_pose()[batch_idx]['joint_0']
        start_dist = self.compute_front_point_dist_given_theta(start_theta)
        end_dist = start_dist + delta_dist
        return self.compute_angle_given_front_point_movement(start_dist, end_dist)
    
    def get_current_pos_given_rest(self, batch_idx: int, rest_pos: np.ndarray):
        curr_pos = super().get_current_pos_given_rest(batch_idx, rest_pos)
        if self.add_noise:
            curr_pos += generate_noise()
        return curr_pos  
    
    def simulate(self, simulation_time, save_freq = -1, **kwargs):
        return super().simulate(simulation_time, save_state_freq= save_freq, compute_ee_pose= compute_ee_pose_clipper,
         uv_func =uv_func, print_progress= self.print_progress)
        
    def get_state(self) -> List[dict]:
        sim_state= super().get_state()
        front_point_batch = self.get_scissors_front_point()
        cut_direction_batch = self.get_scissors_cut_direction()
        for batch_idx in range(self.batch_size):
            sim_state[batch_idx]['completed'] = self.completed_batch[batch_idx].get()
            sim_state[batch_idx]['front_point'] = front_point_batch[batch_idx]
            sim_state[batch_idx]['cut_direction'] = cut_direction_batch[batch_idx][1]
            sim_state[batch_idx]['pose'] = get_scissor_pose_from_state(sim_state[batch_idx])
        return sim_state
    
def generate_noise(noise_range=(-0.0025, 0.0025)):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=3)
    return noise

class config(object):
    def __init__(self, cfg_dict):
        for k, v in sorted(cfg_dict.items()):
            setattr(self, k, v)
            # pprint(f"{k:20}: {v}")


def get_config(config_name):
    with open(os.path.join(proj_root, f"config/base_shape/{config_name}.yaml"), 'r') as f:
        try:
            cfg_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    
    cfg = config(cfg_dict)
        
    return cfg

def get_scissor_pose_from_state(state: dict):
    if isinstance(state["_sim_env"]["ti_objects"], list):
        for ti_object in state["_sim_env"]["ti_objects"]:
            if ti_object["class"] == "Scissor":
                scissor_state = ti_object
    if isinstance(state["_sim_env"]["ti_objects"], dict):
        scissor_state = state["_sim_env"]["ti_objects"]["Scissor"]

    result = np.zeros((8, ), np.float32)
    result[0] = scissor_state["direct_cfg"]["joint_0"]
    result[1:4] = scissor_state["direct_cfg"]["joint_1"][0:3]
    result[4:8] = rotate2quat_numpy(
        np.array(scissor_state["direct_cfg"]["joint_1"][3:6]))
    return result