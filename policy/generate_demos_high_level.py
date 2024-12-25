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
from policy.generate_demos import check_detach_batch, pre_cut, fix_top_edge, fix_square_pad, random_generate_goal

class HighLevelGym():
    def __init__(self, simulation_cfg: DictConfig, gym_cfg) -> None:
        self.gym = gym(simulation_cfg, gym_cfg, debug = 'False', output_mode = None)
        self.cut_length = 0.015  
        self.state_num = 0
    
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

    def high2low(self, action_high_level):
        action_high_batch = copy.deepcopy(action_high_level)
        action_low_batch = []
        for batch_idx in range(self.gym.batch_size):
            if action_high_batch[batch_idx]['Action'] == 'Rotate'  or action_high_batch[batch_idx]['Action'] == 'Tune':
                action_low = dict(Action = 'Rotate')
                pos, current_direction, current_axs = self.gym.get_scissors_cut_direction(batch_idx)
                action_low['Displacement']= list(compute_relative_rotation(current_direction, action_high_batch[batch_idx]['direction']))
                action_low_batch.append(action_low)

            elif action_high_batch[batch_idx]['Action'] == 'Push':
                action_low = dict(Action = 'Translate')
                pos, current_direction, current_axs = self.gym.get_scissors_cut_direction(batch_idx)
                action_low['Displacement']= list(current_direction * action_high_batch[batch_idx]['distance'])
                action_low_batch.append(action_low)

            elif action_high_batch[batch_idx]['Action'] == 'Close':
                action_low = dict(Action = 'Close')
                action_low['Angle']= abs(self.gym.compute_angle_from_current(batch_idx, action_high_batch[batch_idx]['distance']))
                action_low_batch.append(action_low)

            elif action_high_batch[batch_idx]['Action'] == 'Open':
                action_low = dict(Action = 'Open')
                current_angle = self.gym.get_scissors_pose(batch_idx)["joint_0"]
                action_low['Angle']=  max(self.gym.MAXIMUM_ANGLE - current_angle, 0.0)
                action_low_batch.append(action_low)
            
            elif action_high_batch[batch_idx]['Action'] == 'None':
                action_low = dict(Action = 'None')
                action_low_batch.append(action_low)

            else:
                raise NotImplementedError()
            
        return action_low_batch  

    def step_batch_action(self, action_type, key , value):
        action_batch = self.value_to_batch_action(action_type, key, value)
        return self.step(action_batch), action_batch
    
    def value_to_batch_action(self, action_type, key , value):
        assert self.gym.batch_size == len(value), f" {self.gym.batch_size} expected, {len(value)} got"
        action_batch = []
        for batch_idx in range(self.gym.batch_size):
            if value[batch_idx] is not None:
                action = dict(Action = action_type)
                action[key] = value[batch_idx]
            else:
                action = dict(Action = 'None')
            action_batch.append(action)
        return action_batch
    
    def step(self, action_batch):
        action_low = self.high2low(action_high_level= action_batch)
        self.gym.step(action_low)
        return self.gym.get_state()

    def rotate_scissor(self, direction_batch):
        return self.step_batch_action(action_type='Rotate', key = 'direction', value= direction_batch)

    def tune_scissor(self, direction_batch):
        return self.step_batch_action(action_type='Tune', key = 'direction', value= direction_batch)

    def push_scissor(self, distance_batch):
        return self.step_batch_action(action_type='Push', key = 'distance', value= distance_batch)

    def close_scissor(self, distance_batch):
        return self.step_batch_action(action_type='Close', key = 'distance', value= distance_batch)
        
    def open_scissor(self, open_batch):
        return self.step_batch_action(action_type='Open', key= 'info', value = open_batch)

class GymAgent():
    def __init__(self) -> None:
        pass

    def compute_direction(self, env:HighLevelGym, target_rest, detach_flag_batch = None):
        current_pos_batch = env.gym.get_scissors_front_point()
        direction_batch = []
        for batch_idx in range(env.gym.batch_size):
            if detach_flag_batch is None or not detach_flag_batch[batch_idx]:
                target_point = env.gym.get_current_pos_given_rest(batch_idx, target_rest[batch_idx])
                target_dirc= target_point - current_pos_batch[batch_idx]
                pos, current_direction, current_axs = env.gym.get_scissors_cut_direction(batch_idx)
                try:
                    angle,_,_ = compute_relative_rotation(current_direction, target_dirc)
                except:
                    raise ValueError(batch_idx, 'relative rotation error, current:{} target:{}'.format(current_direction, target_dirc))
                if angle < env.gym.max_angle_per_step * 2:
                    direction_batch.append(target_dirc)
                elif detach_flag_batch is not None:
                    detach_flag_batch[batch_idx] = True
                    os.mknod(os.path.join(env.gym.batch_dirs[batch_idx], "detached.txt"))
                    print('Detached Triangle During Simulation')
                    direction_batch.append(None)
            else:
                direction_batch.append(None)
        return direction_batch

    def compute_push_distance(self, env:HighLevelGym, target_rest, detach_flag_batch = None):
        current_pos_batch = env.gym.get_scissors_front_point()
        distance_batch = []
        for batch_idx in range(env.gym.batch_size):
            target_point = env.gym.get_current_pos_given_rest(batch_idx, target_rest[batch_idx])
            if detach_flag_batch is None or not detach_flag_batch[batch_idx]:
                dis = np.linalg.norm(target_point - current_pos_batch[batch_idx])
                if dis < env.gym.max_move_per_step:
                    distance_batch.append(dis)
                elif detach_flag_batch is not None:
                    detach_flag_batch[batch_idx] = True
                    os.mknod(os.path.join(env.gym.batch_dirs[batch_idx], "detached.txt"))
                    print('Detached Triangle During Simulation')
                    distance_batch.append(None)
            else:
                distance_batch.append(None)
        return distance_batch

    def compute_close_distance(self, env:HighLevelGym, target_rest, detach_flag_batch = None):
        current_pos_batch = env.gym.get_scissors_front_point()
        distance_batch = []
        for batch_idx in range(env.gym.batch_size):
            current = current_pos_batch[batch_idx]
            final_goal = env.gym.get_current_pos_given_rest(batch_idx, target_rest[batch_idx])
            if detach_flag_batch is None or not detach_flag_batch[batch_idx]:
                distance_batch.append(np.linalg.norm(final_goal - current))
            else:
                distance_batch.append(None)
        return distance_batch

    def ensure_detach_for_open(self, env:HighLevelGym, detach_flag_batch = None):
        open_batch = []
        for batch_idx in range(env.gym.batch_size):
            if detach_flag_batch is None or not detach_flag_batch[batch_idx]:
                open_batch.append('max')
            else:
                open_batch.append(None)   
        return open_batch
    
    def fix_push_distance(self ,env:HighLevelGym, before_state, after_state, detach_flag_batch =None):
        distance_batch = []
        for batch_idx in range(env.gym.batch_size):     
            if detach_flag_batch is None or not detach_flag_batch[batch_idx]:
                fixed_dis = np.linalg.norm(after_state[batch_idx]['front_point'] - before_state[batch_idx]['front_point'])
                distance_batch.append(fixed_dis)
            else:
                distance_batch.append(None)
        return distance_batch
    
def heuristic_w_pre_cut(env:HighLevelGym, goal_edge_set_batch):
    state_list, action_list = pre_cut(env.gym)
    assert len(state_list) == len(action_list)
    for state, action in zip(state_list, action_list):
        env.save_state_action(state, action)
    agent = GymAgent()
    rule_based_cutting(env, goal_edge_set_batch, agent)


def rule_based_cutting(env:HighLevelGym, goal_edge_set_batch, agent:GymAgent):
    detach_flag_batch = [False for i in range(env.gym.batch_size)]
    inter_goal_rest_all_step = np.array(goal_edge_set_batch).transpose([1, 0, 2, 3])[:, : ,1, :] # t, b, 3 
    for i, inter_goal_rest in enumerate(inter_goal_rest_all_step):
        
        # rotate towards A
        before_rotate_state = env.gym.get_state()
        rotate_direction = agent.compute_direction(env, inter_goal_rest, detach_flag_batch = detach_flag_batch)
        before_close_state, rotate_action = env.rotate_scissor(rotate_direction)
        env.save_state_action(before_rotate_state, rotate_action)

        close_distance = agent.compute_close_distance(env, inter_goal_rest, detach_flag_batch)
        # close_distance = [env.cut_length for i in range(env.gym.batch_size)]
        before_open_state, close_action = env.close_scissor(close_distance)
        for batch_idx in range(env.gym.batch_size):
            env.gym.completed_batch[batch_idx].set(i)
            before_open_state[batch_idx]['completed'] = i
        env.save_state_action(before_close_state, close_action)
        
        open_info = agent.ensure_detach_for_open(env, detach_flag_batch)
        before_tune_state, open_action = env.open_scissor(open_info)
        env.save_state_action(before_open_state, open_action)

        if i == inter_goal_rest_all_step.shape[0]- 1:
            break
        
        if i < inter_goal_rest_all_step.shape[0]- 1:
            next_goal_rest= inter_goal_rest_all_step[i + 1]
            detach_flag_batch = check_detach_batch(env.gym, inter_goal_rest, next_goal_rest, detach_flag_batch)

        # Tune to B
        tune_direction = agent.compute_direction(env, inter_goal_rest,detach_flag_batch = detach_flag_batch)
        before_push_state, tune_action =env.tune_scissor(tune_direction)
        env.save_state_action(before_tune_state, tune_action)

        # move to B
        push_distance = agent.compute_push_distance(env, inter_goal_rest, detach_flag_batch)
        after_push_state, push_action = env.push_scissor(push_distance)
        
        # fix push distance 
        # fixed_dis = agent.fix_push_distance(env,before_push_state, after_push_state, detach_flag_batch)
        # fixed_push_action = env.value_to_batch_action(action_type='push', key= 'distance', value = fixed_dis)
        # env.save_state_action(before_push_state, fixed_push_action)
        env.save_state_action(before_push_state, push_action)

    

@hydra.main(config_path="../config", config_name='generate_demos_high_level', version_base='1.3')
def main(gym_cfg: DictConfig):
    simulation_cfg = OmegaConf.load(proj_root+ '/config/paper_cutting_game_realworld_fast.yaml')
    simulation_cfg = OmegaConf.merge(simulation_cfg, gym_cfg) 
    env = HighLevelGym(simulation_cfg, gym_cfg)
    env_gym = env.gym
    batch_size = env_gym.batch_size

    max_demos = gym_cfg.demos.max_num 
    env.reset(init= True)

    random_goal = True
    if hasattr(gym_cfg, 'goal_path_dir'):
        goal_path_set = [os.path.join(gym_cfg.goal_path_dir, file) for file in os.listdir(gym_cfg.goal_path_dir)]
        # seperate goal_path_set into batch_size
        loaded_goal_set = [np.array(OmegaConf.load(path)['goal_edge_set']) for path in goal_path_set]
        loaded_goal_batch_set = [loaded_goal_set[i:i + batch_size] for i in range(0, len(goal_path_set), batch_size)]
        goal_path_batch_set= [goal_path_set[i:i + batch_size] for i in range(0, len(goal_path_set), batch_size)]
        random_goal = False

    for i in tqdm(range(max_demos // batch_size)):
        env_gym.log.info(f"this is the {i} th trajectory")

        if gym_cfg.demos.constraints == 'top_edge':
            fix_top_edge(env_gym)
        elif gym_cfg.demos.constraints == 'square_pad':
            fix_square_pad(env_gym)
        else:
            raise NotImplementedError()
        
        os.mkdir(str(i))
        os.chdir(str(i))

        batch_dirs: List[str] = []
        for batch_idx in range(env_gym.batch_size):
            batch_dir = f"batch_{str(batch_idx).zfill(len(str(env_gym.batch_size - 1)))}"
            os.makedirs(batch_dir)
            batch_dirs.append(batch_dir)
        env_gym.set_batch_dirs(batch_dirs)

        if random_goal:
            goal_edge_set_batch = [
                random_generate_goal(os.path.join(batch_dirs[batch_idx], str(i)), gym_cfg.demos.curve_cfg)
                for batch_idx in range(batch_size)
            ]
        else:
            goal_edge_set_batch = loaded_goal_batch_set[i]
            
            for batch_idx in range(batch_size):
                OmegaConf.save(dict(goal_edge_set = goal_edge_set_batch[batch_idx].tolist(), goal_path = goal_path_batch_set[i][batch_idx]
                                    ), os.path.join(batch_dirs[batch_idx], str(i)+'.yaml'))
        env_gym.goal_edge_set_batch = goal_edge_set_batch
        heuristic_w_pre_cut(env, goal_edge_set_batch)
        env.reset(init= False)
        os.chdir('..')

if __name__ == "__main__":
    main()