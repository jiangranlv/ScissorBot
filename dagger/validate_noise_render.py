import pickle
import sys
import os
import shutil

import hydra
from omegaconf import DictConfig,OmegaConf
from typing import Union

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
from tqdm import tqdm

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from policy.generate_demos import pre_cut, fix_top_edge, fix_square_pad
from policy.generate_demos_high_level import HighLevelGym
from policy.pose_gym import PoseGym

from dagger.baselines import *

def free_cutting(env:PoseGym, policy:PoseBC, tex_file, num_iters):
    for i in range(num_iters):
        # TODO :No completed info
        state = env.gym.get_state()
        policy.receive_state(state, tex_file)
        pose = policy.predict()
        next_state = env.step(pose)
        pose_dict = [dict(next_pose = pose[i]) for i in range(env.gym.batch_size)]
        env.save_state_action(state, pose_dict)

def rule_based_cutting(env:HighLevelGym, goal_edge_set_batch, policy:BasePolicy, tex_file):

    inter_goal_rest_all_step = np.array(goal_edge_set_batch).transpose([1, 0, 2, 3])[:, : ,1, :] # t, b, 3 
    for i, inter_goal_rest in enumerate(inter_goal_rest_all_step):
        
        # rotate towards A
        before_rotate_state = env.gym.get_state()
        policy.receive_state(before_rotate_state, tex_file)
        rotate_direction = policy.predict(action_type= 'rotate')
        before_close_state, rotate_action = env.rotate_scissor(rotate_direction)
        env.save_state_action(before_rotate_state, rotate_action)
        # print(rotate_action)

        # close 
        policy.receive_state(before_close_state, tex_file)
        close_distance = policy.predict(action_type= 'close')
        before_open_state, close_action = env.close_scissor(close_distance)
        for batch_idx in range(env.gym.batch_size):
            env.gym.completed_batch[batch_idx].set(i)
            before_open_state[batch_idx]['completed'] = i
        env.save_state_action(before_close_state, close_action)
        # print(close_action)
        
        # open
        open_info = [dict(info = 'max') for i in range(env.gym.batch_size)]
        policy.receive_state(before_open_state, tex_file, is_open= True)
        before_tune_state, open_action = env.open_scissor(open_info)
        env.save_state_action(before_open_state, open_action)
        # print(open_action)

        if i == inter_goal_rest_all_step.shape[0]- 1:
            break

        # Tune to B
        policy.receive_state(before_tune_state, tex_file)
        tune_direction = policy.predict(action_type= 'tune')
        before_push_state, tune_action = env.tune_scissor(tune_direction)
        env.save_state_action(before_tune_state, tune_action)
        # print(tune_action)

        # move to B
        policy.receive_state(before_push_state, tex_file)
        push_distance = policy.predict(action_type= 'push')
        after_push_state, push_action = env.push_scissor(push_distance)
        env.save_state_action(before_push_state, push_action)
        # print(push_action)

def validate_high(policy:BasePolicy, env:Union[HighLevelGym, PoseGym], gym_cfg, goal_file_list, texture_file_list):
    assert len(goal_file_list) == len(texture_file_list)

    for goal_file, texture_file in tqdm(zip(goal_file_list, texture_file_list)):
        print(f'Validating: {goal_file}')

        if gym_cfg.demos.constraints == 'top_edge':
            fix_top_edge(env.gym)
        elif gym_cfg.demos.constraints == 'square_pad':
            fix_square_pad(env.gym)
        else:
            raise NotImplementedError()
        
        save_dir = '_'.join(goal_file.split('/')[-5: -1])
        os.mkdir(save_dir)
        os.chdir(save_dir)
        shutil.copy(goal_file, './test.yaml')
        shutil.copy(texture_file, './test.png')

        if env.gym.batch_size > 1:
            batch_dirs = []
            for batch_idx in range(env.gym.batch_size):
                batch_dir = f"batch_{str(batch_idx).zfill(len(str(env.gym.batch_size - 1)))}"
                os.makedirs(batch_dir)
                batch_dirs.append(batch_dir)
        else:
            batch_dirs = ['']
        env.gym.set_batch_dirs(batch_dirs)

        goal_edge_set = [np.array(OmegaConf.load(goal_file)['goal_edge_set'])]
        env.gym.goal_edge_set_batch = goal_edge_set
        
        pre_cut_state_list, pre_cut_action_list = pre_cut(env.gym)
        assert len(pre_cut_state_list) == len(pre_cut_action_list)
        for pre_cut_state, pre_cut_action in zip(pre_cut_state_list, pre_cut_action_list):
            # if pre_cut_action[0]['Action'] == 'Open':
            #     continue
            # policy.receive_state(pre_cut_state, tex_file = texture_file)
            env.save_state_action(pre_cut_state, pre_cut_action)

        policy.pre_process(state = pre_cut_state_list[0], tex_file = texture_file, goal_edge_set = goal_edge_set[0])
        try:
            if isinstance(env, PoseGym):
                free_cutting(env, policy, tex_file = texture_file, num_iters= 50)
            elif isinstance(env, HighLevelGym):
                rule_based_cutting(env, goal_edge_set_batch= goal_edge_set, policy= policy, tex_file= texture_file)
        except Exception as e:
            print(e)
            # raise 1
        env.reset(init = False)
        policy.reset()
        os.chdir('..')

def vis_score_distribution(folder_paths):
    scores = []

    folder_paths = folder_paths if isinstance(folder_paths, list) else [folder_paths]
    for folder in folder_paths:
        for traj_dir in os.listdir(folder):
            traj_path = os.path.join(folder, traj_dir)
            for filename in os.listdir(traj_path):
                if filename.endswith('.yaml'):
                    file_path = os.path.join(traj_path, filename)
                    with open(file_path, 'r') as yaml_file:
                        data = yaml.safe_load(yaml_file)
                        if data['sim_fail'] is not True:
                            score = data.get('score')
                            if score is not None:
                                scores.append(score)

    # 绘制分布图
    plt.hist(scores, bins=20, edgecolor='k')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.show()


def get_policy(cfg) ->BasePolicy:
    if cfg.policy == 'MultiFrameBC':
        policy = MultiFrameBC(cfg.net_cfg, cfg.render_cfg, compress= True)
    elif cfg.policy == 'MultiFrameBC_NoTune':
        policy = MultiFrameBC_NoTune(cfg.net_cfg, cfg.render_cfg, compress= True)
    elif cfg.policy == 'PreDetection':
        policy = PreDetection(render_cfg= cfg.render_cfg, compress= True, unit= 0.015)
    elif cfg.policy == 'PreDetectionGT':
        policy = PreDetectionGT(render_cfg= cfg.render_cfg, compress= True, unit= 0.015)
    elif cfg.policy == 'Tracking':
        policy = Tracking(data_cfg=dict(seq_len = 4, pose_dim = 6), render_cfg= cfg.render_cfg, compress= True, unit = 0.015)  
    elif cfg.policy == 'PoseBC':    
        policy = PoseBC(cfg.net_cfg, cfg.render_cfg, compress= True)
    elif cfg.policy == 'BC_ACT':
        policy = BC_ACT(cfg.net_cfg, cfg.render_cfg, compress= True, max_timesteps= cfg.max_timesteps, chunking_weight= cfg.chunking_weight)
    elif cfg.policy == 'BC_Pose_ACT':
        policy = BC_Pose_ACT(cfg.net_cfg, cfg.render_cfg, compress= True, max_timesteps= cfg.max_timesteps, chunking_weight= cfg.chunking_weight)
    else:
        raise NotImplementedError(cfg.policy)
    return policy

def get_env(simulation_cfg, gym_cfg, type = 'high'):
    if type == 'pose':
        env = PoseGym(simulation_cfg, gym_cfg)
    elif type == 'high':
        env = HighLevelGym(simulation_cfg, gym_cfg)
    env.reset(init = True)
    return env


@hydra.main(config_path="../config/validate", version_base='1.3')        
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    with open(cfg.val_set, 'r') as file:
        validate_goal_set = [str(line.strip()) for line in file]

    texture_file_set = [goal_file.replace('yaml', 'png') for goal_file in validate_goal_set]
    simulation_cfg = OmegaConf.load(cfg.simulation_cfg_file)
    simulation_cfg = OmegaConf.merge(simulation_cfg, cfg.gym_cfg) 

    env_type = 'pose' if 'pose' in cfg.policy.lower() else 'high'
    env = get_env(simulation_cfg, cfg.gym_cfg, type = env_type)
    policy = get_policy(cfg)

    validate_high(policy, env, cfg.gym_cfg, validate_goal_set, texture_file_set)

    # raw_data_dir = os.path.join(args.log_dir , 'raw_data')
    # os.makedirs(raw_data_dir, exist_ok= True)

    # prepared_data_dir = os.path.join(args.log_dir , 'prepared_data')
    # os.makedirs(prepared_data_dir, exist_ok= True)

    # filter_prepare_data(tool= test_env, gym = test_env.gym, data_dir= raw_data_dir, \
    #                                 output_dir= prepared_data_dir, ret_dataset= False)

    # vis_score_distribution(prepared_data_dir)

if __name__ == '__main__':
    main()