import shutil
from typing import List
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os
import numpy as np

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__) ))
sys.path.append(proj_root)
sys.path.append(os.path.join(proj_root, 'policy'))

@hydra.main(config_path="../config", config_name='heuristic_debug', version_base='1.3')
def main(gym_cfg: DictConfig):
    if gym_cfg.level == 'low':
        from generate_demos import fix_square_pad, fix_top_edge, random_generate_goal, heuristic_w_pre_cut
        from cutgym import gym
    elif gym_cfg.level == 'high':
        from generate_demos_high_level import fix_square_pad, fix_top_edge, random_generate_goal, heuristic_w_pre_cut, HighLevelGym as gym

    simulation_cfg = OmegaConf.load(proj_root+ '/config/paper_cutting_game_realworld_fast.yaml')
    simulation_cfg = OmegaConf.merge(simulation_cfg, gym_cfg) 
    env = gym(simulation_cfg, gym_cfg)
    env_gym = env.gym if hasattr(env, 'gym') else env
    batch_size = env_gym.batch_size

    env_gym.reset(init= True)
    if gym_cfg.demos.constraints == 'top_edge':
        fix_top_edge(env_gym)
    elif gym_cfg.demos.constraints == 'square_pad':
        fix_square_pad(env_gym)
    else:
        raise NotImplementedError()

    batch_dirs: List[str] = []
    for batch_idx in range(batch_size):
        batch_dir = f"batch_{str(batch_idx).zfill(len(str(batch_size - 1)))}"
        os.makedirs(batch_dir)
        batch_dirs.append(batch_dir)
    env_gym.set_batch_dirs(batch_dirs)

    if gym_cfg.demos.curve_type == 'monotonic' or gym_cfg.demos.curve_type == 'two_peak':
        goal_edge_set_batch = [
            random_generate_goal(os.path.join(batch_dirs[batch_idx], 'debug'), gym_cfg.demos.num_lines, gym_cfg.demos.start.x, gym_cfg.demos.start.y,
                        gym_cfg.demos.len_range, gym_cfg.demos.init_angle, gym_cfg.demos.angle_slot, curve_type= gym_cfg.demos.curve_type)
            for batch_idx in range(batch_size)
        ]
        env_gym.goal_edge_set_batch = goal_edge_set_batch
        heuristic_w_pre_cut(env, goal_edge_set_batch)
    
    else:
        goal_file = f'/data1/jiangranlv/code/cutting_paper_fem/pattern/{gym_cfg.demos.curve_type}.yaml'
        texture_file = goal_file.replace('yaml', 'png')
        shutil.copy(goal_file, './test.yaml')
        shutil.copy(texture_file, './test.png')

        pattern_lines = OmegaConf.load(goal_file)
        for key, line in pattern_lines.items():
            print('Conducting {} in the Pattern'.format(key))
            goal_edge_set_batch = [np.array(line)]
            env_gym.goal_edge_set_batch = goal_edge_set_batch
            heuristic_w_pre_cut(env, goal_edge_set_batch)
            env_gym.reset_scissor()

    env_gym.reset(init= False)
        
    
if __name__ == "__main__":
    main()
