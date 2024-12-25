import pickle
import sys
import os
import shutil
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import multiprocessing as mp
from typing import List

import argparse
import ast
import copy
import time
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch

from dagger.data_utils import DAggerPreparer, GymStateTool

from policy.validate_policy import proj_root
from policy.cutgym import gym

def worker(gpuid: int,
           demos_dirs: List[str], input_base_dir: str, output_dir: str, worker_id: int, simulation_cfg, gym_cfg, render: bool):
    
    print(f"worker {worker_id} start! gpuid:{gpuid}")
    simulation_cfg.setup.cuda = gpuid
    env = gym(simulation_cfg, gym_cfg, output_mode= 'demos')
    env.reset(init= True)

    tool = GymStateTool(gpuid=gpuid, width_height=(512, 512))
    data_preparer = DAggerPreparer(tool, pre_cut_len = gym_cfg.pre_cut_len)
    data_preparer.filter_prepare_data(demos_dirs, input_base_dir, output_dir, device="cuda:0", render_result= render)
    print(f"worker {worker_id} end! gpuid:{gpuid}")


def allocate_demos_dirs(demos_dir: str, num: int) -> List[List[str]]:
    assert isinstance(demos_dir, str)
    assert isinstance(num, int)
    assert num > 0

    ret = [[] for _ in range(num)]
    for i, _d in enumerate(os.listdir(demos_dir)):
        d = os.path.join(demos_dir, _d)
        if os.path.isdir(d):
            for sub_i, sub_d in enumerate(os.listdir(d)):
                sub_dic = os.path.join(d, sub_d)
                if os.path.isdir(sub_dic):
                    ret[sub_i % num].append(os.path.abspath(sub_dic))    
    return ret

def allocate_gpus(gpus: str, num: int) -> List[int]:
    assert isinstance(gpus, str)
    assert isinstance(num, int)
    assert num > 0

    ret = []
    gpu_num = len(gpus)
    for i in range(num):
        ret.append(int(gpus[i % gpu_num]))
    return ret


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--demos-dirs", "-d", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)
    parser.add_argument("--gpuid", "-g", type=str, default="0")
    parser.add_argument("--process-num", "-p", type=int, default=1)
    parser.add_argument("--render", action = 'store_true')

    args = parser.parse_args()
    return args

        
def main():
    """example usage:
    ```
    python rl/prepare_dataset.py -d /DATA/disk1/epic/lvjiangran/code/cutgym/outputs/demos1/ -o ./rloutputs/ -s 1000 -ds 1000 -g 0
    ```"""
    args = get_args()

    cfg = OmegaConf.load(proj_root+ '/config/validate/base_config.yaml')
    simulation_cfg = OmegaConf.load(cfg.simulation_cfg_file)
    simulation_cfg = OmegaConf.merge(simulation_cfg, cfg.gym_cfg) 
    gym_cfg = cfg.gym_cfg
    
    demos_dirs_allocated = allocate_demos_dirs(args.demos_dirs, args.process_num)
    gpus_allocated = allocate_gpus(args.gpuid, args.process_num)
    # print(demos_dirs_allocated)
    # raise 1
    p_list: List[mp.Process] = []
    for worker_id, demos_dirs, gpuid in zip(range(args.process_num), demos_dirs_allocated, gpus_allocated):
        if args.process_num == 1:
            worker(gpuid, demos_dirs, args.demos_dirs, args.output_dir, worker_id, simulation_cfg, gym_cfg, args.render)
        else:
            p = mp.Process(target=worker, 
                        args=(gpuid, demos_dirs, args.demos_dirs, args.output_dir, worker_id, simulation_cfg, gym_cfg, args.render))
            p.start()
            p_list.append(p)

    for p in p_list:
        p.join()

    
if __name__ == '__main__':
    main()