import taichi as ti
import os

import datetime
import time
import numpy as np
import math
import copy
from typing import List, Tuple

import json
import hydra
import pathlib
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from src.cloth import Cloth
from src.cut_simulation_environment import CutSimulationEnvironment

import logging
import inspect


def get_next_scissors_pose(scissors_old_pose: List[dict], now_substep: int, is_stuck: list, cut_sim_cfg: DictConfig) -> Tuple[List[dict], List[dict]]:
    scissors_new_pose = [
        {"joint_0": 0.0,
         "joint_1": [0.37, 0.20,
                     0.06 * math.sin(now_substep * cut_sim_cfg.dt), 0.0, 0.0, 0.0]},
        {"joint_0": 0.0, "joint_1": [0.15, 0.35, -0.03, 1.57, 0.0, 1.57]}
    ]
    scissors_velocity = [
        {"joint_0": 0.0,
            "joint_1": [0.0, 0.0, 0.06 * math.cos(now_substep * cut_sim_cfg.dt), 0.0, 0.0, 0.0]},
        {"joint_0": 0.0, "joint_1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    ]
    return scissors_new_pose, scissors_velocity


def substep(sim_env: CutSimulationEnvironment, scissors_new_pose: List[dict], scissors_velocity: List[dict]) -> list:
    is_stuck = sim_env.substep(scissors_new_pose, scissors_velocity)
    return is_stuck


@hydra.main(config_path="./config", config_name=pathlib.Path(__file__).stem, version_base='1.3')
def main(cfg: DictConfig):
    log = logging.getLogger(__name__)
    log.info(inspect.getsource(get_next_scissors_pose))

    # initialize taichi
    setup_cfg = cfg.setup
    os.environ['CUDA_VISIBLE_DEVICES'] = setup_cfg['cuda']
    if setup_cfg.arch == 'cuda':
        arch = ti.cuda
    else:
        arch = ti.cpu
    ti.init(arch=arch, debug=setup_cfg.debug,
            kernel_profiler=setup_cfg.kernel_profiler, offline_cache=setup_cfg.offline_cache, random_seed=0, device_memory_GB=setup_cfg.device_memory_GB, fast_math=False)

    # output config
    output_cfg = cfg.output

    # output file path setup
    series_prefix = "{}/collision".format(os.getcwd())

    # sim env config
    cut_sim_cfg = cfg.env.cut_sim

    # cloth config
    cloth_cfg = cfg.cloth

    # scissor config
    scissor_cfg = cfg.scissor

    # create sim env
    sim_env = CutSimulationEnvironment(output_cfg=output_cfg,
                                       cut_sim_cfg=cut_sim_cfg, cloth_cfg=cloth_cfg, scissor_cfg=scissor_cfg, log=log)
    is_stuck = [False, False]
    scissors_old_pose = [
        {"joint_0": 0.0, "joint_1": [0.35, 0.20, 0.0, 0.0, -0.1, 0.0]},
        {"joint_0": 0.0, "joint_1": [0.15, 0.37, -0.04, 1.57, 0.0, 1.57]}
    ]

    # start simulation
    if output_cfg['print_info']:
        print("arch={} cuda={}".format(cfg.setup.arch, cfg.setup.cuda))

    frame_i = 0
    now_substep = 0
    substep_per_frame = sim_env.frame_dt / sim_env.dt

    while frame_i < setup_cfg.max_frame_i:

        if not sim_env.is_pause:
            while now_substep < (frame_i + 1) * substep_per_frame:
                scissors_new_pose, scissors_velocity = get_next_scissors_pose(
                    scissors_old_pose, now_substep, is_stuck, cut_sim_cfg)
                is_stuck = substep(
                    sim_env, scissors_new_pose, scissors_velocity)
                scissors_old_pose = scissors_new_pose

                now_substep += 1

        if sim_env.is_write_cloth:
            '''
            sim_env.export_all_ply(
                frame_i=frame_i, series_prefix=series_prefix + "_cloth", export_scissor=False)'''
            sim_env.export_all_ply(
                frame_i=frame_i, series_prefix=series_prefix)

        if output_cfg.save_topo_info:
            with open(series_prefix + "_topo_" + str(frame_i).zfill(6) + ".txt", "w") as f_obj:
                f_obj.write(sim_env.cloth.get_topo_info_str())

        if output_cfg.save_scissor_force_info:
            forces = sim_env.collision.get_scissor_penalty_force()
            with open(series_prefix + "_force_" + str(frame_i).zfill(6) + ".npy", "bw") as f_obj:
                np.save(f_obj, np.array(forces, dtype=object), allow_pickle=True)

        if output_cfg.print_time_cost:
            # sim_env.clock.print_info(drop=(5, 5))
            log.info("\n" + sim_env.clock.get_info_str(drop=(5, 5)))

        if output_cfg.print_info:
            print("[TIME] simulation_time:{:.5f} real_time:{}".format(
                frame_i * sim_env.frame_dt, datetime.datetime.now()))

            if frame_i == output_cfg.print_kernel_info_frame and setup_cfg.kernel_profiler:
                ti.profiler.print_kernel_profiler_info()

        frame_i += 1


if __name__ == "__main__":
    main()
