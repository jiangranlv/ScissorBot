import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from typing import List

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from rl.prepare_dataset import rotate2xyz_numpy
from collections import OrderedDict

ActionLookupTable = {
    "Open": {
        "ActionID": 0,
        "PredIndices": [4],
        "TargetIndices": [1],
    },
    "Close": {
        "ActionID": 1,
        "PredIndices": [5],
        "TargetIndices": [1],
    },
    "Translate": {
        "ActionID": 2,
        "PredIndices": [6, 7, 8],
        "TargetIndices": [1, 2, 3],
    },
    "Rotate": {
        "ActionID": 3,
        "PredIndices": [9, 10, 11],
        "TargetIndices": [1, 2, 3],
    },
}
ActionLookupTable_for9Drot = {
    "Open": {
        "ActionID": 0,
        "PredIndices": [4],
        "TargetIndices": [1],
    },
    "Close": {
        "ActionID": 1,
        "PredIndices": [5],
        "TargetIndices": [1],
    },
    "Translate": {
        "ActionID": 2,
        "PredIndices": [6, 7, 8],
        "TargetIndices": [1, 2, 3],
    },
    "Rotate": {
        "ActionID": 3,
        "PredIndices": [9, 10, 11, 12, 13, 14, 15, 16, 17],
        "TargetIndices": [1, 2, 3],
    },
}

ActionLookupTable_for16D = {
    "Open": {
        "ActionID": 0,
        "PredIndices": [5],
        "TargetIndices": [1],
    },
    "Close": {
        "ActionID": 1,
        "PredIndices": [6],
        "TargetIndices": [1],
    },
    "Translate": {
        "ActionID": 2,
        "PredIndices": [7, 8, 9],
        "TargetIndices": [1, 2, 3],
    },
    "Rotate": {
        "ActionID": 3,
        "PredIndices": [10, 11, 12],
        "TargetIndices": [1, 2, 3],
    },
    "RotateSmall": {
        "ActionID": 4,
        "PredIndices": [13,14, 15],
        "TargetIndices": [1, 2, 3],
    },
}

ActionLookupTable_for18Drot = {
    "Open": {
        "ActionID": 0,
        "PredIndices": [5],
        "TargetIndices": [1],
    },
    "Close": {
        "ActionID": 1,
        "PredIndices": [6],
        "TargetIndices": [1],
    },
    "Translate": {
        "ActionID": 2,
        "PredIndices": [7, 8, 9],
        "TargetIndices": [1, 2, 3],
    },
    "Rotate": {
        "ActionID": 3,
        "PredIndices": [10, 11, 12, 13, 14, 15, 16, 17, 18],
        "TargetIndices": [1, 2, 3],
    },
    "RotateSmall": {
        "ActionID": 4,
        "PredIndices": [19, 20, 21, 22, 23, 24, 25, 26, 27],
        "TargetIndices": [1, 2, 3],
    },
}
ActionNameList = ["Open", "Close", "Translate", "Rotate"]
ActionNameList5Types = ["Open", "Close", "Translate", "Rotate", "RotateSmall"]

def normalize_transform(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """return y = (x - mean) / std, automatically move `mean` and `std` to x.device"""
    return (x - mean.to(x.device)) / std.to(x.device)

def inverse_normalize_transform(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """return x = y * std + mean, automatically move `mean` and `std` to y.device"""
    return y * std.to(y.device) + mean.to(y.device)

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.state_dict()['param_groups'][0]['lr']

def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

def make_input_statistics(input_statistics: dict):
    d_new = {}
    for k, v in input_statistics.items():
        if isinstance(v, dict):
            d_new[k] = make_input_statistics(v)
        elif isinstance(v, list):
            d_new[k] = torch.tensor(v, dtype=torch.float32, device=torch.device("cpu"), requires_grad=False)
        else:
            raise ValueError(f"Unknown value type:{v}")
    return d_new

def get_translation_std_mean(input_statistics: dict):
    std = np.array(input_statistics["action"]["translation"]["std"])
    std_mean = np.mean(std)
    if not np.allclose(std, std_mean):
        print(f'[WARN] translation std is not all close:{std}')
    return std_mean

def get_pc_xyz_std_mean(input_statistics: dict):
    std = np.array(input_statistics["point_cloud"]["std"][:3])
    std_mean = np.mean(std)
    if not np.allclose(std, std_mean):
        print(f'[WARN] translation std is not all close:{std}')
    return std_mean

def get_next_edge_xyz_std_mean(input_statistics: dict):
    std_a = np.array(input_statistics["next_edge"]['a']["std"])
    std_a_mean = np.mean(std_a)
    if not np.allclose(std_a, std_a_mean):
        print(f'[WARN] translation std is not all close:{std_a}')
    
    std_b = np.array(input_statistics["next_edge"]['b']["std"])
    std_b_mean = np.mean(std_b)
    if not np.allclose(std_b, std_b_mean):
        print(f'[WARN] translation std is not all close:{std_b}')
    return std_a_mean, std_b_mean

def log_action_distribution(writer: SummaryWriter, all_action: np.ndarray, global_step: int, action_type_num:int, alt: dict):
    """
    action: [N, 4]
    """
    writer.add_histogram(f"action/Type", all_action[:, 0], global_step, bins="fd")
    for action_id in range(action_type_num):
        idx = np.where(np.round(all_action[:, 0] - action_id) == 0)
        action = all_action[idx]
        if action_id == alt["Open"]["ActionID"]:
            writer.add_histogram(f"action/Open", action[:, 1], global_step, bins="fd")
        elif action_id == alt["Close"]["ActionID"]:
            writer.add_histogram(f"action/Close", action[:, 1], global_step, bins="fd")
        elif action_id == alt["Translate"]["ActionID"]:
            for axis_idx, axis_str in enumerate(["x", "y", "z"]):
                writer.add_histogram(f"action/Translate_{axis_str}", action[:, 1+axis_idx], global_step, bins="fd")
        elif action_id == alt["Rotate"]["ActionID"]:
            xyz = rotate2xyz_numpy(action[:, 1:])
            for axis_idx, axis_str in enumerate(["x", "y", "z"]):
                writer.add_histogram(f"action/Rotate_{axis_str}", xyz[:, axis_idx], global_step, bins="fd")
            writer.add_histogram(f"action/Rotate_ang", np.rad2deg(np.abs(action[:, 1])), global_step, bins="fd")
        elif action_id == alt["RotateSmall"]["ActionID"]:
            xyz = rotate2xyz_numpy(action[:, 1:])
            for axis_idx, axis_str in enumerate(["x", "y", "z"]):
                writer.add_histogram(f"action/RotateSmall_{axis_str}", xyz[:, axis_idx], global_step, bins="fd")
            writer.add_histogram(f"action/RotateSmall_ang", np.rad2deg(np.abs(action[:, 1])), global_step, bins="fd")
        else:
            raise NotImplementedError
    writer.close()

def log_classification(writer: SummaryWriter, gt: np.ndarray, pred: np.ndarray, legend: List[str], global_step=None):
    class_n = len(legend)
    fig = plt.figure()
    correct_cnt = np.zeros((class_n, class_n))
    correct_ratio = np.zeros_like(correct_cnt)
    for pred_i in range(class_n):
        for gt_j in range(class_n):
            correct_cnt[pred_i, gt_j] = np.sum((gt == gt_j) & (pred == pred_i))
        correct_ratio[pred_i, :] = correct_cnt[pred_i, :] / np.sum(correct_cnt[pred_i, :])
    img = plt.imshow(correct_ratio, cmap="Greens")
    for pred_i in range(class_n):
        for gt_j in range(class_n):
            img.axes.text(gt_j, pred_i, "{:.0f}\n{:.3f}".format(correct_cnt[pred_i, gt_j], correct_ratio[pred_i, gt_j]),
                          horizontalalignment='center', verticalalignment='center')
    plt.xticks(np.arange(class_n), legend)
    plt.yticks(np.arange(class_n), legend)
    plt.xlabel("ground truth")
    plt.ylabel("prediction")
    plt.title("classification result")
    plt.colorbar()
    
    writer.add_figure('classification', fig, global_step=global_step)
    writer.close()
    plt.close()

def load_ckpt(model: torch.nn.Module, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for key, value in ckpt['state_dict'].items():
        new_key = key.replace('model.', '')  # 去掉 'model.' 前缀
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    return model