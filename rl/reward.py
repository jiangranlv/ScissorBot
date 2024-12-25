import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from typing import Tuple, Callable
import matplotlib.pyplot as plt
import pickle
import trimesh
import numpy as np
import torch
from PIL import Image, ImageDraw
from omegaconf import OmegaConf

def get_edge_set_on_uv(state: dict, 
                       resolution: np.ndarray,
                       real_size: np.ndarray,
                       axis_idx=[0, 2],
                       new_edge_eps=1e-3) -> np.ndarray:
    if isinstance(state["_sim_env"]["ti_objects"], list):
        cloth_state = state["_sim_env"]["ti_objects"][0]
    if isinstance(state["_sim_env"]["ti_objects"], dict):
        cloth_state = state["_sim_env"]["ti_objects"]["Cloth"]
    
    vr = cloth_state["vertices_rest_pos"]
    e2vf = cloth_state["mesh"]["edges_vid_fid"]
    a1, a2 = axis_idx
    s1, s2 = real_size
    is_new_edge=lambda mid: (mid[:, a1] > new_edge_eps) & (mid[:, a1] < s1 - new_edge_eps) & \
                            (mid[:, a2] > new_edge_eps) & (mid[:, a2] < s2 - new_edge_eps)
    is_new = np.where((e2vf[:, 3] == -1) & is_new_edge((vr[e2vf[:, 0]] + vr[e2vf[:, 1]]) / 2))
    e2v = e2vf[is_new][:, :2]
    en = e2v.shape[0]
    edge = np.zeros((en, 2, 3))
    edge[:, 0, :] = vr[e2v[..., 0], ...]
    edge[:, 1, :] = vr[e2v[..., 1], ...]
    return edge[:, :, axis_idx] * resolution / real_size


def draw_edge_set(edge_set: list, resolution: np.ndarray, edge_width: int) -> Image.Image:
    img = Image.new('L', tuple(resolution.tolist()), color='white')
    draw = ImageDraw.Draw(img)
    for edge in edge_set:
        edge[:, 1] = resolution[1] - edge[:, 1]
        draw.line(tuple(edge.reshape(-1)), width=edge_width)
    return img


def calculate_state_score(state: dict, goal_set: np.ndarray, 
                          real_size=np.array([0.210, 0.297]), axis_idx=[0, 2],
                          grid_size=0.0003, distance_tolerance=0.003,
                          texture_edge_width=0.0015, state_edge_width=0.001,
                          reward_per_meter=1e2, penalty_per_meter=1e2,
                          visualize=False):
    """
    Calculate score of a single state.

    Args:
        state: dict
        goal_set: np.ndarray
        real_size: real size of texture_img
        axis_idx: identify which axes corresponding to `real_size`, default: x and z
        grid_size: used to estimate correct cut and wrong cut
        texture_edge_width: linewidth on texture image
        state_edge_width: linewidth on current state
        distance_tolerance: tolerance to judge correct cut and wrong cut
        reward_per_meter: per meter reward of correct cut
        penalty_per_meter: per meter penalty of wrong cut
    
    Return:
        score: float
        state_img: ndarray, [H, W]
        texture_img: ndarray, [H, W]
        correct_pixel: ndarray, [H, W]
        wrong_pixel: ndarray, [H, W]
    """
    resolution = np.ceil(real_size / grid_size).astype(np.int64)
    grid_size_real = real_size / resolution
    area_per_pixel = grid_size_real[0] * grid_size_real[1]

    # we use the following variables to avoid rounding error
    grid_size_mean = np.mean(grid_size_real)
    state_edge_width_on_img = np.ceil(state_edge_width / grid_size_mean).astype(np.int64)
    texture_edge_width_on_img = np.ceil(texture_edge_width / grid_size_mean).astype(np.int64)
    state_edge_width_real = state_edge_width_on_img * grid_size_mean
    texture_edge_width_real = texture_edge_width_on_img * grid_size_mean

    # extract all edges in state and texture, then draw on a new image
    edge_set_uv = get_edge_set_on_uv(state, resolution, real_size, axis_idx)
    goal_set_uv = goal_set[:, :, axis_idx] * resolution / real_size
    img = draw_edge_set(edge_set_uv, resolution, state_edge_width_on_img)
    img_gt = draw_edge_set(goal_set_uv, resolution, texture_edge_width_on_img)

    # binarize imgs, '_t' refers to target
    x = np.array(img)
    y = np.array(img_gt)
    m, n = x.shape
    is_target=lambda x: x < 255
    x_t:np.ndarray = is_target(x) # current image matrix
    y_t:np.ndarray = is_target(y) # texture image matrix

    # expand imgs to make a wider range judgement
    x_t_expand = x_t.copy()
    y_t_expand = y_t.copy()
    pixel_tolerance = distance_tolerance / grid_size_mean
    pixel_tolerance_floor = int(distance_tolerance / grid_size_mean)
    for i in range(-pixel_tolerance_floor, pixel_tolerance_floor+1):
        for j in range(-pixel_tolerance_floor, pixel_tolerance_floor+1):
            if (i ** 2 + j ** 2) <= pixel_tolerance:
                if i >= 0:
                    si1, si2 = slice(i, m), slice(0, m-i)
                else:
                    si1, si2 = slice(0, m+i), slice(-i, m)
                if j >= 0:
                    sj1, sj2 = slice(j, n), slice(0, n-j)
                else:
                    sj1, sj2 = slice(0, n+j), slice(-j, n)
                x_t_expand[si1, sj1] |= x_t[si2, sj2]
                y_t_expand[si1, sj1] |= y_t[si2, sj2]

    correct_pixel = y_t & x_t_expand 
    wrong_pixel = x_t & ~y_t_expand
    if visualize:
        visualize_score(x, y, correct_pixel, wrong_pixel)

    return (np.sum(correct_pixel) / texture_edge_width_real * reward_per_meter - 
            np.sum(wrong_pixel) / state_edge_width_real * penalty_per_meter) * area_per_pixel, \
                x, y, correct_pixel, wrong_pixel


def visualize_score(state: np.ndarray, texture: np.ndarray, correct: np.ndarray, wrong: np.ndarray, 
                    width_range=slice(0, 800), height_range=slice(400, 600), figsize=(9, 6)):
    ss = np.tile(state[..., np.newaxis], (1, 1, 4)) / 255.
    ss[np.where(state<255)] = [0., 0., 1., .5]
    ss[np.where(state>=255)] = [1., 1., 1., 0.]

    tt = np.tile(texture[..., np.newaxis], (1, 1, 4)) / 255.
    tt[np.where(texture<255)] = [.5, .5, 0., .5]
    tt[np.where(texture>=255)] = [1., 1., 1., 0.]

    cc = np.tile(correct[..., np.newaxis], (1, 1, 4)).astype(np.float64)
    cc[np.where(correct)] = [0., 1., 0., .8]
    cc[np.where(~correct)] = [1., 1., 1., 0.]

    ww = np.tile(wrong[..., np.newaxis], (1, 1, 4)).astype(np.float64)
    ww[np.where(wrong)] = [1., 0., 0., .5]
    ww[np.where(~wrong)] = [1., 1., 1., 0.]

    plt.figure(figsize=figsize)
    plt.imshow(ss[height_range, width_range])
    plt.imshow(tt[height_range, width_range])
    plt.imshow(cc[height_range, width_range])
    plt.imshow(ww[height_range, width_range])
    plt.title("yellow=goal, blue=real, red=wrong, green=correct")
    plt.show()


def calculate_reward(curr_state_path: str, next_state_path: str, goal_set_path: str):
    goal_set = np.array(OmegaConf.to_container(OmegaConf.load(goal_set_path))["goal_edge_set"])
    curr_state = pickle.load(open(curr_state_path, "rb"))
    next_state = pickle.load(open(next_state_path, "rb"))
    return calculate_state_score(next_state, goal_set)[0] - \
        calculate_state_score(curr_state, goal_set)[0]


def calculate_completeness_chamfer(complete_percentage: float,
                                   wrong_percentage: float,
                                   penalty_coeff: float) -> float:
    return complete_percentage - penalty_coeff * wrong_percentage


def cal_state_diff_chamfer(state1: dict, state2: dict, goal_set: np.ndarray, 
                           real_size=np.array([0.210, 0.297]), axis_idx=[0, 2],
                           sample_on_goal=2000, sample_on_edge=2000,
                           correct_func=lambda x: x < 0.001,
                           wrong_func=lambda x: x > 0.001,
                           use_torch=True, torch_device="cpu",
                           visualize=False, pass_debug_info=None):
    """
    Example usage:
        `
        >>> state1 = pickle.load(open("/DATA/disk1/epic/lvjiangran/code/cutgym/outputs/demos_0717/174013/0/state15.pkl", "rb"))
        >>> state2 = pickle.load(open("/DATA/disk1/epic/lvjiangran/code/cutgym/outputs/demos_0717/174013/0/state20.pkl", "rb"))
        >>> goal_set_path = "/DATA/disk1/epic/lvjiangran/code/cutgym/outputs/demos_0717/174013/0/0.yaml"
        >>> goal_set = np.array(OmegaConf.to_container(OmegaConf.load(goal_set_path))["goal_edge_set"])
        >>> cd, gd, ed, cp, wp, cdp = cal_state_diff_chamfer(state1, state2, goal_set, torch_device="cuda:0", visualize=False)
        >>> score = calculate_completeness_chamfer(cp, wp, 0.2)
        `

    Return:
        - chamfer distance: scalar OR 0.0
        - goal_distance: [sample_on_goal, ] OR None
        - edge_distance: [sample_on_edge, ] OR None
        - correct percentage: scalar OR 0.0
        - wrong percentage: scalar OR 0.0
        - chamfer distance percentage: scalar OR 0.0
    """
    def uniform_sample_on_edge_set(edge_set: np.ndarray, sample_n: int):
        assert edge_set.shape[1:] == (2, 3)

        # sample
        weight = np.linalg.norm(edge_set[:, 1, :] - edge_set[:, 0, :], axis=1)
        coeff = np.random.random(sample_n)
        edge_idx = np.random.choice(edge_set.shape[0], size=sample_n, p=weight / np.sum(weight))
        sample = edge_set[edge_idx, 0, :] * coeff[:, None] + edge_set[edge_idx, 1, :] * (1 - coeff[:, None])
        return sample
    
    def get_edge_rest_pos(state: dict, new_edge_eps=1e-4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return
            - all edge [e, 2, 3]
            - new edge idx [n]
        """
        if isinstance(state["_sim_env"]["ti_objects"], list):
            cloth_state = state["_sim_env"]["ti_objects"][0]
        if isinstance(state["_sim_env"]["ti_objects"], dict):
            cloth_state = state["_sim_env"]["ti_objects"]["Cloth"]
        
        vr = cloth_state["vertices_rest_pos"]
        e2vf = cloth_state["mesh"]["edges_vid_fid"]
        
        a1, a2 = axis_idx
        s1, s2 = real_size
        is_new_edge=lambda mid: (mid[:, a1] > new_edge_eps) & (mid[:, a1] < s1 - new_edge_eps) & \
                                (mid[:, a2] > new_edge_eps) & (mid[:, a2] < s2 - new_edge_eps)
        is_new = np.where((e2vf[:, 3] == -1) & is_new_edge((vr[e2vf[:, 0]] + vr[e2vf[:, 1]]) / 2))[0]
        e2v = e2vf[:, :2]
        en = e2v.shape[0]
        edge = np.zeros((en, 2, 3))
        edge[:, 0, :] = vr[e2v[:, 0], :]
        edge[:, 1, :] = vr[e2v[:, 1], :]
        return edge, is_new
    
    def calc_cd(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        N, D = x.shape
        M, D_ = y.shape
        assert D == D_
        if not use_torch:
            d = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
            return np.mean(np.min(d, axis=0)) + np.mean(np.min(d, axis=1))
        else:
            x_torch = torch.tensor(x, dtype=torch.float32, device=torch_device)
            y_torch = torch.tensor(y, dtype=torch.float32, device=torch_device)
            d = torch.linalg.norm(x_torch[:, None, :] - y_torch[None, :, :], dim=2)
            yd = torch.min(d, dim=0)[0].detach().cpu().numpy()
            xd = torch.min(d, dim=1)[0].detach().cpu().numpy()
            return np.mean(xd) + np.mean(yd), xd, yd

    def calc_edge_set_length(edge_set: np.ndarray):
        return np.sum(np.linalg.norm(edge_set[:, 1, :] - edge_set[:, 0, :], axis=1))
    
    def correct_wrong_func_wrapper(func: Callable, x: np.ndarray):
        x_shape = x.shape
        y = func(x)
        assert y.shape == x_shape
        return np.clip(y, 0., 1.)

    complete1 = state1["completed"]
    complete2 = state2["completed"]
    # print(f"complete1 {complete1}, complete2 {complete2}")
    if complete1 < complete2:
        goal_n = goal_set.shape[0]
        if complete1 < -1 or complete2 > goal_n - 1:
            print(f"[WARN] in calculate_state_different_cd, complete1({complete1}), complete2({complete2}), goal_n({goal_n}),", 
                  f"complete number out of range pass_debug_info:{pass_debug_info}")
        goal_pc = uniform_sample_on_edge_set(goal_set[complete1+1:complete2+1, :, :], sample_on_goal)
    else:
        print(f"[WARN] in calculate_state_different_cd, complete1({complete1}) >= complete2({complete2}),",
              f"pass_debug_info:{pass_debug_info}")
        goal_pc = uniform_sample_on_edge_set(goal_set[:, :, :], sample_on_goal)
    assert goal_pc.shape == (sample_on_goal, 3)

    epos1, eidx1 = get_edge_rest_pos(state1)
    epos2, eidx2 = get_edge_rest_pos(state2)
    new_break_edge_idx = np.array(sorted(list(set(eidx2.tolist()) - set(eidx1.tolist()))))
    # print(f"eidx1.shape:{eidx1.shape} eidx2.shape:{eidx2.shape}")
    
    if len(new_break_edge_idx) == 0: # no new break edge
        return 0., None, None, 0., 0., 0.

    else:
        edge_pc = uniform_sample_on_edge_set(epos2[new_break_edge_idx], sample_on_edge)
        assert edge_pc.shape == (sample_on_edge, 3)

        cd, gd, ed = calc_cd(goal_pc, edge_pc)
        correct_result = correct_wrong_func_wrapper(correct_func, gd)
        wrong_result = correct_wrong_func_wrapper(wrong_func, ed)
        cp = np.sum(correct_result) / sample_on_goal
        wp = np.sum(wrong_result) / sample_on_edge
        cdp = cd / calc_edge_set_length(goal_set)

        if visualize:
            fig = plt.figure(figsize=(9, 3))
            ax = fig.gca()
            plt.scatter(x=goal_pc[:, axis_idx[0]], y=goal_pc[:, axis_idx[1]], s=2, c=np.array([[.5, .5, 0., .5]]))
            plt.scatter(x=edge_pc[:, axis_idx[0]], y=edge_pc[:, axis_idx[1]], s=2, c=np.array([[0., 0., 1., .5]]))

            wrong_idx = np.where(wrong_result > 0.5)[0]
            plt.scatter(x=edge_pc[wrong_idx, axis_idx[0]], y=edge_pc[wrong_idx, axis_idx[1]], s=2, c=np.array([[1., 0., 0., .5]]))

            correct_idx = np.where(correct_result > 0.5)[0]
            plt.scatter(x=goal_pc[correct_idx, axis_idx[0]], y=goal_pc[correct_idx, axis_idx[1]], s=2, c=np.array([[0., 1., 0., .8]]))

            plt.legend(["goal", "real", "wrong", "correct"])
            ax.set_aspect(1.0)
            plt.grid()
            plt.show()

        return cd, gd, ed, cp, wp, cdp

def cal_state_diff_chamfer_given_goal_pc(state1: dict, state2: dict, goal_pc: np.ndarray, 
                           real_size=np.array([0.210, 0.297]), axis_idx=[0, 2],
                           sample_on_goal=2000, sample_on_edge=2000,
                           correct_func=lambda x: x < 0.001,
                           wrong_func=lambda x: x > 0.001,
                           use_torch=True, torch_device="cpu",
                           visualize=False, pass_debug_info=None):
    """
    Example usage:
        `
        >>> state1 = pickle.load(open("/DATA/disk1/epic/lvjiangran/code/cutgym/outputs/demos_0717/174013/0/state15.pkl", "rb"))
        >>> state2 = pickle.load(open("/DATA/disk1/epic/lvjiangran/code/cutgym/outputs/demos_0717/174013/0/state20.pkl", "rb"))
        >>> goal_set_path = "/DATA/disk1/epic/lvjiangran/code/cutgym/outputs/demos_0717/174013/0/0.yaml"
        >>> goal_set = np.array(OmegaConf.to_container(OmegaConf.load(goal_set_path))["goal_edge_set"])
        >>> cd, gd, ed, cp, wp, cdp = cal_state_diff_chamfer(state1, state2, goal_set, torch_device="cuda:0", visualize=False)
        >>> score = calculate_completeness_chamfer(cp, wp, 0.2)
        `

    Return:
        - chamfer distance: scalar OR 0.0
        - goal_distance: [sample_on_goal, ] OR None
        - edge_distance: [sample_on_edge, ] OR None
        - correct percentage: scalar OR 0.0
        - wrong percentage: scalar OR 0.0
        - chamfer distance percentage: scalar OR 0.0
    """
    def uniform_sample_on_edge_set(edge_set: np.ndarray, sample_n: int):
        assert edge_set.shape[1:] == (2, 3)

        # sample
        weight = np.linalg.norm(edge_set[:, 1, :] - edge_set[:, 0, :], axis=1)
        coeff = np.random.random(sample_n)
        edge_idx = np.random.choice(edge_set.shape[0], size=sample_n, p=weight / np.sum(weight))
        sample = edge_set[edge_idx, 0, :] * coeff[:, None] + edge_set[edge_idx, 1, :] * (1 - coeff[:, None])
        return sample
    
    def get_edge_rest_pos(state: dict, new_edge_eps=1e-4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return
            - all edge [e, 2, 3]
            - new edge idx [n]
        """
        if isinstance(state["_sim_env"]["ti_objects"], list):
            cloth_state = state["_sim_env"]["ti_objects"][0]
        if isinstance(state["_sim_env"]["ti_objects"], dict):
            cloth_state = state["_sim_env"]["ti_objects"]["Cloth"]
        
        vr = cloth_state["vertices_rest_pos"]
        e2vf = cloth_state["mesh"]["edges_vid_fid"]
        
        a1, a2 = axis_idx
        s1, s2 = real_size
        is_new_edge=lambda mid: (mid[:, a1] > new_edge_eps) & (mid[:, a1] < s1 - new_edge_eps) & \
                                (mid[:, a2] > new_edge_eps) & (mid[:, a2] < s2 - new_edge_eps)
        is_new = np.where((e2vf[:, 3] == -1) & is_new_edge((vr[e2vf[:, 0]] + vr[e2vf[:, 1]]) / 2))[0]
        e2v = e2vf[:, :2]
        en = e2v.shape[0]
        edge = np.zeros((en, 2, 3))
        edge[:, 0, :] = vr[e2v[:, 0], :]
        edge[:, 1, :] = vr[e2v[:, 1], :]
        return edge, is_new
    
    def calc_cd(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        N, D = x.shape
        M, D_ = y.shape
        assert D == D_
        if not use_torch:
            d = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
            return np.mean(np.min(d, axis=0)) + np.mean(np.min(d, axis=1))
        else:
            x_torch = torch.tensor(x, dtype=torch.float32, device=torch_device)
            y_torch = torch.tensor(y, dtype=torch.float32, device=torch_device)
            d = torch.linalg.norm(x_torch[:, None, :] - y_torch[None, :, :], dim=2)
            yd = torch.min(d, dim=0)[0].detach().cpu().numpy()
            xd = torch.min(d, dim=1)[0].detach().cpu().numpy()
            return np.mean(xd) + np.mean(yd), xd, yd
    
    def correct_wrong_func_wrapper(func: Callable, x: np.ndarray):
        x_shape = x.shape
        y = func(x)
        assert y.shape == x_shape
        return np.clip(y, 0., 1.)

    assert goal_pc.shape == (sample_on_goal, 3), f"goal_pc.shape:{goal_pc.shape}"

    epos1, eidx1 = get_edge_rest_pos(state1)
    epos2, eidx2 = get_edge_rest_pos(state2)
    new_break_edge_idx = np.array(sorted(list(set(eidx2.tolist()) - set(eidx1.tolist()))))
    # print(f"eidx1.shape:{eidx1.shape} eidx2.shape:{eidx2.shape}")
    
    if len(new_break_edge_idx) == 0: # no new break edge
        return 0., None, None, 0., 0., 0.

    else:
        edge_pc = uniform_sample_on_edge_set(epos2[new_break_edge_idx], sample_on_edge)
        assert edge_pc.shape == (sample_on_edge, 3)

        cd, gd, ed = calc_cd(goal_pc, edge_pc)
        correct_result = correct_wrong_func_wrapper(correct_func, gd)
        wrong_result = correct_wrong_func_wrapper(wrong_func, ed)
        cp = np.sum(correct_result) / sample_on_goal
        wp = np.sum(wrong_result) / sample_on_edge
        cdp = None # Not implemented

        if visualize:
            fig = plt.figure(figsize=(9, 3))
            ax = fig.gca()
            plt.scatter(x=goal_pc[:, axis_idx[0]], y=goal_pc[:, axis_idx[1]], s=2, c=np.array([[.5, .5, 0., .5]]))
            plt.scatter(x=edge_pc[:, axis_idx[0]], y=edge_pc[:, axis_idx[1]], s=2, c=np.array([[0., 0., 1., .5]]))

            wrong_idx = np.where(wrong_result > 0.5)[0]
            plt.scatter(x=edge_pc[wrong_idx, axis_idx[0]], y=edge_pc[wrong_idx, axis_idx[1]], s=2, c=np.array([[1., 0., 0., .5]]))

            correct_idx = np.where(correct_result > 0.5)[0]
            plt.scatter(x=goal_pc[correct_idx, axis_idx[0]], y=goal_pc[correct_idx, axis_idx[1]], s=2, c=np.array([[0., 1., 0., .8]]))

            plt.legend(["goal", "real", "wrong", "correct"])
            ax.set_aspect(1.0)
            plt.grid()
            plt.show()

        return cd, gd, ed, cp, wp, cdp