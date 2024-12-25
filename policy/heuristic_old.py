import pathlib
import math
from typing import List, Dict, Tuple, Union
from pprint import pprint
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import copy
import os
from PIL import Image, ImageDraw
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__) ))
sys.path.append(proj_root)
sys.path.append(os.path.join(proj_root, 'policy'))

from src.maths import *
import yaml
from tqdm import tqdm
from cutgym import gym
import random

def random_generate_goal(save_name, n = 8, start_x = 0, start_y = 0.2):
    PAPER_LENGTH = np.array([0.210, 0.297])
    eps = 1e-4

    def generate_trajctory(func, n, start_x, start_y):
        x_values, z_values = func(n, start_x, start_y)
        
        samples = np.column_stack((x_values, np.zeros_like(x_values)))
        samples = np.column_stack((samples,z_values))
        samples[0, 0] = 0.0
        edges = np.stack((samples[:-1], samples[1:]), 1)
        with open(f'{save_name}.yaml', 'a') as f:
            try:
                yaml.safe_dump(dict(goal_edge_set = edges.tolist()), f)
            except yaml.YAMLError as e:
                print(e)
        return edges
    
    def generate_texture_img(edges):

        goal_edge_set = edges[:,:,[0, 2]] * 1024 / PAPER_LENGTH
        
        img = Image.new('RGB', (1024, 1024), color='white')
        draw = ImageDraw.Draw(img)
        
        for edge in goal_edge_set:
            edge[:, 1] = 1024 - edge[:, 1]
            draw.line(tuple(edge.reshape(-1)), fill='red', width=5)
    
        img.save(f'{save_name}.png')
    
    def generate_line_segments(n, start_x, start_y):
        x_list = [start_x]
        y_list = [start_y]
        
        angle = random.uniform(-math.radians(40), math.radians(40))
        total_angle = 0
        
        for _ in range(n):
            length = random.uniform(0.02, 0.03)
            
            x = x_list[-1] + length * math.cos(angle)
            y = y_list[-1] + length * math.sin(angle)
            total_angle += angle
            
            angle = random.uniform(-math.radians(30), math.radians(30))    
            while abs(total_angle + angle) > math.radians(50):
                angle = random.uniform(-math.radians(30), math.radians(30))
            
            if x > PAPER_LENGTH[0] or y > PAPER_LENGTH[1] or y < 0:
                break
            x_list.append(x)
            y_list.append(y)
    
        return x_list, y_list
    
    edges = generate_trajctory(generate_line_segments, n, start_x, start_y)
    generate_texture_img(edges)
    
    return edges   

def heuristic_wo_pre_cut(env:gym, goal_edge_set):
    pre_cut(env)
    rule_based_cutting(env, goal_edge_set)
    
def rule_based_cutting(env:gym, goal_edge_set):
    for i in range(len(goal_edge_set)):
        if i != 0 :
            start_point = env.get_current_pos_given_rest(goal_edge_set[i][0])
            
            # first open the scissor
            open_to_max_angle(env)
            
            # rotate the scissor to direction of current->A
            current_pos = env.get_scissors_front_point()
            pos, current_direction, axs = env.get_scissors_cut_direction()[0]
            rot_between_dirc_move = compute_relative_rotation(current_direction, (start_point - current_pos))
            if rot_between_dirc_move is not None:
                action_rotate_c2a = dict(Action = 'Rotate', Displacement = 
                                    list(rot_between_dirc_move))
                env.step(action_rotate_c2a)
            
            # move to A
            current_pos = env.get_scissors_front_point()
            start_point = env.get_current_pos_given_rest(goal_edge_set[i][0])
            if i==0 :
                safe_move(env, current_pos, start_point)
            elif (goal_edge_set[i][0] == goal_edge_set[i-1][1]).all():
                move_scissor(env, current_pos, start_point)
                if check_detached(env, goal_edge_set[i][0], goal_edge_set[i][1]):
                    os.mknod("detached.txt")
                    print('Detached Triangle During Simulation, Resetting ...')
                    break
            else:
                raise NotImplementedError()
            
        # rotate the scissor to direction of A->B 
        pos, current_direction, axs = env.get_scissors_cut_direction()[0]
        end_point = env.get_current_pos_given_rest(goal_edge_set[i][1])
        current_pos = env.get_scissors_front_point()
        action_rotate_a2b = dict(Action = 'Rotate', Displacement = 
                            list(compute_relative_rotation(current_direction, (end_point - current_pos))))
        env.step(action_rotate_a2b)

        # close the scissor to cut 
        cut_line(env,current_pos, goal_edge_set[i][1])
        
        env.completed.set(i)
    
    #finally open the scissor
    open_to_max_angle(env)
        
def normalize(vec):
    return vec / np.linalg.norm(vec)

def compute_relative_rotation(current, target):
    current = current / np.linalg.norm(current)
    target = target / np.linalg.norm(target)

    axis = np.cross(current, target)
    axis = axis / np.linalg.norm(axis)
    
    if np.dot(current, target) > 1 or np.dot(current, target) < -1:
        return None
    
    angle = np.arccos(np.dot(current, target))
    # print(current, axis, angle)
    theta, phi = direc_to_theta_phi(axis)
    
    return angle, theta, phi

def open_to_max_angle(env:gym, save_action = False):
    scissors_pose = env.get_scissors_pose()
    current_angle = scissors_pose[0]["joint_0"]
    if current_angle < env.MAXIMUM_ANGLE:
        action_open = dict(Action = 'Open', Angle = env.MAXIMUM_ANGLE - current_angle)
        env.step(action_open)
        if save_action: 
            return [action_open]
        else:
            return [None]
        
def cut_line(env:gym, start, end_uv):

    current = start
    final_goal = env.get_current_pos_given_rest(end_uv)
    last_dis = math.dist(current, final_goal)
    while(math.dist(current, final_goal) > env.MAX_CUT_LENGTH):
        env.step(dict(Action = 'Close', Angle = env.MAXIMUM_ANGLE - env.MINIMUM_ANGLE))
        temp_goal = env.get_scissors_front_point()
        open_to_max_angle(env)
        current = env.get_scissors_front_point()
        move_scissor(env, current, temp_goal)
        current = env.get_scissors_front_point() 
        
        # rotate to temp->goal
        pos, current_direction, axs = env.get_scissors_cut_direction()[0]
        final_goal = env.get_current_pos_given_rest(end_uv)
        rot_between_dirc_move = compute_relative_rotation(current_direction, (final_goal - current))
        if rot_between_dirc_move is not None:
            action_rotate_c2a = dict(Action = 'Rotate', Displacement = 
                                list(rot_between_dirc_move))
            env.step(action_rotate_c2a)
            
        curr_dis = math.dist(current, final_goal)
        if curr_dis >= last_dis: 
            print('Warning: this cut may be failure because the front point is far away from the goal')
            if not os.path.exists("fail.txt"):
                os.mknod("fail.txt")
            return 
        last_dis = curr_dis
    env.step(dict(Action = 'Close', Angle = abs(env.compute_angle_from_current(math.dist(current, final_goal)))))

def move_scissor(env:gym, start, end):
    action_translate = dict(Action = 'Translate', Displacement = (end - start).tolist())
    env.step(action_translate)
    # final_pos = env.get_scissors_front_point()
    # if np.linalg.norm(final_pos - end) / np.linalg.norm(end - start)  > 0.1:
    #     print(f'scissor may be stuck in state {env.state_num -1}')
                        

def safe_move(env:gym, start, end, save_action = False, save_state = False):
    dis = (end - start).tolist()
    vert_dis = [0, dis[1], 0]
    action_translate1 = dict(Action = 'Translate', Displacement = vert_dis)
    env.step(action_translate1)
    state_list = []
    state_list.append(env.get_state()) if save_state else None
    
    hori_dis = [dis[0], 0, dis[2]]
    action_translate2 = dict(Action = 'Translate', Displacement = hori_dis)
    env.step(action_translate2)
    # state_list.append(env.get_state()) if save_state else None
    
    if save_state and save_action:
        return state_list, [action_translate1, action_translate2]
    if save_state: 
        return state_list
    elif save_action :
        return [action_translate1, action_translate2]
    else:
        return [None]
    
    # vert_dis = [0, 0.015, 0 ]
    # action_translate = dict(Action = 'Translate', Displacement = vert_dis)
    # env.step(action_translate)
    
def pre_cut(env:gym, save_action = False, save_state = False):
    start_point = env.get_current_pos_given_rest(env.goal_edge_set[0][0])
    scissors_pose = env.get_scissors_pose()
    current_angle = scissors_pose[0]["joint_0"]
    
    state_list = [env.get_state()] if save_state else []
    # first open the scissor
    action_list = []
    if current_angle < 0:
        action_open = open_to_max_angle(env, save_action) 
        action_list += action_open
        state_list.append(env.get_state()) if save_state else None
        
    pos, current_direction, current_axs = env.get_scissors_cut_direction()[0]
    paper_edge_uv_a = np.array([0 , 0, env.goal_edge_set[0][0][2]- 0.015])
    paper_edge_uv_b = np.array([0 , 0, env.goal_edge_set[0][0][2]+ 0.015])
    paper_edge_dirc = env.get_current_pos_given_rest(paper_edge_uv_b) \
                                - env.get_current_pos_given_rest(paper_edge_uv_a)
    action_rotate_axis_parrell_edge = dict(Action = 'Rotate', Displacement = 
                            list(compute_relative_rotation(current_axs, paper_edge_dirc)))
    env.step(action_rotate_axis_parrell_edge)
    state_list.append(env.get_state()) if save_state else None
    
    # rotate the scissor to direction of A->B 
    pos, current_direction, current_axs = env.get_scissors_cut_direction()[0]
    end_point = env.get_current_pos_given_rest(env.goal_edge_set[0][1])
    current_pos = env.get_scissors_front_point()
    action_rotate_a2b = dict(Action = 'Rotate', Displacement = 
                            list(compute_relative_rotation(current_direction, (end_point - start_point))))
    env.step(action_rotate_a2b)
    state_list.append(env.get_state()) if save_state else None
    # move to A
    start_point = env.get_current_pos_given_rest(env.goal_edge_set[0][0])
    current_pos = env.get_scissors_front_point()
    move_result = safe_move(env, current_pos, start_point, save_action, save_state)
    
    
    if save_state and save_action:
        action_list.append(action_rotate_axis_parrell_edge)
        action_list.append(action_rotate_a2b)
        action_list += move_result[1]
        
        state_list = state_list + move_result[0] 
        return state_list, action_list
    
    if save_state:
        state_list = state_list + move_result
        return state_list
    
    if save_action:
        action_list.append(action_rotate_axis_parrell_edge)
        action_list.append(action_rotate_a2b)
        action_list += move_result[1]
        return action_list

    
def fix_square_pad(env:gym):
    def vid_to_old_pos(vid: int) -> np.ndarray:
        i, k = vid // 31, vid % 31
        return np.array([i / 21 * 0.210, 0, k / 30 * 0.297], dtype=float)
    
    vid_list = [10 * 31 + 30, 11 * 31 + 30, 10 * 31 + 29, 11 * 31 + 29]   
    constraint =  dict()  
    for vid in vid_list:
        constraint[vid] = vid_to_old_pos(vid)
    env.append_constraints(constraint, 1.0)    

def fix_top_edge(env:gym):
    def vid_to_old_pos(vid: int) -> np.ndarray:
        i, k = vid // 31, vid % 31
        return np.array([i / 21 * 0.210, 0, k / 30 * 0.297], dtype=float)
    
    constraint =  dict() 
    for top_vertex in range(22):
        vid = top_vertex * 31 + 30
        constraint.update({vid: vid_to_old_pos(vid)})
    env.append_constraints(constraint, 1.0)   

def fix_2_corner(env:gym):
    env.append_constraints({
            30: np.array([0.0, 0.0, 0.297]),
            681: np.array([0.210, 0.0, 0.297]),
                }, 1.0)   
    
def check_detached(env:gym , uv_start: np.ndarray, uv_target):
    num_samples = 5
    thres = 5
    samples_x = np.linspace(uv_start[0], uv_target[0], num_samples)
    samples_y = np.linspace(uv_start[1], uv_target[1], num_samples)
    samples_z = np.linspace(uv_start[2], uv_target[2], num_samples)
    samples = np.column_stack((samples_x, samples_y, samples_z))
    deformation = np.zeros(num_samples -1)
    for i in range(1, num_samples):
        deformation[i-1] = compute_deformation(env, uv_start, samples[i])
    overall_deform = remove_max_and_mean(deformation)
    if overall_deform > thres:
        return True
    return False

def compute_deformation(env:gym, uv1, uv2):
    pos1 = env.get_current_pos_given_rest(uv1)
    pos2 = env.get_current_pos_given_rest(uv2)
    return np.linalg.norm(pos1 - pos2) / np.linalg.norm(uv1 - uv2)    

def remove_max_and_mean(array):
    max_value = np.max(array)
    filtered_array = array[array < max_value]
    result = np.mean(filtered_array)
    return result

def rotate_vertical_with_paper(env:gym, start_uv, end_uv):
    distance = 0.005
    def compute_perpendicular_bisector(p1, p2, distance):
        mid_point = (p1 + p2) / 2
        direction = p2 - p1

        perpendicular_direction = np.array([-direction[2], 0, direction[0]])

        perpendicular_direction /= np.linalg.norm(perpendicular_direction)

        point1 = mid_point + distance * perpendicular_direction
        point2 = mid_point - distance * perpendicular_direction

        return point1, point2
    
    pos, current_direction, current_axs = env.get_scissors_cut_direction()[0]
    paper_edge_uv_a, paper_edge_uv_b= compute_perpendicular_bisector(start_uv, end_uv, distance)
    paper_edge_dirc = env.get_current_pos_given_rest(paper_edge_uv_b) \
                                - env.get_current_pos_given_rest(paper_edge_uv_a)
    action_rotate_axis_parrell_edge = dict(Action = 'Rotate', Displacement = 
                            list(compute_relative_rotation(current_axs, paper_edge_dirc)))
    env.step(action_rotate_axis_parrell_edge)