import pathlib
import math
from typing import List, Dict, Tuple, Union
from pprint import pprint
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import copy
import os
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__) ))
sys.path.append(proj_root)

from src.maths import *
import yaml

from cutgym import gym
# def uv_func(x: np.ndarray) -> np.ndarray:
#         return x[:, 0:2] / 0.3
def uv_func(x):
    return (x / np.array([0.210, 1.0, 0.297]))[:, [0, 2]]

@hydra.main(config_path="../config", config_name='cutgym', version_base='1.3')
def main(gym_cfg: DictConfig):
    simulation_cfg = OmegaConf.load(proj_root+ '/config/paper_cutting_game_fast.yaml')
    simulation_cfg = OmegaConf.merge(simulation_cfg, gym_cfg) 
    # simulation_cfg.cloth.cloth_file = "./assets/simple30cm_15mm.obj"
    simulation_cfg.cloth.cloth_file = "./assets/vertical_a4_2_10mm.obj"
    simulation_cfg.env.cut_sim.position_bounds = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    env = gym(simulation_cfg, gym_cfg)

    goal_edge_set = env.goal_edge_set

    env.reset(init= True)
    MINIMUM_ANGLE, MAXIMUM_ANGLE = (env.MINIMUM_ANGLE, env.MAXIMUM_ANGLE)
    MAX_CUT_LENGTH = env.MAXIMUM_ANGLE
    
    # first holding the paper
    # hold_paper(env)
    # fix_4_corner_points(env)
    mat = env.get_robot_base_cfg()
    mat[:3, 3] = np.array([-0.3, 0.3, -1.0]) # (-0.5, 0.0, -1.0)
    env.set_robot_base_cfg(mat)
    # rotate2vertical(env)
    fix_square_pad(env)
    for i in range(len(goal_edge_set)):
    
        # if i == len(goal_edge_set) //2:
        #     print(env.get_scissors_front_point())
        #     move_holding_space(env= env, relative_motion= 
        #                        np.array([0, 0, env.get_scissors_front_point()[2] - 0]))
            
        print('\n','=' * 30, f'Cutting Edge No.{i+1}', '=' * 30)
        if i == 0 :
            pre_cut(env)
        else:
            start_point = env.get_current_pos_given_rest(goal_edge_set[i][0])
        
            scissors_pose = env.get_scissors_pose()
            current_angle = scissors_pose[0]["joint_0"]
            # angle, theta, phi = scissors_pose[0]["joint_1"][3:]
            
            # first open the scissor
            if current_angle < 0:
                action_open = dict(Action = 'Open', Angle = MAXIMUM_ANGLE - current_angle)
                env.step(action_open)
            
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

def open_to_max_angle(env:gym):
    scissors_pose = env.get_scissors_pose()
    current_angle = scissors_pose[0]["joint_0"]
    if current_angle < env.MAXIMUM_ANGLE:
        action_open = dict(Action = 'Open', Angle = env.MAXIMUM_ANGLE - current_angle)
        env.step(action_open)
        
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
            return 
        last_dis = curr_dis
    env.step(dict(Action = 'Close', Angle = abs(env.compute_angle_from_current(math.dist(current, final_goal)))))

def move_scissor(env:gym, start, end):
    action_translate = dict(Action = 'Translate', Displacement = (end - start).tolist())
    env.step(action_translate)
                            

def safe_move(env:gym, start, end):
    dis = (end - start).tolist()
    vert_dis = [0, dis[1], 0]
    action_translate = dict(Action = 'Translate', Displacement = vert_dis)
    env.step(action_translate)
    
    hori_dis = [dis[0], 0, dis[2]]
    action_translate = dict(Action = 'Translate', Displacement = hori_dis)
    env.step(action_translate)
    
    # vert_dis = [0, 0.015, 0 ]
    # action_translate = dict(Action = 'Translate', Displacement = vert_dis)
    # env.step(action_translate)
    
def pre_cut(env:gym):
    start_point = env.get_current_pos_given_rest(env.goal_edge_set[0][0])
    scissors_pose = env.get_scissors_pose()
    current_angle = scissors_pose[0]["joint_0"]
    
    # first open the scissor
    if current_angle < 0:
        open_to_max_angle(env)
    
    pos, current_direction, current_axs = env.get_scissors_cut_direction()[0]
    paper_edge_uv_a = np.array([0 , 0, env.goal_edge_set[0][0][2]- 0.015])
    paper_edge_uv_b = np.array([0 , 0, env.goal_edge_set[0][0][2]+ 0.015])
    paper_edge_dirc = env.get_current_pos_given_rest(paper_edge_uv_b) \
                                - env.get_current_pos_given_rest(paper_edge_uv_a)
    action_rotate_axis_parrell_edge = dict(Action = 'Rotate', Displacement = 
                            list(compute_relative_rotation(current_axs, paper_edge_dirc)))
    env.step(action_rotate_axis_parrell_edge)
    
    # rotate the scissor to direction of A->B 
    pos, current_direction, current_axs = env.get_scissors_cut_direction()[0]
    end_point = env.get_current_pos_given_rest(env.goal_edge_set[0][1])
    current_pos = env.get_scissors_front_point()
    action_rotate_a2b = dict(Action = 'Rotate', Displacement = 
                            list(compute_relative_rotation(current_direction, (end_point - start_point))))
    env.step(action_rotate_a2b)
    
    # move to A
    start_point = env.get_current_pos_given_rest(env.goal_edge_set[0][0])
    current_pos = env.get_scissors_front_point()
    safe_move(env, current_pos, start_point)

    
def hold_paper(env:gym, center_pos = None):
    c0 = {}
    for i in range(17, 21):
        for j in range(8, 13):
            y = 0.015 * i
            x = 0.015 * j
            z = 0.0
            c0[21*j+i] = np.array([x, y, z])

    c1 = {}
    r = 0.015 / 2 / math.cos(67.5 * math.pi / 180)
    for i in range(17, 21):
        for j in range(8, 13):
            theta = (j - 2) * math.pi / 4
            y = 0.015 * i
            x = (8+ (13 - 8)// 2)* 0.015 + r * math.sin(theta)
            z = r * (1 - math.cos(theta))
            c1[21*j+i] = np.array([x, y, z])

    c1_plus = copy.deepcopy(c1)
    c1_plus[10*21+0] = np.array([0.150, 0.0, 0.150])
    # c1_plus[0] = np.array([0.0, 0.0, 0.150])
    # c1_plus[20*21] = np.array([0.150, 0.0, 0.150])
    
    env.append_constraints(
        old_constraints=c0,
        new_constraints=c1,
        constrain_time=0.5
    )
    env.simulate(0.5, uv_func =uv_func)  

    env.append_constraints(
        old_constraints=c1,
        new_constraints=c1_plus,
        constrain_time=2.0
    )
    env.simulate(2.0, uv_func =uv_func)  
    env.append_constraints(
        old_constraints=c1_plus,
        new_constraints=c1,
        constrain_time=0.5
    )
    
    env.simulate(0.5, uv_func =uv_func)   

def fix_4_corner_points(env:gym):
    env.append_constraints({
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([0.3, 0.0, 0.0]),
            2: np.array([0.3, 0.3, 0.0]),
            3: np.array([0.0, 0.3, 0.0]),
                }, 1.0)

def move_holding_space(env:gym, relative_motion):
    old_holding_space = env._get_cloth_constraint()   
    updated_holding_space = {}
    for holding_point in old_holding_space:
        updated_holding_space[holding_point] = old_holding_space[holding_point] + relative_motion
    
    env.append_constraints(
        old_constraints= old_holding_space, 
        new_constraints=updated_holding_space, 
        constrain_time= 1)
    env.simulate(1, uv_func =uv_func)
    
def merge_dict(dict1, dict2):
    """
    
    """
    for key in dict2:
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            merge_dict(dict1[key], dict2[key])
        else:
            dict1[key] = dict2[key]
    return dict1

def rot_y(x: np.ndarray, theta: float, center: np.ndarray) -> np.ndarray:
        direction = np.array([0., 1., 0.], dtype=float)
        mat = tra.rotation_matrix(theta, direction, center)
        return (mat @ point_to_homo(x))[:3]

def vid_to_old_pos(vid: int) -> np.ndarray:
    i, k = vid // 22, vid % 22
    return np.array([i / 30 * 0.297, 0, k / 21 * 0.210], dtype=float)

def theta_to_constraints(theta: float, center: np.ndarray, vid_list: List[int]) -> Dict[int, np.ndarray]:
    return {vid: rot_y(vid_to_old_pos(vid), theta, center) for vid in vid_list}

def get_rot_center() -> np.ndarray:
    return np.array([0.5 / 30 * 0.297, 0, 10.5 / 21 * 0.210], dtype=float)

def theta_list_to_constraints(theta_list: list) -> List[dict]:
    vid_list = [10, 11, 10+22, 11+22]
    all_constraints = []
    for i in range(len(theta_list) - 1):
        theta1 = theta_list[i]
        theta2 = theta_list[i + 1]
        all_constraints.append((
            theta_to_constraints(theta1, get_rot_center(), vid_list),
            theta_to_constraints(theta2, get_rot_center(), vid_list)))
    return all_constraints

def rotate2vertical(env):
    num_rotate = 200
    theta_lists = [smooth_interp(0.0, math.pi/4, num_rotate).tolist(),
                   smooth_interp(math.pi/4, math.pi/2,num_rotate).tolist()]
    for theta_list in theta_lists:
        cons_list = theta_list_to_constraints(theta_list)
        for old_cons, new_cons in cons_list:
            env.append_constraints(new_cons, 0.0, old_cons)
        env.simulate(2.0, uv_func =uv_func)  

def fix_2_corner(env:gym):
    env.append_constraints({
            30: np.array([0.0, 0.0, 0.297]),
            681: np.array([0.210, 0.0, 0.297]),
                }, 1.0)   
    
def fix_square_pad(env:gym):
    def vid_to_old_pos(vid: int) -> np.ndarray:
        i, k = vid // 31, vid % 31
        return np.array([i / 30 * 0.297, 0, k / 21 * 0.210], dtype=float)
    
    vid_list = [10* 30, 11 * 30, 10*30 -1, 11 * 30 -1]   
    constraint =  dict()  
    for vid in vid_list:
        constraint[vid] = vid_to_old_pos(vid)
    env.append_constraints(constraint, 1.0)    

if __name__ == "__main__":
    main()
