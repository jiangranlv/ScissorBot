import math
from typing import List
import hydra
from omegaconf import DictConfig, OmegaConf
import sys 
import os
from PIL import Image, ImageDraw
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__) ))
sys.path.append(proj_root)
sys.path.append(os.path.join(proj_root, 'policy'))

from src.maths import compute_relative_rotation
import numpy as np
import yaml
from tqdm import tqdm
from cutgym import gym
import random

# from debug_tool import timing_decorator

@hydra.main(config_path="../config", config_name='generate_demos', version_base='1.3')
def main(gym_cfg: DictConfig):
    simulation_cfg = OmegaConf.load(proj_root+ '/config/paper_cutting_game_realworld_fast.yaml')
    simulation_cfg = OmegaConf.merge(simulation_cfg, gym_cfg) 
    env = gym(simulation_cfg, gym_cfg)
    batch_size = env.batch_size

    max_demos = 8 # TODO: increase this
    env.reset(init= True)
    for i in tqdm(range(max_demos)):
        env.log.info(f"this is the {i} th trajectory")

        if gym_cfg.demos.constraints == 'top_edge':
            fix_top_edge(env)
        elif gym_cfg.demos.constraints == 'square_pad':
            fix_square_pad(env)
        else:
            raise NotImplementedError()
        
        os.mkdir(str(i))
        os.chdir(str(i))

        batch_dirs: List[str] = []
        for batch_idx in range(env.batch_size):
            batch_dir = f"batch_{str(batch_idx).zfill(len(str(env.batch_size - 1)))}"
            os.makedirs(batch_dir)
            batch_dirs.append(batch_dir)
        env.set_batch_dirs(batch_dirs)

        goal_edge_set_batch = [
            random_generate_goal(os.path.join(batch_dirs[batch_idx], str(i)), gym_cfg.demos.num_lines, gym_cfg.demos.start.x, gym_cfg.demos.start.y,
                     gym_cfg.demos.len_range, gym_cfg.demos.init_angle, gym_cfg.demos.angle_slot)
            for batch_idx in range(batch_size)
        ]

        env.goal_edge_set_batch = goal_edge_set_batch
        heuristic_w_pre_cut(env, goal_edge_set_batch)
        env.reset(init= False)
        os.chdir('..')

def monotonic_curve(num_lines , start_x, start_y , len_range , init_angle, angle_slot, **kwargs):
    direction = np.random.choice([1, -1])
    init_lower = init_angle[0] * direction
    init_upper = init_angle[-1] * direction
    x_list = [start_x]
    y_list = [start_y]
    total_angle = random.uniform(init_lower, init_upper)
    for _ in range(num_lines):
        length = random.uniform(len_range[0], len_range[-1])

        x = x_list[-1] + length * math.cos(math.radians(total_angle))
        y = y_list[-1] + length * math.sin(math.radians(total_angle))
        random_angle = random.uniform(0.5* angle_slot, 1.5 *angle_slot)
        total_angle += random_angle * direction
        x_list.append(x)
        y_list.append(y)

    return x_list, y_list

def two_peak_curve(num_lines , start_x, start_y , len_range , init_angle, angle_slot, **kwargs):
    direction = np.random.choice([1, -1])
    init_lower = init_angle[0] * direction
    init_upper = init_angle[-1] * direction
    x_list = [start_x]
    y_list = [start_y]
    total_angle = random.uniform(init_lower, init_upper)
    for _ in range(int(num_lines/ 2)):
        length = random.uniform(len_range[0], len_range[-1])

        x = x_list[-1] + length * math.cos(math.radians(total_angle))
        y = y_list[-1] + length * math.sin(math.radians(total_angle))
        random_angle = random.uniform(0.5* angle_slot, 1.5 *angle_slot)
        total_angle += random_angle * direction
        x_list.append(x)
        y_list.append(y)

    for _ in range(int(num_lines/ 2)):
        length = random.uniform(len_range[0], len_range[-1])

        x = x_list[-1] + length * math.cos(math.radians(total_angle))
        y = y_list[-1] + length * math.sin(math.radians(total_angle))
        random_angle = random.uniform(0.5* angle_slot, 1.5 *angle_slot)
        total_angle += random_angle * direction * (-1)
        x_list.append(x)
        y_list.append(y)

    return x_list, y_list

def multi_peak_curve(num_lines, start_x, start_y , len_range , init_angle, angle_slot, num_peak, **kwargs):
    direction = np.random.choice([1, -1])
    init_lower = init_angle[0] * direction
    init_upper = init_angle[-1] * direction
    x_list = [start_x]
    y_list = [start_y]
    total_angle = random.uniform(init_lower, init_upper)
    for i in range(num_peak):
        direction *= -1
        for _ in range(int(num_lines/ num_peak)):
            length = random.uniform(len_range[0], len_range[-1])

            x = x_list[-1] + length * math.cos(math.radians(total_angle))
            y = y_list[-1] + length * math.sin(math.radians(total_angle))
            random_angle = random.uniform(0.5* angle_slot, 1.5 *angle_slot)
            total_angle += random_angle * direction
            x_list.append(x)
            y_list.append(y)

    return x_list, y_list

def random_generate_goal(save_name, curve_cfg):
    PAPER_LENGTH = np.array([0.210, 0.297])
    # PAPER_LENGTH = np.array([0.5, 0.5])

    def generate_trajctory(x_values, z_values):    
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
    
    if curve_cfg.type == 'monotonic' :
        curve_func = monotonic_curve 
    elif curve_cfg.type == 'two_peak':
        curve_func = two_peak_curve
    elif curve_cfg.type == 'multi_peak':
        curve_func = multi_peak_curve
    else:
        raise NotImplementedError(curve_cfg.type)


    edges = generate_trajctory(*curve_func(**curve_cfg))
    generate_texture_img(edges)
    
    return edges   

def heuristic_w_pre_cut(env:gym, goal_edge_set_batch):
    pre_cut(env)
    rule_based_cutting(env, goal_edge_set_batch)


def rule_based_cutting(env:gym, goal_edge_set_batch):
    detach_flag_batch = [False for i in range(env.batch_size)]
    inter_goal_rest_all_step = np.array(goal_edge_set_batch).transpose([1, 0, 2, 3])[:, : ,1, :] # t, b, 3 
    for i, inter_goal_rest in enumerate(inter_goal_rest_all_step):
        
        # rotate towards A
        rotate_given_target_point(env, inter_goal_rest, detach_flag_batch)
        # close the scissor to cut 
        close_scissor(env, inter_goal_rest, detach_flag_batch)

        for batch_idx in range(env.batch_size):
            env.completed_batch[batch_idx].set(i)
        
        # open the scissor
        open_to_max_angle(env, detach_flag_batch)
        
        if i == inter_goal_rest_all_step.shape[0]- 1:
            break

        # ensure the detach after cutting
        if i < inter_goal_rest_all_step.shape[0]- 2:
            next_goal_rest= inter_goal_rest_all_step[i + 1]
            next_next_goal_rest= inter_goal_rest_all_step[i + 2]
            detach_flag_batch = check_detach_batch(env, next_goal_rest, next_next_goal_rest, detach_flag_batch)

        # rotate towards B
        rotate_given_target_point(env, next_goal_rest, detach_flag_batch)

        # move to B
        move_scissor(env, next_goal_rest, detach_flag_batch)
        
        
def normalize(vec):
    return vec / np.linalg.norm(vec)

# @timing_decorator
def open_to_max_angle(env:gym, detach_flag_batch= None):
    scissors_pose = env.get_scissors_pose()
    action_open_batch = []
    for batch_idx in range(env.batch_size):
        current_angle = scissors_pose[batch_idx]["joint_0"]
        if detach_flag_batch is None or not detach_flag_batch[batch_idx]:
            action_open_batch.append(dict(Action = 'Open', Angle = max(env.MAXIMUM_ANGLE - current_angle, 0.0)))
        else:
            action_open_batch.append(dict(Action = 'None'))
    env.step(action_open_batch)

    return env.get_state(), action_open_batch

# @timing_decorator
def rotate_given_target_point(env:gym, target_rest, detach_flag_batch = None):
    # rotate the scissor to direction of current->A
    current_pos_batch = env.get_scissors_front_point()
    target_dirc_batch = []
    for batch_idx in range(env.batch_size):
        target_point = env.get_current_pos_given_rest(batch_idx, target_rest[batch_idx])
        target_dirc_batch.append(target_point - current_pos_batch[batch_idx])
    state,action = rotate_given_target_dirc(env, target_dirc_batch, detach_flag_batch)
    return state, action

def rotate_given_target_dirc(env:gym, target_dirc, detach_flag_batch = None):
    action_rotate_c2a_batch = []
    for batch_idx in range(env.batch_size):
        pos, current_direction, current_axs = env.get_scissors_cut_direction(batch_idx)
        angle = compute_relative_rotation(current_direction, target_dirc[batch_idx])
        if detach_flag_batch is None or not detach_flag_batch[batch_idx]:
            action_rotate_c2a_batch.append(dict(Action = 'Rotate', Displacement = list(angle if angle is not None else [0., 0., 0.])))
        else:
            action_rotate_c2a_batch.append(dict(Action = 'None'))
    env.step(action_rotate_c2a_batch)
    return env.get_state(), action_rotate_c2a_batch

# @timing_decorator
def close_scissor(env:gym, target_rest, detach_flag_batch = None):
    current_pos_batch = env.get_scissors_front_point()
    close_action_batch = []
    for batch_idx in range(env.batch_size):
        current = current_pos_batch[batch_idx]
        final_goal = env.get_current_pos_given_rest(batch_idx, target_rest[batch_idx])
        if detach_flag_batch is None or not detach_flag_batch[batch_idx]:
            close_action_batch.append(dict(Action = 'Close', 
                                        Angle = abs(env.compute_angle_from_current(batch_idx, math.dist(current, final_goal)))))
        else:
            close_action_batch.append(dict(Action = 'None'))
    env.step(close_action_batch)

    return env.get_state(), close_action_batch

# @timing_decorator
def move_scissor(env:gym, target_rest, detach_flag_batch = None):
    current_pos_batch = env.get_scissors_front_point()
    action_translate_batch = []
    for batch_idx in range(env.batch_size):
        target_point = env.get_current_pos_given_rest(batch_idx, target_rest[batch_idx])
        if detach_flag_batch is None or not detach_flag_batch[batch_idx]:
            action_translate_batch.append(dict(Action = 'Translate', Displacement = (target_point - current_pos_batch[batch_idx]).tolist()))
        else:
            action_translate_batch.append(dict(Action = 'None'))
    env.step(action_translate_batch)
    # final_pos = env.get_scissors_front_point()
    # if np.linalg.norm(final_pos - end) / np.linalg.norm(end - start)  > 0.1:
    #     print(f'scissor may be stuck in state {env.state_num -1}')
    return env.get_state(), action_translate_batch    

def safe_move(env:gym, start_batch: List[np.ndarray], end_batch: List[np.ndarray]):
    action_translate1_batch = []
    for batch_idx in range(env.batch_size):
        dis = (end_batch[batch_idx] - start_batch[batch_idx]).tolist()
        vert_dis = [0, dis[1], 0]
        action_translate1_batch.append(dict(Action = 'Translate', Displacement = vert_dis))

    env.step(action_translate1_batch, warn_info= False)
    state_list = []
    state_list.append(env.get_state())
    
    action_translate2_batch = []
    for batch_idx in range(env.batch_size):
        dis = (end_batch[batch_idx] - start_batch[batch_idx]).tolist()
        hori_dis = [dis[0], 0, dis[2]]
        action_translate2_batch.append(dict(Action = 'Translate', Displacement = hori_dis))
    env.step(action_translate2_batch, warn_info= False)
    # state_list.append(env.get_state()) if save_state else None
    
    return state_list, [action_translate1_batch, action_translate2_batch]
    
    
def pre_cut(env:gym):
    a_rest = [goal_edge_set[0][0] for goal_edge_set in env.goal_edge_set_batch]
    b_rest = [goal_edge_set[0][1] for goal_edge_set in env.goal_edge_set_batch]

    state_list = [env.get_state()] 
    action_list = []
    
    # first open the scissor
    if env.get_scissors_pose()[0]["joint_0"] < env.MAXIMUM_ANGLE:
        state_open, action_open = open_to_max_angle(env)
    state_list.append(state_open)
    action_list.append(action_open)
    
    #rotate to vertical
    action_rotate_axis_parrell_edge_batch = []
    for batch_idx in range(env.batch_size):
        pos, current_direction, current_axs = env.get_scissors_cut_direction(batch_idx)
        paper_edge_uv_a = np.array([0 , 0, a_rest[batch_idx][2]- 0.015])
        paper_edge_uv_b = np.array([0 , 0, a_rest[batch_idx][2]+ 0.015])
        paper_edge_dirc = env.get_current_pos_given_rest(batch_idx, paper_edge_uv_b) \
                        - env.get_current_pos_given_rest(batch_idx, paper_edge_uv_a)
        action_rotate_axis_parrell_edge_batch.append(dict(Action = 'Rotate', Displacement = 
                                list(compute_relative_rotation(current_axs, paper_edge_dirc))))
    env.step(action_rotate_axis_parrell_edge_batch)
    state_list.append(env.get_state())
    action_list.append(action_rotate_axis_parrell_edge_batch)
    
    # rotate the scissor to direction of A->B 
    dirc_a2b = [(env.get_current_pos_given_rest(batch_idx, b_rest[batch_idx]) - env.get_current_pos_given_rest(batch_idx, a_rest[batch_idx]))
                                for batch_idx in range(env.batch_size)]
    state_rot_a2b, action_rot_a2b = rotate_given_target_dirc(env, dirc_a2b)
    state_list.append(state_rot_a2b)
    action_list.append(action_rot_a2b)

    # move to A
    start_point_batch = []
    current_pos_batch = env.get_scissors_front_point()
    for batch_idx in range(env.batch_size):
        start_point_batch.append(env.get_current_pos_given_rest(batch_idx, env.goal_edge_set_batch[batch_idx][0][0]))
    state_move_list, action_move_list = safe_move(env, current_pos_batch, start_point_batch)
    state_list+= state_move_list
    action_list+= action_move_list
    
    return state_list, action_list
    

def fix_square_pad(env:gym):
    def vid_to_old_pos(vid: int) -> np.ndarray:
        i, k = vid // 31, vid % 31
        return np.array([i / 21 * 0.210, 0, k / 30 * 0.297], dtype=float)
    
    vid_list = [10 * 31 + 30, 11 * 31 + 30, 10 * 31 + 29, 11 * 31 + 29]   
    constraint =  dict()  
    for vid in vid_list:
        constraint[vid] = vid_to_old_pos(vid)
    for batch_idx in range(env.batch_size):
        env.append_constraints(batch_idx, constraint, 30.0)    

def fix_top_edge(env:gym):
    def vid_to_old_pos(vid: int) -> np.ndarray:
        i, k = vid // 31, vid % 31
        return np.array([i / 21 * 0.210, 0, k / 30 * 0.297], dtype=float)
    
    constraint =  dict() 
    for top_vertex in range(22):
        vid = top_vertex * 31 + 30
        constraint.update({vid: vid_to_old_pos(vid)})
    for batch_idx in range(env.batch_size):
        env.append_constraints(batch_idx, constraint, 30.0)   

def fix_top_edge_50(env:gym):
    def vid_to_old_pos(vid: int) -> np.ndarray:
            i, k = vid // 51, vid % 51
            return np.array([i / 50 * 0.5, 0, k / 50 * 0.5], dtype=float)

    constraint =  dict() 
    for top_vertex in range(50):
        vid = top_vertex * 51 + 50
        constraint.update({vid: vid_to_old_pos(vid)})
    for batch_idx in range(env.batch_size):
        env.append_constraints(batch_idx, constraint, 130.0)  

def fix_2_corner(env:gym):
    env.append_constraints({
            30: np.array([0.0, 0.0, 0.297]),
            681: np.array([0.210, 0.0, 0.297]),
                }, 1.0)   

# @timing_decorator
def check_detach_batch(env:gym, a_rest, b_rest, detach_flag_batch):
    for batch_idx in range(env.batch_size):
        if not detach_flag_batch[batch_idx]:
            if check_detached(env, batch_idx, a_rest[batch_idx], b_rest[batch_idx]):
                detach_flag_batch[batch_idx] = True
                os.mknod(os.path.join(env.batch_dirs[batch_idx], "detached.txt"))
                print('Detached Triangle During Simulation')
    return detach_flag_batch

def check_detached(env:gym, batch_idx: int, uv_start: np.ndarray, uv_target):
    num_samples = 5
    thres = 3
    uv_mid = uv_start + 0.5* (uv_target - uv_start)
    samples_x = np.linspace(uv_mid[0], uv_target[0], num_samples)
    samples_y = np.linspace(uv_mid[1], uv_target[1], num_samples)
    samples_z = np.linspace(uv_mid[2], uv_target[2], num_samples)
    samples = np.column_stack((samples_x, samples_y, samples_z))
    deformation = np.zeros(num_samples -1)
    for i in range(1, num_samples):
        deformation[i-1] = compute_deformation(env, batch_idx, uv_start, samples[i])
    overall_deform = remove_max_and_mean(deformation)
    if overall_deform > thres:
        return True
    return False

def compute_deformation(env:gym, batch_idx, uv1, uv2):
    pos1 = env.get_current_pos_given_rest(batch_idx, uv1)
    pos2 = env.get_current_pos_given_rest(batch_idx, uv2)
    return np.linalg.norm(pos1 - pos2) / np.linalg.norm(uv1 - uv2)    

def remove_max_and_mean(array):
    max_value = np.max(array)
    filtered_array = array[array < max_value]
    result = np.mean(filtered_array)
    return result

def rotate_vertical_with_paper(env:gym, start_uv, end_uv):
    raise NotImplementedError
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
    
if __name__ == "__main__":
    main()
