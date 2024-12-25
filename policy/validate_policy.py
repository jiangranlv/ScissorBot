from omegaconf import DictConfig, OmegaConf
import sys
import copy
import os
import shutil
import random
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__) ))
sys.path.append(proj_root)
sys.path.append(os.path.join(proj_root, 'policy'))
import omegaconf

from src.maths import *
import yaml
from tqdm import tqdm
from cutgym import gym

from generate_demos import random_generate_goal, pre_cut, fix_square_pad, fix_top_edge

import pickle
import trimesh.transformations as tra

import numpy as np

import argparse
from src.sapien_renderer import SapienRender
import sapien.core as sapien
from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset
from rl.pytorch_utils import to_numpy, ActionLookupTable, ActionLookupTable_for16D
# from rl.behavior_clone import encode_action_4D_to_8D
from rl.prepare_dataset import rotate2quat_numpy, default_action_dict_to_4D
# from rl.behavior_clone import PointNetBC
from rl.bc_beta import PointNetTransformerBC

POINT_DIM = 4
POINT_RES = 512
def compress_pc(pc: np.ndarray, fp: np.ndarray, n_sample: int):
    assert pc.shape == (4096, 4) and fp.shape == (3, ), f'{pc.shape}, {fp.shape}'
    dist = np.linalg.norm(pc[:, :3] - fp[np.newaxis, :], axis=1)
    argsorted = np.argsort(dist)
    select_pc = pc[argsorted[:n_sample], :]
    return select_pc

class TestDataLoader(Dataset):
    def __init__(self, seq_len, pose_dim) -> None:
        super().__init__()
        self.pc_buffer = []
        self.pose_buffer = []
        self._seq_len = seq_len
        self.pose_dim = pose_dim
        self.buffer_len = 0
        self.action_buffer = []
        self.debug_pc_buffer = []
        
    def __getitem__(self, idx):
        pc_seq = np.zeros((self._seq_len, POINT_RES, POINT_DIM), np.float32) # [T, POINT_RES, POINT_DIM]
        pose_seq = np.zeros((self._seq_len, self.pose_dim), np.float32) # [T, 8] Rotate = quaternion

        for prev_data_index in range(idx , idx - self._seq_len, -1):
            idx_in_seq = prev_data_index - (idx - self._seq_len + 1)
            if 0 <= prev_data_index:
                pc_seq[idx_in_seq, :] = self.pc_buffer[prev_data_index]
                pose_seq[idx_in_seq, :] = self.pose_buffer[prev_data_index]
            elif idx_in_seq < self._seq_len - 1:
                pc_seq[idx_in_seq, ...] = pc_seq[idx_in_seq + 1, ...].copy()
                pose_seq[idx_in_seq, ...] = pose_seq[idx_in_seq + 1, ...].copy()
            else:
                raise RuntimeError(f"Unexpected idx_in_seq {idx_in_seq}, prev_data_index {prev_data_index}")
        
        # sanity check
        assert pc_seq.shape == (self._seq_len, POINT_RES, POINT_DIM), f"{pc_seq.shape}"
        assert pose_seq.shape == (self._seq_len,self.pose_dim), f"{pose_seq.shape}"

        assert False not in np.isfinite(pc_seq)
        assert False not in np.isfinite(pose_seq)

        return pc_seq, pose_seq
    
    def add_data(self, pc, pose):
        self.pc_buffer.append(pc)
        self.pose_buffer.append(pose)
        self.buffer_len +=1
    
    def get_data(self):
        return self.pc_buffer[-1], self.pose_buffer[-1]
    
    def reset(self):
        self.pc_buffer = []
        self.pose_buffer = []
        self.buffer_len = 0
        self.action_buffer = []
        self.debug_pc_buffer = []

class Render():
    def __init__(self, gpuid, width_height = (512, 512),
                   ee_urdf_path= "assets/endeffector/clipper_body/urdf/clipper_no_limit_floating.urdf"
                   ) -> None:
        print(f"Using GPU:{gpuid} for render observation")
        self.engine = SapienRender()
        self.engine.build_scene(robot_urdf_path=None, ee_urdf_path= ee_urdf_path, width_height= width_height)

    def __call__(self, state, texture_file, output_type = 'pc', compress = False, front_point = None, crop_around_fp = False, center_fp = False,
                 jittor_pc = True, jittor_blade= True, highlight_seg = False, resample = True, num_sample = 512, visualize = False):
        
        camera_pose = self.random_camera_pose(euler =  [0.0, np.pi * 0.125, -np.pi * 0.5], trans = [0.05, 0.18, 0.32])
        tmp_obj_file = f"tmp_{os.getpid()}_" + ''.join(random.choices('0123456789', k=6)) + '.obj'
        self.engine.set_scene(state, camera_pose, tmp_filename= tmp_obj_file, texture_file= texture_file)
        calculate_rgba = True if output_type == 'rgb' else False
        render_result = self.engine.render(calculate_rgba=calculate_rgba, get_segment= True)

        if output_type == 'pc':
            pc_position, pc_color, pc_segment = render_result['pc_position'],render_result['pc_color'], render_result['pc_segment']
            if highlight_seg:
                pc_color = self.highlight_pc_seg(pc_color, pc_segment)
            if crop_around_fp:
                box = []
                box.append([front_point[0] - 0.02, front_point[0] + 0.05])
                box.append([front_point[1] - 0.02, front_point[1] + 0.02])
                box.append([front_point[2] - 0.03, front_point[2] + 0.03])
                _, index = self.crop_box(pc_position, box)
                pc_position = pc_position[index]
                pc_color = pc_color[index]
                pc_segment = pc_segment[index]
            # if jittor_pc:
            #     pc_position = self.jittor_pc(pc_position, noise_factor= 0.004)
            if jittor_blade:
                pc_position = self.jittor_blade(pc_position, pc_segment)
            if center_fp:
                pc_position -= front_point
            if resample:
                ret_pc = self.resample_pc(pc_position, pc_color, num_sample= num_sample, use_rgb_cutline=visualize) 
            # sanity check
                # if compress:
                #     self._point_cloud_number = 512
                #     ret_pc = compress_pc(ret_pc, front_point, n_sample= self._point_cloud_number)
                #     ret_pc[:, :3] -= front_point
                
            else:
                ret_pc = np.concatenate([pc_position, pc_color], axis = 1)    
            # assert ret_pc.shape == (self._point_cloud_number, 4)
            return ret_pc
        elif output_type == 'rgb':
            return render_result['rgba'], render_result['segment']

    def random_camera_pose(self, euler =  [0.0, np.pi * 0.125, -np.pi * 0.5], trans = [0.05, 0.18, 0.32], 
                           random_euler = [0.05, 0.05, 0.05], random_trans = [0.03, 0.03, 0.03]):
        camera_euler = np.array(euler) + 2 * (np.random.random(size=1) - 0.5) * np.array(random_euler)
        camera_trans = np.array(trans) + 2 * (np.random.random(size=1) - 0.5) * np.array(random_trans)
        camera_pose = sapien.Pose(camera_trans, tra.quaternion_from_euler(*camera_euler))
        return camera_pose
    
    def highlight_pc_seg(self, pc_color, pc_segment, obj_id = 6, color = [0, 255, 127]):
        pc_color[pc_segment == obj_id] = np.array(color)/ 256
        return pc_color

    def resample_pc(self, pc_position, pc_color, num_sample, use_rgb_cutline = False):
        import open3d as o3d
        
        self._point_cloud_number=num_sample
        self._cut_line_max_number=128
        self._where_cut_line=lambda x: x[..., 0] > (x[..., 1] + x[..., 2]) * 3  
               
        pc_cat = np.concatenate([pc_position, pc_color], axis=1)
        if pc_cat.shape[0] < self._point_cloud_number:
            print(
                f"Not enough points in the view: {pc_cat.shape[0]} < {self._point_cloud_number}")
            return None

        if use_rgb_cutline:
            cut_line = pc_cat[self._where_cut_line(pc_color), :]
        else:
            cut_line = pc_cat[self._where_cut_line(pc_color), :]
            cut_line[:, 3:] = np.array([1.0, 0.0, 0.0])
        
        if cut_line.shape[0] > self._cut_line_max_number:
            cut_line = np.random.permutation(
                cut_line)[:self._cut_line_max_number, :]
            
        other_point = pc_cat[~self._where_cut_line(pc_color), :]
        # other_point = np.random.permutation(other_point)[:self._point_cloud_number - cut_line.shape[0], :]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(other_point[..., :3])
        if use_rgb_cutline:
            pcd.colors = o3d.utility.Vector3dVector(other_point[..., 3:])
        else:
            pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(other_point[..., 3:]))

        pcd_down = pcd.farthest_point_down_sample(
            self._point_cloud_number - cut_line.shape[0])
        other_point = np.concatenate(
            [np.asarray(pcd_down.points), np.asarray(pcd_down.colors)], axis=1)

        if use_rgb_cutline:
            ret_pc = np.concatenate([cut_line, other_point], axis=0, dtype=np.float32) # point cloud
        else:
            ret_pc = np.concatenate([cut_line, other_point], axis=0, dtype=np.float32)[:, :4] # point cloud
        
        if ret_pc.shape != (self._point_cloud_number, 6 if use_rgb_cutline else 4 ):
            print('Not legal Shape of return pc', ret_pc.shape)
            return None
        return ret_pc         
    
    def jittor_blade(self, pc, pc_seg, obj_id = 6):
        def add_noise_to_point_cloud(points, noise_factor= 0.003, direction = np.array([0,-1,0])):
            noise = abs(torch.rand(1))
            try:
                weight = (points[:, 0] / (points[:, 0].max() - points[:, 0].min())).reshape(-1, 1)
            except:
                raise ValueError(points.shape)
            noisy_points = points + noise_factor * (weight +noise).repeat(1, 3) * direction
            return noisy_points
        obj_index = (pc_seg ==obj_id)
        if np.count_nonzero(obj_index) > 0 :
            obj_pc = pc[obj_index]
            noise_factor = abs(torch.rand(1)) * 0.002 + 0.002
            pc[obj_index] = add_noise_to_point_cloud(obj_pc, noise_factor= noise_factor)
        return pc
    
    def jittor_pc(self, pc, noise_factor= 0.002):
        direction = torch.rand(3)
        direction = direction / torch.norm(direction)
        noise = torch.rand(1)* noise_factor * direction
        noisy_points = pc + noise
        return noisy_points
        
        
    
    def crop_box(self, pc, box):
        x_min, x_max = box[0]
        y_min, y_max = box[1]
        z_min, z_max = box[2]
        index = (pc[:, 0] > x_min) & (pc[:, 0] <x_max) &(pc[:, 1] >y_min) &(pc[:, 1] <y_max) & (pc[:, 2] > z_min) & (pc[:, 2] < z_max)
        # print(f'number of points left {np.count_nonzero(index)} After Crop')
        
        crop_box_pc = pc[index]
        return crop_box_pc, index
    
class GymStateTool():
    def __init__(self, gpuid, width_height=(512, 512)) -> None:
        self.texture_file = None
        self.render = Render(gpuid, width_height)
    
    def get_state_obs(self, state, crop_point = None):
        pc = self.render(state, texture_file= self.texture_file, output_type='pc', jittor_blade= True, highlight_seg= False, 
                      crop_around_fp= True, front_point= crop_point, center_fp= True,
                      resample= True, num_sample = 512, visualize= False)
        pose = np.concatenate([state['front_point'], state['cut_direction']])
        return pc, pose
    
    def get_whole_pc(self, state):
        pc = self.render(state, texture_file= self.texture_file, output_type='pc', jittor_blade= True, highlight_seg= False, 
                      crop_around_fp= False, front_point= None, center_fp= False,
                      resample= False, num_sample = None, visualize= False)
        return pc
    
class TestEnv():
    def __init__(self, gpuid, log_dir, gym_cfg, offline = False, goal_file = None, compress= False, output_mode = None) -> None:
        self.log_dir = log_dir
        self.workspace = self.log_dir
        self.goal_file = goal_file
        if goal_file is not None:
            self.texture_file = goal_file.replace('yaml', 'png') 
        elif self.log_dir is not None:
            self.texture_file = self.log_dir + "/test.png"
        else:
            self.texture_file = None

        if goal_file is not None:
            shutil.copy(self.goal_file, self.log_dir+ '/test.yaml')
            shutil.copy(self.texture_file, self.log_dir+ '/test.png')
            
        self.gym = self.gym_init(gpuid, goal_file = goal_file, gym_cfg= gym_cfg, output_mode= output_mode) if not offline else None

        self.render = Render(gpuid= gpuid, width_height= (640, 480))
        self.state_num = 0
        self.compress= compress
    
    def offline_init(self, data_dir, seq_len, debug= False):
        # self.texture_file = os.path.join(data_dir, data_dir.split('/')[-1]+'.png')
        self.dataloader = TestDataLoader(seq_len= seq_len)    
        for state_file in tqdm(os.listdir(data_dir)):
            if 'pkl' in state_file:
                state = pickle.load(open(os.path.join(data_dir, state_file), 'rb'))
                self.dataloader.add_data(*self.get_state_obs(state, compress= self.compress))
                self.dataloader.action_buffer.append(state['action'])
                # self.dataloader.debug_pc_buffer.append(self.render_debug(state)) if debug else None
        return self.dataloader
    
    def step(self, action: np.ndarray):
        action = copy.deepcopy(action)
        action = decode_action(action)
        print(action)
        action_batch = [action]
        states = self.gym.step(action_batch)
        if 'video' == 'demos': #TODO
            for idx, state in enumerate(states):
                pickle.dump(state, open(os.path.join(self.workspace, f'state{self.state_num + idx}.pkl'), 'bw'))
            self.state_num += len(states)
        return self.get_obs(compress = self.compress)
    
    def set_state(self, state):
        self.gym.set_state(state)
        
    def _get_scissor_pose(self, state: dict):
        if isinstance(state["_sim_env"]["ti_objects"], list):
            for ti_object in state["_sim_env"]["ti_objects"]:
                if ti_object["class"] == "Scissor":
                    scissor_state = ti_object
        if isinstance(state["_sim_env"]["ti_objects"], dict):
            scissor_state = state["_sim_env"]["ti_objects"]["Scissor"]

        result = np.zeros((8, ), np.float32)
        result[0] = scissor_state["direct_cfg"]["joint_0"]
        result[1:4] = scissor_state["direct_cfg"]["joint_1"][0:3]
        result[4:8] = rotate2quat_numpy(
            np.array(scissor_state["direct_cfg"]["joint_1"][3:6]))
        return result
        
    def get_obs(self, compress =True):
        state = self.gym.get_state()[0]
        pose = self._get_scissor_pose(state) # scissor pose
        front_point = self.gym.get_scissors_front_point()[0] if compress else None
        pc = self.render(state, self.texture_file, output_type = 'pc', compress = compress, front_point = front_point)
        return pc, pose
    
    def get_state_obs(self, state, compress =True):
        pose = self._get_scissor_pose(state) # scissor pose
        if compress:
            front_point = np.array(state['scissor_front_point'])
        else:
            front_point = None
        pc = self.render(state, self.texture_file,output_type = 'pc', compress = compress, front_point = front_point)
        return pc, pose
    
    def gym_init(self, gpuid, gym_cfg, goal_file = None, output_mode = 'render'):
        print(f"Using GPU:{gpuid} for Physical simulation")
        simulation_cfg = OmegaConf.load(proj_root+ '/config/paper_cutting_game_realworld_fast.yaml')
        gym_cfg.setup.cuda = gpuid
        self.gym_cfg = gym_cfg
        simulation_cfg = OmegaConf.merge(simulation_cfg, gym_cfg) 

        env = gym(simulation_cfg, gym_cfg, output_mode= output_mode)
        if goal_file is not None:
            goal_edge_set = np.array(OmegaConf.load(goal_file).goal_edge_set)
        else:
            goal_edge_set = random_generate_goal(save_name = self.log_dir+ '/test')
        
        env.goal_edge_set_batch = [goal_edge_set]
        
        env.batch_dirs = ['']
        env.workspace = self.workspace
        return env

    def pre_policy_step(self, seq_len):
        env = self.gym
        env.reset(init= True)
        if self.gym_cfg.demos.constraints == 'top_edge':
            fix_top_edge(env)
        elif self.gym_cfg.demos.constraints == 'square_pad':
            fix_square_pad(env)
        else:
            raise NotImplementedError()

        self.pre_cut_state_list,pre_cut_action_list = pre_cut(env)
        self.dataloader = TestDataLoader(seq_len= seq_len)
        for idx, state_batch in enumerate(self.pre_cut_state_list):
            state = state_batch[0]
            self.dataloader.add_data(*self.get_state_obs(state, compress = self.compress))
        after_pre_cut_state = self.gym.get_state()[0]
        self.dataloader.add_data(*self.get_state_obs(after_pre_cut_state, compress = self.compress))
            # pickle.dump(state_batch, open(os.path.join(self.workspace, f'state{self.state_num + idx}.pkl'), 'bw'))
        # self.state_num += len(self.pre_cut_state_list)
        return env
    
    def export_images(self):
        if not os.path.exists(self.log_dir + '/images'):
            os.mkdir(self.log_dir + '/images')
        for file in tqdm(os.listdir(self.log_dir)):
            if 'pkl' in file:
                state = pickle.load(open(self.log_dir + '/' + file, "rb"))
                save_name = file.split('.')[0].zfill(6)
                img = self.render(state, self.texture_file,'rgb')
                img.save(self.log_dir + '/images/' + save_name + '.png')
    
    def export_meshes(self, name = None):
        if not os.path.exists(self.log_dir + '/meshes'):
            os.mkdir(self.log_dir + '/meshes')
        if name is not None:
            state = pickle.load(open(self.log_dir + '/' + name + '.pkl', "rb"))
            self.gym.set_state(state)
            mesh = self.gym.get_all_mesh()
            mesh.export(self.log_dir + '/meshes/'+ name + '.ply' )
        else:
            for file in tqdm(os.listdir(self.log_dir)):
                if 'pkl' in file:
                    state = pickle.load(open(self.log_dir + '/' + file, "rb"))
                    save_name = file.split('.')[0].zfill(6)
                    self.gym.set_state(state)
                    mesh = self.gym.get_all_mesh()
                    mesh.export(self.log_dir + '/meshes/'+ save_name + '.ply' )
        
def policy_init(checkpoint_path, model_yaml_path, gpuid, log_dir = None, arch = 'transformer'):
    print(f"Using GPU:{gpuid} for run policy")
    model_kwargs = omegaconf.OmegaConf.load(model_yaml_path)
    omegaconf.OmegaConf.save(model_kwargs, os.path.join(log_dir, "model_config.yaml")) if log_dir is not None else None
    model_kwargs = omegaconf.OmegaConf.to_container(model_kwargs)
    
    if arch == 'pointnet':
        model = PointNetBC.load_from_checkpoint(
            checkpoint_path, **model_kwargs, map_location=torch.device('cuda:0'))

    if arch == 'transformer':
        model = PointNetTransformerBC.load_from_checkpoint(
            checkpoint_path, **model_kwargs, map_location=torch.device('cuda:0'))
        
    model.eval()
    return model

def decode_action(encoded_action: np.ndarray):
    if encoded_action.shape == (12, ):
        action_idx = encoded_action[0:4].argmax().item()
        if action_idx == 0:
            value = encoded_action[ActionLookupTable['Open']['PredIndices']]
            return {"Action": 'Open', "Angle": min(max(value[0], 0.0), 0.55)}
        elif action_idx == 1:
            value = encoded_action[ActionLookupTable['Close']['PredIndices']]
            return {"Action": "Close", "Angle": min(max(value[0], 0.0), 0.55)}
        elif action_idx == 2:
            value = encoded_action[ActionLookupTable['Translate']['PredIndices']]
            return {"Action": "Translate", "Displacement": value.tolist()}
        elif action_idx == 3:
            value = encoded_action[ActionLookupTable['Rotate']['PredIndices']]
            return {"Action": "Rotate", "Displacement": value.tolist()}
        else:
            raise ValueError(f"unknown action type: {action_idx}")
    elif encoded_action.shape == (16, ):
        action_idx = encoded_action[0:5].argmax().item()
        if action_idx == 0:
            value = encoded_action[ActionLookupTable_for16D['Open']['PredIndices']]
            return {"Action": 'Open', "Angle": min(max(value[0], 0.0), 0.55)}
        elif action_idx == 1:
            value = encoded_action[ActionLookupTable_for16D['Close']['PredIndices']]
            return {"Action": "Close", "Angle": min(max(value[0], 0.0), 0.55)}
        elif action_idx == 2:
            value = encoded_action[ActionLookupTable_for16D['Translate']['PredIndices']]
            return {"Action": "Translate", "Displacement": value.tolist()}
        elif action_idx == 3:
            value = encoded_action[ActionLookupTable_for16D['Rotate']['PredIndices']]
            return {"Action": "Rotate", "Displacement": value.tolist()}
        elif action_idx == 4:
            value = encoded_action[ActionLookupTable_for16D['RotateSmall']['PredIndices']]
            return {"Action": "Rotate", "Displacement": value.tolist()}
        else:
            raise ValueError(f"unknown action type: {action_idx}")
    else:
        raise ValueError(f"invalid action shape:{encoded_action.shape}")

action_order = [3, 1, 0, 4, 2]
# action_order = [3, 1, 0, 3, 2]
def one_hot_encode(number, num_classes = 5):
    one_hot = np.zeros(num_classes)
    one_hot[number] = 1
    return one_hot


def get_args():
    parser = argparse.ArgumentParser()
    # configuration
    parser.add_argument("--gpuid", "-g", type=int, default=-1)
    parser.add_argument("--checkpoint-path", "-ckpt", type=str, default="./rloutputs/test_bc/batch_valid_debug/epoch=22-step=120000.ckpt")
    parser.add_argument("--model-yaml-path", "-y", type=str,
                        default="./config/rl/bc_beta.yaml")
    parser.add_argument("--log_dir", "-d", type=str, default="./rloutputs/test_bc/")
    parser.add_argument("--exp_name", type=str, default="exp0")
    parser.add_argument("--render_mode", action= "store_true")
    parser.add_argument("--export_mode", type=str, choices= ['none', 'image', 'mesh', 'both'], default = 'image')
    parser.add_argument('--goal_path', type =str, default= None)
    parser.add_argument('--compress', action= 'store_true')

    parser.add_argument('--num_actions', type = int, default= 30)
    parser.add_argument('--val_set', type = str, default= "./val_set.txt")
    parser.add_argument('--pre_cut_len', type = int, default= 5)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    args.log_dir = os.path.join(args.log_dir, args.exp_name)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    env = TestEnv(args.gpuid, args.log_dir, goal_file = args.goal_path, compress= args.compress)
    policy: PointNetTransformerBC = policy_init(checkpoint_path= args.checkpoint_path, model_yaml_path= args.model_yaml_path, 
                                                log_dir= args.log_dir, gpuid= args.gpuid)
    
    model_kwargs = omegaconf.OmegaConf.load(args.model_yaml_path)
    model_kwargs = omegaconf.OmegaConf.to_container(model_kwargs)
    env.pre_policy_step(seq_len= model_kwargs["seq_len"])
    
    for i in range(50):
        pc_seq, pose_seq = env.dataloader.__getitem__(env.dataloader.buffer_len - 1)
        pc_seq = torch.from_numpy(pc_seq[None, :]).cuda()
        pose_seq = torch.from_numpy(pose_seq[None, :]).cuda()
        with torch.no_grad():
            action = to_numpy(policy.predict(pc_seq, pose_seq)[0])

        action_type = one_hot_encode(action_order[i%5], num_classes= 5)
        action[:5] = action_type
        pc, pose = env.step(action)
        env.dataloader.add_data(pc, pose)
        
    
    if args.export_mode == 'image':    
        env.export_images()
    elif args.export_mode == 'mesh':
        env.export_meshes()
    elif args.export_mode == 'both':
        env.export_images()
        env.export_meshes()
    else:
        print('No Exportation')

def validate(args, gym_cfg):

    args.log_dir = os.path.join(args.log_dir, args.exp_name)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    env = TestEnv(args.gpuid, args.log_dir, offline= False, gym_cfg= gym_cfg, goal_file = None, compress= True, output_mode= 'demos')
    policy: PointNetTransformerBC = policy_init(checkpoint_path= args.checkpoint_path, model_yaml_path= args.model_yaml_path, 
                                                log_dir= args.log_dir, gpuid= args.gpuid)
    
    model_kwargs = omegaconf.OmegaConf.load(args.model_yaml_path)
    model_kwargs = omegaconf.OmegaConf.to_container(model_kwargs)

    with open(args.val_set, 'r') as file:
        validate_goal_set = [str(line.strip()) for line in file]
    
    for valid_goal_file in validate_goal_set:
        print(f'Validating: {valid_goal_file}')
        local_goal_dir = ''.join(valid_goal_file.split('/')[-4: -1])
        env.workspace = os.path.join(env.log_dir, local_goal_dir)
        os.makedirs(env.workspace, exist_ok=True)
        env.gym.workspace = env.workspace
        goal_edge_set = np.array(OmegaConf.load(valid_goal_file)['goal_edge_set'])
        env.gym.goal_edge_set_batch = [goal_edge_set]
        env.goal_file = valid_goal_file
        env.texture_file = valid_goal_file.replace('yaml', 'png')
        env.pre_policy_step(seq_len= model_kwargs["seq_len"])

        for i in tqdm(range(args.num_actions - args.pre_cut_len)):
            pc_seq, pose_seq = env.dataloader.__getitem__(env.dataloader.buffer_len - 1)
            pc_seq = torch.from_numpy(pc_seq[None, :]).cuda()
            pose_seq = torch.from_numpy(pose_seq[None, :]).cuda()
            with torch.no_grad():
                action = to_numpy(policy.predict(pc_seq, pose_seq)[0])

            action_type = one_hot_encode(action_order[i%5], num_classes= 5)
            action[:5] = action_type
            pc, pose = env.step(action)
            env.dataloader.add_data(pc, pose)
    
    return env

def debug():
    args= get_args()
    validate(args= args)

if __name__ == '__main__':
    debug()
    