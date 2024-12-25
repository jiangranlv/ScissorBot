import os
import numpy as np
from omegaconf import OmegaConf
import torch
from sklearn.linear_model import RANSACRegressor

from policy.validate_policy import GymStateTool,TestDataLoader
from rl.pytorch_utils import to_numpy

def fit_direction_from_pc(points):

    ransac = RANSACRegressor()
    ransac.fit(points[:, 0].reshape(-1, 1), points[:, 1])  # 在三维空间中拟合直线
    coeff_y = ransac.estimator_.coef_[0]
    
    ransac.fit(points[:, 0].reshape(-1, 1), points[:, 2])
    coeff_z = ransac.estimator_.coef_[0]

    direction = np.array([1, coeff_y, coeff_z])
    direction = direction / np.linalg.norm(direction)
    return direction

class BasePolicy():
    def __init__(self, data_cfg, render_cfg, compress) -> None:
        self.dataloader = TestDataLoader(**data_cfg)
        self.compress = compress
        self.render = GymStateTool(**render_cfg)
        
    def predict(self, action_type):
        pass

    def receive_state(self, state, tex_file, is_open = False):
        if isinstance(state, list) and len(state) == 1 and isinstance(state[0], dict):
            state = state[0]
        # front_point = state['front_point']
        # cut_direction = state['cut_direction']
        # pc = self.render(state, texture_file= tex_file, output_type = 'pc', compress = self.compress, front_point = front_point)
        # self.dataloader.add_data(pc, np.concatenate([front_point, cut_direction]))
        self.render.texture_file = tex_file
        crop_point = self.dataloader.get_data()[-1][:3] if is_open else state['front_point']
        pc, pose = self.render.get_state_obs(state, crop_point= crop_point)
        self.dataloader.add_data(pc, pose)
    
    def reset(self):
        self.dataloader.reset()

    def pre_process(self, **kwargs):
        pass

class MultiFrameBC(BasePolicy):
    def __init__(self, net_cfg, render_cfg, compress) -> None:
        data_cfg = dict(seq_len= net_cfg.seq_len, pose_dim= net_cfg.pose_dim)
        super().__init__(data_cfg, render_cfg, compress)
        self.net = self.net_init(**net_cfg)
    
    def net_init(self, checkpoint_path, model_yaml_path, gpuid, log_dir = None, arch = 'transformer', **kwargs):
        from dagger.bc_delta import PointNetTransformerBC
        print(f"Using GPU:{gpuid} for run policy")
        model_kwargs = OmegaConf.load(model_yaml_path)
        OmegaConf.save(model_kwargs, os.path.join(log_dir, "model_config.yaml")) if log_dir is not None else None
        model_kwargs = OmegaConf.to_container(model_kwargs)

        model = PointNetTransformerBC.load_from_checkpoint(
            checkpoint_path, **model_kwargs, map_location=torch.device('cuda:0'))
            
        model.eval()
        print('Init Model Completed')
        return model
    
    def predict(self, action_type):
        pc_seq, pose_seq = self.dataloader.__getitem__(self.dataloader.buffer_len - 1)
        # from debug_tool import convert_4d_point_cloud_to_open3d
        # convert_4d_point_cloud_to_open3d(pc_seq[-1], '/data1/jiangranlv/code/cutting_paper_fem/debug_pc', name= str(self.dataloader.buffer_len - 1) + '_' + action_type)
        pc_seq = torch.from_numpy(pc_seq[None, :]).cuda()
        pose_seq = torch.from_numpy(pose_seq[None, :]).cuda()
        with torch.no_grad():
            pred = to_numpy(self.net.predict(pc_seq, pose_seq)[0])
        action_value = self.decode_action(pred = pred, action_type = action_type) 
        return [action_value]
    
    def decode_action(self, pred:np.ndarray, action_type:str):
        action2idx = dict(close = 0, push = 1, rotate = np.array([2,3,4]), tune = np.array([5,6,7]))
        idx = action2idx[action_type]
        return pred[idx]
           
class MultiFrameBC_NoTune(MultiFrameBC):
    def __init__(self, net_cfg, render_cfg, compress) -> None:
        super().__init__(net_cfg, render_cfg, compress)
    
    def predict(self, action_type):
        if action_type =='tune':
            pc_seq, pose_seq = self.dataloader.__getitem__(self.dataloader.buffer_len - 1)
            return [pose_seq[-1, 3:]]
        else:
            return super().predict(action_type)

class PoseBC(MultiFrameBC):
    def __init__(self, net_cfg, render_cfg, compress) -> None:
        super().__init__(net_cfg, render_cfg, compress)
    
    def net_init(self, checkpoint_path, model_yaml_path, gpuid, log_dir = None, arch = 'transformer', **kwargs):
        from dagger.bc_pose import PointNetTransformerBC
        print(f"Using GPU:{gpuid} for run policy")
        model_kwargs = OmegaConf.load(model_yaml_path)
        OmegaConf.save(model_kwargs, os.path.join(log_dir, "model_config.yaml")) if log_dir is not None else None
        model_kwargs = OmegaConf.to_container(model_kwargs)

        model = PointNetTransformerBC.load_from_checkpoint(
            checkpoint_path, **model_kwargs, map_location=torch.device('cuda:0'))
            
        model.eval()
        print('Init Model Completed')
        return model
    
    def predict(self):
        return super().predict(None)
    
    def decode_action(self, pred: np.ndarray, **kwargs):
        return pred

class BC_ACT(MultiFrameBC):
    def __init__(self, net_cfg, render_cfg, compress, max_timesteps, chunking_weight) -> None:
        super().__init__(net_cfg, render_cfg, compress)
        self.chunking_size = len(chunking_weight)
        self.all_time_pred = torch.zeros([max_timesteps, max_timesteps + self.chunking_size, 20]).cuda()
        self.step = 0
        self.chunking_weight = np.array(chunking_weight)

    def net_init(self, checkpoint_path, model_yaml_path, gpuid, log_dir = None, arch = 'transformer', **kwargs):
        from dagger.bc_act import PointNetTransformerBC
        print(f"Using GPU:{gpuid} for run policy")
        model_kwargs = OmegaConf.load(model_yaml_path)
        OmegaConf.save(model_kwargs, os.path.join(log_dir, "model_config.yaml")) if log_dir is not None else None
        model_kwargs = OmegaConf.to_container(model_kwargs)

        model = PointNetTransformerBC.load_from_checkpoint(
            checkpoint_path, **model_kwargs, map_location=torch.device('cuda:0'))
            
        model.eval()
        print('Init Model Completed')
        return model
    
    def predict(self, action_type):
        pc_seq, pose_seq = self.dataloader.__getitem__(self.dataloader.buffer_len - 1)
        pc_seq = torch.from_numpy(pc_seq[None, :]).cuda()
        pose_seq = torch.from_numpy(pose_seq[None, :]).cuda()
        with torch.no_grad():
            all_pred = self.net.predict(pc_seq, pose_seq)[0]

        assert all_pred.shape[0] == self.chunking_size
        self.all_time_pred[self.step, self.step:self.step + self.chunking_size] = all_pred
        
        # if action_type == 'close':
        #     print(all_pred[-1, 0])
        # elif action_type == 'rotate':
        #     print(all_pred[-2, 0])
        # elif action_type == 'push':
        #     print(all_pred[-3, 0])
        # elif action_type == 'tune':
        #     print(all_pred[-4, 0])
        pred_for_curr_step = self.all_time_pred[:, self.step]
        pred_populated = torch.all(pred_for_curr_step != 0, axis=1)
        pred_for_curr_step = pred_for_curr_step[pred_populated] # N, action_dim

        actions_for_curr_step = to_numpy(self.pred_to_action(pred_for_curr_step, pose= pose_seq[:, -1, :].repeat(pred_for_curr_step.shape[0], 1)))
        action = self.decode_seq_action(actions_for_curr_step, action_type = action_type)
        
        self.step += 1
        return [action]
    
    def decode_action(self, pred: np.ndarray, **kwargs):
        return pred
    
    def pred_to_action(self, pred_seq: torch.Tensor, pose: torch.Tensor):
        return self.net.model._postprocess_action_pred(pred_seq, pose)

    def decode_seq_action(self, action_seq: torch.Tensor, action_type:str):
        action2idx = dict(close = 0, push = 1, rotate = np.array([2,3,4]), tune = np.array([5,6,7]))
        idx = action2idx[action_type]
        return self.ensemble_actions(action_seq[:, idx], action_type)

    def ensemble_actions(self, action_values: np.ndarray, action_type:str):
        curr_chunking_size = action_values.shape[0]
        assert curr_chunking_size > 0 and curr_chunking_size <= self.chunking_size
        if action_type == 'tune' or action_type == 'rotate':
            # action_value: N, d  weights: N  do weighted average
            vector = np.sum(action_values * self.chunking_weight[-curr_chunking_size:, None]/ np.sum(self.chunking_weight[-curr_chunking_size:]), axis=0)
            return vector / np.linalg.norm(vector)
        else:
            return np.sum(action_values * self.chunking_weight[-curr_chunking_size:] /np.sum(self.chunking_weight[-curr_chunking_size:]), axis=0)

    def reset(self):
        self.step = 0
        self.all_time_pred = torch.zeros_like(self.all_time_pred).cuda()
        return super().reset()

class BC_Pose_ACT(BC_ACT):
    def __init__(self, net_cfg, render_cfg, compress, max_timesteps, chunking_weight) -> None:
        super().__init__(net_cfg, render_cfg, compress, max_timesteps, chunking_weight)
        self.all_time_pred = torch.zeros([max_timesteps, max_timesteps + self.chunking_size, 7]).cuda()
    
    def net_init(self, checkpoint_path, model_yaml_path, gpuid, log_dir=None, arch='transformer', **kwargs):
        from dagger.bc_pose_act import PointNetTransformerBC
        print(f"Using GPU:{gpuid} for run policy")
        model_kwargs = OmegaConf.load(model_yaml_path)
        OmegaConf.save(model_kwargs, os.path.join(log_dir, "model_config.yaml")) if log_dir is not None else None
        model_kwargs = OmegaConf.to_container(model_kwargs)

        model = PointNetTransformerBC.load_from_checkpoint(
            checkpoint_path, **model_kwargs, map_location=torch.device('cuda:0'))
            
        model.eval()
        print('Init Model Completed')
        return model
    
    def predict(self):
        return super().predict(None)

    def pred_to_action(self, pred_seq: torch.Tensor, **kwargs):
        return pred_seq
    
    def decode_seq_action(self, action_seq: torch.Tensor, **kwargs):
        return self.ensemble_actions(action_seq)

    def ensemble_actions(self, action_values: np.ndarray,  **kwargs):
        curr_chunking_size = action_values.shape[0]
        assert curr_chunking_size > 0 and curr_chunking_size <= self.chunking_size

        pose = np.sum(action_values * self.chunking_weight[-curr_chunking_size:, None]/ np.sum(self.chunking_weight[-curr_chunking_size:]), axis=0)
        pose[-3:] = pose[-3:]/ np.linalg.norm(pose[-3:])
        return pose


class Tracking(BasePolicy):
    def __init__(self, data_cfg, render_cfg, compress, unit) -> None:
        super().__init__(data_cfg, render_cfg, compress)
        self.unit = unit
        self.last_direc = np.array([1, 0, 0])

    def predict(self, action_type):
        if action_type == 'rotate':
            try:
                pc_seq, pose_seq = self.dataloader.__getitem__(self.dataloader.buffer_len - 1)           
                pc_current = pc_seq[-1]
                pc_mask = pc_current[pc_current[:, -1].astype(bool), :]
                pc_mask_around_fp = pc_mask[(pc_mask[:, 0] > 0) & (pc_mask[:, 0] < self.unit)]
                if pc_mask_around_fp.shape[0] > 2:
                    direc = fit_direction_from_pc(pc_mask_around_fp)

                    # from debug_tool import convert_4d_point_cloud_to_6d, save_6d_point_cloud
                    # patch_pc = convert_4d_point_cloud_to_6d(pc_current)
                    # save_6d_point_cloud(patch_pc,'/data1/jiangranlv/code/cutting_paper_fem/test_pc_seg', 'debug_tracking_patch')
                    
                    # raise 1
                    self.last_direc = direc.copy()
                    return [direc]
                else: # can't see enough points
                    self.last_direc = pose_seq[-1, 3:].copy()
                    return [pose_seq[-1, 3:]]
            except Exception as e:
                print('Error during predict' , e)

            return [self.last_direc]

        elif action_type == 'tune':
            try:
                pc_seq, pose_seq = self.dataloader.__getitem__(self.dataloader.buffer_len - 1)
                return [pose_seq[-1, 3:]]
            
            except Exception as e:
                return [self.last_direc]
            
        elif action_type == 'push' or action_type == 'close':
            return [self.unit]
        
        else:
            raise NotImplementedError()

class PreDetection(BasePolicy):
    def __init__(self, render_cfg, compress, unit) -> None:
        data_cfg = dict(seq_len = 1, pose_dim = 6)
        super().__init__(data_cfg, render_cfg, compress)
        self.unit = unit
        self.direction_list = []
        self.current = 0

    def pre_process(self, state, tex_file, **kwargs):
        if isinstance(state, list) and len(state) == 1 and isinstance(state[0], dict):
            state = state[0]
        self.render.texture_file = tex_file
        pc = self.render.get_whole_pc(state)

        # print('pc', pc.shape)

        # save_path = '/data1/jiangranlv/code/cutting_paper_fem/test_pc_seg/'
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()

        # pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:])
        # o3d.io.write_point_cloud(f'{save_path}/pre_detect.ply', pcd)

        _where_cut_line=lambda x: x[..., 0] > (x[..., 1] + x[..., 2]) * 3 
        red_line_pc = pc[_where_cut_line(pc[:, 3:]), :]

        # print('red', red_line_pc.shape)
        # pcd.points = o3d.utility.Vector3dVector(red_line_pc[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector(red_line_pc[:, 3:])
        # o3d.io.write_point_cloud(f'{save_path}/pre_detect_red.ply', pcd)

        self.direction_list = self.segment_directions(red_line_pc[:, :3])
    
    def predict(self, action_type):
        if action_type == 'rotate':
            result = [self.direction_list[self.current]]
            self.current+= 1
            return result

        if action_type =='tune':
            return [self.direction_list[self.current]]
        
        elif action_type == 'push' or action_type == 'close':
            return [self.unit]
        
        else:
            raise NotImplementedError()

    def segment_directions(self, line_pc):
        num_points = line_pc.shape[0]
        sorted_indices = np.argsort(line_pc[:, 0])
        sorted_line = line_pc[sorted_indices]

        distance = np.linalg.norm(sorted_line[:, None, :] - sorted_line[None, :, :], axis = -1)
        negetive_score = np.abs(distance - self.unit)
        pivot_points = []
        current_point = 0
        while True:
            if current_point >= num_points:
                break
            next_point = np.argmin(negetive_score[current_point, current_point:]) + current_point
            if negetive_score[current_point, next_point] < 0.5 * self.unit:
                pivot_points.append(next_point)
                current_point = next_point
            else:
                break
        
        current_point = 0
        direction_list = []
        for pivot_point in pivot_points:
            direction_list.append(fit_direction_from_pc(sorted_line[current_point:pivot_point, :]))
            current_point = pivot_point

        return direction_list
        
    def reset(self):
        self.current = 0
        self.direction_list = []
        return super().reset()

class PreDetectionGT(PreDetection):
    def __init__(self, render_cfg, compress, unit) -> None:
        super().__init__(render_cfg, compress, unit)

    def pre_process(self, goal_edge_set, **kwargs):
        self.direction_list = self.compute_direction(goal_edge_set)

    def compute_direction(self, goal_edge_set:np.ndarray):

        edge_vector = goal_edge_set[:, 1, :] - goal_edge_set[:, 0, :]
        edge_vector = edge_vector/ np.linalg.norm(edge_vector, axis= -1)[:, None]

        return edge_vector.tolist()