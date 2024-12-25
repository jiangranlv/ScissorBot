import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import ast
from typing import Literal, Tuple, List
import copy
import time

import omegaconf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler, PyTorchProfiler
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor, Callback, BaseFinetuning
import transformers

from rl.pointnet_utils import PointNetModule
from rl.loss_utils import ActionLoss, HelperPointLoss, GoalPredLoss
from rl.pytorch_utils import *
from rl.prepare_dataset import DataForest, rotate2quat_numpy
from rl.so3 import rotate9D_to_angle_theta_phi

# global constants
POINT_DIM = 4
PRE_CUT_LEN = 5
POINT_RES = 512

class ModifyStepCallback(Callback):
    def __init__(self, step_offset=0) -> None:
        super().__init__()
        self.step_offset = step_offset

    def on_train_start(self, trainer, pl_module):
        trainer.fit_loop.epoch_loop._batches_that_stepped += self.step_offset # for logger
        trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed += self.step_offset # this change global_step, for ckpt name


class BehaviorCloningTransformerDataset(Dataset):
    def __init__(self, data_index_table: np.ndarray, seq_len: int, action_type_num: int, mode: Literal["train", "eval"],
                 point_cloud: DataForest, pose: DataForest, action: DataForest, helper_point: DataForest, info: DataForest,
                 aux: DataForest) -> None:
        super().__init__()

        self._data_index_table = data_index_table.copy()
        self._size = len(self._data_index_table)
        self._seq_len = seq_len
        self._action_type_num = action_type_num

        assert mode in ["train", "eval"]
        self._mode = mode
        self._pc = point_cloud
        self._pose = pose
        self._ac = action
        self._hp = helper_point
        self._info = info
        self._aux = aux

        print(f"dataset {mode}: len {self._size}")

    def __len__(self):
        return self._size

    def _is_same_trajectory(self, data_index1:int, data_index2:int):
        return os.path.dirname(self._info[data_index1].get_path()) == \
            os.path.dirname(self._info[data_index2].get_path())
    
    def _is_small_rotation(self, prev_ac_type: int, ac_gt: np.ndarray) -> bool:
        old_rotation_id = 3
        old_open_id = 0
        return ac_gt[0] == old_rotation_id and prev_ac_type == old_open_id

    @property
    def max_data_index(self) -> int:
        return len(self._ac)

    def __getitem__(self, index):
        data_index = int(self._data_index_table[index])
        pc_seq = np.zeros((self._seq_len, POINT_RES, POINT_DIM), np.float32) # [T, POINT_RES, POINT_DIM]
        pose_seq = np.zeros((self._seq_len, 8), np.float32) # [T, 8] Rotate = quaternion
        hp_seq_gt = np.zeros((self._seq_len, 32, 3), np.float32) # [T, 32, 3]
        ac_gt: np.ndarray = np.load(self._ac[data_index].get_path()) # [4, ] Rotate = ANGLE THETA PHI

        aux = np.load(self._aux[data_index].get_path(), allow_pickle= True).tolist()
        front_point: np.ndarray = aux["front_point"].astype(np.float32)
        cut_direction: np.ndarray = aux["cut_direction"].astype(np.float32)
        next_edge: np.ndarray = aux["next_edge"].astype(np.float32)
        data_idx = np.array([data_index], np.int64)
        
        prev_ac_type = -1.0
        for prev_data_index in range(data_index, data_index - self._seq_len, -1):
            idx_in_seq = prev_data_index - (data_index - self._seq_len + 1)
            if 0 <= prev_data_index and prev_data_index < self.max_data_index and \
                self._is_same_trajectory(prev_data_index, data_index):
                pc_seq[idx_in_seq, ...] = np.load(self._pc[prev_data_index].get_path())
                pose_seq[idx_in_seq, ...] = np.load(self._pose[prev_data_index].get_path())
                hp_seq_gt[idx_in_seq, ...] = np.load(self._hp[prev_data_index].get_path())
                
                if prev_data_index == data_idx - 1:
                    prev_ac_type = np.load(self._ac[prev_data_index].get_path())[0] # get action type
            elif idx_in_seq < self._seq_len - 1:
                pc_seq[idx_in_seq, ...] = pc_seq[idx_in_seq + 1, ...].copy()
                pose_seq[idx_in_seq, ...] = pose_seq[idx_in_seq + 1, ...].copy()
                hp_seq_gt[idx_in_seq, ...] = hp_seq_gt[idx_in_seq + 1, ...].copy()
            else:
                raise RuntimeError(f"Unexpected idx_in_seq {idx_in_seq}")

        # IMPORTANT! modify ac_gt
        if self._action_type_num == 5 and self._is_small_rotation(prev_ac_type, ac_gt):
            ac_gt[0] = 4.0
            
        # sanity check
        assert pc_seq.shape == (self._seq_len, POINT_RES, POINT_DIM), f"{pc_seq.shape}"
        assert pose_seq.shape == (self._seq_len, 8), f"{pose_seq.shape}"
        assert hp_seq_gt.shape == (self._seq_len, 32, 3), f"{hp_seq_gt.shape}"
        assert ac_gt.shape == (4, ), f"{ac_gt.shape}"
        assert data_idx.shape == (1, ), f"{data_idx.shape}"
        assert front_point.shape == (3, ), f"{front_point.shape}"
        assert cut_direction.shape == (3, ), f"{cut_direction.shape}"
        assert next_edge.shape == (2, 3), f"{next_edge.shape}"

        assert False not in np.isfinite(pc_seq)
        assert False not in np.isfinite(pose_seq)
        assert False not in np.isfinite(hp_seq_gt)
        assert False not in np.isfinite(ac_gt)
        assert False not in np.isfinite(data_idx)
        assert False not in np.isfinite(front_point)
        assert False not in np.isfinite(cut_direction)
        assert False not in np.isfinite(next_edge)

        return pc_seq, pose_seq, hp_seq_gt, ac_gt, data_idx, cut_direction, front_point, next_edge
    

def make_single_dataset(data_path: List[str], mode: Literal["train", "eval"], seq_len: int, 
                        pre_cut_len: int, action_type_num: int, verbose=1, aux_path: List[str] = None, compress_path = None):
    if verbose == 1:
        print(f"scan all {mode} data ...")
    
    pc_path = data_path if compress_path is None else compress_path
    pc_file_name = 'point_cloud' if compress_path is None else "compressed"  
    f_pc = DataForest(pc_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name= pc_file_name)
    
    f_pose = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="scissor_pose")
    f_action = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="action")
    f_hp = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="helper_point")
    f_info = DataForest(data_path, target_file_suffix=[".yaml"], info_file_suffix=[], target_file_name="info")
    
    f_aux = DataForest(aux_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="auxiliary")
    all_data_size = len(f_pc)
    assert len(f_pose) == all_data_size and len(f_action) == all_data_size and \
        len(f_hp) == all_data_size and len(f_info) == all_data_size and len(f_aux) == all_data_size, \
            f"{len(f_pc)} {len(f_pose)} {len(f_action)} {len(f_hp)} {len(f_info)} {len(f_aux)}"

    # drop pre_cut
    all_data_idx = []
    for i in range(all_data_size):
        filename = os.path.split(f_info[i].get_path())[1]
        assert filename[-10:] == "_info.yaml"
        if int(filename[:-10]) >= pre_cut_len:
            all_data_idx.append(i)
    all_data_idx = np.array(all_data_idx)

    if verbose == 1:
        print(f"make dataset {mode} ...")

    return BehaviorCloningTransformerDataset(all_data_idx, seq_len, action_type_num, mode, f_pc, f_pose, f_action, f_hp, f_info, f_aux)

  
def get_bc_transformer_dataset(train_data_path: List[str], eval_data_path: List[str], seq_len: int, 
        pre_cut_len: int, action_type_num: int, train_aux_path: List[str], eval_aux_path: List[str],
        train_compress_path: List[str], eval_compress_path: List[str]):
    return make_single_dataset(train_data_path, "train", seq_len, pre_cut_len, action_type_num, aux_path= train_aux_path, 
                               compress_path= train_compress_path), \
        make_single_dataset(eval_data_path, "eval", seq_len, pre_cut_len, action_type_num, aux_path= eval_aux_path, 
                            compress_path= eval_compress_path)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model.
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        if self.pe.device != x.device:
            self.pe.to(x.device)
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TokenAssembler(nn.Module):
    def __init__(self, output_dim) -> None:
        r"""Perform concatenate and layer normalization."""
        super().__init__()
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, list_token: List[torch.Tensor]):
        return self.ln(torch.concatenate(list_token, dim= -1))
    

class TransformerModule(nn.Module):
    def __init__(self, token_dim, PE_cfg, transformer_cfg) -> None:
        super().__init__()
        
        self.token_dim = token_dim
        self.pe = PositionalEncoding(d_model=self.token_dim, **PE_cfg)
        self.token_assembler = TokenAssembler(self.token_dim)

        transformer_layer = nn.TransformerEncoderLayer(batch_first=True, d_model=self.token_dim, **(transformer_cfg["layer"]))
        self.transformer = nn.TransformerEncoder(transformer_layer, transformer_cfg["num_layers"] - 1)
        self.transformer_shallow = nn.TransformerEncoder(transformer_layer, num_layers= 1)
    
    def forward(self, list_token_seq: List[torch.Tensor]):
        input_seq: torch.Tensor = self.token_assembler(list_token_seq) # [B, T, self.token_dim]
        shallow_feat =  self.transformer_shallow(self.pe(input_seq))
        feat = self.transformer(shallow_feat)
        return shallow_feat, feat
        
class ActionDeTokenizer(nn.Module):
    def __init__(self, action_token_dim: int, num_classes: int, rotation_dim: int, cls_head_cfg = {},
                 rot_head_cfg = {}, trans_head_cfg= {}, open_head_cfg ={}, close_head_cfg={}) -> None:
        super().__init__()

        self.rotation_dim = rotation_dim
        self.cls_head = MLP(input_dim= action_token_dim, output_dim= num_classes, **cls_head_cfg)
        if rotation_dim == 18:
            self.rotation_head = MLP(input_dim= action_token_dim, output_dim= 9, **rot_head_cfg)
            self.rotation_small_head = MLP(input_dim= action_token_dim, output_dim= 9, **rot_head_cfg)
        else:
            self.rotation_head = MLP(input_dim= action_token_dim, output_dim= rotation_dim, **rot_head_cfg)
        self.trans_head = MLP(input_dim= action_token_dim, output_dim= 3, **trans_head_cfg)
        self.open_head = MLP(input_dim= action_token_dim, output_dim= 1, **open_head_cfg)
        self.close_head = MLP(input_dim= action_token_dim, output_dim= 1, **close_head_cfg)
        
    def forward(self, action_token: torch.Tensor) -> torch.Tensor:
        x = action_token
        cls = self.cls_head(x, head = True)
        rot = self.rotation_head(x, head = True)
        
        trans = self.trans_head(x, head = True)
        open = self.open_head(x, head = True)
        close = self.close_head(x, head = True)
        
        action_pred = torch.concat([cls, open, close, trans, rot], dim = -1)
        if self.rotation_dim == 18:
            rot_small = self.rotation_small_head(x) 
            action_pred = torch.concat([action_pred, rot_small], dim = -1)
        return action_pred 


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim=[64], act_func = 'relu') -> None:
        super().__init__()

        self.mlp_dim = [input_dim] + hidden_dim + [output_dim]
        self.layer_n = len(hidden_dim) + 1
        self.fcs = nn.ModuleList([nn.Linear(self.mlp_dim[i], self.mlp_dim[i + 1]) for i in range(self.layer_n)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.mlp_dim[i + 1]) for i in range(self.layer_n)])
        if act_func == 'relu':
            self.act_func = F.relu
        elif 'leaky' in act_func:
            self.act_func = F.leaky_relu
        else:
            raise NotImplementedError(act_func)
        
    def forward(self, input: torch.Tensor, head = False) -> torch.Tensor:
        x = input
        for i in range(self.layer_n - 1):
            x = self.act_func(self.bns[i](self.fcs[i](x)))
        x = self.fcs[-1](x)
        x = x if head else self.act_func(self.bns[-1](x))
        return x
        
class HelperNet(nn.Module):
    def __init__(self, input_dim: int, npoints=32, hidden_dim=[]) -> None:
        super().__init__()

        self.npoints = npoints
        self.output_dim = npoints * 3
        self.mlp_dim = [input_dim] + hidden_dim + [self.output_dim]
        self.layer_n = len(hidden_dim) + 1
        self.fcs = nn.ModuleList([nn.Linear(self.mlp_dim[i], self.mlp_dim[i + 1]) for i in range(self.layer_n)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.mlp_dim[i]) for i in range(self.layer_n)])

    def forward(self, x: torch.Tensor):
        for i in range(self.layer_n):
            x = self.fcs[i](F.relu(self.bns[i](x)))
        return x

class State2ActionModule(nn.Module):
    def __init__(self, action_type_num, rotation_dim, input_statistics={}, feat_dim = 128, 
                 action_detokenizer_cfg ={}) -> nn.Module:
        super().__init__()
        
        self.action_type_num = action_type_num
        self.rotation_dim = rotation_dim
        self.input_statistics = make_input_statistics(input_statistics)
        self.position_net = MLP(input_dim= 4, output_dim= feat_dim, hidden_dim= [feat_dim])
        self.goal_net = MLP(input_dim= 6, output_dim= feat_dim, hidden_dim= [feat_dim])
        self.direction_net = MLP(input_dim= 3, output_dim= feat_dim, hidden_dim= [feat_dim])
        self.merge_net = MLP(input_dim= feat_dim * 2, output_dim= feat_dim, hidden_dim= [feat_dim])
        self.action_detokenizer = ActionDeTokenizer(action_token_dim=feat_dim * 2, 
                num_classes= self.action_type_num, rotation_dim= self.rotation_dim, **action_detokenizer_cfg)
        
    def _preprocess_state_input(self, front_point, cut_direction) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:

        """
        front_point = normalize_transform(front_point, self.input_statistics["front_point"]["mean"],
                                 self.input_statistics["front_point"]["std"])
        cut_direction = normalize_transform(cut_direction, self.input_statistics["cut_direction"]["mean"],
                                   self.input_statistics["cut_direction"]["std"])
        
        return front_point, cut_direction

    def forward(self, pose:torch.Tensor, front_point:torch.Tensor, cut_direction:torch.Tensor, 
                next_edge_a:torch.Tensor, next_edge_b:torch.Tensor) \
                -> Tuple[torch.Tensor, torch.Tensor]:
        # sanity check
        assert pose.shape[1:] == (8, ), f"(?, 3) expected, ({pose.shape}) got"
        assert front_point.shape[1:] == (3, ), f"(?, 3) expected, ({front_point.shape}) got"
        assert cut_direction.shape[1:] == (3, ), f"(?, 3) expected, ({cut_direction.shape}) got"
        assert next_edge_a.shape[1:] == (3, ), f"(?, 3) expected, ({next_edge_a.shape}) got"
        assert next_edge_b.shape[1:] == (3, ), f"(?, 3) expected, ({next_edge_b.shape}) got"
        B = front_point.shape[0]

        front_point, cut_direction = self._preprocess_state_input(front_point, cut_direction)

        pos_feat = self.position_net(torch.concat([front_point, pose[:, 0, None]], dim = -1))
        goal_feat = self.goal_net(torch.concat([next_edge_a,next_edge_b], dim = -1))
        dir_feat = self.direction_net(cut_direction)
        mid_feat = self.merge_net(torch.concat([pos_feat, goal_feat], dim = -1))
        ac_pred = self.action_detokenizer(torch.concat([dir_feat, mid_feat], dim = -1))
        
        return ac_pred

            
class PointNetTransformerModule(nn.Module):
    def __init__(self, seq_len=6, input_statistics={},
                 pn_output_dim=64, pose_token_dim=64, 
                 PN_cfg={}, helper_net_cfg={}, 
                 helper_loss_cfg={}, action_loss_cfg={},goal_pred_loss_cfg = {}, goal_pred_offset = False,
                 pose_tokenizer_cfg={}, PE_cfg={},
                 transformer_cfg={"layer":{"nhead": 8, "dim_feedforward": 512}, "num_layers":6},
                 goal_predictor_cfg = {}, state2action_cfg = {}, state2action_ckpt =None, use_helper = False) -> None:
        super().__init__()

        rotation_dim = action_loss_cfg["rotation_dim"]
        self.rotation_dim = rotation_dim
        if rotation_dim == 3:
            self.alt = ActionLookupTable
            self.action_type_num = 4
            self.anl = ActionNameList
        elif rotation_dim == 9:
            self.alt = ActionLookupTable_for9Drot
            self.action_type_num = 4
            self.anl = ActionNameList
        elif rotation_dim == 18:
            self.alt = ActionLookupTable_for18Drot
            self.action_type_num = 5
            self.anl = ActionNameList5Types
        else:
            raise ValueError(f"rotation_dim: {rotation_dim}")
        self.model_output_dim = self.action_type_num + 5 + self.rotation_dim
        
        self.input_statistics = make_input_statistics(input_statistics)
        self.seq_len = seq_len
        self.token_dim = pn_output_dim + pose_token_dim

        self.net = PointNetModule(output_dim=pn_output_dim, **PN_cfg)
        
        self.use_helper = use_helper
        if use_helper:
            self.helper_net = HelperNet(self.token_dim, **helper_net_cfg)
            self.helper_net_npoints = self.helper_net.npoints
            self.helper_loss = HelperPointLoss(seq_len=seq_len, xyz_unit=get_pc_xyz_std_mean(input_statistics), **helper_loss_cfg)

        self.pose_tokenizer = MLP(output_dim=pose_token_dim, input_dim=8, **pose_tokenizer_cfg)
        self.transformer = TransformerModule(self.token_dim, PE_cfg, transformer_cfg)
        # self.mlp = MLP(input_dim=self.token_dim*self.seq_len, output_dim=self.token_dim, hidden_dim= [self.token_dim, self.token_dim])
        
        self.goal_predictor = MLP(input_dim=self.token_dim* seq_len, output_dim= 6, **goal_predictor_cfg)
        self.goal_pred_offset = goal_pred_offset
        
        self.state2action = State2ActionModule(self.action_type_num, rotation_dim, input_statistics, **state2action_cfg)
        if state2action_ckpt is not None:
            self.state2action = load_ckpt(self.state2action, ckpt_path=state2action_ckpt)
        self.action_loss = ActionLoss(**action_loss_cfg, translation_unit=get_translation_std_mean(input_statistics))
        self.goal_pred_loss = GoalPredLoss(*get_next_edge_xyz_std_mean(input_statistics),**goal_pred_loss_cfg)

    def _preprocess_action_gt(self, action_gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_gt: [?, 4]
        """
        action_idx = torch.round(action_gt[..., 0]).long()
        idx1 = torch.where(action_idx == self.alt["Translate"]["ActionID"])[0][:, None]
        idx2 = torch.tensor(self.alt["Translate"]["TargetIndices"], dtype=torch.int64)[None, :]
        mean = self.input_statistics["action"]["translation"]["mean"]
        std = self.input_statistics["action"]["translation"]["std"]
        action_gt = action_gt.clone()
        action_gt[idx1, idx2] = normalize_transform(action_gt[idx1, idx2], mean, std)
        return action_gt

    def _preprocess_helper_point_gt(self, hp_gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hp_gt: [?, seq_len, 32, 3]
        """
        return normalize_transform(hp_gt, self.input_statistics["point_cloud"]["mean"][:3],
                                   self.input_statistics["point_cloud"]["std"][:3])
    
    def _preprocess_next_edge_gt(self, next_edge):
        next_edge_a = next_edge[:, 0, :]
        next_edge_b = next_edge[:, 1, :]
        next_edge_a = normalize_transform(next_edge_a, self.input_statistics["next_edge"]["a"]["mean"],
                                   self.input_statistics["next_edge"]["a"]["std"])
        next_edge_b = normalize_transform(next_edge_b, self.input_statistics["next_edge"]["b"]["mean"],
                                   self.input_statistics["next_edge"]["b"]["std"])
        return torch.concat([next_edge_a, next_edge_b], dim = -1)
    
    def _postprocess_action_pred(self, action_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_pred: [?, ACTION_PRED_DIM]
        Output:
            action_pred: [?, 12]
        """
        idxt = self.alt["Translate"]["PredIndices"]
        mean = self.input_statistics["action"]["translation"]["mean"]
        std = self.input_statistics["action"]["translation"]["std"]
        action_pred = action_pred.clone()
        action_pred[..., idxt] = inverse_normalize_transform(action_pred[..., idxt], mean, std)
        if self.rotation_dim == 3:
            ret_val = action_pred
        elif self.rotation_dim == 9:
            idxr = self.alt["Rotate"]["PredIndices"]
            action_pred[..., idxr[:3]] = rotate9D_to_angle_theta_phi(action_pred[..., idxr])
            ret_val = action_pred[..., :12] # WARNING! possible bugs here.
        else:
            raise NotImplementedError
        assert ret_val.shape[-1] == 12, f"{ret_val.shape}"
        return ret_val

    
    def _preprocess_input(self, pc: torch.Tensor, pose: torch.Tensor) \
                -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pc: [?, POINT_RES, POINT_DIM]
            pose: [?, 8]
        """
        pc = normalize_transform(pc, self.input_statistics["point_cloud"]["mean"],
                                 self.input_statistics["point_cloud"]["std"])
        pose = normalize_transform(pose, self.input_statistics["scissor_pose"]["mean"],
                                   self.input_statistics["scissor_pose"]["std"])
        return pc, pose

    def forward(self, pc_seq: torch.Tensor, pose_seq: torch.Tensor, cd: torch.Tensor, fp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # sanity check
        assert pc_seq.shape[1:] == (self.seq_len, POINT_RES, POINT_DIM), f"(?, {self.seq_len}, POINT_RES, {POINT_DIM}) expected, ({pc_seq.shape}) got"
        assert pose_seq.shape[1:] == (self.seq_len, 8), f"(?, {self.seq_len}, 8) expected, ({pose_seq.shape}) got"
        B, T = pc_seq.shape[0:2]

        hp_seq_pred =None 
        goal_pred = None
        
        pc, pose = self._preprocess_input(pc_seq.view(B*T, POINT_RES, POINT_DIM), pose_seq.view(B*T, 8))
        pc_token: torch.Tensor = self.net(pc) # [B*T, self.feature_dim]
        pose_token: torch.Tensor = self.pose_tokenizer(pose) # [B*T, self.pose_token_dim]
        
        shallow_feat, transformer_output = self.transformer([pc_token.view(B, T, -1), pose_token.view(B, T, -1)]) # [B, T, self.token_dim]
        flatten_output = transformer_output.view(B, T * self.token_dim)
        # flatten_output = self.mlp(torch.concat([pc_token.view(B,-1), pose_token.view(B, -1)], dim = -1))
        
        if self.use_helper:
            obs_seq = torch.concat([pc_token, pose_token], dim = -1)
            hp_seq_pred: torch.Tensor = self.helper_net(obs_seq).view(B* T, self.helper_net_npoints, 3) # [B * T, #helper_net_npoints, 3]
            
        goal_pred = self.goal_predictor(flatten_output, head = True) #B, 6
        if self.goal_pred_offset:
            goal_pred += fp.repeat(1, 2)
        pose_now = pose.view(B, T, 8)[:, -1, :]
        ac_pred = self.state2action(pose_now, fp, cd, goal_pred[:, :3], goal_pred[:, 3:])

        return ac_pred, hp_seq_pred, goal_pred

    def predict(self, pc_seq: torch.Tensor, pose_seq: torch.Tensor) -> torch.Tensor:
        ac_pred = self(pc_seq, pose_seq)[0]
        return self._postprocess_action_pred(ac_pred)

    def forward_loss(self, ac_pred: torch.Tensor, ac_gt: torch.Tensor, 
                     hp_seq_pred: torch.Tensor, hp_seq_gt: torch.Tensor,
                     goal_pred: torch.Tensor, next_edge_gt: torch.Tensor, 
                     reduce: bool=True) -> Tuple[torch.Tensor, dict]:
        assert ac_pred.shape[1:] == (self.model_output_dim, ) and ac_gt.shape[1:] == (4, )
        
        ac_gt = self._preprocess_action_gt(ac_gt)
        action_loss, action_info = self.action_loss(ac_pred, ac_gt, reduce)
        info = copy.deepcopy(action_info)
        loss_all = action_loss
        
        if self.use_helper:
            B, T, num_points,_ = hp_seq_gt.shape
            hp_seq_gt = self._preprocess_helper_point_gt(hp_seq_gt)
            assert hp_seq_pred.shape[1:] == (self.helper_net_npoints, 3), \
                f"(?, {self.helper_net_npoints}, 3) expected, ({hp_seq_pred.shape[1:]}) got"
            assert hp_seq_gt.shape[1:] == (self.seq_len, 32, 3)    
            helper_loss, helper_info = self.helper_loss(hp_seq_pred, hp_seq_gt.view(B*T, num_points, 3), reduce)
            info.update(helper_info)
            loss_all += helper_loss
            
        next_edge_gt = self._preprocess_next_edge_gt(next_edge_gt)  
        assert goal_pred.shape[1:] == (6, ) and next_edge_gt.shape[1:] == (6, )  
        goal_pred_loss, goal_pred_info = self.goal_pred_loss(goal_pred, next_edge_gt) 
        loss_all += goal_pred_loss
        info.update(goal_pred_info)
        return loss_all, info


class PointNetTransformerBC(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, 
                 optimizer_name="Adam", optimizer_kwargs={},
                 schedule_name="ExponentialLR", schedule_kwargs={"gamma": 1.0}, **kwargs):
        super().__init__()

        self.model = PointNetTransformerModule(**kwargs)

        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = copy.deepcopy(optimizer_kwargs)
        self.optimizer_kwargs["lr"] = self.learning_rate
        self.schedule_name = schedule_name
        self.schedule_kwargs = copy.deepcopy(schedule_kwargs)

        self.automatic_optimization = False
        self.save_hyperparameters()

        self.validation_step_outputs = {"action_gt": [], "action_class_gt": [], "action_class_pred": []}
        self.action_gt_is_logged = False

    def forward(self, pc_seq: torch.Tensor, pose_seq: torch.Tensor, cd: torch.Tensor, fp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward(pc_seq, pose_seq, cd, fp)

    def predict(self, pc_seq: torch.Tensor, pose_seq: torch.Tensor) -> torch.Tensor:
        return self.model.predict(pc_seq, pose_seq)
    
    def forward_loss(self, ac_pred, ac_gt, hp_seq_pred, hp_seq_gt, goal_pred, next_edge_gt,reduce=True) -> Tuple[torch.Tensor, dict]:
        return self.model.forward_loss(ac_pred, ac_gt, hp_seq_pred, hp_seq_gt, goal_pred, next_edge_gt, reduce)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(
            filter(lambda p: p.requires_grad, self.parameters()), **(self.optimizer_kwargs))
        if hasattr(torch.optim.lr_scheduler, self.schedule_name):
            schedule = getattr(torch.optim.lr_scheduler, self.schedule_name)(
                optimizer=optimizer, **(self.schedule_kwargs))
        else:
            schedule = getattr(transformers, self.schedule_name)(
                optimizer=optimizer, **(self.schedule_kwargs))
        return [optimizer], [schedule]

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        pc_seq, pose_seq, hp_seq_gt, ac_gt, data_idx, cd, fp, next_edge = batch
        ac_pred, hp_seq_pred, goal_pred = self.forward(pc_seq, pose_seq, cd, fp)
        train_loss, loss_info = self.forward_loss(ac_pred, ac_gt, hp_seq_pred, hp_seq_gt, goal_pred, next_edge)

        action_class_gt = torch.round(ac_gt[:, 0]).long()
        action_class_pred = torch.argmax(ac_pred[:, :self.model.action_type_num], dim=1)
        train_acc = torch.mean((action_class_gt == action_class_pred).float())

        self.log_dict({"train/loss": train_loss.detach().cpu(), "train/acc": train_acc.detach().cpu()})
        self.log_dict({"train/" + k: v for k, v in loss_info.items()})

        self.manual_backward(train_loss)
        opt.step()

    def validation_step(self, batch, batch_idx):
        pc_seq, pose_seq, hp_seq_gt, ac_gt, data_idx, cd, fp, next_edge = batch
        ac_pred, hp_seq_pred, goal_pred = self.forward(pc_seq, pose_seq, cd, fp)
        eval_loss, loss_info = self.forward_loss(ac_pred, ac_gt, hp_seq_pred, hp_seq_gt, goal_pred, next_edge)

        action_class_gt = torch.round(ac_gt[:, 0]).long()
        action_class_pred = torch.argmax(ac_pred[:, :self.model.action_type_num], dim=1)
        eval_acc = torch.mean((action_class_gt == action_class_pred).float())

        self.log_dict({"val/loss": eval_loss.detach().cpu(), "val/acc": eval_acc.detach().cpu()}, sync_dist=True)
        self.log_dict({"val/" + k: v for k, v in loss_info.items()}, sync_dist=True)

        self.validation_step_outputs["action_gt"].append(to_numpy(ac_gt))
        self.validation_step_outputs["action_class_gt"].append(to_numpy(action_class_gt))
        self.validation_step_outputs["action_class_pred"].append(to_numpy(action_class_pred))
    
    def on_validation_epoch_end(self):
        if self.global_step > 0:
            writer:SummaryWriter = self.logger.experiment
            log_classification(writer, 
                               np.concatenate(self.validation_step_outputs["action_class_gt"], axis=0),
                               np.concatenate(self.validation_step_outputs["action_class_pred"], axis=0),
                               self.model.anl, self.global_step)
            if not self.action_gt_is_logged:
                all_action = np.concatenate(self.validation_step_outputs["action_gt"], axis=0)
                log_action_distribution(writer, all_action, self.global_step, self.model.action_type_num, self.model.alt)
                self.action_gt_is_logged = True

        for key in self.validation_step_outputs.keys():
            self.validation_step_outputs[key].clear()

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        sch = self.lr_schedulers()
        sch.step() # update lr per batch

class State2ActionFintune(BaseFinetuning):
     def __init__(self, unfreeze_at_epoch=10):
         super().__init__()
         self._unfreeze_at_epoch = unfreeze_at_epoch

     def freeze_before_training(self, pl_module: PointNetTransformerBC):
         self.freeze(pl_module.model.state2action, train_bn= True)

     def finetune_function(self, pl_module: PointNetTransformerBC, current_epoch, optimizer):
         # When `current_epoch` is 10, feature_extractor will start training.
         if current_epoch == self._unfreeze_at_epoch:
             self.unfreeze_and_add_param_group(
                 modules=pl_module.model.state2action,
                 optimizer=optimizer,
                 train_bn=True,
             )
             
def get_args():
    parser = argparse.ArgumentParser()
    # hardware configuration
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpuid", "-g", type=int, nargs="*", default=-1)
    parser.add_argument("--precision", "-p", type=str,
                        default="high", choices=["highest", "high", "medium"])

    # directory configuration
    parser.add_argument("--common-data-path", "-dp", type=str, default="./rloutputs/data/numpy_data", 
                        help="common data path for training data and evaluation data")
    parser.add_argument("--compress-data-path", "-cdp", type=str, default='./rloutputs/data/test/compressed', 
                        help="common data path for training data and evaluation data")
    parser.add_argument("--aux-data-path", "-adp", type=str, default="./rloutputs/aux", 
                        help="common data path for training data and evaluation data")
    parser.add_argument("--train-data-path", "-tp", nargs="*", type=str, default=["train"],
                        help="specify where to find training data, enter 1 or several paths")
    parser.add_argument("--eval-data-path", "-ep", nargs="*", type=str, default=["eval"],
                        help="specify where to find evaluation data, enter 1 or several paths")
    parser.add_argument("--output-dir", "-o", type=str, default="./rloutputs/")
    parser.add_argument("--exp-name", "-en", type=str, default="test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-path", "-ckpt", type=str, default=None)
    parser.add_argument("--start_finetune_epoch", "-sfe", type=int, default=0)
    parser.add_argument("--model-yaml-path", "-y", type=str, default="./config/rl/bct_config.yaml")

    # model configuration
    parser.add_argument("--model-param", "-m", type=ast.literal_eval,
                        help='overwrite params in yaml, example: -m "{\'pn_output_dim\':64, \'loss_cfg\':{\'probability_weight\':1e1}}"')

    # train or evaluation
    parser.add_argument("--eval", action="store_true", help="only evaluation")

    # debug configuration
    parser.add_argument("--profiler", "-pf", type=str,
                        default="Simple", choices=["Advanced", "Simple", "PyTorch"])
    parser.add_argument("--log-every-n-steps", "-ls", type=int, default=50)
    parser.add_argument("--ckpt-every-n-steps", "-cs", type=int, default=10000)
    parser.add_argument("--val-check-interval", "-vi", type=int, default=1000, 
                        help="How often to check the validation set. Pass an int to check after a fixed number of training batches.")
    parser.add_argument("--limit-train-batches", "-ltb", type=float, default=1.0, help="use how much data to train")
    parser.add_argument("--limit-val-batches", "-lvb", type=float, default=0.5, help="use how much data to validate")

    # optimization configuration
    parser.add_argument("--optimizer-name", "-on", type=str, default="Adam")
    parser.add_argument("--optimizer-kwargs", "-oa", type=ast.literal_eval, default={})
    parser.add_argument("--schedule-name", "-sn", type=str, default="ExponentialLR")
    parser.add_argument("--schedule-kwargs", "-sa", type=ast.literal_eval, default={"gamma": 1.0})

    # training configuration
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--learning-rate", "-l", type=float, default=1e-4,
                        help="learning rate. lr_schedule and learning_rate will restart every time.")
    parser.add_argument("--disable-drop-last", "-ddl", action="store_true")
    parser.add_argument("--num-workers", "-n", type=int, default=2)
    parser.add_argument("--max-steps", "-s", type=int, default=60,
                        help="How many steps to train in this process. Does not include step_offset. Actually, last step is max_step + step_offset")
    parser.add_argument("--step-offset", "-so", type=int, default=0)

    # miscellaneous
    parser.add_argument("--seed", "-sd", type=int,
                        default=time.time_ns() % (2 ** 32))
    
    # modify global_variables
    parser.add_argument("--global-variable", "-gv", type=ast.literal_eval, default={})
    
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # modify global_variables
    for k, v in args.global_variable.items():
        exec(f"global {k}; {k}=v")

    # init logger
    logger = pl_loggers.TensorBoardLogger(args.output_dir, name=args.exp_name)
    os.makedirs(logger.log_dir)
    omegaconf.OmegaConf.save(
        omegaconf.DictConfig(
            {"command line": " ".join(sys.argv), "working dir": os.getcwd(), "args": vars(args)}),
        os.path.join(logger.log_dir, "command_line.yaml"))
    profiler_name2class = {"Advanced": AdvancedProfiler, "Simple": SimpleProfiler, "PyTorch": PyTorchProfiler}
    profiler = profiler_name2class[args.profiler](
        dirpath=logger.log_dir, filename="perf_logs")

    # make model configuration
    model_kwargs_overwrite = args.model_param
    if model_kwargs_overwrite is None:
        model_kwargs_overwrite = {}
    assert isinstance(model_kwargs_overwrite, dict)
    model_kwargs = omegaconf.OmegaConf.merge(omegaconf.OmegaConf.load(args.model_yaml_path),
                                             omegaconf.DictConfig(model_kwargs_overwrite))
    omegaconf.OmegaConf.save(model_kwargs, os.path.join(
        logger.log_dir, "model_config.yaml"))
    model_kwargs = omegaconf.OmegaConf.to_container(model_kwargs)
    lralg_kwargs = {"learning_rate": args.learning_rate,
                    "optimizer_name": args.optimizer_name, "optimizer_kwargs": args.optimizer_kwargs,
                    "schedule_name": args.schedule_name, "schedule_kwargs": args.schedule_kwargs}

    # init dataset and dataloader
    if model_kwargs["action_loss_cfg"]["rotation_dim"] in [3, 9]:
        action_type_num = 4 
    elif model_kwargs["action_loss_cfg"]["rotation_dim"] in [18]:
        action_type_num = 5
    else:
        raise NotImplementedError
    train_data, eval_data = get_bc_transformer_dataset(
        [os.path.join(args.common_data_path, tdp) for tdp in args.train_data_path],
        [os.path.join(args.common_data_path, edp) for edp in args.eval_data_path],
        model_kwargs["seq_len"], PRE_CUT_LEN, action_type_num,
        train_aux_path = [os.path.join(args.aux_data_path, tdp) for tdp in args.train_data_path],
        eval_aux_path = [os.path.join(args.aux_data_path, edp) for edp in args.eval_data_path],
        train_compress_path = [os.path.join(args.compress_data_path, tdp) for tdp in args.train_data_path] if args.compress_data_path else None,
        eval_compress_path = [os.path.join(args.compress_data_path, edp) for edp in args.eval_data_path] if args.compress_data_path else None)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=not args.disable_drop_last)
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=not args.disable_drop_last)

    # init numpy and pytorch, after get_bc_transformer_dataset
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision(args.precision)
    
    # init model
    if args.checkpoint_path is not None:
        model = PointNetTransformerBC.load_from_checkpoint(
            args.checkpoint_path, **model_kwargs, **lralg_kwargs)
    else:
        model = PointNetTransformerBC(**model_kwargs, **lralg_kwargs)
    model.example_input_array = \
        torch.randn([args.batch_size] + list(train_data[0][0].shape)), \
        torch.randn([args.batch_size] + list(train_data[0][1].shape)), \
        torch.randn([args.batch_size] + list(train_data[0][-3].shape)), \
        torch.randn([args.batch_size] + list(train_data[0][-2].shape))
    
    # init trainer
    trainer_kwargs = {
        "accelerator": "cuda" if args.cuda else "cpu",
        "devices": args.gpuid if args.cuda else "auto",
        "max_steps": args.max_steps + args.step_offset,
        "logger": logger,
        "profiler": profiler,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "log_every_n_steps": args.log_every_n_steps,
        "val_check_interval": args.val_check_interval,
        "check_val_every_n_epoch": None,
        "callbacks": [ModelCheckpoint(every_n_train_steps=args.ckpt_every_n_steps, save_top_k=-1), 
                      ModelSummary(max_depth=4), LearningRateMonitor(logging_interval='step'),
                      ModifyStepCallback(args.step_offset),
                      State2ActionFintune(args.start_finetune_epoch)],
    }
    trainer = pl.Trainer(**trainer_kwargs)

    # train
    if not args.eval:
        print("start fitting model...")
        ckpt_path = args.checkpoint_path if args.resume else None
        trainer.fit(model=model, train_dataloaders=train_loader,
                    val_dataloaders=eval_loader, ckpt_path = ckpt_path)
    else:
        trainer.validate(model=model, dataloaders=eval_loader)


if __name__ == "__main__":
    main()
