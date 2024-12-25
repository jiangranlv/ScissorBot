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
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor, Callback
import transformers

from rl.pointnet_utils import PointNetModule
from rl.loss_utils import ActionLoss, HelperPointLoss
from rl.pytorch_utils import *
from rl.prepare_dataset import DataForest, rotate2quat_numpy
from rl.so3 import rotate9D_to_angle_theta_phi

# global constants
POINT_DIM = 4
PRE_CUT_LEN = 5


class ModifyStepCallback(Callback):
    def __init__(self, step_offset=0) -> None:
        super().__init__()
        self.step_offset = step_offset

    def on_train_start(self, trainer, pl_module):
        trainer.fit_loop.epoch_loop._batches_that_stepped += self.step_offset # for logger
        trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed += self.step_offset # this change global_step, for ckpt name


class BehaviorCloningTransformerDataset(Dataset):
    def __init__(self, data_index_table: np.ndarray, seq_len: int, action_type_num: int, mode: Literal["train", "eval"],
                 point_cloud: DataForest, pose: DataForest, action: DataForest, helper_point: DataForest, info: DataForest) -> None:
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
        pc_seq = np.zeros((self._seq_len, 4096, POINT_DIM), np.float32) # [T, 4096, POINT_DIM]
        pose_seq = np.zeros((self._seq_len, 8), np.float32) # [T, 8] Rotate = quaternion
        hp_seq_gt = np.zeros((self._seq_len, 32, 3), np.float32) # [T, 32, 3]
        ac_gt: np.ndarray = np.load(self._ac[data_index].get_path()) # [4, ] Rotate = ANGLE THETA PHI
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
        assert pc_seq.shape == (self._seq_len, 4096, POINT_DIM), f"{pc_seq.shape}"
        assert pose_seq.shape == (self._seq_len, 8), f"{pose_seq.shape}"
        assert hp_seq_gt.shape == (self._seq_len, 32, 3), f"{hp_seq_gt.shape}"
        assert ac_gt.shape == (4, ), f"{ac_gt.shape}"
        assert data_idx.shape == (1, ), f"{data_idx.shape}"

        assert False not in np.isfinite(pc_seq)
        assert False not in np.isfinite(pose_seq)
        assert False not in np.isfinite(hp_seq_gt)
        assert False not in np.isfinite(ac_gt)
        assert False not in np.isfinite(data_idx)

        return pc_seq, pose_seq, hp_seq_gt, ac_gt, data_idx
    

def make_single_dataset(data_path: List[str], mode: Literal["train", "eval"], seq_len: int, 
                        pre_cut_len: int, action_type_num: int, verbose=1):
    if verbose == 1:
        print(f"scan all {mode} data ...")
    f_pc = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="point_cloud")
    f_pose = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="scissor_pose")
    f_action = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="action")
    f_hp = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="helper_point")
    f_info = DataForest(data_path, target_file_suffix=[".yaml"], info_file_suffix=[], target_file_name="info")
    
    all_data_size = len(f_pc)
    assert len(f_pose) == all_data_size and len(f_action) == all_data_size and \
        len(f_hp) == all_data_size and len(f_info) == all_data_size, \
            f"{len(f_pc)} {len(f_pose)} {len(f_action)} {len(f_hp)} {len(f_info)}"

    # drop pre_cut
    all_data_idx = []
    for i in range(all_data_size):
        filename = os.path.split(f_pc[i].get_path())[1]
        assert filename[-16:] == "_point_cloud.npy"
        if int(filename[:-16]) >= pre_cut_len:
            all_data_idx.append(i)
    all_data_idx = np.array(all_data_idx)

    if verbose == 1:
        print(f"make dataset {mode} ...")

    return BehaviorCloningTransformerDataset(all_data_idx, seq_len, action_type_num, mode, f_pc, f_pose, f_action, f_hp, f_info)
    

def get_bc_transformer_dataset(train_data_path: List[str], eval_data_path: List[str], seq_len: int, pre_cut_len: int, action_type_num: int):
    return make_single_dataset(train_data_path, "train", seq_len, pre_cut_len, action_type_num), \
        make_single_dataset(eval_data_path, "eval", seq_len, pre_cut_len, action_type_num),


class PoseTokenizer(nn.Module):
    def __init__(self, pose_token_dim: int, input_dim:int, hidden_dim=[64]) -> None:
        super().__init__()

        self.mlp_dim = [input_dim] + hidden_dim + [pose_token_dim]
        self.layer_n = len(hidden_dim) + 1
        self.fcs = nn.ModuleList([nn.Linear(self.mlp_dim[i], self.mlp_dim[i + 1]) for i in range(self.layer_n)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.mlp_dim[i + 1]) for i in range(self.layer_n - 1)])

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        for i in range(self.layer_n - 1):
            x = F.relu(self.bns[i](self.fcs[i](pose)))
        x = self.fcs[-1](x)
        return x


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

    def forward(self, feature, pose_token):
        return self.ln(torch.concatenate((feature, pose_token), dim=1))
    
    
class ActionDeTokenizer(nn.Module):
    def __init__(self, action_token_dim: int, output_dim: int, hidden_dim=[64]) -> None:
        super().__init__()

        self.mlp_dim = [action_token_dim] + hidden_dim + [output_dim]
        self.layer_n = len(hidden_dim) + 1
        self.fcs = nn.ModuleList([nn.Linear(self.mlp_dim[i], self.mlp_dim[i + 1]) for i in range(self.layer_n)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.mlp_dim[i + 1]) for i in range(self.layer_n - 1)])

    def forward(self, action_token: torch.Tensor) -> torch.Tensor:
        for i in range(self.layer_n - 1):
            x = F.relu(self.bns[i](self.fcs[i](action_token)))
        x = self.fcs[-1](x)
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


class PointNetTransformerModule(nn.Module):
    def __init__(self, seq_len=6, input_statistics={},
                 pn_output_dim=64, pose_token_dim=64, 
                 PN_cfg={}, helper_net_cfg={}, 
                 helper_loss_cfg={}, action_loss_cfg={}, 
                 pose_tokenizer_cfg={}, action_detokenizer_cfg={}, PE_cfg={},
                 transformer_cfg={"layer":{"nhead": 8, "dim_feedforward": 512}, "num_layers":6}) -> None:
        super().__init__()

        self.input_statistics = make_input_statistics(input_statistics)
        self.seq_len = seq_len
        self.token_dim = pn_output_dim + pose_token_dim

        self.net = PointNetModule(output_dim=pn_output_dim, **PN_cfg)
        self.helper_net = HelperNet(pn_output_dim, **helper_net_cfg)
        self.helper_net_npoints = self.helper_net.npoints
        self.helper_loss = HelperPointLoss(seq_len=seq_len, xyz_unit=get_pc_xyz_std_mean(input_statistics), **helper_loss_cfg)

        self.pose_tokenizer = PoseTokenizer(pose_token_dim=pose_token_dim, input_dim=8, **pose_tokenizer_cfg)
        self.pe = PositionalEncoding(d_model=self.token_dim, **PE_cfg)
        self.token_assembler = TokenAssembler(self.token_dim)

        transformer_layer = nn.TransformerEncoderLayer(batch_first=True, d_model=self.token_dim, **(transformer_cfg["layer"]))
        self.transformer = nn.TransformerEncoder(transformer_layer, transformer_cfg["num_layers"])
        
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
        self.action_detokenizer = ActionDeTokenizer(action_token_dim=self.token_dim*self.seq_len, output_dim=self.model_output_dim, **action_detokenizer_cfg)
        self.action_loss = ActionLoss(**action_loss_cfg, translation_unit=get_translation_std_mean(input_statistics))

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

    def _postprocess_action_pred(self, action_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_pred: [?, ACTION_PRED_DIM]
        Output:
            if rotation_dim in [3, 9]:
                action_pred: [?, 4 + 1 + 1 + 3 + 3 (12)]
            if rotation_dim == 18:
                action_pred: [?, 5 + 1 + 1 + 3 + 3 + 3 (16)]
        """
        idxt = self.alt["Translate"]["PredIndices"]
        mean = self.input_statistics["action"]["translation"]["mean"]
        std = self.input_statistics["action"]["translation"]["std"]
        action_pred = action_pred.clone()
        action_pred[..., idxt] = inverse_normalize_transform(action_pred[..., idxt], mean, std)
        if self.rotation_dim == 3:
            ret_val = action_pred.clone()
        elif self.rotation_dim == 9:
            ret_val = action_pred.clone()[..., :12]
            idxr = self.alt["Rotate"]["PredIndices"]
            ret_val[..., 9:12] = rotate9D_to_angle_theta_phi(action_pred[..., idxr])
        elif self.rotation_dim == 18:
            ret_val = action_pred.clone()[..., :16]
            idxr = self.alt["Rotate"]["PredIndices"]
            ret_val[..., 10:13] = rotate9D_to_angle_theta_phi(action_pred[..., idxr])
            idxr = self.alt["RotateSmall"]["PredIndices"]
            ret_val[..., 13:16] = rotate9D_to_angle_theta_phi(action_pred[..., idxr])
        else:
            raise NotImplementedError
        
        if self.rotation_dim == 18:
            assert ret_val.shape[-1] == 16, f"{ret_val.shape}"
        else:
            assert ret_val.shape[-1] == 12, f"{ret_val.shape}"
        return ret_val

    
    def _preprocess_input(self, pc: torch.Tensor, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pc: [?, 4096, POINT_DIM]
            pose: [?, 8]
        """
        pc = normalize_transform(pc, self.input_statistics["point_cloud"]["mean"],
                                 self.input_statistics["point_cloud"]["std"])
        pose = normalize_transform(pose, self.input_statistics["scissor_pose"]["mean"],
                                   self.input_statistics["scissor_pose"]["std"])
        return pc, pose

    def forward(self, pc_seq: torch.Tensor, pose_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # sanity check
        assert pc_seq.shape[1:] == (self.seq_len, 4096, POINT_DIM), f"(?, {self.seq_len}, 4096, {POINT_DIM}) expected, ({pc_seq.shape}) got"
        assert pose_seq.shape[1:] == (self.seq_len, 8), f"(?, {self.seq_len}, 8) expected, ({pose_seq.shape}) got"
        B, T = pc_seq.shape[0:2]

        pc, pose = self._preprocess_input(pc_seq.view(B*T, 4096, POINT_DIM), pose_seq.view(B*T, 8))
        feat: torch.Tensor = self.net(pc) # [B*T, self.feature_dim]
        hp_seq_pred: torch.Tensor = self.helper_net(feat).view(B, T, self.helper_net_npoints, 3) # [B, T, #helper_net_npoints, 3]

        pose_token: torch.Tensor = self.pose_tokenizer(pose) # [B*T, self.pose_token_dim]
        token_seq: torch.Tensor = self.token_assembler(feat, pose_token).view(B, T, self.token_dim) # [B, T, self.token_dim]
        transformer_output: torch.Tensor = self.transformer(self.pe(token_seq)) # [B, T, self.token_dim]
        ac_pred = self.action_detokenizer(transformer_output.view(B, T * self.token_dim)) # [B, ACTION_PRED_DIM]
        return ac_pred, hp_seq_pred

    def predict(self, pc_seq: torch.Tensor, pose_seq: torch.Tensor) -> torch.Tensor:
        ac_pred = self(pc_seq, pose_seq)[0]
        return self._postprocess_action_pred(ac_pred)

    def forward_loss(self, ac_pred: torch.Tensor, ac_gt: torch.Tensor, 
                     hp_seq_pred: torch.Tensor, hp_seq_gt: torch.Tensor,
                     reduce: bool=True) -> Tuple[torch.Tensor, dict]:
        assert ac_pred.shape[1:] == (self.model_output_dim, ) and ac_gt.shape[1:] == (4, )
        assert hp_seq_pred.shape[1:] == (self.seq_len, self.helper_net_npoints, 3) and hp_seq_gt.shape[1:] == (self.seq_len, 32, 3)
        ac_gt = self._preprocess_action_gt(ac_gt)
        hp_seq_gt = self._preprocess_helper_point_gt(hp_seq_gt)
        action_loss, action_info = self.action_loss(ac_pred, ac_gt, reduce)
        helper_loss, helper_info = self.helper_loss(hp_seq_pred, hp_seq_gt, reduce)
        info = copy.deepcopy(action_info)
        info.update(helper_info)
        return action_loss + helper_loss, info


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

    def forward(self, pc_seq: torch.Tensor, pose_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward(pc_seq, pose_seq)

    def predict(self, pc_seq: torch.Tensor, pose_seq: torch.Tensor) -> torch.Tensor:
        return self.model.predict(pc_seq, pose_seq)
    
    def forward_loss(self, ac_pred, ac_gt, hp_seq_pred, hp_seq_gt, reduce=True) -> Tuple[torch.Tensor, dict]:
        return self.model.forward_loss(ac_pred, ac_gt, hp_seq_pred, hp_seq_gt, reduce)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(
            self.parameters(), **(self.optimizer_kwargs))
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

        pc_seq, pose_seq, hp_seq_gt, ac_gt, data_idx = batch
        ac_pred, hp_seq_pred = self(pc_seq, pose_seq)
        train_loss, loss_info = self.forward_loss(ac_pred, ac_gt, hp_seq_pred, hp_seq_gt)

        action_class_gt = torch.round(ac_gt[:, 0]).long()
        action_class_pred = torch.argmax(ac_pred[:, :self.model.action_type_num], dim=1)
        train_acc = torch.mean((action_class_gt == action_class_pred).float())

        self.log_dict({"train/loss": train_loss.detach().cpu(), "train/acc": train_acc.detach().cpu()})
        self.log_dict({"train/" + k: v for k, v in loss_info.items()})

        self.manual_backward(train_loss)
        opt.step()

    def validation_step(self, batch, batch_idx):
        pc_seq, pose_seq, hp_seq_gt, ac_gt, data_idx = batch
        ac_pred, hp_seq_pred = self(pc_seq, pose_seq)
        eval_loss, loss_info = self.forward_loss(ac_pred, ac_gt, hp_seq_pred, hp_seq_gt)

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
    parser.add_argument("--train-data-path", "-tp", nargs="*", type=str, default=["train"],
                        help="specify where to find training data, enter 1 or several paths")
    parser.add_argument("--eval-data-path", "-ep", nargs="*", type=str, default=["eval"],
                        help="specify where to find evaluation data, enter 1 or several paths")
    parser.add_argument("--output-dir", "-o", type=str, default="./rloutputs/")
    parser.add_argument("--exp-name", "-en", type=str, default="test")
    parser.add_argument("--checkpoint-path", "-ckpt", type=str, default=None)
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
    parser.add_argument("--ckpt-every-n-steps", "-cs", type=int, default=1000)
    parser.add_argument("--val-check-interval", "-vi", type=int, default=1000, 
                        help="How often to check the validation set. Pass an int to check after a fixed number of training batches.")
    parser.add_argument("--limit-train-batches", "-ltb", type=float, default=1.0, help="use how much data to train")
    parser.add_argument("--limit-val-batches", "-lvb", type=float, default=1.0, help="use how much data to validate")

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
        model_kwargs["seq_len"], PRE_CUT_LEN, action_type_num)
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
        torch.randn([args.batch_size] + list(train_data[0][1].shape))
    
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
                      ModifyStepCallback(args.step_offset)],
    }
    trainer = pl.Trainer(**trainer_kwargs)

    # train
    if not args.eval:
        print("start fitting model...")
        trainer.fit(model=model, train_dataloaders=train_loader,
                    val_dataloaders=eval_loader)
    else:
        trainer.validate(model=model, dataloaders=eval_loader)


if __name__ == "__main__":
    main()
