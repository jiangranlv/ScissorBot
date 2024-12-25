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
from rl.bc_dataset import *
from rl.so3 import rotate9D_to_angle_theta_phi

# global constants
POINT_DIM = 4
PRE_CUT_LEN = 5
POINT_RES = 512

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
        
    

class PointNetTransformerModule(nn.Module):
    def __init__(self, seq_len=6, input_statistics={},
                 pn_output_dim=64, pose_token_dim=64, 
                 PN_cfg={}, helper_net_cfg={}, 
                 helper_loss_cfg={}, action_loss_cfg={}, 
                 pose_tokenizer_cfg={}, action_detokenizer_cfg={}, PE_cfg={},
                 transformer_cfg={"layer":{"nhead": 8, "dim_feedforward": 512}, "num_layers":6},
                 use_helper = False) -> None:
        super().__init__()

        self.input_statistics = make_input_statistics(input_statistics)
        self.seq_len = seq_len
        self.token_dim = pn_output_dim + pose_token_dim

        self.net = PointNetModule(output_dim=pn_output_dim, **PN_cfg)
        self.pose_tokenizer = MLP(output_dim=pose_token_dim, input_dim=8, **pose_tokenizer_cfg)
        
        self.use_helper = use_helper
        if use_helper:
            self.helper_net = MLP(self.token_dim, **helper_net_cfg)
            self.helper_net_npoints = self.helper_net.npoints
            self.helper_loss = HelperPointLoss(seq_len=seq_len, xyz_unit=get_pc_xyz_std_mean(input_statistics), **helper_loss_cfg)
        
        # self.transformer = TransformerModule(self.token_dim, PE_cfg, transformer_cfg)
        self.mlp = MLP(input_dim=self.token_dim*self.seq_len, output_dim=self.token_dim*self.seq_len, 
                       hidden_dim= [self.token_dim*self.seq_len, self.token_dim*self.seq_len])
        
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
        self.action_detokenizer = ActionDeTokenizer(action_token_dim=self.token_dim*self.seq_len, 
                num_classes= self.action_type_num, rotation_dim= self.rotation_dim, **action_detokenizer_cfg)
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
            pc: [?, POINT_RES, POINT_DIM]
            pose: [?, 8]
        """
        pc = normalize_transform(pc, self.input_statistics["point_cloud"]["mean"],
                                 self.input_statistics["point_cloud"]["std"])
        pose = normalize_transform(pose, self.input_statistics["scissor_pose"]["mean"],
                                   self.input_statistics["scissor_pose"]["std"])
        return pc, pose

    def forward(self, pc_seq: torch.Tensor, pose_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # sanity check
        assert pc_seq.shape[1:] == (self.seq_len, POINT_RES, POINT_DIM), f"(?, {self.seq_len}, POINT_RES, {POINT_DIM}) expected, ({pc_seq.shape}) got"
        assert pose_seq.shape[1:] == (self.seq_len, 8), f"(?, {self.seq_len}, 8) expected, ({pose_seq.shape}) got"
        B, T = pc_seq.shape[0:2]

        hp_seq_pred =None 
        
        pc, pose = self._preprocess_input(pc_seq.view(B*T, POINT_RES, POINT_DIM), pose_seq.view(B*T, 8))
        pc_token: torch.Tensor = self.net(pc) # [B*T, self.feature_dim]
        
        pose_token: torch.Tensor = self.pose_tokenizer(pose) # [B*T, self.pose_token_dim]
        # shallow_feat, transformer_output = self.transformer([pc_token.view(B, T, -1), pose_token.view(B, T, -1)]) # [B, T, self.token_dim]
        # flatten_output = transformer_output.view(B, T * self.token_dim)
        flatten_output = self.mlp(torch.concat([pc_token.view(B,-1), pose_token.view(B, -1)], dim = -1))
        
        if self.use_helper:
            obs_seq = torch.concat([pc_token, pose_token], dim = -1)
            hp_seq_pred: torch.Tensor = self.helper_net(obs_seq).view(B* T, self.helper_net_npoints, 3) # [B * T, #helper_net_npoints, 3]
            
        ac_pred = self.action_detokenizer(flatten_output) # [B, ACTION_PRED_DIM]
        
        return ac_pred, hp_seq_pred

    def predict(self, pc_seq: torch.Tensor, pose_seq: torch.Tensor) -> torch.Tensor:
        ac_pred = self(pc_seq, pose_seq)[0]
        return self._postprocess_action_pred(ac_pred)

    def forward_loss(self, ac_pred: torch.Tensor, ac_gt: torch.Tensor, 
                     hp_seq_pred: torch.Tensor, hp_seq_gt: torch.Tensor,
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
        ac_pred, hp_seq_pred = self.forward(pc_seq, pose_seq)
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
        ac_pred, hp_seq_pred = self.forward(pc_seq, pose_seq)
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
    parser.add_argument("--resume", action="store_true")
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


class ModifyStepCallback(Callback):
    def __init__(self, step_offset=0) -> None:
        super().__init__()
        self.step_offset = step_offset

    def on_train_start(self, trainer, pl_module):
        trainer.fit_loop.epoch_loop._batches_that_stepped += self.step_offset # for logger
        trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed += self.step_offset # this change global_step, for ckpt name

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
        model_kwargs["seq_len"], PRE_CUT_LEN, action_type_num,)
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
        ckpt_path = args.checkpoint_path if args.resume else None
        trainer.fit(model=model, train_dataloaders=train_loader,
                    val_dataloaders=eval_loader, ckpt_path = ckpt_path)
    else:
        trainer.validate(model=model, dataloaders=eval_loader)


if __name__ == "__main__":
    main()
