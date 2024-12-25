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

from rl.loss_utils import ActionLoss
from rl.pytorch_utils import *
from rl.prepare_dataset import DataForest

class ModifyStepCallback(Callback):
    def __init__(self, step_offset=0) -> None:
        super().__init__()
        self.step_offset = step_offset

    def on_train_start(self, trainer, pl_module):
        trainer.fit_loop.epoch_loop._batches_that_stepped += self.step_offset # for logger
        trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed += self.step_offset # this change global_step, for ckpt name
        
class StateActionDataset(Dataset):
    def __init__(self, data_index_table: np.ndarray, pose_df:DataForest, action_df: DataForest, aux_df: DataForest, info:DataForest,  
                    mode: Literal["train", "eval"], action_type_num = 5) -> None:
        super().__init__()

        self._data_index_table = data_index_table.copy()
        self._size = len(self._data_index_table)

        assert mode in ["train", "eval"]
        self._mode = mode
        self._pose = pose_df
        self._action = action_df
        self._aux = aux_df
        self._action_type_num = action_type_num
        self._info = info

        print(f"dataset {mode}: len {self._size}")

    def __len__(self):
        return self._size
    
    def __getitem__(self, index):
        data_index = int(self._data_index_table[index])
        pose: np.ndarray = np.load(self._pose[data_index].get_path())
        action: np.ndarray= np.load(self._action[data_index].get_path())
        aux_data = np.load(self._aux[data_index].get_path(), allow_pickle= True).tolist()
        front_point: np.ndarray = aux_data["front_point"].astype(np.float32)
        cut_direction: np.ndarray = aux_data["cut_direction"].astype(np.float32)
        next_edge: np.ndarray = aux_data["next_edge"].astype(np.float32)
        data_idx: np.ndarray = np.array([data_index], np.int64)
        
        if self._action_type_num == 5:
            prev_data_index = data_index - 1
            if prev_data_index >= 0 and self._is_same_trajectory(prev_data_index, data_index):
                prev_ac_type = np.load(self._action[prev_data_index].get_path())[0]
                if self._is_small_rotation(prev_ac_type, action):
                    action[0] = 4.0
            
        # sanity check
        assert front_point.shape == (3, ), f"{front_point.shape}"
        assert cut_direction.shape == (3, ), f"{cut_direction.shape}"
        assert next_edge.shape == (2, 3), f"{next_edge.shape}"
        assert action.shape == (4, ), f"{action.shape}"
        assert data_idx.shape == (1, ), f"{data_idx.shape}"
        assert pose.shape == (8, ), f"{pose.shape}"

        assert False not in np.isfinite(front_point)
        assert False not in np.isfinite(cut_direction)
        assert False not in np.isfinite(next_edge)
        assert False not in np.isfinite(action)
        assert False not in np.isfinite(data_idx)
        assert False not in np.isfinite(pose)

        return pose, front_point, cut_direction, next_edge, action, data_idx
    
    def _is_small_rotation(self, prev_ac_type: int, ac_gt: np.ndarray) -> bool:
        old_rotation_id = 3
        old_open_id = 0
        return ac_gt[0] == old_rotation_id and prev_ac_type == old_open_id
    
    def _is_same_trajectory(self, data_index1:int, data_index2:int):
        return os.path.dirname(self._info[data_index1].get_path()) == \
            os.path.dirname(self._info[data_index2].get_path())
            
def make_single_dataset(data_path: List[str], mode: Literal["train", "eval"], aux_path = List[str], verbose=1, pre_cut_len = 5):
    if verbose == 1:
        print(f"scan all {mode} data ...") 
    
    pose_df = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="scissor_pose")
    action_df = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="action")
    aux_df = DataForest(aux_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="auxiliary")
    f_info = DataForest(data_path, target_file_suffix=[".yaml"], info_file_suffix=[], target_file_name="info")
    if verbose == 1:
        print(f"make dataset {mode} ...")

    all_data_size = len(pose_df)
    assert len(action_df) == all_data_size and len(aux_df) == all_data_size ,\
            f"{len(action_df)} {len(aux_df)} "
            
    # drop pre_cut
    all_data_idx = []
    for i in range(all_data_size):
        filename = os.path.split(f_info[i].get_path())[1]
        assert filename[-10:] == "_info.yaml"
        if int(filename[:-10]) >= pre_cut_len:
            all_data_idx.append(i)
    all_data_idx = np.array(all_data_idx)
            
    return StateActionDataset(all_data_idx, pose_df, action_df, aux_df, f_info, mode)
    

def get_rotation_dataset(train_data_path: List[str], eval_data_path: List[str], train_aux_path: List[str], eval_aux_path: List[str]):
    return make_single_dataset(train_data_path, "train", train_aux_path), \
        make_single_dataset(eval_data_path, "eval", eval_aux_path),


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
        
class State2ActionModule(nn.Module):
    def __init__(self, input_statistics={}, feat_dim = 128, action_loss_cfg={}, action_detokenizer_cfg ={}) -> None:
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
        self.position_net = MLP(input_dim= 4, output_dim= feat_dim, hidden_dim= [feat_dim])
        self.goal_net = MLP(input_dim= 6, output_dim= feat_dim, hidden_dim= [feat_dim])
        self.direction_net = MLP(input_dim= 3, output_dim= feat_dim, hidden_dim= [feat_dim])
        self.merge_net = MLP(input_dim= feat_dim * 2, output_dim= feat_dim, hidden_dim= [feat_dim])
        self.action_detokenizer = ActionDeTokenizer(action_token_dim=feat_dim * 2, 
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
        
        return None

    
    def _preprocess_input(self, pose, front_point, cut_direction, next_edge_a, next_edge_b) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pc: [?, 4096, POINT_DIM]
            pose: [?, 8]
        """
        pose = normalize_transform(pose, self.input_statistics["scissor_pose"]["mean"],
                                 self.input_statistics["scissor_pose"]["std"])
        front_point = normalize_transform(front_point, self.input_statistics["front_point"]["mean"],
                                 self.input_statistics["front_point"]["std"])
        cut_direction = normalize_transform(cut_direction, self.input_statistics["cut_direction"]["mean"],
                                   self.input_statistics["cut_direction"]["std"])
        next_edge_a = normalize_transform(next_edge_a, self.input_statistics["next_edge"]["a"]["mean"],
                                   self.input_statistics["next_edge"]["a"]["std"])
        next_edge_b = normalize_transform(next_edge_b, self.input_statistics["next_edge"]["b"]["mean"],
                                   self.input_statistics["next_edge"]["b"]["std"])
        return pose, front_point, cut_direction, next_edge_a, next_edge_b

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

        pose, front_point, cut_direction, next_edge_a,next_edge_b = self._preprocess_input(
            pose, front_point, cut_direction, next_edge_a, next_edge_b)

        pos_feat = self.position_net(torch.concat([front_point, pose[:, 0, None]], dim = -1))
        goal_feat = self.goal_net(torch.concat([next_edge_a,next_edge_b], dim = -1))
        dir_feat = self.direction_net(cut_direction)
        mid_feat = self.merge_net(torch.concat([pos_feat, goal_feat], dim = -1))
        ac_pred = self.action_detokenizer(torch.concat([dir_feat, mid_feat], dim = -1))
        
        return ac_pred
    

    def forward_loss(self, ac_pred: torch.Tensor, ac_gt: torch.Tensor, reduce: bool=True) -> Tuple[torch.Tensor, dict]:
        assert ac_pred.shape[1:] == (self.model_output_dim, ), f"(?, {self.model_output_dim}) expected, {ac_pred.shape} got"
        assert ac_gt.shape[1:] == (4, ), f"(?, 4) expected, {ac_gt.shape} got"
        ac_gt = self._preprocess_action_gt(ac_gt)
        action_loss, action_info = self.action_loss(ac_pred, ac_gt, reduce)
        info = copy.deepcopy(action_info)
        return action_loss, info
    
class State2Action(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, 
                 optimizer_name="Adam", optimizer_kwargs={},
                 schedule_name="ExponentialLR", schedule_kwargs={"gamma": 1.0}, **kwargs):
        super().__init__()

        self.model = State2ActionModule(**kwargs)

        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = copy.deepcopy(optimizer_kwargs)
        self.optimizer_kwargs["lr"] = self.learning_rate
        self.schedule_name = schedule_name
        self.schedule_kwargs = copy.deepcopy(schedule_kwargs)

        self.automatic_optimization = False
        self.save_hyperparameters()

        self.validation_step_outputs = {"action_gt": []}
        self.action_gt_is_logged = False

    def forward(self, pose, front_point, cut_direction, next_edge_a, next_edge_b) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward(pose, front_point, cut_direction, next_edge_a, next_edge_b)

    # def predict(self, front_point, cut_direction, next_edge) -> torch.Tensor:
    #     return self.model.predict(front_point, cut_direction, next_edge)
    
    def forward_loss(self, ac_pred, ac_gt, reduce=True) -> Tuple[torch.Tensor, dict]:
        return self.model.forward_loss(ac_pred, ac_gt, reduce)

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

        pose, front_point, cut_direction, next_edge, ac_gt, data_idx = batch
        next_edge_a = next_edge[:, 0, :]
        next_edge_b = next_edge[:, 1, :]
        ac_pred = self.forward(pose, front_point, cut_direction, next_edge_a, next_edge_b)
        train_loss, loss_info = self.forward_loss(ac_pred, ac_gt)
        self.log_dict({"train/loss": train_loss.detach().cpu()})
        self.log_dict({"train/" + k: v for k, v in loss_info.items()})

        self.manual_backward(train_loss)
        opt.step()

    def validation_step(self, batch, batch_idx):
        pose, front_point, cut_direction, next_edge, ac_gt, data_idx = batch
        next_edge_a = next_edge[:, 0, :]
        next_edge_b = next_edge[:, 1, :]
        ac_pred = self.forward(pose, front_point, cut_direction, next_edge_a, next_edge_b)
        eval_loss, loss_info = self.forward_loss(ac_pred, ac_gt)


        self.log_dict({"val/loss": eval_loss.detach().cpu()}, sync_dist=True)
        self.log_dict({"val/" + k: v for k, v in loss_info.items()}, sync_dist=True)

        self.validation_step_outputs["action_gt"].append(to_numpy(ac_gt))
    
    def on_validation_epoch_end(self):
        if self.global_step > 0:
            writer:SummaryWriter = self.logger.experiment
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
    parser.add_argument("--aux-data-path", "-adp", type=str, default="./rloutputs/aux", 
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
    parser.add_argument("--ckpt-every-n-steps", "-cs", type=int, default=10000)
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
    train_data, eval_data = get_rotation_dataset(
        [os.path.join(args.common_data_path, tdp) for tdp in args.train_data_path],
        [os.path.join(args.common_data_path, edp) for edp in args.eval_data_path],
        [os.path.join(args.aux_data_path, tdp) for tdp in args.train_data_path],
        [os.path.join(args.aux_data_path, edp) for edp in args.eval_data_path])
    
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
        model = State2Action.load_from_checkpoint(
            args.checkpoint_path, **model_kwargs, **lralg_kwargs)
    else:
        model = State2Action(**model_kwargs, **lralg_kwargs)
    model.example_input_array = \
        torch.randn([args.batch_size] + list(train_data[0][0].shape)), \
        torch.randn([args.batch_size] + list(train_data[0][1].shape)), \
        torch.randn([args.batch_size] + list(train_data[0][2].shape)), \
        torch.randn([args.batch_size] + list(train_data[0][3][0].shape)),\
        torch.randn([args.batch_size] + list(train_data[0][3][1].shape)),
    
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