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
        
class RotationDataset(Dataset):
    def __init__(self, data_index_table: np.ndarray, rotation_df: DataForest, mode: Literal["train", "eval"],) -> None:
        super().__init__()

        self._data_index_table = data_index_table.copy()
        self._size = len(self._data_index_table)

        assert mode in ["train", "eval"]
        self._mode = mode
        self._rotation = rotation_df

        print(f"dataset {mode}: len {self._size}")

    def __len__(self):
        return self._size
    
    def __getitem__(self, index):
        data_index = int(self._data_index_table[index])
        rot_data = np.load(self._rotation[data_index].get_path(), allow_pickle=True).item()
        
        front_point: np.ndarray = rot_data["front_point"].astype(np.float32)
        cut_direction: np.ndarray = rot_data["cut_direction"].astype(np.float32)
        next_edge: np.ndarray = rot_data["next_edge"].astype(np.float32)
        action: np.ndarray = rot_data["action"].astype(np.float32)
        data_idx: np.ndarray = np.array([data_index], np.int64)

        # sanity check
        assert front_point.shape == (3, ), f"{front_point.shape}"
        assert cut_direction.shape == (3, ), f"{cut_direction.shape}"
        assert next_edge.shape == (2, 3), f"{next_edge.shape}"
        assert action.shape == (4, ), f"{action.shape}"
        assert data_idx.shape == (1, ), f"{data_idx.shape}"

        assert False not in np.isfinite(front_point)
        assert False not in np.isfinite(cut_direction)
        assert False not in np.isfinite(next_edge)
        assert False not in np.isfinite(action)
        assert False not in np.isfinite(data_idx)

        return front_point, cut_direction, next_edge, action, data_idx
    
def make_single_dataset(data_path: List[str], mode: Literal["train", "eval"], verbose=1):
    if verbose == 1:
        print(f"scan all {mode} data ...")
    df = DataForest(data_path, [".npy"], [], target_file_name="rotation")

    if verbose == 1:
        print(f"make dataset {mode} ...")

    return RotationDataset(np.arange(len(df)), df, mode)
    

def get_rotation_dataset(train_data_path: List[str], eval_data_path: List[str]):
    return make_single_dataset(train_data_path, "train"), \
        make_single_dataset(eval_data_path, "eval"),


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
    
class State2RotationModule(nn.Module):
    def __init__(self, input_statistics={}, feat_dim = 64, action_loss_cfg={}) -> None:
        super().__init__()
        
        self.alt = ActionLookupTable_for9Drot
        self.action_type_num = 4

        self.input_statistics = make_input_statistics(input_statistics)
        self.model_output_dim = action_loss_cfg["rotation_dim"] + 9
        self.position_net = MLP(input_dim= 3, output_dim= feat_dim, hidden_dim= [])
        self.goal_net = MLP(input_dim= 3, output_dim= feat_dim, hidden_dim= [])
        self.direction_net = MLP(input_dim= 3, output_dim= feat_dim, hidden_dim= [])
        self.merge_net = MLP(input_dim= feat_dim * 2, output_dim= feat_dim, hidden_dim= [])
        self.rotation_head = MLP(input_dim= feat_dim * 2, output_dim= 9, hidden_dim= [feat_dim, feat_dim], act_func= 'leaky_relu')

        self.action_loss = ActionLoss(**action_loss_cfg, translation_unit=get_translation_std_mean(input_statistics))

    def _preprocess_action_gt(self, action_gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_gt: [?, 4]
        """
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

    
    def _preprocess_input(self, front_point, cut_direction, next_edge_b) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pc: [?, 4096, POINT_DIM]
            pose: [?, 8]
        """
        front_point = normalize_transform(front_point, self.input_statistics["front_point"]["mean"],
                                 self.input_statistics["front_point"]["std"])
        cut_direction = normalize_transform(cut_direction, self.input_statistics["cut_direction"]["mean"],
                                   self.input_statistics["cut_direction"]["std"])
        next_edge_b = normalize_transform(next_edge_b, self.input_statistics["next_edge"]["b"]["mean"],
                                   self.input_statistics["next_edge"]["b"]["std"])
        return front_point, cut_direction, next_edge_b

    def forward(self, front_point:torch.Tensor, cut_direction:torch.Tensor, next_edge_b:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # sanity check
        assert front_point.shape[1:] == (3, ), f"(?, 3) expected, ({front_point.shape}) got"
        assert cut_direction.shape[1:] == (3, ), f"(?, 3) expected, ({cut_direction.shape}) got"
        assert next_edge_b.shape[1:] == (3, ), f"(?, 3) expected, ({next_edge_b.shape}) got"
        B = front_point.shape[0]

        front_point, cut_direction, next_edge_b = self._preprocess_input(front_point, cut_direction, next_edge_b)

        pos_feat = self.position_net(front_point)
        goal_feat = self.goal_net(next_edge_b)
        dir_feat = self.direction_net(cut_direction)
        mid_feat = self.merge_net(torch.concat([pos_feat, goal_feat], dim = -1))
        rotation = self.rotation_head(torch.concat([dir_feat, mid_feat], dim = -1), head = True)
        
        left = torch.zeros([B, 9]).to(rotation.device)
        ac_pred = torch.concat([left, rotation], dim = -1)
        return ac_pred
    

    def forward_loss(self, ac_pred: torch.Tensor, ac_gt: torch.Tensor, reduce: bool=True) -> Tuple[torch.Tensor, dict]:
        assert ac_pred.shape[1:] == (self.model_output_dim, ), f"(?, {self.model_output_dim}) expected, {ac_pred.shape} got"
        assert ac_gt.shape[1:] == (4, ), f"(?, 4) expected, {ac_gt.shape} got"
        ac_gt = self._preprocess_action_gt(ac_gt)
        action_loss, action_info = self.action_loss(ac_pred, ac_gt, reduce)
        info = copy.deepcopy(action_info)
        return action_loss, info
    
class State2Rotation(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, 
                 optimizer_name="Adam", optimizer_kwargs={},
                 schedule_name="ExponentialLR", schedule_kwargs={"gamma": 1.0}, **kwargs):
        super().__init__()

        self.model = State2RotationModule(**kwargs)

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

    def forward(self, front_point, cut_direction, next_edge_b) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward(front_point, cut_direction, next_edge_b)

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

        front_point, cut_direction, next_edge, ac_gt, data_idx = batch
        next_edge_b = next_edge[:, 1, :]
        ac_pred = self.forward(front_point, cut_direction, next_edge_b)
        train_loss, loss_info = self.forward_loss(ac_pred, ac_gt)

        self.log_dict({"train/loss": train_loss.detach().cpu()})
        self.log_dict({"train/" + k: v for k, v in loss_info.items()})

        self.manual_backward(train_loss)
        opt.step()

    def validation_step(self, batch, batch_idx):
        front_point, cut_direction, next_edge, ac_gt, data_idx = batch
        next_edge = next_edge[:, 1,:]
        ac_pred = self(front_point, cut_direction, next_edge)
        eval_loss, loss_info = self.forward_loss(ac_pred, ac_gt)


        self.log_dict({"val/loss": eval_loss.detach().cpu()}, sync_dist=True)
        self.log_dict({"val/" + k: v for k, v in loss_info.items()}, sync_dist=True)

        self.validation_step_outputs["action_gt"].append(to_numpy(ac_gt))
    
    # def on_validation_epoch_end(self):
    #     if self.global_step > 0:
    #         writer:SummaryWriter = self.logger.experiment
    #         if not self.action_gt_is_logged:
    #             all_action = np.concatenate(self.validation_step_outputs["action_gt"], axis=0)
    #             log_action_distribution(writer, all_action, self.global_step, self.model.action_type_num, self.model.alt)
    #             self.action_gt_is_logged = True

    #     for key in self.validation_step_outputs.keys():
    #         self.validation_step_outputs[key].clear()

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
    parser.add_argument("--common-data-path", "-dp", type=str, default="./rloutputs/rotation", 
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
    train_data, eval_data = get_rotation_dataset(
        [os.path.join(args.common_data_path, tdp) for tdp in args.train_data_path],
        [os.path.join(args.common_data_path, edp) for edp in args.eval_data_path])
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
        model = State2Rotation.load_from_checkpoint(
            args.checkpoint_path, **model_kwargs, **lralg_kwargs)
    else:
        model = State2Rotation(**model_kwargs, **lralg_kwargs)
    model.example_input_array = \
        torch.randn([args.batch_size] + list(train_data[0][0].shape)), \
        torch.randn([args.batch_size] + list(train_data[0][1].shape)), \
        torch.randn([args.batch_size] + list(train_data[0][2][1].shape)),
    
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