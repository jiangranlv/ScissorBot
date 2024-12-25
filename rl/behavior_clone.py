import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import ast
from typing import Literal, Tuple
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

from rl.pointnet_utils import PointNetModule
from rl.pytorch_utils import *
from rl.prepare_dataset import DataForest, rotate2quat_numpy
from rl.loss_utils import ActionLoss


action_8D_translate_slice = slice(1, 4)
def encode_action_4D_to_8D(action: np.ndarray) -> np.ndarray:
    """
    Args:
        action [4, ]
    Return:
        action8D [8, ]
    """
    assert action.shape == (4, ), f"action.shape={action.shape}"
    ret = np.zeros(8, np.float32)
    if ActionNameList[int(np.round(action[0]))] == "Open":
        ret[0] = action[1]
    elif ActionNameList[int(np.round(action[0]))] == "Close":
        ret[0] = -action[1]
    elif ActionNameList[int(np.round(action[0]))] == "Translate":
        ret[1:4] = action[1:4]
    elif ActionNameList[int(np.round(action[0]))] == "Rotate":
        ret[4:8] = rotate2quat_numpy(action[1:4])
    else:
        raise ValueError(f"Invalid action {action}")
    return ret


class BehaviorCloningDataset(Dataset):
    def __init__(self, data_index_table: np.ndarray, prev_action_n: int, mode: Literal["train", "eval"],
                 point_cloud: DataForest, pose: DataForest, action: DataForest, info: DataForest) -> None:
        super().__init__()

        self._data_index_table = data_index_table.copy()
        self._size = len(self._data_index_table)
        self._prev_action_n = prev_action_n

        assert mode in ["train", "eval"]
        self._mode = mode
        self._pc = point_cloud
        self._pose = pose
        self._ac = action
        self._info = info

        print(f"dataset {mode}: len {self._size}")

    def __len__(self):
        return self._size

    def _is_same_trajectory(self, data_index1:int, data_index2:int):
        return os.path.dirname(self._info[data_index1].get_path()) == \
            os.path.dirname(self._info[data_index2].get_path())

    def __getitem__(self, index):
        data_index = int(self._data_index_table[index])
        pc: np.ndarray = np.load(self._pc[data_index].get_path()) # [4096, 6]
        pose: np.ndarray = np.load(self._pose[data_index].get_path()) # [8, ] Rotate = quaternion
        prev_ac = np.zeros((self._prev_action_n, 8), np.float32) # [prev_action, 8] Rotate = quaternion
        ac: np.ndarray = np.load(self._ac[data_index].get_path()) # [4, ] Rotate = ANGLE THETA PHI
        for i, prev_data_index in enumerate(range(data_index-self._prev_action_n, data_index)):
            if 0 <= prev_data_index and prev_data_index < len(self._ac) and \
                self._is_same_trajectory(prev_data_index, data_index):
                prev_ac[i] = encode_action_4D_to_8D(np.load(self._ac[prev_data_index].get_path()))
            else:
                prev_ac[i] = np.zeros(8, np.float32)

        # sanity check
        assert pc.shape == (4096, 6)
        assert pose.shape == (8, )
        assert prev_ac.shape == (self._prev_action_n, 8)
        assert ac.shape == (4, )

        assert False not in np.isfinite(pc)
        assert False not in np.isfinite(pose)
        assert False not in np.isfinite(prev_ac)
        assert False not in np.isfinite(ac)

        return pc, pose, prev_ac, ac
    

def get_bc_dataset(data_path: str, prev_action_n:int, eval_size=30000, seed=0, verbose=1):
    if verbose == 1:
        print("scan all data ...")
    f_pc = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="point_cloud")
    f_pose = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="scissor_pose")
    f_action = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="action")
    f_info = DataForest(data_path, target_file_suffix=[".yaml"], info_file_suffix=[], target_file_name="info")
    
    data_size = len(f_pc)
    assert len(f_pose) == data_size and len(f_action) == data_size
    train_size = data_size - eval_size
    assert eval_size >= 1 and train_size >= 1

    np.random.seed(seed)
    all_data_idx = np.random.permutation(data_size)
    train_data_idx = all_data_idx[:train_size]
    eval_data_idx = all_data_idx[train_size:]
    
    if verbose == 1:
        print("make dataset ...")
    train_data = BehaviorCloningDataset(train_data_idx, prev_action_n, "train", f_pc, f_pose, f_action, f_info)
    eval_data = BehaviorCloningDataset(eval_data_idx, prev_action_n, "eval", f_pc, f_pose, f_action, f_info)
    return train_data, eval_data

class ModifyStepCallback(Callback):
    def __init__(self, step_offset=0) -> None:
        super().__init__()
        self.step_offset = step_offset

    def on_train_start(self, trainer, pl_module):
        trainer.fit_loop.epoch_loop._batches_that_stepped += self.step_offset # for logger
        trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed += self.step_offset # this change global_step, for ckpt name


class PoseMLP(nn.Module):
    def __init__(self, feat_dim, prev_action_size, pose_dim, output_dim,
                 hidden_dim=[64, 64]) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.prev_action_size = prev_action_size
        self.pose_dim = pose_dim
        self.output_dim = output_dim

        self.layer_n = len(hidden_dim) + 1
        self.mlp_dim = [feat_dim] + [i for i in hidden_dim] + [self.output_dim]

        self.fcs = nn.ModuleList([nn.Linear(self.mlp_dim[i] + self.pose_dim + self.prev_action_size,
                                            self.mlp_dim[i + 1]) for i in range(self.layer_n)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(self.mlp_dim[i + 1]) for i in range(self.layer_n - 1)])
        self.feat_bn = nn.BatchNorm1d(self.feat_dim)

    def forward(self, feat: torch.Tensor, pose: torch.Tensor, prev_action: torch.Tensor):
        prev_action = prev_action.view(prev_action.shape[0], self.prev_action_size)
        x = F.relu(self.feat_bn(feat))
        for i in range(self.layer_n - 1):
            x = F.relu(self.bns[i](self.fcs[i](
                torch.concatenate((x, pose, prev_action), dim=1))))
        x = self.fcs[-1](torch.concatenate((x, pose, prev_action), dim=1))
        return x


class PointNetMLPModule(nn.Module):
    def __init__(self, pn_output_dim=32, prev_action_n=1, input_statistics={},
                 PN_cfg={}, posemlp_cfg={}, loss_cfg={}) -> None:
        super().__init__()
        self.net = PointNetModule(output_dim=pn_output_dim, **PN_cfg)
        self.prev_action_n = prev_action_n
        self.posemlp = PoseMLP(feat_dim=pn_output_dim,
                               prev_action_size=prev_action_n*8, pose_dim=8, output_dim=12, **posemlp_cfg)
        self.loss = ActionLoss(**loss_cfg)
        self.input_statistics = make_input_statistics(input_statistics)

    def _preprocess_action_gt(self, action_gt: torch.Tensor) -> torch.Tensor:
        action_idx = torch.round(action_gt[..., 0]).long()
        idx1 = torch.where(action_idx == ActionLookupTable["Translate"]["ActionID"])[0][:, None]
        idx2 = torch.tensor(ActionLookupTable["Translate"]["TargetIndices"], dtype=torch.int64)[None, :]
        mean = self.input_statistics["action"]["translation"]["mean"]
        std = self.input_statistics["action"]["translation"]["std"]
        action_gt = action_gt.clone()
        action_gt[idx1, idx2] = normalize_transform(action_gt[idx1, idx2], mean, std)
        return action_gt

    def _postprocess_action_pred(self, action_pred: torch.Tensor) -> torch.Tensor:
        idx2 = ActionLookupTable["Translate"]["PredIndices"]
        mean = self.input_statistics["action"]["translation"]["mean"]
        std = self.input_statistics["action"]["translation"]["std"]
        action_pred = action_pred.clone()
        action_pred[..., idx2] = inverse_normalize_transform(action_pred[..., idx2], mean, std)
        return action_pred
    
    def _preprocess_input(self, pc: torch.Tensor, pose: torch.Tensor, prev_ac: torch.Tensor) -> torch.Tensor:
        pc = normalize_transform(pc, self.input_statistics["point_cloud"]["mean"],
                                 self.input_statistics["point_cloud"]["std"])
        pose = normalize_transform(pose, self.input_statistics["scissor_pose"]["mean"],
                                   self.input_statistics["scissor_pose"]["std"])
        prev_ac = prev_ac.clone()
        prev_ac[..., action_8D_translate_slice] = normalize_transform(prev_ac[..., action_8D_translate_slice],
            self.input_statistics["action"]["translation"]["mean"], self.input_statistics["action"]["translation"]["std"])
        return pc, pose, prev_ac

    def forward(self, pc: torch.Tensor, pose: torch.Tensor, prev_ac: torch.Tensor) -> torch.Tensor:
        # sanity check
        assert pc.shape[1:] == (4096, 6), f"(?, 4096, 6) expected, ({pc.shape}) got"
        assert pose.shape[1:] == (8, ), f"(?, 8) expected, ({pose.shape}) got"
        assert prev_ac.shape[1:] == (self.prev_action_n, 8), f"(?, {self.prev_action_n}, 8) expected, ({prev_ac.shape}) got"
        
        pc, pose, prev_ac = self._preprocess_input(pc, pose, prev_ac)
        return self.posemlp(self.net(pc), pose, prev_ac)

    def predict(self, pc: torch.Tensor, pose: torch.Tensor, prev_ac: torch.Tensor) -> torch.Tensor:
        ac_pred = self.forward(pc, pose, prev_ac)
        return self._postprocess_action_pred(ac_pred)

    def forward_loss(self, ac_pred, ac_gt) -> Tuple[torch.Tensor, dict]:
        ac_gt = self._preprocess_action_gt(ac_gt)
        return self.loss(ac_pred, ac_gt)



class PointNetBC(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, 
                 optimizer_name="Adam", optimizer_kwargs={},
                 schedule_name="ExponentialLR", schedule_kwargs={"gamma": 1.0}, **kwargs):
        super().__init__()
        self.model = PointNetMLPModule(**kwargs)

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

    def forward(self, pc: torch.Tensor, pose: torch.Tensor, prev_ac: torch.Tensor) -> torch.Tensor:
        # sanity check
        return self.model.forward(pc, pose, prev_ac)

    def predict(self, pc: torch.Tensor, pose: torch.Tensor, prev_ac: torch.Tensor) -> torch.Tensor:
        return self.model.predict(pc, pose, prev_ac)

    def forward_loss(self, ac_pred, ac_gt) -> Tuple[torch.Tensor, dict]:
        return self.model.forward_loss(ac_pred, ac_gt)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(
            self.parameters(), **(self.optimizer_kwargs))
        schedule = getattr(torch.optim.lr_scheduler, self.schedule_name)(
            optimizer=optimizer, **(self.schedule_kwargs))
        return [optimizer], [schedule]

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        pc, pose, prev_ac, ac = batch
        ac_hat = self.forward(pc, pose, prev_ac)
        train_loss, loss_info = self.forward_loss(ac_hat, ac)

        action_class_gt = torch.round(ac[:, 0]).long()
        action_class_pred = torch.argmax(ac_hat[:, :4], dim=1)
        train_acc = torch.mean((action_class_gt == action_class_pred).float())

        self.log_dict({"train/loss": train_loss.detach().cpu(), "train/acc": train_acc.detach().cpu()})
        self.log_dict({"train/" + k: v for k, v in loss_info.items()})

        self.manual_backward(train_loss)
        opt.step()

    def validation_step(self, batch, batch_idx):
        pc, pose, prev_ac, ac = batch
        ac_hat = self.forward(pc, pose, prev_ac)
        eval_loss, loss_info = self.forward_loss(ac_hat, ac)

        action_class_gt = torch.round(ac[:, 0]).long()
        action_class_pred = torch.argmax(ac_hat[:, :4], dim=1)
        eval_acc = torch.mean((action_class_gt == action_class_pred).float())

        self.log_dict({"val/loss": eval_loss.detach().cpu(), "val/acc": eval_acc.detach().cpu()}, sync_dist=True)
        self.log_dict({"val/" + k: v for k, v in loss_info.items()}, sync_dist=True)

        self.validation_step_outputs["action_gt"].append(to_numpy(ac))
        self.validation_step_outputs["action_class_gt"].append(to_numpy(action_class_gt))
        self.validation_step_outputs["action_class_pred"].append(to_numpy(action_class_pred))
    
    def on_validation_epoch_end(self):
        if self.global_step > 0:
            writer:SummaryWriter = self.logger.experiment
            log_classification(writer, 
                               np.concatenate(self.validation_step_outputs["action_class_gt"], axis=0),
                               np.concatenate(self.validation_step_outputs["action_class_pred"], axis=0),
                               ActionNameList, self.global_step)
            if not self.action_gt_is_logged:
                all_action = np.concatenate(self.validation_step_outputs["action_gt"], axis=0)
                log_action_distribution(writer, all_action, self.global_step)
                self.action_gt_is_logged = True

        for key in self.validation_step_outputs.keys():
            self.validation_step_outputs[key].clear()

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        sch = self.lr_schedulers()
        sch.step() # update lr per batch


def main():
    parser = argparse.ArgumentParser()
    # hardware configuration
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpuid", "-g", type=int, nargs="*", default=-1)
    parser.add_argument("--precision", "-p", type=str,
                        default="high", choices=["highest", "high", "medium"])

    # directory configuration
    parser.add_argument("--data-path", "-d", nargs="*", type=str, default=["./rloutputs/data/numpy_data"],
                        help="specify where to find data, enter 1 or several paths")
    parser.add_argument("--output-dir", "-o", type=str, default="./rloutputs/")
    parser.add_argument("--exp-name", "-en", type=str, default="test")
    parser.add_argument("--checkpoint-path", "-ckpt", type=str, default=None)
    parser.add_argument("--model-yaml-path", "-y", type=str, default="./config/rl/bc_config.yaml")

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
    parser.add_argument("--eval-size", "-es", type=int, default=30000)
    parser.add_argument("--num-workers", "-n", type=int, default=2)
    parser.add_argument("--max-steps", "-s", type=int, default=60,
                        help="How many steps to train in this process. Does not include step_offset. Actually, last step is max_step + step_offset")
    parser.add_argument("--step-offset", "-so", type=int, default=0)

    # miscellaneous
    parser.add_argument("--seed", "-sd", type=int,
                        default=time.time_ns() % (2 ** 32))
    args = parser.parse_args()

    # init logger
    logger = pl_loggers.TensorBoardLogger(
        args.output_dir, name=args.exp_name, log_graph=True)
    os.makedirs(logger.log_dir)
    omegaconf.OmegaConf.save(
        omegaconf.DictConfig(
            {"command line": " ".join(sys.argv), "working dir": os.getcwd(), "args": vars(args)}),
        os.path.join(logger.log_dir, "command_line.yaml"))
    profiler_name2class = {"Advanced": AdvancedProfiler, "Simple": SimpleProfiler, "PyTorch": PyTorchProfiler}
    profiler = profiler_name2class[args.profiler](
        dirpath=logger.log_dir, filename="perf_logs")

    # init numpy and pytorch
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision(args.precision)

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
    train_data, eval_data = get_bc_dataset(
        args.data_path, model_kwargs["prev_action_n"], args.eval_size, seed=0) # fix seed
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=not args.disable_drop_last)
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=not args.disable_drop_last)
    
    # init model
    if args.checkpoint_path is not None:
        model = PointNetBC.load_from_checkpoint(
            args.checkpoint_path, **model_kwargs, **lralg_kwargs)
    else:
        model = PointNetBC(**model_kwargs, **lralg_kwargs)
    model.example_input_array = \
        torch.randn([args.batch_size] + list(train_data[0][0].shape)), \
        torch.randn([args.batch_size] + list(train_data[0][1].shape)), \
        torch.randn([args.batch_size] + list(train_data[0][2].shape))

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
