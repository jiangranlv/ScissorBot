import pickle
import sys
import os
import shutil
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import ast
import copy
import time
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch

from torch.utils.data.dataloader import DataLoader
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler, PyTorchProfiler
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor, Callback

from rl.pytorch_utils import to_numpy
from rl.bc_beta import PointNetTransformerBC, ModifyStepCallback
from rl.prepare_dataset import rotate2quat_numpy, default_action_dict_to_4D
from rl.so3 import compute_err_deg_from_quats

from policy.cutgym import gym
from policy.validate_policy import TestDataLoader, TestEnv, one_hot_encode, decode_action, proj_root, policy_init, action_order
from policy.generate_demos import rule_based_cutting, pre_cut, fix_square_pad, random_generate_goal

from dagger.data_utils import DAggerPreparer, make_single_dagger_dataset, filter_prepare_data


def compute_error(pred, gt):
    pred = copy.deepcopy(pred)
    gt = copy.deepcopy(gt)
    for key in pred.keys():
        if key != 'Action':
            pred[key]= np.array(pred[key])
            gt[key] = np.array(gt[key])
    if pred["Action"] == "Rotate":
        quat_pred = torch.from_numpy(rotate2quat_numpy(pred["Displacement"]))[None, :]
        quat_gt = torch.from_numpy(rotate2quat_numpy(gt["Displacement"]))[None,:]
        err = compute_err_deg_from_quats(quat_pred, quat_gt).item()
    elif pred["Action"] =="Translate":
        err = np.linalg.norm(pred["Displacement"] - gt["Displacement"])
    else:
        err = (pred["Angle"] - gt["Angle"]) /3.14 * 180
    print(pred["Action"],'err:',"{:.2f}".format(err))
    
class DAggerEnv():
    def __init__(self, simulation_cfg: DictConfig, gym_cfg, weight, seq_len, render_gpu, policy_cfg, output_dir = None) -> None:
        super().__init__()
        self.dataloader = TestDataLoader(seq_len= seq_len)
        self.gym = gym(simulation_cfg, gym_cfg, output_mode= 'demos')
        self.tool = TestEnv(gpuid=render_gpu, log_dir= None, offline= True)
        self.completed = self.gym.completed
        self.bc_policy = policy_init(**policy_cfg)
        assert weight <= 1 and weight >= 0
        self.weight = weight
        self.MINIMUM_ANGLE, self.MAXIMUM_ANGLE, self.MAX_CUT_LENGTH = self.gym.MINIMUM_ANGLE, self.gym.MAXIMUM_ANGLE, self.gym.MAX_CUT_LENGTH

        self.output_dir = output_dir
        
    def pre_policy_step(self):
        env = self.gym
        env.reset(init= True)
        fix_square_pad(env)
        self.pre_cut_state_action_list = pre_cut(env, save_action= True, save_state= True)
        for i, (state,action) in enumerate(zip(*self.pre_cut_state_action_list)):
            pc, pose = self.tool.get_state_obs(state, compress = False)
            self.dataloader.add_data(pc, pose)
            self._save_simulation_result(pc, pose, action, output_dir= self.output_dir, output_zfill=5, data_idx= i)
        return env
        
    def net_predict(self, pc_seq, pose_seq, idx):
        with torch.no_grad():
            action = to_numpy(self.bc_policy.predict(pc_seq, pose_seq)[0])
        action_type = one_hot_encode(action_order[idx%5], num_classes= 5)
        action[:5] = action_type
        action = decode_action(action)   
        # print(action)
        return action
    
    def step_mixture(self, action, delta_action):
        if action['Action'] != delta_action["Action"]:
            os.mknod('fail.txt') if not os.path.exists("fail.txt") else None
            return None
        action_real = copy.deepcopy(action)
        # compute_error(action, delta_action)
        if action_real['Action'] == 'Open' or action_real['Action'] == 'Close':
            action_real["Angle"] = (1 - self.weight) * action_real["Angle"] + self.weight * delta_action['Angle']
        elif action_real["Action"] == 'Rotate' or action_real["Action"] == 'Translate':
            action_real["Displacement"] = \
            [(1 - self.weight) * x + self.weight * y for x, y in zip(action["Displacement"], delta_action["Displacement"])]
        return self.gym.step(action_real)  
         
    def step(self, action):
        if os.path.exists("fail.txt") or os.path.exists('detached.txt'):
            return None
        
        state = self.gym.get_state()
        pc, pose = self.tool.get_state_obs(state, compress = False)
        self.dataloader.add_data(pc, pose)
        self._save_simulation_result(pc, pose, action, output_dir= self.output_dir, output_zfill=5)
        
        idx = self.dataloader.buffer_len - 1
        pc_seq, pose_seq = self.dataloader.__getitem__(idx)
        pc_seq = torch.from_numpy(pc_seq[None, :]).cuda()
        pose_seq = torch.from_numpy(pose_seq[None, :]).cuda()
        delta_action= self.net_predict(pc_seq, pose_seq, idx= idx)
        
        states = self.step_mixture(action, delta_action)
        
        #fix action
        saved_state = pickle.load(open(f"state{self.gym.state_num -1}.pkl", "rb"))
        saved_state['action'] = action
        pickle.dump(saved_state, open(f"state{self.gym.state_num -1}.pkl", "bw"))
        
        return states
    
    def _np_save_wrapper(self, path: str, arr: np.ndarray, save_txt: bool, allow_pickle: bool):
        """path does not include file suffix like '.npy' or '.txt'! """
        if save_txt:
            np.savetxt(path + ".txt", arr)
        else:
            np.save(path + ".npy", arr, allow_pickle=allow_pickle)  

    def _save_simulation_result(self, pc, pose, action, output_dir, output_zfill, save_txt = False, data_idx =None):
        
        current_directory = os.getcwd()
        folder_name = os.path.basename(current_directory)
        output_traj_folder = os.path.join(output_dir, folder_name)
        os.makedirs(output_traj_folder, exist_ok=True)
        
        ret_dict = {}
        ret_dict["point_cloud"] = pc
        ret_dict["pose"] = pose

        ac = default_action_dict_to_4D(action)
        assert ac.shape == (4, )
        ret_dict["action"] = ac

        fp = self.gym.get_scissors_front_point()
        ret_dict["front_point"] = fp

        data_idx = self.gym.state_num if data_idx is None else data_idx
        new_data_prefix = str(data_idx).zfill(output_zfill) 
        self._np_save_wrapper(os.path.join(output_traj_folder, f"{new_data_prefix}_point_cloud"),
                                ret_dict["point_cloud"], save_txt, False)
        self._np_save_wrapper(os.path.join(output_traj_folder, f"{new_data_prefix}_scissor_pose"),
                                ret_dict["pose"], save_txt, False)
        self._np_save_wrapper(os.path.join(output_traj_folder, f"{new_data_prefix}_action"),
                                ret_dict["action"], save_txt, False)
        self._np_save_wrapper(os.path.join(output_traj_folder, f"{new_data_prefix}_auxiliary"),
                                {"front_point": ret_dict["front_point"]}, False, True)
        
    def reset(self, init=False):
        self.dataloader.reset()
        self.gym.reset(init)
    
    def update_model(self, ckpt_path):
        self.bc_policy = PointNetTransformerBC.load_from_checkpoint(ckpt_path)

    def get_scissors_pose(self):
        return self.gym.get_scissors_pose()
    
    def get_scissors_cut_direction(self):
        return self.gym.get_scissors_cut_direction()
    
    def get_current_pos_given_rest(self, uv):
        return self.gym.get_current_pos_given_rest(uv)
    
    def get_scissors_front_point(self):
        return self.gym.get_scissors_front_point()
    
    def compute_angle_from_current(self, delta_dist):
        return self.gym.compute_angle_from_current(delta_dist= delta_dist)
    
def env_init(args):
    gym_cfg = OmegaConf.load(proj_root+ '/config/generate_demos.yaml')
    simulation_cfg = OmegaConf.load(proj_root+ '/config/paper_cutting_game_fast.yaml')
    simulation_cfg = OmegaConf.merge(simulation_cfg, gym_cfg) 
    simulation_cfg.cloth.cloth_file = "./assets/vertical_a4_2_10mm.obj"
    simulation_cfg.env.cut_sim.position_bounds = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    simulation_cfg.setup.cuda = args.sim_gpu
    
    policy_cfg = dict(checkpoint_path= args.checkpoint_path, model_yaml_path= args.model_yaml_path, 
                                                log_dir= None, gpuid= args.policy_gpu)
    
    env = DAggerEnv(simulation_cfg, gym_cfg, weight=args.weight, seq_len= args.seq_len, render_gpu= args.render_gpu, policy_cfg= policy_cfg)        
    env.reset(init = True)
    return env

def collect_data(env:DAggerEnv, output_path, max_demos):
    
    os.chdir(output_path)
    for i in tqdm(range(max_demos)):
        env.gym.log.info(f"this is the {i} th trajectory")
        os.mkdir(str(i))
        os.chdir(str(i))
        goal_edge_set = random_generate_goal(i)
        # goal_edge_set = OmegaConf.load('/DATA/disk1/epic/lvjiangran/code/cutgym/rloutputs/exp/dagger/test3/collected_stage_0/4/4.yaml')
        # goal_edge_set = np.array(goal_edge_set['goal_edge_set'])
        env.gym.goal_edge_set = goal_edge_set
        env.tool.goal_file = os.path.join(output_path, str(i), f'{i}.yaml')
        env.tool.texture_file = os.path.join(output_path, str(i), f'{i}.png')
        # env.tool.goal_file = '/DATA/disk1/epic/lvjiangran/code/cutgym/rloutputs/exp/dagger/test3/collected_stage_0/4/4.yaml'
        # env.tool.texture_file = '/DATA/disk1/epic/lvjiangran/code/cutgym/rloutputs/exp/dagger/test3/collected_stage_0/4/4.png'
        env.pre_policy_step()
        rule_based_cutting(env, goal_edge_set)
        env.reset(init= False)
        os.chdir('..')
        

            
def training_model(dataset, model, checkpoint_path, args):
    
    logger = pl_loggers.TensorBoardLogger(args.output_dir, name=args.exp_name)
    os.makedirs(logger.log_dir)
    OmegaConf.save(DictConfig(
            {"command line": " ".join(sys.argv), "working dir": os.getcwd(), "args": vars(args)}),
        os.path.join(logger.log_dir, "command_line.yaml"))
    profiler_name2class = {"Advanced": AdvancedProfiler, "Simple": SimpleProfiler, "PyTorch": PyTorchProfiler}
    profiler = profiler_name2class[args.profiler](
        dirpath=logger.log_dir, filename="perf_logs")

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=not args.disable_drop_last)
    
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision(args.precision)
    
    model.example_input_array = \
        torch.randn([args.batch_size] + list(dataset[0][0].shape)), \
        torch.randn([args.batch_size] + list(dataset[0][1].shape))
    
    # init trainer
    trainer_kwargs = {
        "accelerator": "cuda" if args.cuda else "cpu",
        "devices": args.policy_gpu if args.cuda else "auto",
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
        trainer.fit(model=model, train_dataloaders=train_loader, ckpt_path = None)


def get_args():
    parser = argparse.ArgumentParser()
    # hardware configuration
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--precision", "-p", type=str,
                        default="high", choices=["highest", "high", "medium"])

    # directory configuration
    parser.add_argument("--common-data-path", "-dp", type=str, default="./rloutputs/data/numpy_data", 
                        help="common data path for training data and evaluation data")
    parser.add_argument("--train-data-path", "-tp", nargs="*", type=str, default=["train"],
                        help="specify where to find training data, enter 1 or several paths")
    parser.add_argument("--eval-data-path", "-ep", nargs="*", type=str, default=["eval"],
                        help="specify where to find evaluation data, enter 1 or several paths")
    parser.add_argument("--output-dir", "-o", type=str, default="./rloutputs/exp/dagger")
    parser.add_argument("--exp-name", "-en", type=str, default="test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-path", "-ckpt", type=str, default="./rloutputs/exp/bc_beta_compress/epoch=51-step=120000.ckpt")
    parser.add_argument("--model-yaml-path", "-y", type=str,
                        default="./rloutputs/exp/bc_beta_compress/model_config.yaml")

    # model configuration
    parser.add_argument("--model-param", "-m", type=ast.literal_eval,
                        help='overwrite params in yaml, example: -m "{\'pn_output_dim\':64, \'loss_cfg\':{\'probability_weight\':1e1}}"')

    # train or evaluation
    parser.add_argument("--eval", action="store_true", help="only evaluation")

    # debug configuration
    parser.add_argument("--profiler", "-pf", type=str,
                        default="Simple", choices=["Advanced", "Simple", "PyTorch"])
    parser.add_argument("--log-every-n-steps", "-ls", type=int, default=50)
    parser.add_argument("--ckpt-every-n-steps", "-cs", type=int, default=3000)
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
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    parser.add_argument("--learning-rate", "-l", type=float, default=1e-4,
                        help="learning rate. lr_schedule and learning_rate will restart every time.")
    parser.add_argument("--disable-drop-last", "-ddl", action="store_true")
    parser.add_argument("--num-workers", "-n", type=int, default=16)
    parser.add_argument("--max-steps", "-s", type=int, default=1000,
                        help="How many steps to train in this process. Does not include step_offset. Actually, last step is max_step + step_offset")
    parser.add_argument("--step-offset", "-so", type=int, default=0)

    # miscellaneous
    parser.add_argument("--seed", "-sd", type=int,
                        default=time.time_ns() % (2 ** 32))
    
    # modify global_variables
    parser.add_argument("--global-variable", "-gv", type=ast.literal_eval, default={})
    

    parser.add_argument("--weight", type=float, default=0.2)  # Example
    parser.add_argument("--seq_len", type=int, default=4)    # Example
    parser.add_argument("--sim_gpu", type=int, default=0)     # Example
    parser.add_argument("--render_gpu", type=int, default=2)     # Example
    parser.add_argument("--action_type_num", type=int, default=5)  # Example
    parser.add_argument("--num_stages", type=int, default=2)  # Example
    parser.add_argument("--max_demos", type=int, default=400)  # Example
    parser.add_argument("--policy_gpu", "-g", type=int, nargs="*", default=1)
    parser.add_argument("--thres", type=float, default= 0.9)
    args = parser.parse_args()
    return args
        
def main():
    
    args = get_args()
    env = env_init(args)
    if not os.path.exists(os.path.join(args.output_dir, args.exp_name)):
        os.mkdir(os.path.join(args.output_dir, args.exp_name))
    elif args.exp_name == 'debug':
        shutil.rmtree(os.path.join(args.output_dir, 'debug'))
        os.mkdir(os.path.join(args.output_dir, 'debug'))
        
    for i in range(args.num_stages):
        print(f'--------The {i} Stage begins--------')
        collected_data_dir = os.path.join(proj_root, args.output_dir, f'{args.exp_name}/collected_stage_{i}')
        prepared_data_dir = os.path.join(proj_root, args.output_dir, f'{args.exp_name}/prepared_stage_{i}')
        if not os.path.exists(collected_data_dir):
            os.mkdir(collected_data_dir)
            os.mkdir(prepared_data_dir)
        env.output_dir = prepared_data_dir
        collect_data(env, collected_data_dir, args.max_demos)
        # dirs = ['091801', '091802', '091803', '091804']
        # prepared_data_dir = [f'/DATA/disk1/epic/lvjiangran/code/cutgym/rloutputs/exp/dagger/{dir}/prepared_stage_0' for dir in dirs]
        dataset = filter_prepare_data(tool= env.tool, gym = env.gym, data_dir= collected_data_dir, \
                                      output_dir= prepared_data_dir, thres = args.thres)
        # checkpoint_path = os.path.join(args.output_dir, f'{args.exp_name}/checkpoints/')
        checkpoint_path = args.checkpoint_path
        
        training_model(dataset, env.bc_policy, checkpoint_path, args)
        dataset.score_thres += 0.02
if __name__ == '__main__':
    main()