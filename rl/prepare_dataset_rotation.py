import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pickle
import functools
import argparse

import tqdm
import numpy as np
from omegaconf import OmegaConf

from src.paper_cutting_environment import PaperCuttingEnvironment
from rl.prepare_dataset import DataForest, DataNode, default_action_dict_to_4D, get_neighbor_state_path
from rl.reward import calculate_state_score

class RotationPrepareDataset:
    def __init__(self, 
                 data_forest: DataForest,
                 cfg_file="./config/paper_cutting_game_fast.yaml") -> None:
        self._data_forest = data_forest

        cfg = OmegaConf.load(cfg_file)
        cfg.robot.use_robot=False
        cfg.setup.arch="cpu"
        cfg.setup.pop("cuda")
        self._env = PaperCuttingEnvironment(cfg)
        self._action_embed = default_action_dict_to_4D

    def __len__(self):
        return len(self._data_forest)

    @staticmethod
    @functools.lru_cache
    def _calculate_score(state_path: str, goal_set_path:str, **calculate_score_kwargs):
        goal_set = np.array(OmegaConf.to_container(OmegaConf.load(goal_set_path))["goal_edge_set"])
        state = pickle.load(open(state_path, "rb"))
        return calculate_state_score(state, goal_set, **calculate_score_kwargs)[0]

    @staticmethod
    @functools.lru_cache
    def _calculate_total_length(goal_set_path: str):
        goal_set = np.array(OmegaConf.to_container(OmegaConf.load(goal_set_path))["goal_edge_set"]) # [?, 2, 3]
        return np.sum(np.linalg.norm(goal_set[:, 0, :] - goal_set[:, 1, :], axis=1))
    
    def _is_failed(self, index:int, node: DataNode, max_seq_len: int, ratio_threshold: float, 
                   print_score_info: bool, visualize_reward: bool) -> bool:
        txt_files = []
        if ".txt" in node.get_parent().info.keys():
            txt_files = [os.path.split(txt_path)[1] for txt_path in node.get_parent().info[".txt"]]
        if ("fail.txt" in txt_files) or ("detached.txt" in txt_files):
            return True
        
        seq_len = node.get_parent().get_size()
        if seq_len > max_seq_len:
            return True
        
        curr_path = node.get_path()
        head, tail = os.path.split(curr_path)
        assert tail[:5] == "state" and tail[-4:] == ".pkl", tail
        final_state_path = os.path.join(head, f"state{seq_len-1}.pkl")
        goal_set_path = os.path.join(head, f"{os.path.basename(head)}.yaml")
        score = self._calculate_score(final_state_path, goal_set_path, 
                                      reward_per_meter=1.0, penalty_per_meter=0.2, distance_tolerance=0.004,
                                      visualize=visualize_reward)
        total_length = self._calculate_total_length(goal_set_path)
        
        if print_score_info:
            if curr_path != final_state_path:
                return True
            print(f"index:{index} score:{score:.4f} total_length:{total_length:.4f} ratio:" +
                  f"{(score/total_length):.4f} visualize_reward:{visualize_reward} path:{final_state_path}")
        
        if score / total_length < ratio_threshold:
            return True
        
        return False
    
    def _get_next_edge(self, node: DataNode, completed: int):
        curr_path = node.get_path()
        head, tail = os.path.split(curr_path)
        assert tail[:5] == "state" and tail[-4:] == ".pkl", tail
        goal_set_path = os.path.join(head, f"{os.path.basename(head)}.yaml")
        goal_set = np.array(OmegaConf.to_container(OmegaConf.load(goal_set_path))["goal_edge_set"]) # [?, 2, 3]
        
        next_edge_rest = goal_set[completed + 1, :, :]
        next_edge_xyz = np.zeros_like(next_edge_rest)
        next_edge_xyz[0, :] = self._env.get_current_pos_given_rest(next_edge_rest[0, :])
        next_edge_xyz[1, :] = self._env.get_current_pos_given_rest(next_edge_rest[1, :])
        return next_edge_xyz

    def _is_small_rotation(self, prev_ac_type: int, ac_gt: np.ndarray) -> bool:
        old_rotation_id = 3
        old_open_id = 0
        return ac_gt[0] == old_rotation_id and prev_ac_type == old_open_id

    def get_item(self, index: int, keep_failure=False, max_seq_len=50, ratio_threshold=0.6,
                 print_score_info=False, visualize_reward=False) -> dict:
        ret_dict = {}
        node = self._data_forest[index]
        path = node.get_path()
        state: dict = pickle.load(open(path, "rb"))
        if state["action"]["Action"] == "Rotate":
            prev_state: dict = pickle.load(open(get_neighbor_state_path(path, -1), "rb"))
            if prev_state["action"]["Action"] == "Open":
                return None # only save large rotation
        else:
            return None # only save rotation
        
        head, tail = os.path.split(path)
        assert tail[:5] == "state" and tail[-4:] == ".pkl", tail
        state_idx = int(tail[5: -4])
        if not state_idx >= 5: # ignore precut
            return None
        
        is_failed = self._is_failed(index, node, max_seq_len, ratio_threshold,
                                    print_score_info, visualize_reward)
        
        if is_failed and not keep_failure:
            return None

        self._env.set_state(state)
        ret_dict["front_point"] = self._env.get_scissors_front_point("world")[0]
        ret_dict["cut_direction"] = self._env.get_scissors_cut_direction()[0][1]
        ret_dict["next_edge"] = self._get_next_edge(node, state["completed"])

        ret_ac = self._action_embed(state["action"]) # action label
        assert ret_ac.shape == (4, )
        ret_dict["action"] = ret_ac

        info = {
            "path": path,
            "fail": is_failed,
        } # info is always returned
        ret_dict["info"] = info

        return ret_dict
    
def main():
    """example usage:
    ```
    python rl/prepare_dataset.py -d /DATA/disk1/epic/lvjiangran/code/cutgym/outputs/demos1/ -o ./rloutputs/ -s 1000 -ds 1000 -g 0
    ```"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos-dirs", "-d", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)

    parser.add_argument("--start-idx", "-s", type=int, required=True)
    parser.add_argument("--data-step", "-ds", type=int, required=True)

    parser.add_argument("--gpuid", "-g", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--keep-failure", "-kf", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    data_forest = DataForest([args.demos_dirs], ".pkl", [".png", ".txt", ".yaml"])
    prepare_dataset = RotationPrepareDataset(data_forest)
    zfill_length = len(f"{len(data_forest)}")

    os.makedirs(args.output_dir, exist_ok=True)

    for data_idx in tqdm.tqdm(range(args.start_idx, len(prepare_dataset), args.data_step)):
        data = prepare_dataset.get_item(data_idx, keep_failure=args.keep_failure)
        if data is not None:
            pickle_path, pickle_name = os.path.split(os.path.abspath(data["info"]["path"]))
            root_path = str(os.path.abspath(args.demos_dirs))
            pickle_path = str(pickle_path)
            assert root_path in pickle_path, f"root:{root_path} pickle:{pickle_path}"
            relative_path = os.path.relpath(pickle_path, root_path)
            new_data_folder = os.path.join(args.output_dir, relative_path)
            assert "state" == os.path.splitext(pickle_name)[0][:5]
            new_data_prefix = str(os.path.splitext(pickle_name)[0][5:]).zfill(zfill_length)

            if os.path.exists(args.output_dir):
                os.makedirs(new_data_folder, exist_ok=True)
            np.save(os.path.join(new_data_folder, f"{new_data_prefix}_rotation.npy"), data)

        else:
            print(f"[WARN] get_item reture None: {data_forest[data_idx].get_path()}")

if __name__ == "__main__":
    main()