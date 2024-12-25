import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path = list(set(sys.path)) # remove repeat path

import pickle
import functools
from typing import List, Callable, Literal, Union
import argparse
from tqdm import tqdm

import numpy as np
import omegaconf
from omegaconf import OmegaConf

from torch.utils.data.dataset import Dataset

from policy.validate_policy import GymStateTool

from rl.prepare_dataset import DataForest, DataNode, judge_internal_cut, default_action_dict_to_4D
from rl.reward import cal_state_diff_chamfer, calculate_completeness_chamfer
from rl.bc_dataset import POINT_DIM, POINT_RES, compress_canonicalize_pc

from scipy.spatial.transform import Rotation

    
def idx_of_path(path: str) -> int:
    """
    >>> path = '/foo/bar/state23.pkl'
    >>> print(idx_of_path(path)) # 23
    """
    assert isinstance(path, str)
    basename = os.path.basename(path)
    assert basename[:5] == "state" and basename[-4:] == ".pkl"
    return int(basename[5: -4])


class ComboActionInfo:
    def __init__(self, path_list: List[str], score: float, completed: int, score_multi_level: List[float], chamfer : float) -> None:
        self.path_list = path_list[:]
        self.score = score
        self.completed = completed
        self.score_multi_level = score_multi_level
        self.chamfer = chamfer

    def __repr__(self) -> str:
        return f"<combo> score:{self.score} completed:{self.completed} " + \
            f"path_list:{os.path.dirname(self.path_list[0])} {[idx_of_path(p) for p in self.path_list]}"


class DAggerPreparer:
    def __init__(self, tool:GymStateTool, pre_cut_len) -> None:
        self.tool = tool
        self.pre_cut_len = pre_cut_len

    def _analyze_single_traj(self, traj_folder_node: DataNode, device: str, visualize=False) -> List[ComboActionInfo]:
        """
        Return score of every cut in this trajectory. Length is about 6 or 7, depend on the goal.
        """
        goal_set_path = traj_folder_node.info[".yaml"][0]
        goal_set = np.array(OmegaConf.to_container(OmegaConf.load(goal_set_path))["goal_edge_set"])

        sorted_path_list = []
        for i, c in enumerate(traj_folder_node.get_child()):
            sorted_path_list.append(c.get_path())
        sorted_path_list = sorted(sorted_path_list, key=idx_of_path)

        curr_completed = -1
        combo_action_info_list = []

        path_list = []
        prev_state_path = None
        prev_state_dict = None

        for state_path_idx, state_path in enumerate(sorted_path_list):
            state_dict: dict = pickle.load(open(state_path, "rb"))
            completed = state_dict["completed"]

            if completed == curr_completed + 1:
                curr_completed += 1
                if prev_state_path is not None and prev_state_dict is not None:
                    cd, gd, ed, cp, wp, cdp = cal_state_diff_chamfer(
                        prev_state_dict, state_dict, goal_set,
                        torch_device=device, visualize=visualize,
                        correct_func=lambda x: np.clip((0.003 - x) / 0.002, 0.0, 1.0),
                        wrong_func=lambda x: np.clip((x - 0.001) / 0.002, 0.0, 1.0),
                        pass_debug_info={"prev_state_path":prev_state_path,
                                         "state_path":state_path})
                    score = calculate_completeness_chamfer(cp, wp, 0.2)
                    score_multi_level = []
                    for level in [0, 0.2, 0.5, 0.8, 1]:
                        score_multi_level.append(calculate_completeness_chamfer(cp, wp, level))

                    chamfer = cd

                else:
                    raise RuntimeError(traj_folder_node.get_path())

                combo_action_info_list.append(ComboActionInfo(
                    path_list, score, completed, score_multi_level, chamfer))
                path_list = []

            elif completed == curr_completed:
                pass
            else:
                raise RuntimeError(traj_folder_node.get_path())
            
            path_list.append(state_path)
            prev_state_path = state_path
            prev_state_dict = state_dict

        if path_list != []:
            combo_action_info_list.append(ComboActionInfo(path_list, -1.0, curr_completed, score_multi_level=[-1.0] * 5, chamfer= -1)) # last cut

        return combo_action_info_list

    def _is_simulator_error(self, node: DataNode, verbose=0) -> bool:
        # if ".txt" in node.get_parent().info.keys():
        #     txt_files = [os.path.basename(txt_path) for txt_path in node.get_parent().info[".txt"]]
        #     if "fail.txt" in txt_files:
        #         if verbose != 0:
        #             print("[WARN] fail", node.get_parent().get_path())
        #         return True
        #     if "detached.txt" in txt_files:
        #         if verbose != 0:
        #             print("[WARN] detached", node.get_parent().get_path())
        #         return True
        id = '{:06d}'.format(node.get_parent().get_size() - 1)
        path = os.path.join(node.get_parent().get_path(), f"state{id}.pkl")
        if judge_internal_cut(pickle.load(open(path, "rb"))):
            if verbose != 0:
                print("[WARN] break inside", node.get_parent().get_path())
            return True
        
        return False
    
    def _render_and_extract_simulator_info(self, state_path: str) -> dict:
        # only return point_cloud, pose, action
        ret_dict = {}
        state: dict = pickle.load(open(state_path, "rb"))
        if state['action']['Action'] == 'None':
            return None

        if self.tool is not None:
            idx_state = idx_of_path(state_path)
            state_name = os.path.basename(state_path)
            if state['action']['Action'].lower() =='open' and idx_state > 0:   
                prev_state_name = state_name.replace(str(idx_state), (str(idx_state - 1)).zfill(len(str(idx_state))))
                prev_state_path = state_path.replace(state_name, prev_state_name)
                prev_state_front_point = pickle.load(open(prev_state_path, "rb"))['front_point']
                crop_point = prev_state_front_point
            else:
                crop_point = state['front_point']
            pc, pose = self.tool.get_state_obs(state, crop_point= crop_point) # TODO: check this
        else:
            pc, pose = np.random.randn(4096, POINT_DIM), np.random.randn(8) # for debug
        ret_dict["point_cloud"] = pc
        ret_dict["pose"] = pose

        def new_action_dict_to_4D(a: dict) -> np.ndarray:
            try:
                if a["Action"] == "Open":
                    return np.array([-1., 0., 0., 0.], dtype=np.float32)
                elif a["Action"] == "Close":
                    return np.array([0., a["distance"], 0., 0.], dtype=np.float32)
                elif a["Action"] == "Push":
                    return np.array([1., a["distance"], 0., 0.], dtype=np.float32)
                elif a["Action"] == "Rotate":
                    return np.array([2., *a["direction"]], dtype=np.float32)
                elif a["Action"] == "Tune":
                    return np.array([3., *a["direction"]], dtype=np.float32)
                else:
                    raise ValueError
            except Exception as e:
                print(f"Cannot parse action: {a}")
                raise e

        if idx_of_path(state_path)>= self.pre_cut_len:
            ac = new_action_dict_to_4D(state["action"])
            assert ac.shape == (4, )
        else:
            ac = str(state['action'])
        ret_dict["action"] = ac

        ret_dict["front_point"] = state['front_point']

        ret_dict["state"] = state

        # save next_pose info 7 DoF Pose [angle, fp, cd]
        state_name = os.path.basename(state_path)
        idx_state = idx_of_path(state_path)
        next_state_name = state_name.replace(str(idx_state).zfill(len(str(idx_state + 1))), (str(idx_state + 1)))
        next_state_path = state_path.replace(state_name, next_state_name)
        if os.path.exists(next_state_path):
            next_state = pickle.load(open(next_state_path, mode = "rb"))
            next_angle = next_state['pose'][0]
            next_front_point = state['front_point'] if state['action']['Action'].lower() =='close' else next_state['front_point']
            next_cut_direction = next_state['cut_direction']
            ret_dict['next_angle_pose'] = np.concatenate([[next_angle], next_front_point, next_cut_direction])
        else:
            ret_dict['next_angle_pose'] = np.zeros(7, np.float32)

        return ret_dict

    def _np_save_wrapper(self, path: str, arr: np.ndarray, save_txt: bool, allow_pickle: bool):
        """path does not include file suffix like '.npy' or '.txt'! """
        if save_txt:
            np.savetxt(path + ".txt", arr)
        else:
            np.save(path + ".npy", arr, allow_pickle=allow_pickle)  

    def filter_prepare_data(self, data_dir: Union[str, List[str]], input_base_dir: str, output_dir: str, device: str, 
                            output_zfill=5, render_result=False, save_txt=False, verbose=1, pre_cut_len = 5) -> dict:
        """
        Args:
            base_dir: 
                if `None`:
                    if `data_dir` is `str`: base_dir = dat
                    elif `data_dir` is `List[str]`: raise error
        Return:
            debug info
        """
        assert isinstance(data_dir, (str, list)) and isinstance(output_dir, str)
        if isinstance(data_dir, list):
            for dd in data_dir:
                assert isinstance(dd, str)
            data_dir_list = data_dir
        else:
            data_dir_list = [data_dir]
        assert len(data_dir_list) > 0

        debug_info = {"all_combo": []}

        os.makedirs(output_dir, exist_ok=True)
        data_forest = DataForest(data_dir_list, ".pkl", [".png", ".txt", ".yaml"], "state")

        analyzed_traj_set = set()
        for i in range(len(data_forest)):
            parent_node = data_forest[i].get_parent()

            # loop over all trajectories
            if parent_node not in analyzed_traj_set:
                analyzed_traj_set.add(parent_node)
                
                if self.tool is not None:
                    traj_id = parent_node.get_path().split('/')[-2]
                    self.tool.texture_file = os.path.abspath(os.path.join(parent_node.get_path(), f'{traj_id}.png'))

                # judge simulator error
                id = '{:06d}'.format(parent_node.get_size() - 1)
                final_state_path = os.path.abspath(os.path.join(
                    parent_node.get_path(),
                    f"state{id}.pkl"
                ))
                final_state_node = data_forest._path2node[final_state_path]
                
                # sim_fail = True if self._is_simulator_error(final_state_node, verbose) else False
                sim_fail = False

                # create folder
                relpath = os.path.relpath(parent_node.get_path(), input_base_dir)
                output_traj_folder = os.path.join(output_dir, relpath)
                os.makedirs(output_traj_folder, exist_ok=True)

                # render all
                fail_render_idx = np.inf
                if render_result:
                    # all_ret_dict = {}
                    for brother_node in tqdm(parent_node.get_child()):
                        brother_path = brother_node.get_path()
                        if idx_of_path(brother_path) < pre_cut_len:
                            continue
                        try:
                            ret_dict = self._render_and_extract_simulator_info(brother_path) # can be optimized
                        except Exception as e:
                            print(e)
                            raise ValueError(brother_path)
                        if ret_dict is not None and ret_dict['point_cloud'] is not None:
                            new_data_prefix = str(idx_of_path(brother_path)).zfill(output_zfill)
                            self._np_save_wrapper(os.path.join(output_traj_folder, f"{new_data_prefix}_point_cloud"),
                                                ret_dict["point_cloud"], save_txt, False)
                            self._np_save_wrapper(os.path.join(output_traj_folder, f"{new_data_prefix}_scissor_pose"),
                                                ret_dict["pose"], save_txt, False)
                            self._np_save_wrapper(os.path.join(output_traj_folder, f"{new_data_prefix}_action"),
                                                ret_dict["action"], save_txt, False)
                            self._np_save_wrapper(os.path.join(output_traj_folder, f"{new_data_prefix}_auxiliary"),
                                                {"front_point": ret_dict["front_point"]}, False, True)
                            self._np_save_wrapper(os.path.join(output_traj_folder, f"{new_data_prefix}_next_angle_pose"),
                                            ret_dict["next_angle_pose"], save_txt, False)
                        # all_ret_dict[brother_path] = ret_dict
                        else:
                            fail_render_idx = idx_of_path(brother_path)
                            break

                # save meta data
                traj_combo_list = self._analyze_single_traj(parent_node, device)
                for combo in traj_combo_list:
                    for tmp_path in combo.path_list:
                        if idx_of_path(tmp_path) < fail_render_idx and idx_of_path(tmp_path) >= pre_cut_len:
                            state_dict = pickle.load(open(tmp_path, 'rb'))
                            info_data = {
                                "path": str(tmp_path),
                                "sim_fail": sim_fail,
                                "score": float(combo.score),
                                "state_action": str(state_dict["action"]["Action"]),
                                "state_completed": int(state_dict["completed"]),
                                "score_multi_level": [float(x) for x in combo.score_multi_level],
                                "chamfer": float(combo.chamfer)
                            }
                            new_data_prefix = str(idx_of_path(tmp_path)).zfill(output_zfill)
                            if state_dict["action"]["Action"] !='None':
                                OmegaConf.save(omegaconf.DictConfig(info_data), os.path.join(output_traj_folder, f"{new_data_prefix}_info.yaml"))
                    debug_info["all_combo"].append(combo)

        return debug_info
    

class DAggerDataset(Dataset):
    def __init__(self, data_index_table: np.ndarray, seq_len: int, action_type_num: int, score_thres: float, mode: Literal["train", "eval"],
                 point_cloud: DataForest, pose: DataForest, action: DataForest, info: DataForest, aux: DataForest, 
                 point_res_store = 4096, point_res_use = 512, pose_dim = 6, action_dim = 4, rot_jitter = 0, chunking_size = 1) -> None:
        super().__init__()

        self._data_index_table = data_index_table.copy()
        self._size = len(self._data_index_table)
        self._seq_len = seq_len
        # self._action_type_num = action_type_num

        self._score_thres = score_thres
        self._getitem_debug_info = {"success":0, "fail":0}

        assert mode in ["train", "eval"]
        self._mode = mode
        self._pc = point_cloud
        self._pose = pose
        self._ac = action
        self._info = info
        self._aux = aux

        print(f"dataset {mode}: len {self._size}")

        self.point_res_store = point_res_store
        self.point_res_use = point_res_use
        self.pose_dim = pose_dim
        self.action_dim = action_dim

        self.rot_jitter = rot_jitter
        self.chunking_size = chunking_size

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
    
    def _clear_debug_info(self):
        self._getitem_debug_info["success"] = 0
        self._getitem_debug_info["fail"] = 0

    @property
    def score_thres(self) -> float:
        return self._score_thres

    @score_thres.setter
    def score_thres(self, new_score_thres: float):
        self._clear_debug_info()
        self._score_thres = new_score_thres
    
    def _get_validated_score(self, index: int):
        data_index = int(self._data_index_table[index])
        info = OmegaConf.load(self._info[data_index].get_path())
        if info['sim_fail'] is True:
            return None
        
        score = float(info["score"]) 
        return score
    
    def _getitem_impl(self, index: int) -> Union[None, tuple]:
        data_index = int(self._data_index_table[index])
        
        score = self._get_validated_score(index)
        if score is None or score < self.score_thres:
            return None

        pc_seq = np.zeros((self._seq_len, self.point_res_store, POINT_DIM), np.float32) # [T, POINT_RES, POINT_DIM]
        pose_seq = np.zeros((self._seq_len, self.pose_dim), np.float32) # [T, 8] Rotate = quaternion
        if self.chunking_size > 1:
            if self.action_dim == 4:
                ac_gt, future_pose_seq = self.load_action_chunking_seq(data_index, self.chunking_size)
            elif self.action_dim == 7:
                ac_gt = self.load_next_pose_chunking_seq(data_index, self.chunking_size)
        else:
            ac_gt: np.ndarray = np.load(self._ac[data_index].get_path()).astype(np.float32) # [4, ] Rotate = ANGLE THETA PHI
        data_idx = np.array([data_index], np.int64)
        fp_seq = np.zeros((self._seq_len, 3), np.float32) # [T, 3] front_point
        
        # prev_ac_type = -1.0 
        for prev_data_index in range(data_index, data_index - self._seq_len, -1):
            idx_in_seq = prev_data_index - (data_index - self._seq_len + 1)
            if 0 <= prev_data_index and prev_data_index < self.max_data_index and \
                self._is_same_trajectory(prev_data_index, data_index):
                pc_seq[idx_in_seq, ...] = np.load(self._pc[prev_data_index].get_path())
                pose_seq[idx_in_seq, ...] = np.load(self._pose[prev_data_index].get_path())
                aux = np.load(self._aux[prev_data_index].get_path(), allow_pickle=True).tolist()
                fp_seq[idx_in_seq, ...] = aux["front_point"].astype(np.float32)
                # if prev_data_index == data_idx - 1:
                #     prev_ac_type = np.load(self._ac[prev_data_index].get_path())[0] # get action type
            elif idx_in_seq < self._seq_len - 1:
                pc_seq[idx_in_seq, ...] = pc_seq[idx_in_seq + 1, ...].copy()
                pose_seq[idx_in_seq, ...] = pose_seq[idx_in_seq + 1, ...].copy()
                fp_seq[idx_in_seq, ...] = fp_seq[idx_in_seq + 1, ...].copy()
            else:
                raise RuntimeError(f"Unexpected idx_in_seq {idx_in_seq}, prev_data_index {prev_data_index}")

        # IMPORTANT! modify ac_gt
        # if self._action_type_num == 5 and self._is_small_rotation(prev_ac_type, ac_gt):
        #     ac_gt[0] = 4.0
        
        # compress and canonicalize point cloud
        if self.point_res_store != self.point_res_use:
            compress_pc_seq = np.zeros((self._seq_len, self.point_res_use, POINT_DIM), np.float32) # [T, POINT_RES, POINT_DIM]
            for i in range(self._seq_len):
                compress_pc_seq[i, ...] = compress_canonicalize_pc(pc_seq[i], fp_seq[i], pose_seq[i][4:8], 
                                            n_sample= self.point_res_use)    
        else:
            compress_pc_seq = pc_seq

        if self.rot_jitter > 0:
            compress_pc_seq, pose_seq, ac_gt = add_random_rotation(compress_pc_seq, pose_seq = pose_seq, gt= ac_gt, min_angle= 0, max_angle= self.rot_jitter)
                
        # sanity check
        assert compress_pc_seq.shape == (self._seq_len, self.point_res_use, POINT_DIM), f"{compress_pc_seq.shape}"
        assert pose_seq.shape == (self._seq_len, self.pose_dim), f"{pose_seq.shape}"
        if self.chunking_size > 1:
            assert ac_gt.shape == (self.chunking_size, self.action_dim), f"{ac_gt.shape}"
        else:
            assert ac_gt.shape == (self.action_dim, ), f"{ac_gt.shape}"
        assert data_idx.shape == (1, ), f"{data_idx.shape}"

        assert False not in np.isfinite(compress_pc_seq)
        assert False not in np.isfinite(pose_seq)
        assert False not in np.isfinite(ac_gt)
        assert False not in np.isfinite(data_idx)

        if self.chunking_size > 1:
            assert pose_seq.shape == (self.chunking_size, self.pose_dim), f"{pose_seq.shape}"
            if self.action_dim == 4:
                return compress_pc_seq, pose_seq, 0, ac_gt, data_idx, future_pose_seq

        return compress_pc_seq, pose_seq, 0, ac_gt, data_idx
    
    def load_action_chunking_seq(self, data_index: int, chunking_size: int) -> np.ndarray:
        action_seq = np.zeros((chunking_size + 1, self.action_dim), np.float32)
        pose_seq = np.zeros((chunking_size + 1, self.pose_dim), np.float32)
        # deal with special case for "open" action
        # -2 for special token which has no loss
        for next_data_index in range(data_index, data_index + chunking_size + 1):
            idx_in_seq = next_data_index - data_index 
            if next_data_index < self.max_data_index and \
                self._is_same_trajectory(next_data_index, data_index):
                action_seq[idx_in_seq, ...] = np.load(self._ac[next_data_index].get_path()).astype(np.float32)
                pose_seq[idx_in_seq, ...] = np.load(self._pose[next_data_index].get_path()).astype(np.float32)

            elif idx_in_seq > 0:
                action_seq[idx_in_seq, ...] = np.array([-2, 0, 0, 0])
                pose_seq[idx_in_seq, ...] = np.zeros_like(pose_seq[idx_in_seq - 1, ...])
            else:
                raise RuntimeError(f"Unexpected idx_in_seq {idx_in_seq}, next_data_index {next_data_index}")
        
        ret_action_seq = action_seq[action_seq[:, 0] != -1, :]
        ret_pose_seq = pose_seq[action_seq[:, 0] != -1, :]

        assert ret_action_seq.shape[0] > 0
        
        return ret_action_seq, ret_pose_seq
    
    def load_next_pose_chunking_seq(self, data_index: int, chunking_size: int) -> np.ndarray:
        action_seq = np.zeros((chunking_size, self.action_dim), np.float32)

        for next_data_index in range(data_index, data_index + chunking_size):
            idx_in_seq = next_data_index - data_index 
            if next_data_index < self.max_data_index and \
                self._is_same_trajectory(next_data_index, data_index):
                loaded_next_pose = np.load(self._ac[next_data_index].get_path()).astype(np.float32)
                if loaded_next_pose.shape == (7, ):
                    action_seq[idx_in_seq, ...]  = loaded_next_pose
                else:
                    action_seq[idx_in_seq, ...] = np.zeros(7, np.float32)
            elif idx_in_seq > 0:
                action_seq[idx_in_seq, ...] = np.zeros(7, np.float32)
            else:
                raise RuntimeError(f"Unexpected idx_in_seq {idx_in_seq}, next_data_index {next_data_index}")
    
        assert action_seq.shape[0] > 0
        return action_seq
            
    def _check_threshold(self):
        if self._getitem_debug_info["fail"] >= 1e2 and \
            self._getitem_debug_info["fail"] >= self._getitem_debug_info["success"]:
            print(f"[WARN] below curr score_thres:{self.score_thres},",
                  "fail:", self._getitem_debug_info["fail"],
                  "success:", self._getitem_debug_info["success"],
                  "Please consider decrease score_thres")
            self._clear_debug_info()
    
    def __getitem__(self, index):
        while True:
            self._check_threshold()
            ret = self._getitem_impl(int(index))
            if ret is not None:
                self._getitem_debug_info["success"] += 1
                return ret
            else:
                self._getitem_debug_info["fail"] += 1
                index = np.random.randint(0, self._size)

def add_random_rotation(pc_seq, pose_seq, gt, min_angle=0, max_angle=10):
    assert pose_seq.shape[1] == 6
    # Generate a random rotation axis (unit vector)
    random_axis = np.random.rand(3)
    random_axis /= np.linalg.norm(random_axis)

    # Generate a random rotation angle within the specified range
    random_angle = np.random.uniform(min_angle, max_angle)

    # Create a rotation matrix based on the random axis and angle
    rotation_matrix = Rotation.from_rotvec(random_angle * random_axis).as_matrix()

    # Apply the rotation to the entire point cloud]
    rotated_pc_seq = np.copy(pc_seq)
    rotated_pose_seq = np.copy(pose_seq)
    rotated_gt = np.copy(gt)
    for i in range(pc_seq.shape[0]):
        rotated_pc_seq[i, :, :3] = np.dot(rotation_matrix, pc_seq[i, :, :3].T).T
        rotated_pose_seq[i] = np.concatenate([np.dot(rotation_matrix, pose_seq[i, :3].T).T, np.dot(rotation_matrix, pose_seq[i, 3:].T).T])
        rotated_gt[1:] = np.dot(rotation_matrix, gt[1:].T).T if (gt[0] == 2 or gt[0] == 3) else gt[1:]
    return rotated_pc_seq, rotated_pose_seq, rotated_gt

def make_single_dagger_dataset(data_path: List[str], mode: Literal["train", "eval"], seq_len: int, 
                        pre_cut_len: int, action_type_num: int, score_thres: float, rot_jitter = 0, 
                        use_angle_pose = False, chunking_size= 1 , verbose=1):
    if verbose == 1:
        print(f"scan all {mode} data ...")
    
    f_pc = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="point_cloud")
    f_pose = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="scissor_pose")
    if use_angle_pose:
        f_action = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="next_angle_pose")
    else:
        f_action = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="action")
    f_info = DataForest(data_path, target_file_suffix=[".yaml"], info_file_suffix=[], target_file_name="info")
    f_aux = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="auxiliary")

    all_data_size = len(f_info)
    assert len(f_pose) == all_data_size and len(f_action) == all_data_size and \
        len(f_info) == all_data_size and len(f_aux) == all_data_size, \
            f"{len(f_pc)} {len(f_pose)} {len(f_action)} {len(f_info)} {len(f_aux)}"

    # drop pre_cut, open
    all_data_idx = []
    for i in range(all_data_size):
        filename = os.path.split(f_info[i].get_path())[1]
        assert filename[-10:] == "_info.yaml"
        file_idx = int(filename[:-10])
        if file_idx >= pre_cut_len and (file_idx % 5 != 2 or use_angle_pose):
            all_data_idx.append(i)
    all_data_idx = np.array(all_data_idx)

    if verbose != 0:
        print(f"make dataset {mode} ...")

    return DAggerDataset(all_data_idx, seq_len, action_type_num, score_thres, mode, f_pc, f_pose, f_action, f_info, f_aux, 
            point_res_store = 512, point_res_use = 512, pose_dim = 6, action_dim= 7 if use_angle_pose else 4, 
            rot_jitter= rot_jitter, chunking_size= chunking_size)
    

def get_dagger_dataset(train_data_path: List[str], eval_data_path: List[str], seq_len: int, 
        pre_cut_len: int, action_type_num: int, score_thres: float, rot_jitter, use_angle_pose = False, chunking_size= 1) -> tuple:
        
    return make_single_dagger_dataset(train_data_path, "train", seq_len, pre_cut_len, action_type_num, score_thres, 
                                      rot_jitter = rot_jitter, use_angle_pose= use_angle_pose, chunking_size= chunking_size), \
        make_single_dagger_dataset(eval_data_path, "eval", seq_len, pre_cut_len, action_type_num, score_thres, 
                                   use_angle_pose= use_angle_pose, chunking_size= chunking_size)
        


def filter_prepare_data(tool, gym, data_dir, output_dir, ret_dataset = True,  thres = 0.98):

    data_preparer = DAggerPreparer(tool, gym)
    data_preparer.filter_prepare_data(data_dir, output_dir, device="cuda:0")
    output_dirs = [output_dir] if isinstance(output_dir, str) else output_dir

    if ret_dataset:
        dataset = make_single_dagger_dataset(output_dirs, "train", seq_len=4, pre_cut_len=5, action_type_num=5, score_thres=thres)
        return dataset 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", action="store_true")
    parser.add_argument("-ds", action="store_true")
    args = parser.parse_args()

    if args.dp:
        dp = DAggerPreparer(None, None)
        dp.filter_prepare_data("/DATA/disk1/epic/lvjiangran/code/cutgym/outputs/demos_0819/035201", 
                               "/home/chenyuxing/research/cutting_paper_fem/dagger/test/test",
                               device="cuda:3")
    if args.ds:
        ds = make_single_dagger_dataset(["/home/chenyuxing/research/cutting_paper_fem/dagger/test/test"], "train", 4, 5, 5, 0.0)
        ds.score_thres = 0.98
        for i in range(300):
            x = ds[np.random.randint(0, len(ds))]
        ds.score_thres = 0.8
        for i in range(300):
            x = ds[np.random.randint(0, len(ds))]