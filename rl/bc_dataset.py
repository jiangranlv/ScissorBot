import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from typing import Literal, Tuple, List
import numpy as np
from torch.utils.data.dataset import Dataset

from rl.pytorch_utils import *
from rl.prepare_dataset import DataForest

from trimesh.transformations import quaternion_matrix

# global constants
POINT_DIM = 4
PRE_CUT_LEN = 5
POINT_RES = 512

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
        pc_seq = np.zeros((self._seq_len, 4096, POINT_DIM), np.float32) # [T, POINT_RES, POINT_DIM]
        pose_seq = np.zeros((self._seq_len, 8), np.float32) # [T, 8] Rotate = quaternion
        hp_seq_gt = np.zeros((self._seq_len, 32, 3), np.float32) # [T, 32, 3]
        ac_gt: np.ndarray = np.load(self._ac[data_index].get_path()) # [4, ] Rotate = ANGLE THETA PHI
        data_idx = np.array([data_index], np.int64)
        fp_seq = np.zeros((self._seq_len, 3), np.float32) # [T, 3] front_point
        
        
        prev_ac_type = -1.0
        for prev_data_index in range(data_index, data_index - self._seq_len, -1):
            idx_in_seq = prev_data_index - (data_index - self._seq_len + 1)
            if 0 <= prev_data_index and prev_data_index < self.max_data_index and \
                self._is_same_trajectory(prev_data_index, data_index):
                pc_seq[idx_in_seq, ...] = np.load(self._pc[prev_data_index].get_path())
                pose_seq[idx_in_seq, ...] = np.load(self._pose[prev_data_index].get_path())
                hp_seq_gt[idx_in_seq, ...] = np.load(self._hp[prev_data_index].get_path())
                aux = np.load(self._aux[prev_data_index].get_path(), allow_pickle= True).tolist()
                fp_seq[idx_in_seq, ...] = aux["front_point"].astype(np.float32)
                if prev_data_index == data_idx - 1:
                    prev_ac_type = np.load(self._ac[prev_data_index].get_path())[0] # get action type
            elif idx_in_seq < self._seq_len - 1:
                pc_seq[idx_in_seq, ...] = pc_seq[idx_in_seq + 1, ...].copy()
                pose_seq[idx_in_seq, ...] = pose_seq[idx_in_seq + 1, ...].copy()
                hp_seq_gt[idx_in_seq, ...] = hp_seq_gt[idx_in_seq + 1, ...].copy()
                fp_seq[idx_in_seq, ...] = fp_seq[idx_in_seq + 1, ...].copy()
            else:
                raise RuntimeError(f"Unexpected idx_in_seq {idx_in_seq}")

        # IMPORTANT! modify ac_gt
        if self._action_type_num == 5 and self._is_small_rotation(prev_ac_type, ac_gt):
            ac_gt[0] = 4.0
        
        # compress and canonicalize point cloud
        compress_pc_seq = np.zeros((self._seq_len, POINT_RES, POINT_DIM), np.float32) # [T, POINT_RES, POINT_DIM]
        for i in range(self._seq_len):
            compress_pc_seq[i, ...] = compress_canonicalize_pc(pc_seq[i], fp_seq[i], pose_seq[i][4:8], 
                                        n_sample= POINT_RES)    
            
        # sanity check
        assert compress_pc_seq.shape == (self._seq_len, POINT_RES, POINT_DIM), f"{pc_seq.shape}"
        assert pose_seq.shape == (self._seq_len, 8), f"{pose_seq.shape}"
        assert hp_seq_gt.shape == (self._seq_len, 32, 3), f"{hp_seq_gt.shape}"
        assert ac_gt.shape == (4, ), f"{ac_gt.shape}"
        assert data_idx.shape == (1, ), f"{data_idx.shape}"

        assert False not in np.isfinite(compress_pc_seq)
        assert False not in np.isfinite(pose_seq)
        assert False not in np.isfinite(hp_seq_gt)
        assert False not in np.isfinite(ac_gt)
        assert False not in np.isfinite(data_idx)

        return compress_pc_seq, pose_seq, hp_seq_gt, ac_gt, data_idx
    

def make_single_dataset(data_path: List[str], mode: Literal["train", "eval"], seq_len: int, 
                        pre_cut_len: int, action_type_num: int, verbose=1):
    if verbose == 1:
        print(f"scan all {mode} data ...")
    
    pc_path = data_path
    pc_file_name = 'point_cloud' 
    f_pc = DataForest(pc_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name= pc_file_name)
    
    f_pose = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="scissor_pose")
    f_action = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="action")
    f_hp = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="helper_point")
    f_info = DataForest(data_path, target_file_suffix=[".yaml"], info_file_suffix=[], target_file_name="info")
    f_aux = DataForest(data_path, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="auxiliary")
    
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
        pre_cut_len: int, action_type_num: int):
        
    return make_single_dataset(train_data_path, "train", seq_len, pre_cut_len, action_type_num), \
        make_single_dataset(eval_data_path, "eval", seq_len, pre_cut_len, action_type_num)

def compress_canonicalize_pc(pc: np.ndarray, fp: np.ndarray, scissor_pose: np.ndarray, n_sample: int, rotate = False):
    assert pc.shape == (4096, 4) and fp.shape == (3, ) , f"{pc.shape} {fp.shape} "

    # select `n_sample` nearest points
    dist = np.linalg.norm(pc[:, :3] - fp[np.newaxis, :], axis=1)
    argsorted = np.argsort(dist)
    select_pc = pc[argsorted[:n_sample], :]

    # affine transformations such that 
    # `fp` is move to 0 
    select_pc[:, :3] -= fp
    
    # scissor is rotated to origin pose
    if rotate:
        assert scissor_pose.shape == (4, ), f"{scissor_pose.shape}"
        rot_mat = quaternion_matrix(scissor_pose)[:3, :3]
        select_pc[:, :3] = select_pc[:, :3] @ rot_mat
    return select_pc