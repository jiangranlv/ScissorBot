import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from typing import List, Union, Callable, Tuple, Dict, Literal
import copy
import pickle
import functools

import numpy as np
from sklearn import linear_model

import argparse
import tqdm
import sapien.core as sapien
import omegaconf
from omegaconf import OmegaConf
import trimesh.transformations as tra

from src.sapien_renderer import SapienRender
from src.paper_cutting_environment import PaperCuttingEnvironment
from rl.reward import calculate_state_score


def rotate2xyz_numpy(rotate: np.ndarray):
    """
    Input:
        rotate: [..., 3]
    Output:
        axis-angle representation xyz: [..., 3]
    """
    angle = rotate[..., 0]
    theta = rotate[..., 1]
    phi = rotate[..., 2]
    return np.stack([angle * np.sin(theta) * np.cos(phi),
                     angle * np.sin(theta) * np.sin(phi),
                     angle * np.cos(theta)], axis=-1)

def rotate2quat_numpy(rotate: np.ndarray):
    """
    Input:
        rotate: [..., 3]
    Output:
        quaternion: [..., 4]
    """
    xyz = rotate2xyz_numpy(rotate)
    angles: np.ndarray = np.linalg.norm(xyz, axis=-1)
    angles = angles[..., np.newaxis]
    half_angles = angles * 0.5
    sin_half_angles_over_angles = np.zeros_like(angles)
    small_angles = np.abs(angles) < 1e-6
    sin_half_angles_over_angles[~small_angles] = np.sin(half_angles)[~small_angles] / angles[~small_angles]
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return np.concatenate([np.cos(half_angles), (xyz * sin_half_angles_over_angles)], axis=-1)


def get_neighbor_state_path(curr_state_path: str, delta_state:int):
    head, tail = os.path.split(curr_state_path)
    root, ext = os.path.splitext(tail)
    assert root[0:5] == "state"
    return os.path.join(head, f"state{int(root[5:])+delta_state}{ext}")


def default_action_dict_to_4D(a: Dict[str, Union[str, List[int]]]):
    if a["Action"] == "Open":
        return np.array([0, a["Angle"], 0, 0], dtype=np.float32)
    elif a["Action"] == "Close":
        return np.array([1, a["Angle"], 0, 0], dtype=np.float32)
    elif a["Action"] == "Translate":
        return np.array([2, *a["Displacement"]], dtype=np.float32)
    elif a["Action"] == "Rotate":
        return np.array([3, *a["Displacement"]], dtype=np.float32)
    else:
        raise ValueError(f"unknown action:{a['Action']}")


def judge_internal_cut(state: dict, start_eid = 1030) -> bool:
    """
    >>> cloth_data = state["_sim_env"]["ti_objects"][0]
    >>> print(cloth_data["vertices_rest_pos"][mesh_data["edges_vid_fid"][1030, 0:2]]) # [[0.1   0.    0.297], [0.11  0.    0.297]]

    if break inside, return true
    """
    if isinstance(state["_sim_env"]["ti_objects"], list):
        cloth_data = state["_sim_env"]["ti_objects"][0]
    if isinstance(state["_sim_env"]["ti_objects"], dict):
        cloth_data = state["_sim_env"]["ti_objects"]["Cloth"]
    mesh_data = cloth_data["mesh"]

    start_fid = mesh_data["edges_vid_fid"][start_eid, 2]
    assert mesh_data["edges_vid_fid"][start_eid, 3] == -1, f"start_eid:{start_eid} is not an outside edge"

    # find all connected faces (start from 'start_fid')
    def dfs_connected_faces(curr_fid: int):
        stack = [curr_fid]
        visited = set([curr_fid])
        while len(stack) > 0:
            curr_fid = stack.pop()
            for vid in mesh_data["faces_vid"][curr_fid]:
                for fid_idx in range(mesh_data["vertices_fid_cnt"][vid]):
                    next_fid = mesh_data["vertices_fid"][vid, fid_idx]
                    if next_fid not in visited:
                        visited.add(next_fid)
                        stack.append(next_fid)
        return visited
    
    faces_set = dfs_connected_faces(start_fid)

    # find all outside edges in 'faces_set'
    outside_eid_faces_set = set()
    for eid in range(mesh_data["n_edges"]):
        if (mesh_data["edges_vid_fid"][eid, 2] in faces_set) and (mesh_data["edges_vid_fid"][eid, 3] == -1):
            outside_eid_faces_set.add(eid)
    outside_eid_faces = np.array(sorted(list(outside_eid_faces_set)))

    # find all connected outside edges (start from 'start_eid')
    def dfs_outside_egde(curr_eid: int):
        stack = [curr_eid]
        outside_edge_set = set([curr_eid])
        while len(stack) > 0:
            curr_eid = stack.pop()
            for vid in mesh_data["edges_vid_fid"][curr_eid, 0: 2]:
                for eid_idx in range(mesh_data["vertices_eid_cnt"][vid]):
                    next_eid = mesh_data["vertices_eid"][vid, eid_idx]
                    if mesh_data["edges_vid_fid"][next_eid, 3] != -1: # not outside edge, skip
                        continue
                    if next_eid not in outside_edge_set:
                        outside_edge_set.add(next_eid)
                        stack.append(next_eid)
        return outside_edge_set
    
    outside_edge_set = dfs_outside_egde(start_eid)
    outside_eid_edges = np.array(sorted(list(outside_edge_set)))

    return (len(outside_eid_edges) != len(outside_eid_faces)) or (outside_eid_edges != outside_eid_faces).any()
    

class DataNode:
    def __init__(self, path: str, tree_root_path: str) -> None:
        self._parent: Union[None, DataNode] = None
        self._child: List[DataNode] = []
        self._path: str = path
        self._size: int = 0
        self.tree_root_path: str = tree_root_path
        self.info: dict = {}

    def __hash__(self) -> int:
        return hash(self._path)

    def __len__(self) -> int:
        return self._size

    def get_path(self) -> str:
        return self._path

    def set_size(self, size: int):
        self._size = size

    def get_size(self) -> int:
        return self._size

    def get_parent(self):
        return self._parent

    def get_child(self):
        return self._child

    def __repr__(self) -> str:
        return f"<Node> path:<{self._path}> size:{self._size} parent:<{self._parent._path if self._parent is not None else None}> info:<{self.info}>"

    def append_child(self, child):
        self._child.append(child)
        child._parent = self
        self._size += child._size


class DataTree:
    def __init__(self, root_path: str, target_file_suffix: Union[List[str], str], info_file_suffix: Union[List[str], str], target_file_name: str = "") -> None:
        if isinstance(target_file_suffix, str):
            target_file_suffix = [target_file_suffix]
        if isinstance(info_file_suffix, str):
            info_file_suffix = [info_file_suffix]
        assert isinstance(target_file_suffix, list) and isinstance(
            info_file_suffix, list)
        assert isinstance(target_file_name, str)

        self._folder_directly_have_target = set()

        self.tree_root_path = root_path
        self._tree, _ = self._build_tree(
            root_path, target_file_suffix, info_file_suffix, target_file_name, True)
        self._calculate_accumulated_sum(self._tree)

    def _build_tree(self,
                    root_path: str,
                    target_file_suffix: List[str],
                    info_file_suffix: List[str],
                    target_file_name: str,
                    is_root=False) -> Tuple[Union[DataNode, None], bool]:
        is_target_file = False
        root_path = os.path.abspath(root_path)
        root_node = DataNode(root_path, self.tree_root_path)
        if not os.path.isfile(root_path):
            for subpath in sorted(os.listdir(root_path)):
                subpath_suffix = os.path.splitext(subpath)[-1]
                child_path = os.path.join(root_path, subpath)
                child, child_is_target_file = self._build_tree(
                    child_path, target_file_suffix, info_file_suffix, target_file_name, False)

                if child is not None:
                    root_node.append_child(child)
                
                if child_is_target_file:
                    self._folder_directly_have_target.add(root_path)

                if subpath_suffix in info_file_suffix:
                    if subpath_suffix in root_node.info.keys():
                        root_node.info[subpath_suffix].append(child_path)
                    else:
                        root_node.info[subpath_suffix] = [child_path]

        elif os.path.splitext(root_path)[-1] in target_file_suffix and \
                target_file_name in os.path.split(root_path)[1]:
            root_node.set_size(1)
            is_target_file = True

        if root_node.get_size() > 0 or is_root:
            return root_node, is_target_file
        else:
            return None, is_target_file

    def _calculate_accumulated_sum(self, root: DataNode):
        accumulated_sum = [0]
        for child in root.get_child():
            self._calculate_accumulated_sum(child)
            accumulated_sum.append(accumulated_sum[-1] + child.get_size())
        root.info["acc_sum"] = np.array(accumulated_sum)

    def _get_data(self, root: DataNode, idx: int) -> DataNode:
        if len(root.get_child()) == 0:
            return root
        which_child = np.searchsorted(
            root.info["acc_sum"], v=idx, side="right") - 1
        return self._get_data(root.get_child()[which_child], idx - root.info["acc_sum"][which_child])

    def __len__(self):
        return self._tree.get_size()

    def __getitem__(self, idx: int) -> DataNode:
        assert isinstance(idx, int), f"type(idx)={type(idx)}, but int expected"
        assert idx >= 0 and idx < self.__len__(
        ), f"idx:{idx} out of range:{self.__len__()}"
        return self._get_data(self._tree, idx)
    
    def get_folder_directly_have_target_cnt(self):
        return len(self._folder_directly_have_target)


class DataForest:
    def __init__(self, root_paths: List[str], target_file_suffix: Union[List[str], str], info_file_suffix: Union[List[str], str], target_file_name: str = "") -> None:
        assert isinstance(root_paths, list) and \
            isinstance(target_file_suffix, (list, str)) and isinstance(info_file_suffix, (list, str)) and isinstance(target_file_name, str)
        self._data_forest: List[DataTree] = []
        self._accumulated_size: List[int] = [0]

        for root_path in root_paths:
            self._data_forest.append(
                DataTree(root_path, target_file_suffix, info_file_suffix, target_file_name))
            self._accumulated_size.append(
                self._accumulated_size[-1] + len(self._data_forest[-1]))
            
        self.folder_directly_have_target_cnt = sum([tree.get_folder_directly_have_target_cnt()
                                                    for tree in self._data_forest])
        self._build_path2node_idx()

    def _build_path2node_idx(self):
        self._path2node = {}
        self._path2idx = {}
        for index in range(self.__len__()):
            node = self.__getitem__(index)
            path = node.get_path()
            self._path2node[path] = node
            self._path2idx[path] = index

    def __len__(self):
        return self._accumulated_size[-1]

    def __getitem__(self, index: Union[int, str]) -> DataNode:
        if isinstance(index, int):
            assert index < self.__len__() and index >= 0, f"index {index} out of range [0, {self.__len__()})"
            which_tree = np.searchsorted(
                self._accumulated_size, v=index, side="right") - 1
            return self._data_forest[which_tree][index - self._accumulated_size[which_tree]]
        elif isinstance(index, str):
            assert index in self._path2node.keys(), f"idxstr {index} not found."
            return self._path2node[index]
        else:
            raise ValueError(f"Invalid index type{type(index)}")
        
    def __contains__(self, path: str) -> bool:
        if isinstance(path, str):
            return path in self._path2node
        else:
            raise ValueError(f"Invalid path type{type(path)}")


class PointCloudPrepareDataset:
    def __init__(self,
                 data_forest: DataForest,
                 texture_suffix=".png",
                 point_cloud_number=4096,
                 cut_line_max_number=128,
                 camera_pose=sapien.Pose(
                     [0.21/2, 0.9, 0.297/2], tra.quaternion_from_euler(*[0.0, 0.0, -np.pi * 0.5])),
                 where_cut_line=lambda x: x[..., 0] > (x[..., 1] + x[..., 2]) * 3,
                 prepare_which:List[Literal["point_cloud", "pose", "action", "helper_point", "auxiliary"]]=[],
                 cfg_file = "./config/paper_cutting_game_fast.yaml",
                 **render_args) -> None:
        
        self._data_forest = data_forest
        self._texture_suffix = texture_suffix
        self._prepare_which = copy.deepcopy(prepare_which)
        assert isinstance(prepare_which, list)
        print(f"preparing {prepare_which} ...")

        if "point_cloud" in self._prepare_which:
            self._render_args = copy.deepcopy(render_args)
            self._renderer = SapienRender(**render_args)
            self._renderer.build_scene(**render_args)

            self._camera_pose = sapien.Pose(camera_pose.p, camera_pose.q)
            self._where_cut_line = where_cut_line
            self._point_cloud_number = point_cloud_number
            self._cut_line_max_number = cut_line_max_number

        if ("helper_point" in self._prepare_which) or ("auxiliary" in self._prepare_which):
            cfg = OmegaConf.load(cfg_file)
            cfg.robot.use_robot=False
            cfg.setup.arch="cpu"
            cfg.setup.pop("cuda")
            self._env = PaperCuttingEnvironment(cfg)

        if "helper_point" in self._prepare_which:
            self._ransac = linear_model.RANSACRegressor()

        if "action" in self._prepare_which:
            self._action_embed = default_action_dict_to_4D

    def set_camera_pose(self, camera_pose: sapien.Pose):
        self._camera_pose = sapien.Pose(camera_pose.p, camera_pose.q)

    @property
    def camera_pose(self):
        return self._camera_pose

    def set_where_cut_line(self, where_cut_line: Callable):
        assert callable(where_cut_line)
        self._where_cut_line = where_cut_line

    @property
    def where_cut_line(self):
        return self._where_cut_line

    def set_point_cloud_number(self, point_cloud_number: int):
        assert isinstance(point_cloud_number, int)
        self._point_cloud_number = point_cloud_number

    @property
    def point_cloud_number(self):
        return self._point_cloud_number

    def set_cut_line_max_number(self, cut_line_max_number: int):
        assert isinstance(cut_line_max_number, int)
        self._cut_line_max_number = cut_line_max_number

    @property
    def cut_line_max_number(self):
        return self._cut_line_max_number

    def set_action_embed(self, action_embed: Callable):
        assert callable(action_embed)
        self._action_embed = action_embed

    @property
    def action_embed(self):
        return self._action_embed

    def _get_scissor_pose(self, state: dict, path: str):
        for ti_object in state["_sim_env"]["ti_objects"]:
            if ti_object["class"] == "Scissor":
                result = np.zeros((8, ), np.float32)
                result[0] = ti_object["direct_cfg"]["joint_0"]
                result[1:4] = ti_object["direct_cfg"]["joint_1"][0:3]
                result[4:8] = rotate2quat_numpy(
                    np.array(ti_object["direct_cfg"]["joint_1"][3:6]))
                return result
        raise ValueError(f"no scissor information in state. path:{path}")

    def _get_helper_point(self, state: dict, goal_edge_set:list, 
                          samples_per_meter=2e3, total_sample_n=32,
                          backward_tolerance=0e-2, forward_tolerance=3e-2,
                          outlier_tolerance=1e-2) -> np.ndarray:
        """
        gaurantee to have `sample_n` samples as long as:
            (length of cutting line) x (`samples_per_meter`) > sample_n
        """
        self._env.set_state(state)
        front_point = self._env.get_scissors_front_point("world")[0]
        cut_direc = self._env.get_scissors_cut_direction()[0][1]

        good_samples: List[np.ndarray] = []
        all_samples: List[np.ndarray] = []
        all_scores: List[float] = []
        
        for edge in goal_edge_set:
            edge = np.array(edge) # [2, 3]
            edge_rest_length = np.linalg.norm(edge[0, :] - edge[1, :])
            sample_n = int(edge_rest_length * samples_per_meter) + 1
            t = np.arange(sample_n) / sample_n
            rest_positions = (1 - t[:, np.newaxis]) * edge[[0], :] + t[:, np.newaxis] * edge[[1], :]

            curr_positions = []
            for rest_position in rest_positions:
                curr_positions.append(self._env.get_current_pos_given_rest(rest_position))
            curr_positions = np.array(curr_positions)

            # Robust regression
            t_reshape = t.reshape(-1, 1)
            self._ransac.fit(t_reshape, curr_positions) 
            
            for i in range(sample_n):
                backward = np.clip(np.dot(front_point - curr_positions[i], cut_direc), a_min=backward_tolerance, a_max=np.inf)
                forward = np.clip(np.dot(curr_positions[i] - front_point, cut_direc), a_min=forward_tolerance, a_max=np.inf)
                outlier = np.linalg.norm((self._ransac.predict(t_reshape[[i], :]) - curr_positions[[i], :])[0])
                score = backward + outlier + forward # smaller is better
                all_samples.append(curr_positions[i])
                all_scores.append(score)
                if outlier <= outlier_tolerance and backward <= backward_tolerance \
                    and forward <= forward_tolerance:
                    good_samples.append(curr_positions[i])
        
        if len(good_samples) > total_sample_n:
            import open3d as o3d
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(np.array(good_samples))
            return np.asarray(pc.farthest_point_down_sample(total_sample_n).points)
        
        else:
            selected = np.argsort(np.array(all_scores))[:total_sample_n]
            return np.array(all_samples)[selected, :]

    def __len__(self):
        return len(self._data_forest)

    def __getitem__(self, index):
        raise NotImplementedError("Please use get_item method instead.")
    
    @functools.lru_cache
    def _calculate_score(self, state_path: str, goal_set_path:str, **calculate_score_kwargs):
        goal_set = np.array(OmegaConf.to_container(OmegaConf.load(goal_set_path))["goal_edge_set"])
        state = pickle.load(open(state_path, "rb"))
        return calculate_state_score(state, goal_set, **calculate_score_kwargs)[0]

    @functools.lru_cache
    def _calculate_total_length(self, goal_set_path: str):
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
            if curr_path == final_state_path:
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
    
    def get_item(self, index: int, keep_failure=False, max_seq_len=50, ratio_threshold=0.6,
                 print_score_info=False, visualize_reward=False, use_rgb_cutline=True):
        ret_dict = {}
        node = self._data_forest[index]
        path = node.get_path()
        state = pickle.load(open(path, "rb"))

        is_failed = self._is_failed(index, node, max_seq_len, ratio_threshold,
                                    print_score_info, visualize_reward)
        
        if is_failed and not keep_failure:
            return
        
        if "point_cloud" in self._prepare_which:
            tmp_filename = path.replace("/", "_").replace(".", "_") + "_tmp.obj"
            texture_path = node.get_parent().info[self._texture_suffix][0]
            self._renderer.set_scene(state,
                                    camera_pose=self._camera_pose,
                                    tmp_filename=tmp_filename,
                                    texture_file=texture_path,
                                    **self._render_args)
            result = self._renderer.render(calculate_rgba=False)

            pc_cat = np.concatenate(
                [result["pc_position"], result["pc_color"]], axis=1)
            if pc_cat.shape[0] < self._point_cloud_number:
                print(
                    f"Not enough points in the view: {pc_cat.shape[0]} < {self._point_cloud_number}")
                return

            if use_rgb_cutline:
                cut_line = pc_cat[self._where_cut_line(result["pc_color"]), :]
            else:
                cut_line = pc_cat[self._where_cut_line(result["pc_color"]), :]
                cut_line[:, 3:] = np.array([1.0, 0.0, 0.0])
            
            if cut_line.shape[0] > self._cut_line_max_number:
                cut_line = np.random.permutation(
                    cut_line)[:self._cut_line_max_number, :]
                
            other_point = pc_cat[~self._where_cut_line(result["pc_color"]), :]
            # other_point = np.random.permutation(other_point)[:self._point_cloud_number - cut_line.shape[0], :]
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(other_point[..., :3])
            if use_rgb_cutline:
                pcd.colors = o3d.utility.Vector3dVector(other_point[..., 3:])
            else:
                pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(other_point[..., 3:]))

            pcd_down = pcd.farthest_point_down_sample(
                self._point_cloud_number - cut_line.shape[0])
            other_point = np.concatenate(
                [np.asarray(pcd_down.points), np.asarray(pcd_down.colors)], axis=1)

            if use_rgb_cutline:
                ret_pc = np.concatenate([cut_line, other_point], axis=0, dtype=np.float32) # point cloud
            else:
                ret_pc = np.concatenate([cut_line, other_point], axis=0, dtype=np.float32)[:, :4] # point cloud
            
            assert ret_pc.shape == (self._point_cloud_number, 6 if use_rgb_cutline else 4)
            ret_dict["point_cloud"] = ret_pc

        if "pose" in self._prepare_which:
            ret_ps = self._get_scissor_pose(state, path) # scissor pose
            assert ret_ps.shape == (8, )
            ret_dict["pose"] = ret_ps

        if "action" in self._prepare_which:
            ret_ac = self._action_embed(state["action"]) # action label
            assert ret_ac.shape == (4, )
            ret_dict["action"] = ret_ac

        if "helper_point" in self._prepare_which:
            goal_path = node.get_parent().info[".yaml"][0]
            # dirname, filename = os.path.split(goal_path)
            # assert os.path.split(dirname)[1] == filename[:-5] and \
            #     filename[-5:] == ".yaml", f"goal_path:{goal_path}"
            goal = OmegaConf.load(goal_path)
            goal_edge_set = OmegaConf.to_container(goal)["goal_edge_set"]
            ret_hp = self._get_helper_point(state, goal_edge_set)
            assert ret_hp.shape == (32, 3)
            ret_dict["helper_point"] = ret_hp

        if "auxiliary" in self._prepare_which:
            self._env.set_state(state)
            auxiliary = {}
            auxiliary["front_point"] = self._env.get_scissors_front_point("world")[0]
            auxiliary["cut_direction"] = self._env.get_scissors_cut_direction()[0][1]
            auxiliary["next_edge"] = self._get_next_edge(node, state["completed"])
            ret_dict["auxiliary"] = auxiliary
        
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
    parser.add_argument("--use-rgb-cutline", "-rgbc", action="store_true")
    parser.add_argument("--save-which", "-sw", nargs="*", default=[])
    parser.add_argument("--zfill-length", "-zl", type=int, default=5)
    args = parser.parse_args()

    prepare_which = []
    for sw in args.save_which:
        assert sw in ["point_cloud", "pose", "action", "helper_point", "auxiliary", "info"], f"{args.save_which}"
        if not sw == "info":
            prepare_which.append(sw)

    np.random.seed(args.seed)

    data_forest = DataForest([args.demos_dirs], ".pkl", [".png", ".txt", ".yaml"])
    prepare_dataset = PointCloudPrepareDataset(data_forest,
                                               robot_urdf_path=None,
                                               device=f"cuda:{args.gpuid}",
                                               prepare_which=prepare_which)
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "total_trajectory.txt"), "w") as f_obj:
        f_obj.write(str(data_forest.folder_directly_have_target_cnt))
    zfill_length = args.zfill_length
    if zfill_length == -1:
        zfill_length = len(f"{len(data_forest)}")

    for data_idx in tqdm.tqdm(range(args.start_idx, len(prepare_dataset), args.data_step)):
        data = prepare_dataset.get_item(data_idx, keep_failure=args.keep_failure, use_rgb_cutline=args.use_rgb_cutline)
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

            if "point_cloud" in args.save_which:
                np.save(os.path.join(new_data_folder,
                        f"{new_data_prefix}_point_cloud.npy"), data["point_cloud"], allow_pickle=False)
            if "pose" in args.save_which:
                np.save(os.path.join(new_data_folder,
                        f"{new_data_prefix}_scissor_pose.npy"), data["pose"], allow_pickle=False)
            if "action" in args.save_which:
                np.save(os.path.join(new_data_folder,
                        f"{new_data_prefix}_action.npy"), data["action"], allow_pickle=False)
            if "helper_point" in args.save_which:
                np.save(os.path.join(new_data_folder,
                        f"{new_data_prefix}_helper_point.npy"), data["helper_point"], allow_pickle=False)
            if "auxiliary" in args.save_which:
                np.save(os.path.join(new_data_folder,
                        f"{new_data_prefix}_auxiliary.npy"), data["auxiliary"], allow_pickle=True)
            if "info" in args.save_which:
                OmegaConf.save(omegaconf.DictConfig(data["info"]), 
                                            os.path.join(new_data_folder, f"{new_data_prefix}_info.yaml"))
        else:
            print(f"[WARN] get_item reture None: {data_forest[data_idx].get_path()}")


if __name__ == "__main__":
    main()
