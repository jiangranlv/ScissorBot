import numpy as np
import time
import ctypes
import trimesh
import taichi as ti
import torch
from typing import Dict

CUBE_COORDINATE_EPS = 1e-3
SDF_SHAPE = 4
ANGLE_IDX = 3
THETA_IDX = 4
PHI_IDX = 5

class Clock:
    def __init__(self) -> None:
        self._data = {}
        self._start_time = {}

    def start_clock(self, target_name: str):
        self._start_time[target_name] = time.time()

    def end_clock(self, target_name: str):
        assert target_name in self._start_time.keys(), \
            "[ERROR] please call start_clock first."

        if target_name not in self._data.keys():
            self._data[target_name] = []
        self._data[target_name].append(
            time.time() - self._start_time[target_name])

        self._start_time.pop(target_name)

    def get_info_str(self, drop):
        """print average time cost. drop some large values and small values"""
        assert isinstance(drop[0], int) and isinstance(drop[1], int)
        assert drop[0] >= 0 and drop[1] >= 0

        ret_str = ""
        for key in self._data.keys():
            tmp = np.array(sorted(self._data[key]))
            if tmp.shape[0] > drop[0] + drop[1]:
                if ret_str != "":
                    ret_str += "\n"
                ret_str += "[TIME] {} {}".format(
                    key, tmp[drop[0]:tmp.shape[0] - drop[1]].mean())
        return ret_str

    def print_info(self, drop):
        """print average time cost. drop some large values and small values"""
        print(self.get_info_str(drop))

    def reset(self):
        self._data = {}
        self._start_time = {}


class Queue:
    def __init__(self) -> None:
        self._data = []

    def empty(self) -> bool:
        return len(self._data) == 0
    
    def get(self) -> None:
        if self.empty():
            return None
        else:
            return self._data.pop(0)
    
    def put(self, item) -> None:
        self._data.append(item)


def get_raster_points(res=[64, 64, 64], bound=[[-1.0, -1.0, -1.0], [+1.0, +1.0, +1.0]]):
    points = np.meshgrid(
        np.linspace(bound[0][0], bound[1][0], res[0]),
        np.linspace(bound[0][1], bound[1][1], res[1]),
        np.linspace(bound[0][2], bound[1][2], res[2])
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose().astype(np.float32)
    return points


@ti.func
def remove_item_in_field_func(f: ti.template(), f_len: ti.template(), item: ti.i32) -> bool:
    """remove 'item' in a dynamic length field 'f' with length 'f_len'."""
    is_found = False
    ti.loop_config(serialize=True)
    for i in range(f_len[None]):
        if f[i] == item:
            is_found = True
        if is_found:
            f[i] = f[i + 1]
    if is_found:
        f_len[None] -= 1
    return is_found


@ti.kernel
def remove_item_in_field_kernel(f: ti.template(), f_len: ti.template(), item: ti.i32) -> bool:
    """remove 'item' in a dynamic length field 'f' with length 'f_len'."""
    return remove_item_in_field_func(f, f_len, item)


def numpy_array_to_ptr(arr: np.ndarray, ctype):
    return arr.ctypes.data_as(ctypes.POINTER(ctype))

@ti.func
def in_simulation_region_func(pos: ti.types.vector(3, ti.f32), position_bounds: ti.template()) -> bool:
    ret_val = True
    for i in ti.static(range(3)):
        if not (position_bounds[0][i] <= pos[i] and pos[i] <= position_bounds[1][i]):
            ret_val = False
    return ret_val

def slice_array(x: np.ndarray, n: int):
    """return x[:n, ...]"""
    return x[:n, ...]

def slice2D_array(x: np.ndarray, n1: int, n2: int):
    """return x[:n1, :n2, ...]"""
    return x[:n1, :n2, ...]

def pad_array(x: np.ndarray, m: int):
    """
    Pad `x` such that the first dim changes to `m` from `n`. 

    Return shape=(max(n, m), ...)
    """
    assert len(x.shape) >= 1
    n = x.shape[0]
    return np.pad(x, [(0, max(0, m-n))] + 
                     [(0, 0)] * (len(x.shape) - 1))

def pad2D_array(x: np.ndarray, m1: int, m2: int):
    """
    Pad `x` such that the first dim changes to `m1` from `n1` and second dim changes to `m2` from `n2` 

    Return shape=(max(n1, m1), max(n2, m2), ...)
    """
    assert len(x.shape) >= 2
    n1, n2 = x.shape[0:2]
    return np.pad(x, [(0, max(0, m1-n1))] + 
                     [(0, max(0, m2-n2))] + 
                     [(0, 0)] * (len(x.shape) - 2))

def export_obj(file_name, vert: np.ndarray, face: np.ndarray, verttext: np.ndarray=None):
    with open(file_name, "w") as f_obj:
        nv = vert.shape[0]
        f_obj.write(("v {:.8f} {:.8f} {:.8f}\n" * nv).format(*(vert.reshape(-1))))

        if verttext is not None:
            nvt = verttext.shape[0]
            f_obj.write(("vt {:.8f} {:.8f}\n" * nvt).format(
                *(verttext.reshape(-1))))

            nf = face.shape[0]
            f_obj.write(("f {}/{} {}/{} {}/{}\n" * nf).format(
                *(np.repeat(face + 1, 2, axis=1).reshape(-1))))
            
        else:
            nf = face.shape[0]
            f_obj.write(("f {} {} {}\n" * nf).format(
                *((face + 1).reshape(-1))))
            

def double_side_mesh(mesh: trimesh.Trimesh):
    """return a new mesh object which contain double sides of origin mesh.
    
    Example:
        {v: ..., f:[[0, 1, 2], ...]} -> {v: ..., f:[[0, 1, 2], [0, 2, 1], ...]}"""
    return trimesh.Trimesh(vertices=mesh.vertices, 
                           faces=np.concatenate([mesh.faces, mesh.faces[:, [0, 2, 1]]]))

def torch_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def floating_pose6D_to_pose7D(pose6D_dict: Dict[str, torch.Tensor], eps=1e-7) -> Dict[str, torch.Tensor]:
    pose7D_dict = {}
    for joint_name in pose6D_dict.keys():
        pose6D = pose6D_dict[joint_name]
        if len(pose6D.shape) != 2 or pose6D.shape[1] != 6:
            pose7D_dict[joint_name] = pose6D.clone()
        else:
            B, D = pose6D.shape
            assert D == 6
            translate = pose6D[:, :3]

            quaternion = torch.zeros((B, 4), dtype=pose6D.dtype, device=pose6D.device)
            quaternion[:, 1] = torch.sin(pose6D[:, THETA_IDX]) * torch.cos(pose6D[:, PHI_IDX])
            quaternion[:, 2] = torch.sin(pose6D[:, THETA_IDX]) * torch.sin(pose6D[:, PHI_IDX])
            quaternion[:, 3] = torch.cos(pose6D[:, THETA_IDX])
            qlen = quaternion.norm(dim=1)
            angle = pose6D[:, ANGLE_IDX]
            quaternion *= (torch.sin(angle / 2.0) / torch.clamp_min(qlen, eps))[:, None]
            quaternion[:, 0] = torch.cos(angle / 2.0)

            pose7D_dict[joint_name] = torch.concat([translate, quaternion], dim=1)
    return pose7D_dict