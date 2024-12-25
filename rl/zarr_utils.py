# some code borrowed from https://github.com/columbia-ai-robotics/irp.git 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import zarr
import numpy as np
from typing import List
from rl.prepare_dataset import DataForest
import tqdm

def make_zarr(npy_paths: List[str], zarr_path: str, 
              point_cloud_size=(4096, 6), point_cloud_block_dim=128,
              pose_size=(8, ), pose_block_dim=128,
              action_size=(4, ), action_block_dim=128,
              eval_ratio=0.2, eval_size=None,
              seed=0, print_info=True):
    np.random.seed(seed)
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store, overwrite=True)
    train_group = root.create_group('train')
    eval_group = root.create_group('eval')

    assert isinstance(npy_paths, list)
    f_pc = DataForest(npy_paths, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="point_cloud")
    f_pose = DataForest(npy_paths, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="scissor_pose")
    f_action = DataForest(npy_paths, target_file_suffix=[".npy"], info_file_suffix=[], target_file_name="action")
    
    data_size = len(f_pc)
    assert len(f_pose) == data_size and len(f_action) == data_size
    if eval_size is None:
        eval_size = data_size * eval_ratio
    eval_size = int(eval_size)
    train_size = data_size - eval_size
    assert eval_size >= 1 and train_size >= 1

    all_data_idx = np.random.permutation(data_size)
    train_data_idx = all_data_idx[:train_size]
    eval_data_idx = all_data_idx[train_size:]

    train_pc = train_group.zeros("point_cloud", shape=(train_size, *point_cloud_size), 
                           chunks=(point_cloud_block_dim, *point_cloud_size), dtype=np.float32)
    train_pose = train_group.zeros("pose", shape=(train_size, *pose_size), 
                           chunks=(pose_block_dim, *pose_size), dtype=np.float32)
    train_action = train_group.zeros("action", shape=(train_size, *action_size), 
                               chunks=(action_block_dim, *action_size), dtype=np.float32)
    
    eval_pc = eval_group.zeros("point_cloud", shape=(eval_size, *point_cloud_size), 
                         chunks=(point_cloud_block_dim, *point_cloud_size), dtype=np.float32)
    eval_pose = eval_group.zeros("pose", shape=(eval_size, *pose_size), 
                           chunks=(pose_block_dim, *pose_size), dtype=np.float32)
    eval_action = eval_group.zeros("action", shape=(eval_size, *action_size), 
                             chunks=(action_block_dim, *action_size), dtype=np.float32)
    
    def write_data(zarr_data: zarr.Array, zarridx2dataidx: np.ndarray, dataidx2path: DataForest, desired_shape: tuple):
        for chunk_idx in tqdm.tqdm(range(zarr_data.nchunks)):
            npdata_list = []
            zarr_idx_start = chunk_idx * zarr_data.chunks[0]
            zarr_idx_end = min((chunk_idx + 1) * zarr_data.chunks[0], zarr_data.shape[0])
            assert zarr_idx_start < zarr_idx_end
            for data_idx_ptr in range(zarr_idx_start, zarr_idx_end):
                data_idx = int(zarridx2dataidx[data_idx_ptr])
                npdata = np.load(dataidx2path[data_idx].get_path())
                assert npdata.shape == desired_shape
                npdata_list.append(npdata[np.newaxis, ...])
            zarr_data[zarr_idx_start:zarr_idx_end] = np.concatenate(npdata_list).astype(np.float32)

    write_data(train_pc, train_data_idx, f_pc, point_cloud_size)
    write_data(train_pose, train_data_idx, f_pose, pose_size)
    write_data(train_action, train_data_idx, f_action, action_size)

    write_data(eval_pc, eval_data_idx, f_pc, point_cloud_size)
    write_data(eval_pose, eval_data_idx, f_pose, pose_size)
    write_data(eval_action, eval_data_idx, f_action, action_size)

    if print_info:
        print("make successful. merged zarr:")
        print(f"{train_pc.info}\n{train_pose.info}\n{train_action.info}")
        print(f"{eval_pc.info}\n{eval_pose.info}\n{eval_action.info}")
    
    return train_data_idx, eval_data_idx

def parse_bytes(s):
    """Parse byte string to numbers

    >>> from dask.utils import parse_bytes
    >>> parse_bytes('100')
    100
    >>> parse_bytes('100 MB')
    100000000
    >>> parse_bytes('100M')
    100000000
    >>> parse_bytes('5kB')
    5000
    >>> parse_bytes('5.4 kB')
    5400
    >>> parse_bytes('1kiB')
    1024
    >>> parse_bytes('1e6')
    1000000
    >>> parse_bytes('1e6 kB')
    1000000000
    >>> parse_bytes('MB')
    1000000
    >>> parse_bytes(123)
    123
    >>> parse_bytes('5 foos')
    Traceback (most recent call last):
        ...
    ValueError: Could not interpret 'foos' as a byte unit
    """
    if isinstance(s, (int, float)):
        return int(s)
    s = s.replace(" ", "")
    if not any(char.isdigit() for char in s):
        s = "1" + s

    for i in range(len(s) - 1, -1, -1):
        if not s[i].isalpha():
            break
    index = i + 1

    prefix = s[:index]
    suffix = s[index:]

    try:
        n = float(prefix)
    except ValueError as e:
        raise ValueError("Could not interpret '%s' as a number" % prefix) from e

    try:
        multiplier = byte_sizes[suffix.lower()]
    except KeyError as e:
        raise ValueError("Could not interpret '%s' as a byte unit" % suffix) from e

    result = n * multiplier
    return int(result)

byte_sizes = {
    "kB": 10 ** 3,
    "MB": 10 ** 6,
    "GB": 10 ** 9,
    "TB": 10 ** 12,
    "PB": 10 ** 15,
    "KiB": 2 ** 10,
    "MiB": 2 ** 20,
    "GiB": 2 ** 30,
    "TiB": 2 ** 40,
    "PiB": 2 ** 50,
    "B": 1,
    "": 1,
}
byte_sizes = {k.lower(): v for k, v in byte_sizes.items()}
byte_sizes.update({k[0]: v for k, v in byte_sizes.items() if k and "i" not in k})
byte_sizes.update({k[:-1]: v for k, v in byte_sizes.items() if k and "i" in k})


def open_cached(zarr_path, mode='a', cache_size='4GB'):
    cache_bytes = 0
    if cache_size:
        cache_bytes = parse_bytes(cache_size)
    
    store = zarr.DirectoryStore(zarr_path)
    chunk_store = store
    if cache_bytes > 0:
        chunk_store = zarr.LRUStoreCache(store, max_size=cache_bytes)

    group = zarr.open_group(store=store, mode=mode, chunk_store=chunk_store)
    return group

def merge_zarr(zarr_paths: List[str], merge_path: str,
               point_cloud_size=(4096, 6), point_cloud_block_dim=128,
               pose_size=(7, ), pose_block_dim=128,
               action_size=(4, ), action_block_dim=128,
               print_info=True,
               ):
    store = zarr.DirectoryStore(merge_path)
    assert not os.path.exists(merge_path), f"Please delele {merge_path} first"
    
    root = zarr.group(store=store)
    train_group = root.create_group('train')
    eval_group = root.create_group('eval')

    train_pc = train_group.zeros("point_cloud", shape=(0, *point_cloud_size), 
                           chunks=(point_cloud_block_dim, *point_cloud_size), dtype=np.float32)
    train_pose = train_group.zeros("pose", shape=(0, *pose_size), 
                           chunks=(pose_block_dim, *pose_size), dtype=np.float32)
    train_action = train_group.zeros("action", shape=(0, *action_size), 
                               chunks=(action_block_dim, *action_size), dtype=np.float32)
    eval_pc = eval_group.zeros("point_cloud", shape=(0, *point_cloud_size), 
                         chunks=(point_cloud_block_dim, *point_cloud_size), dtype=np.float32)
    eval_pose = eval_group.zeros("pose", shape=(0, *pose_size), 
                           chunks=(pose_block_dim, *pose_size), dtype=np.float32)
    eval_action = eval_group.zeros("action", shape=(0, *action_size), 
                             chunks=(action_block_dim, *action_size), dtype=np.float32)
    
    for zarr_path in tqdm.tqdm(zarr_paths):
        group = zarr.open_group(zarr_path, "r")

        train_pc.append(group["train/point_cloud"])
        train_pose.append(group["train/pose"])
        train_action.append(group["train/action"])

        eval_pc.append(group["eval/point_cloud"])
        eval_pose.append(group["eval/pose"])
        eval_action.append(group["eval/action"])
    
    if print_info:
        print("merge successful. merged zarr:")
        print(f"{train_pc.info}\n{train_pose.info}\n{train_action.info}")
        print(f"{eval_pc.info}\n{eval_pose.info}\n{eval_action.info}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["make", "merge"], default="make")
    parser.add_argument("--numpy-data", "-n", nargs="*", type=str, default=[], help="example: ./rloutputs/data/")
    parser.add_argument("--zarr-data", "-z", nargs="*", type=str, default=[], help="example: ./rloutputs/zarrdata/")
    parser.add_argument("--merge-output-path", "-o", type=str, default="", help="example: ./rloutputs/zarr_merge/")
    parser.add_argument("--eval-ratio", "-er", type=float, default=0.2)
    parser.add_argument("--eval-size", "-es", type=int, default=None, help="overwrite eval ratio")
    parser.add_argument("--block-dim", "-b", type=int, default=128)
    args = parser.parse_args()

    if args.mode == "make":
        make_zarr(args.numpy_data, args.zarr_data[0],
                  eval_ratio=args.eval_ratio, eval_size=args.eval_size,
                  point_cloud_block_dim=args.block_dim,
                  pose_block_dim=args.block_dim,
                  action_block_dim=args.block_dim,)
    elif args.mode == "merge":
        merge_zarr(args.zarr_data, args.merge_output_path,
                   point_cloud_block_dim=args.block_dim,
                  pose_block_dim=args.block_dim,
                  action_block_dim=args.block_dim,)
    else:
        raise ValueError

if __name__ == "__main__":
    main()