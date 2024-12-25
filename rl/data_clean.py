import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pprint
import shutil
from rl.prepare_dataset import *


def shape_check(npy_paths: List[str], pc_shape=(4096, 6), pose_shape=(7, ), action_shape=(4, )):
    assert isinstance(npy_paths, list)
    f_pc = DataForest(npy_paths, target_file_suffix=[
                      ".npy"], info_file_suffix=[], target_file_name="point_cloud")
    f_pose = DataForest(npy_paths, target_file_suffix=[
                        ".npy"], info_file_suffix=[], target_file_name="scissor_pose")
    f_action = DataForest(npy_paths, target_file_suffix=[
                          ".npy"], info_file_suffix=[], target_file_name="action")

    data_size = len(f_pc)
    assert data_size == len(f_pose) and data_size == len(f_action)
    delete_path_list = []

    def append_delete_path(delete_path_list, index):
        delete_path_list.append(f_pc[i].get_path())
        delete_path_list.append(f_pose[i].get_path())
        delete_path_list.append(f_action[i].get_path())
    for i in range(data_size):
        try:
            pc = np.load(f_pc[i].get_path())
            ps = np.load(f_pose[i].get_path())
            ac = np.load(f_action[i].get_path())
        except ValueError:
            print(
                f"[WARNING] When loading {f_pc[i].get_path()} {f_pose[i].get_path()} {f_action[i].get_path()}, ValueError raised.")
            append_delete_path(delete_path_list, i)
        except Exception:
            raise RuntimeError("Unknown exception.")

        try:
            assert pc.shape == pc_shape and ps.shape == pose_shape and ac.shape == action_shape
        except AttributeError:
            print(
                f"[WARNING] When checking shape of {f_pc[i].get_path()} {f_pose[i].get_path()} {f_action[i].get_path()}, AttributeError raised.")
            append_delete_path(delete_path_list, i)
        except AssertionError:
            print(
                f"[WARNING] When checking shape {pc.shape} {ps.shape} {ac.shape}, not equal to desired shape {pc_shape} {pose_shape} {action_shape}, AssertionError raised.")
            append_delete_path(delete_path_list, i)
    
    print("Error list:")
    pprint.pprint(delete_path_list)

    while True:
        response = input("Delete all?: y/n\n")
        if response in "yY":
            for path in delete_path_list:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            return
        elif response in "nN":
            return
        else:
            print(f"Unknown response:{response}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--numpy-data", "-n", nargs="*",
                        type=str, default=[], help="example: ./rloutputs/data/")
    args = parser.parse_args()
    shape_check(args.numpy_data)


if __name__ == "__main__":
    main()
