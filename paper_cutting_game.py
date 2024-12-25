import pathlib
import math
import pickle

import hydra
from omegaconf import DictConfig
from src.maths import *
from src.scissor import default_compute_ee_pose


from src.paper_cutting_environment import PaperCuttingEnvironment


def uv_func(x: np.ndarray) -> np.ndarray:
    return x[:, 0:2] / 0.3


@hydra.main(config_path="./config", config_name=pathlib.Path(__file__).stem, version_base='1.3')
def main(cfg: DictConfig):
    env = PaperCuttingEnvironment(cfg)

    for reset_iter in range(1):
        env.reset()

        if cfg.robot.use_robot:
            mat = env.get_robot_base_cfg()
            mat[:3, 3] = np.array([-0.3, 0.3, -1.0])
            env.set_robot_base_cfg(mat)

        env.append_constraints({
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([0.3, 0.0, 0.0]),
            2: np.array([0.3, 0.3, 0.0]),
            3: np.array([0.0, 0.3, 0.0]),
        }, 1.0)
        env.set_scissors_pose([{
            "joint_0": 0.0,
            "joint_1": [0.32, 0.098, 0.0, 0.0, 0.0, 0.0]
        }], default_compute_ee_pose)

        rotate_angle_avg = math.pi / 8

        for i in range(6):
            env.append_scissors_action([{
                "Action": "Close",
                "Angle": 0.45,
                "Time": 0.5
            }])

            env.append_scissors_action([{
                "Action": "Open",
                "Angle": 0.45,
                "Time": 0.5
            }])

            print("[TEST] current position of [0.1, 0.1, 0.001]:",
                env.get_current_pos_given_rest(np.array([0.1, 0.1, 0.001])))
            print("[TEST] range", env.get_scissor_open_range())

            states = env.simulate(1.0, 1, compute_ee_pose=default_compute_ee_pose, uv_func=uv_func)
            # env.replay(states, replay_frequency=5)

            np.save("edge{}.npy".format(i), np.array(
                env.get_split_edge_rest_position(), dtype=object), allow_pickle=True)
            np.save("proj{}.npy".format(i), np.array(
                env.get_front_point_projection(), dtype=object), allow_pickle=True)
            pickle.dump(states[-1], open(f"state{i}.pkl", "bw"))

            print("[TEST]", env.get_current_cloth_state())
            print("[TEST]", env.get_split_edge_rest_position())
            print("[TEST]", env.get_current_scissors_model_penetration())
            print("[TEST]", env.get_scissors_mesh())
            print("[TEST]", env.get_cloth_mesh())
            print("[TEST] front point at theta = -0.25:",
                env.compute_scissors_front_point_given_theta("world", [-0.25]))
            print("[TEST] theta at dist = 0.03:",
                env.compute_theta_given_front_point_dist([0.03]))

            scissors_pose = env.get_scissors_pose()
            pos, vec, axs = env.get_scissors_cut_direction()[0]
            theta_z = math.atan2(vec[1], vec[0])

            env.append_scissors_action([{
                "Action": "Move",
                "Displacement": [0.07 * math.cos(theta_z), 0.07 * math.sin(theta_z), 0.0, 0.0, 0.0, 0.0],
                "Time": 1.0
            }])

            env.append_scissors_action([{
                "Action": "Move",
                "Displacement": [-0.005 * math.cos(theta_z), -0.005 * math.sin(theta_z), 0.0, 0.0, 0.0, 0.0],
                "Time": 0.2
            }])

            env.append_scissors_action([{
                "Action": "Move",
                "Displacement": [0.0, 0.0, 0.0, rotate_angle_avg, 0.0, 0.0],
                "Time": 0.6
            }])

            env.simulate(2.0, True, compute_ee_pose=default_compute_ee_pose, uv_func=uv_func)

        env.simulate(6.0, True, compute_ee_pose=default_compute_ee_pose, uv_func=uv_func)


if __name__ == "__main__":
    main()
