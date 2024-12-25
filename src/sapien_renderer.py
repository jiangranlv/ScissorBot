import os

import numpy as np
import trimesh.transformations as tra
import trimesh
from typing import Union, List, Callable
from PIL import Image

from omegaconf import OmegaConf
import sapien.core as sapien

from src.endeffector import EndEffector
from src.utils import *
from src.maths import *

from PIL import ImageColor, Image
import random
class SapienRender():
    def __init__(self,
                 offscreen_only=True,
                 device="cuda:0",
                 camera_shader="ibl",
                 uv_func=lambda x: x[:, [0, 2]] / np.array([[0.21, 0.297]]),
                 **kwargs) -> None:
        
        self._proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._engine = sapien.Engine()
        self._renderer = sapien.SapienRenderer(
            offscreen_only=offscreen_only, device=device)
        self._engine.set_renderer(self._renderer)
        self._cloth = None

        sapien.render_config.camera_shader_dir = camera_shader
        self._uv_func = uv_func

    def build_scene(self,
                    robot_urdf_path="assets/robot/ur10e/ur10e.urdf",
                    ee_urdf_path="assets/endeffector/rm65_scissors_description/urdf/rm65_scissors_description_no_mimic_floating_modify_limit.urdf",
                    cfg_path="config/paper_cutting_game_realworld_fast.yaml",
                    scissor_urdf_path="assets/scissor/only_scissors/urdf/only_scissors_no_floating_modify_limit.urdf",
                    point_light=[[[+2.0, +2.0, 4.0], [10.0, 10.0, 10.0]],
                                 [[-2.0, +2.0, 4.0], [10.0, 10.0, 10.0]],
                                 [[+2.0, -2.0, 4.0], [10.0, 10.0, 10.0]],
                                 [[-2.0, -2.0, 4.0], [10.0, 10.0, 10.0]],],
                    ambient_light=[[0.5, 0.5, 0.5]],
                    near_far=(0.1, 100),
                    width_height=(512, 512),
                    fovy=np.deg2rad(90),
                    **kwargs):
        cwd = os.getcwd()
        os.chdir(self._proj_root)

        scene_config = sapien.SceneConfig()
        self._scene = self._engine.create_scene(scene_config)

        loader: sapien.URDFLoader = self._scene.create_urdf_loader()
        loader.fix_root_link = True

        if hasattr(self, "_robot"):
            del self._robot
        if robot_urdf_path is not None:
            self._robot: sapien.KinematicArticulation = loader.load_kinematic(robot_urdf_path)

        if hasattr(self, "_endeffector"):
            del self._endeffector
        if hasattr(self, "_my_ee"):
            del self._my_ee
        if ee_urdf_path is not None:
            self._endeffector: sapien.KinematicArticulation = loader.load_kinematic(ee_urdf_path)
            cfg = OmegaConf.load(cfg_path)
            self._my_ee = EndEffector(batch_size= cfg.env.batch_size, endeffector_cfg=cfg.endeffector, output_cfg= cfg.output)                                            
        if hasattr(self, "_scissor"):
            del self._scissor
        if scissor_urdf_path is not None:
            self._scissor: sapien.KinematicArticulation = loader.load_kinematic(scissor_urdf_path)

        for position, color in point_light:
            self._scene.add_point_light(position, color)
        for color in ambient_light:
            self._scene.set_ambient_light(color)

        near, far = near_far
        width, height = width_height
        self._camera = self._scene.add_camera(
            name="camera",
            width=width,
            height=height,
            fovy=fovy,
            near=near,
            far=far,
        )

        os.chdir(cwd)

    def set_scene(self,
                  state: dict,
                  camera_pose: sapien.Pose,
                  tmp_filename="tmp.obj",
                  texture_file=None,
                  paper_metallic=0.01,
                  paper_roughness=0.5,
                  **kwargs):
        self._camera.set_pose(camera_pose)
        if self._cloth is not None:
            self._scene.remove_actor(self._cloth)
            self._renderer.clear_cached_resources()

        for ti_object_data in \
            state["_sim_env"]["ti_objects"].values() if isinstance(state["_sim_env"]["ti_objects"], dict) \
                else state["_sim_env"]["ti_objects"]:
            if ti_object_data["class"] == "Robot" and hasattr(self, "_robot"):
                robot_data = ti_object_data
                root_mat = robot_data["base_transform_matrix"] @ np.linalg.inv(
                    robot_data["base_origin_matrix"])
                self._robot.set_root_pose(sapien.Pose(
                    root_mat[:3, 3], tra.quaternion_from_matrix(root_mat)))
                self._robot.set_qpos(np.arctan2(
                    np.sin(robot_data["yrdf_cfg"]), np.cos(robot_data["yrdf_cfg"])))

            if ti_object_data["class"] == "EndEffector" and \
                    hasattr(self, "_endeffector") and hasattr(self, "_my_ee"):
                ee_data = ti_object_data
                self._my_ee.set_state([ee_data]) if isinstance(ee_data, dict) else self._my_ee.set_state(ee_data)
                root_cfg = ee_data["yrdf_cfg"]["base_joint"]
                root_mat = np.eye(4, dtype=float)
                root_mat[:3, 3] = root_cfg[0:3]
                root_mat[:3, :3] = tra.rotation_matrix(root_cfg[ANGLE_IDX], theta_phi_to_direc(
                    root_cfg[THETA_IDX], root_cfg[PHI_IDX]))[:3, :3]
                self._endeffector.set_root_pose(sapien.Pose(
                    root_mat[:3, 3], tra.quaternion_from_matrix(root_mat)))
                sapien_cfg = []
                for joint_sapien in self._endeffector.get_joints():
                    joint_name = joint_sapien.get_name()
                    if joint_name == "":
                        continue
                    j = self._my_ee.yrdf.joint_map[joint_name]
                    if j.mimic is not None:
                        sapien_cfg.append(
                            ee_data["yrdf_cfg"][j.mimic.joint] * j.mimic.multiplier + j.mimic.offset)
                    elif j.type not in ["fixed", "floating"]:
                        sapien_cfg.append(ee_data["yrdf_cfg"][joint_name])
                self._endeffector.set_qpos(sapien_cfg)

            if ti_object_data["class"] == "Scissor" and hasattr(self, "_scissor"):
                scissor_data = ti_object_data
                root_cfg = scissor_data["direct_cfg"]["joint_1"]
                root_mat = np.eye(4, dtype=float)
                root_mat[:3, 3] = root_cfg[0:3]
                root_mat[:3, :3] = tra.rotation_matrix(root_cfg[ANGLE_IDX], theta_phi_to_direc(
                    root_cfg[THETA_IDX], root_cfg[PHI_IDX]))[:3, :3]
                self._scissor.set_root_pose(sapien.Pose(
                    root_mat[:3, 3], tra.quaternion_from_matrix(root_mat)))
                self._scissor.set_qpos([scissor_data["direct_cfg"]["joint_0"]])

            if ti_object_data["class"] == "Cloth":
                cloth_data = ti_object_data
                vertices = cloth_data["vertices_pos"]
                faces = cloth_data["mesh"]["faces_vid"]
                faces = np.concatenate([faces, faces[:, [0, 2, 1]]], axis=0)
                export_obj(tmp_filename, vertices, faces, 
                           self._uv_func(cloth_data["vertices_rest_pos"]))
                builder = self._scene.create_actor_builder()
                material = self._renderer.create_material()
                material.metallic = paper_metallic
                material.roughness = paper_roughness
                if texture_file is not None:
                    material.diffuse_texture_filename = texture_file
                builder.add_visual_from_file(
                    filename=tmp_filename, material=material)
                self._cloth = builder.build_static(name='cloth')
                self._cloth.set_pose(sapien.Pose(
                    [0.0, 0.0, 0.0], tra.quaternion_from_matrix(np.eye(4, dtype=float))))
                if os.path.exists(tmp_filename):
                    os.remove(tmp_filename)

    def render(self, 
               calculate_rgba=True, 
               calculate_point_cloud=False,
               calculate_depth = True,
               get_segment = False,
               filter_depth = True,
               global_noise = True,
               **kwargs):
        return_val = {}
        self._scene.step()
        self._scene.update_render()
        self._camera.take_picture()

        if calculate_point_cloud or calculate_rgba or calculate_depth:
            rgba = self._camera.get_float_texture('Color')  # [H, W, 4]

        if calculate_rgba:
            rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
            rgba_pil = Image.fromarray(rgba_img)
            return_val["rgba"] = rgba_pil
            if get_segment:
                segment = self._camera.get_visual_actor_segmentation()
                colormap = sorted(set(ImageColor.colormap.values()))
                color_palette = np.array([ImageColor.getrgb(color) for color in colormap],
                                        dtype=np.uint8)
                
                label0_image = segment[..., 0].astype(np.uint8)  # mesh-level
                label1_image = segment[..., 1].astype(np.uint8)  # actor-level
                label0_image = (label0_image == 6) * 10
                label0_pil = Image.fromarray(color_palette[label0_image])
                label1_pil = Image.fromarray(color_palette[label1_image])
                return_val['segment'] = [label0_pil, label1_pil]

        if calculate_point_cloud:
            position = self._camera.get_float_texture('Position')  # [H, W, 4]
            points_opengl = position[..., :3][position[..., 3] < 1]
            points_color = rgba[position[..., 3] < 1][..., :3]
            model_matrix = self._camera.get_model_matrix()
            points_world = points_opengl @ model_matrix[:3, :3].T + \
                model_matrix[:3, 3]
            return_val["pc_position"] = points_world
            return_val["pc_color"] = points_color
            if get_segment:
                segment = self._camera.get_visual_actor_segmentation()
                points_seg = segment[position[..., 3] < 1][..., 0].astype(np.uint8)
                # print(segment.shape, np.unique(segment[..., 1]))
                # print(segment.shape, np.unique(segment[..., 0]))
                # counts = np.bincount(segment[position[..., 3] < 1][..., 0].reshape(-1))
                # most_common_value = np.argmax(counts)
                # print('most_common', most_common_value)
                # colormap = sorted(set(ImageColor.colormap.values()))
                # color_palette = np.array([ImageColor.getrgb(color) for color in colormap],
                #                         dtype=np.uint8)
                # points_seg = color_palette[points_seg]
                return_val['pc_segment'] = points_seg

        if calculate_depth:
            position = self._camera.get_float_texture('Position')  # [H, W, 4]
            depth = -position[..., 2]
            depth = torch.from_numpy(depth)
            if filter_depth:
                depth = local_filter_depth(depth)
            if global_noise:
                depth += 2 * (torch.rand(depth.shape) - 0.5)  * 0.001 # random noise
            
            xyz_map = depth_to_xyz(depth, self._camera.get_intrinsic_matrix()) 
            pc_world, valid_index = xyz_to_pc(xyz_map, self._camera.get_model_matrix())
            pc_color = rgba[valid_index][:, :3]
            return_val['pc_position'] = pc_world
            return_val['pc_color'] = pc_color

            if get_segment:
                segment = self._camera.get_visual_actor_segmentation()
                depth_seg = segment[:, :, 0].astype(np.uint8)
                pc_seg = depth_seg[valid_index]
                return_val['pc_segment'] = pc_seg
                

        return return_val

def local_filter_depth(depth):
    depth[np.where(depth == 0)] = np.inf

    kernel = random.choice([1, 3, 5, 7, 9]) 
    depth = torch.nn.functional.pad(depth , pad = (kernel//2, kernel//2), mode= 'replicate').unsqueeze(-1)
    filter_depth= torch.nn.functional.avg_pool2d(depth, kernel_size = (kernel, 1), stride = 1).squeeze(-1)
    return filter_depth
    
def xyz_to_pc(xyz_map, extrinsic):
    valid_index = xyz_map[..., 2] < np.inf
    # transfrom to opengl camera space
    points_camera_space = xyz_map[..., :3][valid_index].type(torch.float64) @ tra.euler_matrix(np.pi, 0, 0)[:3, :3]

    model_matrix = extrinsic
    points_world = points_camera_space @ model_matrix[:3, :3].T + \
        model_matrix[:3, 3]
    
    return points_world, valid_index
    
def depth_to_xyz(depth_map, intrinsics):
    # Create grid of pixel coordinates
    height, width = depth_map.shape
    y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    y = y.float()
    x = x.float()

    # Convert pixel coordinates to normalized device coordinates (NDC)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    ndc_x = (x - cx) / fx
    ndc_y = (y - cy) / fy

    # Convert NDC to 3D coordinates
    depth = depth_map # Convert depth from millimeters to meters
    x_3d = ndc_x * depth
    y_3d = ndc_y * depth
    z_3d = depth

    # Stack 3D coordinates
    xyz = torch.stack((x_3d, y_3d, z_3d), dim=-1)

    return xyz




