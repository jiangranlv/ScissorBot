import bpy
import math
import numpy as np
import os
import sys
import os
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__) ))
sys.path.append(proj_root)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", type=str, required=True,help="Specify folder name.")
parser.add_argument("-s", "--start-idx", type=int, default=0, help="Rendering start index.")
parser.add_argument("-e", "--end-idx", type=int, default=5000, help="Rendering start index.")
parser.add_argument("--tex_file", type=str, help="Rendering start index.")
args, unknown = parser.parse_known_args()

folder_name = args.folder
start_idx = args.start_idx
end_idx = args.end_idx
_, exp_name = os.path.split(folder_name)
texture_name = exp_name[:exp_name.rfind('_')]

in_dir_ply = "{}/".format(folder_name)
out_dir = "{}/".format(folder_name)
file_prefix = "xgame_"

obj = bpy.data.objects['Cube']
bpy.data.objects.remove(obj)

# bpy.data.objects['Camera'].rotation_euler = np.array(
#     [0.9, 0.0, 1.57 * 1.5 - 3.14])  # XYZ
# bpy.data.objects['Camera'].location = np.array([-0.6, -0.6, 0.75])
# bpy.data.objects['Light'].location = np.array([2.0, -2.0, 4.0])

bpy.data.objects['Camera'].rotation_euler = np.array(
    [1.5, 0.0, 1.57 * 2.2])  # XYZ
bpy.data.objects['Camera'].location = np.array([-0.25, 0.9, 0.22])
bpy.data.objects['Light'].location = np.array([0.0, 2.0, 4.0])

scene = bpy.context.scene
scene.render.resolution_x = 1920                   # 输出图片的长宽
scene.render.resolution_y = 1080

scene.render.image_settings.file_format = 'PNG'    # 保存格式为PNG
scene.render.image_settings.color_mode = 'RGB'     # 设置图片具有RGB三个通道
scene.render.image_settings.color_depth = '8'      # 使用8位的颜色
scene.sequencer_colorspace_settings.name = 'sRGB'  # 保存时使用sRGB格式编码


for i in range(start_idx, end_idx):
    for obj in bpy.data.objects:
        if file_prefix in obj.name:
            bpy.data.objects.remove(obj)

    scene_filename = file_prefix + str(i).zfill(6)
    try:
        bpy.ops.import_mesh.ply(filepath=scene_filename + ".ply", files=[
                            {'name': scene_filename + ".ply"}], directory=in_dir_ply, filter_glob="*.ply")
    except:
        cloth_filename = file_prefix + "cloth_" + str(i).zfill(6)
        scissor_filename = file_prefix + "scissor_" + str(i).zfill(6)
        
        bpy.ops.import_mesh.ply(filepath=cloth_filename + ".ply", files=[
                                {'name': cloth_filename + ".ply"}], directory=in_dir_ply, filter_glob="*.ply")
        bpy.ops.import_mesh.ply(filepath=scissor_filename + ".ply", files=[
                                {'name': scissor_filename + ".ply"}], directory=in_dir_ply, filter_glob="*.ply")

        cloth = bpy.data.objects[cloth_filename]
    # uv_co = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
    if args.tex_file is not None :
        new_mat = bpy.data.materials.new(name="new_mat")
        new_mat.use_nodes = True
        bsdf = new_mat.node_tree.nodes["Principled BSDF"]
        texImage = new_mat.node_tree.nodes.new("ShaderNodeTexImage")
        texImage.image = bpy.data.images.load(filepath=args.tex_file)
        new_link = new_mat.node_tree.links.new(bsdf.inputs["Base Color"], texImage.outputs["Color"])

        uv_filename = 'cloth_' + "vertuv_" + str(i).zfill(6)
        uv_co = np.load(in_dir_ply + uv_filename + ".npy")
        new_uv = cloth.data.uv_layers.new(name='NewUV')
        for poly in cloth.data.polygons:
            for loop_index in poly.loop_indices:
                # print(uv_co[cloth.data.loops[loop_index].vertex_index])
                new_uv.data[loop_index].uv = uv_co[cloth.data.loops[loop_index].vertex_index]
    
        cloth.data.materials.append(new_mat)
    
    bpy.ops.render.render()
    bpy.data.images["Render Result"].save_render(
        out_dir + "{}.png".format(str(i).zfill(6)))

