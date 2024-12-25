import bpy
import math
import numpy as np
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", type=str, required=True,help="Specify folder name.")
parser.add_argument("-s", "--start-idx", type=int, default=0, help="Rendering start index.")
parser.add_argument("-e", "--end-idx", type=int, default=5000, help="Rendering start index.")
args, unknown = parser.parse_known_args()

folder_name = args.folder
start_idx = args.start_idx
end_idx = args.end_idx

in_dir_ply = "{}/".format(folder_name)
out_dir = "{}/".format(folder_name)
file_prefix = "xgame_"
file_format = '.ply'

obj = bpy.data.objects['Cube']
bpy.data.objects.remove(obj)

bpy.data.objects['Camera'].rotation_euler = np.array(
    [1.3, 0.0, 1.57 * 1.5])  # XYZ
bpy.data.objects['Camera'].location = np.array([0.8, 0.8, 0.2])
bpy.data.objects['Light'].location = np.array([2.0, 2.0, 4.0])

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

    filename = file_prefix + str(i).zfill(6) + file_format
    bpy.ops.import_mesh.ply(filepath=filename, files=[
                            {'name': filename}], directory=in_dir_ply, filter_glob="*.ply")

    bpy.ops.render.render()
    bpy.data.images["Render Result"].save_render(
        out_dir + "{}.png".format(str(i).zfill(6)))
