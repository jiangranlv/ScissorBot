import numpy as np
import yaml
from PIL import Image, ImageDraw
import sys
import sys
import os
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__) ))
sys.path.append(proj_root)
# PAPER_LENGTH = 0.3
PAPER_LENGTH = np.array([0.210, 0.297])
eps = 1e-4
# x= [0, 0.034, 0.086, 0.120, 0.142, 0.113, 0.083, 0.0742, 0.133, 0.229, 0.05, 0.022, 0.041]
# y = [0.2, 0.190, 0.177, 0.161, 0.140, 0.123,0.106, 0.0181, 0.007,0.006, 0.1, 0.069,0.035]
# x = [0, 0.099, 0.057, 0.213, 0.123, 0.256, 0.189]
# y = [0.2, 0.263, 0.134, 0.223, 0.065, 0.139, 0]

# x = [0.1, 0.139, 0.080, 0.139, 0.190, 0.201]
# z = [0, 0.047, 0.089, 0.151, 0.084, 0.132]

x = [0, 0.019, 0.035, 0.049, 0.061, 0.080, 0.099, 0.119, 0.141, 0.160, 0.176]
z = [0.160, 0.174, 0.180, 0.179, 0.173, 0.160, 0.148, 0.142, 0.146, 0.156, 0.160]
def generate_trajctory(goal_file, func):
    # y_values = np.linspace(0, 0.2, 10)
    # # 计算相应的纵坐标
    # x_values = func(y_values)
    # x_values = x
    # z_values = z
    x_values, z_values = func()
    
    samples = np.column_stack((x_values, np.zeros_like(x_values)))
    samples = np.column_stack((samples,z_values))
    # samples = np.column_stack((x_values, y_values))
    # samples = np.column_stack((samples, np.zeros_like(x_values)))
    # sorted_indices = np.argsort(y_values)[::-1]
    # samples = samples[sorted_indices]
    samples[0, 0] = 0.0
    edge = np.stack((samples[:-1], samples[1:]), 1)
    with open(goal_file, 'a') as f:
        try:
            yaml.safe_dump(dict(goal_edge_set = edge.tolist()), f)
        except yaml.YAMLError as e:
            print(e)

def generate_texture_img(goal_file):
    with open(goal_file, 'r') as f:
        try:
            data = yaml.safe_load(f)
            print(data)
        except yaml.YAMLError as e:
            print(e)

    goal_edge_set = np.array(data['goal_edge_set'])[:,:,[0, 2]] * 1024 / PAPER_LENGTH
    
    img = Image.new('RGB', (1024, 1024), color='white')
    draw = ImageDraw.Draw(img)
    
    for edge in goal_edge_set:
        edge[:, 1] = 1024 - edge[:, 1]
        draw.line(tuple(edge.reshape(-1)), fill='red', width=5)
    
    _, file_name = os.path.split(goal_file)
    save_name = file_name.replace('yaml', 'png')
    img.save(proj_root + f'/assets/texture/{save_name}')
    
def semi_circle(x):
    return np.sqrt(0.01 - (x- 0.1)  ** 2 + eps )

def sigmoid(x):
    y =  1 / (1 + np.exp(-  60* (0.1 - x)))
    return normalize(y, 0, 0.3)

def normalize(x, lower, upper):
    x = (x -min(x)) /  (max(x) - min(x))
    return (upper- lower) * x / 2 + lower /2

def s_curve(x):
    a = 0.81
    b = -6.0
    c = 14.125
    d = -7.0
    y = a*(x-0.15)*3 + b*(x-0.15)*2 + c*(x-0.15) + d
    return normalize(y, 0, 0.3)
    

def oval():
    x = np.linspace(0.1, 0.297, 10)
    z = np.sqrt(0.197* 0.197 - (x- 0.297)  ** 2 + eps )
    return x, z

def vertival_circle2():
    z = np.linspace(0.0, 0.17, 6)
    z = np.concatenate((z, np.linspace(0.175, 0.197, 4)))
    x = np.sqrt(0.197* 0.197 - z  ** 2 + eps )
    z = z[::-1]
    x = x[::-1]
    return x, z

if __name__ == '__main__':
    goal_file = sys.argv[-1]
    generate_trajctory(goal_file, vertival_circle2)
    generate_texture_img(goal_file)

