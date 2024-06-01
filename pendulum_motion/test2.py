#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/pendulum_motion/test2.py
Project: /workspace/code/pendulum_motion
Created Date: Friday May 31st 2024
Author: Kaixu Chen
-----
Comment:
这里我们不需要考虑摆动中的能量损失，也就是说，摆动的幅度不会逐渐减小。

Have a good code time :)
-----
Last Modified: Friday May 31st 2024 7:30:04 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2, os, json, shutil
import itertools
import threading

def one_cycle(max_left_theta, max_right_theta, L, dt):

    x_list = []
    y_list = []
    L = 1
    # left to right 
    left_degree = max_left_theta
    while left_degree > 0:
        theta = np.radians(left_degree)
        x_list.append(-L * np.sin(theta))
        y_list.append(-L * np.cos(theta))
        left_degree -= dt

    right_degree = 0
    while right_degree < max_right_theta:
        theta = np.radians(right_degree)
        x_list.append(L * np.sin(theta))
        y_list.append(-L * np.cos(theta))
        right_degree += dt

    # right to left
    while right_degree > 0:
        theta = np.radians(right_degree)
        x_list.append(L * np.sin(theta))
        y_list.append(-L * np.cos(theta))
        right_degree -= dt

    left_degree = 0
    while left_degree < max_left_theta:
        theta = np.radians(left_degree)
        x_list.append(-L * np.sin(theta))
        y_list.append(-L * np.cos(theta))
        left_degree += dt
    
    return x_list, y_list

def save_index(x_list, file_path):

    left_index = x_list.index(min(x_list))
    right_index = x_list.index(max(x_list))

    json_info = {
        "frames": len(x_list),
        "left_index": left_index,
        "right_index": right_index
    }

    with open(file_path, 'w') as f:
        json.dump(json_info, f, indent=4)

# def modify_gap(x, y, gap):
    
#     # 根据点计算出角度
#     degree = np.degrees(np.arctan2(y,x))

#     # 根据角度来计算新的gap
#     if x < 0:
#         x += np.sin(np.radians(degree)) * gap
#     else:
#         x -= np.sin(np.radians(degree)) * gap

#     y += np.cos(np.radians(degree)) * gap

#     return x, y

def save_mp4(frames, shape_instance, shape_path, x_list, y_list, L):

    # 创建动画
    fig, ax = plt.subplots()
    plt.axis('off')
    fig.set_size_inches(5.2, 5.2)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 0)

    line, = ax.plot([], [], lw=5)
    line.set_data([], [])

    ax.add_patch(shape_instance)

    width, height = fig.get_size_inches() * fig.get_dpi()

    out = cv2.VideoWriter(shape_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(width), int(height)))

    for i in range(int(frames)):
        
        x = x_list[i]
        y = y_list[i]

        shape = shape_path.split('_')[-1].split('.')[0]
        # 根据摆的位置更新图形
        line.set_data([0, x], [0, y])

        # 感觉这边gap有问题
        if 'circle' in shape:
            shape_instance.set_center([x, y])
        elif 'rect' in shape:
            shape_instance.set_xy([x, y])
        elif 'polygon' in shape:
            shape_instance.set_xy([[0, 0], [x, y], [x+0.5, y+0.5]])            

        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        out.write(image)

    out.release()
    print(f"save {shape_path} done!")
    # # 复制文件
    # for i in range(5):
    #     shutil.copyfile(filename, f"{filename.replace('.mp4', f'_{i}.mp4')}")

def main(shape: str, infos: list):

    # 设置单摆的参数
    # g = 9.81  # 重力加速度 (m/s^2)
    
    dt = 0.1  # 时间步长

    # 非对称的运动方程
    degree = [90, 45]
    
    save_config = []

    for (left_degree, right_degree) in list(itertools.product(degree, degree)):

        print(f"left_degree: {left_degree}, right_degree: {right_degree}")

        # 保存动画到文件
        path = os.path.join("/workspace/data/pendulum", f"left{left_degree}_right{right_degree}")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # 保存index，作为融合的索引
        index_path = os.path.join("/workspace/data/pendulum/index_mapping", )

        if not os.path.exists(index_path):
            os.makedirs(index_path, exist_ok=True)

        if shape == 'circle':
            circle_config = []
            for idx, info in enumerate(infos):
                print(f"shape: {shape}, info: {info}")
                shape_instance = plt.Circle((0, 0), info[0], fc='r')  # 使用圆形表示摆的端点
                l_gap = info[1]

                # 保存圆形的配置
                config = {
                    'left_degree': left_degree,
                    'right_degree': right_degree,
                    'radius': info[0],
                    'l_gap': l_gap
                }

                circle_config.append(config)

                shape_path = os.path.join(path, f"left{left_degree}_right{right_degree}_{shape}{idx}.mp4")
                shape_index_path = os.path.join(index_path, f"left{left_degree}_right{right_degree}_{shape}{idx}.json")

                process_one_sample(left_degree, right_degree, l_gap, dt, shape_instance, shape_path, shape_index_path)

        elif shape == 'rect':
            rect_config = []
            for idx, info in enumerate(infos):

                width = info[0]
                height = info[1]
                shape_instance = plt.Rectangle((0, 0), width, height, fc='r')
                l_gap = info[2]

                config = {
                    'left_degree': left_degree,
                    'right_degree': right_degree,
                    'width': width,
                    'height': height,
                    'l_gap': l_gap
                }
                rect_config.append(config)

                shape_path = os.path.join(path, f"left{left_degree}_right{right_degree}_{shape}{idx}.mp4")
                shape_index_path = os.path.join(index_path, f"left{left_degree}_right{right_degree}_{shape}{idx}.json")

                process_one_sample(left_degree, right_degree, l_gap, dt, shape_instance, shape_path, shape_index_path)

        elif shape == 'polygon':
            polygon_config = []
            for idx, info in enumerate(infos):

                one_point = info[0]
                two_point = info[1]
                shape_instance = plt.Polygon([[0, 0], [one_point, one_point], [two_point, two_point]], closed=True, fill=True, edgecolor='r')
                l_gap = info[2]

                config = {
                    'left_degree': left_degree,
                    'right_degree': right_degree,
                    'one_point': one_point,
                    'two_point': two_point,
                    'l_gap': l_gap
                }
                polygon_config.append(config)

                shape_path = os.path.join(path, f"left{left_degree}_right{right_degree}_{shape}{idx}.mp4")
                shape_index_path = os.path.join(index_path, f"left{left_degree}_right{right_degree}_{shape}{idx}.json")

                process_one_sample(left_degree, right_degree, l_gap, dt, shape_instance, shape_path, shape_index_path)

    # save config 
    save_config = {
        'circle': circle_config,
        'rect': rect_config,
        'polygon': polygon_config
    }
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(save_config, f, indent=4)

def process_one_sample(left_degree, right_degree, L, dt, shape_value, shape_path, shape_index_path):   

    # 一次单摆运动
    # 按照4个象限来分别的话，从左到右的摆动和从右到左的摆动
    # 角度的变化是角边小，变大，半个周期结束，再变小，再变大，一个周期结束。

    x_list, y_list = one_cycle(left_degree, right_degree, L, dt)

    # 推测一周周期的frame
    total_degree = left_degree + right_degree
    frames = total_degree * 2 / dt 
    
    save_mp4(frames, shape_value, shape_path, x_list, y_list, L)
    save_index(x_list, shape_index_path)

def random_shape():

    # 配置形状，从三个形状里面选择一个
    # 一个形状里面，大小不同，以及在轴上的大小不同
    shape = ['circle', 'rect', 'polygon']

    # 随机生成形状，以及配置
    # circle
    random_radius = np.linspace(0.1, 0.5, 10)
    circle_list = list(itertools.product(random_radius, random_radius))
    
    # rect
    rect_list = list(itertools.product(random_radius, random_radius, random_radius))    
    
    # polygon
    polygon_list = list(itertools.product(random_radius, random_radius, random_radius))

    shape_info = {
        'circle': circle_list,
        'rect': rect_list,
        'polygon': polygon_list
    }
    
    return shape_info

if '__main__' == __name__:

    threads = []

    shape_info = random_shape()

    # for shape, shape_info in shape_info.items():
    #     main(shape, shape_info) # only for test

    for shape, shape_info in shape_info.items():

        thread = threading.Thread(target=main, args=(shape, shape_info))
        threads.append(thread)

    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
