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
import cv2

# 设置单摆的参数
g = 9.81  # 重力加速度 (m/s^2)
L = 1.0   # 单摆长度 (m)
theta0 = np.pi / 6  # 初始角度 (45度)
omega0 = 0.0        # 初始角速度
asymmetry = 0.1  # 不对称系数

# 非对称的运动方程
left_theta0 = 90
right_theta0 = 90

# 初始化
dt = 0.1  # 时间步长

def one_cycle(max_left_theta, max_right_theta, L, dt):

    x_list = []
    y_list = []

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

# 推测一周周期的frame
total_degree = left_theta0 + right_theta0
frames = total_degree * 2 / dt 

for _ in range(1):
    # 一次单摆运动
    # 按照4个象限来分别的话，从左到右的摆动和从右到左的摆动
    # 角度的变化是角边小，变大，半个周期结束，再变小，再变大，一个周期结束。
    x_list, y_list = one_cycle(left_theta0, right_theta0, L, dt)

print(f"x_min: {min(x_list)}, x_max: {max(x_list)}")

# 创建动画
fig, ax = plt.subplots()
ax.set_xlim(-L-0.1, L+0.1)
ax.set_ylim(-L-0.1, 0)
line, = ax.plot([], [], lw=5)

# 这里可以换成其他的形状
pendulum_bob = plt.Circle((0, 0), 0.1, fc='r')  # 使用圆形表示摆的端点
pandulum_rect = plt.Rectangle((0, 0), 0.2, 0.1, fc='r')
polygon = plt.Polygon([[0, 0], [0.1, 0.1], [0.1, -0.1]], closed=True, fill=True, edgecolor='r')

def init():
    line.set_data([], [])
    pendulum_bob.set_center((0, 0))
    pandulum_rect.set_xy([0, 0])
    polygon.set_xy([[0, 0], [0.1, 0.1], [0.1, -0.1]])

    ax.add_patch(pendulum_bob)
    ax.add_patch(pandulum_rect)
    ax.add_patch(polygon)

    return line, pendulum_bob

def update(frame):
    x = x_list[frame]
    y = y_list[frame]
    line.set_data([0, x], [0, y])
    pendulum_bob.set_center((x, y))
    pandulum_rect.set_xy([x-0.1, y-0.1])
    polygon.set_xy([[x, y], [x+0.1, y+0.1], [x+0.1, y-0.1]])
    # print(f"frame: {frame}, x: {x}, y: {y}")
    return line, pendulum_bob

def save_mp4():
    filename = "asymmetric_pendulum_motion.mp4"
    fps = 30 
    width, height = fig.get_size_inches() * fig.get_dpi()
    size = (int(width), int(height))

    init()

    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(int(frames)):
        update(i)

        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        out.write(image)
    
# 保存动画到文件
save_mp4()