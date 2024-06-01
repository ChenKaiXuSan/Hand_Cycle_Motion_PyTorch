# #!/usr/bin/env python3
# # -*- coding:utf-8 -*-
# '''
# File: /workspace/code/pendulum_motion/test.py
# Project: /workspace/code/pendulum_motion
# Created Date: Friday May 31st 2024
# Author: Kaixu Chen
# -----
# Comment:

# Have a good code time :)
# -----
# Last Modified: Friday May 31st 2024 7:01:45 am
# Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
# -----
# Copyright (c) 2024 The University of Tsukuba
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	---------------------------------------------------------
# '''
# import numpy as np
# import matplotlib.pyplot as plt

# # 创建时间数组
# t = np.linspace(0, 2*np.pi, 1000)  # 从0到2π生成1000个点

# # 创建周期性运动函数，这里使用正弦函数
# amplitude = 1.0  # 振幅
# frequency = 1.0  # 频率
# phase = 0.0      # 相位
# periodic_motion = amplitude * np.sin(frequency * t + phase)

# # 绘制图形
# plt.plot(t, periodic_motion, label='Periodic Motion')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Periodic Motion: Sinusoidal Function')
# plt.grid(True)
# plt.legend()
# plt.savefig('periodic_motion.png')
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

def plot_pendulum(amplitude):
    fig, ax = plt.subplots()

    # 绘制单摆线
    ax.plot([0, 0], [-1, 1], 'k', lw=2)

    # 绘制小锤
    circle = plt.Circle((0, amplitude), 0.1, color='b')
    ax.add_artist(circle)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1 + amplitude)
    ax.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('pendulum.png')
    plt.show()

# 指定振幅并绘制
amplitude = 0.5
plot_pendulum(amplitude)
