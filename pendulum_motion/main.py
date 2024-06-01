
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

# 设置单摆的参数
g = 9.81  # 重力加速度 (m/s^2)
L = 1.0   # 单摆长度 (m)
theta0 = np.pi / 6  # 初始角度 (45度)
omega0 = 0.0        # 初始角速度
asymmetry = 0.1  # 不对称系数

# 定义运动方程
def pendulum_motion(theta, omega, t, dt):
    omega_new = omega - (g / L) * np.sin(theta) * dt
    theta_new = theta + omega_new * dt
    return theta_new, omega_new

# 初始化
dt = 0.01  # 时间步长
t_max = 9000  # 模拟总时间
T = 2 * np.pi * np.sqrt(L / g)  # 单摆周期
n_steps = int(t_max / dt)
theta = theta0
omega = omega0

# 存储运动轨迹
theta_list = []
x_list = []
y_list = []

# 计算单摆运动
for i in range(n_steps):
    theta, omega = pendulum_motion(theta, omega, i*dt, dt)
    theta_list.append(theta)
    x_list.append(L * np.sin(theta))
    y_list.append(-L * np.cos(theta))

print(f"x_min: {min(x_list)}, x_max: {max(x_list)}")

# 创建动画
fig, ax = plt.subplots()
ax.set_xlim(-L-0.2, L+0.2)
ax.set_ylim(-L-0.2, 0)
line, = ax.plot([], [], lw=5)
pendulum_bob = plt.Circle((0, 0), 0.1, fc='r')  # 使用圆形表示摆的端点

def init():
    line.set_data([], [])
    pendulum_bob.set_center((0, 0))
    ax.add_patch(pendulum_bob)
    return line, pendulum_bob

def update(frame):
    x = x_list[frame]
    y = y_list[frame]
    line.set_data([0, x], [0, y])
    pendulum_bob.set_center((x, y))
    # print(f"frame: {frame}, x: {x}, y: {y}")
    return line, pendulum_bob

# ani = animation.FuncAnimation(fig, update, frames=n_steps, init_func=init, blit=True, interval=dt*1000)

def save_mp4():
    filename = "asymmetric_pendulum_motion.mp4"
    fps = 30 
    width, height = fig.get_size_inches() * fig.get_dpi()
    size = (int(width), int(height))

    init()

    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(n_steps):
        update(i)
        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        out.write(image)
    
# 保存动画到文件
# ani.save('asymmetric_pendulum_motion.gif', writer="pillow", fps=30)
save_mp4()