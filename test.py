"""
测试文件：加载训练好的PPO模型，运行25步测试，
生成UAV飞行轨迹图和用户任务处理时延图像。
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from train import UAVEnv, num_uavs, num_users

# 自动检测系统中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 测试参数
TEST_STEPS = 25
MODEL_PATH = "ppo_uav_ris_10"


def run_test(model_path=MODEL_PATH, test_steps=TEST_STEPS):
    """加载模型并运行测试，返回记录的数据"""

    # 加载模型
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        raise FileNotFoundError(
            f"模型文件 '{model_path}.zip' 不存在，请先运行 train.py 训练模型。"
        )

    model = PPO.load(model_path)

    # 创建测试环境
    env = UAVEnv()
    obs, info = env.reset()

    # 数据记录
    # UAV轨迹: (test_steps+1, num_uavs, 2)，包含初始位置
    uav_trajectories = np.zeros((test_steps + 1, num_uavs, 2))
    uav_trajectories[0] = env.uav_positions.copy()

    # 用户时延记录: 通信时延、计算时延、回传时延、总时延
    comm_delays = np.zeros((test_steps, num_users))
    comp_delays = np.zeros((test_steps, num_users))
    return_delays = np.zeros((test_steps, num_users))
    total_delays = np.zeros(test_steps)

    # 用户位置（每步可能不同，记录最后一步用于标注）
    user_positions_record = np.zeros((test_steps, num_users, 2))

    print(f"\n{'=' * 70}")
    print(f"开始测试：共 {test_steps} 步")
    print(f"{'=' * 70}")

    for step in range(test_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        # 记录UAV位置
        uav_trajectories[step + 1] = env.uav_positions.copy()

        # 记录时延
        comm_delays[step] = np.array(env.users_comm_delay)
        comp_delays[step] = np.array(env.users_comp_delay)
        return_delays[step] = np.array(env.users_return_delay)
        total_delays[step] = env.total_time

        # 记录用户位置
        user_positions_record[step] = env.user_positions.copy()

        print(f"Step {step + 1:3d} | Reward: {reward:.4f} | "
              f"Total Delay: {env.total_time:.4f}s | "
              f"Decisions: {info['user_decisions']}")

        if done:
            obs, info = env.reset()

    print(f"\n{'=' * 70}")
    print(f"测试完成")
    print(f"{'=' * 70}\n")

    return {
        'uav_trajectories': uav_trajectories,
        'comm_delays': comm_delays,
        'comp_delays': comp_delays,
        'return_delays': return_delays,
        'total_delays': total_delays,
        'user_positions': user_positions_record,
        'test_steps': test_steps,
    }


def plot_uav_trajectories(data, save_path="./uav_trajectories.png"):
    """绘制UAV飞行轨迹图"""
    trajectories = data['uav_trajectories']
    user_positions = data['user_positions']
    test_steps = data['test_steps']

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    colors = ['#e74c3c', '#2ecc71', '#3498db']  # 红、绿、蓝
    markers = ['o', 's', '^']

    for i in range(num_uavs):
        xs = trajectories[:, i, 0]
        ys = trajectories[:, i, 1]

        # 绘制轨迹线
        ax.plot(xs, ys, color=colors[i], linewidth=1.5, alpha=0.7,
                label=f'UAV {i + 1} trajectory')

        # 起点标记
        ax.scatter(xs[0], ys[0], color=colors[i], marker=markers[i],
                   s=200, zorder=5, edgecolors='black', linewidths=1.5)
        ax.annotate(f'UAV{i + 1} start', (xs[0], ys[0]),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=9, fontweight='bold', color=colors[i])

        # 终点标记
        ax.scatter(xs[-1], ys[-1], color=colors[i], marker='*',
                   s=300, zorder=5, edgecolors='black', linewidths=1.5)
        ax.annotate(f'UAV{i + 1} end', (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(10, -15),
                    fontsize=9, fontweight='bold', color=colors[i])

        # 步数标注（每5步标注一次）
        for t in range(0, test_steps + 1, 5):
            if t > 0:
                ax.annotate(f'{t}', (xs[t], ys[t]),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=7, color=colors[i], alpha=0.8)

    # 绘制最后一步的用户位置
    last_user_pos = user_positions[-1]
    ax.scatter(last_user_pos[:, 0], last_user_pos[:, 1],
               color='gray', marker='x', s=80, zorder=4, label='Ground Users')
    for k in range(num_users):
        ax.annotate(f'GT{k}', (last_user_pos[k, 0], last_user_pos[k, 1]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color='gray')

    # RIS位置
    ax.scatter(0, 0, color='gold', marker='D', s=200, zorder=5,
               edgecolors='black', linewidths=1.5, label='RIS')
    ax.annotate('RIS', (0, 0), textcoords="offset points", xytext=(10, 10),
                fontsize=10, fontweight='bold', color='gold')

    ax.set_xlim(-420, 420)
    ax.set_ylim(-420, 420)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'UAV Flight Trajectories ({test_steps} steps)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"UAV飞行轨迹图已保存至: {save_path}")


def plot_user_delays(data, save_path="./user_task_delays.png"):
    """绘制用户任务处理时延图像"""
    comm_delays = data['comm_delays']
    comp_delays = data['comp_delays']
    return_delays = data['return_delays']
    total_delays = data['total_delays']
    test_steps = data['test_steps']

    steps = np.arange(1, test_steps + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)

    # === 子图1：各用户通信时延 ===
    ax1 = axes[0, 0]
    for k in range(num_users):
        ax1.plot(steps, comm_delays[:, k], linewidth=1.2, alpha=0.8, label=f'GT{k}')
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Communication Delay (s)', fontsize=11)
    ax1.set_title('Communication Delay per User', fontsize=13)
    ax1.legend(fontsize=7, ncol=2, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # === 子图2：各用户计算时延 ===
    ax2 = axes[0, 1]
    for k in range(num_users):
        ax2.plot(steps, comp_delays[:, k], linewidth=1.2, alpha=0.8, label=f'GT{k}')
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Computation Delay (s)', fontsize=11)
    ax2.set_title('Computation Delay per User', fontsize=13)
    ax2.legend(fontsize=7, ncol=2, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # === 子图3：各用户回传时延 ===
    ax3 = axes[1, 0]
    for k in range(num_users):
        ax3.plot(steps, return_delays[:, k], linewidth=1.2, alpha=0.8, label=f'GT{k}')
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Return Delay (s)', fontsize=11)
    ax3.set_title('Return Delay per User', fontsize=13)
    ax3.legend(fontsize=7, ncol=2, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # === 子图4：系统总时延 ===
    ax4 = axes[1, 1]
    ax4.plot(steps, total_delays, 'r-o', linewidth=2, markersize=4, label='Total System Delay')
    ax4.fill_between(steps, 0, total_delays, alpha=0.2, color='red')
    ax4.set_xlabel('Step', fontsize=11)
    ax4.set_ylabel('Total System Delay (s)', fontsize=11)
    ax4.set_title('Total System Delay per Step', fontsize=13)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'User Task Processing Delays ({test_steps} steps)', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"用户任务处理时延图已保存至: {save_path}")


if __name__ == "__main__":
    data = run_test(model_path=MODEL_PATH, test_steps=TEST_STEPS)
    plot_uav_trajectories(data, save_path="./uav_trajectories.png")
    plot_user_delays(data, save_path="./user_task_delays.png")
