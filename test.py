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

    # UAV负载记录: (test_steps, num_uavs)
    uav_loads = np.zeros((test_steps, num_uavs))
    # 用户决策记录: (test_steps, num_users)
    user_decisions_record = np.zeros((test_steps, num_users), dtype=int)

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

        # 记录UAV负载和用户决策
        uav_loads[step] = np.array(env.uav_L)
        user_decisions_record[step] = env.user_decisions.copy()

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
        'uav_loads': uav_loads,
        'user_decisions': user_decisions_record,
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
                   s=250, zorder=5, edgecolors='black', linewidths=2)
        ax.annotate(f'UAV{i + 1} 起点', (xs[0], ys[0]),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=10, fontweight='bold', color=colors[i],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # 终点标记
        ax.scatter(xs[-1], ys[-1], color=colors[i], marker='*',
                   s=350, zorder=5, edgecolors='black', linewidths=2)
        ax.annotate(f'UAV{i + 1} 终点', (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(10, -15),
                    fontsize=10, fontweight='bold', color=colors[i],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

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


def plot_uav_load_balance(data, save_path="./uav_load_balance.png"):
    """绘制每一步UAV的负载均衡图"""
    uav_loads = data['uav_loads']
    test_steps = data['test_steps']

    steps = np.arange(1, test_steps + 1)
    colors = ['#e74c3c', '#2ecc71', '#3498db']

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    bar_width = 0.25
    for i in range(num_uavs):
        offsets = steps + (i - 1) * bar_width
        ax.bar(offsets, uav_loads[:, i], width=bar_width, color=colors[i],
               alpha=0.85, label=f'UAV {i + 1}', edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Load (Mbit)', fontsize=12)
    ax.set_title(f'UAV Load Balance per Step ({test_steps} steps)', fontsize=14)
    ax.set_xticks(steps)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"UAV负载均衡图已保存至: {save_path}")


def plot_offload_distribution(data, save_path="./offload_distribution.png"):
    """绘制每一步所有用户的本地计算和卸载计算分布"""
    user_decisions = data['user_decisions']
    test_steps = data['test_steps']

    steps = np.arange(1, test_steps + 1)

    # 统计每步中 本地计算用户数 和 卸载到各UAV的用户数
    local_counts = np.zeros(test_steps)
    uav_counts = np.zeros((test_steps, num_uavs))

    for t in range(test_steps):
        for k in range(num_users):
            dec = user_decisions[t, k]
            if dec == 0:
                local_counts[t] += 1
            else:
                uav_counts[t, dec - 1] += 1

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    # 堆叠柱状图
    ax.bar(steps, local_counts, color='#95a5a6', label='Local', edgecolor='white', linewidth=0.5)
    bottom = local_counts.copy()
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    for i in range(num_uavs):
        ax.bar(steps, uav_counts[:, i], bottom=bottom, color=colors[i],
               label=f'Offload to UAV {i + 1}', edgecolor='white', linewidth=0.5)
        bottom += uav_counts[:, i]

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Number of Users', fontsize=12)
    ax.set_title(f'User Computation Distribution per Step ({test_steps} steps)', fontsize=14)
    ax.set_xticks(steps)
    ax.set_yticks(range(0, num_users + 1))
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"用户计算分布图已保存至: {save_path}")


def plot_local_vs_offload_delay(data, save_path="./local_vs_offload_delay.png"):
    """绘制每一步本地时延和卸载时延的对比折线图"""
    user_decisions = data['user_decisions']
    comm_delays = data['comm_delays']
    comp_delays = data['comp_delays']
    return_delays = data['return_delays']
    test_steps = data['test_steps']

    steps = np.arange(1, test_steps + 1)

    # 每步的本地总时延 和 卸载总时延
    local_delay_per_step = np.zeros(test_steps)
    offload_delay_per_step = np.zeros(test_steps)

    for t in range(test_steps):
        for k in range(num_users):
            user_total = comm_delays[t, k] + comp_delays[t, k] + return_delays[t, k]
            if user_decisions[t, k] == 0:
                local_delay_per_step[t] += user_total
            else:
                offload_delay_per_step[t] += user_total

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    ax.plot(steps, local_delay_per_step, 'o-', color='#e67e22', linewidth=2,
            markersize=5, label='Local Computation Delay')
    ax.plot(steps, offload_delay_per_step, 's-', color='#2980b9', linewidth=2,
            markersize=5, label='Offload Computation Delay')

    ax.fill_between(steps, local_delay_per_step, alpha=0.15, color='#e67e22')
    ax.fill_between(steps, offload_delay_per_step, alpha=0.15, color='#2980b9')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Total Delay (s)', fontsize=12)
    ax.set_title(f'Local vs Offload Delay per Step ({test_steps} steps)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"本地与卸载时延对比图已保存至: {save_path}")


if __name__ == "__main__":
    data = run_test(model_path=MODEL_PATH, test_steps=TEST_STEPS)
    plot_uav_trajectories(data, save_path="./uav_trajectories.png")
    plot_user_delays(data, save_path="./user_task_delays.png")
    plot_uav_load_balance(data, save_path="./uav_load_balance.png")
    plot_offload_distribution(data, save_path="./offload_distribution.png")
    plot_local_vs_offload_delay(data, save_path="./local_vs_offload_delay.png")
