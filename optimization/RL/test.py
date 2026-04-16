import numpy as np
import matplotlib.pyplot as plt
from ies_rl_env import IES_14Node_Env


def run_test():
    # 1. 实例化环境
    env = IES_14Node_Env()
    obs = env.reset()
    print(f"环境初始化成功！共有 {env.T_total} 个时间步。")

    # 用于记录数据的列表
    log_Tin_N6 = []
    log_Tin_N14 = []
    log_Ts_N1 = []
    log_Ts_N14 = []
    log_Reward = []

    done = False
    step = 0

    print("\n--- 开始仿真测试 ---")
    while not done:
        # 2. 构造测试动作 (Heuristic Action)
        # 动作空间约束在 [0, 1]

        if step < (7 * 60 / env.dt_min):
            # 0-7点 (谷电): EB 全开，CHP/FC 关，阀门全开(储热)
            act_src = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            act_valves = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        else:
            # 7点以后: EB 关，FC 满发，CHP 辅助，阀门关小(防过热/节流)
            act_src = np.array([0.4, 1.0, 0.0], dtype=np.float32)
            act_valves = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        action = {
            'src': act_src,
            'valves': act_valves
        }

        # 3. 与环境交互
        next_obs, reward, done, info = env.step(action)

        # 4. 记录数据
        log_Tin_N6.append(env.Tin['N6'])
        log_Tin_N14.append(env.Tin['N14'])
        log_Ts_N1.append(env.Ts_node['N1'])
        log_Ts_N14.append(env.Ts_node['N14'])
        log_Reward.append(reward)

        # 打印关键步状态
        if step % 8 == 0 or done:
            hour = step * env.dt_min / 60.0
            print(f"时间: {hour:04.2f}h | "
                  f"源温N1: {env.Ts_node['N1']:5.1f}°C | "
                  f"荷温N14: {env.Ts_node['N14']:5.1f}°C | "
                  f"室温N14: {env.Tin['N14']:5.2f}°C | "
                  f"奖励: {reward:6.2f}")

        step += 1

    print(f"\n仿真结束！总步数: {step} | 总奖励: {sum(log_Reward):.2f}")

    # ==========================================
    # 5. 绘制验证图表
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    time_axis = np.arange(step) * env.dt_min / 60.0  # 转换为小时

    # 图 1: 管网温度动态 (验证 FDM 和延时)
    ax1 = axes[0]
    ax1.plot(time_axis, log_Ts_N1, 'r-', lw=2.5, label='Source Supply Temp (N1)')
    ax1.plot(time_axis, log_Ts_N14, 'orange', ls='--', lw=2.5, label='Terminal Supply Temp (N14)')
    ax1.set_ylabel('Water Temp (°C)')
    ax1.set_title('Network Thermal Propagation (Source vs Terminal)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 图 2: 室内温度 (验证 RC 模型和换热)
    ax2 = axes[1]
    ax2.plot(time_axis, log_Tin_N6, 'g-', lw=2, label='Indoor Temp N6 (Near)')
    ax2.plot(time_axis, log_Tin_N14, 'b-', lw=2, label='Indoor Temp N14 (Far)')
    ax2.axhline(20.0, color='gray', ls='--', label='Min Comfort (20°C)')
    ax2.axhline(24.0, color='gray', ls='--', label='Max Comfort (24°C)')
    ax2.set_ylabel('Indoor Temp (°C)')
    ax2.set_title('Building Thermal Dynamics (Response to Valve & Supply)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # 图 3: 环境反馈奖励 (验证 Reward 函数)
    ax3 = axes[2]
    ax3.plot(time_axis, log_Reward, 'purple', lw=2, label='Step Reward')
    ax3.set_ylabel('Reward')
    ax3.set_xlabel('Time (Hour)')
    ax3.set_title('Agent Reward Signal over Time')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_test()