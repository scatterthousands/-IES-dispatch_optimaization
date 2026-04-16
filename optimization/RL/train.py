import numpy as np
import matplotlib.pyplot as plt
from ies_rl_env import IES_14Node_Env
from PPO_LSTM import PPO_LSTM_Agent


def train_marl():
    print("=== 初始化综合能源环境 (IES 14-Node) ===")
    env = IES_14Node_Env()

    # 状态维度：时间(1) + 电价(1) + 风电(1) + 建筑温度(3) + 阀门前水温(3) = 9维
    state_dim = 9

    # 智能体 1：源侧控制 (CHP, FC, EB)，动作维度 = 3
    agent_source = PPO_LSTM_Agent(agent_id="Source", state_dim=state_dim, action_dim=3, lr=3e-4)

    # 智能体 2：荷侧控制 (N6, N11, N14 的阀门)，动作维度 = 3
    agent_valve = PPO_LSTM_Agent(agent_id="Valves", state_dim=state_dim, action_dim=3, lr=3e-4)

    MAX_EPISODES = 30000

    # 用于记录数据的列表
    history_rewards = []
    history_tin_N14 = []
    history_tin_N11 = []
    history_tin_N6  = []

    print("\n=== 开始 PPO-LSTM 多智能体训练 ===")
    for ep in range(MAX_EPISODES):
        obs = env.reset()

        # 每个 Episode 必须重置 LSTM 隐状态 (清空昨天记忆)
        h_src = agent_source.get_init_hidden()
        h_val = agent_valve.get_init_hidden()

        ep_reward = 0

        # --- 仿真一天 (96 个 15分钟步长) ---
        for t in range(env.T_total):
            # 1. 智能体根据观测值和 LSTM 隐状态选择动作
            act_src, logp_src, val_src, h_src = agent_source.select_action(obs, h_src)
            act_val, logp_val, val_val, h_val = agent_valve.select_action(obs, h_val)

            # 2. 动作映射 (神经网络输出[-1, 1]，环境需要 [0, 1] 和 [0.1, 1])
            env_act_src = (act_src + 1.0) / 2.0  # 映射到[0, 1]
            env_act_val = (act_val + 1.0) / 2.0 * 0.9 + 0.1  # 映射到 [0.1, 1.0]，防止死水

            actions = {
                'src': env_act_src,
                'valves': env_act_val
            }

            # 3. 与环境交互
            next_obs, reward, done, info = env.step(actions)

            # 4. 存储记忆
            agent_source.store_transition(obs, act_src, logp_src, reward, val_src)
            agent_valve.store_transition(obs, act_val, logp_val, reward, val_val)

            obs = next_obs
            ep_reward += reward

            if done:
                break

        # 5. 回合结束：执行 BPTT (Backpropagation Through Time) 梯度更新
        agent_source.update()
        agent_valve.update()

        # 记录与打印
        history_rewards.append(ep_reward)
        final_tin = env.Tin['N14']  # 取 N14 作为代表建筑
        history_tin_N14.append(final_tin)
        final_tin_N11 = env.Tin['N11']
        history_tin_N11.append(final_tin_N11)
        final_tin_N6 = env.Tin['N6']
        history_tin_N6.append(final_tin_N6)


        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(history_rewards[-10:])
            avg_tin = np.mean(history_tin_N14[-10:])
            avg_tin_N11 = np.mean(history_tin_N11[-10:])
            avg_tin_N6 = np.mean(history_tin_N6[-10:])
            print(
                f"Episode: {ep + 1:>4d} | Avg Reward (Last 10): {avg_reward:>8.2f} | Final T_in(N14): {avg_tin:>5.2f}°C| Final T_in(N11): {avg_tin_N11:>5.2f}°C| Final T_in(N16): {avg_tin_N6:>5.2f}°C")

    # ==========================================
    # 训练结束，绘制学习曲线
    # ==========================================
    print("\n=== 训练结束，正在生成学习曲线 ===")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # 曲线 1: 奖励收敛曲线
    axes[0].plot(history_rewards, color='purple', alpha=0.3)
    # 计算滑动平均线使其平滑
    window = 20
    if len(history_rewards) >= window:
        moving_avg = np.convolve(history_rewards, np.ones(window) / window, mode='valid')
        axes[0].plot(range(window - 1, len(history_rewards)), moving_avg, color='purple', lw=2,
                     label=f'Moving Avg ({window} eps)')
    axes[0].set_title('MARL Training Reward Convergence (PPO-LSTM)', fontweight='bold')
    axes[0].set_ylabel('Episode Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 曲线 2: 建筑温度达标情况
    axes[1].plot(history_tin_N14, color='teal', alpha=0.5, label='Final T_in (N14)')
    axes[1].axhline(20.0, color='gray', linestyle='--', label='Min Comfort Bound (20°C)')
    axes[1].set_title('Building End-of-Day Temperature Evolution', fontweight='bold')
    axes[1].set_xlabel('Episodes')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_marl()