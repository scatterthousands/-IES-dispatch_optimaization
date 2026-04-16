from ies_rl_env_old import IES_RL_Environment
from PPO_LSTM import PPOLSTMAgent


def train_marl():
    # 1. 实例化环境 (假设 env 代码已在 ies_rl_env_old.py 中)
    # from ies_rl_env import IES_RL_Environment
    env = IES_RL_Environment()

    # 获取状态维度 (假设为 8维：时间、电价、风电、3室温、2来水温)
    state_dim = 8

    # 2. 实例化三个异构智能体
    agent_N1 = PPOLSTMAgent("Agent_N1", state_dim, action_dim=2)  # 控制 CHP, FC
    agent_N15 = PPOLSTMAgent("Agent_N15", state_dim, action_dim=1)  # 控制 EB
    agent_valves = PPOLSTMAgent("Agent_Valves", state_dim, action_dim=3)  # 控制 3个阀门

    num_episodes = 1000

    for ep in range(num_episodes):
        obs = env.reset()

        # === 核心：每个 Episode 开始时，重置每个智能体的 LSTM 记忆 ===
        h_N1 = agent_N1.get_init_hidden()
        h_N15 = agent_N15.get_init_hidden()
        h_valves = agent_valves.get_init_hidden()

        ep_reward = 0

        for t in range(env.T_total):
            # 1. 智能体根据观测值和自己的记忆，选择动作，并更新记忆
            act_N1, lp_N1, val_N1, h_N1 = agent_N1.select_action(obs, h_N1)
            act_N15, lp_N15, val_N15, h_N15 = agent_N15.select_action(obs, h_N15)
            act_v, lp_v, val_v, h_valves = agent_valves.select_action(obs, h_valves)

            # 2. 组合动作，输入环境
            joint_actions = {
                'N1': act_N1,
                'N15': act_N15,
                'Valves': act_v
            }

            next_obs, reward, done, _ = env.step(joint_actions)

            # 3. 将单步数据存入记忆库 (准备进行 BPTT)
            agent_N1.store_transition(obs, act_N1, lp_N1, reward, val_N1)
            agent_N15.store_transition(obs, act_N15, lp_N15, reward, val_N15)
            agent_valves.store_transition(obs, act_v, lp_v, reward, val_v)

            obs = next_obs
            ep_reward += reward

            if done:
                break

        # === 核心：回合结束，三个智能体分别沿着时间序列回传梯度 (BPTT) ===
        agent_N1.update()
        agent_N15.update()
        agent_valves.update()

        if (ep + 1) % 100 == 0:
            print(f"Episode: {ep + 1} | Total Reward: {ep_reward:.2f} | Final T_in(N6): {env.Tin['N6']:.2f} | Final T_in(N11): {env.Tin['N11']:.2f} | Final T_in(N14): {env.Tin['N14']:.2f}")


if __name__ == "__main__":
    train_marl()