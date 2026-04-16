import ies_rl_env_old
import numpy as np

if __name__ == "__main__":
    env = ies_rl_env_old.IES_RL_Environment()
    obs = env.reset()

    total_reward = 0
    for t in range(env.T_total):
        # 模拟强化学习算法输出动作
        dummy_action = {
            'N1': np.random.uniform(-1, 1, 2),
            'N15': np.random.uniform(-1, 1, 1),
            'Valves': np.random.uniform(0, 1, 3)  # 阀门开度
        }

        next_obs, reward, done, _ = env.step(dummy_action)
        total_reward += reward

        if t % 20 == 0:
            print(f"Step {t:02d} | T_in(N14): {env.Tin['N14']:.2f}°C | Reward: {reward:.2f}")

    print(f"Episode Done. Total Reward: {total_reward:.2f}")