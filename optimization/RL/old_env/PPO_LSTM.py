import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ==========================================
# 1. 带有 LSTM 的 Actor-Critic 神经网络
# ==========================================
class ActorCriticLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCriticLSTM, self).__init__()

        # 共享特征提取层
        self.fc1 = nn.Linear(state_dim, hidden_size)

        # LSTM 记忆层 (核心创新点)
        # batch_first=True 意味着输入维度为 (batch, seq_len, feature)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Actor 分支 (输出动作的均值)
        self.actor_fc = nn.Linear(hidden_size, action_dim)
        # 动作的标准差 (可训练参数，用于探索)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic 分支 (输出状态的价值 V)
        self.critic_fc = nn.Linear(hidden_size, 1)

        self.relu = nn.ReLU()

    def forward(self, x, hidden_state):
        # x shape: (batch, seq_len, state_dim)
        x = self.relu(self.fc1(x))

        # 经过 LSTM，输出新的特征和更新后的记忆 (hx, cx)
        lstm_out, hidden_state = self.lstm(x, hidden_state)

        # 提取各个分支的输出
        action_mean = torch.tanh(self.actor_fc(lstm_out))  # 限制在 [-1, 1] 之间
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)

        state_value = self.critic_fc(lstm_out)

        return action_mean, action_std, state_value, hidden_state


# ==========================================
# 2. PPO-LSTM 智能体类
# ==========================================
class PPOLSTMAgent:
    def __init__(self, agent_id, state_dim, action_dim, hidden_size=64, lr=3e-4):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticLSTM(state_dim, action_dim, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # PPO 超参数
        self.gamma = 0.99
        self.eps_clip = 0.2

        # 记忆库 (用于存放一个 Episode 的数据进行 BPTT 更新)
        self.memory = []

    def get_init_hidden(self, batch_size=1):
        """初始化 LSTM 的隐藏状态 (h0, c0)"""
        return (torch.zeros(1, batch_size, self.hidden_size).to(self.device),
                torch.zeros(1, batch_size, self.hidden_size).to(self.device))

    def select_action(self, state, hidden_state):
        """在环境中执行动作时调用"""
        self.network.eval()
        with torch.no_grad():
            state_ts = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, dim)

            action_mean, action_std, value, new_hidden = self.network(state_ts, hidden_state)

            # 从正态分布中采样动作 (带探索)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(dim=-1)

        # 截断到 [-1, 1] 并转为 numpy
        action_np = torch.clamp(action, -1.0, 1.0).cpu().numpy().flatten()

        return action_np, action_logprob.item(), value.item(), new_hidden

    def store_transition(self, state, action, logprob, reward, value):
        self.memory.append((state, action, logprob, reward, value))

    def update(self):
        """回合结束后进行 PPO 网络更新 (BPTT 沿时间反向传播)"""
        self.network.train()

        # 整理记忆库
        states = torch.FloatTensor(np.array([m[0] for m in self.memory])).unsqueeze(0).to(
            self.device)  # (1, seq_len, dim)
        actions = torch.FloatTensor(np.array([m[1] for m in self.memory])).unsqueeze(0).to(self.device)
        old_logprobs = torch.FloatTensor(np.array([m[2] for m in self.memory])).unsqueeze(0).to(self.device)
        rewards = [m[3] for m in self.memory]
        values = [m[4] for m in self.memory]

        # 计算优势函数 (Advantage) 和 目标价值 (Return)
        returns = []
        discounted_sum = 0
        for reward in reversed(rewards):
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        returns = torch.FloatTensor(returns).unsqueeze(0).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # 归一化加速收敛

        values_ts = torch.FloatTensor(values).unsqueeze(0).to(self.device)
        advantages = returns - values_ts

        # PPO 训练循环 (沿整个轨迹进行 BPTT)
        for _ in range(4):  # PPO 通常对同一批数据更新几轮 (Epochs)
            init_hidden = self.get_init_hidden()

            # 将整个序列喂给 LSTM
            action_mean, action_std, state_values, _ = self.network(states, init_hidden)

            dist = torch.distributions.Normal(action_mean, action_std)
            new_logprobs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            # PPO 截断目标函数
            ratios = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values.squeeze(-1), returns)

            # 损失函数: Actor_Loss + 0.5 * Critic_Loss - 0.01 * Entropy (鼓励探索)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory.clear()