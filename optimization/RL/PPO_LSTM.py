import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


# ==========================================
# 1. 包含 LSTM 记忆层的 Actor-Critic 神经网络
# ==========================================
class ActorCriticLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCriticLSTM, self).__init__()

        # 共享特征提取层
        self.fc1 = nn.Linear(state_dim, hidden_size)

        # LSTM 层 (核心！用于跨越长时滞)
        # batch_first=True 意味着输入维度为 (batch_size, sequence_length, feature_dim)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Actor 分支 (策略网络) - 输出动作的均值
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        # 动作的标准差 (独立的可训练参数，用于探索)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic 分支 (价值网络) - 输出状态价值 V
        self.critic_value = nn.Linear(hidden_size, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state_seq, hidden_state):
        # state_seq: (batch_size, seq_len, state_dim)
        x = self.relu(self.fc1(state_seq))

        # 经过 LSTM，获得时序融合特征和新的隐藏状态
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)

        # 限制动作均值在 [-1, 1]
        action_mean = self.tanh(self.actor_mean(lstm_out))
        # 扩展标准差的维度以匹配序列长度
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)

        state_value = self.critic_value(lstm_out)

        return action_mean, action_std, state_value, new_hidden_state


# ==========================================
# 2. PPO-LSTM 智能体类
# ==========================================
class PPO_LSTM_Agent:
    def __init__(self, agent_id, state_dim, action_dim, hidden_size=64, lr=3e-4):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticLSTM(state_dim, action_dim, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # PPO 核心超参数
        self.gamma = 0.99  # 折扣因子
        self.gae_lambda = 0.95  # GAE 平滑参数
        self.clip_eps = 0.2  # 截断范围
        self.K_epochs = 4  # 每批数据更新次数

        self.memory = []  # 存放一条轨迹 (Trajectory)

    def get_init_hidden(self, batch_size=1):
        """获取 LSTM 初始状态 (h0, c0)，每个 Episode 开始时调用"""
        return (torch.zeros(1, batch_size, self.hidden_size).to(self.device),
                torch.zeros(1, batch_size, self.hidden_size).to(self.device))

    def select_action(self, state, hidden_state):
        """与环境交互时调用 (单步推理)"""
        self.network.eval()
        with torch.no_grad():
            # 增加 batch_size 和 seq_len 维度: (1, 1, state_dim)
            state_ts = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)

            action_mean, action_std, value, new_hidden = self.network(state_ts, hidden_state)

            # 构造高斯分布并采样动作
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(dim=-1)

        # 转回 numpy，并保证在 [-1, 1] 范围内
        action_np = torch.clamp(action, -1.0, 1.0).squeeze().cpu().numpy()
        if self.action_dim == 1: action_np = np.array([action_np])

        return action_np, action_logprob.item(), value.item(), new_hidden

    def store_transition(self, state, action, logprob, reward, value):
        self.memory.append((state, action, logprob, reward, value))

    def update(self):
        """回合结束后调用，沿时间轴进行 BPTT 梯度更新"""
        if len(self.memory) == 0: return

        self.network.train()

        # 1. 整理整个 Episode 的序列数据，增加 Batch 维度 -> (1, seq_len, dim)
        states = torch.FloatTensor(np.array([m[0] for m in self.memory])).unsqueeze(0).to(self.device)
        actions = torch.FloatTensor(np.array([m[1] for m in self.memory])).unsqueeze(0).to(self.device)
        old_logprobs = torch.FloatTensor(np.array([m[2] for m in self.memory])).unsqueeze(0).to(self.device)
        rewards = [m[3] for m in self.memory]
        values = [m[4] for m in self.memory]

        # 2. 计算 GAE (广义优势估计) 和 目标价值 (Returns)
        returns = []
        advantages = []
        gae = 0
        # 最后一个状态的 next_value 设为 0 (回合结束)
        next_value = 0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            next_value = values[i]

        returns = torch.FloatTensor(returns).unsqueeze(0).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(0).to(self.device)

        # 优势函数归一化 (大幅稳定训练)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. PPO 网络更新 (K_epochs 轮)
        for _ in range(self.K_epochs):
            # LSTM 必须从零开始读取整个序列，以保持时间因果关系
            init_hidden = self.get_init_hidden()

            action_mean, action_std, state_values, _ = self.network(states, init_hidden)

            dist = Normal(action_mean, action_std)
            new_logprobs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            # PPO 概率比率
            ratios = torch.exp(new_logprobs - old_logprobs)

            # 截断损失 (Surrogate Loss)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值损失 (MSE Loss)
            critic_loss = nn.MSELoss()(state_values.squeeze(-1), returns.squeeze(-1))

            # 总损失 = 策略损失 + 0.5 * 价值损失 - 0.01 * 熵 (鼓励探索)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            # 梯度下降
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止 LSTM 梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

        # 清空记忆库
        self.memory.clear()