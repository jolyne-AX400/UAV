import numpy as np
import torch

class RolloutBuffer:
    """
    RolloutBuffer 用于存储强化学习中一次或多次环境交互的数据（即“经验”），
    通常用于基于策略梯度（如PPO）算法的训练过程。

    每个时间步都会记录：
    - 智能体的节点特征（obs）
    - 图结构的邻接矩阵（adjs）
    - 动作（actions）
    - 动作的log概率（log_probs）
    - 奖励（rewards）
    - 价值函数估计（values）
    - 终止标志（dones）

    该类还负责根据GAE（Generalized Advantage Estimation）
    计算优势函数（advantages）和回报（returns）。
    """

    def __init__(self, num_agents, node_feat_dim, device='cpu', gamma=0.99, lam=0.95):
        """
        初始化RolloutBuffer。

        参数：
        - num_agents: 智能体数量（即图中节点数N）
        - node_feat_dim: 每个节点的特征维度
        - device: 计算设备（cpu或cuda）
        - gamma: 折扣因子（reward discount factor）
        - lam: GAE平滑系数（用于控制优势估计的偏差与方差）
        """
        self.num_agents = num_agents
        self.node_feat_dim = node_feat_dim
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clear()  # 初始化时清空缓存

    def clear(self):
        """
        清空缓冲区中的所有已存数据。
        通常在每个训练epoch开始时调用。
        """
        self.obs = []        # 存储每步的节点特征张量 [N, feat]
        self.adjs = []       # 存储每步的邻接矩阵 [N, N]
        self.positions = []  # 存储每步的节点位置 [N,2]
        self.actions = []    # 每步的动作
        self.log_probs = []  # 每步动作对应的log概率
        self.rewards = []    # 每步的奖励（可以是每个智能体的reward）
        self.values = []     # 每步的价值估计
        self.dones = []      # 每步是否终止标志（1表示结束，0表示未结束）

    def add(self, node_feats, adj, actions, log_probs, rewards, values, dones, positions=None):
        """
        添加一次时间步的数据到缓存中。

        参数：
        - node_feats: 当前时间步的节点特征 [N, feat]
        - adj: 当前图的邻接矩阵 [N, N]
        - actions: 智能体执行的动作 [N, act_dim]
        - log_probs: 动作的log概率 [N]
        - rewards: 当前时间步的奖励，可以是标量或每个agent的list
        - values: 当前状态的价值估计 [N]
        - dones: episode是否结束（bool或0/1）
        - positions: (可选) 节点的连续位置张量 [N,2]，用于重算物理掩码
        """
        # detach()防止梯度回传到策略网络
        self.obs.append(node_feats.detach().cpu())
        self.adjs.append(adj.detach().cpu())
        # positions 优先使用调用方传入的 positions 参数；若未提供则填零占位
        if positions is not None:
            self.positions.append(positions.detach().cpu())
        else:
            self.positions.append(torch.zeros((node_feats.size(0), 2), dtype=torch.float32))
        self.actions.append(actions.detach().cpu())
        self.log_probs.append(log_probs.detach().cpu())
        self.rewards.append(torch.tensor(rewards, dtype=torch.float32))
        self.values.append(values.detach().cpu())
        self.dones.append(torch.tensor(dones, dtype=torch.float32))

    def compute_returns_and_advantages(self, last_value):
        """
        使用GAE（Generalized Advantage Estimation）算法计算优势函数（advantages）
        和回报（returns = advantage + value）。

        参数：
        - last_value: 最后一个状态的价值估计，用于bootstrap（避免最后一步信息丢失）
        """
        T = len(self.rewards)  # 时间步数
        self.returns = [None] * T
        self.advantages = [None] * T
        next_value = last_value.detach().cpu()
        gae = 0.0  # 初始化GAE累计值

        # 从后往前计算（逆序时间步）
        for t in reversed(range(T)):
            # 奖励和价值都可能是tensor或数值类型，确保统一
            rewards_t = self.rewards[t].sum() if isinstance(self.rewards[t], torch.Tensor) else torch.tensor(self.rewards[t]).sum()
            value_t = self.values[t].sum() if isinstance(self.values[t], torch.Tensor) else torch.tensor(self.values[t]).sum()

            mask = 1.0 - self.dones[t]  # 如果done=1，则mask=0，防止越界传播
            # δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
            delta = rewards_t + self.gamma * next_value * mask - value_t
            # 计算GAE累计值
            gae = delta + self.gamma * self.lam * mask * gae
            # 存储当前时间步的优势值和回报
            self.advantages[t] = gae
            self.returns[t] = gae + value_t
            # 更新下一步的value为当前value
            next_value = value_t

        # 将列表转成tensor以便批处理（如输入PPO更新）
        self.advantages = torch.tensor(self.advantages, dtype=torch.float32)
        self.returns = torch.tensor(self.returns, dtype=torch.float32)

    def get_batches(self):
        """
        将存储的数据整合成可直接输入神经网络训练的批数据。
        这里返回“整条序列”的 batch（未做随机 mini-batch 切分；mini-batch 在 train.py 中完成）。

        返回：
        - obs_tensor:   [T, N, feat]
        - adjs_tensor:  [T, N, N]
        - actions_tensor: [T, N, act_dim]
        - logp_tensor:  [T, N]
        - values_tensor:[T]        # 注意：本实现使用“联合价值（标量V）”，因此是 [T]
        - returns:      [T]
        - advantages:   [T]
        """
        obs_tensor = torch.stack(self.obs)         # [T, N, feat]
        positions_tensor = torch.stack(self.positions) # [T, N, 2]
        adjs_tensor = torch.stack(self.adjs)       # [T, N, N]
        actions_tensor = torch.stack(self.actions) # [T, N, act_dim]
        logp_tensor = torch.stack(self.log_probs)  # [T, N]

        # 你在采样时把 value 以标量或 [1] 形式存入，这里统一压到 [T]
        values_tensor = torch.stack(self.values).view(-1)  # [T]

        returns = self.returns       # [T]
        advantages = self.advantages # [T]
        return obs_tensor, positions_tensor, adjs_tensor, actions_tensor, logp_tensor, values_tensor, returns, advantages

