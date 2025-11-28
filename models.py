import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGATLayer(nn.Module):
    """
    简单图注意力层（Graph Attention Layer）
    
    核心思想：
        让每个节点 i 按"注意力权重" αᵢⱼ 聚合邻居 j 的信息，得到带邻居上下文的新表示 hᵢ。
        注意力机制允许模型自动学习哪些邻居更重要，而不是简单地平均所有邻居的特征。
    
    输入：
        x: [N, d_in]   —— N 个节点（无人机），每行是该节点的特征
        adj: [N, N]    —— 邻接矩阵（0/1 或权重都行）；若传 None 表示完全连通
                          adj[i,j] = 1 表示节点 i 和 j 之间存在连接（邻居关系）

    输出：
        h: [N, d_out]  —— 每个节点的新嵌入（融合了邻居信息后的表示）
        
    工作流程：
        1. 线性变换：将每个节点的特征从 d_in 维映射到 d_out 维
        2. 注意力计算：对每对节点 (i,j) 计算注意力得分 eᵢⱼ
        3. 掩码过滤：只保留邻居节点的注意力，非邻居设为负无穷
        4. Softmax归一化：将注意力得分转化为概率分布（和为1）
        5. 加权聚合：用注意力权重对邻居特征进行加权求和
    """
    def __init__(self, in_dim, out_dim, leaky_relu_neg_slope=0.2):
        """
        参数：
            in_dim: 输入特征维度（每个节点的原始特征数）
            out_dim: 输出特征维度（经过GAT层后的特征数）
            leaky_relu_neg_slope: LeakyReLU的负斜率参数（默认0.2）
        """
        super().__init__()
        
        # ========== 第一部分：特征变换层 ==========
        # 线性变换参数 W，用于将节点特征从 in_dim → out_dim
        # 这是一个共享的变换矩阵，所有节点使用相同的权重
        self.W = nn.Linear(in_dim, out_dim, bias=False)

        # ========== 第二部分：注意力机制参数 ==========
        # 注意力机制参数 a，用于计算节点对之间的注意力系数
        # 计算公式：e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        # 其中 || 表示拼接操作，将节点i和节点j的变换后特征拼接在一起
        # 输入维度是 out_dim * 2（因为拼接了两个节点的特征）
        # 输出维度是 1（得到一个标量注意力得分）
        self.a = nn.Linear(out_dim * 2, 1, bias=False)

        # ========== 第三部分：激活函数 ==========
        # LeakyReLU 激活函数，用于注意力得分的非线性变换
        # 相比ReLU，LeakyReLU在负数区域保留小的梯度，防止神经元"死亡"
        self.leaky_relu = nn.LeakyReLU(leaky_relu_neg_slope)

    def forward(self, x, adj):
        """
        前向传播过程
        
        参数：
            x: [N, d_in] - N个节点的输入特征矩阵
            adj: [N, N] - 邻接矩阵，表示节点间的连接关系
            
        返回：
            h: [N, d_out] - 聚合邻居信息后的节点特征
        """
        # ========== 步骤1：线性变换 ==========
        # 对每个节点的特征进行线性变换：x → Wx
        # 输入 x: [N, d_in]，输出 Wh: [N, out_dim]
        Wh = self.W(x)
        N = Wh.size(0)  # 获取节点数量

        # ========== 步骤2：构造节点对特征 ==========
        # 为了计算每对节点(i, j)之间的注意力，需要将它们的特征拼接起来
        # 这里使用了广播机制来高效地构造所有节点对的特征拼接
        
        # Wh_i: 将每个节点的特征复制N次（按列复制）
        # unsqueeze(1)将形状从[N, out_dim]变为[N, 1, out_dim]
        # expand(-1, N, -1)将形状扩展为[N, N, out_dim]
        # 结果：Wh_i[i, j, :] = Wh[i, :]（第i个节点的特征，重复N次）
        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1)  # [N, N, out_dim]
        
        # Wh_j: 将每个节点的特征复制N次（按行复制）
        # unsqueeze(0)将形状从[N, out_dim]变为[1, N, out_dim]
        # expand(N, -1, -1)将形状扩展为[N, N, out_dim]
        # 结果：Wh_j[i, j, :] = Wh[j, :]（第j个节点的特征，重复N次）
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1)  # [N, N, out_dim]
        
        # 在最后一个维度上拼接i和j的特征
        # cat: [N, N, 2*out_dim]，其中cat[i, j, :]包含了节点i和节点j的完整特征信息
        cat = torch.cat([Wh_i, Wh_j], dim=-1)

        # ========== 步骤3：计算原始注意力得分 ==========
        # 通过小型MLP（self.a）计算每对节点的注意力得分
        # self.a(cat): [N, N, 1] → squeeze后得到 [N, N]
        # e[i, j]表示节点j对节点i的重要性原始分数（越大表示越重要）
        e = self.leaky_relu(self.a(cat).squeeze(-1))  # [N, N]

        # ========== 步骤4：应用邻接掩码 ==========
        # 只有真正的邻居节点才应该参与注意力计算
        # 非邻居节点的注意力得分需要被屏蔽掉
        
        if adj is None:
            # 如果没有提供邻接矩阵，假设是完全连通图（所有节点都互相连接）
            mask = torch.ones((N, N), dtype=torch.bool, device=Wh.device)
        else:
            # 将邻接矩阵转换为布尔掩码（>0的位置为True）
            mask = (adj > 0)
            
            # 添加自环：允许每个节点也"关注"自己的特征
            # 创建单位矩阵作为自环掩码
            eye = torch.eye(N, dtype=torch.bool, device=Wh.device)
            # 使用逻辑或操作将自环添加到掩码中
            mask = mask | eye

        # 对非邻居节点的注意力得分赋值为负无穷
        # 这样在后续的softmax操作中，这些位置的权重会接近0
        NEG_INF = -9e15  # 一个很大的负数，模拟负无穷
        e = torch.where(mask, e, torch.full_like(e, NEG_INF))

        # ========== 步骤5：Softmax归一化 ==========
        # 对每个节点i的所有邻居j进行softmax归一化
        # dim=1表示对每一行进行归一化（即对节点i的所有邻居进行归一化）
        # alpha[i, j]表示节点j对节点i的归一化注意力权重
        # 每一行的和为1：Σⱼ alpha[i, j] = 1
        alpha = torch.softmax(e, dim=1)  # [N, N]

        # ========== 步骤6：加权聚合邻居特征 ==========
        # 使用注意力权重对邻居的变换后特征进行加权求和
        # matmul(alpha, Wh): [N, N] × [N, out_dim] = [N, out_dim]
        # h[i, :] = Σⱼ alpha[i, j] * Wh[j, :]
        # 这样每个节点的新表示就融合了其邻居的信息，且重要的邻居权重更大
        h = torch.matmul(alpha, Wh)  # [N, out_dim]
        
        return h


class ActorGAT(nn.Module):
    """
    基于图注意力网络的Actor（策略网络）
    
    功能说明：
        将每个节点的自身特征与邻居信息融合，输出动作分布的参数（均值和标准差），
        形成连续动作空间的高斯策略。这是强化学习中Actor-Critic架构的Actor部分。
    
    应用场景：
        多无人机协同控制系统，其中：
        - 每个无人机是一个节点
        - 无人机之间的通信关系由邻接矩阵描述
        - 每个无人机需要输出连续动作（如速度、方向等）
    
    网络架构：
        输入层 → GAT层1 → GAT层2 → MLP → 输出（均值 + 标准差）
        
    输出说明：
        - 均值(mean): 动作的期望值
        - 标准差(std): 动作的不确定性（探索程度）
        - 两者共同定义一个高斯分布，从中采样得到实际执行的动作
    """
    def __init__(self, node_in_dim=8, gat_hidden=64, action_dim=3):
        """
        参数：
            node_in_dim: 每个节点的输入特征维度（默认8）
                        例如：[位置x, 位置y, 速度vx, 速度vy, 能量, ...]
            gat_hidden: GAT层的隐藏层维度（默认64）
            action_dim: 动作空间的维度（默认3）
                       例如：[dx, dy, 目标距离] 或 [vx, vy, vz]
        """
        super().__init__()
        
        # ========== 第一部分：图注意力层 ==========
        # 使用两层GAT来逐步聚合和提炼邻居信息
        
        # 第1层 GAT：将输入特征映射到隐藏空间
        # 作用：初步融合每个节点与其一阶邻居的信息
        self.gat1 = SimpleGATLayer(node_in_dim, gat_hidden)
        
        # 第2层 GAT：在隐藏空间中进一步聚合信息
        # 作用：融合二阶邻居信息（邻居的邻居），扩大感受野
        self.gat2 = SimpleGATLayer(gat_hidden, gat_hidden)

        # ========== 第二部分：动作输出头 ==========
        # 每个节点独立的MLP，将图特征转化为动作均值
        # 注意：这里是"去中心化"的设计，每个节点独立决策
        self.mlp = nn.Sequential(
            nn.Linear(gat_hidden, 64),    # 隐藏层
            nn.ReLU(),                     # 激活函数
            nn.Linear(64, action_dim)      # 输出层：输出动作均值
        )

        # ========== 第三部分：可学习的标准差参数 ==========
        # log_std: 动作分布的对数标准差（共享参数）
        # 
        # 设计思路：
        #   - 使用对数形式是为了保证 std = exp(log_std) > 0（标准差必须为正）
        #   - 初始化为-1.0意味着初始标准差约为exp(-1)≈0.37（适度的探索）
        #   - 这是一个可学习的参数，会随着训练自动调整探索程度
        #   - 所有节点共享相同的log_std（简化设计，也可以改为每节点独立）
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))


    def forward(self, node_feats, adj):
        """
        前向传播：从节点特征和邻接关系生成动作分布
        
        参数：
            node_feats: [N, node_in_dim] - 每个节点的输入特征
                       例如：N个无人机的状态信息（位置、速度、能量等）
            adj: [N, N] - 邻接矩阵
                       adj[i,j]=1表示无人机i和j之间可以通信

        输出：
            mean: [N, action_dim] - 每个节点的动作均值
                                   这是策略网络认为的"最优动作"
            std:  [N, action_dim] - 每个节点的动作标准差
                                   控制探索的程度（标准差越大，探索越随机）
                                   
        使用方式：
            在训练/执行时，从N(mean, std²)分布中采样得到实际动作：
            action = mean + std * epsilon, 其中 epsilon ~ N(0,1)
        """
        
        # ========== 步骤1：第一层GAT聚合 ==========
        # 每个节点融合其一阶邻居的信息
        x = F.elu(self.gat1(node_feats, adj))  # [N, gat_hidden]
        # 使用ELU激活（相比ReLU更平滑，有助于训练稳定性）
        
        # ========== 步骤2：第二层GAT聚合 ==========
        # 进一步融合二阶邻居信息，获得更广阔的"视野"
        x = F.elu(self.gat2(x, adj))  # [N, gat_hidden]

        # ========== 步骤3：生成动作均值 ==========
        # 通过MLP将融合后的图特征映射到动作空间
        mean = self.mlp(x)  # [N, action_dim]
        # mean[i, :]表示第i个节点（无人机）应该执行的动作期望值

        # ========== 步骤4：生成动作标准差 ==========
        # 从对数标准差计算实际标准差，并扩展到所有节点
        # exp(log_std): [action_dim] → std的实际值（始终为正）
        # unsqueeze(0): [action_dim] → [1, action_dim]（增加批次维度）
        # expand_as(mean): [1, action_dim] → [N, action_dim]（复制到每个节点）
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        # 所有节点共享相同的标准差（简化设计）

        return mean, std


class CriticCentral(nn.Module):
    """
    集中式Critic（价值函数网络）
    
    功能说明：
        输入全局状态特征，输出一个标量 V(s)，评估"当前全局状态的好坏"。
        这个价值估计用于训练Actor，告诉Actor当前状态下期望能获得多少累积奖励。
    
    在多智能体系统中的作用：
        - 训练阶段：Critic可以访问全局信息（所有智能体的状态）
        - 执行阶段：Actor只使用局部信息（符合CTDE: 集中训练、分布执行原则）
        
    CTDE (Centralized Training with Decentralized Execution):
        - Centralized Training: 训练时Critic看到全局信息，提供更准确的价值估计
        - Decentralized Execution: 执行时Actor只用局部观测，无需全局通信
    """
    def __init__(self, global_in_dim, hidden=256):
        """
        参数：
            global_in_dim: 全局特征的维度
                          通常是所有智能体特征拼接后的维度
                          例如：N个智能体 × 每个8维 = global_in_dim
            hidden: 隐藏层的宽度（默认256）
        """
        super().__init__()
        
        # ========== 简单的三层MLP结构 ==========
        # 设计理念：Critic不需要太复杂，主要是拟合价值函数
        self.net = nn.Sequential(
            nn.Linear(global_in_dim, hidden),  # 输入层 → 隐藏层1
            nn.ReLU(),                          # 激活函数
            nn.Linear(hidden, hidden),          # 隐藏层1 → 隐藏层2
            nn.ReLU(),                          # 激活函数
            nn.Linear(hidden, 1)                # 隐藏层2 → 输出（单个价值）
        )

    def forward(self, global_feat):
        """
        前向传播：估计全局状态价值
        
        参数：
            global_feat: [B, global_in_dim] 或 [global_in_dim]
                        批量的全局状态特征或单个全局状态
                        
        输出：
            value: [B] 或标量 - 状态价值估计 V(s)
                   数值越大表示当前状态越"好"（期望获得更多奖励）
                   
        价值函数的意义：
            V(s) = E[Σ γᵗ r_t | s_0 = s]
            即从状态s开始，按照当前策略执行，期望能获得的折扣累积奖励
        """
        return self.net(global_feat).squeeze(-1)
    

# ==================== 新增模块：带全局统计的Critic ====================

def concat_nodes_and_extras(node_feats, extras):
    """
    辅助函数：将所有智能体的特征扁平化，并与全局统计量进行拼接
    
    设计目的：
        为Critic网络构造输入，使其同时看到：
        1. 所有智能体的局部特征（拼成一条长向量）
        2. 环境的全局统计量（如覆盖率、总能量、时间进度等）
    
    优势：
        ✅ Critic在训练时获得更全局的状态视图，提供更准确的价值估计
        ✅ 符合CTDE思想：训练时集中信息，执行时各智能体仍保持分布式
        ✅ 全局统计量提供了"大局观"，帮助评估团队整体表现

    参数：
        node_feats: 节点特征张量
            - 形状1: [N, d] - N个节点，每个节点d维特征（单个时间步）
            - 形状2: [B, N, d] - B个批次，每批N个节点（批量数据）
        extras: 全局统计量张量
            - 形状1: [E] - 一条全局统计（单个时间步）
            - 形状2: [B, E] - 每个批次有各自的全局统计

    返回：
        global_feat: 拼接后的全局特征向量
            - 若输入为 [N, d] 和 [E]，输出 [N*d + E]
            - 若输入为 [B, N, d] 和 [B, E]，输出 [B, N*d + E]
    """

    # ========== 情况1：无批次输入（单个时间步）==========
    if node_feats.dim() == 2:
        N, d = node_feats.shape                  # N个节点，每个d维
        flat = node_feats.reshape(N * d)         # 展平成一维向量 [N*d]

        if extras.dim() == 1:
            # 典型场景：单步数据，单条全局统计
            # 例如：5个无人机×8维特征=40维，再加3维全局统计=43维
            return torch.cat([flat, extras], dim=0)  # [N*d + E]
            
        elif extras.dim() == 2:
            # extras带批次 [B, E]，需要将节点特征也扩展到批次
            B = extras.size(0)
            flatB = flat.unsqueeze(0).expand(B, -1)  # [B, N*d]
            return torch.cat([flatB, extras], dim=1)  # [B, N*d + E]
        else:
            raise ValueError("extras 必须是 [E] 或 [B, E] 的形式")

    # ========== 情况2：带批次输入（多个样本）==========
    elif node_feats.dim() == 3:
        B, N, d = node_feats.shape                # B个样本，每个N个节点
        flat = node_feats.reshape(B, N * d)       # 每个样本展平 [B, N*d]

        if extras.dim() == 1:
            # extras只有一条[E]，需要复制到每个批次
            extras = extras.unsqueeze(0).expand(B, -1)  # [B, E]
            
        elif extras.dim() == 2:
            # extras已经是[B, E]，检查批次数是否匹配
            if extras.size(0) != B:
                raise ValueError(f"extras的批次数{extras.size(0)}必须与node_feats的批次数{B}一致")
        else:
            raise ValueError("extras 必须是 [E] 或 [B, E] 的形式")

        # 拼接得到Critic的全局输入
        return torch.cat([flat, extras], dim=1)  # [B, N*d + E]

    # ========== 输入格式错误 ==========
    else:
        raise ValueError("node_feats 必须是 [N, d] 或 [B, N, d] 形状")



class CriticCentralGlobal(nn.Module):
    """
    增强版集中式Critic：融合局部特征和全局统计
    
    ✳️ 核心创新：
        相比基础版CriticCentral，本版本额外接收环境的全局统计量，
        例如：探索覆盖率、团队总能量、任务完成度、时间进度等。
        这些全局信息帮助Critic更准确地评估当前状态的价值。
    
    ✳️ 适用场景：
        多智能体协同任务，其中：
        - 每个智能体有自己的局部状态（位置、速度、能量等）
        - 存在团队级的全局指标（覆盖率、完成度、剩余时间等）
        - 需要综合评估"个体表现"+"团队表现"
    
    ✳️ CTDE架构中的定位：
        - 训练阶段：Critic看到所有局部特征+全局统计，提供准确的V(s)
        - 执行阶段：Actor仍然只基于局部观测，保持分布式决策
        
    ✳️ 输入构成：
        [智能体1特征, 智能体2特征, ..., 智能体N特征, 全局统计量]
         ←────────── N*d 维 ──────────→  ←── E 维 ──→
    """
    def __init__(self, num_agents: int, node_in_dim: int, extras_dim: int = 3, hidden: int = 256):
        """
        初始化增强版Critic
        
        参数：
            num_agents: 智能体数量 N（例如：5个无人机）
            node_in_dim: 每个智能体的特征维度 d（例如：8维状态）
            extras_dim: 全局统计量的维度 E（默认3）
                       例如：[覆盖率, 能量比例, 时间进度]
            hidden: MLP隐藏层宽度（默认256）
        """
        super().__init__()
        self.num_agents = num_agents
        self.node_in_dim = node_in_dim
        self.extras_dim = extras_dim

        # ========== 计算Critic的总输入维度 ==========
        # 总维度 = 所有智能体的特征维度 + 全局统计维度
        # 例如：5个智能体×8维特征 + 3维全局统计 = 43维
        global_in_dim = num_agents * node_in_dim + extras_dim

        # ========== 构建价值网络（三层MLP）==========
        self.net = nn.Sequential(
            nn.Linear(global_in_dim, hidden),  # 输入层：全局特征→隐藏层
            nn.ReLU(),
            nn.Linear(hidden, hidden),         # 隐藏层：进一步提取特征
            nn.ReLU(),
            nn.Linear(hidden, 1)               # 输出层：输出单个价值V(s)
        )

    def forward(self, node_feats, extras):
        """
        前向传播：估计包含全局统计的状态价值
        
        输入：
            node_feats: 所有智能体的特征
                - [N, D] - 单个时间步，N个智能体
                - [B, N, D] - B个批次，每批N个智能体
            extras: 全局统计量
                - [E] - 单条全局统计
                - [B, E] - 每个批次的全局统计
                
        输出：
            value: 状态价值估计
                - 标量或[B] - 对应输入的批次数
                
        工作流程：
            1. 检查输入维度的合法性
            2. 将node_feats展平并与extras拼接
            3. 通过MLP计算价值V(s)
        """

        # ========== 输入验证：检查node_feats形状 ==========
        if node_feats.dim() == 2:
            # 单个时间步的情况
            N, D = node_feats.shape
            assert N == self.num_agents and D == self.node_in_dim, \
                f"node_feats 形状应为 [N={self.num_agents}, D={self.node_in_dim}]，" \
                f"实际为 {node_feats.shape}"

        elif node_feats.dim() == 3:
            # 批量数据的情况
            _, N, D = node_feats.shape
            assert N == self.num_agents and D == self.node_in_dim, \
                f"node_feats 形状应为 [B, N={self.num_agents}, D={self.node_in_dim}]，" \
                f"实际为 {node_feats.shape}"

        else:
            raise ValueError("node_feats 必须是 [N, D] 或 [B, N, D]")

        # ========== 输入验证：检查extras形状 ==========
        if extras.dim() not in (1, 2) or extras.size(-1) != self.extras_dim:
            raise ValueError(
                f"extras 最后一维必须是 extras_dim={self.extras_dim}，"
                f"实际为 {extras.shape}"
            )

        # ========== 特征拼接 ==========
        # 调用辅助函数：把节点特征展平并拼接全局统计
        # 输出形状：[N*D + E] 或 [B, N*D + E]
        global_feat = concat_nodes_and_extras(node_feats, extras)

        # ========== 价值估计 ==========
        # 通过MLP输出价值，squeeze(-1)去掉最后的单维度
        value = self.net(global_feat).squeeze(-1)  # [B] 或 标量
        return value


class AdjPredictor(nn.Module):
    """
    可学习的邻接矩阵预测器（动态图生成器）
    
    核心功能：
        根据节点的当前特征，动态预测哪些节点之间应该建立连接。
        这允许通信拓扑根据任务需求自适应调整，而不是使用固定的邻接矩阵。
    
    应用场景：
        - 多无人机系统：根据位置和任务动态建立通信链路
        - 社交网络：预测用户之间的潜在关系
        - 分子图：预测原子间的化学键
    
    工作原理：
        1. 将每个节点的特征编码到隐藏空间
        2. 对每对节点(i,j)，拼接它们的编码并打分
        3. 通过sigmoid将分数转化为概率（0-1之间）
        4. 可选：只保留top-k个最强的连接（稀疏化）
    
    输入：
        node_feats: [N, d] - N个节点的特征
        
    输出：
        adj_probs: [N, N] - 邻接概率矩阵（0-1实值）
        logits: [N, N] - 未归一化的分数（用于计算损失）
    
    注意：
        当前实现的复杂度为O(N²)，适合中小规模图。
        对于大规模图（N>1000），建议使用top-k采样或低秩近似。
    """
    def __init__(self, node_in_dim: int, hidden: int = 64):
        """
        参数：
            node_in_dim: 节点输入特征维度
            hidden: 节点编码的隐藏层维度（默认64）
        """
        super().__init__()
        
        # ========== 节点编码器 ==========
        # 将节点特征映射到隐藏空间，提取关键信息
        self.node_enc = nn.Linear(node_in_dim, hidden)
        
        # ========== 节点对打分器 ==========
        # 对拼接后的节点对特征进行打分，预测连接概率
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),  # 输入：两个节点的编码拼接
            nn.ReLU(),
            nn.Linear(hidden, 1)            # 输出：连接分数（标量）
        )

    def forward(self, node_feats, top_k: int = None):
        """
        前向传播：预测邻接矩阵
        
        参数：
            node_feats: [N, node_in_dim] - 节点特征
            top_k: 可选参数，如果指定，每个节点只保留top_k个最强连接
                   这可以强制生成稀疏图，减少通信开销
        
        返回：
            probs: [N, N] - 连接概率矩阵
                   probs[i,j] ∈ [0,1] 表示节点i到j的连接概率
            logits: [N, N] - 原始分数（未经sigmoid）
                    用于计算二元交叉熵损失
        
        如果提供top_k：
            使用straight-through estimator技巧：
            - 前向传播时使用硬掩码（只保留top-k）
            - 反向传播时梯度仍通过原始概率传递
            这样既保证了稀疏性，又保持了梯度流动
        """
        N = node_feats.size(0)  # 节点数量
        
        # ========== 步骤1：编码所有节点 ==========
        h = F.relu(self.node_enc(node_feats))  # [N, hidden]

        # ========== 步骤2：构造所有节点对的特征 ==========
        # 使用广播机制高效构造
        hi = h.unsqueeze(1).expand(-1, N, -1)  # [N, N, hidden] - 行复制
        hj = h.unsqueeze(0).expand(N, -1, -1)  # [N, N, hidden] - 列复制
        cat = torch.cat([hi, hj], dim=-1)      # [N, N, 2*hidden] - 拼接

        # ========== 步骤3：计算连接分数 ==========
        logits = self.pair_mlp(cat).squeeze(-1)  # [N, N]

        # ========== 步骤4：消除自环 ==========
        # 通常不希望节点连接自己（自环），将对角线设为负无穷
        idx = torch.arange(N, device=logits.device)
        logits[idx, idx] = -9e15

        # ========== 步骤5：转化为概率 ==========
        probs = torch.sigmoid(logits)  # [N, N]，值域[0, 1]

        # ========== 可选步骤：Top-K稀疏化 ==========
        if top_k is not None and top_k > 0:
            # 对每一行（每个节点）选择概率最大的top_k个连接
            # values: [N, top_k] - top_k个最大概率值
            # indices: [N, top_k] - 对应的列索引
            values, indices = torch.topk(probs, k=min(top_k, N-1), dim=1)
            
            # 构造硬掩码：只有top_k位置为1，其余为0
            hard_mask = torch.zeros_like(probs)
            hard_mask.scatter_(1, indices, 1.0)  # 将top_k位置设为1

            # Straight-Through Estimator (STE)技巧：
            # - 前向：使用离散的hard_mask
            # - 反向：梯度通过连续的probs
            # 实现方式：probs_st = hard_mask + (probs - probs).detach()
            #                     = hard_mask + 0 (前向)
            #                     梯度时：∂probs_st/∂θ = ∂probs/∂θ
            probs_st = hard_mask.detach() - probs.detach() + probs
            return probs_st, logits

        # 不使用top-k时，直接返回连续概率
        return probs, logits


class InfoController(nn.Module):
    """
    信息流控制器（AC2C风格的门控机制）
    
    核心思想：
        为每个节点生成一个"门控值"（gate），控制该节点是否/以多大程度
        参与信息交换。这可以用于：
        - 自适应通信：节点根据自身状态决定是否发送信息
        - 带宽管理：限制总通信量，节省网络资源
        - 隐私保护：敏感节点可以降低信息共享程度
    
    AC2C (Actor-Critic with Communication Control):
        一种多智能体强化学习框架，其中智能体不仅学习动作策略，
        还学习何时、如何与其他智能体通信。
    
    输入：
        node_feats: [N, d] - 每个节点的特征
        
    输出：
        gates: [N] - 每个节点的门控值，范围[0, 1]
               0表示完全阻断信息，1表示完全共享
    """
    def __init__(self, node_in_dim: int, hidden: int = 32):
        """
        参数：
            node_in_dim: 节点输入特征维度
            hidden: 隐藏层维度（默认32，较小因为任务简单）
        """
        super().__init__()
        
        # ========== 简单的两层MLP + Sigmoid ==========
        # 输出需要在[0,1]之间，因此最后使用Sigmoid激活
        self.net = nn.Sequential(
            nn.Linear(node_in_dim, hidden),  # 输入层
            nn.ReLU(),                        # 激活
            nn.Linear(hidden, 1),             # 输出层（单个标量）
            nn.Sigmoid()                      # 压缩到[0,1]区间
        )

    def forward(self, node_feats):
        """
        前向传播：为每个节点计算门控值
        
        参数：
            node_feats: [N, d] - N个节点的特征
            
        返回：
            gates: [N] - 每个节点的门控值
                   例如：[0.8, 0.3, 0.9, 0.1, 0.6]
                   表示节点0、2、4通信意愿强，节点1、3通信意愿弱
        
        使用示例：
            gates = info_controller(node_feats)
            # 将门控应用到邻接矩阵
            adj_gated = adj * gates.unsqueeze(1)  # 发送端门控
            或
            adj_gated = adj * gates.unsqueeze(0)  # 接收端门控
        """
        # node_feats: [N, d]
        gates = self.net(node_feats).squeeze(-1)  # [N]
        return gates


class UDEAdjDynamics(nn.Module):
    """
    基于通用微分方程（UDE）的邻接矩阵动力学建模器
    
    核心概念：
        UDE (Universal Differential Equations) 是一种结合物理方程和神经网络的方法。
        这里用于建模邻接矩阵的连续时间演化：
        
        dA/dt = f_θ(A, node_feats, t)
        
        其中f_θ是一个神经网络，学习邻接矩阵如何随时间变化。
    
    应用场景：
        - 动态网络：社交关系、通信拓扑随时间演化
        - 物理系统：粒子间相互作用强度的变化
        - 生物网络：神经连接的可塑性
    
    优势：
        - 可以捕捉复杂的时间演化模式
        - 物理可解释性：dA/dt有明确的变化率含义
        - 可以进行长期预测和轨迹优化
    
    技术依赖：
        需要 torchdiffeq 包来求解常微分方程（ODE）
        安装：pip install torchdiffeq
    
    输入：
        A0: [N, N] - 初始邻接矩阵（t=0时刻）
        node_feats: [N, d] - 节点特征（影响演化规律）
        t_span: (t_start, t_end) - 时间区间
        
    输出：
        A_t: [N, N] - 演化后的邻接矩阵（t=t_end时刻）
    """
    def __init__(self, node_feat_dim: int, hidden: int = 64):
        """
        参数：
            node_feat_dim: 节点特征维度
            hidden: 隐藏层维度（默认64）
        """
        super().__init__()
        
        # ========== 节点编码器 ==========
        # 提取节点的关键特征，用于决定邻接矩阵的变化趋势
        self.node_enc = nn.Linear(node_feat_dim, hidden)
        
        # ========== 节点对变化率打分器 ==========
        # 对每对节点(i,j)，预测它们连接强度的变化率 dA[i,j]/dt
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def _odefunc(self, node_feats, N):
        """
        内部方法：构造ODE的右端函数 f(t, A)
        
        参数：
            node_feats: [N, d] - 节点特征（固定）
            N: 节点数量
            
        返回：
            f: 函数(t, a_flat) -> da_flat/dt
               输入：t (时间), a_flat (展平的邻接矩阵)
               输出：da_flat/dt (邻接矩阵的变化率)
        """
        # 预先计算节点编码（在整个ODE求解过程中保持不变）
        h = F.relu(self.node_enc(node_feats))  # [N, hidden]

        def f(t, a_flat):
            """
            ODE右端函数：dA/dt = f(t, A)
            
            参数：
                t: 当前时间（标量）
                a_flat: 当前邻接矩阵（展平成一维向量）
                
            返回：
                dA/dt: 邻接矩阵的变化率（展平）
            """
            # ========== 步骤1：恢复矩阵形状 ==========
            A = a_flat.view(N, N)  # [N*N] → [N, N]
            
            # ========== 步骤2：计算节点对的变化率 ==========
            # 构造所有节点对的特征拼接
            hi = h.unsqueeze(1).expand(-1, N, -1)  # [N, N, hidden]
            hj = h.unsqueeze(0).expand(N, -1, -1)  # [N, N, hidden]
            cat = torch.cat([hi, hj], dim=-1)      # [N, N, 2*hidden]
            
            # 通过MLP预测变化率
            dA = self.pair_mlp(cat).squeeze(-1)    # [N, N]
            
            # ========== 步骤3：稳定性处理 ==========
            # 移除自环的变化（对角线保持不变）
            idx = torch.arange(N, device=dA.device)
            dA[idx, idx] = 0.0
            
            # 使用tanh限制变化率的幅度，防止数值爆炸
            # tanh将值压缩到(-1, 1)范围内
            dA = torch.tanh(dA)
            
            # ========== 步骤4：展平返回 ==========
            return dA.view(-1)  # [N, N] → [N*N]

        return f

    def integrate(self, A0, node_feats, t_span=(0.0, 1.0), method='rk4'):
        """
        对邻接矩阵进行时间积分，模拟其演化过程
        
        参数：
            A0: [N, N] - 初始邻接矩阵（可以是logits或概率）
            node_feats: [N, d] - 节点特征
            t_span: (t_start, t_end) - 积分时间区间（默认从0到1）
            method: ODE求解方法（默认'rk4'即四阶龙格库塔法）
                    其他选项：'dopri5'（自适应步长），'euler'（欧拉法）
        
        返回：
            A_t_probs: [N, N] - 时刻t_end的邻接概率矩阵（经过sigmoid）
            a_end: [N, N] - 原始logits（未归一化）
        
        工作流程：
            1. 检查torchdiffeq是否已安装
            2. 构造ODE函数 dA/dt = f(A, node_feats)
            3. 从t_start积分到t_end
            4. 将结果通过sigmoid转为概率
        """
        # ========== 检查依赖 ==========
        try:
            from torchdiffeq import odeint
        except Exception as e:
            raise RuntimeError(
                "UDEAdjDynamics 需要安装 'torchdiffeq' 包。\n"
                "请运行：pip install torchdiffeq"
            ) from e

        # ========== 准备ODE求解 ==========
        N = A0.size(0)
        f = self._odefunc(node_feats, N)  # 获取ODE右端函数

        # 注意：不要在这里断开梯度，否则上游生成A0的网络无法收到梯度。
        # 如果需要保护原始张量不被就地修改，可使用clone()，但保留requires_grad以允许反向传播。
        a0 = A0.clone()
        a0_flat = a0.view(-1)  # 展平初始状态

        # ========== 执行ODE积分 ==========
        # 定义时间点：起点和终点
        t = torch.tensor([t_span[0], t_span[1]], device=a0.device, dtype=a0.dtype)
        
        # 调用ODE求解器
        # a_t: [2, N*N] - 两个时间点的状态（起点和终点）
        a_t = odeint(f, a0_flat, t, method=method)
        
        # 提取终点时刻的状态
        a_end = a_t[-1].view(N, N)  # [N, N]

        # ========== 转化为概率 ==========
        # 通过sigmoid将logits转为[0,1]的概率
        return torch.sigmoid(a_end), a_end
