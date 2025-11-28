import torch

def build_adj_matrix(positions, comm_range):
    """
    构建无人机之间的邻接矩阵（Adjacency Matrix）。

    功能说明：
        这个函数根据每个无人机的位置，计算它们之间的欧式距离，
        如果两个无人机之间的距离小于通信半径 comm_range，
        就认为它们之间可以通信（记为 1），否则为 0。

        输出的邻接矩阵 adj 用于图神经网络（例如 GAT），
        表示节点之间的连接关系。
    
    """

    # 如果输入 positions 不是 tensor，则将其转为 tensor
    # 方便后续在 GPU 上进行向量化计算
    if not torch.is_tensor(positions):
        positions = torch.tensor(positions, dtype=torch.float32)

    # ===========================
    # 1️⃣ 计算两两无人机之间的欧式距离矩阵
    # torch.cdist(X, X, p=2) 会自动计算每对 (i,j) 间的 L2 距离
    # dist[i, j] = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
    # 结果是 [N, N] 的对称矩阵
    # ===========================
    dist = torch.cdist(positions, positions, p=2)

    # ===========================
    # 2️⃣ 根据通信半径构造邻接矩阵
    # 若两无人机间距离 ≤ 通信范围，则可通信 → 1；否则 0
    # to(dtype=torch.float32) 将 bool 类型转为浮点型，方便后续计算
    # ===========================
    adj = (dist <= comm_range).to(dtype=torch.float32)

    # ===========================
    # 3️⃣ 移除自环（自己到自己）
    # 因为在 GAT 层中通常会手动加入 self-loop（见 GATLayer forward）
    # 所以这里先将对角线元素置为 0，防止重复计算
    # ===========================
    idx = torch.arange(adj.size(0), device=adj.device)  # [0, 1, 2, ..., N-1]
    adj[idx, idx] = 0.0

    # ===========================
    # 4️⃣ 返回邻接矩阵
    # 举例：
    # 若有 3 架无人机：
    # dist =
    # [[0.0, 1.5, 3.2],
    #  [1.5, 0.0, 2.1],
    #  [3.2, 2.1, 0.0]]
    # comm_range = 2.0
    # → adj =
    # [[0, 1, 0],
    #  [1, 0, 1],
    #  [0, 1, 0]]
    # ===========================
    return adj
