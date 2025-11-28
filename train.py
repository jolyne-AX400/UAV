# 使用带全局统计拼接的 Critic（CTDE：训练期特权信息仅用于 Critic）
from models import ActorGAT, CriticCentralGlobal, AdjPredictor

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from env import SimpleEnv
# 下面这行导入了旧版 CriticCentral，但当前脚本实际上未使用（保留不影响运行）
from models import ActorGAT, CriticCentral
from graph_utils import build_adj_matrix
from buffer import RolloutBuffer
from utils import save_checkpoint, ensure_dir, save_logs
import csv


def node_features_from_obs(obs, max_energy=None):
    """
    把环境给的一步观测（每个 UAV 的状态 + 一张全局探索图）整理成每个无人机一行的数值特征矩阵，方便直接喂给 Actor（GAT）和 Critic 使用。
    它做了几件事：抽取关键信息 → 做归一化 → 计算局部探索密度 → 加上 agent 身份（one-hot）→ 组装成 [N, feat_dim] 张量。
    """
    feats = []
    for o in obs:
        # 连续位置（浮点）并做网格化相关的归一化使用
        x, y = float(o['uav_position'][0]), float(o['uav_position'][1])
        energy = float(o['energy'])
        explored = o['explored_map']  # 全局探索二值图（同一张图，各 agent 共享）
        gy, gx = explored.shape       # 注意：数组形状为 [行(y), 列(x)]，与常用(x,y)顺序相反

        # 以当前浮点位置四舍五入到最近网格
        xi, yi = int(round(x)), int(round(y))

        # r=3 的局部窗口，统计局部已探索的格子数
        r = 3
        x0, x1 = max(0, xi-r), min(gx, xi+r+1)
        y0, y1 = max(0, yi-r), min(gy, yi+r+1)
        local_sum = float(explored[y0:y1, x0:x1].sum())

        # 能量归一化分母：优先使用 max_energy；否则用当前能量（至少为1避免除零）
        denom = float(max_energy) if max_energy is not None else max(energy, 1.0)

        # 构建单个智能体的特征行：
        # 位置/地图归一化 + 能量归一化 + 局部探索密度 + one-hot agent id
        feats.append([x/gx, y/gy, energy/denom, local_sum/((2*r+1)**2), *o['agent_id']])
        # local_ratio = local_sum / ((2*r+1)**2)这给策略一个“附近新不新鲜”的信号，有助于减少重复覆盖、鼓励去没走过的地方。

    # 聚合为 [N, feat_dim] 的 float32 张量
    return torch.tensor(feats, dtype=torch.float32)


# def flatten_global(obs, max_energy=None):
#     """
#     将节点特征展平成一条 [1, N*feat_dim] 的向量（旧版 CriticCentral 使用）。
#     当前训练流程改为使用 CriticCentralGlobal（拼 extras），
#     这个函数在本文件中已不再调用，保留以兼容历史代码/调试。
#     """
#     N = len(obs)
#     feats = node_features_from_obs(obs, max_energy=max_energy)  # [N, feat_dim]
#     return feats.reshape(1, -1)  # [1, N*feat_dim]


def train(args):
    # 自动选择设备（优先用 CUDA）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建环境（你的 SimpleEnv 包含 n 个智能体、探索图、能量、步计数等）
    seed_value = getattr(args, 'seed', 42)
    env = SimpleEnv(seed_value=seed_value)
    num_agents = env.n

    # 节点特征维度 = 4（x,y,energy,local_sum） + N（one-hot agent id）
    # 注意：随 num_agents 增大，这个维度也会增加（O(N)）
    node_feat_dim = 4 + num_agents

    # Actor：两层 GAT + MLP 输出每个智能体的动作分布（均值与共享方差）
    actor = ActorGAT(node_in_dim=node_feat_dim, gat_hidden=64, action_dim=3).to(device)

    # 可学习的邻接预测器（PoC 版本）和信息价值控制器
    adj_predictor = AdjPredictor(node_in_dim=node_feat_dim, hidden=64).to(device)
    info_controller = None
    use_ude = getattr(args, 'use_ude', False)

    if use_ude:
        # try to create UDE model
        try:
            from models import UDEAdjDynamics
            ude_model = UDEAdjDynamics(node_feat_dim=node_feat_dim, hidden=64).to(device)
        except Exception:
            ude_model = None
    else:
        ude_model = None

    try:
        # InfoController 在 models.py 中新增
        from models import InfoController
        info_controller = InfoController(node_in_dim=node_feat_dim, hidden=32).to(device)
    except Exception:
        info_controller = None

    # Critic：使用“全局统计拼接”的集中式 Critic（仅训练期使用特权信息）
    # extras_dim=3: [coverage_ratio, energy_total_norm, t_norm]
    # 注意：这不改变执行期的通信受限，因为 extras 只进 Critic，不进 Actor
    critic = CriticCentralGlobal(
        num_agents=num_agents,
        node_in_dim=node_feat_dim,
        extras_dim=3,
        hidden=256
    ).to(device)

    # 优化器：联合优化 Actor + Critic 参数
    params = list(actor.parameters()) + list(critic.parameters()) + list(adj_predictor.parameters())
    if info_controller is not None:
        params += list(info_controller.parameters())
    if ude_model is not None:
        params += list(ude_model.parameters())

    optimizer = torch.optim.Adam(params, lr=3e-4)

    # 余弦退火学习率调度器（T_max=100个 step）；当前用 ep_reward 调 step（见下方注释）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)

    # 经验回放缓冲区（按时间展开的 on-policy rollouts），包含 GAE(γ,λ)
    buffer = RolloutBuffer(
        num_agents=num_agents, node_feat_dim=node_feat_dim,
        device=device, gamma=0.99, lam=0.95
    )

    # 准备输出目录和日志容器
    ensure_dir("checkpoints")
    logs = {"episode": [], "total_reward": [], "explored_ratio": []}
    ensure_dir("logs")

    # 构造输出文件前缀，避免不同实验互相覆盖
    mode_str = 'ude' if use_ude else 'no_ude'
    out_suffix = f"{mode_str}_{args.episodes}_seed{args.seed}"
    training_logs_path = f"logs/training_logs_{out_suffix}.csv"
    adj_stats_path = f"logs/adj_stats_{out_suffix}.csv"
    final_ckpt_path = f"checkpoints/final_{out_suffix}.pt"

    # 用于记录邻接统计（每步）：列表的元素为 (ep, step, adj_mean, adj_diff)
    adj_stats = []

    # 熵正则系数：前期探索更强，逐渐衰减（见循环内线性退火）
    base_entropy_coef = 3e-2
    entropy_coef = 2e-2  # 初始值（会在循环中被覆盖）

    # 从环境配置里读最大能量上限（用于能量归一化的 denom）
    max_energy = env.sg.V.get('ENERGY', 400.0)

    # ===== 主训练循环 =====
    for ep in range(1, args.episodes+1):

        # 每个 episode 开始时更新熵系数（100 回合线性衰减到 ~1e-3）
        entropy_coef = max(1e-3, base_entropy_coef * (1 - ep / 1000.0))

        # 环境复位，获取初始观测（list，长度=N）
        obs = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        prev_adj = None

        # 清空 rollout buffer，准备收集新一条轨迹
        buffer.clear()

        # 记录每个时间步的全局统计（extras），供 PPO 更新阶段重算 Critic 用
        extras_seq = []

        # ======== 采样一条 episode 轨迹（或到最大步数截断）========
        while not done and steps < args.steps_per_episode:

            # 构造当前时刻的节点特征 [N, feat_dim]（包含 one-hot id）
            feats = node_features_from_obs(obs, max_energy=max_energy).to(device)

            # 基于当前 N 个位置和通信半径，构造邻接矩阵 [N, N]
            positions = torch.tensor([o['uav_position'] for o in obs], dtype=torch.float32).to(device)

            # 使用可学习的邻接预测器 + 物理掩码
            with torch.no_grad():
                if ude_model is not None:
                    # use UDE to integrate adjacency dynamics; use adj_predictor to provide initial A0 logits
                    a0_probs, a0_logits = adj_predictor(feats, top_k=None)
                    a0_logits = a0_logits.to(device)
                    phys_mask = build_adj_matrix(positions, comm_range=args.comm_range).to(device)
                    # initial A0 masked with phys_mask
                    a0_logits_masked = a0_logits * phys_mask
                    try:
                        adj_probs, adj_logits = ude_model.integrate(a0_logits_masked, feats, t_span=(0.0, 1.0))
                    except RuntimeError as e:
                        raise RuntimeError(str(e) + "\nInstall dependency via: pip install torchdiffeq")
                    # 记录邻接统计（UDE 路径）
                    try:
                        adj_mean = float(adj_probs.mean().detach().cpu().item())
                    except Exception:
                        adj_mean = float(adj_probs.mean())

                    if prev_adj is None:
                        adj_diff = 0.0
                    else:
                        cur = adj_probs.detach().cpu()
                        adj_diff = float(torch.norm(cur - prev_adj).item())
                    adj_stats.append((ep, steps, adj_mean, adj_diff))
                    prev_adj = adj_probs.detach().cpu()
                else:
                    raw_adj, logits = adj_predictor(feats, top_k=args.top_k)  # [N,N]
                    phys_mask = build_adj_matrix(positions, comm_range=args.comm_range).to(device)

                    if info_controller is not None:
                        gates = info_controller(feats)  # [N]
                        gate_outer = gates.unsqueeze(1) * gates.unsqueeze(0)  # [N,N]
                    else:
                        gate_outer = 1.0

                    adj_probs = raw_adj * phys_mask * gate_outer

                    # 记录邻接统计
                    try:
                        adj_mean = float(adj_probs.mean().detach().cpu().item())
                    except Exception:
                        adj_mean = float(adj_probs.mean())

                    if prev_adj is None:
                        adj_diff = 0.0
                    else:
                        # prev_adj 存为 CPU tensor
                        cur = adj_probs.detach().cpu()
                        adj_diff = float(torch.norm(cur - prev_adj).item())
                    # 保存为 CPU 便于后续分析
                    adj_stats.append((ep, steps, adj_mean, adj_diff))
                    prev_adj = adj_probs.detach().cpu()
                # 二值化邻接（简单阈值），可替换为 top-k
                adj_mask = (adj_probs > args.adj_threshold).to(dtype=torch.float32)

                # 策略网络前向：得到动作分布参数；按分布采样动作；拿到 logp（每个智能体一行）
                mean, std = actor(feats, adj_mask)                 # [N,3], [N,3]
                dist = Normal(mean, std)
                actions = dist.sample()                       # [N,3]  采样连续动作
                logp = dist.log_prob(actions).sum(dim=-1)     # [N]    对动作维度求和

                # === 计算仅供 Critic 使用的全局统计量（CTDE 的“特权信息”）===
                explored_map = obs[0]['explored_map']         # 全局探索图：任意 o['explored_map'] 等价
                coverage_ratio = float(explored_map.mean())   # 覆盖率 ∈ [0,1]
                total_energy = sum(float(o['energy']) for o in obs)
                energy_total_norm = total_energy / (num_agents * max_energy + 1e-6)  # 平均能量比例 ∈ [0,1]
                # 时间步归一化（也可用 steps/args.steps_per_episode）
                t_norm = float(env.t) / float(args.steps_per_episode + 1e-6)

                # extras_t: [3]（张量，放到同一 device）
                extras_t = torch.tensor(
                    [coverage_ratio, energy_total_norm, t_norm],
                    dtype=torch.float32, device=device
                )

                # Critic 估值：V(s_t)（标量），输入为 [N,feat_dim] + extras_t
                value = critic(feats, extras_t)

            # ======== 与环境交互：执行动作，推进到下一步 ========
            # 将动作张量转为 Python list（环境 step 需要 Python 原生数值）
            actions_np = actions.cpu().numpy().tolist()
            next_obs, rewards, done, info = env.step(actions_np)

            # 将当前步的数据推入 buffer（注意：这里按原逻辑保持 CPU 存储）
            # - feats: 节点特征 [N,feat]
            # - adj  : 邻接矩阵 [N,N]
            # - actions: [N,3]
            # - logp   : [N]
            # - rewards: list/np.array，长度=N 或 标量（你的 env 返回为 per-agent）
            # - value  : 标量（V(s)）
            # - done   : 标量（终止标志）
            # 将当前步的数据推入 buffer（注意：这里按原逻辑保持 CPU 存储）
            buffer.add(
                feats.cpu(),
                adj_mask.cpu(),
                actions.detach().cpu(),                      # use detach().cpu() to avoid torch.tensor(tensor) warning
                logp.cpu(),
                rewards,
                value.detach().cpu().unsqueeze(0),       # keep as tensor [1]
                float(done),
                positions.cpu()
            )

            # 累计 episode 奖励（把 per-agent rewards 求和）
            ep_reward += float(np.sum(rewards))

            # 状态推进
            obs = next_obs
            steps += 1

            # 记录本步 extras（在 PPO 更新阶段用于重算 Critic）
            extras_seq.append(extras_t.detach().cpu())

        # ======== 轨迹结束后：bootstrap 最后一刻的价值，用于 GAE ========
        with torch.no_grad():
            # 终止时刻的全局统计（与循环内一致的构造方式）
            explored_map = obs[0]['explored_map']
            coverage_ratio = float(explored_map.mean())
            total_energy = sum(float(o['energy']) for o in obs)
            energy_total_norm = total_energy / (num_agents * max_energy + 1e-6)
            t_norm = float(env.t) / float(args.steps_per_episode + 1e-6)
            final_extras = torch.tensor(
                [coverage_ratio, energy_total_norm, t_norm],
                dtype=torch.float32, device=device
            )
            # 最后一刻的节点特征
            last_feats = node_features_from_obs(obs, max_energy=max_energy).to(device)
            # V(s_T)
            last_value = critic(last_feats, final_extras)

        # 基于 last_value 做 GAE(Returns & Advantages) 计算（在 CPU 上）
        buffer.compute_returns_and_advantages(last_value.cpu())

        # —— 训练开始前，确保用 Setting 的 ENERGY 作为归一化分母（与 eval 一致）
        max_energy = env.sg.V.get('ENERGY', 3000.0)

        # ================= PPO 更新阶段 =================
        obs_tensor, positions_tensor, adjs_tensor, actions_tensor, logp_tensor, values_old, returns, advantages = buffer.get_batches()
        obs_tensor = obs_tensor.to(device)      # [T, N, feat_dim]
        positions_tensor = positions_tensor.to(device)  # [T, N, 2]
        adjs_tensor = adjs_tensor.to(device)    # [T, N, N]
        actions_tensor = actions_tensor.to(device)  # [T, N, act_dim]
        logp_tensor = logp_tensor.to(device)        # [T, N]
        values_old = values_old.to(device)          # [T]  —— 联合价值
        returns = returns.to(device)                # [T]
        advantages = advantages.to(device)          # [T]

        # 优势标准化提高稳定性
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ppo_epochs = args.ppo_epochs
        clip_eps = args.clip_eps
        vf_clip = getattr(args, 'vf_clip', 0.2)     # 新增：价值剪切阈值
        batch_size = getattr(args, 'batch_size', 256)

        T = obs_tensor.size(0)
        idx_all = torch.arange(T, device=device)

        # ——（可选但推荐）在进入 PPO epoch 前做一次优势归一化（只需一次）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # ★

        early_stop_ppo = False  # ★ 为 KL 早停准备

        for _ in range(ppo_epochs):
            perm = torch.randperm(T, device=device)
            for start in range(0, T, batch_size):
                idx = perm[start:start+batch_size]

                feats_b = obs_tensor[idx]        # [B, N, feat]
                adjs_b = adjs_tensor[idx]        # [B, N, N]
                positions_b = positions_tensor[idx]  # [B, N, 2]
                acts_b = actions_tensor[idx]     # [B, N, act]
                old_logp_b = logp_tensor[idx]    # [B, N]
                adv_b = advantages[idx]          # [B]
                ret_b = returns[idx]             # [B]
                v_old_b = values_old[idx]        # [B]
                extras_b = torch.stack([extras_seq[i] for i in idx.tolist()]).to(device)  # [B, 3]

                new_logps = []
                entropies = []
                values_pred = []
                sparsity_sum = 0.0

                # —— 逐时间步前向（你可以保持这种写法）
                for b in range(feats_b.size(0)):
                    # 重新用 AdjPredictor 计算邻接（使其在更新中可被优化）
                    raw_adj_b, logits_b = adj_predictor(feats_b[b], top_k=args.top_k)
                    phys_mask_b = build_adj_matrix(positions_b[b], comm_range=args.comm_range).to(device)

                    if info_controller is not None:
                        gates_b = info_controller(feats_b[b])
                        gate_outer_b = gates_b.unsqueeze(1) * gates_b.unsqueeze(0)
                    else:
                        gate_outer_b = 1.0

                    adj_b = (raw_adj_b * phys_mask_b * gate_outer_b)
                    adj_mask_b = (adj_b > args.adj_threshold).to(dtype=torch.float32)

                    mean_t, std_t = actor(feats_b[b], adj_mask_b)         # [N,act], [N,act]
                    dist_t = Normal(mean_t, std_t + 1e-6)                # ★ 数值微量，防止 0 方差
                    logp_t = dist_t.log_prob(acts_b[b]).sum(dim=-1)      # [N]，只在 act 维求和
                    # 熵：改为在 act 维均值，再在 agent 维均值，解耦 N 的尺度
                    entropy_t = dist_t.entropy().mean(dim=-1)            # [N]（act 平均）
                    entropies.append(entropy_t.mean())                   # ★ agent 维均值 → 标量

                    v_t = critic(feats_b[b], extras_b[b])                # 标量
                    values_pred.append(v_t)
                    new_logps.append(logp_t)
                    sparsity_sum = sparsity_sum + raw_adj_b.mean()

                new_logps = torch.stack(new_logps)     # [B, N]
                entropy = torch.stack(entropies).mean()# ★ 整个 batch 的均值熵（不随 N 放大）
                values_pred = torch.stack(values_pred).view(-1)  # [B]

                # —— 策略：逐 agent 的 ratio，然后在 agent 维做均值（解耦 N 的尺度）★
                r = torch.exp(new_logps - old_logp_b)                  # [B, N]
                adv_expanded = adv_b.unsqueeze(1)                      # [B, 1]
                surr1 = (r * adv_expanded).mean(dim=1)                 # [B]
                surr2 = (torch.clamp(r, 1.0 - clip_eps, 1.0 + clip_eps) * adv_expanded).mean(dim=1)
                policy_loss = -torch.min(surr1, surr2).mean()

                # —— 价值剪切（保留你的做法）
                v_clipped = v_old_b + (values_pred - v_old_b).clamp(-vf_clip, vf_clip)
                v_loss1 = F.mse_loss(values_pred, ret_b)
                v_loss2 = F.mse_loss(v_clipped, ret_b)
                value_loss = torch.max(v_loss1, v_loss2)

                # —— KL 早停：用与上面相同的“均值口径”估一个近似 KL（越大表示步子太猛）★
                approx_kl = (old_logp_b - new_logps).mean()            # 标量（均值口径）
                if approx_kl.abs().item() > 0.05:                      # 阈值 0.01~0.05 可调
                    early_stop_ppo = True
                    # 直接跳出 mini-batch 循环，进入下一 epoch
                    break

                # 平均稀疏损失（batch 内均值）
                sparsity_loss = sparsity_sum / float(feats_b.size(0))

                loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy + args.alpha_sparse * sparsity_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()) + list(adj_predictor.parameters()),
                    max_norm=0.5
                )
                optimizer.step()

            if early_stop_ppo:  # ★ 若某个 batch 的 KL 过大，提前结束本 epoch 的剩余更新
                break


        # ======== 记录日志 & 调度学习率 ========
        logs["episode"].append(ep)
        logs["total_reward"].append(ep_reward)
        logs["explored_ratio"].append(float(info.get('explored_ratio', 0.0)))

        # 正确推进 CosineAnnealingLR：不传入 ep_reward
        scheduler.step()


        # 定期保存 checkpoint 与打印日志
        if ep % args.save_interval == 0:
            save_checkpoint({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': ep,
                'node_feat_dim': node_feat_dim
            }, f"checkpoints/ckpt_{mode_str}_ep{ep}.pt")

        if ep % args.log_interval == 0:
            print(f"EP {ep} | reward {ep_reward:.2f} | explored_ratio {logs['explored_ratio'][-1]:.4f}")

    # 训练结束后保存日志与最终模型
    os.makedirs("logs", exist_ok=True)
    save_logs(logs, training_logs_path)

    # 将邻接统计写入 CSV 供后续分析
    if len(adj_stats) > 0:
        with open(adj_stats_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'step', 'adj_mean', 'adj_diff'])
            for row in adj_stats:
                writer.writerow(row)

    save_checkpoint({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': args.episodes,
        'node_feat_dim': node_feat_dim
    }, final_ckpt_path)
    print(f'Training finished, models saved to {final_ckpt_path}')


if __name__ == '__main__':
    # 命令行参数（回合数、每回合步数、通信半径、PPO 超参、保存/日志间隔等）
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20000)
    parser.add_argument('--steps-per-episode', type=int, default=2000)
    parser.add_argument('--comm-range', type=float, default=25.0)
    parser.add_argument('--ppo-epochs', type=int, default=4)
    parser.add_argument('--clip-eps', type=float, default=0.2)
    parser.add_argument('--adj-threshold', type=float, default=0.5,
                        help='Threshold to binarize predicted adjacency probabilities')
    parser.add_argument('--alpha-sparse', type=float, default=1e-2,
                        help='Weight for adjacency sparsity regularization')
    parser.add_argument('--top-k', type=int, default=3,
                        help='If >0, keep top-k predicted edges per node (PoC sparse)')
    parser.add_argument('--use-ude', action='store_true', help='Use UDE adjacency dynamics (requires torchdiffeq)')
    parser.add_argument('--save-interval', type=int, default=50)
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for the environment')
    args = parser.parse_args()

    # 启动训练
    train(args)
