import sys
import os
import torch
import torch.nn as nn
import traceback

# Ensure project root is on sys.path so `import models` works when running from scripts/
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import AdjPredictor, UDEAdjDynamics

def run_test():
    try:
        device = torch.device('cpu')
        N = 4
        d = 6
        # 随机节点特征
        node_feats = torch.randn(N, d, device=device)

        # 简单的 AdjPredictor
        adj_pred = AdjPredictor(node_in_dim=d, hidden=16).to(device)

        # 生成 logits（注意：AdjPredictor 返回 probs, logits）
        probs, logits = adj_pred(node_feats)
        # logits 是 [N, N]

        ude = UDEAdjDynamics(node_feat_dim=d, hidden=16).to(device)

        # 运行 UDE 积分（短 t_span）
        A_t_probs, a_end = ude.integrate(logits, node_feats, t_span=(0.0, 0.2), method='rk4')

        # 简单损失：终点概率的均值（标量）
        loss = A_t_probs.mean()
        loss.backward()

        # 检查 AdjPredictor 中参数是否有梯度
        total_grad = 0.0
        for name, p in adj_pred.named_parameters():
            if p.grad is None:
                print(f"param {name} grad is None")
            else:
                gnorm = p.grad.detach().abs().sum().item()
                print(f"param {name} grad sum abs: {gnorm:.6e}")
                total_grad += gnorm

        print(f"total grad sum abs in AdjPredictor: {total_grad:.6e}")

        # UDE 参数梯度
        ude_total = 0.0
        for name, p in ude.named_parameters():
            if p.grad is None:
                print(f"ude param {name} grad is None")
            else:
                g = p.grad.detach().abs().sum().item()
                print(f"ude param {name} grad sum abs: {g:.6e}")
                ude_total += g
        print(f"total grad sum abs in UDE: {ude_total:.6e}")

        if total_grad > 0:
            print('SUCCESS: gradients flowed to AdjPredictor')
        else:
            print('FAIL: no gradients to AdjPredictor')

    except Exception as e:
        print('Exception during test:')
        traceback.print_exc()

if __name__ == '__main__':
    run_test()
