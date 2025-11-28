
# UAV Multi-Agent Exploration (Modified A+C+D)

关键改动：
- **A（步长-能量解耦）**：用 `STEP_SCALE` 与 `ENERGY_COST_PER_UNIT` 控制位移与能量。
- **C（奖励增强）**：加入共享的全局覆盖率增量奖励，并对碰撞/重叠惩罚使用暖启动日程。
- **D（训练稳定）**：PPO 加入 **熵正则** 与 **ReduceLROnPlateau** 学习率调度；修正观测归一化。

运行：
```bash
pip install -r requirements.txt
python train.py --episodes 200 --steps-per-episode 200
python eval.py
```
