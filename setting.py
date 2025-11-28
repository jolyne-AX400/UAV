#setting.py
import random
from typing import List, Tuple
class Setting:
    def __init__(self):
        self.V = {
            'MAP_X': 100,
            'MAP_Y': 100,
            'OBSTACLE': [
                [10, 20, 5, 5],[30, 10, 10, 5],[50, 25, 10, 5],[70, 10, 5, 5],[6, 65, 5, 5],
                [20, 65, 2, 10],[15, 50, 5, 10],[40, 45, 5, 10],[60, 55, 10, 5],[80, 50, 10, 10],
                [20, 80, 10, 10],[75, 75, 15, 5],[10, 90, 5, 5],[60, 90, 10, 5],[90, 90, 5, 5],
                [25, 25, 7, 7],[35, 60, 8, 4],[55, 40, 6, 6],[65, 70, 5, 8],[85, 80, 7, 5],
                [5, 5, 4, 4],[15, 15, 6, 3],[45, 10, 5, 5],[70, 40, 6, 6],
                [90, 60, 4, 4],[12, 35, 6, 6],[28, 50, 5, 5],[42, 70, 7, 3],[58, 80, 6, 4],
                [75, 55, 5, 5],[85, 65, 6, 6],[3, 10, 4, 4],[7, 15, 5, 3],[85, 10, 6, 6],[90, 20, 5, 5]
            ],
            'CHANNLE': 3,
            'NUM_UAV': 6,
            'NUM_AGENTS': 6,
            'INIT_POSITION': (50, 50),

            # ====== 改动A：更合理的能量和步长参数 ======
            'ENERGY': 2000,                 # 总能量
            'STEP_SCALE': 2.0,             # 单步位移尺度（配合动作的 dist∈[0,1]）
            'ENERGY_COST_PER_UNIT': 1.0,   # 每移动 1 单位消耗的能量
            'PENALTY_WARMUP_STEPS': 1200,   # 惩罚暖启动步数
            'MARK_RADIUS': 2,              # 新增：配合改动（扩大每步覆盖面积以加速探索）
            # 原 MAXDISTANCE 保留兼容（不再直接用于位移）
            'MAXDISTANCE': 2000,
        }

'''
import random
from typing import List, Tuple
class Setting:
    def __init__(self):
        self.V = {
            'MAP_X': 100,
            'MAP_Y': 100,
            'OBSTACLE': [
                [30, 10, 10, 5],[50, 25, 10, 5],[70, 10, 5, 5],
                [20, 65, 2, 10],[40, 45, 5, 10],[60, 55, 10, 5],[80, 50, 10, 10],
                [20, 80, 10, 10],[75, 75, 15, 5],[10, 90, 5, 5],[60, 90, 10, 5],[90, 90, 5, 5],
                [25, 25, 7, 7],[35, 60, 8, 4],[55, 40, 6, 6],[65, 70, 5, 8],[85, 80, 7, 5],
                [45, 10, 5, 5],[70, 40, 6, 6],
                [90, 60, 4, 4],[28, 50, 5, 5],[42, 70, 7, 3],[58, 80, 6, 4],
                [75, 55, 5, 5],[85, 65, 6, 6],[85, 10, 6, 6],[90, 20, 5, 5]
            ],
            'CHANNLE': 3,
            'NUM_UAV': 6,
            'NUM_AGENTS': 6,
            'INIT_POSITION': (50, 50),

            # ====== 改动A：更合理的能量和步长参数 ======
            'ENERGY': 3000,                 # 总能量
            'STEP_SCALE': 2.0,             # 单步位移尺度（配合动作的 dist∈[0,1]）
            'ENERGY_COST_PER_UNIT': 1.0,   # 每移动 1 单位消耗的能量
            'PENALTY_WARMUP_STEPS': 100,   # 惩罚暖启动步数
            'MARK_RADIUS': 1,              # 新增：配合改动
            # 原 MAXDISTANCE 保留兼容（不再直接用于位移）
            'MAXDISTANCE': 1000,

            #15:00加入的
            'OB_PROX_WEIGHT': 0.2,
            'OB_PROX_SHAPE':'exp',
            'OB_PROX_SIGMA':1,
            'OB_PROX_RANGE':1,
            'OB_PROX_SKIP_ON_COLLISION':True,
        }
'''
# Rect = List[int]  # [x, y, w, h]

# # ---------- 公共工具 ----------
# def aabb_overlap(a: Rect, b: Rect, gap: int = 0) -> bool:
#     ax, ay, aw, ah = a
#     bx, by, bw, bh = b
#     # 带可选最小间隙 gap 的AABB重叠判定
#     return not (ax + aw + gap <= bx or
#                 bx + bw + gap <= ax or
#                 ay + ah + gap <= by or
#                 by + bh + gap <= ay)

# def try_place_rects(
#     map_x: int, map_y: int,
#     n: int,
#     w_range: Tuple[int, int],
#     h_range: Tuple[int, int],
#     padding: int,
#     seed: int | None,
#     forbid_sets: List[List[Rect]],   # 不能与这些集合中的矩形重叠
#     no_overlap_within: bool = True,  # 是否同类之间也不重叠
#     min_gap: int = 0                 # 可选最小间隙（含同类与禁区）
# ) -> List[Rect]:
#     rng = random.Random(seed)
#     placed: List[Rect] = []

#     # 可行性检查
#     if w_range[0] <= 0 or h_range[0] <= 0:
#         raise ValueError("w_range 和 h_range 的下界必须 > 0")
#     if map_x - 2*padding <= 0 or map_y - 2*padding <= 0:
#         raise ValueError("padding 太大导致可放置空间为0")

#     max_attempts = max(200, n * 400)
#     attempts = 0

#     while len(placed) < n and attempts < max_attempts:
#         attempts += 1
#         w = rng.randint(*w_range)
#         h = rng.randint(*h_range)

#         if w > map_x - 2*padding or h > map_y - 2*padding:
#             continue

#         x = rng.randint(padding, map_x - w - padding)
#         y = rng.randint(padding, map_y - h - padding)
#         r = [x, y, w, h]

#         # 与禁区（障碍物等）检查
#         conflict = False
#         for s in forbid_sets:
#             if any(aabb_overlap(r, o, min_gap) for o in s):
#                 conflict = True
#                 break
#         if conflict:
#             continue

#         # 同类之间是否允许重叠
#         if no_overlap_within and any(aabb_overlap(r, p, min_gap) for p in placed):
#             continue

#         placed.append(r)

#     if len(placed) < n:
#         raise RuntimeError(
#             f"仅放置了 {len(placed)}/{n} 个。请减少数量、缩小尺寸范围，或增大地图/减小 padding / 减小 min_gap。"
#         )
#     return placed


# # ---------- 你的 Setting ----------
# class Setting:
#     def __init__(self,
#                  # 地图
#                  map_x: int = 150,
#                  map_y: int = 150,
#                  padding: int = 0,
#                  seed: int | None = None,
#                  # 障碍物
#                  n_obstacles: int = 30,
#                  obs_w_range: Tuple[int, int] = (3, 10),
#                  obs_h_range: Tuple[int, int] = (3, 10),
#                  obs_min_gap: int = 5,        # 障碍物间最小缝隙（0 表示允许贴边但不重叠）
#                  # 信息区域
#                  n_info: int = 6,
#                  info_w_range: Tuple[int, int] = (2, 6),
#                  info_h_range: Tuple[int, int] = (2, 6),
#                  info_min_gap_to_obs: int = 5,   # 信息区与障碍物之间最小缝隙
#                  info_min_gap_between: int = 5,  # 信息区彼此最小缝隙
#                  info_no_overlap_within: bool = True  # 信息区之间不重叠
#                  ):
#         self.MAP_X = map_x
#         self.MAP_Y = map_y

#         # 1) 先放障碍物（互不重叠，可选最小缝隙）
#         obstacles = try_place_rects(
#             map_x, map_y,
#             n=n_obstacles,
#             w_range=obs_w_range, h_range=obs_h_range,
#             padding=padding,
#             seed=None if seed is None else seed + 101,  # 让两类用不同随机流
#             forbid_sets=[],
#             no_overlap_within=True,
#             min_gap=obs_min_gap
#         )

#         # 2) 再放信息区域：不得与障碍物重叠（可设与障碍物间的最小缝隙），
#         #    默认信息区之间也不重叠（可关）
#         info_regions = try_place_rects(
#             map_x, map_y,
#             n=n_info,
#             w_range=info_w_range, h_range=info_h_range,
#             padding=padding,
#             seed=None if seed is None else seed + 202,
#             forbid_sets=[obstacles],                 # 禁止与障碍物重叠/靠太近
#             no_overlap_within=info_no_overlap_within,
#             min_gap=max(info_min_gap_to_obs, info_min_gap_between)
#             # 这里用一个 min_gap，如果你想区分“对障碍物的间隙”和“彼此间的间隙”，
#             # 可把 try_place_rects 拆成两次检查（示例下方给出）
#         )

#         self.V = {
#             'MAP_X': self.MAP_X,
#             'MAP_Y': self.MAP_Y,
#             'OBSTACLE': obstacles,
#             'INFO': info_regions,
#             'CHANNLE': 3,
#             'NUM_UAV': 6,
#             'NUM_AGENTS': 6,
#             'INIT_POSITION': (50, 50),

#             # ====== 改动A：更合理的能量和步长参数 ======
#             'ENERGY': 3000,                 # 总能量
#             'STEP_SCALE': 2.0,             # 单步位移尺度（配合动作的 dist∈[0,1]）
#             'ENERGY_COST_PER_UNIT': 1.0,   # 每移动 1 单位消耗的能量
#             'PENALTY_WARMUP_STEPS': 1000,   # 惩罚暖启动步数
#             'MARK_RADIUS': 1,              # 新增：配合改动
#             # 原 MAXDISTANCE 保留兼容（不再直接用于位移）
#             'MAXDISTANCE': 1000,
#         }

#     # 可选：若你希望 “信息区对障碍物的间隙” 与 “信息区彼此间的间隙” 分开控制，
#     # 可以改成先用 min_gap=info_min_gap_to_obs 放置时只对 forbid_sets 检查，
#     # 然后再逐个检查 placed 内部用 info_min_gap_between；或直接拷贝 try_place_rects
#     # 做两个不同的 gap 分支。


# # ---------- 用法示例 ----------
# if __name__ == "__main__":
#     s = Setting(
#         map_x=100, map_y=100, padding=1, seed=42,
#         n_obstacles=30, obs_w_range=(4,10), obs_h_range=(4,10), obs_min_gap=0,
#         n_info=6, info_w_range=(2,6), info_h_range=(2,6),
#         info_min_gap_to_obs=1, info_min_gap_between=0, info_no_overlap_within=True
#     )
#     print("OBSTACLE:", s.V['OBSTACLE'][:5], " ...")
#     print("INFO    :", s.V['INFO'])
