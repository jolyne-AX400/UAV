import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import pandas as pd
from setting import Setting

#10.24 15:50备份
class SimpleEnv:
    """
    SimpleEnv 是一个简化的多智能体环境，旨在模拟多无人机（UAV）探索未知地图的过程。
    环境包含多个智能体（无人机），每个智能体有独立的状态（位置、能量等），并且共享探索区域的奖励。
    """

    def __init__(self, seed_value=1):
        """
        初始化环境，设置地图尺寸、智能体数量、障碍物等环境变量。
        
        参数:
        - seed_value: 随机种子，用于初始化环境的随机性
        """
        self.prev_explored_count = 0  # 上一轮探索的区域数量
        self.sg = Setting()  # 设置文件实例，获取环境设置
        self.V = self.sg.V  # 从设置文件中加载配置

        # 地图的宽度和高度
        self.mapx = self.V['MAP_X']
        self.mapy = self.V['MAP_Y']
        # 无人机数量
        self.n = self.V.get('NUM_AGENTS', 4)

        # 探索地图（记录每个点是否被探索）
        self.explored_map = np.zeros((self.mapx, self.mapy), dtype=np.int8)
        self.explored_maps = [np.zeros((self.mapx, self.mapy), dtype=np.int8) for _ in range(self.n)]#个体探索图，只记录第 i 个智能体自己探索到的格子（用于个人探索奖励）。
        self.prev_explored_maps = [np.zeros((self.mapx, self.mapy), dtype=np.int8) for _ in range(self.n)]

        # UAV（无人机）状态
        self.uav = [list(self.V['INIT_POSITION']) for _ in range(self.n)]
        self.start_position = [pos.copy() for pos in self.uav]
        self.uav_paths = [[list(pos)] for pos in self.uav]

        # ====== 改动A：将位移尺度与能量单位解耦 ======
        # 单位步长尺度（每步可走的最大“比例” * 该尺度）
        self.step_scale = float(self.V.get('STEP_SCALE', 2.0))  # 每步最大物理位移尺度（网格单位）
        # 每移动 1 单位所需能量
        self.energy_cost_per_unit = float(self.V.get('ENERGY_COST_PER_UNIT', 1.0))
        self.energy = [float(self.V.get('ENERGY', 3000.0)) for _ in range(self.n)]

        # 兼容原字段
        self.maxdistance = self.V.get('MAXDISTANCE', 1000)

        # 总奖励
        self.total_reward = 0

        # 障碍物地图（1为障碍物，0为空地）
        self.OB = 1
        self.mapob = np.zeros((self.mapx, self.mapy), dtype=np.int8)
        # 根据配置的障碍物位置设置障碍物
        for x_start, y_start, width, height in self.V['OBSTACLE']:
            self.mapob[y_start:y_start + height, x_start:x_start + width] = self.OB

        # 日志记录
        self.reward_log = []  # 奖励日志
        self.energy_log = []  # 能量日志
        self.collision_log = []  # 碰撞日志
        self.visited_log = []  # 访问日志
        self.explored_ratio_log = []  # 探索比例日志

        # 训练步计数（用于逐步加大惩罚强度）
        self.t = 0
        self.penalty_warmup_steps = int(self.V.get('PENALTY_WARMUP_STEPS', 1500))

        if seed_value is not None:
            np.random.seed(seed_value)  # 设置随机种子

    def reset(self):
        """重置环境，初始化所有无人机的位置和能量等"""
        self.uav = [list(self.V['INIT_POSITION']) for _ in range(self.n)]
        self.start_position = [pos.copy() for pos in self.uav]

        # 统一与 __init__ / Setting 一致的能量上限（默认 3000），避免训练/评估尺度漂移
        self.energy = [float(self.V.get('ENERGY', 3000.0)) for _ in range(self.n)]

        self.uav_paths = [[list(pos)] for pos in self.uav]

        self.explored_map = np.zeros((self.mapx, self.mapy), dtype=np.int8)
        self.explored_maps = [np.zeros((self.mapx, self.mapy), dtype=np.int8) for _ in range(self.n)]
        self.prev_explored_maps = [np.zeros((self.mapx, self.mapy), dtype=np.int8) for _ in range(self.n)]

        self.prev_explored_count = 0
        self.total_reward = 0
        self.t = 0

        return self.get_state()  # 返回初始状态


    def get_state(self):
        """返回所有无人机的状态"""
        states = []
        for i in range(self.n):
            one_hot = np.zeros(self.n, dtype=np.float32)
            one_hot[i] = 1.0  # 当前智能体的one-hot表示
            state = {
                'uav_position': self.uav[i],
                'energy': self.energy[i],
                'explored_map': self.explored_map.copy(),
                'agent_id': one_hot,
            }
            states.append(state)
        return states

    def calculate_reward(self, move_distances, collisions):
        """
        计算每个无人机的奖励，包括探索奖励、碰撞惩罚、重叠惩罚等。
        
        参数：
        - move_distances: 每个无人机移动的距离
        - collisions: 每个无人机是否发生碰撞
        
        返回：
        - rewards: 每个无人机的奖励
        - reward_components: 奖励组成部分的详细信息
        """
        rewards = []
        reward_components = []
        explored_area = np.sum(self.explored_map)  # 当前已探索区域
        total_area = self.mapx * self.mapy  # 地图总面积
        explored_ratio = explored_area / total_area  # 探索比例

        delta_coverage = explored_area - self.prev_explored_count  # 本轮探索新增的区域
        self.prev_explored_count = explored_area

        # ====== 改动C：共享覆盖率增量奖励 + 温和惩罚日程 ======
        penalty_scale = min(1.0, float(self.t) / max(1, self.penalty_warmup_steps))  # 随时间增长的惩罚强度

        # 权重设定
        collision_penalty_weight = 0.3 * penalty_scale
        overlap_penalty_weight = 0.1 * penalty_scale
        exploration_reward_weight = 0.5
        global_coverage_reward_weight = 0.25  # 共享给所有agent

        # 碰撞惩罚
        collision_penalties = np.array(collisions, dtype=np.float32) * collision_penalty_weight

        # 重叠惩罚（智能体之间的距离过近会产生重叠惩罚）
        overlap_penalties = np.zeros(self.n, dtype=np.float32)
        min_distance_threshold = 2.0
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    dist_ij = np.linalg.norm(np.array(self.uav[i]) - np.array(self.uav[j])) # 计算两架无人机的距离
                    if dist_ij < min_distance_threshold:
                        overlap_penalties[i] += overlap_penalty_weight * (min_distance_threshold - dist_ij)

        # 探索奖励（个体）
        exploration_rewards = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            delta_explored = np.sum(np.clip(self.explored_maps[i] - self.prev_explored_maps[i], 0, 1))
            exploration_rewards[i] = exploration_reward_weight * float(delta_explored)

        # 共享覆盖率增量奖励
        shared = global_coverage_reward_weight * float(delta_coverage) / max(1, self.n)

        # 计算每个智能体的总奖励
        for i in range(self.n):
            total_reward = exploration_rewards[i] + shared - collision_penalties[i] - overlap_penalties[i]
            rewards.append(float(total_reward))
            reward_components.append({
                'explore_reward': float(exploration_rewards[i]),
                'shared_coverage': float(shared),
                'collision_penalty': float(collision_penalties[i]),
                'overlap_penalty': float(overlap_penalties[i]),
                'total': float(total_reward)
            })

        return rewards, reward_components

    def step(self, actions):
        """
        根据智能体的动作更新环境状态，计算奖励，并检查是否达到终止条件（能量耗尽）。
        
        参数：
        - actions: 每个智能体的动作（方向和距离）
        
        返回：
        - next_state: 环境的下一个状态
        - rewards: 每个智能体的奖励
        - done: 是否结束（所有智能体的能量都耗尽时）
        - info: 额外的信息（如碰撞、奖励细节等）
        """
        collisions = [0 for _ in range(self.n)]  # 碰撞情况
        newly_explored = 0  # 新探索的区域数
        move_distances = [0.0 for _ in range(self.n)]  # 每个智能体的移动距离

        new_positions = []
        for i, action in enumerate(actions):
            dx, dy, dist = action
            norm = np.sqrt(dx ** 2 + dy ** 2)
            if norm > 0:
                dx /= norm
                dy /= norm
            dist = np.clip(dist, 0.1, 1.0)

            # ====== 改动A：使用 step_scale + energy_cost_per_unit ======
            move_distance = float(dist) * self.step_scale
            available_units = max(0.0, float(self.energy[i]) / self.energy_cost_per_unit)  # 当前可用单位位移
            actual_distance = max(0.0, min(move_distance, available_units))#看能量是否够要移动的距离，取小的那个
            #计算新坐标
            new_x = self.uav[i][0] + dx * actual_distance
            new_y = self.uav[i][1] + dy * actual_distance
            #边界限制
            new_x = float(np.clip(new_x, 0, self.mapx - 1))
            new_y = float(np.clip(new_y, 0, self.mapy - 1))

            x_int, y_int = int(round(new_x)), int(round(new_y))
            if self.mapob[y_int, x_int] == self.OB:  # 检查是否撞到障碍物
                new_pos = self.uav[i]
                collisions[i] += 1
                actual_distance = 0.0  # 碰到障碍不移动也不扣能量
            else:
                new_pos = [new_x, new_y]

            new_positions.append(new_pos)
            move_distances[i] = actual_distance

        # 更新无人机状态和能量消耗
        for i in range(self.n):
            self.uav[i] = new_positions[i]
            self.uav_paths[i].append(list(new_positions[i]))
            # 扣除能量
            self.energy[i] = max(0.0, float(self.energy[i]) - move_distances[i] * self.energy_cost_per_unit)

            x_int, y_int = int(round(new_positions[i][0])), int(round(new_positions[i][1]))
            radius = self.V.get('MARK_RADIUS', 2)  # 扩大探索范围（原为1）
            for dx_ in range(-radius, radius + 1):
                for dy_ in range(-radius, radius + 1):
                    nx, ny = x_int + dx_, y_int + dy_
                    if 0 <= nx < self.mapx and 0 <= ny < self.mapy:
                        if self.mapob[ny, nx] != self.OB:
                            if self.explored_map[ny, nx] == 0:
                                self.explored_map[ny, nx] = 1
                                newly_explored += 1
                            if self.explored_maps[i][ny, nx] == 0:
                                self.explored_maps[i][ny, nx] = 1

        reward, reward_components = self.calculate_reward(move_distances, collisions)

        self.total_reward += sum(reward)
        self.reward_log.append(reward)
        self.energy_log.append(sum(self.energy) / self.n)
        self.collision_log.append(collisions)
        self.visited_log.append(0)

        # 计算当前探索比例
        explored_area = np.sum(self.explored_map)
        total_area = self.mapx * self.mapy
        explored_ratio = explored_area / total_area
        self.explored_ratio_log.append(explored_ratio)

        for i in range(self.n):
            self.prev_explored_maps[i] = np.copy(self.explored_maps[i])

        done = all(e <= 0 for e in self.energy)  # 如果所有无人机的能量都用完，结束游戏

        info = {
            'collisions': collisions,
            'total_reward': self.total_reward,
            'newly_explored': newly_explored,
            'explored_ratio': explored_ratio,
            'reward_components': reward_components
        }

        # 增加一步数（用于调整惩罚）
        self.t += 1

        return self.get_state(), reward, done, info

    def render(self, save_path=None):
        """
        渲染当前环境状态，显示地图、无人机的位置和路径。
        
        参数：
        - save_path: 可选的保存路径，如果提供则将图像保存为文件
        """
        map_data = np.zeros((self.mapx, self.mapy))
        map_data[self.mapob == self.OB] = 0
        map_data[self.mapob != self.OB] = 1

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(map_data, cmap='gray', origin='lower')

        for i in range(self.n):
            ax.scatter(self.uav[i][0], self.uav[i][1], s=100, label=f"UAV {i}")

        for i in range(self.n):
            if len(self.uav_paths[i]) > 1:
                uav_x, uav_y = zip(*self.uav_paths[i])
                ax.plot(uav_x, uav_y, linewidth=2, label=f"UAV {i} Path")

        ax.set_title("UAV Environment Map")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=100)
        plt.show()

    def save_logs_to_csv(self, file_path="env_logs.csv"):
        """
        将日志保存为CSV文件，便于后续分析和可视化。
        
        参数：
        - file_path: 保存日志的文件路径
        """
        def _sum_list(x): 
            return float(np.sum(x)) if isinstance(x, (list, np.ndarray)) else float(x)
        def _mean_list(x): 
            return float(np.mean(x)) if isinstance(x, (list, np.ndarray)) else float(x)

        data = {
            "step": list(range(len(self.reward_log))),
            "reward_sum": [ _sum_list(r) for r in self.reward_log ],
            "reward_mean": [ _mean_list(r) for r in self.reward_log ],
            "energy_mean": self.energy_log,
            "collisions_sum": [ _sum_list(c) for c in self.collision_log ],
            "visited_points": self.visited_log,
            "explored_ratio": self.explored_ratio_log,
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        print(f"✅ 日志已保存至: {file_path}")
'''
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import pandas as pd
from setting import Setting

class SimpleEnv:
    """
    SimpleEnv 是一个简化的多智能体环境，旨在模拟多无人机（UAV）探索未知地图的过程。
    环境包含多个智能体（无人机），每个智能体有独立的状态（位置、能量等），并且共享探索区域的奖励。
    """

    def __init__(self, seed_value=1):
        """
        初始化环境，设置地图尺寸、智能体数量、障碍物等环境变量。
        
        参数:
        - seed_value: 随机种子，用于初始化环境的随机性
        """
        self.prev_explored_count = 0  # 上一轮探索的区域数量
        self.sg = Setting()  # 设置文件实例，获取环境设置
        self.V = self.sg.V  # 从设置文件中加载配置

        #15：00之后加入
        # === 避障：半径门控 + 平滑惩罚 的默认超参（可在 Setting.V 覆盖） ===
        self.V.setdefault('OB_PROX_RANGE', self.V.get('MARK_RADIUS', 2))   # 感知半径（只在半径内触发）
        self.V.setdefault('OB_PROX_WEIGHT', 0.2)                            # 靠近惩罚强度
        self.V.setdefault('OB_PROX_SHAPE', 'exp')                           # 'exp' / 'linear' / 'quadratic'
        self.V.setdefault('OB_PROX_SIGMA', max(1.0, float(self.V['OB_PROX_RANGE'])/2.0))  # 指数形的尺度
        self.V.setdefault('OB_PROX_SKIP_ON_COLLISION', True)                # 碰撞当步是否跳过靠近惩罚

        # 地图的宽度和高度
        self.mapx = self.V['MAP_X']
        self.mapy = self.V['MAP_Y']
        # 无人机数量
        self.n = self.V.get('NUM_AGENTS', 4)

        # 探索地图（记录每个点是否被探索）
        self.explored_map = np.zeros((self.mapx, self.mapy), dtype=np.int8)
        self.explored_maps = [np.zeros((self.mapx, self.mapy), dtype=np.int8) for _ in range(self.n)]#个体探索图，只记录第 i 个智能体自己探索到的格子（用于个人探索奖励）。
        self.prev_explored_maps = [np.zeros((self.mapx, self.mapy), dtype=np.int8) for _ in range(self.n)]

        # UAV（无人机）状态
        self.uav = [list(self.V['INIT_POSITION']) for _ in range(self.n)]
        self.start_position = [pos.copy() for pos in self.uav]
        self.uav_paths = [[list(pos)] for pos in self.uav]

        # ====== 改动A：将位移尺度与能量单位解耦 ======
        # 单位步长尺度（每步可走的最大“比例” * 该尺度）
        self.step_scale = float(self.V.get('STEP_SCALE', 2.0))  # 每步最大物理位移尺度（网格单位）
        # 每移动 1 单位所需能量
        self.energy_cost_per_unit = float(self.V.get('ENERGY_COST_PER_UNIT', 1.0))
        self.energy = [float(self.V.get('ENERGY', 3000.0)) for _ in range(self.n)]

        # 兼容原字段
        self.maxdistance = self.V.get('MAXDISTANCE', 1000)

        # 总奖励
        self.total_reward = 0

        # 障碍物地图（1为障碍物，0为空地）
        self.OB = 1
        self.mapob = np.zeros((self.mapx, self.mapy), dtype=np.int8)
        # 根据配置的障碍物位置设置障碍物
        for x_start, y_start, width, height in self.V['OBSTACLE']:
            self.mapob[y_start:y_start + height, x_start:x_start + width] = self.OB

        # 日志记录
        self.reward_log = []  # 奖励日志
        self.energy_log = []  # 能量日志
        self.collision_log = []  # 碰撞日志
        self.visited_log = []  # 访问日志
        self.explored_ratio_log = []  # 探索比例日志

        # 训练步计数（用于逐步加大惩罚强度）
        self.t = 0
        self.penalty_warmup_steps = int(self.V.get('PENALTY_WARMUP_STEPS', 200))

        if seed_value is not None:
            np.random.seed(seed_value)  # 设置随机种子

    def reset(self):
        """重置环境，初始化所有无人机的位置和能量等"""
        self.uav = [list(self.V['INIT_POSITION']) for _ in range(self.n)]
        self.start_position = [pos.copy() for pos in self.uav]
        self.energy = [float(self.V.get('ENERGY', 400.0)) for _ in range(self.n)]
        self.uav_paths = [[list(pos)] for pos in self.uav]

        self.explored_map = np.zeros((self.mapx, self.mapy), dtype=np.int8)
        self.explored_maps = [np.zeros((self.mapx, self.mapy), dtype=np.int8) for _ in range(self.n)]
        self.prev_explored_maps = [np.zeros((self.mapx, self.mapy), dtype=np.int8) for _ in range(self.n)]

        self.prev_explored_count = 0
        self.total_reward = 0
        self.t = 0

        return self.get_state()  # 返回初始状态

    def get_state(self):
        """返回所有无人机的状态"""
        states = []
        for i in range(self.n):
            one_hot = np.zeros(self.n, dtype=np.float32)
            one_hot[i] = 1.0  # 当前智能体的one-hot表示
            state = {
                'uav_position': self.uav[i],
                'energy': self.energy[i],
                'explored_map': self.explored_map.copy(),
                'agent_id': one_hot,
            }
            states.append(state)
        return states

    #15：00之后加入
    def calculate_reward(self, move_distances, collisions, proximity_penalties=None):
        """
        计算每个无人机的奖励，包括探索奖励、碰撞惩罚、重叠惩罚等。
        额外支持：proximity_penalties（半径门控 + 平滑惩罚）
        
        参数：
        - move_distances: 每个无人机移动的距离
        - collisions: 每个无人机是否发生碰撞
        - proximity_penalties: ndarray/list，逐智能体的靠近惩罚（若为 None 则视为 0）
        
        返回：
        - rewards: 每个无人机的奖励
        - reward_components: 奖励组成部分的详细信息
        """
        #15：00之后加入
        if proximity_penalties is None:
            proximity_penalties = np.zeros(self.n, dtype=np.float32)

        rewards = []
        reward_components = []
        explored_area = np.sum(self.explored_map)  # 当前已探索区域
        total_area = self.mapx * self.mapy  # 地图总面积
        explored_ratio = explored_area / total_area  # 探索比例

        delta_coverage = explored_area - self.prev_explored_count  # 本轮探索新增的区域
        self.prev_explored_count = explored_area

        # ====== 改动C：共享覆盖率增量奖励 + 温和惩罚日程 ======
        penalty_scale = min(1.0, float(self.t) / max(1, self.penalty_warmup_steps))  # 随时间增长的惩罚强度

        # 权重设定
        collision_penalty_weight = 0.3 * penalty_scale
        overlap_penalty_weight = 0.1 * penalty_scale
        exploration_reward_weight = 0.5
        global_coverage_reward_weight = 0.25  # 共享给所有agent

        # 碰撞惩罚
        collision_penalties = np.array(collisions, dtype=np.float32) * collision_penalty_weight

        # 重叠惩罚（智能体之间的距离过近会产生重叠惩罚）
        overlap_penalties = np.zeros(self.n, dtype=np.float32)
        min_distance_threshold = 2.0
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    dist_ij = np.linalg.norm(np.array(self.uav[i]) - np.array(self.uav[j])) # 计算两架无人机的距离
                    if dist_ij < min_distance_threshold:
                        overlap_penalties[i] += overlap_penalty_weight * (min_distance_threshold - dist_ij)

        # 探索奖励（个体）
        exploration_rewards = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            delta_explored = np.sum(np.clip(self.explored_maps[i] - self.prev_explored_maps[i], 0, 1))
            exploration_rewards[i] = exploration_reward_weight * float(delta_explored)

        # 共享覆盖率增量奖励
        shared = global_coverage_reward_weight * float(delta_coverage) / max(1, self.n)

        # 计算每个智能体的总奖励
        for i in range(self.n):
            #15：00之后加入
            total_reward = (
                exploration_rewards[i]
                + shared
                - collision_penalties[i]
                - overlap_penalties[i]
                - float(proximity_penalties[i])   # 新增：靠近障碍惩罚
            )
            rewards.append(float(total_reward))
            reward_components.append({
                'explore_reward': float(exploration_rewards[i]),
                'shared_coverage': float(shared),
                'collision_penalty': float(collision_penalties[i]),
                'overlap_penalty': float(overlap_penalties[i]),
                #15：00之后加入
                'proximity_penalty': float(proximity_penalties[i]),
                'total': float(total_reward)
            })

        return rewards, reward_components

    def step(self, actions):
        """
        根据智能体的动作更新环境状态，计算奖励，并检查是否达到终止条件（能量耗尽）。
        
        参数：
        - actions: 每个智能体的动作（方向和距离）
        
        返回：
        - next_state: 环境的下一个状态
        - rewards: 每个智能体的奖励
        - done: 是否结束（所有智能体的能量都耗尽时）
        - info: 额外的信息（如碰撞、奖励细节等）
        """
        collisions = [0 for _ in range(self.n)]  # 碰撞情况
        newly_explored = 0  # 新探索的区域数
        move_distances = [0.0 for _ in range(self.n)]  # 每个智能体的移动距离

        new_positions = []
        for i, action in enumerate(actions):
            dx, dy, dist = action
            norm = np.sqrt(dx ** 2 + dy ** 2)
            if norm > 0:
                dx /= norm
                dy /= norm
            dist = np.clip(dist, 0.1, 1.0)

            # ====== 改动A：使用 step_scale + energy_cost_per_unit ======
            move_distance = float(dist) * self.step_scale
            available_units = max(0.0, float(self.energy[i]) / self.energy_cost_per_unit)  # 当前可用单位位移
            actual_distance = max(0.0, min(move_distance, available_units))#看能量是否够要移动的距离，取小的那个
            #计算新坐标
            new_x = self.uav[i][0] + dx * actual_distance
            new_y = self.uav[i][1] + dy * actual_distance
            #边界限制
            new_x = float(np.clip(new_x, 0, self.mapx - 1))
            new_y = float(np.clip(new_y, 0, self.mapy - 1))

            x_int, y_int = int(round(new_x)), int(round(new_y))
            if self.mapob[y_int, x_int] == self.OB:  # 检查是否撞到障碍物
                new_pos = self.uav[i]
                collisions[i] += 1
                actual_distance = 0.0  # 碰到障碍不移动也不扣能量
            else:
                new_pos = [new_x, new_y]

            new_positions.append(new_pos)
            move_distances[i] = actual_distance

        # 更新无人机状态和能量消耗
        for i in range(self.n):
            self.uav[i] = new_positions[i]
            self.uav_paths[i].append(list(new_positions[i]))
            # 扣除能量
            self.energy[i] = max(0.0, float(self.energy[i]) - move_distances[i] * self.energy_cost_per_unit)

            x_int, y_int = int(round(new_positions[i][0])), int(round(new_positions[i][1]))
            radius = self.V.get('MARK_RADIUS', 2)  # 扩大探索范围（原为1）
            for dx_ in range(-radius, radius + 1):
                for dy_ in range(-radius, radius + 1):
                    nx, ny = x_int + dx_, y_int + dy_
                    if 0 <= nx < self.mapx and 0 <= ny < self.mapy:
                        if self.mapob[ny, nx] != self.OB:
                            if self.explored_map[ny, nx] == 0:
                                self.explored_map[ny, nx] = 1
                                newly_explored += 1
                            if self.explored_maps[i][ny, nx] == 0:
                                self.explored_maps[i][ny, nx] = 1

        #15：00之后加入
        # ==== 避障：半径门控 + 平滑惩罚（无需距离场，局部窗口 O(R^2)）====
        R_sense = float(self.V.get('OB_PROX_RANGE', self.V.get('MARK_RADIUS', 2)))
        w_prox  = float(self.V.get('OB_PROX_WEIGHT', 0.2))
        shape   = self.V.get('OB_PROX_SHAPE', 'exp')      # 'exp' / 'linear' / 'quadratic'
        sigma   = float(self.V.get('OB_PROX_SIGMA', max(1.0, R_sense/2.0)))
        skip_on_collision = bool(self.V.get('OB_PROX_SKIP_ON_COLLISION', True))

        # 与其余惩罚保持一致的“热身”缩放
        penalty_scale = min(1.0, float(self.t) / max(1, self.penalty_warmup_steps))
        w_prox *= penalty_scale

        proximity_penalties = np.zeros(self.n, dtype=np.float32)

        # 局部窗口里搜索最近障碍距离（单位：格）
        def nearest_obst_dist_window(xi: int, yi: int, R: int) -> float:
            dmin = float('inf')
            for dx_ in range(-R, R+1):
                for dy_ in range(-R, R+1):
                    nx, ny = xi + dx_, yi + dy_
                    if 0 <= nx < self.mapx and 0 <= ny < self.mapy and self.mapob[ny, nx] == self.OB:
                        d = (dx_**2 + dy_**2) ** 0.5  # 欧氏距离
                        if d < dmin:
                            dmin = d
            return dmin if dmin < float('inf') else (R + 1.0)

        R_int = int(np.ceil(R_sense))
        for i in range(self.n):
            if skip_on_collision and collisions[i] > 0:
                continue
            xi = int(round(new_positions[i][0]))
            yi = int(round(new_positions[i][1]))
            d = nearest_obst_dist_window(xi, yi, R_int)

            # —— 半径门控：仅在感知半径内触发
            if d < R_sense:
                if shape == 'exp':
                    # Nav2 膨胀层同源：指数衰减，近处陡、远处缓
                    ratio = float(np.exp(-max(d, 0.0) / max(1e-6, sigma)))
                else:
                    p = 2 if shape == 'quadratic' else 1
                    ratio = float(((R_sense - d) / max(1e-6, R_sense)) ** p)
                proximity_penalties[i] = w_prox * ratio

        #15：00之后加入
        # 将靠近惩罚作为额外负项并入奖励计算
        reward, reward_components = self.calculate_reward(move_distances, collisions, proximity_penalties=proximity_penalties)

        self.total_reward += sum(reward)
        self.reward_log.append(reward)
        self.energy_log.append(sum(self.energy) / self.n)
        self.collision_log.append(collisions)
        self.visited_log.append(0)

        # 计算当前探索比例
        explored_area = np.sum(self.explored_map)
        total_area = self.mapx * self.mapy
        explored_ratio = explored_area / total_area
        self.explored_ratio_log.append(explored_ratio)

        for i in range(self.n):
            self.prev_explored_maps[i] = np.copy(self.explored_maps[i])

        done = all(e <= 0 for e in self.energy)  # 如果所有无人机的能量都用完，结束游戏

        info = {
            'collisions': collisions,
            'total_reward': self.total_reward,
            'newly_explored': newly_explored,
            'explored_ratio': explored_ratio,
            'reward_components': reward_components
        }

        # 增加一步数（用于调整惩罚）
        self.t += 1

        return self.get_state(), reward, done, info

    def render(self, save_path=None):
        """
        渲染当前环境状态，显示地图、无人机的位置和路径。
        
        参数：
        - save_path: 可选的保存路径，如果提供则将图像保存为文件
        """
        map_data = np.zeros((self.mapx, self.mapy))
        map_data[self.mapob == self.OB] = 0
        map_data[self.mapob != self.OB] = 1

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(map_data, cmap='gray', origin='lower')

        for i in range(self.n):
            ax.scatter(self.uav[i][0], self.uav[i][1], s=100, label=f"UAV {i}")

        for i in range(self.n):
            if len(self.uav_paths[i]) > 1:
                uav_x, uav_y = zip(*self.uav_paths[i])
                ax.plot(uav_x, uav_y, linewidth=2, label=f"UAV {i} Path")

        ax.set_title("UAV Environment Map")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=100)
        plt.show()

    def save_logs_to_csv(self, file_path="env_logs.csv"):
        """
        将日志保存为CSV文件，便于后续分析和可视化。
        
        参数：
        - file_path: 保存日志的文件路径
        """
        def _sum_list(x): 
            return float(np.sum(x)) if isinstance(x, (list, np.ndarray)) else float(x)
        def _mean_list(x): 
            return float(np.mean(x)) if isinstance(x, (list, np.ndarray)) else float(x)

        data = {
            "step": list(range(len(self.reward_log))),
            "reward_sum": [ _sum_list(r) for r in self.reward_log ],
            "reward_mean": [ _mean_list(r) for r in self.reward_log ],
            "energy_mean": self.energy_log,
            "collisions_sum": [ _sum_list(c) for c in self.collision_log ],
            "visited_points": self.visited_log,
            "explored_ratio": self.explored_ratio_log,
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        print(f"✅ 日志已保存至: {file_path}")
'''