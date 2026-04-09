import numpy as np


class IES_RL_Environment:
    def __init__(self):
        # 1. 仿真时间与步长设置
        self.dt_min = 15
        self.dt_sec = self.dt_min * 60.0
        self.T_total = int(24 * 60 / self.dt_min)
        self.current_step = 0

        # 2. 设备参数
        self.Cap_CHP = 20000.0  # kW (电)
        self.Alpha_CHP = 1.2
        self.Cap_FC = 15000.0  # kW (电)
        self.Alpha_FC = 1.0
        self.Cap_EB = 50000.0  # kW (热)

        # 3. 建筑参数 (N6, N11, N14)
        self.bldgs = {
            'N6': {'R': 0.015, 'C': 4000.0, 'max_flow': 15.0},
            'N11': {'R': 0.012, 'C': 5000.0, 'max_flow': 15.0},
            'N14': {'R': 0.018, 'C': 4500.0, 'max_flow': 15.0}
        }

        # 4. 管道拓扑定义 (VF-VT模式：保留长度和管径，流量随动作动态变化)
        # 拓扑: 1-2, 2-3, 2-4, 2-5, 3-8, 4-6(Load), 8-7, 15(EB)-7, 7-5, 5-9, 9-10, 9-12, 10-11(Load), 12-13, 13-14(Load)
        self.pipes_info = [
            {'id': 'P1', 'fr': 'N1', 'to': 'N2', 'L': 1000, 'D': 0.25},
            {'id': 'P2', 'fr': 'N2', 'to': 'N3', 'L': 1500, 'D': 0.15},
            {'id': 'P3', 'fr': 'N2', 'to': 'N4', 'L': 800, 'D': 0.15},
            {'id': 'P4', 'fr': 'N2', 'to': 'N5', 'L': 2000, 'D': 0.20},
            {'id': 'P5', 'fr': 'N3', 'to': 'N8', 'L': 1000, 'D': 0.15},
            {'id': 'P6', 'fr': 'N4', 'to': 'N6', 'L': 500, 'D': 0.15},  # to Load N6
            {'id': 'P7', 'fr': 'N8', 'to': 'N7', 'L': 1000, 'D': 0.15},
            {'id': 'P15', 'fr': 'N15', 'to': 'N7', 'L': 1000, 'D': 0.20},  # New Branch from EB
            {'id': 'P8', 'fr': 'N7', 'to': 'N5', 'L': 1000, 'D': 0.15},
            {'id': 'P9', 'fr': 'N5', 'to': 'N9', 'L': 2500, 'D': 0.25},
            {'id': 'P10', 'fr': 'N9', 'to': 'N10', 'L': 800, 'D': 0.15},
            {'id': 'P11', 'fr': 'N9', 'to': 'N12', 'L': 1000, 'D': 0.20},
            {'id': 'P12', 'fr': 'N10', 'to': 'N11', 'L': 500, 'D': 0.15},  # to Load N11
            {'id': 'P13', 'fr': 'N12', 'to': 'N13', 'L': 800, 'D': 0.20},
            {'id': 'P14', 'fr': 'N13', 'to': 'N14', 'L': 500, 'D': 0.20}  # to Load N14
        ]

        # 预计算固定物理参数，并在最大流速下划分布局，保证变流量时 Co 永远 <= 1
        self.pipes = {}
        for p in self.pipes_info:
            Area = np.pi * (p['D'] / 2) ** 2
            # 假设该管道可能出现的最大流速上限为 2.0 m/s
            v_max = 2.0
            N_seg = max(1, int(np.ceil(p['L'] / (v_max * self.dt_sec))))
            self.pipes[p['id']] = {
                'Area': Area, 'L': p['L'], 'N': N_seg, 'dx': p['L'] / N_seg,
                'Ts': np.full(N_seg + 1, 60.0),  # 供水网格温度初始化
                'Tr': np.full(N_seg + 1, 40.0)  # 回水网格温度初始化
            }

        # 节点温度字典
        self.nodes = [f'N{i}' for i in range(1, 16)]
        self.Ts_node = {n: 60.0 for n in self.nodes}
        self.Tr_node = {n: 40.0 for n in self.nodes}
        self.Tin = {n: 21.0 for n in self.bldgs.keys()}

    def get_time_series(self, t_idx):
        """获取当前环境数据 (室外温度, 电价, 风电, 电负荷)"""
        hour = (t_idx * self.dt_min) / 60.0
        T_amb = -10 + 6 * np.sin((hour - 8) * np.pi / 12)
        Price = 0.02 if hour < 7 else (0.30 if 18 <= hour < 21 else 0.15)
        Wind = max(0, 1800 + 1200 * np.cos((hour - 3) * np.pi / 12))
        Load = 1500 + 800 * np.exp(-0.1 * (hour - 10) ** 2) + 1000 * np.exp(-0.1 * (hour - 19) ** 2)
        return T_amb, Price, Wind, Load

    def reset(self):
        """强化学习 Reset 接口"""
        self.current_step = 0
        for p in self.pipes.values():
            p['Ts'].fill(80.0)
            p['Tr'].fill(40.0)
        for n in self.nodes:
            self.Ts_node[n], self.Tr_node[n] = 60.0, 40.0
        for b in self.Tin:
            self.Tin[b] = 23.0
        return self._get_obs()

    def step(self, actions):
        """强化学习 Step 接口 (执行动作，计算物理引擎，返回奖励)"""
        # 1. 解析动作 (映射到物理区间)
        # 动作空间假定为 [-1, 1]
        act_N1 = np.clip(actions['N1'], -1, 1)  # [CHP, FC]
        act_N15 = np.clip(actions['N15'], -1, 1)  # [EB]
        act_valves = np.clip(actions['Valves'], -1, 1)  # [Valve6, Valve11, Valve14]
        # # 强制阀门全开（初期调试用）
        # act_valves = np.array([1.0, 1.0, 1.0])

        P_chp = (act_N1[0] + 1) / 2 * self.Cap_CHP
        P_fc = (act_N1[1] + 1) / 2 * self.Cap_FC
        Q_eb = (act_N15[0] + 1) / 2 * self.Cap_EB

        # 阀门开度转流量
        m_loads = {
            'N6': (act_valves[0] + 1) / 2 * self.bldgs['N6']['max_flow'],
            'N11': (act_valves[1] + 1) / 2 * self.bldgs['N11']['max_flow'],
            'N14': (act_valves[2] + 1) / 2 * self.bldgs['N14']['max_flow']
        }

        # 2. 水力模块：根据阀门流量，反推全网流量分布 (KCL)
        m_pipes = self._solve_hydraulics(m_loads, Q_eb)

        # 3. 能量转换与源侧温度更新
        Q_chp = P_chp * self.Alpha_CHP
        Q_fc = P_fc * self.Alpha_FC

        # N1 节点供水升温
        if m_pipes['P1'] > 0:
            dT_N1 = (Q_chp + Q_fc) / (4.186 * m_pipes['P1'])
            self.Ts_node['N1'] = min(100.0, self.Tr_node['N1'] + dT_N1)

        # N15 节点供水升温 (EB)
        if m_pipes['P15'] > 0:
            dT_N15 = Q_eb / (4.186 * m_pipes['P15'])
            self.Ts_node['N15'] = min(100.0, self.Tr_node['N15'] + dT_N15)

        # 4. FDM 管网温度场更新 (供水 & 回水)
        self._update_thermal_network(m_pipes)

        # 5. 建筑热力学模块 (RC 模型更新)
        T_amb, Price, Wind, Elec_Load = self.get_time_series(self.current_step)
        comfort_penalty = 0.0

        for n, bldg in self.bldgs.items():

            # 【关键 1】：在更新前，先保存当前的室温作为 old_Tin
            old_temp = self.Tin[n]

            # 建筑换热量 (假设回水温度高于室温 3 度)
            Tr_bldg = max(self.Tin[n] + 3.0, self.Ts_node[n] - 20.0)  # 简化换热逻辑
            if self.Ts_node[n] > Tr_bldg:
                Q_sup = 4.186 * m_loads[n] * (self.Ts_node[n] - Tr_bldg)
            else:
                Q_sup = 0
                Tr_bldg = self.Ts_node[n]

            # 更新回水网络入口
            if n == 'N6':
                self.pipes['P6']['Tr'][-1] = Tr_bldg
            elif n == 'N11':
                self.pipes['P12']['Tr'][-1] = Tr_bldg
            elif n == 'N14':
                self.pipes['P14']['Tr'][-1] = Tr_bldg

            # RC 方程
            Q_loss = (self.Tin[n] - T_amb) / bldg['R']
            self.Tin[n] += (self.dt_sec / 3600.0 / bldg['C']) * (Q_sup - Q_loss)

            # 【关键 2】：计算温度的改善量 (新温度 - 旧温度)
            temp_improvement = self.Tin[n] - old_temp

            # 计算舒适度惩罚
            if self.Tin[n] < 20.0:
                # 惩罚绝对温差的平方
                base_penalty = (20.0 - self.Tin[n]) ** 2

                # 如果温度正在上升(improvement > 0)，就减免一部分惩罚；如果还在降温，就加重惩罚
                # 这里的 5.0 是引导系数，可以根据训练情况微调
                comfort_penalty += base_penalty - 5.0 * temp_improvement
            elif self.Tin[n] > 24.0:
                comfort_penalty += (self.Tin[n] - 24.0) ** 2
            else:
                # 只要在舒适区内，给予存活奖励！鼓励它保持在这里
                comfort_penalty -= 1.0

            # # 密集奖励塑形
            # if self.Tin[n] < 20.0:
            #     # 如果温度上升了，给予稍微的正向引导抵消部分惩罚
            #     temp_improvement = self.Tin[n] - old_Tin[n]  # old_Tin 是更新前保存的温度
            #     comfort_penalty += (20.0 - self.Tin[n]) ** 2 - 5.0 * temp_improvement
            # elif self.Tin[n] > 24.0:
            #     comfort_penalty += (self.Tin[n] - 24.0) ** 2
            # else:
            #     # 只要在舒适区内，给予存活奖励！鼓励它保持在这里
            #     comfort_penalty -= 1.0

        # 6. 计算经济奖励 (Reward)
        P_eb = Q_eb / 0.98
        P_buy = Elec_Load + P_eb - P_chp - P_fc - Wind
        if P_buy < 0: P_buy = 0  # 简化不考虑卖电

        cost_gas = (P_chp / 0.35 + P_fc / 0.50) * 0.04
        cost_elec = P_buy * Price
        cost_carbon = (P_buy * 0.85 + (P_chp / 0.35) * 0.20 + (P_fc / 0.5) * 0.05) * 0.02

        # # 核心：RL 奖励是负的成本
        # reward = - (cost_gas + cost_elec + cost_carbon + 100.0 * comfort_penalty)

        # 修改为：缩小惩罚倍数，并将整体奖励缩放到[-10, 0] 级别
        penalty_weight = 10.0  # 降低惩罚权重
        raw_reward = - (cost_gas + cost_elec + cost_carbon + penalty_weight * comfort_penalty)

        # [关键] 将 reward 除以 1000，防止梯度爆炸
        reward = raw_reward / 1000.0

        # 7. 步长前进
        self.current_step += 1
        done = self.current_step >= self.T_total

        return self._get_obs(), reward, done, {}

    def _solve_hydraulics(self, m_loads, Q_eb):
        """简化水力求解器：基于阀门设定，自下而上推导全网流量分布"""
        m = {}
        m['P6'] = m_loads['N6'];
        m['P12'] = m_loads['N11'];
        m['P14'] = m_loads['N14']
        m['P3'] = m['P6']
        m['P10'] = m['P12'];
        m['P13'] = m['P14'];
        m['P11'] = m['P13']
        m['P9'] = m['P10'] + m['P11']

        # 并联环路 (N2->N5) 和 (N2->N3...->N7->N5)
        # 为了避免水力迭代，假设自然分流比为 50%
        m['P4'] = 0.5 * m['P9']
        m['P8'] = 0.5 * m['P9']

        # N15 (EB) 流量根据加热功率自适应 (假设 EB 设计温差 40度)
        m['P15'] = Q_eb / (4.186 * 40.0) if Q_eb > 0 else 0.0
        m['P15'] = min(m['P15'], m['P8'])  # 防止倒流

        m['P7'] = m['P8'] - m['P15']
        m['P5'] = m['P7']
        m['P2'] = m['P5']
        m['P1'] = m['P2'] + m['P3'] + m['P4']
        return m

    def _update_thermal_network(self, m_pipes):
        """FDM 热力学引擎更新"""
        # 1. 供水温度计算与混合
        for pid in ['P1', 'P15']:  # 源支路
            self.pipes[pid]['Ts'][0] = self.Ts_node['N1'] if pid == 'P1' else self.Ts_node['N15']

        # 依次向下游计算 (显式前向推进)
        for pid in ['P1', 'P15', 'P2', 'P3', 'P4', 'P5', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14']:
            # 更新管内 FDM
            self._fdm_step(pid, m_pipes[pid], direction='Supply')
            # 根据拓扑更新节点混合 (此处略去冗长的 if-else，思路同 Gurobi 中的 enthalpy_sum)
            # 例如 N5 混合:
            # Ts_node['N5'] = (m_P4 * Ts_P4_out + m_P8 * Ts_P8_out) / (m_P4 + m_P8 + 1e-6)
            # ...

        # 2. 回水温度计算与混合 (同理，由负载端反向计算至源端)
        # ...

    def _fdm_step(self, pid, m, direction='Supply'):
        """单根管道的有限差分步进"""
        p = self.pipes[pid]
        v = m / (1000.0 * p['Area'])
        Co = min(1.0, v * self.dt_sec / p['dx'])  # 保证稳定性
        mu = 0.02 * self.dt_sec / 3600.0 / p['N']  # 经验热损

        arr = p['Ts'] if direction == 'Supply' else p['Tr']
        arr_new = np.copy(arr)

        # 迎风格式
        for k in range(1, p['N'] + 1):
            arr_new[k] = (1 - Co - mu) * arr[k] + Co * arr[k - 1] + mu * 10.0

        if direction == 'Supply':
            self.pipes[pid]['Ts'] = arr_new
        else:
            self.pipes[pid]['Tr'] = arr_new

    def _get_obs(self):
        """构建强化学习的状态观测向量"""
        T_amb, Price, Wind, Load = self.get_time_series(self.current_step)
        obs = [
            self.current_step / self.T_total,  # 归一化时间
            Price / 0.30,  # 归一化电价
            Wind / 3000.0,
            self.Tin['N6'] / 24.0, self.Tin['N11'] / 24.0, self.Tin['N14'] / 24.0,
            self.Ts_node['N6'] / 100.0, self.Ts_node['N11'] / 100.0  # 阀门前的来水温度感知
        ]
        return np.array(obs, dtype=np.float32)

