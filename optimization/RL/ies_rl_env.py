import numpy as np


class IES_14Node_Env:
    """
    14节点 变流量-变水温 (VF-VT) 综合能源多智能体强化学习环境
    包含：非线性水力压降、水泵功耗、换热站传热极限、FDM管网时滞、建筑RC模型
    """

    def __init__(self):
        # ---------------------------------------------------------
        # 1. 仿真基础设置 (15分钟级分辨率)
        # ---------------------------------------------------------
        self.dt_min = 15
        self.dt_sec = self.dt_min * 60.0
        self.T_total = int(24 * 60 / self.dt_min)  # 96 steps
        self.current_step = 0

        # 物理常数
        self.rho = 1000.0  # kg/m3
        self.cp = 4186.0  # J/(kg.K)
        self.cp_kw = 4.186  # kJ/(kg.K)
        self.T_ground = 10.0
        self.lambda_f = 0.02  # 管道沿程摩擦系数
        self.eta_pump = 0.75  # 水泵综合效率

        # ---------------------------------------------------------
        # 2. 设备参数 (全集中在 N1 节点)
        # ---------------------------------------------------------
        self.Cap_CHP_E = 2000.0;
        self.Alpha_CHP = 1.2
        self.Cap_FC_E = 1500.0;
        self.Alpha_FC = 1.0;
        self.Eta_FC = 0.50
        self.Cap_EB_Q = 5000.0;
        self.Eta_EB = 0.98

        # ---------------------------------------------------------
        # 3. 建筑与换热站参数 (N6, N11, N14)
        # ---------------------------------------------------------
        self.bldgs = {
            'N6': {'R': 0.015, 'C': 4000.0, 'm_max': 10.0, 'K_eq': 150.0},
            'N11': {'R': 0.012, 'C': 5000.0, 'm_max': 12.0, 'K_eq': 180.0},
            'N14': {'R': 0.018, 'C': 4500.0, 'm_max': 15.0, 'K_eq': 200.0}
        }
        self.dT_app = 3.0  # 换热器最小趋近温差

        # ---------------------------------------------------------
        # 4. 网络拓扑与管道初始化 (14节点新结构)
        # ---------------------------------------------------------
        # 拓扑: 1-2, 2-3, 2-4, 2-5, 3-8, 4-6(荷), 8-7, 7-5, 5-9, 9-10, 9-12, 10-11(荷), 12-13, 13-14(荷)
        self.nodes = [f'N{i}' for i in range(1, 15)]
        self.pipes_info = [
            {'id': 'P1', 'fr': 'N1', 'to': 'N2', 'L': 1000, 'D': 0.25},
            {'id': 'P2', 'fr': 'N2', 'to': 'N3', 'L': 1500, 'D': 0.15},
            {'id': 'P3', 'fr': 'N2', 'to': 'N4', 'L': 800, 'D': 0.15},
            {'id': 'P4', 'fr': 'N2', 'to': 'N5', 'L': 2000, 'D': 0.20},
            {'id': 'P5', 'fr': 'N3', 'to': 'N8', 'L': 1000, 'D': 0.15},
            {'id': 'P6', 'fr': 'N4', 'to': 'N6', 'L': 500, 'D': 0.15},
            {'id': 'P7', 'fr': 'N8', 'to': 'N7', 'L': 1000, 'D': 0.15},
            {'id': 'P8', 'fr': 'N7', 'to': 'N5', 'L': 1000, 'D': 0.15},
            {'id': 'P9', 'fr': 'N5', 'to': 'N9', 'L': 2500, 'D': 0.25},
            {'id': 'P10', 'fr': 'N9', 'to': 'N10', 'L': 800, 'D': 0.15},
            {'id': 'P11', 'fr': 'N9', 'to': 'N12', 'L': 1000, 'D': 0.20},
            {'id': 'P12', 'fr': 'N10', 'to': 'N11', 'L': 500, 'D': 0.15},
            {'id': 'P13', 'fr': 'N12', 'to': 'N13', 'L': 800, 'D': 0.20},
            {'id': 'P14', 'fr': 'N13', 'to': 'N14', 'L': 500, 'D': 0.20}
        ]

        # 预计算管道物理常数，并基于最大流速(假设2m/s)划分最大网格数，保证变流速时Co<=1
        self.pipes = {}
        for p in self.pipes_info:
            A = np.pi * (p['D'] / 2) ** 2
            # 水力阻抗 S_p = (8 * lambda * L) / (pi^2 * D^5 * rho)
            S_res = (8 * self.lambda_f * p['L']) / ((np.pi ** 2) * (p['D'] ** 5) * self.rho)
            # 热损系数基准
            Re_thermal = (1 / (2 * np.pi * 1.5)) * np.log(4 * 1.5 / p['D'])
            mu_base = self.dt_sec / (self.rho * A * self.cp * Re_thermal)

            # 最大可能流速预估网格数
            N_seg = max(1, int(np.ceil(p['L'] / (2.0 * self.dt_sec))))

            self.pipes[p['id']] = {
                'fr': p['fr'], 'to': p['to'], 'L': p['L'], 'A': A, 'S': S_res,
                'N': N_seg, 'dx': p['L'] / N_seg, 'mu_total': mu_base,
                'Ts': np.full(N_seg + 1, 60.0),  # 供水状态
                'Tr': np.full(N_seg + 1, 40.0)  # 回水状态
            }

        # 节点状态字典
        self.Ts_node = {n: 60.0 for n in self.nodes}
        self.Tr_node = {n: 40.0 for n in self.nodes}
        self.Tin = {n: 20.0 for n in self.bldgs.keys()}

    def get_environment_conditions(self, step):
        """生成时序外界条件"""
        hour = step * self.dt_min / 60.0
        T_amb = -10 + 6 * np.sin((hour - 8) * np.pi / 12)
        Price = 0.02 if hour < 7 else (0.30 if 18 <= hour < 21 else 0.15)
        Wind = max(0, 1800 + 1200 * np.cos((hour - 3) * np.pi / 12))
        Elec_Load = 1500 + 800 * np.exp(-0.1 * (hour - 10) ** 2) + 1000 * np.exp(-0.1 * (hour - 19) ** 2)
        return T_amb, Price, Wind, Elec_Load

    def reset(self):
        """环境重置"""
        self.current_step = 0
        for p in self.pipes.values():
            p['Ts'].fill(65.0)
            p['Tr'].fill(40.0)
        for n in self.nodes:
            self.Ts_node[n], self.Tr_node[n] = 65.0, 40.0
        for b in self.Tin:
            self.Tin[b] = 20.0  # 强制冷启动
        return self._get_obs()

    def step(self, actions):
        """
        环境步进交互 (核心动力学引擎)
        actions: dict包含 {'src': [a_chp, a_fc, a_eb], 'valves':[v_6, v_11, v_14]}
        取值范围皆假设被算法映射到了 [0, 1]
        """
        # ==========================================
        # 1. 动作解析与物理映射
        # ==========================================
        a_src = np.clip(actions['src'], 0.0, 1.0)
        # 阀门开度限制在 [0.1, 1.0] 防止除零和死水断流
        a_valves = np.clip(actions['valves'], 0.1, 1.0)

        P_chp = a_src[0] * self.Cap_CHP_E
        P_fc = a_src[1] * self.Cap_FC_E
        Q_eb = a_src[2] * self.Cap_EB_Q
        P_eb = Q_eb / self.Eta_EB

        m_load = {
            'N6': a_valves[0] * self.bldgs['N6']['m_max'],
            'N11': a_valves[1] * self.bldgs['N11']['m_max'],
            'N14': a_valves[2] * self.bldgs['N14']['m_max']
        }

        # ==========================================
        # 2. 变流量非线性水力模型 (Hydraulics & Pump)
        # ==========================================
        m, P_pump = self._solve_hydraulics(m_load)

        # ==========================================
        # 3. 源侧产热与 PHE 模型
        # ==========================================
        Q_src_total = P_chp * self.Alpha_CHP + P_fc * self.Alpha_FC + Q_eb
        m_src = m['P1']

        # PHE 换热升温
        dT_src = Q_src_total / (self.cp_kw * m_src)
        self.Ts_node['N1'] = min(100.0, self.Tr_node['N1'] + dT_src)  # 物理上限截断

        # ==========================================
        # 4. FDM 管道与节点热力分布网络
        # ==========================================
        self._update_thermal_network(m)

        # ==========================================
        # 5. 荷侧 SHE 模型与建筑 RC
        # ==========================================
        T_amb, Price, Wind, Elec_Load = self.get_environment_conditions(self.current_step)
        comfort_penalty = 0.0

        for n, bldg in self.bldgs.items():

            old_temp = self.Tin[n]

            T_s_node = self.Ts_node[n]
            T_indoor = self.Tin[n]
            m_l = m_load[n]

            # --- 换热站等效热阻截断模型 (精细化核心) ---
            # 1. 潜在自然换热能力
            Q_pot = bldg['K_eq'] * max(0, T_s_node - T_indoor)
            # 2. 物理极限换热能力 (受限于流量和端差)
            Q_max = self.cp_kw * m_l * max(0, T_s_node - T_indoor - self.dT_app)
            # 3. 实际提取热量
            Q_sup = min(Q_pot, Q_max)

            # 反推回水温度 (反馈给热网)
            self.Tr_node[n] = T_s_node - Q_sup / (self.cp_kw * m_l)

            # RC 室内温度演进
            Q_loss = (T_indoor - T_amb) / bldg['R']
            self.Tin[n] += (self.dt_sec / 3600.0 / bldg['C']) * (Q_sup - Q_loss)

            # 舒适度惩罚 (抛物线型)
            if self.Tin[n] < 20.0:
                comfort_penalty = 50 + (20.0 - self.Tin[n]) ** 2
                comfort_penalty += comfort_penalty - 5.0 * (self.Tin[n] - old_temp)
            elif self.Tin[n] > 24.0:
                comfort_penalty += 50 + (self.Tin[n] - 24.0) ** 2
            else:
                comfort_penalty -= 5  # 存活奖励

        # ==========================================
        # 6. 计算联合奖励 (Reward Formulation)
        # ==========================================
        # 购电平衡
        P_buy = max(0, Elec_Load + P_eb + P_pump - P_chp - P_fc - Wind)

        # 经济成本
        cost_gas = (P_chp / 0.35 + P_fc / 0.5) * 0.04
        cost_elec = P_buy * Price
        cost_carbon = (P_buy * 0.85 + (P_chp / 0.35) * 0.20 + (P_fc / 0.5) * 0.05) * 0.02

        # 奖励组合 (成本为负)
        penalty_weight = 50.0 #原本为10
        raw_reward = - (cost_gas + cost_elec + cost_carbon + penalty_weight * comfort_penalty)
        reward = raw_reward / 1000.0  # 奖励缩放，利于RL收敛

        self.current_step += 1
        done = self.current_step >= self.T_total

        # 将一些监控数据传出，方便 Debug 和绘图
        info = {'P_pump_kW': P_pump, 'Q_src': Q_src_total, 'Tin': self.Tin.copy(), 'Ts_N14': self.Ts_node['N14']}

        return self._get_obs(), reward, done, info

    def _solve_hydraulics(self, m_load):
        """
        基于14节点拓扑自下而上的水力流量推导与非线性水泵电耗计算
        """
        m = {}
        # 负荷分支
        m['P6'] = m_load['N6'];
        m['P12'] = m_load['N11'];
        m['P14'] = m_load['N14']
        m['P13'] = m['P14'];
        m['P11'] = m['P13']
        m['P10'] = m['P12']
        m['P9'] = m['P10'] + m['P11']

        # 环路并联拆分 (简化: 根据支路热阻经验分配比)
        # N5 接收来自 P4(N2直接) 和 P8(经由N3,N8,N7)。假设 60% 走主路P4
        m['P4'] = 0.6 * m['P9']
        m['P8'] = 0.4 * m['P9']

        m['P7'] = m['P8'];
        m['P5'] = m['P7']
        m['P2'] = m['P5'];
        m['P3'] = m['P6']
        m['P1'] = m['P2'] + m['P3'] + m['P4']

        # ---------------------------------------------------------
        # 计算水泵功耗 (三次幂惩罚源)
        # 寻找最不利压降路径。由于树状+简化的环，近似计算总沿程阻力
        # P_pump = Delta P * Vol_flow / eta
        # ---------------------------------------------------------
        max_dp = 0.0
        # 路径1: 1->2->4->6
        dp1 = sum(self.pipes[pid]['S'] * m[pid] ** 2 for pid in ['P1', 'P3', 'P6'])
        # 路径2: 1->2->5->9->10->11
        dp2 = sum(self.pipes[pid]['S'] * m[pid] ** 2 for pid in ['P1', 'P4', 'P9', 'P10', 'P12'])
        # 路径3: 1->2->3->8->7->5->9->12->13->14 (最长路径)
        dp3 = sum(
            self.pipes[pid]['S'] * m[pid] ** 2 for pid in ['P1', 'P2', 'P5', 'P7', 'P8', 'P9', 'P11', 'P13', 'P14'])

        max_dp = max(dp1, dp2, dp3)
        # P_pump = max_dp (Pa) * m (kg/s) / (rho * eta).  W 转 kW 除以 1000
        P_pump_kw = (max_dp * m['P1']) / (self.rho * self.eta_pump) / 1000.0

        return m, P_pump_kw

    def _update_thermal_network(self, m_pipes):
        """使用迎风差分格式与节点混合定律更新全网温度场"""
        # 1. 供水正向推进
        self.pipes['P1']['Ts'][0] = self.Ts_node['N1']

        # 定义供水拓扑推进顺序 (上游 -> 下游)
        supply_seq = ['P1', 'N2', 'P2', 'P3', 'P4', 'N3', 'N4', 'P5', 'P6', 'N8', 'P7',
                      'N7', 'P8', 'N5', 'P9', 'N9', 'P10', 'P11', 'N10', 'N12', 'P12', 'P13', 'N13', 'P14']

        for item in supply_seq:
            if item.startswith('P'):
                self._fdm_step(item, m_pipes[item], 'Supply')
            else:
                self._mix_nodes(item, m_pipes, 'Supply')

        # 2. 回水反向推进 (荷 -> 源)
        # N6, N11, N14 的回水温度已在 SHE 模型中更新至对应管道末端
        return_seq = ['P14', 'N13', 'P13', 'P12', 'N12', 'N10', 'P11', 'P10', 'N9', 'P9',
                      'N5', 'P8', 'P4', 'N7', 'P7', 'N8', 'P5', 'N3', 'P2', 'N6', 'P6', 'N4', 'P3', 'N2', 'P1']

        for item in return_seq:
            if item.startswith('P'):
                self._fdm_step(item, m_pipes[item], 'Return')
            else:
                self._mix_nodes(item, m_pipes, 'Return')

    def _fdm_step(self, pid, m, direction):
        """单管差分计算：动态库朗数"""
        p = self.pipes[pid]
        v = m / (self.rho * p['A'])
        # 动态 Co
        Co = min(1.0, v * self.dt_sec / p['dx'])
        mu = p['mu_total'] / p['N']

        arr = p['Ts'] if direction == 'Supply' else p['Tr']
        new_arr = np.copy(arr)

        for k in range(1, p['N'] + 1):
            new_arr[k] = (1 - Co - mu) * arr[k] + Co * arr[k - 1] + mu * self.T_ground

        if direction == 'Supply':
            self.pipes[pid]['Ts'] = new_arr
        else:
            self.pipes[pid]['Tr'] = new_arr

    def _mix_nodes(self, nid, m_pipes, direction):
        """硬编码的 14节点拓扑混合 (此处仅展示核心交叉节点，直通节点直接赋值)"""
        if direction == 'Supply':
            if nid == 'N2':
                self.Ts_node['N2'] = self.pipes['P1']['Ts'][-1]
                self.pipes['P2']['Ts'][0] = self.Ts_node['N2']
                self.pipes['P3']['Ts'][0] = self.Ts_node['N2']
                self.pipes['P4']['Ts'][0] = self.Ts_node['N2']
            elif nid == 'N5':
                # N5 混合 P4 和 P8
                m_tot = m_pipes['P4'] + m_pipes['P8']
                enth = m_pipes['P4'] * self.pipes['P4']['Ts'][-1] + m_pipes['P8'] * self.pipes['P8']['Ts'][-1]
                self.Ts_node['N5'] = enth / (m_tot + 1e-6)
                self.pipes['P9']['Ts'][0] = self.Ts_node['N5']
            # ... 其他直通节点略写为等值赋值
            elif nid == 'N9':
                self.Ts_node['N9'] = self.pipes['P9']['Ts'][-1]
                self.pipes['P10']['Ts'][0] = self.Ts_node['N9']
                self.pipes['P11']['Ts'][0] = self.Ts_node['N9']
            else:
                in_p = {'N3': 'P2', 'N4': 'P3', 'N8': 'P5', 'N7': 'P7', 'N10': 'P10', 'N12': 'P11', 'N13': 'P13'}.get(
                    nid)
                out_p = {'N3': 'P5', 'N4': 'P6', 'N8': 'P7', 'N7': 'P8', 'N10': 'P12', 'N12': 'P13', 'N13': 'P14'}.get(
                    nid)
                if in_p and out_p:
                    self.Ts_node[nid] = self.pipes[in_p]['Ts'][-1]
                    self.pipes[out_p]['Ts'][0] = self.Ts_node[nid]

        else:  # Return 混合 (与Supply完全对称但反向，例如 N9 混合 P10, P11回水)
            if nid == 'N9':
                m_tot = m_pipes['P10'] + m_pipes['P11']
                enth = m_pipes['P10'] * self.pipes['P10']['Tr'][-1] + m_pipes['P11'] * self.pipes['P11']['Tr'][-1]
                self.Tr_node['N9'] = enth / (m_tot + 1e-6)
                self.pipes['P9']['Tr'][-1] = self.Tr_node['N9']
            elif nid == 'N2':
                m_tot = m_pipes['P2'] + m_pipes['P3'] + m_pipes['P4']
                enth = m_pipes['P2'] * self.pipes['P2']['Tr'][-1] + m_pipes['P3'] * self.pipes['P3']['Tr'][-1] + \
                       m_pipes['P4'] * self.pipes['P4']['Tr'][-1]
                self.Tr_node['N2'] = enth / (m_tot + 1e-6)
                self.pipes['P1']['Tr'][-1] = self.Tr_node['N2']
            # ... 其余节点反向传值
            elif nid == 'N5':
                self.Tr_node['N5'] = self.pipes['P9']['Tr'][-1]
                self.pipes['P4']['Tr'][-1] = self.Tr_node['N5']
                self.pipes['P8']['Tr'][-1] = self.Tr_node['N5']
            else:
                in_p_ret = {'N3': 'P5', 'N4': 'P6', 'N8': 'P7', 'N7': 'P8', 'N10': 'P12', 'N12': 'P13',
                            'N13': 'P14'}.get(nid)
                out_p_ret = {'N3': 'P2', 'N4': 'P3', 'N8': 'P5', 'N7': 'P7', 'N10': 'P10', 'N12': 'P11',
                             'N13': 'P13'}.get(nid)
                if in_p_ret and out_p_ret:
                    self.Tr_node[nid] = self.pipes[in_p_ret]['Tr'][-1]
                    self.pipes[out_p_ret]['Tr'][-1] = self.Tr_node[nid]

    def _get_obs(self):
        """智能体观测状态"""
        T_amb, Price, Wind, Load = self.get_environment_conditions(self.current_step)
        obs = [
            self.current_step / self.T_total, Price / 0.30, Wind / 3000.0,
            self.Tin['N6'] / 24.0, self.Tin['N11'] / 24.0, self.Tin['N14'] / 24.0,
            self.Ts_node['N6'] / 100.0, self.Ts_node['N11'] / 100.0, self.Ts_node['N14'] / 100.0
        ]
        return np.array(obs, dtype=np.float32)