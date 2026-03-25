import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import config

# ==========================================
# 1. 拓扑解析与参数计算
# ==========================================
ts_data = config.get_time_series_data()
T = config.SIM_SETTINGS['T']
dt_sec = config.SIM_SETTINGS['dt_sec']
dt_h = config.SIM_SETTINGS['dt_hour']
t_ax = ts_data['t_hours']
T_amb, Price_Elec = ts_data['T_amb'], ts_data['Price_Elec']
Wind_Power, Elec_Load = ts_data['Wind_Power'], ts_data['Elec_Load']


def calc_pipe_params_fdm(cfg):
    phy = config.PHYSICS
    Re = (1 / (2 * np.pi * phy['Lambda_soil'])) * np.log(4 * phy['Depth'] / cfg['D'])
    Area = np.pi * (cfg['D'] / 2) ** 2
    mu_total = dt_sec / (phy['Rho'] * Area * phy['Cp'] * Re)
    v = cfg['m'] / (phy['Rho'] * Area)

    # [精度优化] 因为dt变小了，N会自动变大
    N_seg = max(1, int(np.floor(cfg['L'] / (v * dt_sec))))
    dx = cfg['L'] / N_seg
    Co = min(1.0, v * dt_sec / dx)
    return {'v': v, 'N': N_seg, 'Co': Co, 'mu': mu_total / N_seg, 'delay_h': cfg['L'] / v / 3600.0}


Pipes = {}
in_pipes = {n: [] for n in config.NODES}
out_pipes = {n: [] for n in config.NODES}

print(f"--- 仿真设置: 步长 {dt_sec / 60:.0f} 分钟, 共 {T} 步 ---")
for cfg in config.PIPE_CONFIGS:
    params = calc_pipe_params_fdm(cfg)
    Pipes[cfg['id']] = {**cfg, **params}
    in_pipes[cfg['to']].append(cfg['id'])
    out_pipes[cfg['fr']].append(cfg['id'])
    print(
        f"Pipe {cfg['id']} ({cfg['fr']}->{cfg['to']}): N={params['N']}, Co={params['Co']:.2f}, Delay={params['delay_h']:.2f}h")

# 获取节点真实流量守恒值
m_in = {n: sum(Pipes[p]['m'] for p in in_pipes[n]) for n in config.NODES}
m_out = {n: sum(Pipes[p]['m'] for p in out_pipes[n]) for n in config.NODES}

# ==========================================
# 2. Gurobi 模型构建
# ==========================================
model = gp.Model("IES_HighRes_Robust")
lim = config.LIMITS;
dev = config.DEVICES;
bldg = config.BUILDINGS;
cp_kw = config.PHYSICS['Cp_kW']

# --- A. 变量定义 ---
P_chp = model.addVars(T, lb=0, ub=dev['N1']['Cap_CHP_Elec'], name="P_chp")
Q_chp = model.addVars(T, lb=0, name="Q_chp")
Q_gb = model.addVars(T, lb=0, ub=dev['N1']['Cap_GB_Heat'], name="Q_gb")
P_fc = model.addVars(T, lb=0, ub=dev['N2']['Cap_FC_Elec'], name="P_fc")
Q_fc = model.addVars(T, lb=0, name="Q_fc")
Q_eb = model.addVars(T, lb=0, ub=dev['N3']['Cap_EB_Heat'], name="Q_eb")
P_eb = model.addVars(T, lb=0, name="P_eb")
P_buy = model.addVars(T, lb=0, name="P_buy")
P_curt = model.addVars(T, lb=0, ub=Wind_Power, name="P_curt")

Ts_node = model.addVars(config.NODES, T + 1, lb=lim['Ts_min'], ub=lim['Ts_max'], name="Ts_node")
Tr_node = model.addVars(config.NODES, T + 1, lb=lim['Tr_min'], ub=lim['Ts_max'], name="Tr_node")
Ts_pipe = {p: model.addVars(Pipes[p]['N'] + 1, T + 1, lb=lim['Ts_min'], ub=lim['Ts_max']) for p in Pipes}
Tr_pipe = {p: model.addVars(Pipes[p]['N'] + 1, T + 1, lb=lim['Tr_min'], ub=lim['Ts_max']) for p in Pipes}

T_in = model.addVars(config.LOADS, T + 1, lb=lim['Tin_min'], ub=lim['Tin_max'], name="T_in")
Q_sup = model.addVars(config.LOADS, T, lb=0, name="Q_sup")
Q_slack = model.addVars(config.LOADS, T, lb=0, name="Q_slack")  # 直接向建筑注热的虚拟加热器
T_slack = model.addVars(config.LOADS, T, lb=0, name="T_slack")
Tr_bldg = model.addVars(config.LOADS, T, lb=10.0, ub=lim['Ts_max'], name="Tr_bldg")
T_bp_slack = model.addVars(config.LOADS, T, lb=0, name="T_bp_slack")

# --- B. 约束条件 ---
if config.SIM_SETTINGS['Cycle_Constraints']:
    for n in config.LOADS: model.addConstr(T_in[n, 0] == T_in[n, T])
    for p in Pipes:
        for k in range(Pipes[p]['N'] + 1):
            model.addConstr(Ts_pipe[p][k, 0] == Ts_pipe[p][k, T])
            model.addConstr(Tr_pipe[p][k, 0] == Tr_pipe[p][k, T])

for t in range(T):
    model.addConstr(Q_chp[t] == P_chp[t] * dev['N1']['Alpha_CHP'])
    model.addConstr(Q_fc[t] == P_fc[t] * dev['N2']['Alpha_FC'])
    model.addConstr(Q_eb[t] == P_eb[t] * dev['N3']['Eta_EB'])
    model.addConstr(Elec_Load[t] + P_eb[t] == P_chp[t] + P_fc[t] + (Wind_Power[t] - P_curt[t]) + P_buy[t])

    for n in config.NODES:
        # 供水混合
        if m_in[n] > 0:
            enth_in = gp.quicksum(Pipes[p]['m'] * Ts_pipe[p][Pipes[p]['N'], t] for p in in_pipes[n])
            model.addConstr(Ts_node[n, t] * m_in[n] == enth_in)

        # 回水混合
        ret_enth_in = gp.quicksum(Pipes[p]['m'] * Tr_pipe[p][0, t] for p in out_pipes[n])
        if n in config.LOADS: ret_enth_in += (m_in[n] - m_out[n]) * Tr_bldg[n, t]

        m_ret_total = m_out[n] + (m_in[n] - m_out[n] if n in config.LOADS else 0)
        if m_ret_total > 0:
            model.addConstr(Tr_node[n, t] * m_ret_total == ret_enth_in)

        # 管道无缝连接
        for p in out_pipes[n]: model.addConstr(Ts_pipe[p][0, t] == Ts_node[n, t])
        for p in in_pipes[n]:  model.addConstr(Tr_pipe[p][Pipes[p]['N'], t] == Tr_node[n, t])

        # 源荷接口
        if n in config.SOURCES:
            m_src = m_out[n] - m_in[n]
            Q_inj = (Q_chp[t] + Q_gb[t]) if n == 'N1' else (Q_fc[t] if n == 'N2' else Q_eb[t])
            model.addConstr(Q_inj == m_src * cp_kw * (Ts_node[n, t] - Tr_node[n, t]))

        elif n in config.LOADS:
            m_ld = m_in[n] - m_out[n]
            # 从水里提取的热量
            model.addConstr(Q_sup[n, t] == m_ld * cp_kw * (Ts_node[n, t] - Tr_bldg[n, t]))
            model.addConstr(Ts_node[n, t] >= Tr_bldg[n, t])
            model.addConstr(Tr_bldg[n, t] >= T_in[n, t] + 3.0 - T_bp_slack[n, t])

            # [关键修复] 乘以时间步长系数 dt_h，确保物理意义正确
            net_heat_to_room = Q_sup[n, t] + Q_slack[n, t] - (T_in[n, t] - T_amb[t]) / bldg[n]['R']
            model.addConstr(T_in[n, t + 1] == T_in[n, t] + (dt_h / bldg[n]['C']) * net_heat_to_room)
            model.addConstr(T_in[n, t + 1] >= lim['Tin_min'] - T_slack[n, t])

    # FDM 方程
    for p in Pipes.keys():
        cfg = Pipes[p]
        for k in range(1, cfg['N'] + 1):
            model.addConstr(
                Ts_pipe[p][k, t + 1] == (1 - cfg['Co'] - cfg['mu']) * Ts_pipe[p][k, t] + cfg['Co'] * Ts_pipe[p][
                    k - 1, t] + cfg['mu'] * 10.0)
            model.addConstr(
                Tr_pipe[p][k - 1, t + 1] == (1 - cfg['Co'] - cfg['mu']) * Tr_pipe[p][k - 1, t] + cfg['Co'] * Tr_pipe[p][
                    k, t] + cfg['mu'] * 10.0)

# --- 目标函数 ---
eco = config.ECONOMICS
# [关键修复] 成本累加必须乘以 dt_h 转换为真正的 kWh 能量计费
cost_fuel = gp.quicksum((P_chp[t] / 0.35 + Q_gb[t] / 0.9 + P_fc[t] / 0.5) * eco['Cost_Gas'] * dt_h for t in range(T))
cost_grid = gp.quicksum(P_buy[t] * Price_Elec[t] * dt_h for t in range(T))
cost_pen = gp.quicksum(
    (eco['Cost_Curtail']*P_curt[t] +
    sum(eco['Cost_Slack_Q']*Q_slack[n,t] + eco['Cost_Slack_T']*T_slack[n,t] + 500.0*T_bp_slack[n,t] for n in config.LOADS)) * dt_h
    for t in range(T)
)
emi = gp.quicksum((P_buy[t] * eco['Factor_Grid'] + (P_chp[t] / 0.35 + Q_gb[t] / 0.9) * eco['Factor_Gas'] + (
            P_fc[t] / 0.5) * eco['Factor_FC']) * dt_h for t in range(T))
model.setObjective(cost_fuel + cost_grid + cost_pen + emi * eco['Carbon_Tax'], GRB.MINIMIZE)

# ==========================================
# 3. 求解与绘图
# ==========================================
model.Params.NumericFocus = 1      # 让 Gurobi 更小心地处理数值计算
model.Params.BarHomogeneous = 1    # 启用齐次障碍法，专门对付难解的/看似不可行的模型

model.optimize()

if model.status == GRB.OPTIMAL:
    print("--- 优化成功 ---")
    Tin_res = {n: [T_in[n, t].X for t in range(T)] for n in config.LOADS}

    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)

    # 1. 建筑温度
    ax1 = axes[0]
    ax1.set_title("(a) Multi-Node Building Status (15-min Resolution)", fontweight='bold')
    colors = ['#00695c', '#d84315', '#283593']
    for idx, n in enumerate(config.LOADS): ax1.plot(t_ax, Tin_res[n], '-', color=colors[idx], label=f'Temp {n}')
    ax1.fill_between(t_ax, lim['Tin_min'], lim['Tin_max'], color='#e0f2f1', alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.plot(t_ax, Price_Elec, color='gray', linestyle='--', label='Price')
    ax1.legend(loc='upper left');
    ax1.set_ylabel('Temp (C)')

    # 2. 热源调度
    ax2 = axes[1]
    ax2.set_title("(b) 3-Machine Heat Dispatch", fontweight='bold')
    q_gb, q_chp, q_fc, q_eb = [Q_gb[t].X for t in range(T)], [Q_chp[t].X for t in range(T)], [Q_fc[t].X for t in
                                                                                              range(T)], [Q_eb[t].X for
                                                                                                          t in range(T)]
    ax2.stackplot(t_ax, q_gb, q_chp, q_fc, q_eb, labels=['GB (N1)', 'CHP (N1)', 'FC (N2)', 'EB (N3)'],
                  colors=['gray', 'orange', 'cyan', 'purple'], alpha=0.8)
    ax2.legend(loc='upper right');
    ax2.set_ylabel('Heat (kW)')

    # 3. 电力平衡
    ax3 = axes[2]
    ax3.set_title("(c) Power Balance", fontweight='bold')
    p_wind = [Wind_Power[t] - P_curt[t].X for t in range(T)]
    p_chp, p_fc, p_buy = [P_chp[t].X for t in range(T)], [P_fc[t].X for t in range(T)], [P_buy[t].X for t in range(T)]
    ax3.stackplot(t_ax, p_wind, p_chp, p_fc, p_buy, labels=['Wind', 'CHP', 'FC', 'Grid Buy'],
                  colors=['#1E90FF', 'orange', 'cyan', 'green'], alpha=0.7)
    tot_load = Elec_Load + np.array([P_eb[t].X for t in range(T)])
    ax3.plot(t_ax, tot_load, color='black', linewidth=2, label='Total Load')
    ax3.legend(loc='upper left', ncol=2);
    ax3.set_ylabel('Power (kW)')

    # 4. 温度动态
    ax4 = axes[3]
    ax4.set_title("(d) Network Topology Temp Dynamics", fontweight='bold')
    ax4.plot(t_ax, [Ts_node['N2', t].X for t in range(T)], 'r-', lw=2.5, label='Source N2 (FC)')
    ax4.plot(t_ax, [Ts_node['N7', t].X for t in range(T)], 'm--', lw=2, label='Junction N7 (Delayed)')
    ax4.plot(t_ax, [Ts_node['N8', t].X for t in range(T)], 'orange', ls='-.', lw=2, label='Load N8 (Terminal)')
    ax4.legend();
    ax4.set_ylabel('Temp (C)')

    plt.tight_layout()
    plt.show()
else:
    print("Infeasible")