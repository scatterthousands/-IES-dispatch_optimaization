import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 引入配置文件
import complex_network_data as config

# ==========================================
# 1. 初始化与参数计算
# ==========================================
ts_data = config.get_time_series_data()
T = config.SIM_SETTINGS['T']
T_amb = ts_data['T_amb']
Price_Elec = ts_data['Price_Elec']
Wind_Power = ts_data['Wind_Power']
Elec_Load = ts_data['Elec_Load']

m_flow_total = (config.BUILDING['Q_ref'] * 1000) / \
               (config.PHYSICS['Cp'] * config.BUILDING['dT_design']) * 1.1

print(f"--- 系统初始化 ---")
print(f"设计总流量: {m_flow_total:.2f} kg/s")


def calc_pipe_params_fdm(cfg, m_flow_pipe):
    phy = config.PHYSICS
    dt = config.SIM_SETTINGS['dt']
    Re = (1 / (2 * np.pi * phy['Lambda_soil'])) * np.log(4 * phy['Depth'] / cfg['D'])
    Area = np.pi * (cfg['D'] / 2) ** 2
    mu_total = dt / (phy['Rho'] * Area * phy['Cp'] * Re)
    v = m_flow_pipe / (phy['Rho'] * Area)

    N_seg = int(np.floor(cfg['L'] / (v * dt)))
    if N_seg < 1: N_seg = 1

    dx = cfg['L'] / N_seg
    Co = v * dt / dx
    if Co > 1.0: Co = 1.0

    return {
        'v': v, 'N': N_seg, 'Co': Co, 'mu': mu_total / N_seg,
        'delay_h': cfg['L'] / v / 3600.0, 'm': m_flow_pipe
    }


Pipes = {}
print("--- 管道参数 ---")
for cfg in config.PIPE_CONFIGS:
    m_p = m_flow_total * cfg['ratio']
    # 强制修正：如果是汇流后的总管，流量设为总流量
    if cfg['id'] == 'P4':
        m_p = m_flow_total

    params = calc_pipe_params_fdm(cfg, m_p)
    Pipes[cfg['id']] = {**cfg, **params}
    print(f"Pipe {cfg['id']}: Delay={params['delay_h']:.2f}h, N={params['N']}, Co={params['Co']:.3f}, m={m_p:.2f}")

# ==========================================
# 2. Gurobi 建模
# ==========================================
model = gp.Model("IES_Configured")

# --- 变量 ---
dev = config.DEVICES
lim = config.LIMITS

# 源侧
P_chp = model.addVars(T, lb=0, ub=dev['Cap_CHP_Elec'], name="P_chp")
Q_chp = model.addVars(T, lb=0, name="Q_chp")
Q_gb = model.addVars(T, lb=0, ub=dev['Cap_GB_Heat'], name="Q_gb")
P_fc = model.addVars(T, lb=0, ub=dev['Cap_FC_Elec'], name="P_fc")
Q_fc = model.addVars(T, lb=0, name="Q_fc")
Q_eb = model.addVars(T, lb=0, ub=dev['Cap_EB_Heat'], name="Q_eb")
P_eb = model.addVars(T, lb=0, name="P_eb")

# 电网
P_buy = model.addVars(T, lb=0, name="P_buy")
P_curt = model.addVars(T, lb=0, ub=Wind_Power, name="P_curt")

# 温度场
Ts_node = model.addVars(config.NODES, T + 1, lb=lim['Ts_min'], ub=lim['Ts_max'], name="Ts_node")
Tr_node = model.addVars(config.NODES, T + 1, lb=lim['Tr_min'], ub=lim['Tr_max'], name="Tr_node")

Ts_pipe = {}
Tr_pipe = {}
for pid, p in Pipes.items():
    Ts_pipe[pid] = model.addVars(p['N'] + 1, T + 1, lb=lim['Ts_min'] - 10, ub=lim['Ts_max'], name=f"Ts_{pid}")
    Tr_pipe[pid] = model.addVars(p['N'] + 1, T + 1, lb=lim['Tr_min'] - 10, ub=lim['Tr_max'], name=f"Tr_{pid}")

T_in = model.addVars(T + 1, lb=lim['Tin_min'], ub=lim['Tin_max'], name="T_in")
Q_load = model.addVars(T, lb=0, name="Q_load")
Q_slack = model.addVars(T, lb=0, name="Q_slack")
T_slack = model.addVars(T, lb=0, name="T_slack")

# --- 约束 ---

# 1. 初始状态与周期性
if config.SIM_SETTINGS['Cycle_Constraints']:
    model.addConstr(T_in[0] == T_in[T], name="Cyclic_Tin")
    # 周期性约束
    for pid, p in Pipes.items():
        for k in range(p['N'] + 1):
            model.addConstr(Ts_pipe[pid][k, 0] == Ts_pipe[pid][k, T])
            model.addConstr(Tr_pipe[pid][k, 0] == Tr_pipe[pid][k, T])

    # 强制初始室温为下限，观察爬升能力
    model.addConstr(T_in[0] == 20.0)
else:
    model.addConstr(T_in[0] == lim['Init_Tin'])
    for pid, p in Pipes.items():
        for k in range(p['N'] + 1):
            model.addConstr(Ts_pipe[pid][k, 0] == lim['Init_Ts'])
            model.addConstr(Tr_pipe[pid][k, 0] == lim['Init_Tr'])

for t in range(T):
    # A. 设备
    model.addConstr(Q_chp[t] == P_chp[t] * dev['Alpha_CHP'])
    model.addConstr(Q_fc[t] == P_fc[t] * dev['Alpha_FC'])
    model.addConstr(Q_eb[t] == P_eb[t] * dev['Eta_EB'])

    # B. 源侧热平衡
    cp_kw = config.PHYSICS['Cp_kW']
    # N1: CHP+GB+FC
    model.addConstr(Q_chp[t] + Q_gb[t] + Q_fc[t] ==
                    Pipes['P1']['m'] * cp_kw * (Ts_node['N1', t] - Tr_node['N1', t]))
    # N2: EB
    model.addConstr(Q_eb[t] == Pipes['P2']['m'] * cp_kw * (Ts_node['N2', t] - Tr_node['N2', t]))
    # N3: 无源 (备用)
    model.addConstr(0 == Pipes['P3']['m'] * cp_kw * (Ts_node['N3', t] - Tr_node['N3', t]))

    # C. 电力平衡
    model.addConstr(Elec_Load[t] + P_eb[t] ==
                    P_chp[t] + P_fc[t] + (Wind_Power[t] - P_curt[t]) + P_buy[t])

    # D. 管道 FDM
    for pid, p in Pipes.items():
        model.addConstr(Ts_pipe[pid][0, t] == Ts_node[p['fr'], t])
        model.addConstr(Tr_pipe[pid][p['N'], t] == Tr_node[p['to'], t])

        for k in range(1, p['N'] + 1):
            model.addConstr(Ts_pipe[pid][k, t + 1] ==
                            (1 - p['Co'] - p['mu']) * Ts_pipe[pid][k, t] +
                            p['Co'] * Ts_pipe[pid][k - 1, t] +
                            p['mu'] * config.PHYSICS['T_ground'])
            model.addConstr(Tr_pipe[pid][k - 1, t + 1] ==
                            (1 - p['Co'] - p['mu']) * Tr_pipe[pid][k - 1, t] +
                            p['Co'] * Tr_pipe[pid][k, t] +
                            p['mu'] * config.PHYSICS['T_ground'])

    # E. 混合节点 N4
    m_in_sum = sum(Pipes[p]['m'] for p in ['P1', 'P2', 'P3'])
    enth_sum = sum(Pipes[p]['m'] * Ts_pipe[p][Pipes[p]['N'], t] for p in ['P1', 'P2', 'P3'])
    model.addConstr(Ts_node['N4', t] * m_in_sum == enth_sum)
    model.addConstr(Ts_pipe['P4'][0, t] == Ts_node['N4', t])

    model.addConstr(Tr_node['N4', t] == Tr_pipe['P4'][0, t])
    for p in ['P1', 'P2', 'P3']:
        model.addConstr(Tr_pipe[p][Pipes[p]['N'], t] == Tr_node['N4', t])

    # F. 负荷
    model.addConstr(Q_load[t] + Q_slack[t] ==
                    Pipes['P4']['m'] * cp_kw * (Ts_node['N5', t] - Tr_node['N5', t]))
    model.addConstr(Ts_node['N5', t] == Ts_pipe['P4'][Pipes['P4']['N'], t])
    model.addConstr(Tr_pipe['P4'][Pipes['P4']['N'], t] == Tr_node['N5', t])
    model.addConstr(Tr_node['N5', t] >= T_in[t] + 3.0)

    # G. 建筑 RC
    bldg = config.BUILDING
    model.addConstr(T_in[t + 1] == T_in[t] +
                    (1 / bldg['C']) * (Q_load[t] - (T_in[t] - T_amb[t]) / bldg['R']))
    model.addConstr(T_in[t + 1] >= lim['Tin_min'] - T_slack[t])

# --- 目标函数 ---
eco = config.ECONOMICS
cost_fuel = gp.quicksum((P_chp[t] / 0.35 + Q_gb[t] / dev['Eta_GB']) * eco['Cost_Gas'] for t in range(T))
cost_fc_fuel = gp.quicksum((P_fc[t] / dev['Eta_FC_Elec']) * eco['Cost_Gas'] for t in range(T))
cost_grid = gp.quicksum(P_buy[t] * Price_Elec[t] for t in range(T))
cost_pen = gp.quicksum(eco['Cost_Curtail'] * P_curt[t] +
                       eco['Cost_Slack_Q'] * Q_slack[t] +
                       eco['Cost_Slack_T'] * T_slack[t] for t in range(T))
emi_grid = gp.quicksum(P_buy[t] * eco['Factor_Grid'] for t in range(T))
emi_gas = gp.quicksum((P_chp[t] / 0.35 + Q_gb[t] / dev['Eta_GB']) * eco['Factor_Gas'] for t in range(T))
emi_fc = gp.quicksum((P_fc[t] / dev['Eta_FC_Elec']) * eco['Factor_FC'] for t in range(T))
cost_carbon = (emi_grid + emi_gas + emi_fc) * eco['Carbon_Tax']

model.setObjective(cost_fuel + cost_fc_fuel + cost_grid + cost_pen + cost_carbon, GRB.MINIMIZE)

# ==========================================
# 4. 绘图
# ==========================================
model.optimize()

if model.status == GRB.OPTIMAL:
    print("--- 优化成功 ---")

    res_Tin = [T_in[t].X for t in range(T)]

    # 提取数据
    res_Ts_src = [Ts_node['N4', t].X for t in range(T)]
    res_Ts_load = [Ts_node['N5', t].X for t in range(T)]
    res_Tr_load = [Tr_node['N5', t].X for t in range(T)]

    res_Q_eb = [Q_eb[t].X for t in range(T)]
    res_Q_chp = [Q_chp[t].X for t in range(T)]
    res_Q_gb = [Q_gb[t].X for t in range(T)]
    res_Q_fc = [Q_fc[t].X for t in range(T)]

    res_P_chp = [P_chp[t].X for t in range(T)]
    res_P_fc = [P_fc[t].X for t in range(T)]
    res_P_eb = [P_eb[t].X for t in range(T)]
    res_P_buy = [P_buy[t].X for t in range(T)]
    res_Wind_used = [Wind_Power[t] - P_curt[t].X for t in range(T)]

    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    c_eb = '#9370DB'
    c_chp = '#FFA500'
    c_gb = '#696969'
    c_fc = '#00FFFF'
    c_wind = '#1E90FF'
    c_grid = '#2E8B57'

    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    time_axis = np.arange(T)

    # 图 1: 室温
    ax1 = axes[0]
    ax1.set_title("(a) Building Status: Thermal Storage", loc='left', fontweight='bold')
    ax1.fill_between(time_axis, lim['Tin_min'], lim['Tin_max'], color='#e0f2f1', label='Comfort Zone')
    ax1.plot(time_axis, res_Tin, 'o-', color='#00695c', linewidth=2, label='Indoor Temp')
    ax1b = ax1.twinx()
    ax1b.step(time_axis, Price_Elec, where='mid', color='gray', linestyle='--', label='Price')
    ax1b.fill_between(time_axis, Price_Elec, 0, step='mid', color='gray', alpha=0.1)
    ax1.legend(loc='upper left')

    # 图 2: 热源
    ax2 = axes[1]
    ax2.set_title("(b) Heat Source Dispatch", loc='left', fontweight='bold')
    ax2.stackplot(time_axis, res_Q_gb, res_Q_chp, res_Q_fc, res_Q_eb,
                  labels=['GB', 'CHP', 'Fuel Cell', 'EB'],
                  colors=[c_gb, c_chp, c_fc, c_eb], alpha=0.8)
    ax2.set_ylabel('Heat (kW)')
    ax2.legend(loc='upper right')

    # 图 3: 电力平衡
    ax3 = axes[2]
    ax3.set_title("(c) Power Balance: Valley Filling by EB", loc='left', fontweight='bold')
    ax3.stackplot(time_axis, res_Wind_used, res_P_chp, res_P_fc, res_P_buy,
                  labels=['Wind Used', 'CHP', 'Fuel Cell', 'Grid Buy'],
                  colors=[c_wind, c_chp, c_fc, c_grid], alpha=0.7)
    total_load = Elec_Load + np.array(res_P_eb)
    ax3.plot(time_axis, Elec_Load, 'k--', linewidth=1.5, label='Base Elec Load')
    ax3.step(time_axis, total_load, where='mid', color='black', linewidth=2.5, label='Total Load (w/ EB)')
    ax3.fill_between(time_axis, Elec_Load, total_load, step='mid', color=c_eb, alpha=0.5, label='EB Charging')
    ax3.set_ylabel('Power (kW)')
    ax3.legend(loc='upper left', ncol=2)

    # 图 4: 管网
    ax4 = axes[3]
    ax4.set_title(f"(d) Network Temp (Delay ~{Pipes['P4']['delay_h']:.1f}h)", loc='left', fontweight='bold')
    ax4.plot(time_axis, res_Ts_src, color='#d32f2f', linewidth=2.5, label='Mix Source ($T_{s,N4}$)')
    ax4.plot(time_axis, res_Ts_load, color='#ff7043', linewidth=2.5, linestyle='--', label='Load In ($T_{s,N5}$)')
    ax4.plot(time_axis, res_Tr_load, color='#1976d2', linewidth=1.5, linestyle=':', label='Load Out ($T_{r,N5}$)')
    ax4.legend(ncol=3)

    plt.tight_layout()
    plt.show()

elif model.status == GRB.INFEASIBLE:
    print("Model is Infeasible. Computing IIS to diagnose...")
    model.computeIIS()
    model.write("infeasible.ilp")
    print("IIS written to infeasible.ilp")