import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


# 1. 数据生成
def get_data_final(T_sim=24):
    t = np.arange(T_sim)

    # 1. 室外温度: 严寒 (-10 ~ -5)
    T_amb = -10 + 5 * np.sin((t - 9) * np.pi / 12)

    # 2. 电价 ($/kWh): 显著的峰谷差
    # 谷 (0-7点): 0.02 (比气便宜)
    # 平 (7-18点): 0.15
    # 峰 (18-21点): 0.30 放热
    Price_Elec = np.ones(T_sim) * 0.15
    Price_Elec[0:7] = 0.02
    Price_Elec[18:21] = 0.30
    # Price_Elec[24:31] = 0.02
    # Price_Elec[42:45] = 0.30
    # Price_Elec[48:55] = 0.02
    # Price_Elec[66:69] = 0.30

    # 3. 风电: 夜间大风，配合谷电
    Wind_Power = 1200 + 800 * np.cos((t - 3) * np.pi / 12)
    Wind_Power = np.maximum(Wind_Power, 0)

    # 4. 电负荷
    Elec_Load = 1000 + 400 * np.exp(-0.1 * (t - 10) ** 2) + 500 * np.exp(-0.1 * (t - 19) ** 2)

    # 5. 基础热需求参考
    Q_ref = 3000.0

    return T_amb, Elec_Load, Wind_Power, Price_Elec, Q_ref


T_amb, Elec_load, Wind_power, Price_grid, Q_ref = get_data_final(24)
T = len(T_amb)


# 2. 物理参数设置

# --- A. 建筑参数 (增强保温，利于存热) ---
R_bldg = 0.015  # 热阻 (保温较好)
C_bldg = 5000.0  # 热容 (蓄热能力大)
T_in_min, T_in_max = 20.0, 24.0

# --- B. 管网参数 ---
# 长度设为 6km，使延时控制在 3-4小时左右
Length_total = 6000.0
Pipe_Diameter = 0.3  # DN300
Rho = 1000.0
Cp_water = 4186.0
T_ground = -10.0

# 温度边界
T_s_min, T_s_max = 60.0, 90.0  # 供水最高 90
T_r_min, T_r_max = 30.0, 60.0  # 回水放宽到 75，防止卡死

# 初始状态
Init_Ts_Val = 75.0
Init_Tr_Val = 45.0

# 流量设计
dT_design = 30.0
m_flow_kg_s = (Q_ref * 1000) / (Cp_water * dT_design) * 1.1
coeff_flow_kW = m_flow_kg_s * 4.186

# FDM 参数
Area = np.pi * (Pipe_Diameter / 2) ** 2
v_velocity = m_flow_kg_s / (Rho * Area)
dt_sec = 3600.0
tau_phys_hours = Length_total / v_velocity / 3600.0

# 稳定性修正: Co <= 1
N_max_stable = int(np.floor(Length_total / (v_velocity * dt_sec)))
N_segments = max(1, N_max_stable)
dx = Length_total / N_segments
Co = v_velocity * dt_sec / dx
mu = 0.02 / N_segments

print(f"--- 物理参数 ---")
print(f"流速: {v_velocity:.2f} m/s")
print(f"传输延时: {tau_phys_hours:.2f} h")
print(f"网格 N: {N_segments}, Co: {Co:.3f}")


# 3. Gurobi 建模
model = gp.Model("IES_Final_Run")

# --- 变量 ---
# 设备容量
Cap_chp_elec = 5000
Cap_gb_heat = 5000
Cap_eb_heat = 3000

P_chp = model.addVars(T, lb=0, ub=Cap_chp_elec, name="P_chp")
Q_chp = model.addVars(T, lb=0, name="Q_chp")
Q_gb = model.addVars(T, lb=0, ub=Cap_gb_heat, name="Q_gb")
Q_eb = model.addVars(T, lb=0, ub=Cap_eb_heat, name="Q_eb")

P_buy = model.addVars(T, lb=0, name="P_buy")
P_curt = model.addVars(T, lb=0, ub=Wind_power, name="P_curt")
P_eb = model.addVars(T, lb=0, name="P_eb")

# 管网状态
T_supply = model.addVars(N_segments + 1, T, lb=T_s_min, ub=T_s_max, name="Ts")
T_return = model.addVars(N_segments + 1, T, lb=T_r_min, ub=T_r_max, name="Tr")

# 建筑状态
T_in = model.addVars(T + 1, lb=T_in_min, ub=T_in_max, name="T_in")
Q_bldg_supply = model.addVars(T, lb=0, name="Q_bldg_supply")

# 松弛变量
Q_slack = model.addVars(T, lb=0, name="Q_slack")
T_slack_low = model.addVars(T, lb=0, name="T_slack_low")

# --- 约束 ---

# 1. 初始状态 (强制冷启动/中间态)
# 强制初始室温为下限，逼迫系统加热
model.addConstr(T_in[0] == 20.0)

# 2. 管道初始状态固定 (t=0)
for k in range(N_segments + 1):
    model.addConstr(T_supply[k, 0] == Init_Ts_Val)  # 固定为75
    model.addConstr(T_return[k, 0] == Init_Tr_Val)  # 固定为45

for t in range(T):
    # 设备关系
    model.addConstr(Q_eb[t] == P_eb[t] * 0.98)
    model.addConstr(Q_chp[t] == P_chp[t] * 1.2)

    # 能量平衡
    model.addConstr(Elec_load[t] + P_eb[t] == P_chp[t] + (Wind_power[t] - P_curt[t]) + P_buy[t])
    model.addConstr(Q_chp[t] + Q_gb[t] + Q_eb[t] == coeff_flow_kW * (T_supply[0, t] - T_return[0, t]))

    # 管道 FDM (t > 0)
    if t > 0:
        for k in range(1, N_segments + 1):
            # 供水管
            model.addConstr(
                T_supply[k, t] == (1 - Co - mu) * T_supply[k, t - 1] + Co * T_supply[k - 1, t - 1] + mu * T_ground)
            # 回水管
            model.addConstr(
                T_return[k - 1, t] == (1 - Co - mu) * T_return[k - 1, t - 1] + Co * T_return[k, t - 1] + mu * T_ground)

    # 负荷耦合
    model.addConstr(
        Q_bldg_supply[t] + Q_slack[t] == coeff_flow_kW * (T_supply[N_segments, t] - T_return[N_segments, t]))
    # 物理换热限制
    model.addConstr(T_return[N_segments, t] >= T_in[t] + 3.0)

    # 建筑动力学
    model.addConstr(T_in[t + 1] == T_in[t] + (1.0 / C_bldg) * (Q_bldg_supply[t] - (T_in[t] - T_amb[t]) / R_bldg))
    model.addConstr(T_in[t + 1] >= 20.0 - T_slack_low[t])

# --- 目标函数 ---
obj = gp.quicksum(
    (P_chp[t] / 0.35 + Q_gb[t] / 0.9) * 0.04 +  # 气价
    P_buy[t] * Price_grid[t] +  # 电价
    P_curt[t] * 5 +  # 弃风惩罚
    Q_slack[t] * 1e5 +
    T_slack_low[t] * 1e4
    for t in range(T)
)
model.setObjective(obj, GRB.MINIMIZE)


# 4. 绘图
model.optimize()

if model.status == GRB.OPTIMAL:
    print("--- 优化成功：准备全景绘图 ---")

    # 1. 提取基础数据
    res_Tin = [T_in[t].X for t in range(T)]
    res_Ts_src = [T_supply[0, t].X for t in range(T)]
    res_Ts_load = [T_supply[N_segments, t].X for t in range(T)]
    res_Tr_load = [T_return[N_segments, t].X for t in range(T)]

    # 2. 提取能源出力数据
    res_Q_eb = [Q_eb[t].X for t in range(T)]
    res_Q_chp = [Q_chp[t].X for t in range(T)]
    res_Q_gb = [Q_gb[t].X for t in range(T)]

    # 3. 提取电力数据
    res_P_chp = [P_chp[t].X for t in range(T)]
    res_P_eb = [P_eb[t].X for t in range(T)]
    res_P_buy = [P_buy[t].X for t in range(T)]
    res_P_curt = [P_curt[t].X for t in range(T)]
    res_Wind_used = [Wind_power[t] - res_P_curt[t] for t in range(T)]

    # --- 绘图配置 ---
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    # 定义配色方案
    c_eb = '#9370DB'  # 紫色 (电锅炉)
    c_chp = '#FFA500'  # 橙色 (CHP)
    c_gb = '#696969'  # 深灰 (燃气锅炉)
    c_wind = '#1E90FF'  # 蓝色 (风电)
    c_grid = '#2E8B57'  # 绿色 (电网购电)
    c_load = 'black'  # 负荷

    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    time_axis = np.arange(T)

    # ---------------------------
    # 图 1: 室内温度与电价 (保持不变)
    # ---------------------------
    ax1 = axes[0]
    ax1.set_title("(a) Building Status: Indoor Temp & Price Signal", loc='left', fontweight='bold')

    # 舒适区背景
    ax1.fill_between(time_axis, 20, 24, color='#e0f2f1', label='Comfort Zone')
    # 室温曲线
    ax1.plot(time_axis, res_Tin, 'o-', color='#00695c', linewidth=2, label='Indoor Temp ($T_{in}$)')

    ax1b = ax1.twinx()
    # 电价阶梯
    ax1b.step(time_axis, Price_grid, where='mid', color='gray', linestyle='--', alpha=0.7, label='Elec Price')
    ax1b.fill_between(time_axis, Price_grid, 0, step='mid', color='gray', alpha=0.1)

    ax1.set_ylabel('Temp (°C)', color='#00695c')
    ax1b.set_ylabel('Price ($/kWh)', color='gray')
    ax1.legend(loc='upper left')
    ax1.set_ylim(19.5, 24.5)

    # ---------------------------
    # 图 2: 热源调度堆叠图 (Heat Dispatch)
    # ---------------------------
    ax2 = axes[1]
    ax2.set_title("(b) Heat Source Dispatch ", loc='left', fontweight='bold')

    # 堆叠面积图
    ax2.stackplot(time_axis, res_Q_gb, res_Q_chp, res_Q_eb,
                  labels=['Gas Boiler', 'CHP Heat', 'Electric Boiler'],
                  colors=[c_gb, c_chp, c_eb], alpha=0.8)

    # 叠加总热负荷需求 (进入管网的热量)
    total_heat_supply = np.array(res_Q_gb) + np.array(res_Q_chp) + np.array(res_Q_eb)
    ax2.plot(time_axis, total_heat_supply, 'k--', linewidth=1, label='Total Heat Injection')

    ax2.set_ylabel('Heat Power (kW)')
    ax2.legend(loc='upper right')

    # ---------------------------
    # 图 3: 电力平衡堆叠图 (Power Dispatch)
    # ---------------------------
    ax3 = axes[2]
    ax3.set_title("(c) Electric Power Balance", loc='left', fontweight='bold')

    # 正向堆叠：电源 (风电利用量 + CHP发电 + 购电)
    ax3.stackplot(time_axis, res_Wind_used, res_P_chp, res_P_buy,
                  labels=['Wind Used', 'CHP Elec', 'Grid Buy'],
                  colors=[c_wind, c_chp, c_grid], alpha=0.7)

    # 线条：电力负荷 (基础负荷 + EB耗电)
    total_elec_load = Elec_load + np.array(res_P_eb)
    ax3.step(time_axis, total_elec_load, where='mid', color='black', linewidth=2, label='Total Elec Load (Base+EB)')

    ax3.set_ylabel('Electric Power (kW)')
    ax3.legend(loc='upper left')

    # ---------------------------
    # 图 4: 管网温度动态
    # ---------------------------
    ax4 = axes[3]
    ax4.set_title(f"(d) Network Temp Dynamics (Delay ~{tau_phys_hours:.1f}h)", loc='left', fontweight='bold')

    ax4.plot(time_axis, res_Ts_src, color='#d32f2f', linewidth=2.5, label='Source Supply ($T_{s,0}$)')
    ax4.plot(time_axis, res_Ts_load, color='#ff7043', linewidth=2.5, linestyle='--', label='Load Supply ($T_{s,N}$)')
    ax4.plot(time_axis, res_Tr_load, color='#1976d2', linewidth=1.5, linestyle=':', label='Load Return ($T_{r,N}$)')

    ax4.set_ylabel('Temperature (°C)')
    ax4.set_xlabel('Time (Hour)')
    ax4.set_ylim(30, 95)
    ax4.legend(ncol=3)

    plt.tight_layout()
    plt.show()

else:
    print("Infeasible")