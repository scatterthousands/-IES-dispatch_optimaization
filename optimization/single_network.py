import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# 1. 数据生成
def get_data_realistic(T_sim=24):
    # 模拟 3000kW 级别的热负荷
    t = np.arange(T_sim)
    # 热负荷: 凌晨高，中午低
    Q = 2000 + 1200 * np.cos((t - 3) * np.pi / 12)
    # 电力: 双峰
    E = 800 + 400 * np.exp(-0.1 * (t - 10) ** 2) + 500 * np.exp(-0.1 * (t - 19) ** 2)
    # 风电: 反调峰
    W = 400 + 300 * np.cos((t - 2) * np.pi / 12)

    # 保证非负
    Q = np.maximum(Q, 100)
    W = np.maximum(W, 0)

    return Q, E, W


Q_demand, Elec_load, Wind_power = get_data_realistic(24)
T = len(Q_demand)


# 2. 物理参数修正

#管道物理参数
Length_total = 10000.0  # 10km
# 管径
Pipe_Diameter = 0.2  # DN200
Rho = 1000.0
Cp_water = 4186.0
T_ground = 10.0

#流量设计
Q_peak_W = np.max(Q_demand) * 1000
dT_design = 40.0  # 设计温差 40度
m_flow_kg_s = Q_peak_W / (Cp_water * dT_design) * 1.1  # 1.1倍裕度
coeff_flow_kW = m_flow_kg_s * 4.186

#流速与延时计算
Area = np.pi * (Pipe_Diameter / 2) ** 2
v_velocity = m_flow_kg_s / (Rho * Area)
tau_phys_hours = Length_total / v_velocity / 3600.0

print(f"--- 物理参数校验 ---")
print(f"热负荷峰值: {np.max(Q_demand):.1f} kW")
print(f"设计流量: {m_flow_kg_s:.2f} kg/s")
print(f"管内流速: {v_velocity:.2f} m/s (合理范围 0.5-2.0)")
print(f"物理传输延时: {tau_phys_hours:.2f} 小时")

# --- FDM 差分参数 ---
dt = 3600.0  # 1小时
# 自动计算分段数 N，使得 Courant数 (Co) 接近 1
# Co = v * dt / dx = 1  => dx = v * dt
dx_ideal = v_velocity * dt
N_segments = int(np.round(Length_total / dx_ideal))
if N_segments < 1: N_segments = 1

dx = Length_total / N_segments
Co = v_velocity * dt / dx  # 库朗数
mu = 0.02 / N_segments  # 简化的热损系数

print(f"--- 差分网格 ---")
print(f"空间分段 N: {N_segments}")
print(f"库朗数 Co: {Co:.4f}")

# 3. Gurobi 建模
model = gp.Model("IES_FDM_Robust")

# --- 变量 ---
# 热源扩容以确保充足
Cap_chp = np.max(Q_demand) * 0.8
Cap_gb = np.max(Q_demand) * 1.2
P_chp = model.addVars(T, lb=0, ub=Cap_chp / 1.2, name="P_chp")
Q_chp = model.addVars(T, lb=0, name="Q_chp")
Q_gb = model.addVars(T, lb=0, ub=Cap_gb, name="Q_gb")
P_buy = model.addVars(T, lb=0, name="P_buy")
P_curtail = model.addVars(T, lb=0, ub=Wind_power, name="P_curtail")

# 管道温度场 T[空间k, 时间t]
# k=0 (Source), k=N (Load)
T_supply_pipe = model.addVars(N_segments + 1, T, lb=10, ub=110, name="Ts_grid")
T_return_pipe = model.addVars(N_segments + 1, T, lb=10, ub=110, name="Tr_grid")

# 松弛变量 (虚拟电热器) - 必加！
Q_slack = model.addVars(T, lb=0, name="Q_slack")

# --- 约束 ---
for t in range(T):
    # 1. 能量平衡
    model.addConstr(Elec_load[t] == P_chp[t] + (Wind_power[t] - P_curtail[t]) + P_buy[t])
    model.addConstr(Q_chp[t] == P_chp[t] * 1.2)

    # 2. 热源接口 (k=0)
    # 产热 = 流量 * (Ts[0] - Tr[0])
    model.addConstr(Q_chp[t] + Q_gb[t] == coeff_flow_kW * (T_supply_pipe[0, t] - T_return_pipe[0, t]))

    # 3. 负荷接口 (k=N)
    # 需热 = 流量 * (Ts[N] - Tr[N]) + 松弛补热
    model.addConstr(
        coeff_flow_kW * (T_supply_pipe[N_segments, t] - T_return_pipe[N_segments, t]) + Q_slack[t]
        == Q_demand[t]
    )
    # 物理温差约束 (防止交叉)
    model.addConstr(T_supply_pipe[N_segments, t] >= T_return_pipe[N_segments, t] + 0.1)

    # 4. FDM 差分约束 (核心物理方程)
    if t > 0:
        for k in range(1, N_segments + 1):
            # 供水管: 显式一阶迎风
            # T_k^t = (1-Co-mu)*T_k^{t-1} + Co*T_{k-1}^{t-1} + mu*Tg
            model.addConstr(
                T_supply_pipe[k, t] ==
                (1 - Co - mu) * T_supply_pipe[k, t - 1] +
                Co * T_supply_pipe[k - 1, t - 1] +
                mu * T_ground
            )
            # 回水管: 流向相反 (从 k 流向 k-1)
            # 下游(k-1) 依赖于 上游(k)
            model.addConstr(
                T_return_pipe[k - 1, t] ==
                (1 - Co - mu) * T_return_pipe[k - 1, t - 1] +
                Co * T_return_pipe[k, t - 1] +
                mu * T_ground
            )
    else:
        # --- 初始时刻 t=0 的松绑 ---
        # 允许初始温度场是变量，而不是定值
        # 加上简单的物理合理性约束：沿水流方向温度不升高 (不考虑管道中途加热)
        for k in range(N_segments):
            # 供水管: T[k] >= T[k+1]
            model.addConstr(T_supply_pipe[k, 0] >= T_supply_pipe[k + 1, 0])
            # 回水管: T[k+1] >= T[k] (因为是从 N 流向 0)
            model.addConstr(T_return_pipe[k + 1, 0] >= T_return_pipe[k, 0])

        # 限制初始范围，避免极值
        model.addConstr(T_supply_pipe[0, 0] <= 110)
        model.addConstr(T_supply_pipe[N_segments, 0] >= 40)

# --- 目标函数 ---
obj = gp.quicksum(
    (P_chp[t] / 0.35 + Q_gb[t] / 0.9) * 0.04 +
    P_buy[t] * 0.15 +
    P_curtail[t] * 0.5 +
    Q_slack[t] * 10000.0  # 强惩罚松弛变量
    for t in range(T)
)
model.setObjective(obj, GRB.MINIMIZE)


# 4. 求解与绘图
model.optimize()

if model.status == GRB.OPTIMAL:
    print("\n--- 优化成功 ---")
    if sum(v.X for v in Q_slack.values()) > 1:
        print(f"警告: 仍有热缺口，总量 {sum(v.X for v in Q_slack.values()):.2f} kW (可能是初始状态调整期)")

    # 数据提取
    ts_source = [T_supply_pipe[0, t].X for t in range(T)]
    ts_load = [T_supply_pipe[N_segments, t].X for t in range(T)]

    # 绘图
    plt.style.use('bmh')
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_temp = 'tab:red'
    color_load = 'tab:blue'

    # 左轴: 负荷
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Heat Load (kW)', color=color_load)
    ax1.fill_between(range(T), Q_demand, color=color_load, alpha=0.3, label='Heat Demand')
    ax1.tick_params(axis='y', labelcolor=color_load)

    # 右轴: 温度 (展示延时)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Supply Temperature (°C)', color=color_temp)
    ax2.plot(ts_source, color='tab:red', marker='o', label='Source Temp (x=0)')
    ax2.plot(ts_load, color='tab:orange', linestyle='--', linewidth=2, label=f'Load Temp (x={Length_total / 1000}km)')
    ax2.tick_params(axis='y', labelcolor=color_temp)

    # 标题包含物理信息
    plt.title(
        f'Pipeline Thermal Dynamics (Velocity={v_velocity:.2f}m/s, Delay~{tau_phys_hours:.1f}h)\nFDM Scheme: Upwind Difference',
        fontsize=12)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3)
    plt.tight_layout()
    plt.show()

else:
    print("Infeasible")
    model.computeIIS()
    model.write("iis.ilp")