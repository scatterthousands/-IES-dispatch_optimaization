import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

# ============================
# 1. 参数设定 (Parameters)
# ============================
T = 24  # 调度周期 24小时
dt = 3600  # 时间步长 (s)

# --- 热网参数 ---
Length = 5000  # 管道长度 (m)
v_flow = 1.5  # 流速 (m/s) (定流量模式 CF-VT)
# 计算空间离散化
# 为了保证 Co <= 1, dx >= v * dt
# 设 Co = 1 (理想传输), dx = v * dt
dx = v_flow * dt
N_seg = int(np.ceil(Length / dx))  # 管道分段数
Co = (v_flow * dt) / (Length / N_seg)  # 重新计算实际Co，确保 <= 1
mu = 0.05  # 简化的热损系数 (涵盖了 alpha, beta, surrounding area 等)

print(f"管道分段数: {N_seg}, Courant Number: {Co:.4f}")

# --- 热工与设备参数 ---
cp = 4.186  # 水比热容 (kJ/kg.C)
m1 = 20.0  # 一次侧质量流量 (kg/s)
k_SHE = 1.0 * m1 * cp  # 换热站系数 (假设效率为1)

# CHP参数 (以热定电或热电解耦，这里用固定热电比简化)
alpha_CHP = 1.2  # 热电比 Q/P
eta_CHP_gen = 0.35  # 发电效率 (用于计算燃料成本)

# 建筑参数
C_b = 50000.0  # 建筑热容 (kJ/C)
K_b = 1500.0  # 建筑传热系数 (W/C -> J/s.C -> kJ/s.C * 1000) -> 1.5 kW/C
T_min = 18.0  # 最低室温
T_max = 24.0  # 最高室温

# --- 负荷与环境数据 (模拟数据) ---
np.random.seed(42)
P_load = np.array([500 + 200 * np.sin(t * np.pi / 12) for t in range(T)])  # 电负荷 (kW)
P_wind_max = np.array([300 + 100 * np.random.rand() for t in range(T)])  # 风电 (kW)
T_out = np.array([0 + 5 * np.sin((t - 6) * np.pi / 12) for t in range(T)])  # 室外温度 (C)
T_amb = np.array([5] * T)  # 管道地埋环境温度 (C)

# 价格参数
c_fuel = 0.05  # 燃料成本 ($/kW fuel)
c_grid = np.array([0.1 if 0 <= t <= 6 else 0.3 for t in range(T)])  # 分时电价
c_curt = 0.5  # 弃风惩罚

# ============================
# 2. 模型构建 (Model Building)
# ============================
model = gp.Model("IES_Thermal_Inertia")

# --- 变量声明 ---
# 能源设备 (t)
P_CHP = model.addVars(T, lb=0, ub=800, name="P_CHP")
Q_CHP = model.addVars(T, lb=0, name="Q_CHP")
P_grid = model.addVars(T, lb=0, name="P_grid")  # 买电
P_wind = model.addVars(T, lb=0, name="P_wind")  # 实际消纳风电

# 热网温度 (供水s, 回水r) [k, t]
# k=0 是热源出口/回水入口, k=N_seg 是建筑入口/回水出口
T_s = model.addVars(N_seg + 1, T, lb=40, ub=100, name="Ts")
T_r = model.addVars(N_seg + 1, T, lb=20, ub=80, name="Tr")

# 建筑 (t)
Q_s = model.addVars(T, lb=0, name="Q_s")  # 供给建筑的热量
T_in = model.addVars(T, lb=T_min, ub=T_max, name="T_in")

# --- 目标函数 (Min Cost) ---
# 燃料成本 + 购电成本 + 弃风惩罚
obj = gp.quicksum(c_fuel * P_CHP[t] / eta_CHP_gen +
                  c_grid[t] * P_grid[t] +
                  c_curt * (P_wind_max[t] - P_wind[t]) for t in range(T))
model.setObjective(obj, GRB.MINIMIZE)


# --- 约束条件 ---

# 1. 周期性边界条件 (辅助变量)
# 为了处理 t=0 时刻依赖 t-1 (即 t=23) 的情况
def get_prev_idx(t):
    return T - 1 if t == 0 else t - 1


for t in range(T):
    t_prev = get_prev_idx(t)

    # --- 5.1 电力平衡 ---
    # 假设无电锅炉EB和燃料电池FC，仅CHP+Grid+Wind
    model.addConstr(P_load[t] == P_CHP[t] + P_grid[t] + P_wind[t], name=f"ElecBal_{t}")

    # --- 5.2 可再生能源 ---
    model.addConstr(P_wind[t] <= P_wind_max[t], name=f"WindLim_{t}")

    # --- 5.3 热电耦合 ---
    model.addConstr(Q_CHP[t] == alpha_CHP * P_CHP[t], name=f"CHP_Coupling_{t}")

    # --- 5.4 & 入口边界 ---
    # 热源产热 Q_GEN = Q_CHP
    # Q_PHE = eta * Q_GEN (假设eta=1, 直接注入)
    # T_s,0 计算: Q = m * cp * (Ts,0 - Tr,0)
    # 注意：回水流向是 load -> source，所以热源处的回水是 T_r[0, t] (假设 k=0 是热源位置)
    # 修正：根据模型索引，k=0是热源。
    model.addConstr(Q_CHP[t] == m1 * cp * (T_s[0, t] - T_r[0, t]), name=f"SourceHeat_{t}")

    # --- 5.5 一次供水管网热惯性 (k=0...N-1) ---
    # T_s[k, t] -> T_s[k+1, t+1] (随时间向下游流动)
    # 你的公式是显式差分: T(t) = ... T(t-1)
    # 这里我们对应到 Python 索引：
    # T_s[k, t] 由 T_s[k, t-1] (本地热惯性) 和 T_s[k-1, t-1] (上游流入) 决定
    for k in range(1, N_seg + 1):
        # 供水管: 从 0 流向 N
        model.addConstr(
            T_s[k, t] == (1 - Co - mu) * T_s[k, t_prev] +
            Co * T_s[k - 1, t_prev] +
            mu * T_amb[t],
            name=f"PipeSupply_{k}_{t}"
        )

    # --- 5.6 SHE 换热 (节点 k=N_seg) ---
    # 供热量 Q_s = k_SHE * (T_s[N, t] - T_r[N, t])
    # 注意：这里的 T_r[N, t] 是建筑出口的回水温度，即回水管网的入口
    model.addConstr(Q_s[t] == k_SHE * (T_s[N_seg, t] - T_r[N_seg, t]), name=f"SHE_Heat_{t}")

    # 物理约束补充：回水温度不能低于室内温度 + 换热温差(5度)
    model.addConstr(T_r[N_seg, t] >= T_in[t] + 5.0, name=f"ExchangerLimit_{t}")

    # --- 5.7 一次回水管网热惯性 ---
    # 回水管: 从 N 流向 0
    # T_r[k, t] 由 T_r[k, t-1] 和 T_r[k+1, t-1] (上游是 k+1) 决定
    for k in range(0, N_seg):
        model.addConstr(
            T_r[k, t] == (1 - Co - mu) * T_r[k, t_prev] +
            Co * T_r[k + 1, t_prev] +
            mu * T_amb[t],
            name=f"PipeReturn_{k}_{t}"
        )

    # --- 5.8 建筑热惯性 ---
    # 欧拉前向差分：C * (T_in[t] - T_in[t-1]) = Q_s[t-1] - Loss[t-1]
    # 或者根据你的公式 5.8: C * (T(t+1) - T(t)) = Q - Loss
    # 这里我们统一用: T_in[t] 取决于 T_in[t-1] 和 Q_s[t-1]
    # 注意单位换算：K_b 是 kW/C, Q_s 是 kW

    # 热损项
    heat_loss = K_b * (T_in[t_prev] - T_out[t_prev]) / 1000.0  # 除以1000换算成kW匹配Q_s

    model.addConstr(
        C_b * (T_in[t] - T_in[t_prev]) / dt == Q_s[t_prev] - heat_loss,
        name=f"BuildingDyn_{t}"
    )

# ============================
# 3. 求解与可视化
# ============================
model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimization Successful!")

    # 提取结果
    res_P_CHP = [P_CHP[t].x for t in range(T)]
    res_P_grid = [P_grid[t].x for t in range(T)]
    res_T_in = [T_in[t].x for t in range(T)]
    res_T_s_source = [T_s[0, t].x for t in range(T)]  # 热源出口
    res_T_s_load = [T_s[N_seg, t].x for t in range(T)]  # 建筑入口

    # 绘图
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 图1: 电力平衡
    ax[0].stackplot(range(T), res_P_CHP, res_P_grid, labels=['CHP', 'Grid'])
    ax[0].plot(range(T), P_load, 'k--', label='Load')
    ax[0].set_ylabel('Power (kW)')
    ax[0].legend(loc='upper right')
    ax[0].set_title('Power Balance')

    # 图2: 建筑温度 (展示热惯性)
    ax[1].plot(range(T), res_T_in, 'r-o', label='Indoor Temp')
    ax[1].plot(range(T), T_out, 'b--', label='Outdoor Temp')
    ax[1].axhline(y=T_min, color='gray', linestyle=':', label='Min Comfort')
    ax[1].axhline(y=T_max, color='gray', linestyle=':', label='Max Comfort')
    ax[1].set_ylabel('Temperature (C)')
    ax[1].legend()
    ax[1].set_title('Building Thermal Inertia')

    # 图3: 管网温度 (展示传输延迟)
    ax[2].plot(range(T), res_T_s_source, 'r-', label='Source Supply Temp')
    ax[2].plot(range(T), res_T_s_load, 'orange', linestyle='--', label='Load Supply Temp (Delayed)')
    ax[2].set_ylabel('Temperature (C)')
    ax[2].set_xlabel('Time (h)')
    ax[2].legend()
    ax[2].set_title('Pipeline Thermal Inertia (Delay Effect)')

    plt.tight_layout()
    plt.show()
else:
    print("Optimization Failed or Infeasible")