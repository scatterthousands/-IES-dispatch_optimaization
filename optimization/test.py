import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# ==========================================
# 1. 生成符合常理的数据
# ==========================================
def generate_realistic_data(T=24):
    time = np.arange(T)

    # 1. 电负荷 (典型的双峰曲线：早高峰 9-11点，晚高峰 19-21点)
    P_elec = 200 + \
             150 * np.exp(-0.5 * ((time - 10) / 2) ** 2) + \
             200 * np.exp(-0.5 * ((time - 20) / 2) ** 2)
    # 加上一点随机波动
    P_elec += np.random.normal(0, 10, T)

    # 2. 风电 (具有反调峰特性：夜间大，白天小)
    P_wind = 100 + 150 * np.cos((time - 3) * np.pi / 12)
    # 确保不为负且不过大
    P_wind = np.clip(P_wind, 0, 300)
    # 增加随机性
    P_wind += np.random.normal(0, 15, T)
    return pd.DataFrame({
        'ElecLoad': P_elec,
        'WindPower': P_wind,
    })


# --- 数据准备 ---
# 优先读取本地CSV，如果没有则生成
file_path = os.path.join('data', 'heatload.csv')

if os.path.exists(file_path):
    print(f"读取文件: {file_path}")
    df = pd.read_csv(file_path)
    # 如果CSV里只有热负荷，我们需要补全风电和电负荷
    if 'ElecLoad' not in df.columns:
        print("CSV中缺少电/风数据，使用生成数据补全...")
        df_sim = generate_realistic_data(len(df))
        df['ElecLoad'] = df_sim['ElecLoad']
        df['WindPower'] = df_sim['WindPower']
        # 确保只要 HeatLoad 列存在即可，不用改名
else:
    print("未找到CSV文件，使用生成数据演示...")
    df = generate_realistic_data(24)

# 提取数组
T = len(df)
Elec_load = df['ElecLoad'].values
Wind_power = df['WindPower'].values
Q_demand_target = df['HeatLoad_test'].values

# ==========================================
# 2. 系统参数 (CF-VT模式)
# ==========================================

# 管道延时参数
tau = 2  # 传输延时 (小时)
eta_pipe = 0.98  # 管道每小时保温效率
eta_total = eta_pipe ** tau
T_ground = 10.0  # 土壤温度

# 关键：计算定流量 m (CF-VT)
# 流量必须足够大，保证在最大热负荷时，温差仍在合理范围内 (例如供水100，回水40，温差60)
# m = Q_max / (cp * max_delta_T)
Q_max = np.max(Q_demand_target)
cp = 4.186
max_delta_T = 40.0  # 设计最大温差
m_flow = (Q_max / (cp * max_delta_T)) * 1.2 # 给 1.2 倍的安全裕度
coeff_flow = m_flow * cp
print(f"最大热负荷: {Q_max:.1f} kW, 设定流量: {m_flow:.2f} kg/s (换热系数 {coeff_flow:.2f})")

# 设备容量
Cap_chp_elec = 600
alpha_chp = 1.0  # 热电比 Q/P
eta_chp_elec = 0.35
Cap_gb = 1200  # 燃气锅炉

# 价格参数 ($)
Cost_gas = 0.04
Cost_grid = 0.15
Cost_curtail = 0.5

# 温度边界
T_s_max = 110.0
T_s_min = 60.0
T_r_max = 110.0
T_r_min = 20.0  # 回水温度下限 (重要物理约束，如果回水太低说明供水不足)

# ==========================================
# 3. Gurobi 优化模型
# ==========================================
model = gp.Model("IES_Network_Delay_Only")

# --- 变量 ---
P_chp = model.addVars(T, lb=0, ub=Cap_chp_elec, name="P_chp")
Q_chp = model.addVars(T, lb=0, name="Q_chp")
Q_gb = model.addVars(T, lb=0, ub=Cap_gb, name="Q_gb")

P_buy = model.addVars(T, lb=0, name="P_buy")
P_curtail = model.addVars(T, lb=0, ub=Wind_power, name="P_curtail")

# 管网温度
# 热源出口供水
T_s_source = model.addVars(T, lb=T_s_min, ub=T_s_max, name="T_s_source")
# 负荷入口供水 (状态量)
T_s_load = model.addVars(T, lb=0, ub=T_s_max, name="T_s_load")
# 负荷出口回水 (关键变量，由能量平衡决定)
T_r_load = model.addVars(T, lb=T_r_min, ub=T_r_max, name="T_r_load")
# 热源入口回水
T_r_source = model.addVars(T, lb=0, ub=T_r_max, name="T_r_source")

# --- 约束 ---

for t in range(T):
    # 1. 电力平衡
    model.addConstr(
        Elec_load[t] == P_chp[t] + (Wind_power[t] - P_curtail[t]) + P_buy[t],
        name=f"Elec_Balance_{t}"
    )

    # 2. CHP 耦合 (热电比)
    model.addConstr(Q_chp[t] == P_chp[t] * alpha_chp, name=f"CHP_Couple_{t}")

    # 3. 热源侧热平衡
    # 产热 = 流量 * (供水 - 回水)
    model.addConstr(
        Q_chp[t] + Q_gb[t] == coeff_flow * (T_s_source[t] - T_r_source[t]),
        name=f"Source_Heat_Bal_{t}"
    )

    # 4. 传输延时与损耗 (CF-VT 核心)
    if t >= tau:
        # 供水: Source(t-tau) -> Load(t)
        model.addConstr(
            T_s_load[t] == T_s_source[t - tau] * eta_total + T_ground * (1 - eta_total),
            name=f"Delay_Supply_{t}"
        )
        # 回水: Load(t-tau) -> Source(t)
        model.addConstr(
            T_r_source[t] == T_r_load[t - tau] * eta_total + T_ground * (1 - eta_total),
            name=f"Delay_Return_{t}"
        )
    else:
        # 初始时刻处理 (t < tau)
        # 假设初始时刻系统处于稳态，给定一个合理的历史值，或者允许优化器自行调整边界内数值
        # 为了避免 Infeasible，这里不强制相等，只给范围，或者假设前一天的状态
        model.addConstr(T_s_load[t] >= T_s_min * eta_total + T_ground * (1 - eta_total))
        model.addConstr(T_s_load[t] <= T_s_max * eta_total + T_ground * (1 - eta_total))

        # 初始的回水温度也需要定义，防止自由变量导致无界
        model.addConstr(T_r_source[t] == 50)  # 假设初始流回热源的水是温的

    # 5. 负荷侧强制供需平衡 (代替了RC模型)
    # Demand (Fixed) = Flow * (T_supply_load - T_return_load)
    # 这个约束非常强：它实际上把 T_r_load 变成了 T_s_load 和 Q_demand 的函数
    model.addConstr(
        Q_demand_target[t] == coeff_flow * (T_s_load[t] - T_r_load[t]),
        name=f"Load_Satisfy_{t}"
    )

    #供水温度大于回水温度
    model.addConstr(T_s_load[t] >= T_r_load[t] + 0.1, name=f"Physical_DT_{t}")
    if t < tau:
        # 允许初始回水温度在 20-60度之间自由调整，不要定死
        model.addConstr(T_r_source[t] >= 20.0)
        model.addConstr(T_r_source[t] <= 60.0)

        # 允许初始供水温度(在负荷侧)在合理范围内
        model.addConstr(T_s_load[t] >= 50.0)  # 只要比 T_r_source 高即可
        model.addConstr(T_s_load[t] <= 100.0)

# --- 目标函数 ---
obj = gp.quicksum(
    (P_chp[t] / eta_chp_elec + Q_gb[t] / 0.9) * Cost_gas +
    P_buy[t] * Cost_grid +
    P_curtail[t] * Cost_curtail
    for t in range(T)
)

model.setObjective(obj, GRB.MINIMIZE)

# ==========================================
# 4. 求解与绘图
# ==========================================
model.optimize()

if model.status == GRB.OPTIMAL:
    print("\n优化成功！")

    # 提取结果
    res_P_chp = [P_chp[t].X for t in range(T)]
    res_P_buy = [P_buy[t].X for t in range(T)]
    res_P_curtail = [P_curtail[t].X for t in range(T)]

    res_Ts_source = [T_s_source[t].X for t in range(T)]
    res_Ts_load = [T_s_load[t].X for t in range(T)]
    res_Tr_load = [T_r_load[t].X for t in range(T)]

    # --- 绘图分析 ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # 子图1: 电力供需 (验证数据合理性)
    ax1 = axes[0]
    ax1.plot(Elec_load, 'k-', linewidth=2, label='Elec Load')
    ax1.plot(Wind_power, 'g--', label='Wind Power')
    ax1.bar(range(T), res_P_chp, color='orange', alpha=0.5, label='CHP Gen')
    ax1.bar(range(T), res_P_buy, bottom=res_P_chp, color='blue', alpha=0.3, label='Grid Buy')
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('Electric Balance (Realistic Profile)')
    ax1.legend()
    ax1.grid(True)

    # 子图2: 温度传输 (验证延时)
    ax2 = axes[1]
    ax2.plot(res_Ts_source, 'r-o', label='Source Supply Temp (t)')
    ax2.plot(res_Ts_load, 'r--', label=f'Load Supply Temp (t+{tau}h delay)')
    ax2.plot(res_Tr_load, 'b-', label='Load Return Temp')
    ax2.axhline(T_s_max, color='gray', linestyle=':')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title(f'Network Temperature Dynamics (Delay = {tau}h)')
    ax2.legend()
    ax2.grid(True)

    # 子图3: 供需热平衡
    ax3 = axes[2]
    # 计算实际供热量
    Q_supplied = [coeff_flow * (res_Ts_load[t] - res_Tr_load[t]) for t in range(T)]
    ax3.plot(Q_demand_target, 'k-o', label='Demand (CSV/RC result)')
    ax3.plot(Q_supplied, 'm--', linewidth=2, label='Supplied Heat')
    ax3.set_ylabel('Heat (kW)')
    ax3.set_title('Heat Demand vs Supply (Must Match)')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

else:
    print("模型不可行。请检查流量设置或热负荷是否过大导致温差无法满足。")
    model.computeIIS()
    for c in model.getConstrs():
        if c.IISConstr:
            print(f"冲突约束: {c.ConstrName}")