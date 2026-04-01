import numpy as np

# ==========================================
# 1. 全局仿真设置 (15分钟高分辨率)
# ==========================================
TIME_STEP_MIN = 15
T_TOTAL = int(24 * 60 / TIME_STEP_MIN)
DT_SEC = TIME_STEP_MIN * 60.0
DT_HOUR = TIME_STEP_MIN / 60.0

SIM_SETTINGS = {
    'T': T_TOTAL,
    'dt_sec': DT_SEC,
    'dt_hour': DT_HOUR,
    'Cycle_Constraints': True
}

PHYSICS = {
    'Rho': 1000.0, 'Cp': 4186.0, 'Cp_kW': 4.186,
    'T_ground': 10.0, 'Lambda_soil': 1.5, 'Depth': 1.5
}

LIMITS = {
    'Ts_min': 0.0, 'Ts_max': 100.0,
    'Tr_min': 0.0, 'Tr_max': 100.0,
    'Tin_min': 20.0, 'Tin_max': 24.0
}

# ==========================================
# 2. 网络拓扑精确定义 (14 节点新结构)
# ==========================================
NODES = [f'N{i}' for i in range(1, 15)]  # N1 到 N14
SOURCES = ['N1']
LOADS = ['N6', 'N11', 'N14']
JUNCTIONS = [n for n in NODES if n not in SOURCES and n not in LOADS]

# 遵循 KCL 的流量分配 (kg/s)
PIPE_CONFIGS = [
    {'id': 'P1', 'fr': 'N1', 'to': 'N2', 'L': 1000, 'D': 0.25, 'm': 37.0},
    {'id': 'P2', 'fr': 'N2', 'to': 'N3', 'L': 1500, 'D': 0.15, 'm': 12.0},
    {'id': 'P3', 'fr': 'N2', 'to': 'N4', 'L': 800, 'D': 0.15, 'm': 10.0},
    {'id': 'P4', 'fr': 'N2', 'to': 'N5', 'L': 2000, 'D': 0.20, 'm': 15.0},
    {'id': 'P5', 'fr': 'N3', 'to': 'N8', 'L': 1000, 'D': 0.15, 'm': 12.0},
    {'id': 'P6', 'fr': 'N4', 'to': 'N6', 'L': 500, 'D': 0.15, 'm': 10.0},
    {'id': 'P7', 'fr': 'N8', 'to': 'N7', 'L': 1000, 'D': 0.15, 'm': 12.0},
    {'id': 'P8', 'fr': 'N7', 'to': 'N5', 'L': 1000, 'D': 0.15, 'm': 12.0},
    {'id': 'P9', 'fr': 'N5', 'to': 'N9', 'L': 2500, 'D': 0.25, 'm': 27.0},
    {'id': 'P10', 'fr': 'N9', 'to': 'N10', 'L': 800, 'D': 0.15, 'm': 12.0},
    {'id': 'P11', 'fr': 'N9', 'to': 'N12', 'L': 1000, 'D': 0.20, 'm': 15.0},
    {'id': 'P12', 'fr': 'N10', 'to': 'N11', 'L': 500, 'D': 0.15, 'm': 12.0},
    {'id': 'P13', 'fr': 'N12', 'to': 'N13', 'L': 800, 'D': 0.20, 'm': 15.0},
    {'id': 'P14', 'fr': 'N13', 'to': 'N14', 'L': 500, 'D': 0.20, 'm': 15.0}
]

# ==========================================
# 3. 设备与建筑参数 (GB已移除, 设备集中于 N1)
# ==========================================
DEVICES = {
    'N1': {
        'Cap_CHP_Elec': 2000, 'Alpha_CHP': 1.2,
        'Cap_FC_Elec': 4000, 'Eta_FC_Elec': 0.50, 'Alpha_FC': 1.0,
        'Cap_EB_Heat': 4500, 'Eta_EB': 0.98
    }
}

BUILDINGS = {
    'N6': {'R': 0.015, 'C': 4000.0},
    'N11': {'R': 0.012, 'C': 5000.0},
    'N14': {'R': 0.018, 'C': 4500.0}  # 离得最远，保温最好
}

ECONOMICS = {
    'Cost_Gas': 0.04, 'Cost_Curtail': 0.5, 'Cost_Slack_Q': 1000.0, 'Cost_Slack_T': 1000.0,
    'Carbon_Tax': 0.02,
    'Factor_Grid': 0.85, 'Factor_Gas': 0.20, 'Factor_FC': 0.05
}


# ==========================================
# 4. 高精度时间序列生成
# ==========================================
def get_time_series_data():
    T = SIM_SETTINGS['T']
    t_hours = np.linspace(0, 24, T, endpoint=False)

    T_amb = -10 + 6 * np.sin((t_hours - 8) * np.pi / 12)

    Price_Elec = np.ones(T) * 0.15
    Price_Elec[(t_hours >= 0) & (t_hours < 7)] = 0.02
    Price_Elec[(t_hours >= 18) & (t_hours < 21)] = 0.30

    Wind_Power = 1800 + 1200 * np.cos((t_hours - 3) * np.pi / 12)
    Wind_Power = np.maximum(Wind_Power, 0)

    Elec_Load = 1500 + 800 * np.exp(-0.1 * (t_hours - 10) ** 2) + 1000 * np.exp(-0.1 * (t_hours - 19) ** 2)

    return {'t_hours': t_hours, 'T_amb': T_amb, 'Price_Elec': Price_Elec,
            'Wind_Power': Wind_Power, 'Elec_Load': Elec_Load}