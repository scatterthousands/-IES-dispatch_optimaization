import numpy as np


# ==========================================
# 1. 全局仿真设置
# ==========================================
TIME_STEP_MIN = 15  # 设置时间粒度: 15分钟
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

# 彻底放宽边界，依靠物理方程自我平衡，杜绝 Bounds Conflict
LIMITS = {
    'Ts_min': 0.0, 'Ts_max': 100.0,
    'Tr_min': 0.0, 'Tr_max': 100.0,
    'Tin_min': 20.0, 'Tin_max': 24.0
}

# ==========================================
# 2. 网络拓扑精确定义
# ==========================================
SOURCES = ['N1', 'N2', 'N3']
LOADS = ['N5', 'N6', 'N8']
JUNCTIONS = ['N4', 'N7', 'N9']
NODES = SOURCES + LOADS + JUNCTIONS

PIPE_CONFIGS = [
    {'id': 'P1', 'fr': 'N1', 'to': 'N4', 'L': 2000, 'D': 0.15, 'm': 12.0},
    {'id': 'P2', 'fr': 'N2', 'to': 'N7', 'L': 2000, 'D': 0.15, 'm': 14.0},
    {'id': 'P3', 'fr': 'N3', 'to': 'N9', 'L': 2000, 'D': 0.15, 'm': 14.0},
    {'id': 'P4', 'fr': 'N4', 'to': 'N5', 'L': 3000, 'D': 0.20, 'm': 17.0},
    {'id': 'P5', 'fr': 'N5', 'to': 'N6', 'L': 2000, 'D': 0.10, 'm': 4.0},
    {'id': 'P6', 'fr': 'N7', 'to': 'N6', 'L': 3000, 'D': 0.15, 'm': 8.0},
    {'id': 'P7', 'fr': 'N7', 'to': 'N8', 'L': 3000, 'D': 0.10, 'm': 6.0},
    {'id': 'P8', 'fr': 'N9', 'to': 'N8', 'L': 3000, 'D': 0.15, 'm': 9.0},
    {'id': 'P9', 'fr': 'N9', 'to': 'N4', 'L': 2000, 'D': 0.10, 'm': 5.0}
]

DEVICES = {
    'N1': {'Cap_CHP_Elec': 2000, 'Alpha_CHP': 1.2, 'Cap_GB_Heat': 6000, 'Eta_GB': 0.9},
    'N2': {'Cap_FC_Elec': 1500, 'Eta_FC_Elec': 0.50, 'Alpha_FC': 1.0},
    'N3': {'Cap_EB_Heat': 8000, 'Eta_EB': 0.98}
}

BUILDINGS = {
    'N5': {'R': 0.015, 'C': 4000.0},
    'N6': {'R': 0.012, 'C': 5000.0},
    'N8': {'R': 0.018, 'C': 4500.0}
}

ECONOMICS = {
    'Cost_Gas': 0.04, 'Cost_Curtail': 0.5, 'Cost_Slack_Q': 1000, 'Cost_Slack_T': 500,
    'Carbon_Tax': 0.02,
    'Factor_Grid': 0.85, 'Factor_Gas': 0.20, 'Factor_FC': 0.05
}


# ==========================================
# 3. 时间序列生成
# ==========================================
def get_time_series_data():
    T = SIM_SETTINGS['T']
    # 生成 0到24 的连续时间轴 (如 0, 0.25, 0.5, 0.75, 1.0...)
    t_hours = np.linspace(0, 24, T, endpoint=False)

    T_amb = -10 + 6 * np.sin((t_hours - 8) * np.pi / 12)

    Price_Elec = np.ones(T) * 0.15
    Price_Elec[(t_hours >= 0) & (t_hours < 7)] = 0.02  # 谷电
    Price_Elec[(t_hours >= 18) & (t_hours < 21)] = 0.30  # 峰电

    Wind_Power = 1800 + 1200 * np.cos((t_hours - 3) * np.pi / 12)
    Wind_Power = np.maximum(Wind_Power, 0)

    Elec_Load = 1500 + 800 * np.exp(-0.1 * (t_hours - 10) ** 2) + 1000 * np.exp(-0.1 * (t_hours - 19) ** 2)

    return {'t_hours': t_hours, 'T_amb': T_amb, 'Price_Elec': Price_Elec,
            'Wind_Power': Wind_Power, 'Elec_Load': Elec_Load}