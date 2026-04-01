import pandapower as pp
import pandapower.networks as nw
import pandas as pd
import numpy as np

# ====================== 1. 加载IEEE 30-bus原始电力拓扑（全版本兼容） ======================
# 兼容函数名：ieee30() / case30()
try:
    net = nw.ieee30()
except AttributeError:
    net = nw.case30()

# 查看原始电力拓扑信息（只取肯定存在的字段，避免报错）
print("=== 原始IEEE 30-bus节点信息 ===")
print(net.bus[['name', 'vn_kv']].head())  # name和vn_kv是所有版本都有的字段
print("\n=== 原始IEEE 30-bus支路信息 ===")
print(net.line[['from_bus', 'to_bus', 'length_km']].head())  # 只取基础字段

# ====================== 2. 扩展为IES综合能源节点（电-热-气） ======================
# 2.1 给每个节点添加热网属性（不管原始字段，直接新增）
heat_cols = ['heat_load_mw', 'heat_source_mw', 'supply_temp_c', 'return_temp_c', 'heat_resistance']
for col in heat_cols:
    net.bus[col] = 0.0  # 初始化所有热网字段为0

# 批量赋值热网参数
net.bus['supply_temp_c'] = 80
net.bus['return_temp_c'] = 40
net.bus['heat_resistance'] = 0.01

# 2.2 给每个节点添加气网属性（同理，直接新增）
gas_cols = ['gas_load_mw', 'gas_pressure_bar', 'gas_resistance']
for col in gas_cols:
    net.bus[col] = 0.0

net.bus['gas_pressure_bar'] = 4
net.bus['gas_resistance'] = 0.005

# 2.3 挂载典型IES设备（CHP、燃气锅炉、热泵）
ies_nodes = [2, 7, 13, 22]

# 节点2：CHP机组（电-热耦合）
net.bus.loc[2, 'heat_source_mw'] = 10  # CHP供热10MW
net.bus.loc[2, 'gas_load_mw'] = 15  # CHP耗气15MW

# 节点7：燃气锅炉（纯供热）
net.bus.loc[7, 'heat_source_mw'] = 8
net.bus.loc[7, 'gas_load_mw'] = 9

# 节点13：热泵（电转热）
net.bus.loc[13, 'heat_source_mw'] = 6

# 2.4 添加热/气负荷
net.bus.loc[[5, 8, 10], 'heat_load_mw'] = [3, 2.5, 4]  # 居民热负荷
net.bus.loc[[15, 20], 'gas_load_mw'] = [7, 9]  # 工业气负荷


# ====================== 3. 电-热-气耦合潮流计算 ======================
def ies_power_flow(net):
    """
    简化的IES耦合潮流计算（只保证核心功能，避开版本兼容问题）
    """
    # 第一步：计算电力潮流（pandapower核心功能，所有版本都支持）
    print("\n=== 正在计算电力潮流 ===")
    pp.runpp(net)  # 不管字段名，直接跑潮流

    # 打印电力潮流核心结果（只取肯定存在的字段）
    print("=== 电力潮流计算结果 ===")
    print("节点电压标幺值:\n", net.res_bus[['vm_pu']].head())
    print("线路有功功率（送端）:\n", net.res_line[['p_from_mw']].head())

    # 第二步：计算热网潮流（简化模型，防除零）
    c_water = 4.186  # kJ/(kg·℃)
    density_water = 1000  # kg/m³
    temp_diff = net.bus['supply_temp_c'] - net.bus['return_temp_c']

    # 避免除以0，温差为0时流量设为0
    net.bus['heat_flow_m3h'] = np.where(
        temp_diff != 0,
        (net.bus['heat_load_mw'] * 1000 * 3600) / (c_water * density_water * temp_diff),
        0
    )

    # 第三步：计算气网潮流
    gas_calorific = 35  # MJ/m³
    net.bus['gas_flow_m3h'] = (net.bus['gas_load_mw'] * 3600) / gas_calorific

    # 输出热/气网结果
    print("\n=== 热/气网潮流计算结果 ===")
    print("关键IES节点热负荷与流量:")
    print(net.bus[['heat_load_mw', 'heat_flow_m3h']].loc[ies_nodes])
    print("\n关键IES节点气负荷与流量:")
    print(net.bus[['gas_load_mw', 'gas_flow_m3h']].loc[ies_nodes])

    return net


# 执行IES耦合潮流计算
net = ies_power_flow(net)

# ====================== 4. 结果保存 ======================
# 只保存核心结果，避免字段冲突
save_cols = ['name', 'vn_kv', 'heat_load_mw', 'heat_source_mw', 'gas_load_mw', 'heat_flow_m3h', 'gas_flow_m3h']
net.bus[save_cols].to_csv("IEEE30_IES_bus_results.csv", encoding='utf-8')
print("\n=== 结果已保存到 IEEE30_IES_bus_results.csv ===")

# ====================== 主函数执行 ======================
if __name__ == "__main__":
    ies_power_flow(net)