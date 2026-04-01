import pandapower as pp
import pandapower.networks as nw
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 1. 加载 3机9节点 电力拓扑 =====================
# pandapower 内置 3机9节点算例
net = nw.case9()

# 查看原始拓扑（可选）
print("=== 3机9节点 原始信息 ===")
print(net.bus[['name', 'vn_kv']])
print(net.line[['from_bus', 'to_bus']])

# ===================== 2. 扩展为 IES 综合能源节点 =====================
# 2.1 新增热网属性
heat_cols = ['heat_load_mw', 'heat_source_mw', 'supply_temp_c', 'return_temp_c']
for col in heat_cols:
    net.bus[col] = 0.0
net.bus['supply_temp_c'] = 80
net.bus['return_temp_c'] = 40

# 2.2 新增气网属性
gas_cols = ['gas_load_mw', 'gas_pressure_bar']
for col in gas_cols:
    net.bus[col] = 0.0
net.bus['gas_pressure_bar'] = 4

# 2.3 挂载 IES 能源设备（3机对应3类能源站）
# Node 1 → 燃料电池（FC）：发电+少量供热
net.bus.loc[0, 'heat_source_mw'] = 2   # 余热供热 2MW
net.bus.loc[0, 'gas_load_mw'] = 10     # 耗气 10MW

# Node 2 → CHP：电-热强耦合
net.bus.loc[1, 'heat_source_mw'] = 8   # 供热 8MW
net.bus.loc[1, 'gas_load_mw'] = 12     # 耗气 12MW

# Node 3 → 燃气锅炉（GB）：纯供热
net.bus.loc[2, 'heat_source_mw'] = 6   # 供热 6MW
net.bus.loc[2, 'gas_load_mw'] = 7      # 耗气 7MW

# 2.4 分配热/气负荷
heat_load_nodes = [3, 5, 7]  # Node 4, 6, 8（pandapower 0索引）
net.bus.loc[heat_load_nodes, 'heat_load_mw'] = [3, 2.5, 4]

gas_load_nodes = [4, 6, 8]   # Node 5, 7, 9
net.bus.loc[gas_load_nodes, 'gas_load_mw'] = [2, 3, 2]

# ===================== 3. IES 耦合潮流计算 =====================
def ies_power_flow(net):
    # 3.1 电力潮流计算
    pp.runpp(net)
    print("\n=== 电力潮流结果 ===")
    print(net.res_bus[['vm_pu']])

    # 3.2 热网潮流（简化模型）
    c_water = 4.186
    density_water = 1000
    temp_diff = net.bus['supply_temp_c'] - net.bus['return_temp_c']
    net.bus['heat_flow_m3h'] = np.where(
        temp_diff != 0,
        (net.bus['heat_load_mw'] * 1000 * 3600) / (c_water * density_water * temp_diff),
        0
    )

    # 3.3 气网潮流（简化模型）
    gas_calorific = 35  # MJ/m³
    net.bus['gas_flow_m3h'] = (net.bus['gas_load_mw'] * 3600) / gas_calorific

    print("\n=== 热/气网结果 ===")
    print("热负荷与流量:\n", net.bus[['heat_load_mw', 'heat_flow_m3h']])
    print("气负荷与流量:\n", net.bus[['gas_load_mw', 'gas_flow_m3h']])
    return net

net = ies_power_flow(net)

# ===================== 4. 绘制 3机9节点 IES 拓扑图 =====================
G = nx.Graph()
nodes = list(range(1, 10))
G.add_nodes_from(nodes)

# 真实 3机9节点 线路连接
edges = [(1,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,3), (3,6), (2,5), (2,8)]
G.add_edges_from(edges)

# 固定布局（接近标准3机9节点结构）
pos = {
    1: (0, 8), 2: (6, 8), 3: (3, 0),
    4: (1, 6), 5: (3, 6), 6: (5, 6),
    7: (5, 4), 8: (3, 4), 9: (1, 4)
}

# 节点颜色分类
node_colors = []
node_sizes = []
for n in nodes:
    if n == 1:
        node_colors.append("darkblue")   # 燃料电池
        node_sizes.append(1200)
    elif n == 2:
        node_colors.append("darkorange") # CHP
        node_sizes.append(1200)
    elif n == 3:
        node_colors.append("orange")     # 燃气锅炉
        node_sizes.append(1200)
    elif n in [4,6,8]:
        node_colors.append("lightblue")  # 热负荷
        node_sizes.append(900)
    elif n in [5,7,9]:
        node_colors.append("lightgreen") # 气负荷
        node_sizes.append(900)
    else:
        node_colors.append("lightgray")
        node_sizes.append(700)

# 绘图
plt.figure(figsize=(10, 8))
nx.draw_networkx_edges(G, pos, edge_color="black", width=2)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

# 图例
legend_elements = [
    Patch(facecolor='darkblue', label='Node 1：燃料电池（FC）'),
    Patch(facecolor='darkorange', label='Node 2：CHP 机组'),
    Patch(facecolor='orange', label='Node 3：燃气锅炉（GB）'),
    Patch(facecolor='lightblue', label='热负荷节点'),
    Patch(facecolor='lightgreen', label='气负荷节点')
]
plt.legend(handles=legend_elements, loc="upper right")
plt.title("3机9节点 综合能源系统（IES）拓扑图", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.savefig("3machine9bus_IES_topology.png", dpi=300, bbox_inches="tight")
plt.show()

# 保存结果
net.bus.to_csv("3machine9bus_IES_results.csv", encoding='utf-8')