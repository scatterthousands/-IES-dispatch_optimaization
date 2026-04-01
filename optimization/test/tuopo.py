import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 1. 加载真实 IEEE 30-bus 数据 =====================
import pandapower as pp
import pandapower.networks as nw

try:
    net = nw.ieee30()
except AttributeError:
    net = nw.case30()

# ===================== 2. 手动定义坐标（完全还原你截图的结构） =====================
# 这是根据你提供的 IEEE 30-bus 结构图反推的坐标
# 布局规则：上北下南，左西右东
pos = {
    1: (0, 6), 2: (1, 6), 3: (2, 6), 4: (1.5, 5),
    5: (-1, 5), 6: (2, 4), 7: (-2, 5), 8: (1, 3),
    9: (3, 3), 10: (2, 2), 11: (3.5, 3), 12: (2, 1),
    13: (0, 0), 14: (1, 0), 15: (1.5, -1), 16: (2.5, 1),
    17: (2.5, 0), 18: (2, -1), 19: (3, 1), 20: (3.5, 1),
    21: (3, 0), 22: (4, 1), 23: (5, -1), 24: (4, 0),
    25: (6, -1), 26: (7, -1), 27: (3, 3), 28: (0, 4),
    29: (2.5, 4), 30: (3.5, 4)
}

# ===================== 3. 构建图并绘制 =====================
G = nx.Graph()
# 添加所有节点
G.add_nodes_from(range(1, 31))

# 【关键】从 net 中提取真实的线路连接（这一步必须做，不能瞎连）
edges = []
for idx, row in net.line.iterrows():
    # pandapower 里 from_bus/to_bus 是整数，直接对应节点号
    edges.append((row['from_bus'] + 1, row['to_bus'] + 1)) # 因为pandapower是0索引，这里转1索引

G.add_edges_from(edges)

# ===================== 4. 分类着色（按官方图纸） =====================
node_colors = []
node_sizes = []
labels = {}

for n in range(1, 31):
    # 发电机节点 (根据你截图上的波浪符号)
    if n in [1, 2, 3, 6, 8, 13, 27]:
        node_colors.append('red')
        node_sizes.append(1500)
        labels[n] = str(n)
    # 热负荷 (浅蓝色)
    elif n in [5, 8, 10]:
        node_colors.append('lightblue')
        node_sizes.append(800)
        labels[n] = str(n)
    # 气负荷 (浅绿色)
    elif n in [15, 20]:
        node_colors.append('lightgreen')
        node_sizes.append(800)
        labels[n] = str(n)
    # 普通节点
    else:
        node_colors.append('lightgray')
        node_sizes.append(600)
        labels[n] = str(n)

# ===================== 5. 绘图 =====================
plt.figure(figsize=(14, 10))

# 画电力线路
nx.draw_networkx_edges(G, pos, edge_color='black', width=1.5, alpha=0.7)

# 画节点
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)

# 画编号
nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

# ===================== 6. 论文级图例与标题 =====================
legend_elements = [
    Patch(facecolor='red', label='发电机节点 (平衡节点/机组)'),
    Patch(facecolor='lightblue', label='热负荷节点 (参考代码配置)'),
    Patch(facecolor='lightgreen', label='气负荷节点 (参考代码配置)'),
    Patch(facecolor='lightgray', label='普通电力节点')
]
plt.legend(handles=legend_elements, loc='upper right')
plt.title('IEEE 30-bus 综合能源系统 (IES) 拓扑图 - 标准还原版', fontsize=16)
plt.axis('off')
plt.tight_layout()

# 保存高清图
plt.savefig('IEEE30_IES_TrueStructure.png', dpi=300, bbox_inches='tight')
plt.show()