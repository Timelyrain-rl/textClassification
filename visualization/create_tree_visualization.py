import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

# 设置中文字体
font_path = 'SimHei.ttf'  # 请确保这个字体文件存在或替换为系统中存在的中文字体
if os.path.exists(font_path):
    font = FontProperties(fname=font_path)
else:
    # 尝试使用系统默认中文字体
    font = FontProperties(family='SimHei')

# 读取树结构文件
file_path = 'e:/study/fourNext/CapstoneProject/textClassification/visualization/annot_tree_output/classification_tree.txt'
tree_data = []

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            path, label = parts
            nodes = path.split('|')
            tree_data.append((nodes, label))

# 创建有向图
G = nx.DiGraph()

# 添加节点和边
for nodes, label in tree_data:
    for i in range(len(nodes)):
        G.add_node(nodes[i], label=nodes[i])
        if i > 0:
            G.add_edge(nodes[i-1], nodes[i])

# 使用分层布局
pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

# 绘制图形
plt.figure(figsize=(20, 15))
nx.draw(G, pos, with_labels=False, node_size=100, node_color='skyblue', arrows=False)

# 添加标签
labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_family=font.get_name(), font_size=8)

plt.axis('off')
plt.title('分类树结构可视化', fontproperties=font, fontsize=20)
plt.savefig('e:/study/fourNext/CapstoneProject/textClassification/visualization/classification_tree.png', dpi=300, bbox_inches='tight')
plt.show()