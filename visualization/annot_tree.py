import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import sys,os

path = os.getcwd()
path = path.replace('visualization','')

csv_file = path + '/data/merged_data_cleaned.csv'
output_dir = 'annot_tree_output'
os.makedirs(output_dir, exist_ok=True)

print("读取CSV文件...")
try:
    df = pd.read_csv(csv_file, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(csv_file, encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')

# 提取一二三级分类列
print("提取分类数据...")
df = df[['一级分类', '二级分类', '三级分类']].dropna(subset=['一级分类'])

# 统计每个分类的数量
class_counts = defaultdict(int)
for _, row in df.iterrows():
    level1 = row['一级分类']
    level2 = row['二级分类']
    level3 = row['三级分类']
    
    class_counts[level1] += 1
    if pd.notna(level2):
        class_counts[f"{level1}.{level2}"] += 1
    if pd.notna(level3):
        class_counts[f"{level1}.{level2}.{level3}"] += 1

# 创建唯一的分类集合
level1_set = set(df['一级分类'].dropna())
level2_dict = defaultdict(set)
level3_dict = defaultdict(set)

for _, row in df.iterrows():
    level1 = row['一级分类']
    level2 = row['二级分类']
    level3 = row['三级分类']
    
    if pd.notna(level2):
        level2_dict[level1].add(level2)
    
    if pd.notna(level2) and pd.notna(level3):
        level3_dict[f"{level1}.{level2}"].add(level3)

# 生成树结构文件
print("生成树结构文件...")
tree_file = os.path.join(output_dir, 'classification_tree.txt')
with open(tree_file, 'w', encoding='utf-8') as f:
    # 添加根节点
    f.write("root\troot\n")
    
    # 添加一级分类
    for level1 in level1_set:
        f.write(f"root|{level1}\t{level1}\n")
        
        # 添加二级分类
        for level2 in level2_dict[level1]:
            f.write(f"root|{level1}|{level2}\t{level2}\n")
            
            # 添加三级分类
            for level3 in level3_dict[f"{level1}.{level2}"]:
                f.write(f"root|{level1}|{level2}|{level3}\t{level3}\n")

# 生成注释文件
print("生成注释文件...")
annotation_file = os.path.join(output_dir, 'classification_annot.txt')
with open(annotation_file, 'w', encoding='utf-8') as f:
    # 设置全局属性
    f.write("\n".join([
        "title\t分类层次结构可视化",
        "title_font_size\t15",
        "total_plotted_degrees\t340",
        "start_rotation\t270",
        "clade_separation\t0.5",
        "branch_bracket_depth\t0.8",
        "branch_bracket_width\t0.2",
        "annotation_background_width\t0.2",
        "annotation_background_alpha\t0.1",
        "annotation_background_separation\t0.01",
        "\n"
    ]))
    
    # 设置根节点属性
    f.write("\n".join([
        "root\tclade_marker_size\t40",
        "root\tclade_marker_color\t#777777",
        "root\tclade_marker_edge_width\t0.1",
        "root\tring_width\t1",
        "root\tring_height\t1",
        "root\tring_color\t#777777",
        "\n"
    ]))
    
    # 设置一级分类属性
    level1_colors = plt.cm.tab20(np.linspace(0, 1, len(level1_set)))
    for i, level1 in enumerate(level1_set):
        color = '#%02x%02x%02x' % tuple(int(c*255) for c in level1_colors[i][:3])
        size = min(30, max(10, int(class_counts[level1] / 10)))
        
        f.write(f"root|{level1}\tclade_marker_size\t{size}\n")
        f.write(f"root|{level1}\tclade_marker_color\t{color}\n")
        f.write(f"root|{level1}\tclade_marker_edge_width\t0.5\n")
        f.write(f"root|{level1}\tring_width\t1\n")
        f.write(f"root|{level1}\tring_height\t1\n")
        f.write(f"root|{level1}\tring_color\t{color}\n")
        f.write(f"root|{level1}\tannotation\t{level1}\n")
        f.write(f"root|{level1}\tannotation_font_size\t8\n")
        f.write(f"root|{level1}\tannotation_background_color\t{color}\n\n")
    
    # 设置二级分类属性
    for level1 in level1_set:
        level2_list = list(level2_dict[level1])
        if level2_list:
            level2_colors = plt.cm.tab20c(np.linspace(0, 1, len(level2_list)))
            for j, level2 in enumerate(level2_list):
                color = '#%02x%02x%02x' % tuple(int(c*255) for c in level2_colors[j][:3])
                size = min(20, max(5, int(class_counts[f"{level1}.{level2}"] / 5)))
                
                f.write(f"root|{level1}|{level2}\tclade_marker_size\t{size}\n")
                f.write(f"root|{level1}|{level2}\tclade_marker_color\t{color}\n")
                f.write(f"root|{level1}|{level2}\tclade_marker_edge_width\t0.3\n")
                f.write(f"root|{level1}|{level2}\tannotation\t{level2}\n")
                f.write(f"root|{level1}|{level2}\tannotation_font_size\t6\n")
                f.write(f"root|{level1}|{level2}\tannotation_background_color\t{color}\n\n")
    
    # 设置三级分类属性
    for level1 in level1_set:
        for level2 in level2_dict[level1]:
            level3_list = list(level3_dict[f"{level1}.{level2}"])
            if level3_list:
                level3_colors = plt.cm.tab20b(np.linspace(0, 1, len(level3_list)))
                for k, level3 in enumerate(level3_list):
                    color = '#%02x%02x%02x' % tuple(int(c*255) for c in level3_colors[k][:3])
                    size = min(10, max(3, int(class_counts[f"{level1}.{level2}.{level3}"] / 3)))
                    
                    f.write(f"root|{level1}|{level2}|{level3}\tclade_marker_size\t{size}\n")
                    f.write(f"root|{level1}|{level2}|{level3}\tclade_marker_color\t{color}\n")
                    f.write(f"root|{level1}|{level2}|{level3}\tclade_marker_edge_width\t0.1\n")

print(f"树结构文件已保存到: {tree_file}")
print(f"注释文件已保存到: {annotation_file}")