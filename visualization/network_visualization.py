import pandas as pd
import os
import json
import numpy as np
from collections import defaultdict

path = os.getcwd()
path = path.replace('visualization','')

# 设置文件路径
csv_file = path + '/data/merged_data_cleaned.csv'
tree_file = 'annot_tree_output/classification_tree.txt'
annot_file = 'annot_tree_output/classification_annot.txt'
output_dir = os.path.join(path, 'visualization_output')
os.makedirs(output_dir, exist_ok=True)

# 读取树结构文件
print("读取树结构文件...")
tree_data = {}
with open(tree_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            path, name = parts
            tree_data[path] = name

# 读取注释文件以获取颜色信息
print("读取注释文件...")
color_data = {}
with open(annot_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3 and 'color' in parts[1]:
            path, attr, value = parts
            if 'clade_marker_color' in attr:
                color_data[path] = value

# 读取CSV文件以获取分类计数
print("读取CSV文件...")
try:
    df = pd.read_csv(csv_file, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(csv_file, encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')

# 提取一二三级分类列
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

def build_graph_data():
    nodes = []
    links = []
    categories = []
    node_dict = {}
    category_dict = {}
    
    # 添加根节点
    nodes.append({
        "id": "root",
        "name": "分类体系",
        "symbolSize": 50,
        "category": 0,
        "value": sum(class_counts.values())
    })
    node_dict["root"] = len(nodes) - 1
    categories.append({"name": "根节点"})
    
    # 遍历树结构数据
    for path, name in tree_data.items():
        if path == "root":
            continue
            
        parts = path.split('|')
        current_id = path
        parent_id = "|".join(parts[:-1]) if len(parts) > 1 else "root"
        
        # 设置节点大小（基于数量）和类别
        count = class_counts.get(".".join(parts[1:]), 0)
        symbol_size = max(20, min(50, np.sqrt(count) * 3))
        
        # 设置节点类别
        level = len(parts) - 1
        if level not in category_dict:
            category_dict[level] = len(categories)
            categories.append({"name": f"第{level}级分类"})
        
        # 创建节点
        node = {
            "id": current_id,
            "name": parts[-1],
            "symbolSize": symbol_size,
            "category": category_dict[level],
            "value": count,
            "itemStyle": {
                "color": color_data.get(current_id, "#1f77b4")
            }
        }
        nodes.append(node)
        node_dict[current_id] = len(nodes) - 1
        
        # 创建连接
        if parent_id in node_dict:
            links.append({
                "source": node_dict[parent_id],
                "target": node_dict[current_id],
                "value": count
            })
    
    return {"nodes": nodes, "links": links, "categories": categories}

def generate_graph_html(data):
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>分类网络图</title>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
        <style>
            #main {{height: 800px; width: 100%;}}
            body {{margin: 0; padding: 20px; font-family: Arial, sans-serif;}}
            h1 {{text-align: center; color: #333;}}
            .btn-group {{margin: 20px 0; text-align: center;}}
            .btn {{padding: 8px 16px; margin: 0 10px; background-color: #4CAF50; color: white; 
                  border: none; border-radius: 4px; cursor: pointer;}}
            .btn:hover {{background-color: #45a049;}}
        </style>
    </head>
    <body>
        <h1>分类网络图</h1>
        <div id="main"></div>
        <script>
            var chartDom = document.getElementById('main');
            var myChart = echarts.init(chartDom);
            var option;

            option = {{
                tooltip: {{
                    trigger: 'item',
                    formatter: function(params) {{
                        if (params.dataType === 'node') {{
                            return params.data.name + '<br/>数量: ' + params.data.value;
                        }}
                        return params.data.source + ' -> ' + params.data.target;
                    }}
                }},
                legend: [{{
                    data: {json.dumps([cat["name"] for cat in data["categories"]], ensure_ascii=False)}
                }}],
                animationDuration: 1500,
                animationEasingUpdate: 'quinticInOut',
                series: [
                    {{
                        name: '分类网络图',
                        type: 'graph',
                        layout: 'force',
                        data: {json.dumps(data["nodes"], ensure_ascii=False)},
                        links: {json.dumps(data["links"], ensure_ascii=False)},
                        categories: {json.dumps(data["categories"], ensure_ascii=False)},
                        roam: true,
                        label: {{
                            show: true,
                            position: 'right',
                            formatter: '{{b}}'
                        }},
                        force: {{
                            repulsion: 200,
                            gravity: 0.1,
                            edgeLength: 100,
                            layoutAnimation: true
                        }},
                        emphasis: {{
                            focus: 'adjacency',
                            lineStyle: {{
                                width: 4
                            }}
                        }}
                    }}
                ]
            }};

            option && myChart.setOption(option);
            window.addEventListener('resize', function() {{
                myChart.resize();
            }});
        </script>
    </body>
    </html>
    '''
    
    with open(os.path.join(output_dir, 'network_visualization.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    print("构建分类网络图数据...")
    graph_data = build_graph_data()
    print("生成分类网络图HTML...")
    generate_graph_html(graph_data)
    print("完成！")