import json

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

# 创建树结构
def build_tree():
    root = {"name": "root", "children": []}
    
    for nodes, label in tree_data:
        current = root
        for i, node in enumerate(nodes):
            # 跳过根节点，因为已经创建
            if i == 0 and node == "root":
                continue
                
            # 查找当前节点是否已存在
            found = False
            for child in current.get("children", []):
                if child["name"] == node:
                    current = child
                    found = True
                    break
            
            # 如果不存在，创建新节点
            if not found:
                new_node = {"name": node}
                if i < len(nodes) - 1:  # 不是叶子节点
                    new_node["children"] = []
                
                if "children" not in current:
                    current["children"] = []
                
                current["children"].append(new_node)
                current = new_node
    
    return root

# 生成JSON文件
tree_json = build_tree()
with open('e:/study/fourNext/CapstoneProject/textClassification/visualization/classification_tree.json', 'w', encoding='utf-8') as f:
    json.dump(tree_json, f, ensure_ascii=False, indent=2)

print("JSON文件已生成")