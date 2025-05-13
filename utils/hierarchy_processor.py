import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def extract_hierarchy_from_data(csv_path):
    # 读取原始数据
    df = pd.read_csv(csv_path)
    
    # 初始化多标签二值化器
    mlb_l1 = MultiLabelBinarizer()
    mlb_l2 = MultiLabelBinarizer()
    mlb_l3 = MultiLabelBinarizer()

    # 处理标签格式
    l1_labels = [str(x).split(',') for x in df['一级分类']]
    l2_labels = [str(x).split(',') for x in df['二级分类']]
    l3_labels = [str(x).split(',') for x in df['三级分类']]

    # 拟合标签编码器
    mlb_l1.fit(l1_labels)
    mlb_l2.fit(l2_labels)
    mlb_l3.fit(l3_labels)

    # 构建层级约束矩阵
    parent_constraint = np.zeros((len(mlb_l1.classes_), len(mlb_l2.classes_)))
    grandparent_constraint = np.zeros((len(mlb_l1.classes_), len(mlb_l3.classes_)))

    # 分析层级关系
    hierarchy_tree = {}
    for l1, l2, l3 in zip(l1_labels, l2_labels, l3_labels):
        for l1_cls in l1:
            if l1_cls not in hierarchy_tree:
                hierarchy_tree[l1_cls] = {}
                
            for l2_cls in l2:
                if l2_cls not in hierarchy_tree[l1_cls]:
                    hierarchy_tree[l1_cls][l2_cls] = set()
                
                # 更新约束矩阵
                l1_idx = mlb_l1.classes_.tolist().index(l1_cls)
                l2_idx = mlb_l2.classes_.tolist().index(l2_cls)
                parent_constraint[l1_idx][l2_idx] = 1
                
                for l3_cls in l3:
                    hierarchy_tree[l1_cls][l2_cls].add(l3_cls)
                    
                    l3_idx = mlb_l3.classes_.tolist().index(l3_cls)
                    grandparent_constraint[l1_idx][l3_idx] = 1

    return {
        'mlb_l1': mlb_l1,
        'mlb_l2': mlb_l2,
        'mlb_l3': mlb_l3,
        'parent_constraint': parent_constraint,
        'grandparent_constraint': grandparent_constraint,
        'hierarchy_tree': hierarchy_tree
    }

def save_constraint_matrices(output_dir, matrices):
    np.save(f"{output_dir}/parent_constraint.npy", matrices['parent_constraint'])
    np.save(f"{output_dir}/grandparent_constraint.npy", matrices['grandparent_constraint'])
    print(f"约束矩阵已保存至 {output_dir}")

if __name__ == "__main__":
    matrices = extract_hierarchy_from_data("data/merged_data_cleaned.csv")
    save_constraint_matrices("data", matrices)