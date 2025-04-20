import pandas as pd
import numpy as np
import os

def generate_cooccurrence(df, col1, col2):
    labels1 = df[col1].unique()
    labels2 = df[col2].unique()
    labels1 = [str(l) for l in labels1]
    labels2 = [str(l) for l in labels2]
    
    # 创建每个样本的one-hot标签矩阵
    num_samples = len(df)
    matrix = np.zeros((num_samples, len(labels2)))
    label2_to_idx = {label: idx for idx, label in enumerate(labels2)}
    
    for idx, row in df.iterrows():
        l2 = str(row[col2])
        if l2 in label2_to_idx:
            matrix[idx, label2_to_idx[l2]] = 1
    
    return matrix

def main():
    data_path = os.path.join(os.path.dirname(__file__), '../data/merged_data_cleaned.csv')
    df = pd.read_csv(data_path)
    # 假设列名为category_l1, category_l2, category_l3
    # 如有不同请修改下方列名
    l1_col = '一级分类'
    l2_col = '二级分类'
    l3_col = '三级分类'

    co_l1_l2 = generate_cooccurrence(df, l1_col, l2_col)
    co_l2_l3 = generate_cooccurrence(df, l2_col, l3_col)
    co_l1_l3 = generate_cooccurrence(df, l1_col, l3_col)

    out_dir = os.path.join(os.path.dirname(__file__), '../data')
    np.save(os.path.join(out_dir, 'cooccurrence_l1_l2.npy'), co_l1_l2)
    np.save(os.path.join(out_dir, 'cooccurrence_l2_l3.npy'), co_l2_l3)
    np.save(os.path.join(out_dir, 'cooccurrence_l1_l3.npy'), co_l1_l3)
    print('共现矩阵已保存到data目录下')
    print(f'矩阵形状：l1_l2: {co_l1_l2.shape}, l2_l3: {co_l2_l3.shape}, l1_l3: {co_l1_l3.shape}')

if __name__ == '__main__':
    main()