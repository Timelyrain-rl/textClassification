from nlpcda import Simbert
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm

# 设置日志级别过滤无关警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = {
    'model_path': '/root/autodl-tmp/textClassification/models/chinese_simbert_L-12_H-768_A-12',
    'CUDA_VISIBLE_DEVICES': '0',
    'max_len': 128,
    'seed': 42
}
simbert = Simbert(config=config)

def generate_synonyms_csv(input_csv, output_csv):
    # 读取CSV文件
    df = pd.read_csv(input_csv)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    
    # 创建输出文件并写入表头
    columns = list(df.columns) + ['生成文本', '置信度']
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write(','.join(columns) + '\n')
    
    # 记录成功生成的同义句数量
    success_count = 0
    
    # 处理每个主要内容
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            if pd.notnull(row['主要内容']):
                # 限制输入文本长度，防止形状不兼容错误
                content = str(row['主要内容'])
                if len(content) > 100:  # 限制输入长度
                    content = content[:100]
                
                # 生成同义句
                synonyms = simbert.replace(sent=content, create_num=3)
                
                # 如果生成成功，立即写入文件
                if synonyms:
                    for text, score in synonyms:
                        # 创建新行数据
                        new_row = row.copy()
                        
                        # 将数据转换为字符串列表，准备写入CSV
                        row_values = []
                        for col in df.columns:
                            if col != '生成文本' and col != '置信度':
                                # 处理可能包含逗号的字段
                                value = str(new_row[col]).replace('"', '""')
                                if ',' in value:
                                    value = f'"{value}"'
                                row_values.append(value)
                        
                        # 添加生成文本和置信度
                        text_value = text.replace('"', '""')
                        if ',' in text_value:
                            text_value = f'"{text_value}"'
                        row_values.append(text_value)
                        row_values.append(f"{score:.2f}")
                        
                        # 写入文件
                        with open(output_csv, 'a', encoding='utf-8') as f:
                            f.write(','.join(row_values) + '\n')
                        
                        success_count += 1
        except Exception as e:
            print(f"生成失败[{idx}]: {str(e)}")
            continue
    
    print(f"生成完成，共生成{success_count}条同义句，结果已保存至 {output_csv}")
    if success_count == 0:
        print("没有生成任何有效的同义句，请检查输入数据")

if __name__ == "__main__":
    input_path = "/root/autodl-tmp/textClassification/data/merged_data_cleaned.csv"
    output_path = "/root/autodl-tmp/textClassification/data/sample.csv"
    generate_synonyms_csv(input_path, output_path)