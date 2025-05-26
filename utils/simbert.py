from nlpcda import Simbert
import os
import tensorflow as tf
import pandas as pd
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
    
    # 新建列存储生成结果和置信度
    df['生成文本'] = None
    df['置信度'] = None
    
    # 处理每个主要内容
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            if pd.notnull(row['主要内容']):
                synonyms = simbert.replace(sent=row['主要内容'], create_num=3)
                # 分别存储生成的文本和置信度
                texts = [text for text, score in synonyms]
                scores = [score for text, score in synonyms]
                df.at[idx, '生成文本'] = texts
                df.at[idx, '置信度'] = scores
        except Exception as e:
            print(f"生成失败[{idx}]: {str(e)}")
            continue
    
    # 展开生成结果到多行，同时保持置信度对应
    exploded_df = pd.DataFrame({
        col: df[col].repeat(df['生成文本'].str.len())
        for col in df.columns if col not in ['生成文本', '置信度']
    })
    
    # 展开生成文本和置信度列
    exploded_df['生成文本'] = [text for texts in df['生成文本'].dropna() for text in texts]
    exploded_df['置信度'] = [f"{score:.2f}" for scores in df['置信度'].dropna() for score in scores]
    
    # 保存结果
    exploded_df.to_csv(output_csv, index=False)
    print(f"生成完成，结果已保存至 {output_csv}")

if __name__ == "__main__":
    input_path = "/root/autodl-tmp/textClassification/data/test_samples.csv"
    output_path = "/root/autodl-tmp/textClassification/data/sample.csv"
    generate_synonyms_csv(input_path, output_path)