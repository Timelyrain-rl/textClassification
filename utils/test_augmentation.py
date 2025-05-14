import pandas as pd
import os

def sample_data(input_path, output_path, sample_size=5):
    """随机抽取样本进行测试"""
    try:
        df = pd.read_csv(input_path)
        sampled_df = df.sample(n=sample_size, random_state=42)
        
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        sampled_df.to_csv(output_path, index=False)
        print(f"成功抽取{sample_size}条样本保存至：{os.path.abspath(output_path)}")
    except FileNotFoundError:
        print(f"错误：未找到输入文件 {input_path}")
    except Exception as e:
        print(f"发生异常：{str(e)}")

if __name__ == "__main__":
    # 测试配置
    sample_data(
        input_path="../data/merged_data_cleaned.csv",  # 原始数据路径
        output_path="../data/test_samples.csv",        # 测试样本保存路径
        sample_size=5                                 # 抽取数量
    )