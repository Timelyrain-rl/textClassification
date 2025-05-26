import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("../data/merged_data_cleaned.csv")

def plot_hierarchy_counts():
    """绘制分层分类数量统计图"""
    # 统计各层级唯一值数量
    level_counts = {
        'L1': df['一级分类'].nunique(),
        'L2': df['二级分类'].nunique(),
        'L3': df['三级分类'].nunique()
    }
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    bars = plt.barh(list(level_counts.keys()), list(level_counts.values()))
    
    # 添加数据标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + 0.3, f'{int(width)}', va='center')
    
    # 设置图表样式
    plt.title('分类层级数量统计')
    plt.xlabel('类别数量')
    plt.ylabel('分类层级')
    plt.xlim(0, max(level_counts.values())*1.2)
    plt.grid(axis='x', linestyle='--')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('hierarchy_counts.png')
    plt.close()

if __name__ == "__main__":
    plot_hierarchy_counts()