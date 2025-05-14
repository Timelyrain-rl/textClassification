import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据（与treemap.py使用相同数据源）
df = pd.read_csv("/root/autodl-tmp/textClassification/data/merged_data_cleaned.csv")

# 一级分类柱状图
def plot_level1():
    level1_counts = df['一级分类'].value_counts().sort_values()
    plt.figure(figsize=(10, 6))
    level1_counts.plot(kind='barh')
    plt.title('一级分类工单数量分布')
    plt.xlabel('数量')
    plt.ylabel('分类')
    plt.tight_layout()
    plt.savefig('level1_classification.png')
    plt.close()

# 二级分类TOP5/倒TOP5
def plot_level2():
    level2_counts = df.groupby(['一级分类', '二级分类']).size().reset_index(name='count')
    
    # 全局TOP5
    top5 = level2_counts.sort_values('count', ascending=False).head(20)
    # 全局倒TOP5 
    bottom5 = level2_counts.sort_values('count').head(20)

    # 绘制TOP5
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([f"{row['一级分类']}-{row['二级分类']}" for _, row in top5.iterrows()], top5['count'])
    ax.set_title('TOP5 二级分类统计')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('top5_level2.png')
    plt.close()

    # 绘制倒TOP5（同上，修改数据源）
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([f"{row['一级分类']}-{row['二级分类']}" for _, row in bottom5.iterrows()], bottom5['count'])
    ax.set_title('倒TOP5 二级分类统计')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('bottom5_level2.png')
    plt.close()

# 三级分类TOP5/倒TOP5（实现逻辑与二级分类类似）
def plot_level3():
    level3_counts = df.groupby(['一级分类', '二级分类', '三级分类']).size().reset_index(name='count')
    
    top5 = level3_counts.sort_values('count', ascending=False).head(50)
    bottom5 = level3_counts.sort_values('count').head(50)

    # TOP5图表
    fig, ax = plt.subplots(figsize=(14, 7))
    labels = [f"{row['一级分类']}-{row['二级分类']}\n{row['三级分类']}" for _, row in top5.iterrows()]
    ax.bar(labels, top5['count'])
    ax.set_title('TOP5 三级分类统计')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('top5_level3.png')
    plt.close()

    # 倒TOP5图表
    fig, ax = plt.subplots(figsize=(14, 7))
    labels = [f"{row['一级分类']}-{row['二级分类']}\n{row['三级分类']}" for _, row in bottom5.iterrows()]
    ax.bar(labels, bottom5['count'])
    ax.set_title('倒TOP5 三级分类统计')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('bottom5_level3.png')
    plt.close()

if __name__ == "__main__":
    plot_level1()
    plot_level2()
    plot_level3()
    print("可视化图片已生成：level1_classification.png, top5_level2.png, bottom5_level2.png, top5_level3.png, bottom5_level3.png")
