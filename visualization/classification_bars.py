import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据（与treemap.py使用相同数据源）
df = pd.read_csv("../data/merged_data_cleaned.csv")

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

def plot_level2():
    level2_counts = df.groupby(['一级分类', '二级分类']).size().reset_index(name='count')
    
    top20 = level2_counts.sort_values('count', ascending=False).head(20)
    bottom20 = level2_counts.sort_values('count').head(20)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([f"{row['一级分类']}-{row['二级分类']}" for _, row in top20.iterrows()], top20['count'])
    ax.set_title('TOP20 二级分类统计')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(20))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('top20_level2.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([f"{row['一级分类']}-{row['二级分类']}" for _, row in bottom20.iterrows()], bottom20['count'])
    ax.set_title('倒TOP20 二级分类统计')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(20))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('bottom20_level2.png')
    plt.close()

def plot_level3():
    level3_counts = df.groupby(['一级分类', '二级分类', '三级分类']).size().reset_index(name='count')
    
    top20 = level3_counts.sort_values('count', ascending=False).head(20)
    bottom20 = level3_counts.sort_values('count').head(20)

    fig, ax = plt.subplots(figsize=(14, 7))
    labels = [f"{row['一级分类']}-{row['二级分类']}\n{row['三级分类']}" for _, row in top20.iterrows()]
    ax.bar(labels, top20['count'])
    ax.set_title('TOP20 三级分类统计')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('top20_level3.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 7))
    labels = [f"{row['一级分类']}-{row['二级分类']}\n{row['三级分类']}" for _, row in bottom20.iterrows()]
    ax.bar(labels, bottom20['count'])
    ax.set_title('BOTTOM20 三级分类统计')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('bottom20_level3.png')
    plt.close()

if __name__ == "__main__":
    plot_level1()
    plot_level2()
    plot_level3()

