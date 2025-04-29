import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

csv_file = 'merged_data_cleaned.csv'
output_plot_file = '政务分类_24小时分布.png'
df = pd.read_csv(csv_file, low_memory=False)

df['创建时间'] = pd.to_datetime(df['创建时间'], errors='coerce')
df['hour'] = df['创建时间'].dt.hour

df['政务分类'] = df['政务分类'].apply(lambda x: '政务' if x == '政务' else '非政务')

hourly_distribution = df.groupby(['hour', '政务分类']).size().unstack(fill_value=0)

plt.figure(figsize=(15, 8))

bar_width = 0.35
index = range(24)

plt.bar(index, hourly_distribution['政务'], bar_width, label='政务')
plt.bar([i + bar_width for i in index], hourly_distribution['非政务'], bar_width, label='非政务')

plt.title('政务分类_24小时分布', fontsize=16)
plt.xlabel('小时', fontsize=12)
plt.ylabel('计数', fontsize=12)
plt.xticks([i + bar_width/2 for i in index], index) 
plt.grid(True, linestyle='--', alpha=0.6)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()

try:
    plt.savefig(output_plot_file, bbox_inches='tight')
    print(f"图表已保存至: {output_plot_file}")
except Exception as e:
    print(f"保存图表时出错: {e}")
