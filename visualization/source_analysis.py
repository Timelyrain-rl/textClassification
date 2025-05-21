import pandas as pd
from pyecharts.charts import Bar
from pyecharts import options as opts

# 读取数据文件
df = pd.read_csv(r'e:\study\fourNext\CapstoneProject\textClassification\data\merged_data_cleaned.csv')

# 统计工单来源分布
source_counts = df['工单来源'].value_counts().sort_values(ascending=False)

# 创建柱状图
bar = (
    Bar()
    .add_xaxis(source_counts.index.tolist())
    .add_yaxis("工单数量", source_counts.values.tolist())
    .set_global_opts(
        title_opts=opts.TitleOpts(title="工单来源分布统计"),
        xaxis_opts=opts.AxisOpts(
            name="来源渠道",
            axislabel_opts=opts.LabelOpts(
                rotate=45,  # 倾斜45度
                interval=0,  # 显示所有标签
                font_size=12
            )
        ),
        yaxis_opts=opts.AxisOpts(name="数量"),
        datazoom_opts=[opts.DataZoomOpts()],
    )
)

# 保存为HTML文件
bar.render("source_distribution.html")