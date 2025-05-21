import pandas as pd
from pyecharts.charts import Bar, Grid
from pyecharts import options as opts
from pyecharts.charts import Timeline

# 预处理数据（新增部分）
def preprocess_region_data(df):
    # 提取一级分类
    df['一级分类'] = df['问题分类'].str.split('->').str[0].str.strip()
    # 填充空值并过滤无效数据
    df['被反映区'] = df['被反映区'].fillna('未指定区').str.replace('区', '')
    return df.groupby(['被反映区', '一级分类']).size().reset_index(name='数量')

# 生成区域分析图表
def create_region_chart():
    df = pd.read_csv('../data/merged_data_cleaned.csv')
    region_data = preprocess_region_data(df)
    
    # 获取各区数据量排序
    region_order = region_data.groupby('被反映区')['数量'].sum().sort_values(ascending=False).index.tolist()

    # 创建时间轴（模拟按钮切换）
    timeline = Timeline(
        init_opts=opts.InitOpts(width='1400px', height='600px', theme='essos')  # 加宽整体布局
    )
    timeline.add_schema(
        axis_type='category',
        orient='vertical',
        symbol='roundRect',
        play_interval=3000,
        is_auto_play=True,
        is_loop_play=True,
        pos_left='5%',  # 从15%调整为5%
        pos_bottom='15%',
        width=150,  # 增加时间轴宽度
        height='70%'  # 增加时间轴高度
    )

    # 生成各区域图表
    for region in region_order[:15]:
        data = region_data[region_data['被反映区'] == region].nlargest(5, '数量')
        
        bar = (
            Bar()
            .add_xaxis(data['一级分类'].tolist())
            .add_yaxis("数量", data['数量'].tolist(),
                     category_gap="60%",
                     label_opts=opts.LabelOpts(position="right"))
            .reversal_axis()
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"{region} TOP5 问题分布",
                    pos_left="center"  # 删除subtitle参数
                ),
                xaxis_opts=opts.AxisOpts(
                    name="数量",
                    splitline_opts=opts.SplitLineOpts(is_show=True)),
                yaxis_opts=opts.AxisOpts(
                    name="问题分类",
                    axislabel_opts=opts.LabelOpts(font_size=12))
            )
        )
        timeline.add(bar, region)

    timeline.render("region_top5_interactive.html")

if __name__ == "__main__":
    create_region_chart()
