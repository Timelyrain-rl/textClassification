import pandas as pd
from pyecharts.charts import Bar, Tab
from pyecharts import options as opts

def generate_region_distribution():
    # 读取数据
    df = pd.read_csv('../data/merged_data_cleaned.csv')
    
    # 预处理数据
    df['一级分类'] = df['问题分类'].str.split('->').str[0].str.strip()
    df['被反映区'] = df['被反映区'].fillna('未指定区').str.replace('区', '')
    
    # 过滤非目标区域并分组统计
    filtered_df = df[~df['被反映区'].isin(['未指定区', '非辖属'])]
    region_data = filtered_df.groupby(['被反映区', '一级分类']).size().reset_index(name='数量')
    
    # 创建分页选项卡
    tab = Tab()
    
    # 为每个行政区生成独立图表
    for region in filtered_df['被反映区'].unique():
        region_df = region_data[region_data['被反映区'] == region].sort_values('数量', ascending=True)
        
        bar = (
            Bar(init_opts=opts.InitOpts(width='1200px', height='600px'))
            .add_xaxis(region_df['一级分类'].tolist())
            .add_yaxis("案件数量", region_df['数量'].tolist())
            .reversal_axis()
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{region} - 问题分类分布"),
                xaxis_opts=opts.AxisOpts(name="数量"),
                yaxis_opts=opts.AxisOpts(
                    name="问题分类",
                    axislabel_opts=opts.LabelOpts(font_size=12)),
                datazoom_opts=[opts.DataZoomOpts(
                    range_start=0,
                    range_end=100,
                    start_value=0,
                    end_value=100
                )]
            )
        )
        tab.add(bar, region)
    
    tab.render("region_class_distribution.html")

if __name__ == "__main__":
    generate_region_distribution()
