import pandas as pd
from pyecharts.charts import Bar
from pyecharts import options as opts

def generate_region_top3():
    # 读取数据
    df = pd.read_csv('../data/merged_data_cleaned.csv')
    
    # 预处理数据
    df['一级分类'] = df['问题分类'].str.split('->').str[0].str.strip()
    df['被反映区'] = df['被反映区'].fillna('未指定区').str.replace('区', '')
    
    # 过滤非目标区域
    filtered_df = df[~df['被反映区'].isin(['未指定区', '非辖属'])]
    
    # 按区统计TOP3分类
    region_data = filtered_df.groupby(['被反映区', '一级分类']).size() \
                            .groupby('被反映区', group_keys=False) \
                            .nlargest(3) \
                            .reset_index(name='数量')
    
    # 生成图表
    bar = (
        Bar(init_opts=opts.InitOpts(width='1400px', height='800px'))
        .add_xaxis(region_data['被反映区'].unique().tolist())
        .add_yaxis("TOP1", region_data[region_data.groupby('被反映区').cumcount() == 0]['数量'].tolist())
        .add_yaxis("TOP2", region_data[region_data.groupby('被反映区').cumcount() == 1]['数量'].tolist())
        .add_yaxis("TOP3", region_data[region_data.groupby('被反映区').cumcount() == 2]['数量'].tolist())
        .set_global_opts(
            title_opts=opts.TitleOpts(title="各区TOP3问题分类分布"),
            xaxis_opts=opts.AxisOpts(name="行政区", axislabel_opts=opts.LabelOpts(rotate=45)),
            yaxis_opts=opts.AxisOpts(name="案件数量"),
            datazoom_opts=[opts.DataZoomOpts()]
        )
        .render("region_top3_bar.html")
    )

if __name__ == "__main__":
    generate_region_top3()
    print("图表已生成：region_top3_bar.html")