import pandas as pd
from pyecharts.charts import Map
from pyecharts import options as opts

# 读取数据
df = pd.read_csv('/root/autodl-tmp/textClassification/data/merged_data_cleaned.csv')

# 处理行政区数据（过滤空值并统计频次）
district_counts = df['被反映区'].value_counts().reset_index()
district_counts.columns = ['district', 'count']

# 转换为ECharts需要的格式
data_pairs = list(zip(district_counts['district'].tolist(), 
                     district_counts['count'].tolist()))

# 创建地图实例
m = Map(init_opts=opts.InitOpts(width='1200px', height='800px'))

# 添加地图配置
m.add("诉求分布", 
      data_pair=data_pairs,
      maptype="北京",
      is_map_symbol_show=False)

m.set_global_opts(
    title_opts=opts.TitleOpts(title="北京市民诉求分布地图"),
    visualmap_opts=opts.VisualMapOpts(
        max_=district_counts['count'].max(),
        is_piecewise=True,
        range_text=["高", "低"],
        pieces=[
            {"min": 20, "label": '20+', "color": "#B40404"},
            {"min": 15, "max": 19, "label": '15-19', "color": "#DF0100"},
            {"min": 10, "max": 14, "label": '10-14', "color": "#F78181"},
            {"min": 5, "max": 9, "label": '5-9', "color": "#F5A9A9"},
            {"min": 1, "max": 4, "label": '1-4', "color": "#FFFFCC"}
        ]
    ),
    tooltip_opts=opts.TooltipOpts(
        trigger="item", 
        formatter="{b}<br/>案件数量: {c}")
)

# 保存为HTML文件（可交互）
m.render("district_distribution.html")
