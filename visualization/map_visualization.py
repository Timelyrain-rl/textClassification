import pandas as pd
from pyecharts.charts import Map
from pyecharts import options as opts

# 读取数据
df = pd.read_csv('../data/merged_data_cleaned.csv')

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
        max_=3000,  # 设置最大值上限
        is_piecewise=True,
        range_text=["高", "低"],
        pieces=[
            {"min": 2001, "label": '2001-3000', "color": "#8B0000"},  # 深红
            {"min": 1001, "max": 2000, "label": '1001-2000', "color": "#FF6347"}, # 橙红
            {"min": 501, "max": 1000, "label": '501-1000', "color": "#7B68EE"}, # 过渡紫
            {"min": 101, "max": 500, "label": '101-500', "color": "#4169E1"},   # 皇家蓝
            {"min": 0, "max": 100, "label": '0-100', "color": "#87CEEB"}      # 天蓝
        ]
    ),
    tooltip_opts=opts.TooltipOpts(
        trigger="item", 
        formatter="{b}<br/>案件数量: {c}")
)

# 保存为HTML文件（可交互）
m.render("district_distribution.html")
