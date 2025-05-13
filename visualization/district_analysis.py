import pandas as pd
from pyecharts.charts import Bar, Page
from pyecharts import options as opts
from pyecharts.components import Table

# 读取数据
df = pd.read_csv('/root/autodl-tmp/textClassification/data/merged_data_cleaned.csv')

# 数据预处理
df = df[df['被反映区'].notna()]  # 过滤空值区
district_data = df.groupby('被反映区')['一级分类'].value_counts().groupby(level=0, group_keys=False).nlargest(5)

# 生成可视化页面
page = Page(page_title="各区问题分布TOP5", layout=Page.SimplePageLayout)

# 生成表格汇总
table = Table()
headers = ["行政区", "TOP1", "TOP2", "TOP3", "TOP4", "TOP5"]
rows = []
for district in district_data.index.levels[0]:
    top5 = district_data[district].index.tolist()
    rows.append([district] + top5 + [""]*(5-len(top5)))  # 填充空值保持表格对齐
table.add(headers, rows)
page.add(table)

# 生成交互式柱状图
for district in district_data.index.levels[0]:
    data = district_data[district].reset_index()
    bar = (
        Bar(init_opts=opts.InitOpts(width="1200px"))
        .add_xaxis(data['一级分类'].tolist())
        .add_yaxis("问题数量", data['count'].tolist())
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"{district}问题分类TOP5"),
            datazoom_opts=[opts.DataZoomOpts(type_="inside")],
            toolbox_opts=opts.ToolboxOpts(
                feature={
                    "saveAsImage": {},
                    "dataView": {},
                    "magicType": {"type": ["line", "bar"]}
                })
        )
        .set_series_opts(
            label_opts=opts.LabelOpts(position="top"),
            markpoint_opts=opts.MarkPointOpts(
                data=[
                    opts.MarkPointItem(type_="max", name="最大值"),
                    opts.MarkPointItem(type_="min", name="最小值")
                ])
        )
    )
    page.add(bar)

# 渲染页面
page.render("district_analysis.html")
