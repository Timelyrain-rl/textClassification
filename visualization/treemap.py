from pyecharts import options as opts
from pyecharts.charts import TreeMap
import pandas as pd

# 读取CSV数据
df = pd.read_csv("/root/autodl-tmp/textClassification/data/merged_data_cleaned.csv")

# 统计三级分类数据
hierarchy_counts = df.groupby(['一级分类', '二级分类', '三级分类']).size().reset_index(name='value')

# 构建树形数据结构
data = []
for (l1, l1_group) in hierarchy_counts.groupby('一级分类'):
    l1_total = l1_group['value'].sum()
    children_l2 = []
    
    for (l2, l2_group) in l1_group.groupby('二级分类'):
        l2_total = l2_group['value'].sum()
        children_l3 = [
            {"value": row['value'], "name": f"{row['三级分类']}\n{row['value']}"}
            for _, row in l2_group.iterrows()
        ]
        
        children_l2.append({
            "value": l2_total,
            "name": f"{l2}\n{l2_total}",
            "children": children_l3
        })
    
    data.append({
        "value": l1_total,
        "name": f"{l1}\n{l1_total}",
        "children": children_l2
    })

# 创建矩形树图
tm = (
    TreeMap(init_opts=opts.InitOpts(width="1600px", height="900px"))
    .add(
        series_name="分类统计",
        data=data,
        levels=[
            opts.TreeMapLevelsOpts(
                treemap_itemstyle_opts=opts.TreeMapItemStyleOpts(
                    border_color="#555", border_width=4, gap_width=4
                )
            ),
            opts.TreeMapLevelsOpts(
                color_saturation=[0.3, 0.6],
                treemap_itemstyle_opts=opts.TreeMapItemStyleOpts(
                    border_color="#666", border_width=2, gap_width=2
                )
            ),
            opts.TreeMapLevelsOpts(
                color_saturation=[0.3, 0.5],
                treemap_itemstyle_opts=opts.TreeMapItemStyleOpts(
                    border_color="#777", border_width=1, gap_width=1
                )
            ),
        ]
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="工单分类统计树图", pos_left="center"),
        legend_opts=opts.LegendOpts(is_show=False)
    )
)

tm.render("treemap.html")
