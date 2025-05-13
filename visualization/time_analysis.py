import pandas as pd
from pyecharts.charts import Bar
from pyecharts import options as opts

# 读取数据
df = pd.read_csv('/root/autodl-tmp/textClassification/data/merged_data_cleaned.csv')

# 处理时间和政务分类
df['创建时间'] = pd.to_datetime(df['创建时间'])
df['小时'] = df['创建时间'].dt.hour
df['政务分类'] = df['政务分类'].fillna('非政务')  # 处理空值

# 统计各时段政务/非政务数量
time_dist = df.groupby(['小时', '政务分类']).size().unstack(fill_value=0)

# 创建柱状图
# 原版绝对值图表
bar = Bar(init_opts=opts.InitOpts(width='1200px', height='600px'))
bar.add_xaxis([f"{h:02d}:00" for h in range(24)])
# 原版绝对值图表修改处
bar.add_yaxis("政务类", time_dist['政务'].tolist())  # 移除 stack 参数
bar.add_yaxis("非政务类", time_dist['非政务'].tolist())  # 移除 stack 参数

# 新增图表配置选项
bar.set_global_opts(
    title_opts=opts.TitleOpts(title="政务工单时间分布分析"),
    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow"),
    xaxis_opts=opts.AxisOpts(
        name="时间段",
        axislabel_opts=opts.LabelOpts(rotate=-45),  # 添加标签旋转
        splitline_opts=opts.SplitLineOpts(is_show=True)),
    yaxis_opts=opts.AxisOpts(
        name="工单数量",
        splitline_opts=opts.SplitLineOpts(is_show=True)),
)

# 设置系列选项时添加 category_gap
bar.set_series_opts(
    label_opts=opts.LabelOpts(is_show=False),
    markpoint_opts=opts.MarkPointOpts(
        data=[
            opts.MarkPointItem(type_="max", name="最大值"),
            opts.MarkPointItem(type_="min", name="最小值")
        ]),
    category_gap="50%"  # 控制柱间间距
)

bar.render("time_distribution.html")

# 新增归一化版本图表 --------------------------------------------------
# 计算百分比 (确保每行相加为100%)
# 修改归一化计算部分
# 原代码：
# total = time_dist.sum(axis=1)
# norm_dist = time_dist.div(total, axis=0) * 100

# 改为与Matplotlib一致的归一化方式
norm_dist = time_dist.div(time_dist.sum(axis=1), axis=0) * 100  # 按行归一化
norm_dist = norm_dist.round(2)

# 添加有效性验证
assert not norm_dist.isnull().values.any(), "存在空值，请检查原始数据"
assert (abs(norm_dist.sum(axis=1) - 100) < 0.1).all(), "归一化误差超过阈值"

# 修改后的正确顺序
norm_bar = Bar(init_opts=opts.InitOpts(width='1200px', height='600px'))
norm_bar.add_xaxis([f"{h:02d}:00" for h in range(24)])
norm_bar.add_yaxis("政务类", norm_dist['政务'].tolist(), stack='stack1')
norm_bar.add_yaxis("非政务类", norm_dist['非政务'].tolist(), stack='stack1')

# 配置图表选项应放在创建实例之后
# 修改归一化计算部分（全局归一化）
total = time_dist.sum().sum()  # 计算所有时段的工单总数
norm_dist = (time_dist / total * 100).round(2)  # 全局归一化

# 验证总和为100% ±0.1%
assert abs(norm_dist.sum().sum() - 100) < 0.1, "全局归一化误差超过阈值"

# 修改图表配置
norm_bar.set_global_opts(
    title_opts=opts.TitleOpts(title="政务工单24小时分布（全局归一化）"),
    tooltip_opts=opts.TooltipOpts(
        trigger="axis",
        formatter="""
        function(params) {
            return params[0].name + '<br/>'
                   + params[0].seriesName + ': ' + params[0].value.toFixed(2) + '%<br/>'
                   + params[1].seriesName + ': ' + params[1].value.toFixed(2) + '%<br/>'
                   + '时段占比: ' + (params[0].value + params[1].value).toFixed(2) + '%';
        }
        """
    ),
    yaxis_opts=opts.AxisOpts(
        name="全局占比 (%)",
        axislabel_opts=opts.LabelOpts(formatter="{value}%"),
        splitline_opts=opts.SplitLineOpts(is_show=True))
)

norm_bar.set_series_opts(
    label_opts=opts.LabelOpts(formatter="{c}%")
)

norm_bar.render("time_distribution_normalized.html")

# 修改后的计算方式（保留更多小数位）
# 删除旧的按行归一化代码，仅保留全局归一化部分

# 全局归一化计算 --------------------------------------------------
# 计算所有时段的工单总数
total = time_dist.sum().sum()
# 计算全局占比（保留3位小数）
norm_dist = (time_dist / total * 100).round(3)

# 验证全局总和为100% ±0.1%
assert abs(norm_dist.sum().sum() - 100) < 0.1, "全局归一化误差超过阈值"

# 创建归一化图表
# 计算独立归一化
def normalize_column(col):
    return (col / col.sum() * 100).round(2)

norm_dist = pd.DataFrame({
    '政务': normalize_column(time_dist['政务']),
    '非政务': normalize_column(time_dist['非政务'])
})

# 验证每个类别的总和
assert abs(norm_dist['政务'].sum() - 100) < 0.1, "政务类归一化错误"
assert abs(norm_dist['非政务'].sum() - 100) < 0.1, "非政务类归一化错误"

# 修改图表配置
norm_bar = Bar(init_opts=opts.InitOpts(width='1200px', height='600px'))
norm_bar.add_xaxis([f"{h:02d}:00" for h in range(24)])
# 修改归一化图表的系列添加方式
norm_bar.add_yaxis("政务类", norm_dist['政务'].tolist())  # 移除 stack 参数
norm_bar.add_yaxis("非政务类", norm_dist['非政务'].tolist())  # 移除 stack 参数

# 在系列配置中添加间距控制
norm_bar.set_series_opts(
    label_opts=opts.LabelOpts(formatter="{c}%"),
    category_gap="50%"  # 控制柱间间距
)

norm_bar.set_global_opts(
    title_opts=opts.TitleOpts(title="政务工单时间分布（类别独立归一化）"),
    tooltip_opts=opts.TooltipOpts(
        trigger="axis",
        formatter="""
        function(params) {
            return params[0].name + '<br/>'
                   + params[0].seriesName + ': ' + params[0].value + '%<br/>'
                   + params[1].seriesName + ': ' + params[1].value + '%';
        }
        """
    ),
    yaxis_opts=opts.AxisOpts(
        name="类别占比 (%)",
        axislabel_opts=opts.LabelOpts(formatter="{value}%"))
)

norm_bar.render("time_distribution_normalized.html")