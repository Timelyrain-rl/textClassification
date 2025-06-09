import pandas as pd
from pyecharts.charts import WordCloud, Timeline
from pyecharts import options as opts
import jieba
from collections import Counter

# 加载停用词
def load_stopwords():
    with open('../data/ChineseStopWords.txt', 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f])

def generate_area_wordcloud_carousel():
    # 读取数据
    try:
        # 尝试读取visualization.csv
        df = pd.read_csv('../data/visualization.csv')
        # 确保列名匹配
        if 'major_content' in df.columns and 'area_name' in df.columns:
            content_col = 'major_content'
            area_col = 'area_name'
        else:
            # 尝试使用中文列名
            content_col = '主要内容'
            area_col = '被反映区'
    except Exception as e:
        print(f"无法读取visualization.csv: {e}")
        # 尝试读取替代文件
        df = pd.read_csv('../data/merged_data_cleaned.csv')
        content_col = '主要内容'
        area_col = '被反映区'
    
    # 预处理数据
    df[area_col] = df[area_col].fillna('未指定区').str.replace('区', '')
    
    # 过滤非目标区域
    filtered_df = df[~df[area_col].isin(['未指定区', '非辖属'])]
    
    # 加载停用词
    stopwords = load_stopwords()
    
    # 创建时间轴对象，设置初始化选项以控制整体布局
    timeline = Timeline(init_opts=opts.InitOpts(width="1200px", height="800px"))
    
    # 修改时间轴配置，将其放在左侧并拉长
    timeline.add_schema(
        orient="vertical",  # 垂直方向的时间轴
        is_auto_play=True,
        is_loop_play=True,
        play_interval=3000,  # 3秒切换一次
        pos_left="5%",      # 放在左侧
        pos_right="85%",    # 控制时间轴宽度
        pos_top="10%",      # 顶部留出空间
        pos_bottom="10%",   # 底部留出空间
        axis_type="category",
        symbol_size=10,     # 时间点大小
        linestyle_opts=opts.LineStyleOpts(width=2),  # 时间轴线条样式
        label_opts=opts.LabelOpts(interval=0, font_size=12),  # 标签样式
        itemstyle_opts=opts.ItemStyleOpts(color="#005cc5"),  # 时间点样式
    )
    
    # 为每个区域生成词云图
    for area in filtered_df[area_col].unique():
        # 获取该区域的所有文本
        area_text = ' '.join(filtered_df[filtered_df[area_col] == area][content_col].dropna().astype(str))
        
        # 分词并过滤停用词
        words = [word for word in jieba.lcut(area_text) 
                if len(word) > 1 and word not in stopwords]  # 添加停用词过滤
        
        # 统计词频
        word_counts = Counter(words).most_common(100)  # 每个区域取前100个高频词
        
        if not word_counts:  # 如果没有足够的词，跳过该区域
            continue
            
        # 生成词云，设置位置在右侧
        wordcloud = (
            WordCloud(init_opts=opts.InitOpts(width="1000px", height="700px"))
            .add(
                "", 
                word_counts,
                word_size_range=[15, 80],
                shape="circle",
                textstyle_opts=opts.TextStyleOpts(font_family="SimHei"),
                pos_left="25%",  # 将词云图向右移动
                pos_right="5%",  # 右侧留出空间
                pos_top="10%",   # 顶部留出空间
                pos_bottom="10%" # 底部留出空间
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"{area}区域诉求内容词云",
                    pos_left="center",
                    pos_top="5%"
                ),
                tooltip_opts=opts.TooltipOpts(is_show=True),
            )
        )
        
        # 添加到时间轴
        timeline.add(wordcloud, area)
    
    # 渲染HTML文件
    timeline.render("area_wordcloud_carousel.html")
    print("词云轮播图已生成：area_wordcloud_carousel.html")

if __name__ == "__main__":
    generate_area_wordcloud_carousel()