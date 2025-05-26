import pandas as pd
from pyecharts.charts import WordCloud
from pyecharts import options as opts
import jieba
from collections import Counter

# 读取数据
df = pd.read_csv('../data/merged_data_cleaned.csv')

# 加载停用词
def load_stopwords():
    with open('../data/ChineseStopWords.txt', 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f])

stopwords = load_stopwords()

def generate_wordcloud():
    # 合并文本内容并分词
    text = ' '.join(df['主要内容'].dropna().astype(str))
    words = [word for word in jieba.lcut(text) 
            if len(word) > 1 and word not in stopwords]  # 添加停用词过滤
    
    # 统计词频
    word_counts = Counter(words).most_common(200)
    
    # 生成词云
    c = (
        WordCloud()
        .add("", 
             word_counts,
             word_size_range=[20, 100],
             shape="circle",
             textstyle_opts=opts.TextStyleOpts(font_family="SimHei"))
        .set_global_opts(title_opts=opts.TitleOpts(title="诉求内容词云"))
    )
    
    # 保存为HTML文件
    c.render("wordcloud.html")

if __name__ == "__main__":
    generate_wordcloud()