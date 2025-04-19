import pandas as pd

# 读取原始数据
df = pd.read_csv('merged_data_cleaned.csv')

unique_categories = df['问题分类'].drop_duplicates()
unique_categories.to_csv('unique_question_categories.csv', 
                        index=False, 
                        header=['问题分类'])