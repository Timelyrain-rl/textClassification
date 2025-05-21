from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    # 随机抽取30条数据
    df = pd.read_csv('data/merged_data_cleaned.csv').sample(n=30, random_state=1)
    # 选择需要的列
    data = df[['创建时间', '工单编号', '主要内容', '一级分类', '二级分类', '三级分类']]
    return render_template('index.html', data=data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)