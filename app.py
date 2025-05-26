from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import pandas as pd
import os

app = Flask(__name__)

# 全局变量，用于存储数据
df = None

@app.route('/')
def index():
    return redirect(url_for('page', page_num=1))

@app.route('/page/<int:page_num>')
def page(page_num):
    global df
    # 加载数据（如果尚未加载）
    if df is None:
        df = pd.read_csv('data/merged_data_cleaned.csv')
    
    # 计算分页信息
    total_records = len(df)
    records_per_page = 20
    total_pages = (total_records + records_per_page - 1) // records_per_page
    
    # 确保页码在有效范围内
    if page_num < 1:
        page_num = 1
    elif page_num > total_pages:
        page_num = total_pages
    
    # 计算当前页的数据范围
    start_idx = (page_num - 1) * records_per_page
    end_idx = min(start_idx + records_per_page, total_records)
    
    # 获取当前页的数据
    page_data = df.iloc[start_idx:end_idx]
    # 选择需要的列
    page_data = page_data[['创建时间', '工单编号', '主要内容', '一级分类', '二级分类', '三级分类']]
    
    return render_template('index.html', 
                           data=page_data.to_dict(orient='records'),
                           current_page=page_num,
                           total_pages=total_pages)

@app.route('/save', methods=['POST'])
def save_changes():
    try:
        # 获取前端发送的数据
        data = request.json
        work_id = data.get('work_id')  # 工单编号
        level1 = data.get('level1')    # 一级分类
        level2 = data.get('level2')    # 二级分类
        level3 = data.get('level3')    # 三级分类
        
        # 验证数据
        if not all([work_id, level1, level2, level3]):
            return jsonify({'status': 'error', 'message': '数据不完整'}), 400
        
        # 准备保存的数据
        verification_file = 'data/verification.csv'
        new_data = pd.DataFrame({
            '工单编号': [work_id],
            '一级分类': [level1],
            '二级分类': [level2],
            '三级分类': [level3],
            '修改时间': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
        })
        
        # 检查文件是否存在，如果不存在则创建
        if not os.path.exists(verification_file):
            new_data.to_csv(verification_file, index=False, encoding='utf-8')
        else:
            # 增量保存（追加模式）
            new_data.to_csv(verification_file, mode='a', header=False, index=False, encoding='utf-8')
        
        return jsonify({'status': 'success', 'message': '数据已保存'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 添加访问visualization目录下HTML文件的路由
@app.route('/visualization/<filename>')
def visualization(filename):
    visualization_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualization')
    return send_from_directory(visualization_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)