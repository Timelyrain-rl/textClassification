import spacy
import time

nlp = spacy.load('zh_core_web_trf')

text = """
企业反映....
"""

text_length = len(text)
print(f"文本长度：{text_length}个字符")

start_time = time.time()

doc = nlp(text)

end_time = time.time()
processing_time = end_time - start_time
print(f"处理耗时：{processing_time:.2f}秒")

print("命名实体识别结果：")
print("-" * 50)
for ent in doc.ents:
    print(f"实体文本: {ent.text}")
    print(f"实体类型: {ent.label_}")
    print(f"起始位置: {ent.start_char}")
    print(f"结束位置: {ent.end_char}")
    print("-" * 30)

# 定义一个更友好的中文命名实体识别函数
def perform_chinese_ner(text):
    # 计算文本长度
    text_length = len(text)
    print(f"输入文本长度：{text_length}个字符")
    
    # 开始计时
    start_time = time.time()
    
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entity = {
            'text': ent.text,
            'type': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        }
        entities.append(entity)
    
    # 计算处理时间
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"处理耗时：{processing_time:.2f}秒")
    
    return entities
