from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
from multiprocessing import Pool
import os
from tqdm import tqdm

class BackTranslator:
    def __init__(self, model_path="/root/autodl-tmp/textClassification/models/opus-mt-zh-en"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.en_model = AutoModelForSeq2SeqLM.from_pretrained("/root/autodl-tmp/textClassification/models/opus-mt-en-zh").to(self.device)
        self.en_tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/textClassification/models/opus-mt-en-zh")

    def translate_zh_to_en(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate_en_to_zh(self, text):
        inputs = self.en_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.en_model.generate(**inputs, max_length=512)
        return self.en_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def augment_text(self, text):
        try:
            en_text = self.translate_zh_to_en(text)
            back_translated = self.translate_en_to_zh(en_text)
            return back_translated
        except Exception as e:
            print(f"翻译失败: {str(e)}")
            return None

def batch_augment(args):
    translator, text = args
    try:
        return translator.augment_text(text)
    except Exception as e:
        print(f"处理失败: {str(e)}")
        return None

def main():
    # 初始化翻译器
    translator = BackTranslator()
    
    # 读取原始数据
    input_path = "/root/autodl-tmp/textClassification/data/merged_data_cleaned.csv"
    df = pd.read_csv(input_path)
    
    # 随机抽取10条样本
    sampled_df = df.sample(n=10, random_state=42)  # 新增抽样代码
    print("已抽取10条样本数据:")
    print(sampled_df[['主要内容']].head(10))
    
    # 准备多进程任务
    tasks = [(translator, row['主要内容']) for _, row in sampled_df.iterrows()]
    # 使用spawn上下文创建进程池

    ctx = torch.multiprocessing.get_context('spawn')  # 新增上下文设置
    with ctx.Pool(processes=8) as pool:  # 修改为使用spawn上下文
        results = list(tqdm(pool.imap(batch_augment, tasks), total=len(tasks)))
    
    # 合并增强数据
    augmented_df = sampled_df.copy()
    augmented_df['主要内容'] = [res if res else orig for res, orig in zip(results, sampled_df['主要内容'])]
    
    # 保存增强结果
    output_dir = "/root/autodl-tmp/textClassification/data/augmented"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "back_translated_sample.csv")  # 修改输出文件名
    augmented_df.to_csv(output_path, index=False)
    print(f"样本增强数据已保存至: {output_path}")

if __name__ == "__main__":
    main()