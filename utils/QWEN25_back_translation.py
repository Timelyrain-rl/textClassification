from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os
from tqdm import tqdm
from multiprocessing import Pool
import torch

class BackTranslator:
    def __init__(self, model_path="/root/autodl-tmp/textClassification/models/Qwen2.5-3B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
    def augment_text(self, text, max_length=512, temperature=0.7):
        prompt = f"""将以下文本进行同义改写，保持核心语义不变但使用不同的表达方式。直接输出改写后的文本：
原文：{text}
改写："""
        
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            padding='max_length',  # 改用固定长度padding
            truncation=True,
            max_length=512,
            return_attention_mask=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,  # 显式获取input_ids
                attention_mask=inputs.attention_mask,  # 确保传递attention_mask
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id  # 显式设置结束符
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("改写：")[-1].strip()

def process_csv(input_path, output_dir, batch_size=32):
    # 读取原始数据
    df = pd.read_csv(input_path)
    
    # 随机抽取10条样本
    sampled_df = df.sample(n=10, random_state=42)  # 新增抽样代码
    print(f"已抽取{len(sampled_df)}条样本数据:")
    
    translator = BackTranslator()
    
    # 修改多进程上下文
    ctx = torch.multiprocessing.get_context('spawn')  # 新增代码
    with ctx.Pool(processes=4) as pool:  # 修改为使用spawn上下文
        args = [(translator, text) for text in sampled_df['主要内容']]
        results = list(tqdm(pool.imap(batch_augment, args), total=len(sampled_df)))
        print(results)

    # 保存增强后的数据
    sampled_df['回译内容'] = results
    output_path = os.path.join(output_dir, f"augmented_sample_{os.path.basename(input_path)}")
    
    # 保存增强后的数据
    sampled_df['回译内容'] = results
    output_path = os.path.join(output_dir, f"augmented_sample_{os.path.basename(input_path)}")
    sampled_df.to_csv(output_path, index=False)
    print(f"增强样本已保存至: {output_path}")

# 修正属性错误（第12行）
class BackTranslator:
    def __init__(self, model_path="/root/autodl-tmp/textClassification/models/Qwen2.5-3B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 修正为device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)  # 使用正确的device属性
        self.model.eval()
        
    def augment_text(self, text, max_length=512, temperature=0.7):
        prompt = f"""将以下文本进行同义改写，保持核心语义不变但使用不同的表达方式。只输出改写后的文本：
原文：{text}
改写："""
        
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("改写：")[-1].strip()

def batch_augment(args):
    translator, text = args
    try:
        return translator.augment_text(text)
    except Exception as e:
        print(f"处理失败: {str(e)}")
        return text  # 返回原始文本作为保底

if __name__ == "__main__":
    input_csv = "/root/autodl-tmp/textClassification/data/merged_data_cleaned.csv"
    output_dir = "/root/autodl-tmp/textClassification/data/augmented/"
    process_csv(input_csv, output_dir)