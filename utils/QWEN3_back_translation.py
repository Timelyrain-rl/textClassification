import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from tqdm import tqdm
from multiprocessing import Pool

class BackTranslator:
    def __init__(self, model_path="/root/autodl-tmp/textClassification/models/Qwen3-4B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )
        
    def augment_text(self, text, max_length=512, temperature=0.7):
        prompt = f"""请对以下文本进行同义改写，保持语义不变但用不同的表达方式。只需输出改写后的文本。
原始文本：{text}
改写文本："""
        
        # 关闭思维模式的关键修改
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            use_thinking=False  # ← 修改这里关闭思维模式
        )
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            repetition_penalty=1.1,
            do_sample=True
        )
        # 简化输出解析
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).split("改写文本：")[-1].strip()

def batch_augment(args):
    """多进程批处理函数"""
    translator, text = args  # ← 修改参数接收方式
    try:
        augmented = translator.augment_text(text)
        return augmented  # ← 直接返回增强文本
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return None

def main():
    # 新增启动方法设置
    import multiprocessing
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    
    # 初始化增强器
    translator = BackTranslator()
    
    # 读取原始数据
    input_path = "/root/autodl-tmp/textClassification/data/merged_data_cleaned.csv"
    df = pd.read_csv(input_path)
    
    # 新增随机采样5条（保持可重复性）
    test_sample = df.sample(n=5, random_state=42)  # ← 添加这行实现随机采样
    print("已随机选择5条样本进行测试...")
    
    # 准备多进程参数（简化参数传递）
    tasks = [(translator, row['主要内容']) for _, row in test_sample.iterrows()]  # ← 仅传递主要内容
    
    # 使用多进程加速
    augmented_data = []
    with Pool(processes=16) as pool:
        results = list(tqdm(pool.imap(batch_augment, tasks), total=len(tasks)))
    
    # 收集结果（仅保留有效增强文本）
    augmented_data = [{'主要内容': res} for res in results if res]  # ← 简化数据结构
    
    # 保存增强数据
    output_dir = "/root/autodl-tmp/textClassification/data/augmented"
    os.makedirs(output_dir, exist_ok=True)
    
    # 修改输出路径为测试文件
    output_path = f"{output_dir}/augmented_data_sample.csv"  # ← 添加_sample后缀
    pd.DataFrame(augmented_data).to_csv(output_path, index=False)
    print(f"生成{len(augmented_data)}条增强数据（测试样本），已保存至：{output_path}")

if __name__ == "__main__":
    main()