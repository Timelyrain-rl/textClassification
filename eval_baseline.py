import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.cuda.amp import autocast
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from models.baseline_classifier import BaselineClassifier
from train_baseline import TextClassificationDataset

def evaluate_model(model, eval_loader, device):
    print("开始评估基线模型...")
    model.eval()
    
    all_predictions_l1 = []
    all_predictions_l2 = []
    all_predictions_l3 = []
    all_labels_l1 = []
    all_labels_l2 = []
    all_labels_l3 = []
    
    all_logits_l1 = []  # 新增logits收集
    all_logits_l2 = []
    all_logits_l3 = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if batch_idx % 10 == 0:
                print(f"评估进度: {batch_idx}/{len(eval_loader)}")
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_l1 = batch['labels_l1']
            labels_l2 = batch['labels_l2']
            labels_l3 = batch['labels_l3']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 对于多标签分类，使用0.5作为阈值
            predictions_l1 = (outputs['l1_probs'].cpu() > 0.5).float()
            predictions_l2 = (outputs['l2_probs'].cpu() > 0.5).float()
            predictions_l3 = (outputs['l3_probs'].cpu() > 0.5).float()
            
            all_predictions_l1.extend(predictions_l1.numpy())
            all_predictions_l2.extend(predictions_l2.numpy())
            all_predictions_l3.extend(predictions_l3.numpy())
            all_labels_l1.extend(labels_l1.numpy())
            all_labels_l2.extend(labels_l2.numpy())
            all_labels_l3.extend(labels_l3.numpy())
            
            # 收集原始logits
            all_logits_l1.extend(outputs['l1_logits'].cpu().numpy())
            all_logits_l2.extend(outputs['l2_logits'].cpu().numpy())
            all_logits_l3.extend(outputs['l3_logits'].cpu().numpy())
    
    # 转换为numpy数组
    all_predictions_l1 = np.array(all_predictions_l1)
    all_predictions_l2 = np.array(all_predictions_l2)
    all_predictions_l3 = np.array(all_predictions_l3)
    all_labels_l1 = np.array(all_labels_l1)
    all_labels_l2 = np.array(all_labels_l2)
    all_labels_l3 = np.array(all_labels_l3)
    
    def calculate_metrics(y_true, y_pred, y_logits, level_name):  # 添加y_logits参数
        # 原始指标
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        y_true_single = np.argmax(y_true, axis=1)  # 假设每个样本只有一个真实标签
        y_logits = np.array(y_logits)  # 确保logits是numpy数组
        
        try:
            top3_acc = top_k_accuracy_score(y_true_single, y_logits, k=3)
            top5_acc = top_k_accuracy_score(y_true_single, y_logits, k=5)
            top10_acc = top_k_accuracy_score(y_true_single, y_logits, k=10)
        except Exception as e:
            print(f"计算Top-K时发生错误: {str(e)}")
            top3_acc = 0.0
            top5_acc = 0.0
            top10_acc = 0.0
        
        print(f"\n{level_name} 评估结果:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro-F1: {f1_macro:.4f}")
        print(f"Micro-F1: {f1_micro:.4f}") 
        print(f"Top-3 Accuracy: {top3_acc:.4f}")
        print(f"Top-5 Accuracy: {top5_acc:.4f}")
        print(f"Top-10 Accuracy: {top10_acc:.4f}")  
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc,
            'top10_accuracy': top10_acc
        }
    
    def calculate_top_k(y_true, y_logits, k=5):
        y_true_classes = np.argmax(y_true, axis=1)
        return top_k_accuracy_score(y_true_classes, y_logits, k=k)
    
    # 计算各层级的评估指标
    results = {
        'level1': calculate_metrics(all_labels_l1, all_predictions_l1, all_logits_l1, "一级分类"),
        'level2': calculate_metrics(all_labels_l2, all_predictions_l2, all_logits_l2, "二级分类"),
        'level3': calculate_metrics(all_labels_l3, all_predictions_l3, all_logits_l3, "三级分类")
    }
    
    # 添加Top-K结果
    results['level1']['top5_accuracy'] = calculate_top_k(all_labels_l1, all_logits_l1, 5)
    results['level2']['top5_accuracy'] = calculate_top_k(all_labels_l2, all_logits_l2, 5)
    results['level3']['top5_accuracy'] = calculate_top_k(all_labels_l3, all_logits_l3, 5)
    
    return {
        'results': results,
        'predictions': {
            'l1': all_predictions_l1,
            'l2': all_predictions_l2,
            'l3': all_predictions_l3
        }
    }

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    print("加载数据...")
    df = pd.read_csv('data/merged_data_cleaned.csv')
    texts = df['主要内容'].values

    # --- 修改开始：加载并处理标签 ---
    print("从DataFrame提取并预处理标签数据...")
    # 获取标签列
    raw_labels_l1 = df['一级分类'].values
    raw_labels_l2 = df['二级分类'].values
    raw_labels_l3 = df['三级分类'].values

    # 预处理标签 - 实现标签编码
    print("处理标签数据...")

    # 处理一级标签
    mlb_l1 = MultiLabelBinarizer()
    processed_labels_l1 = [label.split(',') if isinstance(label, str) else [str(label)] for label in raw_labels_l1]
    labels_l1 = mlb_l1.fit_transform(processed_labels_l1)
    num_classes_l1 = len(mlb_l1.classes_)

    # 处理二级标签
    mlb_l2 = MultiLabelBinarizer()
    processed_labels_l2 = [label.split(',') if isinstance(label, str) else [str(label)] for label in raw_labels_l2]
    labels_l2 = mlb_l2.fit_transform(processed_labels_l2)
    num_classes_l2 = len(mlb_l2.classes_)

    # 处理三级标签
    mlb_l3 = MultiLabelBinarizer()
    processed_labels_l3 = [label.split(',') if isinstance(label, str) else [str(label)] for label in raw_labels_l3]
    labels_l3 = mlb_l3.fit_transform(processed_labels_l3)
    num_classes_l3 = len(mlb_l3.classes_)

    print(f"标签维度: L1={labels_l1.shape}, L2={labels_l2.shape}, L3={labels_l3.shape}")
    print(f"类别数量: L1={num_classes_l1}, L2={num_classes_l2}, L3={num_classes_l3}")
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/textClassification/models/chinese-roberta-wwm-ext')
    
    print("初始化模型...")
    model = BaselineClassifier(
        num_labels_l1=num_classes_l1,
        num_labels_l2=num_classes_l2,
        num_labels_l3=num_classes_l3
    )

    print("加载模型权重...")
    # 加载最佳模型
    model_path = 'models/best_baseline_classifier.pth'
    print(f"从 {model_path} 加载模型...")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    print("创建评估数据加载器...")
    # 确认 max_length 是否适合 RoBERTa
    # 将处理后的标签传递给 Dataset
    eval_dataset = TextClassificationDataset(texts, labels_l1, labels_l2, labels_l3, tokenizer, max_length=512)
    eval_loader = DataLoader(eval_dataset,
                           batch_size=16,
                           shuffle=False,
                           num_workers=4,
                           pin_memory=True)

    # 修改评估调用方式
    eval_results = evaluate_model(model, eval_loader, device)
    
    # 保存评估结果
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    with open(f'{results_dir}/baseline_evaluation_results.txt', 'w') as f:
        f.write("基线模型评估结果:\n")
        
        for level_name, metrics in eval_results['results'].items():
            if level_name == 'level1':
                f.write("\n一级标签评估:\n")
            elif level_name == 'level2':
                f.write("\n二级标签评估:\n")
            elif level_name == 'level3':
                f.write("\n三级标签评估:\n")
                
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
    
    print(f"评估结果已保存到 {results_dir}/baseline_evaluation_results.txt")
    
    # 获取预测结果
    all_predictions_l1 = eval_results['predictions']['l1']
    all_predictions_l2 = eval_results['predictions']['l2']
    all_predictions_l3 = eval_results['predictions']['l3']

    # 层级一致性检查
    hierarchy_violations = 0
    for i in range(len(all_predictions_l1)):
        l1_active = np.where(all_predictions_l1[i] > 0.5)[0]
        l2_active = np.where(all_predictions_l2[i] > 0.5)[0]
        l3_active = np.where(all_predictions_l3[i] > 0.5)[0]
        
        # 基线模型没有层级约束，所以这里只是检查一下
        if len(l1_active) > 0 and len(l2_active) > 0:
            if not any(l1_idx in l1_active for l1_idx in range(len(l1_active))):
                hierarchy_violations += 1
        
        if len(l2_active) > 0 and len(l3_active) > 0:
            if not any(l2_idx in l2_active for l2_idx in range(len(l2_active))):
                hierarchy_violations += 1

    print(f"\n层级一致性检查结果:")
    print(f"总样本数: {len(all_predictions_l1)}")
    print(f"层级约束违反次数: {hierarchy_violations}")
    print(f"层级一致性: {1 - hierarchy_violations/(2*len(all_predictions_l1)):.2%}")

if __name__ == "__main__":
    main()