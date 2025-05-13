import torch
from torch.utils.data import Dataset, DataLoader
# from transformers import LongformerTokenizer
from transformers import BertTokenizer # 导入 BertTokenizer
from models.hierarchical_classifier import HierarchicalClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer # 新增导入

class TextDataset(Dataset):
    def __init__(self, texts, labels_l1, labels_l2, labels_l3, tokenizer, max_length=512):
        print(f"开始初始化评估数据集，共有{len(texts)}条数据...")
        self.texts = texts
        self.labels_l1 = labels_l1
        self.labels_l2 = labels_l2
        self.labels_l3 = labels_l3
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print("开始预处理文本编码...")
        total = len(texts)
        self.encodings = []
        for i, text in enumerate(texts, 1):
            if i % 1000 == 0:
                print(f"已处理 {i}/{total} 条文本...")
            encoding = self.tokenizer(str(text), 
                                    max_length=self.max_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='pt')
            self.encodings.append(encoding)
        print("文本编码预处理完成！")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.encodings[idx]
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels_l1': torch.FloatTensor(self.labels_l1[idx]),
            'labels_l2': torch.FloatTensor(self.labels_l2[idx]),
            'labels_l3': torch.FloatTensor(self.labels_l3[idx])
        }

def evaluate_model(model, eval_loader, device):
    print("开始评估模型...")
    model.eval()
    
    all_predictions_l1 = []
    all_predictions_l2 = []
    all_predictions_l3 = []
    all_labels_l1 = []
    all_labels_l2 = []
    all_labels_l3 = []
    
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
            
            predictions_l1 = torch.sigmoid(outputs['l1_logits']).cpu()
            predictions_l2 = torch.sigmoid(outputs['l2_logits']).cpu()
            predictions_l3 = torch.sigmoid(outputs['l3_logits']).cpu()
            
            predictions_l1 = (predictions_l1 > 0.5).float()
            predictions_l2 = (predictions_l2 > 0.5).float()
            predictions_l3 = (predictions_l3 > 0.5).float()
            
            all_predictions_l1.extend(predictions_l1.numpy())
            all_predictions_l2.extend(predictions_l2.numpy())
            all_predictions_l3.extend(predictions_l3.numpy())
            all_labels_l1.extend(labels_l1.numpy())
            all_labels_l2.extend(labels_l2.numpy())
            all_labels_l3.extend(labels_l3.numpy())
    
    all_predictions_l1 = np.array(all_predictions_l1)
    all_predictions_l2 = np.array(all_predictions_l2)
    all_predictions_l3 = np.array(all_predictions_l3)
    all_labels_l1 = np.array(all_labels_l1)
    all_labels_l2 = np.array(all_labels_l2)
    all_labels_l3 = np.array(all_labels_l3)
    
    def calculate_metrics(y_true, y_pred, level_name):
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n{level_name} 评估结果:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def post_process(predictions):
        # 获取各层级预测结果
        l1_preds = (predictions['l1_probs'] > 0.5).float()
        l2_preds = (predictions['l2_probs'] > 0.5).float()
        l3_preds = (predictions['l3_probs'] > 0.5).float()
        
        # 应用层级约束
        l2_preds = l2_preds * l1_preds.unsqueeze(-1)  # 子类必须属于激活的父类
        l3_preds = l3_preds * l2_preds.unsqueeze(-1)  # 孙类必须属于激活的子类
        
        # 新增层级白名单过滤
        for i in range(l2_preds.size(0)):
            active_l1 = l1_preds[i].nonzero().squeeze()
            allowed_l2 = self.allowed_l2_indices[active_l1]  # 从数据生成的允许列表
            l2_preds[i] *= allowed_l2
            
            active_l2 = l2_preds[i].nonzero().squeeze()
            allowed_l3 = self.allowed_l3_indices[active_l2]
            l3_preds[i] *= allowed_l3
        
        return {
            'l1_preds': l1_preds,
            'l2_preds': l2_preds,
            'l3_preds': l3_preds
        }
    
    results = {
        'level1': calculate_metrics(all_labels_l1, all_predictions_l1, "一级分类"),
        'level2': calculate_metrics(all_labels_l2, all_predictions_l2, "二级分类"),
        'level3': calculate_metrics(all_labels_l3, all_predictions_l3, "三级分类")
    }
    
    # 在return前添加预测结果的返回
    return {
        'results': results,
        'predictions': {
            'l1': all_predictions_l1,
            'l2': all_predictions_l2,
            'l3': all_predictions_l3
        }
    }

def extract_constraint_tree(model, mlb_l1, mlb_l2, mlb_l3, filename="constraint_tree.txt"):
    """从模型参数中提取层级约束关系"""
    # 获取约束矩阵权重
    parent_weights = model.parent_constraint.weight.detach().cpu().numpy()
    grandparent_weights = model.grandparent_constraint.weight.detach().cpu().numpy()
    
    # 构建树结构
    tree = {}
    
    # 一级到二级的约束关系
    for l2_idx in range(parent_weights.shape[0]):
        l1_parent = np.argmax(parent_weights[l2_idx])
        l1_name = mlb_l1.classes_[l1_parent]
        l2_name = mlb_l2.classes_[l2_idx]
        
        if l1_name not in tree:
            tree[l1_name] = {}
        tree[l1_name][l2_name] = []
    
    # 二级到三级的约束关系
    for l3_idx in range(grandparent_weights.shape[0]):
        l1_grandparent = np.argmax(grandparent_weights[l3_idx])
        # 修正索引：使用L1父类索引而非L3索引
        l2_parents = np.where(parent_weights[:, l1_grandparent] > 0.5)[0]  # 阈值可调整
        
        for l2_idx in l2_parents:
            l1_name = mlb_l1.classes_[l1_grandparent]
            l2_name = mlb_l2.classes_[l2_idx]
            l3_name = mlb_l3.classes_[l3_idx]
            
            if l1_name in tree and l2_name in tree[l1_name]:
                tree[l1_name][l2_name].append(l3_name)

    # 写入文件
    with open(filename, 'w', encoding='utf-8') as f:
        for l1, children in tree.items():
            f.write(f"┌── {l1}\n")
            for l2, l3_list in children.items():
                f.write(f"│   ├── {l2}\n")
                for l3 in l3_list:
                    f.write(f"│   │   └── {l3}\n")
    print(f"约束树已保存至 {filename}")

# 在main函数中添加调用
def main():
    print("开始加载tokenizer...")
    # tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/textClassification/models/chinese-roberta-wwm-ext') # 加载本地 BertTokenizer

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
    # --- 修改结束 ---

    print("初始化模型...")
    # 注意：HierarchicalClassifier 的 __init__ 默认模型路径已修改为本地路径
    # 使用从 MultiLabelBinarizer 获取的类别数量
    model = HierarchicalClassifier(
        num_labels_l1=num_classes_l1,
        num_labels_l2=num_classes_l2,
        num_labels_l3=num_classes_l3
    )

    print("加载模型权重...")
    # 确保加载的是最佳模型
    model_path = 'models/best_hierarchical_classifier.pth'
    print(f"从 {model_path} 加载模型...")
    model.load_state_dict(torch.load(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)

    print("创建评估数据加载器...")
    # 确认 max_length 是否适合 RoBERTa (通常是 512)
    # 将处理后的标签传递给 Dataset
    eval_dataset = TextDataset(texts, labels_l1, labels_l2, labels_l3, tokenizer, max_length=512)
    eval_loader = DataLoader(eval_dataset,
                           batch_size=16, # 可以根据评估时的显存调整
                           shuffle=False, # 评估时不需要打乱
                           num_workers=4,
                           pin_memory=True)

    # 修改评估调用方式
    eval_results = evaluate_model(model, eval_loader, device)
    
    extract_constraint_tree(model, mlb_l1, mlb_l2, mlb_l3)
    
    # 获取预测结果
    all_predictions_l1 = eval_results['predictions']['l1']
    all_predictions_l2 = eval_results['predictions']['l2']
    all_predictions_l3 = eval_results['predictions']['l3']

    # 层级一致性检查（移动到这里）
    hierarchy_violations = 0
    for i in range(len(all_predictions_l1)):
        l1_active = np.where(all_predictions_l1[i] > 0.5)[0]
        l2_active = np.where(all_predictions_l2[i] > 0.5)[0]
        l3_active = np.where(all_predictions_l3[i] > 0.5)[0]
        
        # 检查子类是否属于激活的父类
        if not set(l2_active).issubset(l1_active):
            hierarchy_violations += 1
        
        # 检查孙类是否属于激活的子类 
        if not set(l3_active).issubset(l2_active):
            hierarchy_violations += 1

    print(f"\n层级一致性检查结果:")
    print(f"总样本数: {len(all_predictions_l1)}")
    print(f"层级约束违反次数: {hierarchy_violations}")
    print(f"层级一致性: {1 - hierarchy_violations/(2*len(all_predictions_l1)):.2%}")


if __name__ == '__main__':
    main()