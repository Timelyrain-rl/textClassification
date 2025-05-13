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
    
    results = {
        'level1': calculate_metrics(all_labels_l1, all_predictions_l1, "一级分类"),
        'level2': calculate_metrics(all_labels_l2, all_predictions_l2, "二级分类"),
        'level3': calculate_metrics(all_labels_l3, all_predictions_l3, "三级分类")
    }
    
    return results

def main():
    print("开始加载tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/textClassification/models/chinese-roberta-wwm-ext')

    print("加载数据...")
    df = pd.read_csv('data/merged_data_cleaned.csv')
    texts = df['主要内容'].values

    # 使用HierarchyBuilder构建层级结构
    print("构建层级结构...")
    from utils.hierarchy_utils import HierarchyBuilder
    hierarchy_builder = HierarchyBuilder()
    hierarchy_builder.build_from_file('visualization/annot_tree_output/classification_tree.txt')
    
    # 获取掩码矩阵
    l1_to_l2_mask, l2_to_l3_mask = hierarchy_builder.get_mask_matrices()
    num_l1, num_l2, num_l3 = hierarchy_builder.get_label_counts()
    
    print(f"标签数量: L1={num_l1}, L2={num_l2}, L3={num_l3}")

    # 处理标签数据
    print("从DataFrame提取并预处理标签数据...")
    raw_labels_l1 = df['一级分类'].values
    raw_labels_l2 = df['二级分类'].values
    raw_labels_l3 = df['三级分类'].values
    
    # 获取有效标签列表
    valid_l1_labels = list(hierarchy_builder.label_to_id['l1'].keys())
    valid_l2_labels = list(hierarchy_builder.label_to_id['l2'].keys())
    valid_l3_labels = list(hierarchy_builder.label_to_id['l3'].keys())
    
    # 处理标签数据
    processed_labels_l1 = [[str(label)] for label in raw_labels_l1]
    processed_labels_l2 = [[str(label)] for label in raw_labels_l2]
    processed_labels_l3 = [[str(label)] for label in raw_labels_l3]
    
    # 使用预定义的类别进行转换
    mlb_l1 = MultiLabelBinarizer(classes=valid_l1_labels)
    mlb_l2 = MultiLabelBinarizer(classes=valid_l2_labels)
    mlb_l3 = MultiLabelBinarizer(classes=valid_l3_labels)
    
    labels_l1 = mlb_l1.fit_transform(processed_labels_l1)
    labels_l2 = mlb_l2.fit_transform(processed_labels_l2)
    labels_l3 = mlb_l3.fit_transform(processed_labels_l3)
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

    # 创建标签索引映射
    label_indices = {
        'l1': {label: idx for idx, label in enumerate(mlb_l1.classes_)},
        'l2': {label: idx for idx, label in enumerate(mlb_l2.classes_)},
        'l3': {label: idx for idx, label in enumerate(mlb_l3.classes_)}
    }

    # 创建层级掩码矩阵
    print("创建层级掩码矩阵...")
    l1_to_l2_mask = torch.zeros(num_classes_l1, num_classes_l2)
    l2_to_l3_mask = torch.zeros(num_classes_l2, num_classes_l3)

    # 填充掩码矩阵
    for l1, l2_list in hierarchy_builder.l1_to_l2_map.items():
        if l1 in label_indices['l1']:
            l1_idx = label_indices['l1'][l1]
            for l2 in l2_list:
                if l2 in label_indices['l2']:
                    l2_idx = label_indices['l2'][l2]
                    l1_to_l2_mask[l1_idx, l2_idx] = 1

    for l2, l3_list in hierarchy_builder.l2_to_l3_map.items():
        if l2 in label_indices['l2']:
            l2_idx = label_indices['l2'][l2]
            for l3 in l3_list:
                if l3 in label_indices['l3']:
                    l3_idx = label_indices['l3'][l3]
                    l2_to_l3_mask[l2_idx, l3_idx] = 1

    print(f"标签维度: L1={labels_l1.shape}, L2={labels_l2.shape}, L3={labels_l3.shape}")
    print(f"类别数量: L1={num_classes_l1}, L2={num_classes_l2}, L3={num_classes_l3}")
    
    print("初始化模型...")
    model = HierarchicalClassifier(
        num_labels_l1=num_classes_l1,  
        num_labels_l2=num_classes_l2,  
        num_labels_l3=num_classes_l3,  
        l1_to_l2_mask=l1_to_l2_mask,
        l2_to_l3_mask=l2_to_l3_mask
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

    evaluate_model(model, eval_loader, device)

if __name__ == '__main__':
    main()