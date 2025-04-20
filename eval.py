import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizer
from models.hierarchical_classifier import HierarchicalClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    
    print("加载数据...")
    df = pd.read_csv('data/merged_data_cleaned.csv')
    texts = df['主要内容'].values
    
    print("加载标签数据...")
    labels_l1 = np.load('data/cooccurrence_l1_l2.npy')
    labels_l2 = np.load('data/cooccurrence_l1_l3.npy')
    labels_l3 = np.load('data/cooccurrence_l1_l3.npy')
    
    print("初始化模型...")
    model = HierarchicalClassifier(
        num_labels_l1=labels_l1.shape[1],
        num_labels_l2=labels_l2.shape[1],
        num_labels_l3=labels_l3.shape[1]
    )
    
    print("加载模型权重...")
    model.load_state_dict(torch.load('models/hierarchical_classifier.pth'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)
    
    print("创建评估数据加载器...")
    eval_dataset = TextDataset(texts, labels_l1, labels_l2, labels_l3, tokenizer)
    eval_loader = DataLoader(eval_dataset, 
                           batch_size=16,
                           shuffle=False,
                           num_workers=4,
                           pin_memory=True)
    
    evaluate_model(model, eval_loader, device)

if __name__ == '__main__':
    main()