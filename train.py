import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizer
from models.hierarchical_classifier import HierarchicalClassifier
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler

# 在 TextDataset 类中添加预处理缓存
class TextDataset(Dataset):
    def __init__(self, texts, labels_l1, labels_l2, labels_l3, tokenizer, max_length=4096):
        print(f"开始初始化数据集，共有{len(texts)}条数据...")
        self.texts = texts
        self.labels_l1 = labels_l1
        self.labels_l2 = labels_l2
        self.labels_l3 = labels_l3
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 预处理并缓存所有文本编码
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

def train_model(model, train_loader, optimizer, device, num_epochs=5):
    print(f"开始训练，总共{num_epochs}个epoch...")
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        total_loss = 0
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader, 1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_l1 = batch['labels_l1'].to(device)
            labels_l2 = batch['labels_l2'].to(device)
            labels_l3 = batch['labels_l3'].to(device)
            
            optimizer.zero_grad()
            
            # 使用混合精度训练
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                loss_l1 = criterion(outputs['l1_logits'], labels_l1)
                loss_l2 = criterion(outputs['l2_logits'], labels_l2)
                loss_l3 = criterion(outputs['l3_logits'], labels_l3)
                
                # Total loss is weighted sum of all levels
                loss = loss_l1 + loss_l2 + loss_l3
            
            # 使用梯度缩放器
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch进度: {batch_idx}/{total_batches}, "
                      f"当前batch损失: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}')

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)
    
    print("创建数据加载器...")
    dataset = TextDataset(texts, labels_l1, labels_l2, labels_l3, tokenizer, max_length=512)
    train_loader = DataLoader(dataset, 
                            batch_size=16,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)
    
    print("初始化优化器...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    print("开始训练过程...")
    train_model(model, train_loader, optimizer, device)
    
    print("保存模型...")
    torch.save(model.state_dict(), 'models/hierarchical_classifier.pth')
    print("训练完成！")

if __name__ == '__main__':
    main()