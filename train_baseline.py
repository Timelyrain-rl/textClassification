import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from models.baseline_classifier import BaselineClassifier

# 数据集类
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels_l1=None, labels_l2=None, labels_l3=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels_l1 = labels_l1
        self.labels_l2 = labels_l2
        self.labels_l3 = labels_l3
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print("开始预处理文本...")
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
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        if self.labels_l1 is not None:
            item['labels_l1'] = torch.FloatTensor(self.labels_l1[idx])
        if self.labels_l2 is not None:
            item['labels_l2'] = torch.FloatTensor(self.labels_l2[idx])
        if self.labels_l3 is not None:
            item['labels_l3'] = torch.FloatTensor(self.labels_l3[idx])
        return item

# 评估函数
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad(): 
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_l1 = batch['labels_l1'].to(device)
            labels_l2 = batch['labels_l2'].to(device)
            labels_l3 = batch['labels_l3'].to(device)

            with autocast(): 
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss_l1 = criterion(outputs['l1_logits'], labels_l1)
                loss_l2 = criterion(outputs['l2_logits'], labels_l2)
                loss_l3 = criterion(outputs['l3_logits'], labels_l3)
                loss = loss_l1 + loss_l2 + loss_l3

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    model.train()
    return avg_val_loss

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=5, patience=3, min_delta=0.001, model_save_path='models/best_baseline_classifier.pth'):
    print(f"开始训练基线模型，总共最多{num_epochs}个epoch...")
    print(f"早停设置: patience={patience}, min_delta={min_delta}")
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()  # 多标签分类用BCEWithLogitsLoss
    # 如果使用softmax，则使用CrossEntropyLoss
    # criterion = torch.nn.CrossEntropyLoss()
    
    scaler = GradScaler()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    model_save_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        print(f"创建目录: {model_save_dir}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        total_train_loss = 0
        total_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader, 1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_l1 = batch['labels_l1'].to(device)
            labels_l2 = batch['labels_l2'].to(device)
            labels_l3 = batch['labels_l3'].to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                loss_l1 = criterion(outputs['l1_logits'], labels_l1)
                loss_l2 = criterion(outputs['l2_logits'], labels_l2)
                loss_l3 = criterion(outputs['l3_logits'], labels_l3)
                
                # 基线模型不使用层级约束
                total_loss = loss_l1 + loss_l2 + loss_l3
                loss = total_loss 

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch进度: {batch_idx}/{total_batches}, "
                      f"当前batch训练损失: {loss.item():.4f}")

        avg_train_loss = total_train_loss / total_batches
        print(f"Epoch {epoch+1} 平均训练损失: {avg_train_loss:.4f}")

        # 验证
        avg_val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} 平均验证损失: {avg_val_loss:.4f}")

        # 早停检查
        if avg_val_loss < best_val_loss - min_delta:
            print(f"验证损失从 {best_val_loss:.4f} 改善到 {avg_val_loss:.4f}，保存模型...")
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1
            print(f"验证损失没有显著改善，已经 {epochs_no_improve} 个epoch没有改善")
            if epochs_no_improve >= patience:
                print(f"早停触发，停止训练。最佳epoch: {best_epoch}，最佳验证损失: {best_val_loss:.4f}")
                break

    print(f"训练完成！最佳模型保存在 {model_save_path}")
    print(f"最佳epoch: {best_epoch}，最佳验证损失: {best_val_loss:.4f}")
    return model

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    print("加载数据...")
    df = pd.read_csv('data/merged_data_cleaned.csv')
    texts = df['主要内容'].values

    # 准备标签
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

    # 划分训练集和验证集
    print("划分训练集和验证集...")
    indices = np.arange(len(texts))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    train_texts = texts[train_indices]
    val_texts = texts[val_indices]
    train_labels_l1 = labels_l1[train_indices]
    val_labels_l1 = labels_l1[val_indices]
    train_labels_l2 = labels_l2[train_indices]
    val_labels_l2 = labels_l2[val_indices]
    train_labels_l3 = labels_l3[train_indices]
    val_labels_l3 = labels_l3[val_indices]
    print(f"训练集大小: {len(train_texts)}, 验证集大小: {len(val_texts)}")
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/textClassification/models/chinese-roberta-wwm-ext')
    
    # 创建数据集
    print("创建训练集...")
    train_dataset = TextClassificationDataset(
        train_texts, train_labels_l1, train_labels_l2, train_labels_l3, tokenizer
    )
    
    print("创建验证集...")
    val_dataset = TextClassificationDataset(
        val_texts, val_labels_l1, val_labels_l2, val_labels_l3, tokenizer
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4, pin_memory=True)
    
    # 创建模型
    print("创建基线模型...")
    model = BaselineClassifier(
        num_labels_l1=num_classes_l1,
        num_labels_l2=num_classes_l2,
        num_labels_l3=num_classes_l3
    )
    model.to(device)
    
    # 创建优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # 训练模型
    trained_model = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=10, patience=3, min_delta=0.001,
        model_save_path='models/best_baseline_classifier.pth'
    )
    
    print("基线模型训练完成！")

if __name__ == "__main__":
    main()