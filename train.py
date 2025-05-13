import torch
from torch.utils.data import Dataset, DataLoader
# from transformers import LongformerTokenizer
from transformers import BertTokenizer
from models.hierarchical_classifier import HierarchicalClassifier
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import MultiLabelBinarizer 

# 在文件顶部添加导入语句
import torch.nn.functional as F  # 添加这行代码

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
        # 确保返回的标签是正确的类型和形状
        encoding = self.encodings[idx]
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        # 确保标签是 FloatTensor
        if self.labels_l1 is not None:
            item['labels_l1'] = torch.FloatTensor(self.labels_l1[idx])
        if self.labels_l2 is not None:
            item['labels_l2'] = torch.FloatTensor(self.labels_l2[idx])
        if self.labels_l3 is not None:
            item['labels_l3'] = torch.FloatTensor(self.labels_l3[idx])
        return item

# 新增评估函数
def evaluate_model(model, val_loader, criterion, device):
    model.eval() # 设置模型为评估模式
    total_val_loss = 0
    with torch.no_grad(): # 在评估阶段不计算梯度
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_l1 = batch['labels_l1'].to(device)
            labels_l2 = batch['labels_l2'].to(device)
            labels_l3 = batch['labels_l3'].to(device)

            with autocast(): # 同样可以使用混合精度进行评估
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss_l1 = criterion(outputs['l1_logits'], labels_l1)
                loss_l2 = criterion(outputs['l2_logits'], labels_l2)
                loss_l3 = criterion(outputs['l3_logits'], labels_l3)
                loss = loss_l1 + loss_l2 + loss_l3

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    model.train() # 将模型设置回训练模式
    return avg_val_loss

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=5, patience=3, min_delta=0.001, model_save_path='models/best_hierarchical_classifier.pth'):
    print(f"开始训练，总共最多{num_epochs}个epoch...")
    print(f"早停设置: patience={patience}, min_delta={min_delta}")
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    # 确保保存模型的目录存在
    model_save_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        print(f"创建目录: {model_save_dir}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train() # 确保模型在训练模式
        total_train_loss = 0
        total_batches = len(train_loader)

        # 保持现有训练循环不变，现在F已经被正确导入
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

                # 原始损失计算
                loss_l1 = F.binary_cross_entropy_with_logits(outputs['l1_logits'], labels_l1)
                loss_l2 = F.binary_cross_entropy_with_logits(outputs['l2_logits'], labels_l2)
                loss_l3 = F.binary_cross_entropy_with_logits(outputs['l3_logits'], labels_l3)
                
                # 层级约束损失
                hierarchy_loss = F.mse_loss(
                    torch.sigmoid(outputs['l2_logits']),
                    model.parent_constraint(torch.sigmoid(outputs['l1_logits']))
                )
                hierarchy_loss += F.mse_loss(
                    torch.sigmoid(outputs['l3_logits']),
                    model.grandparent_constraint(torch.sigmoid(outputs['l1_logits']))
                )
                
                total_loss = loss_l1 + loss_l2 + loss_l3 + 0.5 * hierarchy_loss.mean()
                loss = total_loss  # 替换原有的简单相加损失

            # 使用梯度缩放器
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch进度: {batch_idx}/{total_batches}, "
                      f"当前batch训练损失: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch+1} 训练完成, 平均训练损失: {avg_train_loss:.4f}')

        # --- 验证和早停逻辑 ---
        avg_val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}, 平均验证损失: {avg_val_loss:.4f}')

        # 检查是否有改进
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            # 保存最佳模型
            torch.save(model.state_dict(), model_save_path)
            print(f'验证损失改善，保存最佳模型到 {model_save_path}')
        else:
            epochs_no_improve += 1
            print(f'验证损失没有显著改善 ({epochs_no_improve}/{patience})')

        # 检查是否需要早停
        if epochs_no_improve >= patience:
            print(f'连续 {patience} 个 epochs 验证损失没有改善，提前停止训练')
            print(f'最佳模型保存在 Epoch {best_epoch}，验证损失为: {best_val_loss:.4f}')
            break # 退出训练循环
    
    if epoch == num_epochs - 1 and epochs_no_improve < patience:
         print(f'训练完成所有 {num_epochs} 个 epochs')
         print(f'最佳模型保存在 Epoch {best_epoch}，验证损失为: {best_val_loss:.4f}')


def main():
    print("开始加载tokenizer...")
    # tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/textClassification/models/chinese-roberta-wwm-ext') # 加载本地 BertTokenizer

    print("加载数据...")
    df = pd.read_csv('data/merged_data_cleaned.csv')
    texts = df['主要内容'].values

    print("从DataFrame提取并预处理标签数据...")
    # 获取标签列
    raw_labels_l1 = df['一级分类'].values
    raw_labels_l2 = df['二级分类'].values
    raw_labels_l3 = df['三级分类'].values

    # 预处理标签 - 实现标签编码
    print("处理标签数据...")
    
    # 处理一级标签
    mlb_l1 = MultiLabelBinarizer()
    # 假设标签是字符串，需要转换为列表
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

    # --- 数据划分 ---
    print("划分训练集和验证集...")
    indices = np.arange(len(texts))
    # 使用已经处理好的 NumPy 标签数组进行划分
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

    print("初始化模型...")
    # 使用从预处理中得到的类别数量
    # HierarchicalClassifier 的 __init__ 默认模型路径已修改为本地路径
    model = HierarchicalClassifier(
        num_labels_l1=num_classes_l1,
        num_labels_l2=num_classes_l2,
        num_labels_l3=num_classes_l3
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)

    print("创建数据集和数据加载器...")
    # 将处理好的标签数组传递给 Dataset
    # 当前 max_length=512
    train_dataset = TextDataset(train_texts, train_labels_l1, train_labels_l2, train_labels_l3, tokenizer, max_length=512)
    val_dataset = TextDataset(val_texts, val_labels_l1, val_labels_l2, val_labels_l3, tokenizer, max_length=512)

    train_loader = DataLoader(train_dataset,
                            batch_size=16,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)

    val_loader = DataLoader(val_dataset,
                          batch_size=16,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True)

    print("初始化优化器...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    print("开始训练过程...")
    # 调用修改后的 train_model，传入验证加载器和早停参数
    train_model(model, train_loader, val_loader, optimizer, device, num_epochs=10, patience=2, min_delta=0.005, model_save_path='models/best_hierarchical_classifier.pth')
    print("训练完成！模型已保存在 'models/hierarchical_classifier.pth'")

if __name__ == '__main__':
    main()