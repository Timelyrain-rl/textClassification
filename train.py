import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizer
from models.hierarchical_classifier import HierarchicalClassifier
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler

class TextDataset(Dataset):
    def __init__(self, texts, labels_l1, labels_l2, labels_l3, tokenizer, max_length=4096):
        self.texts = texts
        self.labels_l1 = labels_l1
        self.labels_l2 = labels_l2
        self.labels_l3 = labels_l3
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(text, 
                                 max_length=self.max_length,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels_l1': torch.FloatTensor(self.labels_l1[idx]),
            'labels_l2': torch.FloatTensor(self.labels_l2[idx]),
            'labels_l3': torch.FloatTensor(self.labels_l3[idx])
        }

def train_model(model, train_loader, optimizer, device, num_epochs=5):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
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
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

def main():
    # Initialize tokenizer
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    # Load data
    df = pd.read_csv('data/merged_data_cleaned.csv')
    texts = df['主要内容'].values
    # Load preprocessed labels
    labels_l1 = np.load('data/cooccurrence_l1_l2.npy')
    labels_l2 = np.load('data/cooccurrence_l1_l3.npy')
    labels_l3 = np.load('data/cooccurrence_l1_l3.npy')
    # Initialize model
    model = HierarchicalClassifier(
        num_labels_l1=labels_l1.shape[1],
        num_labels_l2=labels_l2.shape[1],
        num_labels_l3=labels_l3.shape[1]
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, labels_l1, labels_l2, labels_l3, tokenizer, max_length=2048)  # 减小序列长度
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)  # 减小batch size
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Train the model
    train_model(model, train_loader, optimizer, device)
    
    # Save the model
    torch.save(model.state_dict(), 'models/hierarchical_classifier.pth')

if __name__ == '__main__':
    main()