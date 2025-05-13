import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig

class HierarchicalClassifier(nn.Module):
    def __init__(self, num_labels_l1, num_labels_l2, num_labels_l3, l1_to_l2_mask=None, l2_to_l3_mask=None):
        super().__init__()
        # 使用本地模型路径
        model_path = '/root/autodl-tmp/textClassification/models/chinese-roberta-wwm-ext'
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.hidden_size = self.encoder.config.hidden_size
        
        # 一级分类器
        self.classifier_l1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_labels_l1)
        )
        
        # 二级分类器 - 输入包含一级分类的logits
        self.classifier_l2 = nn.Sequential(
            nn.Linear(self.hidden_size + num_labels_l1, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_labels_l2)
        )
        
        # 三级分类器 - 输入包含二级分类的logits
        self.classifier_l3 = nn.Sequential(
            nn.Linear(self.hidden_size + num_labels_l2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_labels_l3)
        )
        
        # 注册掩码作为缓冲区
        if l1_to_l2_mask is None:
            l1_to_l2_mask = torch.ones(num_labels_l1, num_labels_l2)
        if l2_to_l3_mask is None:
            l2_to_l3_mask = torch.ones(num_labels_l2, num_labels_l3)
            
        self.register_buffer('l1_to_l2_mask', l1_to_l2_mask)
        self.register_buffer('l2_to_l3_mask', l2_to_l3_mask)
        
        self.sigmoid = nn.Sigmoid()
        self.threshold = 0.5

    def forward(self, input_ids, attention_mask):
        # 获取编码器输出
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token输出
        
        # 一级分类预测
        l1_logits = self.classifier_l1(pooled_output)
        l1_probs = self.sigmoid(l1_logits)
        
        # 基于一级分类预测创建二级掩码
        l1_pred = (l1_probs > self.threshold).float()
        l2_mask = torch.matmul(l1_pred, self.l1_to_l2_mask)
        l2_mask = (l2_mask > 0).float()
        
        # 二级分类预测（连接pooled_output和l1_logits）
        l2_input = torch.cat([pooled_output, l1_logits], dim=1)
        l2_logits = self.classifier_l2(l2_input)
        l2_logits = l2_logits * l2_mask  # 应用掩码
        l2_probs = self.sigmoid(l2_logits)
        
        # 基于二级分类预测创建三级掩码
        l2_pred = (l2_probs > self.threshold).float()
        l3_mask = torch.matmul(l2_pred, self.l2_to_l3_mask)
        l3_mask = (l3_mask > 0).float()
        
        # 三级分类预测（连接pooled_output和l2_logits）
        l3_input = torch.cat([pooled_output, l2_logits], dim=1)
        l3_logits = self.classifier_l3(l3_input)
        l3_logits = l3_logits * l3_mask  # 应用掩码
        l3_probs = self.sigmoid(l3_logits)
        
        return {
            'l1_logits': l1_logits,
            'l2_logits': l2_logits,
            'l3_logits': l3_logits,
            'l1_probs': l1_probs,
            'l2_probs': l2_probs,
            'l3_probs': l3_probs
        }