import torch
import torch.nn as nn
# from transformers import LongformerModel, LongformerConfig
from transformers import BertModel, BertConfig # 导入 BertModel 和 BertConfig
import numpy as np

class HierarchicalClassifier(nn.Module):
    # def __init__(self, model_name='allenai/longformer-base-4096',
    def __init__(self, model_name='/root/autodl-tmp/textClassification/models/chinese-roberta-wwm-ext',
                 num_labels_l1=12, num_labels_l2=75, num_labels_l3=341,
                 gradient_checkpointing=True,
                 constraint_dir='data'):
        super().__init__()
        
        # 加载约束矩阵
        parent_constraint = np.load(f"{constraint_dir}/parent_constraint.npy")
        grandparent_constraint = np.load(f"{constraint_dir}/grandparent_constraint.npy")
        
        # 初始化约束层并加载预训练权重
        self.parent_constraint = nn.Linear(num_labels_l1, num_labels_l2)
        self.parent_constraint.weight.data = torch.from_numpy(parent_constraint).float()
        self.parent_constraint.bias.data.zero_()
        
        self.grandparent_constraint = nn.Linear(num_labels_l1, num_labels_l3)
        self.grandparent_constraint.weight.data = torch.from_numpy(grandparent_constraint).float()
        self.grandparent_constraint.bias.data.zero_()
    
        # 冻结约束层参数
        for param in self.parent_constraint.parameters():
            param.requires_grad = False
        for param in self.grandparent_constraint.parameters():
            param.requires_grad = False

        # # Longformer encoder
        # self.config = LongformerConfig.from_pretrained(model_name)
        # self.encoder = LongformerModel.from_pretrained(model_name)
        # 使用 Bert encoder
        self.config = BertConfig.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        self.hidden_size = self.config.hidden_size

        # Level 1 classifier (13 classes)
        self.classifier_l1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_labels_l1)
        )

        # Level 2 classifier (77 classes) - Input size adjusted
        # 添加层级约束矩阵
        #self.parent_constraint = nn.Linear(num_labels_l1, num_labels_l2)
        #self.grandparent_constraint = nn.Linear(num_labels_l1, num_labels_l3)
        
        # 确保sigmoid层存在
        self.sigmoid = nn.Sigmoid()

        # 调整分类器输入维度
        self.classifier_l2 = nn.Sequential(
            nn.Linear(self.hidden_size + num_labels_l1, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_labels_l2)
        )
        self.classifier_l3 = nn.Sequential(
            nn.Linear(self.hidden_size + num_labels_l2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_labels_l3)
        )

    def forward(self, input_ids, attention_mask):
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Pool the output (use [CLS] token representation)
        pooled_output = sequence_output[:, 0]

        # Level 1 prediction
        l1_logits = self.classifier_l1(pooled_output)
        l1_probs = self.sigmoid(l1_logits) # Still calculate probs for potential use/output

        # Level 2 prediction (concatenate pooled output with l1 logits)
        l2_input = torch.cat([pooled_output, l1_logits], dim=1) # Use l1_logits instead of l1_probs
        l2_logits = self.classifier_l2(l2_input) + 0.8 * torch.sigmoid(self.parent_constraint(l1_probs))
        #l2_logits = l2_logits * self.parent_mask[l1_preds.argmax(dim=1)]  # parent_mask从数据生成的约束矩阵
        l2_probs = self.sigmoid(l2_logits)

        # Level 3 prediction (concatenate pooled output with l2 logits)
        l3_input = torch.cat([pooled_output, l2_logits], dim=1)
        l3_logits = self.classifier_l3(l3_input) + 1.2 * torch.sigmoid(self.grandparent_constraint(l1_probs))
        #l3_logits = l3_logits * self.grandparent_mask[l1_preds.argmax(dim=1)]
        l3_probs = self.sigmoid(l3_logits)

        return {
            'l1_logits': l1_logits,
            'l2_logits': l2_logits,
            'l3_logits': l3_logits,
            'l1_probs': l1_probs,
            'l2_probs': l2_probs,
            'l3_probs': l3_probs
        }