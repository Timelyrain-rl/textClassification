import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig

class BaselineClassifier(nn.Module):
    def __init__(self, model_name='/root/autodl-tmp/textClassification/models/chinese-roberta-wwm-ext',
                 num_labels_l1=12, num_labels_l2=75, num_labels_l3=341,
                 gradient_checkpointing=True):
        super().__init__()

        self.config = BertConfig.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        self.hidden_size = self.config.hidden_size

        # 简化的分类器，直接从BERT输出到各个层级的标签
        self.classifier_l1 = nn.Linear(self.hidden_size, num_labels_l1)
        self.classifier_l2 = nn.Linear(self.hidden_size, num_labels_l2)
        self.classifier_l3 = nn.Linear(self.hidden_size, num_labels_l3)
        
        # 根据您的需求选择sigmoid或softmax
        self.sigmoid = nn.Sigmoid()  # 多标签分类用sigmoid
        # 或者使用softmax进行多类别分类
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # 使用[CLS]标记的输出

        # 简单的前向传播，没有层级约束
        l1_logits = self.classifier_l1(pooled_output)
        l2_logits = self.classifier_l2(pooled_output)
        l3_logits = self.classifier_l3(pooled_output)
        
        # 使用sigmoid进行多标签分类
        l1_probs = self.sigmoid(l1_logits)
        l2_probs = self.sigmoid(l2_logits)
        l3_probs = self.sigmoid(l3_logits)
        
        # 如果使用softmax进行多类别分类，请使用以下代码
        # l1_probs = F.softmax(l1_logits, dim=1)
        # l2_probs = F.softmax(l2_logits, dim=1)
        # l3_probs = F.softmax(l3_logits, dim=1)

        return {
            'l1_logits': l1_logits,
            'l2_logits': l2_logits,
            'l3_logits': l3_logits,
            'l1_probs': l1_probs,
            'l2_probs': l2_probs,
            'l3_probs': l3_probs
        }