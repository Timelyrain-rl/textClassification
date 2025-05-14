import torch
import torch.nn as nn

from transformers import BertModel, BertConfig # 导入 BertModel 和 BertConfig

class HierarchicalClassifier(nn.Module):
    def __init__(self, model_name='/root/autodl-tmp/textClassification/models/chinese-roberta-wwm-ext',
                 num_labels_l1=12, num_labels_l2=75, num_labels_l3=341,
                 gradient_checkpointing=True):
        super().__init__()

        self.config = BertConfig.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        self.hidden_size = self.config.hidden_size

        self.classifier_l1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_labels_l1)
        )

        self.parent_constraint = nn.Linear(num_labels_l1, num_labels_l2)
        self.grandparent_constraint = nn.Linear(num_labels_l1, num_labels_l3)
        
        self.sigmoid = nn.Sigmoid()

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
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        pooled_output = sequence_output[:, 0]

        l1_logits = self.classifier_l1(pooled_output)
        l1_probs = self.sigmoid(l1_logits)

        l2_input = torch.cat([pooled_output, l1_logits], dim=1)
        l2_logits = self.classifier_l2(l2_input) + torch.sigmoid(self.parent_constraint(l1_probs))
        l2_probs = self.sigmoid(l2_logits)

        l3_input = torch.cat([pooled_output, l2_logits], dim=1)
        l3_logits = self.classifier_l3(l3_input) + torch.sigmoid(self.grandparent_constraint(l1_probs))
        l3_probs = self.sigmoid(l3_logits)

        return {
            'l1_logits': l1_logits,
            'l2_logits': l2_logits,
            'l3_logits': l3_logits,
            'l1_probs': l1_probs,
            'l2_probs': l2_probs,
            'l3_probs': l3_probs
        }