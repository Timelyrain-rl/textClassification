import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerConfig

class HierarchicalClassifier(nn.Module):
    def __init__(self, model_name='allenai/longformer-base-4096', 
                 num_labels_l1=13, num_labels_l2=77, num_labels_l3=340,
                 gradient_checkpointing=True):
        super().__init__()

        # Longformer encoder
        self.config = LongformerConfig.from_pretrained(model_name)
        self.encoder = LongformerModel.from_pretrained(model_name)
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
        self.classifier_l2 = nn.Sequential(
            nn.Linear(self.hidden_size + num_labels_l1, self.hidden_size), # Input uses L1 logits
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_labels_l2)
        )

        # Level 3 classifier (340 classes) - Input size adjusted
        self.classifier_l3 = nn.Sequential(
            nn.Linear(self.hidden_size + num_labels_l2, self.hidden_size), # Input uses L2 logits
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_labels_l3)
        )

        self.sigmoid = nn.Sigmoid()

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
        l2_logits = self.classifier_l2(l2_input)
        l2_probs = self.sigmoid(l2_logits) # Still calculate probs

        # Level 3 prediction (concatenate pooled output with l2 logits)
        l3_input = torch.cat([pooled_output, l2_logits], dim=1) # Use l2_logits instead of l2_probs
        l3_logits = self.classifier_l3(l3_input)
        l3_probs = self.sigmoid(l3_logits) # Still calculate probs

        return {
            'l1_logits': l1_logits,
            'l2_logits': l2_logits,
            'l3_logits': l3_logits,
            'l1_probs': l1_probs,
            'l2_probs': l2_probs,
            'l3_probs': l3_probs
        }