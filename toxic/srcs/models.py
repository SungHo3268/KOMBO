import os
import sys
import torch.nn as nn
from transformers import BertModel

sys.path.append(os.getcwd())
from pretraining.srcs.models import CustomBertModel


class KOLDModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if hasattr(self.config, "embedding_type"):
            self.bert = CustomBertModel(config)
        elif config.model_name == 'bert-base':
            self.bert = BertModel(config)
        elif config.model_name == 'klue-bert-base':
            print("Using klue-bert-base. Loading the model...")
            self.bert = BertModel.from_pretrained("klue/bert-base")
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.class_num = config.__dict__[f"level_{config.label_level}_label_num"]
        self.classifier = nn.Linear(config.hidden_size, self.class_num)

    def forward(self, inputs):
        outputs = self.bert.forward(**inputs, output_hidden_states=True)  # input_ids, attention_mask, token_type_ids
        pooled_output = outputs['pooler_output']

        output_drop = self.dropout(pooled_output)
        logits = self.classifier(output_drop)
        return outputs, logits

