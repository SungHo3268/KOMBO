import os
import sys
import torch.nn as nn
from transformers import BertModel

sys.path.append(os.getcwd())
from pretraining.srcs.models import CustomBertModel


class BEEPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if hasattr(self.config, "embedding_type"):
            self.bert = CustomBertModel(config)
        else:
            self.bert = BertModel(config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        if config.binary:
            self.classifier = nn.Linear(config.hidden_size, 2)
        else:
            self.classifier = nn.Linear(config.hidden_size, 3)

    def forward(self, inputs):
        outputs = self.bert.forward(**inputs,
                                    output_hidden_states=True)  # input_token_ids, attention_mask, token_type_ids
        pooled_output = outputs['pooler_output']

        outputs_drop = self.dropout(pooled_output)
        logits = self.classifier(outputs_drop)

        return outputs, logits


class KMHaSModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if hasattr(self.config, "embedding_type"):
            self.bert = CustomBertModel(config)
        else:
            self.bert = BertModel(config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 9)

    def forward(self, inputs):
        outputs = self.bert.forward(**inputs,
                                    output_hidden_states=True)  # input_token_ids, attention_mask, token_type_ids
        pooled_output = outputs['pooler_output']

        outputs_drop = self.dropout(pooled_output)
        logits = self.classifier(outputs_drop)

        return outputs, logits


class KOLDModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if hasattr(self.config, "embedding_type"):
            self.bert = CustomBertModel(config)
        elif config.model_name == 'bert-base':
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.class_num = config.__dict__[f"level_{config.label_level}_label_num"]
        self.classifier = nn.Linear(config.hidden_size, self.class_num)

    def forward(self, inputs):
        outputs = self.bert.forward(**inputs, output_hidden_states=True)  # input_ids, attention_mask, token_type_ids
        pooled_output = outputs['pooler_output']

        output_drop = self.dropout(pooled_output)
        logits = self.classifier(output_drop)
        return outputs, logits
