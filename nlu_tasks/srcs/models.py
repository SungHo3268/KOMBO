import os
import sys
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

sys.path.append(os.getcwd())
from pretraining.srcs.models import CustomBertModel, CustomBertForPreTraining

import warnings
warnings.filterwarnings("ignore")


class KorNLIModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if hasattr(self.config, "embedding_type"):
            self.bert = CustomBertModel(config)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 3)  # entailment, neutral, contradiction

    def forward(self, inputs):
        outputs = self.bert.forward(**inputs,
                                    output_hidden_states=True)  # input_ids, attention_mask, token_type_ids
        pooled_output = outputs['pooler_output']

        output_drop = self.dropout(pooled_output)
        logits = self.classifier(output_drop)
        return outputs, logits


class KorQuADModel(BertPreTrainedModel):
    """
    BertPreTrainedModel
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    """

    def __init__(self, config):
        super(KorQuADModel, self).__init__(config)
        self.num_labels = config.num_labels  # 2 is a default value. (start_position and end_position)

        if hasattr(self.config, "embedding_type"):
            # self.bert = CustomBertModel(config)
            self.bert = CustomBertForPreTraining(config)
        else:
            self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
    ):
        r"""
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`).
                Position outside of the sequence are not taken into account for computing the loss.
            end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`).
                Position outside of the sequence are not taken into account for computing the loss.
        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1, )`, `optional`, returned when :obj:`labels` is provided):
                Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
            start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
                Span-start scores (before SoftMax).
            end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
                Span-end scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attention weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """

        if hasattr(self.config, "embedding_type"):
            outputs = self.bert.korquad_forward(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        else:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits) + outputs[2:]  # start_logits, end_logits, (hidden_states), (attentions)
        if start_positions is not None and end_positions is not None:  # if train
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class KorSTSModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if hasattr(self.config, "embedding_type"):
            self.bert = CustomBertModel(config)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)  # cosine similarity between two sentences

    def forward(self, inputs):
        outputs = self.bert.forward(**inputs,
                                    output_hidden_states=True)  # input_ids, attention_mask, token_type_ids
        pooled_output = outputs['pooler_output']

        output_drop = self.dropout(pooled_output)
        logits = self.classifier(output_drop)
        logits = logits.view(-1)
        return outputs, logits


class NSMCModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if hasattr(self.config, "embedding_type"):
            self.bert = CustomBertModel(config)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)  # positive or negative

    def forward(self, inputs):
        outputs = self.bert.forward(**inputs,
                                    output_hidden_states=True)  # input_ids, attention_mask, token_type_ids
        pooled_output = outputs['pooler_output']

        outputs_drop = self.dropout(pooled_output)
        logits = self.classifier(outputs_drop)

        return outputs, logits


class PAWS_XModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if hasattr(self.config, "embedding_type"):
            self.bert = CustomBertModel(config)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)  # positive or negative

    def forward(self, inputs):
        outputs = self.bert.forward(**inputs,
                                    output_hidden_states=True)  # input_ids, attention_mask, token_type_ids
        pooled_output = outputs['pooler_output']

        output_drop = self.dropout(pooled_output)
        logits = self.classifier(output_drop)

        return outputs, logits


