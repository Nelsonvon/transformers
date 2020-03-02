from transformers import BertModel, BertPreTrainedModel
from transformers.configuration_bert import BertMultipleLabelConfig
import torch
from torch import nn
from torch.nn import LSTM, CrossEntropyLoss, Sigmoid
import logging

logger = logging.getLogger(__name__)


class BertNerTagPreditionHead(nn.Module):
    def __init__(self, config:BertMultipleLabelConfig):
        super(BertNerTagPreditionHead, self).__init__()
        self.tag_decoder = nn.Linear(config.lstm_hidden_size*2, config.num_tags)
        # TODO: crf_layer
    
    def forward(self, sequence_output):
        output = self.tag_decoder(sequence_output)
        # TODO: crf_layer
        return output

class BertNerMultipleTypePredictionHead(nn.Module):
    def __init__(self, config: BertMultipleLabelConfig):
        super(BertNerMultipleTypePredictionHead, self).__init__()
        self.type_decoder = nn.Linear(config.lstm_hidden_size*2, config.reference_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.reference_size))
        self.sig = Sigmoid()
        # TODO: activation function

        pass

    def forward(self, sequence_output):
        # hidden_states = self.transform(sequence_output) # TODO: rewrite transform function
        hidden_states = self.type_decoder(sequence_output) + self.bias
        hidden_states = self.sig(hidden_states)
        return hidden_states


# TODO: Implement tag prediction with no crf layer first.
class BertForMultipleLabelTokenClassification(BertPreTrainedModel):
    r"""
        **tag_ids**: ``torch.LongTensor`` of shape (batch_size, sequence_length)
        IOB (IOBES) tags of the token.
            I: 0
            O: 1
            B: 2
            E: 3
            S: 4

        **label_type_ids**: ``torch.LongTensor`` of shape (batch_size, sequence_length, reference_size)
        Existance of types of the current token
            ``0`` indicates token w_i is not labeled as type t_j
            ``1`` indicates token w_i is labeled as type t_j

    """
    def __init__(self, config:BertMultipleLabelConfig):
        super(BertForMultipleLabelTokenClassification, self).__init__(config)
        self.bert = BertModel(config)
        # self.linear = nn.Linear(in_features= config.hidden_size, out_features=config.lstm_hidden_size)
        self.bilstm = LSTM(input_size = config.hidden_size,
                            hidden_size = config.lstm_hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            dropout=config.lstm_dropout_prob,
                            bidirectional=True)
        self.num_labels = config.reference_size
        self.num_tags = config.num_tags
        self.tag_prediction = BertNerTagPreditionHead(config)
        self.type_prediction = BertNerMultipleTypePredictionHead(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                tag_ids=None, label_type_ids=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        bert_output = outputs[0]
        logger.info(bert_output.size())
        # lstm_input = self.linear(bert_output)
        # logger.info(lstm_input.size())
        sequence_output,(h_n, c_n) = self.bilstm(input=bert_output)
        outputs = (sequence_output,) + outputs
        logger.info(sequence_output.size())
        logger.info(h_n.size())
        # tag_output = self.tag_prediction(sequence_output)
        type_output = self.type_prediction(sequence_output)
        tag_output = self.tag_prediction(sequence_output)
        outputs = (tag_output, type_output,) + outputs
        loss_fct = CrossEntropyLoss()
        tag_loss = loss_fct(tag_output.view(-1, self.num_tags), tag_ids.view(-1))
        type_loss = loss_fct(type_output.view(-1, 1),label_type_ids.view(-1))

        total_loss = tag_loss + type_loss # TODO: weight the terms

        outputs = (total_loss,) + outputs
        return outputs

