import torch

from pytorch_lightning import LightningModule
from torch.nn import Dropout, Linear, Module, ReLU
from transformers import AutoTokenizer, AutoModel


def masked_average_pooling(hidden_states, attention_mask):
    """
    Perform average pooling on the hidden states, taking the attention mask into account.

    Args:
        hidden_states (torch.Tensor): The hidden states of shape (batch_size, sequence_length, hidden_size).
        attention_mask (torch.Tensor): The attention mask of shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: The pooled hidden states of shape (batch_size, hidden_size).
    """
    # Apply the attention mask to the hidden states
    masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1)

    # Sum the hidden states and the mask
    sum_hidden_states = masked_hidden_states.sum(dim=1)
    sum_mask = attention_mask.sum(dim=1, keepdim=True)

    # Compute the average hidden states
    average_hidden_states = sum_hidden_states / sum_mask
    return average_hidden_states


class HateSpeechMeasurerModule(Module):
    def __init__(self, base='roberta-base', n_dense=128, dropout_rate=0.4):
        super(HateSpeechMeasurerModule, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        self.base_model = AutoModel.from_pretrained(base)
        self.dense_layer = Linear(
            in_features=self.base_model.config.hidden_size,
            out_features=n_dense)
        self.dropout_rate = dropout_rate
        self.dropout = Dropout(p=dropout_rate)
        self.relu = ReLU()
        self.output_layer = Linear(
            in_features=n_dense,
            out_features=1)

    def forward(self, input_ids, attention_mask):
        # Run inputs through base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain hidden state
        x = outputs.last_hidden_state
        # Perform average pooling across the token dimension, with masking
        x = masked_average_pooling(x, attention_mask)
        # Dense layer, with dropout, and ReLU activation
        x = self.dense_layer(x)
        x = self.dropout(x)
        x = self.relu(x)
        # Dense layer to produce score
        score = self.output_layer(x)
        return score

    def tokenize(self, texts):
        encoding = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        return encoding['input_ids'], encoding['attention_mask']

    def predict(self, texts):
        input_ids, attention_mask = self.tokenize(texts)
        with torch.no_grad():
            score = self.forward(input_ids, attention_mask)
        return score