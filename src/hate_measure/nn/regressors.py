import torch
from torch.nn import Dropout, Linear, Module, ReLU
from transformers import AutoTokenizer, AutoModel
from hate_measure.utils import masked_average_pooling


class HateSpeechScorer(Module):
    def __init__(
        self, base='AnswerDotAI/ModernBERT-base', n_dense=128, dropout_rate=0.4,
        max_length=512, padding='max_length', truncation=True
    ):
        super(HateSpeechScorer, self).__init__()
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
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
        encoding = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length)
        return encoding['input_ids'], encoding['attention_mask']

    def predict(self, texts):
        input_ids, attention_mask = self.tokenize(texts)
        with torch.no_grad():
            score = self.forward(input_ids, attention_mask)
        return score