import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from .config import HateSpeechScorerConfig


def masked_average_pooling(hidden_states, attention_mask):
    """Average pool hidden states, weighted by attention mask."""
    masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1)
    sum_hidden_states = masked_hidden_states.sum(dim=1)
    sum_mask = attention_mask.sum(dim=1, keepdim=True)
    return sum_hidden_states / sum_mask


class HateSpeechScorer(PreTrainedModel):
    config_class = HateSpeechScorerConfig

    def __init__(self, config: HateSpeechScorerConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config.encoder_config)
        n_hidden = self.encoder.config.hidden_size
        # Begin fine-tuning layers
        self.dense = nn.Linear(n_hidden, config.n_dense)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.activation = nn.ReLU()
        self.regressor = nn.Linear(config.n_dense, config.num_labels)
        self.loss_fn = nn.MSELoss()
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        # Remove Trainer-specific args that the encoder doesn't accept
        kwargs.pop("num_items_in_batch", None)
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        pooled = masked_average_pooling(out.last_hidden_state, attention_mask)
        x = self.dense(pooled)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.regressor(x)

        loss = None
        if labels is not None:
            if labels.dim() == 1 and x.size(-1) == 1:
                labels = labels.unsqueeze(-1)
            loss = self.loss_fn(x, labels)

        return SequenceClassifierOutput(loss=loss, logits=x)