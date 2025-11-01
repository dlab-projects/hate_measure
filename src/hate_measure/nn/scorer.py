import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from hate_measure.utils import masked_average_pooling
from .config import HateSpeechScorerConfig

class HateSpeechScorer(PreTrainedModel):
    config_class = HateSpeechScorerConfig
    def __init__(self, config):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(config.encoder_model_name_or_path)
        hidden = self.encoder.config.hidden_size
        self.dense = nn.Linear(hidden, config.n_dense)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = nn.ReLU()
        self.regressor = nn.Linear(config.n_dense, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled = masked_average_pooling(out.last_hidden_state, attention_mask)
        logits = self.regressor(self.act(self.dropout(self.dense(pooled))))
        loss = None
        if labels is not None:
            loss = F.mse_loss(logits.view(-1), labels.view(-1).float())
        return SequenceClassifierOutput(loss=loss, logits=logits)