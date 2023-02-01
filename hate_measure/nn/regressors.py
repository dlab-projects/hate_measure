import numpy as np
import tensorflow as tf
import transformers

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense,
                                     Dropout,
                                     GlobalAveragePooling1D,
                                     Input)
from ..utils import initialize_use_layer


class HateSpeechMeasurer(Model):
    """Measures the hatefulness of input text.

    This model stacks a dense layer on top of an input transformer model (e.g.,
    RoBERTa), followed by a regression layer. The regression layer aims to
    directly predict the hate speech score obtained via Item Response Theory.

    Attributes
    ----------
    base_model : tf.keras.Model
        The transformer model that processes input text.

    Parameters
    ----------
    transformer : transformers.models
        Transformer model serving as the input.
    n_dense : int
        The number of feedforward units after the transformer model.
    dropout_rate : float
        The dropout rate applied to the dense layer.
    """
    def __init__(self, transformer='roberta-base', n_dense=64, dropout_rate=0.1):
        super(HateSpeechMeasurer, self).__init__()
        # Instantiate a fresh transformer if provided the correct string.
        if transformer == 'roberta-base' or transformer == 'roberta-large':
            self.transformer = transformers.TFRobertaModel.from_pretrained(transformer)
        else:
            # Otherwise, assume a transformer has been provided
            self.transformer = transformer
        # Transformer input saved for config
        self.transformer_config = transformer
        self.n_dense = n_dense
        self.dense = Dense(n_dense, activation='relu')
        self.dropout_rate = dropout_rate
        self.dropout = Dropout(dropout_rate)
        self.output_layer = Dense(1, activation='linear')

    @classmethod
    def build_model(cls, transformer='roberta-base', max_length=512, n_dense=64,
                    dropout_rate=0.1):
        """Builds a model using the Functional API."""
        input_ids = Input(shape=(max_length,),
                          dtype=tf.int32,
                          name='input_ids')
        attention_mask = Input(shape=(max_length,),
                               dtype=tf.int32,
                               name='attention_mask')
        network = cls(transformer=transformer,
                      n_dense=n_dense,
                      dropout_rate=dropout_rate)
        outputs = network.call(inputs=[input_ids, attention_mask])
        model = Model(inputs=[input_ids, attention_mask],
                      outputs=outputs)
        return model

    def call(self, inputs):
        """Forward pass. Inputs must be a list of length 3, with the first two
        entries being the transformer input, and the third entry as the
        severity.
        """
        input_ids = inputs[0]
        attention_mask = inputs[1]
        # Apply transformer and get classifier output
        x = self.transformer.roberta(input_ids, attention_mask)
        x = GlobalAveragePooling1D()(x.last_hidden_state)
        # Apply dense layer with dropout
        x = self.dense(x)
        x = self.dropout(x)
        # Apply final regressor layer
        x = self.output_layer(x)
        return x

    def get_config(self):
        return {'transformer': self.transformer_input,
                'n_dense': self.n_dense,
                'dropout_rate': self.dropout_rate}