import numpy as np
import tensorflow as tf
import transformers

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense,
                                     Dropout,
                                     GlobalAveragePooling1D,
                                     GlobalMaxPooling1D,
                                     Input,
                                     concatenate)

from .layers import HateConstructLayer, TargetIdentityLayer
from ..utils import initialize_use_layer


class ConstructClassifierSeverity(Model):
    """Classifies input text according to a hate speech construct via item
    responses.

    This model stacks a dense layer on top of an input transformer model (e.g.,
    RoBERTa), followed by a hate speech construct classification layer. See the
    `HateConstructLayer` layer for details on the endpoint of this model.

    Attributes
    ----------
    transformer : transformers.models
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
        super(ConstructClassifierSeverity, self).__init__()
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
        self.dropout = Dropout(dropout_rate)
        self.hate_construct = HateConstructLayer()

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
        severity = Input(shape=(1,), name='severity')
        network = cls(transformer=transformer,
                      n_dense=n_dense,
                      dropout_rate=dropout_rate)
        outputs = network.call(inputs=[input_ids, attention_mask, severity])
        model = Model(inputs=[input_ids, attention_mask, severity],
                      outputs=outputs)
        return model

    def call(self, inputs):
        """Forward pass. Inputs must be a list of length 3, with the first two
        entries being the transformer input, and the third entry as the
        severity.
        """
        # Separate input
        severity = inputs[2]
        # Apply transformer and get classifier output
        x = self.transformer([inputs[0], inputs[1]])
        x = GlobalAveragePooling1D()(x.last_hidden_state)
        # Apply dense layer with dropout
        x = self.dense(x)
        x = self.dropout(x)
        # Incorporate severity input
        x = concatenate([x, severity])
        # Hate construct prediction
        x = self.hate_construct(x)
        return x

    def get_config(self):
        return {'transformer': self.transformer_input,
                'n_dense': self.dense.units,
                'dropout_rate': self.dropout.rate}


class ConstructClassifierUSE(Model):
    """Classifies input text according to a hate speech construct via item
    responses, using the Universal Sentence Encoder (USE).

    This model stacks a dense layer on top of USE followed by a hate speech
    construct classification layer. See the`HateConstructLayer` layer for
    details on the endpoint of this model.

    Attributes
    ----------
    transformer : transformers.models
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
    def __init__(self, version='v4', n_dense=64, dropout_rate=0.1):
        super(ConstructClassifierUSE, self).__init__()
        # Instantiate a fresh USE
        self.version = version
        self.USE = initialize_use_layer(version)
        self.n_dense = n_dense
        self.dense = Dense(n_dense, activation='relu')
        self.dropout = Dropout(dropout_rate)
        self.hate_construct = HateConstructLayer()

    @classmethod
    def build_model(cls, version='v4', n_dense=64, dropout_rate=0.1):
        """Builds a model using the Functional API."""
        text = tf.keras.Input((), dtype=tf.string, name='input_text')
        severity = Input(shape=(1,), name='severity')
        network = cls(version=version,
                      n_dense=n_dense,
                      dropout_rate=dropout_rate)
        outputs = network.call(inputs=[text, severity])
        model = Model(inputs=[text, severity], outputs=outputs)
        return model

    def call(self, inputs):
        """Forward pass. Inputs must be a list of length 2, with the first
        input being the text and the second entry being the severity.
        """
        # Separate input
        severity = inputs[1]
        # Apply transformer and get classifier output
        x = self.USE(inputs[0])
        # Apply dense layer with dropout
        x = self.dense(x)
        x = self.dropout(x)
        # Incorporate severity input
        x = concatenate([x, severity])
        # Hate construct prediction
        x = self.hate_construct(x)
        return x

    def get_config(self):
        return {'version': self.version,
                'n_dense': self.dense.units,
                'dropout_rate': self.dropout.rate}


class TargetIdentityClassifier(Model):
    """Classifies the target identity of text input, if any.

    This model stacks a dense layer on top of an input transformer model (e.g.,
    RoBERTa), followed by a multi-output classification layer. See the
    `TargetIdentityLayer` layer for details on the endpoint of this model.
    Importantly, multiple identities (or none) can be targeted in a given
    text input.

    Attributes
    ----------
    transformer : transformers.models
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
    def __init__(
        self, transformer='roberta-base', n_dense=64, dropout_rate=0.1,
        pooling='max', mask_pool=True
    ):
        super(TargetIdentityClassifier, self).__init__()
        # Instantiate a fresh transformer if provided the correct string.
        self.transformer_name = transformer
        if transformer == 'roberta-base' or transformer == 'roberta-large':
            self.transformer = transformers.TFRobertaModel.from_pretrained(transformer)
        elif transformer == 'bert-base-uncased':
            self.transformer = transformers.TFBertModel.from_pretrained(transformer)
        elif transformer == 'distilbert-base-uncased':
            self.transformer = transformers.TFDistilBertModel.from_pretrained(transformer)
        else:
            # Otherwise, assume a transformer has been provided
            self.transformer_name = 'custom'
            self.transformer = transformer
        # Transformer input saved for config
        self.transformer_config = transformer
        self.n_dense = n_dense
        self.dense = Dense(n_dense, activation='relu')
        self.dropout = Dropout(dropout_rate)
        self.pooling = pooling
        self.mask_pool = mask_pool
        if self.pooling == 'max':
            self.pool = GlobalMaxPooling1D()
        elif self.pooling == 'mean':
            self.pool = GlobalAveragePooling1D()
        self.target_identity = TargetIdentityLayer()

    @classmethod
    def build_model(cls, transformer='roberta-base', max_length=512, n_dense=64,
                    dropout_rate=0.1, pooling='max', mask_pool=True):
        """Builds a model using the Functional API."""
        input_ids = Input(shape=(max_length,),
                          dtype=tf.int32,
                          name='input_ids')
        attention_mask = Input(shape=(max_length,),
                               dtype=tf.int32,
                               name='attention_mask')
        network = cls(transformer=transformer,
                      n_dense=n_dense,
                      dropout_rate=dropout_rate,
                      pooling=pooling,
                      mask_pool=mask_pool)
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
        if self.transformer_name == 'distilbert-base-uncased':
            x = self.transformer.distilbert(input_ids, attention_mask)
        elif self.transformer_name == 'bert-base-uncased':
            x = self.transformer.bert(input_ids, attention_mask)
        elif (self.transformer_name == 'roberta-base') or (self.transformer_name == 'roberta-large'):
            x = self.transformer.roberta(input_ids, attention_mask)
        # Perform pooling
        if self.pooling == 'max' and self.mask_pool:
            mask = tf.cast(tf.expand_dims(attention_mask, axis=-1), 'float32')
            mask_sum = tf.cast(tf.math.reduce_sum(attention_mask, axis=-1, keepdims=True), 'float32')
            x = tf.math.reduce_sum(
                x.last_hidden_state * mask,
                axis=1) / mask_sum
        elif self.pooling == 'mean' and self.mask_pool:
            x = self.pool(x.last_hidden_state, mask=attention_mask)
        else:
            x = self.pool(x.last_hidden_state)
        # Apply dense layer with dropout
        x = self.dense(x)
        x = self.dropout(x)
        # Target identity prediction
        x = self.target_identity(x)
        return x

    def get_config(self):
        return {'transformer': self.transformer_config,
                'n_dense': self.dense.units,
                'dropout_rate': self.dropout.rate}


class TargetIdentityClassifierUSE(Model):
    """Classifies the target identity of text input, if any.

    This model stacks a dense layer on top of an input Universal Sentence
    Encoder (USE) model, followed by a multi-output classification layer. See
    the `TargetIdentityLayer` layer for details on the endpoint of this model.
    Importantly, multiple identities (or none) can be targeted in a given
    text input.

    Attributes
    ----------
    transformer : transformers.models
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
    def __init__(self, version='v4', n_dense=64, dropout_rate=0.1):
        super(TargetIdentityClassifierUSE, self).__init__()
        # Instantiate a fresh USE
        self.version = version
        self.USE = initialize_use_layer(version)
        self.n_dense = n_dense
        self.dense = Dense(n_dense, activation='relu')
        self.dropout = Dropout(dropout_rate)
        self.target_identity = TargetIdentityLayer()

    @classmethod
    def build_model(cls, version='v4', n_dense=64, dropout_rate=0.1):
        """Builds a model using the Functional API."""
        text = tf.keras.Input((), dtype=tf.string, name='input_text')
        network = cls(version=version,
                      n_dense=n_dense,
                      dropout_rate=dropout_rate)
        outputs = network.call(inputs=text)
        model = Model(inputs=text, outputs=outputs)
        return model

    def call(self, inputs):
        """Forward pass. Inputs must be a list of length 3, with the first two
        entries being the transformer input, and the third entry as the
        severity.
        """
        # Apply transformer and get classifier output
        x = self.USE(inputs)
        # Apply dense layer with dropout
        x = self.dense(x)
        x = self.dropout(x)
        # Target identity prediction
        x = self.target_identity(x)
        return x

    def get_config(self):
        return {'version': self.version,
                'n_dense': self.dense.units,
                'dropout_rate': self.dropout.rate}
