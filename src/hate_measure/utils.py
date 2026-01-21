import numpy as np
import torch

from hate_measure.constants import ITEMS as item_names


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
