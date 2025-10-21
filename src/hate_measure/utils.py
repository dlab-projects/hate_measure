import numpy as np
import torch

from hate_measure.constants import items as item_names


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


def recode_responses(
    data, items=None, sentiment={}, respect={}, insult={}, humiliate={},
    status={}, dehumanize={}, violence={}, genocide={}, attack_defend={},
    hatespeech={}
):
    """Recodes the item responses.
    Parameters
    ----------
    data : pd.DataFrame
        Hate speech dataset.
    items : dict of dicts
        A dict whose keys must be the item names, and whose values are additional
        dicts containing the recoding. If this is not provided, it is created
        from the remaining arguments.
    sentiment, respect, insult, humiliate, status, dehumanize, violence, genocide,
    attack_defend, hatespeech : dict
        The recoding for each hate speech item as a dict, with keys denoting
        the values to replace, and values denoting what to replace them with.
    """
    if items is None:
        items = dict(zip(item_names,
                         [sentiment, respect, insult, humiliate, status,
                          dehumanize, violence, genocide, attack_defend, hatespeech]))
    return data.replace(items)


def weighted_correlation(x, y, weights):
    """Calculates a weighted Pearson correlation coefficient.

    Parameters
    ----------
    x, y : np.ndarrays
        The data with which to evaluate the correlation.
    weights : np.ndarray
        The weight to apply to each sample.

    Returns
    -------
    corr : float
        The weighted Pearson correlation coefficient.
    """
    norm = np.sum(weights)
    # Center the data
    x_mean = (x @ weights) / norm
    x_res = x - x_mean
    y_mean = (y @ weights) / norm
    y_res = y - y_mean
    # Calculate variances and covariance
    x_var = (x_res**2 @ weights) / norm
    y_var = (y_res**2 @ weights) / norm
    xy_covar = np.sum(weights * x_res * y_res) / norm
    # Evaluate weighted correlation
    corr = xy_covar / np.sqrt(x_var * y_var)
    return corr
