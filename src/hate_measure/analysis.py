import numpy as np

from .utils import (cv_probs_to_preds_multi_output,
                    weighted_correlation)


def cv_probs_to_scores(probs, hs_measures):
    """Converts probability predictions to hate speech measure proxy scores.

    Parameters
    ----------
    probs : list
        A list of lists. The outer list corresponds to cross-validation folds,
        while the inner list contains the multi-output probability predictions.
        Within the inner list is a series of np.ndarrays, with first dimension
        corresponding to number of samples, and second dimension corresponding
        to number of categories for that output.
    hs_measures : list
        A list of np.ndarrays. Each entry corresponds to a cross-validation
        fold. The arrays denote hate speech measures for the samples in each
        fold.

    Returns
    -------
    scores : np.ndarray
        The sum of predictions across outputs.
    """
    # Convert probabilities to predictions
    preds = cv_probs_to_preds_multi_output(probs)
    # Sum up predicted labels to obtain a hate speech proxy score
    scores = [np.sum(pred, axis=1) for pred in preds]
    return scores


def cv_corr_probs_hate_speech(probs, hs_measures, weights=None):
    """Calculates the correlation between the hate-speech measure and the
    predicted probabilities according to a model.

    This function assumes the predicted probabilities occur over a list of
    CV folds.

    Parameters
    ----------
    probs : list
        A list of lists. The outer list corresponds to cross-validation folds,
        while the inner list contains the multi-output probability predictions.
        Within the inner list is a series of np.ndarrays, with first dimension
        corresponding to number of samples, and second dimension corresponding
        to number of categories for that output.
    hs_measures : list
        A list of np.ndarrays. Each entry corresponds to a cross-validation
        fold. The arrays denote hate speech measures for the samples in each
        fold.
    weights : list of np.ndarrays, default None
        The weights to apply when calculating correlations. If None, all
        weights are set to 1.

    Returns
    -------
    corrs : np.ndarray
        The correlation between hate speech measure and proxy for each
        cross-validation fold.
    """
    if weights is None:
        weights = [np.ones(hs_measure.size) for hs_measure in hs_measures]
    # Sum up predicted labels to obtain a hate speech proxy score
    scores = cv_probs_to_scores(probs, hs_measures)
    # Calculate correlations between proxy and actual hate speech measure
    corrs = np.array([weighted_correlation(score, hs_measure, weight)
                      for score, hs_measure, weight in zip(scores, hs_measures, weights)])
    return corrs
