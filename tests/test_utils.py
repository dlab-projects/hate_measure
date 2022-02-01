import numpy as np

from deephate.utils import accuracy_by_chance, weighted_correlation
from numpy.testing import assert_allclose
from statsmodels.stats.weightstats import DescrStatsW


def test_accuracy_by_chance():
    """Tests that accuracy by chance is correctly calculated."""
    p_train = 0.5
    p_test = 0.5
    acc = accuracy_by_chance(p_train, p_test)
    assert acc == 0.5
    # Empirical analysis
    p_train = 0.4
    p_test = 0.6
    train = np.random.choice(
        [0, 1],
        p=[p_train, 1 - p_train],
        size=100000,
        replace=True)
    test = np.random.choice(
        [0, 1],
        p=[p_test, 1 - p_test],
        size=100000,
        replace=True)
    empirical = (train == test).mean()
    true = accuracy_by_chance(p_train, p_test)
    assert_allclose(empirical, true, rtol=1e-1)


def test_weighted_correlation():
    """Tests that weighted correlations are calculated correctly."""
    D = 100
    # Test that weighted correlation matches numpy's correlation function
    x = np.random.normal(size=(D, 2))
    weights = np.ones(shape=D)
    observed = weighted_correlation(x[:, 0], x[:, 1], weights)
    true = np.corrcoef(x[:, 0], x[:, 1])[0, 1]
    assert_allclose(observed, true)
    # Test weights
    weights = np.random.uniform(low=0, high=1, size=D)
    stats = DescrStatsW(x, weights=weights)
    observed = weighted_correlation(x[:, 0], x[:, 1], weights)
    true = stats.corrcoef[0, 1]
    assert_allclose(observed, true)
