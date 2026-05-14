import numpy as np
from scipy.stats import skew, kurtosis


def ordinary_moments_columns(X):
    """Return mean, variance, skewness, kurtosis per column (excess kurtosis, unbiased)."""
    return {
        "mean": X.mean(axis=0),
        "variance": X.var(axis=0, ddof=1),
        "skewness": skew(X, axis=0, bias=False),
        "kurtosis": kurtosis(X, axis=0, bias=False),
    }
