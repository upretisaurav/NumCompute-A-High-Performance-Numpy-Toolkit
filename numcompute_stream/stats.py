"""Descriptive statistics along an axis, ignoring NaN."""
import numpy as np

def mean(data, axis=0):
    """Mean along axis, ignoring NaN."""
    return np.nanmean(data,axis)

def median(data,axis=0):
    """Median along axis, ignoring NaN."""
    return np.nanmedian(data,axis)

def stan_dev(data,axis=0):
    """Standard deviation along axis, ignoring NaN."""
    return np.nanstd(data,axis)

def min(data, axis=0):
    """Minimum along axis, ignoring NaN."""
    return np.nanmin(data,axis)

def max(data, axis=0):
    """Maximum along axis, ignoring NaN."""
    return np.nanmax(data, axis)

def histogram(data, bins=10):
    """np.histogram of the non-NaN values."""
    return np.histogram(data[~np.isnan(data)],bins)

def quantile(data, q):
    """Quantile(s) q, ignoring NaN."""
    return np.nanquantile(data, q)