"""Classical time-domain condition indicators."""

import numpy as np
from scipy import stats

FEATURE_NAMES = [
    "mean", "std", "rms", "peak", "peak_to_peak",
    "skewness", "kurtosis",
    "crest_factor", "shape_factor", "impulse_factor", "clearance_factor",
]


def time_features(windows, eps=1e-12):
    """Compute 11 standard time-domain features per window.

    Parameters
    ----------
    windows : array_like, shape (n,) or (n_windows, n)

    Returns
    -------
    np.ndarray, shape (n_windows, 11) — columns follow ``FEATURE_NAMES``.
    Kurtosis is the Pearson (non-excess) value: ~3 for Gaussian noise,
    clearly higher for impulsive signals such as localized bearing faults.
    """
    w = np.atleast_2d(np.asarray(windows, dtype=float))
    absw = np.abs(w)
    mean = w.mean(axis=1)
    std = w.std(axis=1)
    rms = np.sqrt(np.mean(w ** 2, axis=1))
    peak = absw.max(axis=1)
    p2p = w.max(axis=1) - w.min(axis=1)
    skew = stats.skew(w, axis=1)
    kurt = stats.kurtosis(w, axis=1, fisher=False)
    mean_abs = absw.mean(axis=1)
    sqrt_amp = np.mean(np.sqrt(absw), axis=1) ** 2
    crest = peak / (rms + eps)
    shape = rms / (mean_abs + eps)
    impulse = peak / (mean_abs + eps)
    clearance = peak / (sqrt_amp + eps)
    return np.column_stack(
        [mean, std, rms, peak, p2p, skew, kurt, crest, shape, impulse, clearance]
    )
