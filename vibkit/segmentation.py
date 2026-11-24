"""Cutting long vibration records into fixed-length samples."""

import numpy as np


def sliding_windows(signal, length, stride=None, max_windows=None):
    """Cut a 1-D signal into windows of ``length`` points.

    Parameters
    ----------
    signal : array_like, shape (n,)
    length : int
        Number of points per window.
    stride : int, optional
        Step between window starts. Defaults to ``length`` (no overlap).
        A smaller stride gives overlapping windows (data augmentation).
    max_windows : int, optional
        Keep at most this many windows (taken from the start).

    Returns
    -------
    np.ndarray, shape (n_windows, length)
    """
    x = np.asarray(signal, dtype=float).ravel()
    stride = length if stride is None else stride
    if length <= 0 or stride <= 0:
        raise ValueError("length and stride must be positive")
    if x.size < length:
        return np.empty((0, length))
    n = 1 + (x.size - length) // stride
    if max_windows is not None:
        n = min(n, max_windows)
    idx = np.arange(length)[None, :] + stride * np.arange(n)[:, None]
    return x[idx]


def normalize(windows, method="zscore", eps=1e-12):
    """Normalize each window independently.

    method : "zscore" (zero mean, unit std), "minmax" (to [-1, 1]) or "none".
    """
    w = np.asarray(windows, dtype=float)
    if method == "none":
        return w
    if method == "zscore":
        mu = w.mean(axis=-1, keepdims=True)
        sd = w.std(axis=-1, keepdims=True)
        return (w - mu) / (sd + eps)
    if method == "minmax":
        lo = w.min(axis=-1, keepdims=True)
        hi = w.max(axis=-1, keepdims=True)
        return 2 * (w - lo) / (hi - lo + eps) - 1
    raise ValueError(f"unknown method: {method}")
