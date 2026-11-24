"""Loaders for public bearing datasets stored locally.

Data files are not included in this repository; download them from the
original providers and point the loaders at your local copy.
"""

from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat

# CWRU sampling rates (Hz)
CWRU_FS = {"12k": 12_000, "48k": 48_000}

_CHANNEL_SUFFIX = {"DE": "_DE_time", "FE": "_FE_time", "BA": "_BA_time"}


def load_cwru_mat(path, channel="DE"):
    """Read one CWRU ``.mat`` file and return (signal, rpm).

    channel : "DE" (drive end), "FE" (fan end) or "BA" (base).
    ``rpm`` is None when the file does not store it.
    """
    mat = loadmat(Path(path))
    suffix = _CHANNEL_SUFFIX[channel]
    keys = [k for k in mat if k.endswith(suffix)]
    if not keys:
        raise KeyError(f"no {channel} channel in {path}; keys: "
                       f"{[k for k in mat if not k.startswith('__')]}")
    sig = np.asarray(mat[keys[0]], dtype=float).ravel()
    rpm_keys = [k for k in mat if re.search(r"RPM$", k)]
    rpm = float(np.asarray(mat[rpm_keys[0]]).ravel()[0]) if rpm_keys else None
    return sig, rpm


def build_dataset(file_labels, length=1024, stride=None, channel="DE",
                  max_windows=None):
    """Turn {path: label} into arrays X (n, length) and y (n,).

    Example
    -------
    >>> X, y = build_dataset({"data/97.mat": 0, "data/130.mat": 1})
    """
    from .segmentation import sliding_windows

    xs, ys = [], []
    for path, label in file_labels.items():
        sig, _ = load_cwru_mat(path, channel)
        w = sliding_windows(sig, length, stride, max_windows)
        xs.append(w)
        ys.append(np.full(len(w), label))
    return np.concatenate(xs), np.concatenate(ys)
