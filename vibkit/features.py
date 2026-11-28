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


FREQ_FEATURE_NAMES = [
    "mean_freq", "freq_std", "rms_freq", "peak_freq", "spectral_entropy",
]


def frequency_features(windows, fs, eps=1e-12):
    """Compute 5 frequency-domain features per window from its power spectrum.

    mean_freq        : spectral centroid (Hz)
    freq_std         : spread of the spectrum around the centroid (Hz)
    rms_freq         : root-mean-square frequency (Hz)
    peak_freq        : frequency of the largest spectral line (Hz)
    spectral_entropy : normalized Shannon entropy of the spectrum, in [0, 1]
                       (low = energy in a few lines, high = noise-like)

    Returns np.ndarray, shape (n_windows, 5) — columns follow ``FREQ_FEATURE_NAMES``.
    """
    w = np.atleast_2d(np.asarray(windows, dtype=float))
    w = w - w.mean(axis=1, keepdims=True)
    p = np.abs(np.fft.rfft(w, axis=1)) ** 2
    f = np.fft.rfftfreq(w.shape[1], d=1.0 / fs)
    prob = p / (p.sum(axis=1, keepdims=True) + eps)
    mean_f = prob @ f
    std_f = np.sqrt(np.sum(prob * (f[None, :] - mean_f[:, None]) ** 2, axis=1))
    rms_f = np.sqrt(prob @ f ** 2)
    peak_f = f[np.argmax(p, axis=1)]
    ent = -np.sum(prob * np.log(prob + eps), axis=1) / np.log(f.size)
    return np.column_stack([mean_f, std_f, rms_f, peak_f, ent])
