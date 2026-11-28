"""Frequency and time-frequency analysis of vibration signals."""

import numpy as np
from scipy import signal as sps


def amplitude_spectrum(x, fs, window="hann"):
    """One-sided amplitude spectrum.

    Returns (freqs, amplitude). A pure sine of amplitude A appears as a
    peak of height ~A (window gain is compensated).
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x - x.mean()
    n = x.size
    win = sps.get_window(window, n) if window else np.ones(n)
    spec = np.abs(np.fft.rfft(x * win)) / win.sum()
    spec[1:] *= 2  # one-sided
    if n % 2 == 0:
        spec[-1] /= 2  # Nyquist bin is not doubled
    return np.fft.rfftfreq(n, d=1.0 / fs), spec


def bandpass(x, fs, low, high, order=4):
    """Zero-phase Butterworth band-pass filter."""
    sos = sps.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, np.asarray(x, dtype=float))


def envelope_spectrum(x, fs, band=None, window="hann"):
    """Envelope (demodulation) spectrum, the standard tool for bearing faults.

    Optionally band-pass filter around a resonance first (``band=(low, high)``),
    then take the Hilbert envelope and its amplitude spectrum. Peaks appear at
    the fault characteristic frequencies (BPFO, BPFI, ...) and their harmonics.
    """
    x = np.asarray(x, dtype=float).ravel()
    if band is not None:
        x = bandpass(x, fs, *band)
    env = np.abs(sps.hilbert(x))
    return amplitude_spectrum(env, fs, window=window)


def spectral_kurtosis(x, fs, nperseg=128):
    """Spectral kurtosis (SK) from the STFT.

    SK(f) = <|X(t,f)|^4> / <|X(t,f)|^2>^2 - 2. It is ~0 for stationary
    Gaussian noise and large in frequency bands where the signal is
    impulsive, which is where bearing-fault impacts excite resonances.

    Returns (freqs, sk).
    """
    f, _, z = sps.stft(np.asarray(x, dtype=float), fs=fs, nperseg=nperseg)
    p = np.abs(z) ** 2
    sk = np.mean(p ** 2, axis=1) / (np.mean(p, axis=1) ** 2 + 1e-24) - 2
    sk[0] = 0.0  # DC bin is not meaningful
    return f, sk


def select_band(x, fs, nperseg_list=(16, 32, 64, 128)):
    """Pick the band with the highest spectral kurtosis for envelope analysis.

    A simplified kurtogram: SK is computed at several frequency resolutions;
    the winning bin defines a band centred on it with width ``fs / nperseg``.

    Finer resolutions use fewer STFT frames, so their SK estimates are noisier.
    To compare resolutions fairly, each SK value is scored against the
    standard deviation of SK for Gaussian noise, ~ 2 / sqrt(n_frames), and the
    most significant bin wins.

    Returns a dict with ``band`` (low, high) in Hz, ``center``, ``sk``,
    ``score`` and ``nperseg``. Use it as
    ``envelope_spectrum(x, fs, band=res["band"])``.
    """
    best = None
    nyq = fs / 2
    n_samples = np.asarray(x).size
    for n in nperseg_list:
        f, sk = spectral_kurtosis(x, fs, nperseg=n)
        bw = fs / n
        valid = (f - bw / 2 > 0) & (f + bw / 2 < nyq)
        if not valid.any():
            continue
        n_frames = max(1, 2 * n_samples // n)  # default STFT overlap is 50%
        score = sk * np.sqrt(n_frames) / 2
        i = np.flatnonzero(valid)[np.argmax(score[valid])]
        if best is None or score[i] > best["score"]:
            best = {"band": (f[i] - bw / 2, f[i] + bw / 2), "center": f[i],
                    "sk": float(sk[i]), "score": float(score[i]), "nperseg": n}
    if best is None:
        raise ValueError("signal too short for the requested resolutions")
    return best


def spectrogram(x, fs, nperseg=256, noverlap=None, log=True):
    """Short-time Fourier transform magnitude.

    Returns (freqs, times, S) with S of shape (n_freqs, n_times); in dB if
    ``log``. Useful as a 2-D input image for CNNs.
    """
    f, t, z = sps.stft(np.asarray(x, dtype=float), fs=fs,
                       nperseg=nperseg, noverlap=noverlap)
    s = np.abs(z)
    if log:
        s = 20 * np.log10(s + 1e-12)
    return f, t, s
