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
