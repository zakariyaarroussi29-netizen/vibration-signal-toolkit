"""Simulate an outer-race bearing fault and detect it with envelope analysis.

Run:  python examples/demo_outer_race_fault.py
Saves figures to examples/figures/.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vibkit import (amplitude_spectrum, envelope_spectrum, spectrogram,  # noqa: E402
                    fault_frequencies, time_features, FEATURE_NAMES, select_band)
from vibkit.bearing import CWRU_DE_6205  # noqa: E402


def simulate_outer_race_fault(fs=12_000, duration=1.0, shaft_rpm=1797,
                              resonance_hz=3_000, damping=600, snr_db=0,
                              seed=0):
    """Periodic decaying impulses at BPFO exciting a structural resonance,
    plus shaft-rate vibration and white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration)) / fs
    shaft_hz = shaft_rpm / 60
    bpfo = fault_frequencies(CWRU_DE_6205, shaft_hz)["BPFO"]

    impulses = np.zeros_like(t)
    for t0 in np.arange(0, duration, 1 / bpfo):
        tau = t - t0
        m = tau >= 0
        impulses[m] += np.exp(-damping * tau[m]) * np.sin(2 * np.pi * resonance_hz * tau[m])

    shaft = 0.3 * np.sin(2 * np.pi * shaft_hz * t)
    clean = impulses + shaft
    noise_power = np.mean(impulses ** 2) / 10 ** (snr_db / 10)
    noise = rng.normal(0, np.sqrt(noise_power), t.size)
    return t, clean + noise, bpfo


def main():
    fs = 12_000
    t, x, bpfo = simulate_outer_race_fault(fs=fs)
    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)

    feats = dict(zip(FEATURE_NAMES, time_features(x)[0]))
    print(f"Theoretical BPFO: {bpfo:.1f} Hz")
    print(f"Kurtosis: {feats['kurtosis']:.2f}  (Gaussian noise ~ 3)")

    f, a = amplitude_spectrum(x, fs)
    res = select_band(x, fs)  # automatic band via spectral kurtosis
    lo, hi = res["band"]
    print(f"Band selected by spectral kurtosis: {lo:.0f}-{hi:.0f} Hz (SK = {res['sk']:.1f})")
    fe, ae = envelope_spectrum(x, fs, band=res["band"])
    sel = (fe > 20) & (fe < 500)
    peak = fe[sel][np.argmax(ae[sel])]
    print(f"Largest envelope-spectrum peak: {peak:.1f} Hz")

    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    axes[0].plot(t[:2400], x[:2400], lw=0.6)
    axes[0].set(title="Raw signal (first 0.2 s)", xlabel="Time [s]", ylabel="Amplitude")
    axes[1].plot(f, a, lw=0.6)
    axes[1].set(title="Amplitude spectrum", xlabel="Frequency [Hz]", xlim=(0, fs / 2))
    axes[2].plot(fe, ae, lw=0.8)
    for k in range(1, 4):
        axes[2].axvline(k * bpfo, color="r", ls="--", lw=0.8,
                        label="BPFO harmonics" if k == 1 else None)
    axes[2].set(title=f"Envelope spectrum (auto band {lo:.0f}–{hi:.0f} Hz)", xlabel="Frequency [Hz]",
                xlim=(0, 500))
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out / "outer_race_fault.png", dpi=120)

    fs_, ts_, s = spectrogram(x, fs, nperseg=256)
    fig2, ax = plt.subplots(figsize=(9, 4))
    ax.pcolormesh(ts_, fs_, s, shading="auto")
    ax.set(title="Spectrogram (dB)", xlabel="Time [s]", ylabel="Frequency [Hz]")
    fig2.tight_layout()
    fig2.savefig(out / "spectrogram.png", dpi=120)
    print(f"Figures saved to {out}")


if __name__ == "__main__":
    main()
