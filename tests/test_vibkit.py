import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vibkit import (sliding_windows, normalize, time_features, FEATURE_NAMES,  # noqa: E402
                    amplitude_spectrum, envelope_spectrum, spectrogram,
                    BearingGeometry, fault_frequencies)
from vibkit.bearing import CWRU_DE_6205  # noqa: E402


def test_sliding_windows_no_overlap():
    w = sliding_windows(np.arange(10), 4)
    assert w.shape == (2, 4)
    assert w[1, 0] == 4


def test_sliding_windows_overlap_and_cap():
    w = sliding_windows(np.arange(10), 4, stride=2)
    assert w.shape == (4, 4)
    assert sliding_windows(np.arange(10), 4, stride=1, max_windows=3).shape == (3, 4)
    assert sliding_windows(np.arange(3), 4).shape == (0, 4)


def test_normalize():
    w = np.random.default_rng(0).normal(5, 3, (4, 100))
    z = normalize(w)
    assert np.allclose(z.mean(1), 0, atol=1e-9) and np.allclose(z.std(1), 1)
    m = normalize(w, "minmax")
    assert np.allclose(m.min(1), -1) and np.allclose(m.max(1), 1, atol=1e-9)


def test_time_features_sine():
    t = np.arange(10_000) / 10_000
    x = np.sin(2 * np.pi * 50 * t)
    f = dict(zip(FEATURE_NAMES, time_features(x)[0]))
    assert f["rms"] == pytest.approx(1 / np.sqrt(2), rel=1e-3)
    assert f["crest_factor"] == pytest.approx(np.sqrt(2), rel=1e-3)
    assert f["kurtosis"] == pytest.approx(1.5, rel=1e-2)


def test_gaussian_kurtosis_near_three():
    x = np.random.default_rng(1).normal(size=200_000)
    assert time_features(x)[0, FEATURE_NAMES.index("kurtosis")] == pytest.approx(3, abs=0.05)


def test_amplitude_spectrum_peak():
    fs = 1000
    t = np.arange(fs) / fs
    f, a = amplitude_spectrum(2.0 * np.sin(2 * np.pi * 100 * t), fs)
    i = np.argmax(a)
    assert f[i] == pytest.approx(100)
    assert a[i] == pytest.approx(2.0, rel=0.01)


def test_cwru_fault_frequencies():
    # Published CWRU multiples of shaft speed: BPFO 3.5848, BPFI 5.4152, BSF 4.7135, FTF 0.39828
    ff = fault_frequencies(CWRU_DE_6205, shaft_hz=1.0)
    assert ff["BPFO"] == pytest.approx(3.5848, rel=1e-3)
    assert ff["BPFI"] == pytest.approx(5.4152, rel=1e-3)
    assert ff["BSF"] == pytest.approx(2.357, rel=1e-3)  # x2 = 4.7135 (two contacts per spin)
    assert ff["FTF"] == pytest.approx(0.39828, rel=1e-3)


def test_envelope_spectrum_finds_bpfo():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from demo_outer_race_fault import simulate_outer_race_fault
    fs = 12_000
    _, x, bpfo = simulate_outer_race_fault(fs=fs, snr_db=-5)
    f, a = envelope_spectrum(x, fs, band=(2_000, 4_000))
    sel = (f > 20) & (f < 500)
    assert f[sel][np.argmax(a[sel])] == pytest.approx(bpfo, abs=2)


def test_spectrogram_shape():
    f, t, s = spectrogram(np.random.default_rng(2).normal(size=4096), fs=1000, nperseg=128)
    assert s.shape == (f.size, t.size)


def test_contact_angle_changes_frequencies():
    g0 = BearingGeometry(9, 0.3, 1.5, 0)
    g15 = BearingGeometry(9, 0.3, 1.5, 15)
    assert fault_frequencies(g15, 30)["BPFO"] > fault_frequencies(g0, 30)["BPFO"]


# ---- v0.2.0: spectral kurtosis band selection & frequency features ----

def test_select_band_finds_resonance():
    from vibkit import select_band
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from demo_outer_race_fault import simulate_outer_race_fault
    fs = 12_000
    _, x, bpfo = simulate_outer_race_fault(fs=fs, snr_db=-5, resonance_hz=3_000)
    res = select_band(x, fs)
    lo, hi = res["band"]
    assert 2_000 < (lo + hi) / 2 < 4_000  # inside the excited resonance region
    f, a = envelope_spectrum(x, fs, band=res["band"])
    sel = (f > 20) & (f < 500)
    assert f[sel][np.argmax(a[sel])] == pytest.approx(bpfo, abs=2)


def test_spectral_kurtosis_gaussian_near_zero():
    from vibkit import spectral_kurtosis
    x = np.random.default_rng(3).normal(size=60_000)
    _, sk = spectral_kurtosis(x, fs=1000, nperseg=128)
    assert np.median(np.abs(sk[1:-1])) < 0.2


def test_frequency_features_sine():
    from vibkit import frequency_features, FREQ_FEATURE_NAMES
    fs = 1000
    t = np.arange(1000) / fs
    sine = dict(zip(FREQ_FEATURE_NAMES, frequency_features(np.sin(2 * np.pi * 120 * t), fs)[0]))
    assert sine["peak_freq"] == pytest.approx(120)
    assert sine["mean_freq"] == pytest.approx(120, abs=1)
    noise = frequency_features(np.random.default_rng(4).normal(size=1000), fs)[0]
    assert sine["spectral_entropy"] < 0.1 < 0.8 < noise[4]
