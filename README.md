# vibration-signal-toolkit

Small, readable Python tools for **vibration-based fault diagnosis of rotating machinery**:
segmenting raw records into samples, classical condition indicators, spectral and
envelope analysis, and bearing fault characteristic frequencies.

Built as a clean preprocessing base for deep-learning bearing diagnosis experiments
(e.g. on the CWRU dataset).

![Envelope analysis of a simulated outer-race fault](examples/figures/outer_race_fault.png)

## Features

| Module | What it does |
|---|---|
| `vibkit.segmentation` | Sliding-window segmentation (with overlap for augmentation), per-sample z-score / min-max normalization |
| `vibkit.features` | 11 time-domain indicators (RMS, kurtosis, crest / impulse / clearance factors, …) and 5 frequency-domain indicators (spectral centroid, spread, RMS frequency, peak frequency, spectral entropy) |
| `vibkit.spectral` | Amplitude spectrum, band-pass filtering, Hilbert **envelope spectrum**, **spectral kurtosis** with automatic band selection, STFT spectrogram |
| `vibkit.bearing` | BPFO, BPFI, BSF, FTF from bearing geometry (CWRU 6205 drive-end geometry included) |
| `vibkit.datasets` | Loader for CWRU `.mat` files → `X, y` arrays ready for training |

## Installation

```bash
git clone https://github.com/zakariyaarroussi29-netizen/vibration-signal-toolkit.git
cd vibration-signal-toolkit
pip install -r requirements.txt
```

## Quick start

```python
import numpy as np
from vibkit import sliding_windows, normalize, time_features, envelope_spectrum, fault_frequencies
from vibkit.bearing import CWRU_DE_6205

fs = 12_000                                   # CWRU 12 kHz drive-end data
x = np.load("my_signal.npy")                  # any 1-D vibration record

# 1) Cut into 1024-point samples with 50% overlap, normalize each one
X = normalize(sliding_windows(x, length=1024, stride=512))

# 2) Classical indicators (n_samples x 11)
F = time_features(X)

# 3) Expected fault frequencies at 1797 rpm
print(fault_frequencies(CWRU_DE_6205, shaft_hz=1797 / 60))
# {'BPFO': 107.4, 'BPFI': 162.2, 'BSF': 70.6, 'FTF': 11.9}

# 4) Envelope spectrum around a resonance band
f, a = envelope_spectrum(x, fs, band=(2_000, 4_000))
```

### Automatic band selection (spectral kurtosis)

Envelope analysis works best when the signal is first filtered around the
resonance that the fault impacts excite. `select_band` finds that band
automatically with a simplified kurtogram: spectral kurtosis is computed at
several frequency resolutions and the most statistically significant band wins.

```python
from vibkit import select_band, envelope_spectrum

res = select_band(x, fs)
print(res["band"])            # e.g. (2625.0, 3375.0) Hz
f, a = envelope_spectrum(x, fs, band=res["band"])
```

On the simulated outer-race fault the selected band finds BPFO in 20/20 runs at
0 dB and −5 dB SNR, and in 14/20 runs at −10 dB, where the fault is deeply buried in noise.

### Loading CWRU data

Download the `.mat` files from the
[Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
into a local `data/` folder (not tracked by git), then:

```python
from vibkit.datasets import build_dataset

X, y = build_dataset(
    {"data/97.mat": 0,    # normal
     "data/105.mat": 1,   # inner race, 0.007"
     "data/118.mat": 2,   # ball, 0.007"
     "data/130.mat": 3},  # outer race, 0.007"
    length=1024, stride=512,
)
```

## Demo

```bash
python examples/demo_outer_race_fault.py
```

Simulates periodic impacts from an outer-race defect exciting a 3 kHz resonance,
buried in noise. Spectral kurtosis selects the band around the resonance, and the
envelope spectrum shows clear peaks at BPFO (107.4 Hz) and its harmonics, although
the fault is hard to see in the raw signal and the plain spectrum.

## Tests

```bash
pytest
```

Tests check the spectra against known sine amplitudes, the features against analytical
values, and the fault frequencies against the published CWRU values.

## Changelog

- **0.2.0**: spectral kurtosis and automatic band selection (`spectral_kurtosis`, `select_band`); frequency-domain features (`frequency_features`); the demo now selects its band automatically.
- **0.1.0**: first release: segmentation, time-domain features, spectra, envelope analysis, bearing fault frequencies, CWRU loader.

## License

MIT — see [LICENSE](LICENSE).

## Author

**Zakariya Arroussi** — PhD student, Mechanical Engineering, East China University of Science and Technology (ECUST).
