"""vibkit: small, dependency-light tools for vibration-based machine diagnosis."""

from .segmentation import sliding_windows, normalize
from .features import (time_features, FEATURE_NAMES,
                       frequency_features, FREQ_FEATURE_NAMES)
from .spectral import (amplitude_spectrum, envelope_spectrum, spectrogram, bandpass,
                       spectral_kurtosis, select_band)
from .bearing import BearingGeometry, fault_frequencies

__version__ = "0.2.0"

__all__ = [
    "sliding_windows",
    "normalize",
    "time_features",
    "FEATURE_NAMES",
    "frequency_features",
    "FREQ_FEATURE_NAMES",
    "amplitude_spectrum",
    "envelope_spectrum",
    "spectrogram",
    "bandpass",
    "spectral_kurtosis",
    "select_band",
    "BearingGeometry",
    "fault_frequencies",
]
