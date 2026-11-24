"""vibkit: small, dependency-light tools for vibration-based machine diagnosis."""

from .segmentation import sliding_windows, normalize
from .features import time_features, FEATURE_NAMES
from .spectral import amplitude_spectrum, envelope_spectrum, spectrogram, bandpass
from .bearing import BearingGeometry, fault_frequencies

__version__ = "0.1.0"

__all__ = [
    "sliding_windows",
    "normalize",
    "time_features",
    "FEATURE_NAMES",
    "amplitude_spectrum",
    "envelope_spectrum",
    "spectrogram",
    "bandpass",
    "BearingGeometry",
    "fault_frequencies",
]
