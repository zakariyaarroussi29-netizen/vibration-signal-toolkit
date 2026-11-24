"""Rolling-element bearing characteristic fault frequencies."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BearingGeometry:
    """Bearing geometry.

    n_balls : number of rolling elements
    ball_diameter, pitch_diameter : same unit (e.g. inches or mm)
    contact_angle_deg : contact angle in degrees (0 for deep-groove ball bearings)
    """
    n_balls: int
    ball_diameter: float
    pitch_diameter: float
    contact_angle_deg: float = 0.0


# Drive-end bearing of the CWRU test rig (SKF 6205-2RS JEM), dimensions in inches.
CWRU_DE_6205 = BearingGeometry(n_balls=9, ball_diameter=0.3126, pitch_diameter=1.537)


def fault_frequencies(geom, shaft_hz):
    """Return BPFO, BPFI, BSF and FTF in Hz for a shaft speed in Hz.

    Assumes a stationary outer race and rotating inner race. Convert rpm
    with ``shaft_hz = rpm / 60``.
    """
    r = geom.ball_diameter / geom.pitch_diameter * math.cos(
        math.radians(geom.contact_angle_deg))
    n = geom.n_balls
    return {
        "BPFO": n / 2 * shaft_hz * (1 - r),  # outer race
        "BPFI": n / 2 * shaft_hz * (1 + r),  # inner race
        "BSF": geom.pitch_diameter / (2 * geom.ball_diameter) * shaft_hz * (1 - r ** 2),  # ball spin
        "FTF": shaft_hz / 2 * (1 - r),  # cage
    }
