"""
casimir_mems.levelB.validity

Validity / refusal logic for Level B (Derivative Expansion).

Primary purpose:
- compute dimensionless diagnostics for a sinusoidal profile:
    kd  = (2π/period) * d
    ak  = (2π/period) * a
    ak2 = (ak)^2
- compute a smooth validity_score in [0, 1] for plotting/diagnostics
- compute warned/refused boolean masks based on kd thresholds

Important:
- refusal is a hard trust boundary: refused rows should be treated as outside
  Level B's domain (and downstream consumers may mask or downgrade them).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class LevelBValidity:
    kd: np.ndarray
    ak: np.ndarray
    ak2: np.ndarray
    validity_score: np.ndarray
    warned: np.ndarray
    refused: np.ndarray


def _sigmoid01(x: np.ndarray, *, x0: float, width: float) -> np.ndarray:
    """
    Smooth step from ~1 to ~0 centered around x0 with transition scale 'width'.

    Returns values in (0,1). Used to create a non-binary validity_score while
    keeping hard warned/refused masks separate.
    """
    w = max(float(width), 1e-12)
    z = (x - float(x0)) / w
    return 1.0 / (1.0 + np.exp(z))


def compute_validity_sinusoid(
    d: np.ndarray,
    *,
    period: float,
    amplitude: float,
    kd_warn: float,
    kd_refuse: float,
) -> LevelBValidity:
    """
    Compute B0 validity diagnostics for a 1D sinusoidal corrugation.

    Parameters
    ----------
    d : array-like
        Separations (m).
    period : float
        Corrugation period (m).
    amplitude : float
        Corrugation amplitude (m).
    kd_warn : float
        Warn threshold on kd (dimensionless).
    kd_refuse : float
        Refusal threshold on kd (dimensionless). Points with kd > kd_refuse
        are outside Level B by contract.

    Returns
    -------
    LevelBValidity
        kd, ak, ak2, validity_score, warned, refused
    """
    d = np.asarray(d, dtype=float)
    if d.ndim != 1 or d.size < 2:
        raise ValueError("d must be a 1D array with >=2 points")
    if np.any(d <= 0.0):
        raise ValueError("All separations must be > 0")
    if not (period > 0.0):
        raise ValueError("period must be > 0")
    if not (amplitude >= 0.0):
        raise ValueError("amplitude must be >= 0")

    k = 2.0 * math.pi / float(period)
    kd = k * d
    ak = np.full_like(d, k * float(amplitude), dtype=float)
    ak2 = ak * ak

    warned = kd > float(kd_warn)
    refused = kd > float(kd_refuse)

    # Smooth score: 1 well inside domain; decays as kd exceeds warn; near-zero by refuse.
    # Use warn as center, and transition width derived from (refuse-warn) if available.
    if math.isfinite(float(kd_refuse)) and float(kd_refuse) > float(kd_warn):
        width = 0.25 * (float(kd_refuse) - float(kd_warn))
    else:
        width = 0.25 * max(float(kd_warn), 1.0)

    score = _sigmoid01(kd, x0=float(kd_warn), width=width)

    # Hard clip refused points to score=0 to respect trust boundary.
    score = np.asarray(score, dtype=float)
    score[refused] = 0.0

    return LevelBValidity(
        kd=kd,
        ak=ak,
        ak2=ak2,
        validity_score=score,
        warned=warned,
        refused=refused,
    )