"""
casimir_mems.levelB.derivative_expansion

Level B (B0): Ideal-conductor electromagnetic Derivative Expansion (DE)
correction for a sinusoidal corrugation.

Implements:
- BETA_EM_IDEAL (Bimonte et al., EPL 97 (2012) 50001) for perfect conductors:
    beta_EM = (2/3) * (1 - 15/pi^2)
- eta_levelB_DE_ideal(d): normalized plane-plane pressure ratio for
  one corrugated plate vs flat baseline:
    η_B(d) = < P(H) [1 + β (∇H)^2] > / P(d)

This is intentionally geometry-only and idealized; it is used as a diagnostic bridge
to highlight where material realism (Level C) matters.
"""

from __future__ import annotations

import math
import numpy as np

from casimir_mems.levelA.plane_plane import P_pp_ideal

# Bimonte et al. (EPL 97 (2012) 50001): EM perfect conductors
BETA_EM_IDEAL: float = (2.0 / 3.0) * (1.0 - 15.0 / (math.pi**2))


def eta_levelB_DE_ideal(
    d: np.ndarray,
    *,
    a: float,
    period: float,
    samples: int = 4096,
    beta: float = BETA_EM_IDEAL,
) -> np.ndarray:
    """
    Level B (B0): ideal-metal EM DE correction for one sinusoidally corrugated surface vs flat.

    Parameters
    ----------
    d : array-like
        Separations (m).
    a : float
        Sinusoid amplitude (m).
    period : float
        Sinusoid period (m).
    samples : int
        Number of x-samples over one corrugation period for averaging.
    beta : float
        DE coefficient (defaults to BETA_EM_IDEAL).

    Returns
    -------
    np.ndarray
        η_B(d) dimensionless ratio.
    """
    d = np.asarray(d, dtype=float)
    if d.ndim != 1 or d.size < 2:
        raise ValueError("d must be a 1D array with >=2 points")
    if np.any(d <= 0.0):
        raise ValueError("All separations must be > 0")
    if not (period > 0.0):
        raise ValueError("period must be > 0")
    if not (a >= 0.0):
        raise ValueError("a must be >= 0")
    if samples < 256:
        raise ValueError("samples too low; use >= 256")

    k = 2.0 * math.pi / float(period)
    x = np.linspace(0.0, 2.0 * math.pi / k, int(samples), endpoint=False)
    sinx = np.sin(k * x)
    cosx = np.cos(k * x)

    grad2 = (float(a) * k * cosx) ** 2  # (∇H)^2 along x for 1D sinusoid

    P0 = P_pp_ideal(d)
    out = np.empty_like(d, dtype=float)

    for i, di in enumerate(d):
        H = float(di) + float(a) * sinx
        if np.any(H <= 0.0):
            raise ValueError(f"Non-positive local gap encountered at d={di:.3e} with amplitude a={a:.3e}")
        PH = P_pp_ideal(H)
        Pcorr = float(np.mean(PH * (1.0 + float(beta) * grad2)))
        out[i] = Pcorr / float(P0[i])

    return out