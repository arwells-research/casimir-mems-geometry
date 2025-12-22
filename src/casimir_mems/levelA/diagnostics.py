"""
casimir_mems.levelA.diagnostics

Engineer-friendly diagnostics for comparisons:
- deviation factor η = F / F_ref
- percent difference
- local log-slope (d log|y| / d log x)
"""

from __future__ import annotations

import numpy as np
from ..types import Array


def deviation_factor(F_geom: Array, F_ref: Array) -> Array:
    """η = F_geom / F_ref (elementwise)."""
    Fg = np.asarray(F_geom, dtype=float)
    Fr = np.asarray(F_ref, dtype=float)
    return Fg / Fr


def percent_difference(F_geom: Array, F_ref: Array) -> Array:
    """100 * (F_geom - F_ref) / F_ref (elementwise)."""
    Fg = np.asarray(F_geom, dtype=float)
    Fr = np.asarray(F_ref, dtype=float)
    return 100.0 * (Fg - Fr) / Fr


def log_slope(x: Array, y: Array) -> Array:
    """
    Local log slope d log|y| / d log x using numpy gradient.

    Notes
    -----
    - Uses abs(y) so attractive forces/gradients are handled naturally.
    - Requires x > 0 and y != 0 (zeros will produce -inf).
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if np.any(x_arr <= 0.0):
        raise ValueError("x must be > 0 for log_slope.")
    if np.any(y_arr == 0.0):
        raise ValueError("y must be nonzero everywhere for log_slope.")

    lx = np.log(x_arr)
    ly = np.log(np.abs(y_arr))
    return np.gradient(ly, lx)
