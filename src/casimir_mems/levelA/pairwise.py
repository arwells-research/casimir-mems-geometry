"""
casimir_mems.levelA.pairwise

Optional additive "foil" baselines (Hamaker-like).
Not part of the locked Level A single-entry-point API.

These are intentionally labeled as non-retarded / Hamaker-like approximations
and are included only for engineering contrasts (additivity vs non-additivity).
"""

from __future__ import annotations

import numpy as np
from ..types import Array, Sphere, RectTrenchGrating, Calibration


def _effective_d(d: Array, calib: Calibration | None) -> Array:
    d_arr = np.asarray(d, dtype=float)
    d0 = 0.0 if calib is None else float(calib.d0)
    d_eff = d_arr + d0
    if np.any(d_eff <= 0.0):
        raise ValueError("Effective separation d + d0 must be > 0 everywhere.")
    return d_eff


def F_sphere_plane_hamaker(d: Array, sphere: Sphere, A: float, calib: Calibration | None = None) -> Array:
    """
    Simple non-retarded Hamaker-like sphere–plane force:
      F ≈ - A R / (6 d^2)

    A is Hamaker constant (J).
    """
    if sphere.R <= 0.0:
        raise ValueError("Sphere radius R must be > 0.")
    if A <= 0.0:
        raise ValueError("Hamaker constant A must be > 0.")
    d_eff = _effective_d(d, calib)
    return -(float(A) * float(sphere.R)) / (6.0 * d_eff**2)


def F_sphere_trench_hamaker_mix(
    d: Array,
    sphere: Sphere,
    grating: RectTrenchGrating,
    A: float,
    calib: Calibration | None = None,
) -> Array:
    """
    Hamaker-like area mixing baseline for trench array:
      F_eff = (1-f)F(d) + fF(d+h)
    where f = w/p is trench fraction.
    """
    if sphere.R <= 0.0:
        raise ValueError("Sphere radius R must be > 0.")
    if A <= 0.0:
        raise ValueError("Hamaker constant A must be > 0.")
    p = float(grating.p)
    w = float(grating.w)
    h = float(grating.h)
    if p <= 0.0:
        raise ValueError("Grating period p must be > 0.")
    if w <= 0.0 or w >= p:
        raise ValueError("Trench width w must satisfy 0 < w < p.")
    if h < 0.0:
        raise ValueError("Trench depth h must be >= 0.")
    f = w / p

    d_eff = _effective_d(d, calib)
    F_top = -(float(A) * float(sphere.R)) / (6.0 * d_eff**2)
    F_bot = -(float(A) * float(sphere.R)) / (6.0 * (d_eff + h) ** 2)
    return (1.0 - f) * F_top + f * F_bot