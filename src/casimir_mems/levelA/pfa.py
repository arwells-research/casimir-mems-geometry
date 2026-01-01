"""
casimir_mems.levelA.pfa

Sphere–plane Proximity Force Approximation (PFA) baselines.

Uses the ideal plane–plane energy/pressure as primitives.

PFA relationships:
  F(d)      ≈ 2π R * E_pp(d)
  dF/dd(d)  ≈ 2π R * P_pp(d)

Sign conventions:
- E_pp_ideal(d) < 0, P_pp_ideal(d) < 0 for attraction
- Therefore F(d) < 0 and dF/dd(d) < 0
"""

from __future__ import annotations

import numpy as np
from ..types import Array, Sphere, Calibration
from .plane_plane import E_pp_ideal, P_pp_ideal


def _effective_d(d: Array, calib: Calibration | None) -> Array:
    d_arr = np.asarray(d, dtype=float)
    d0 = 0.0 if calib is None else float(calib.d0)
    d_eff = d_arr + d0
    if np.any(d_eff <= 0.0):
        raise ValueError("Effective separation d + d0 must be > 0 everywhere.")
    return d_eff


def F_sphere_plane_pfa(d: Array, sphere: Sphere, calib: Calibration | None = None) -> Array:
    """
    Sphere–plane force under PFA:
      F(d) ≈ 2π R * E_pp_ideal(d)

    Parameters
    ----------
    d : array-like
        Separation(s) in meters.
    sphere : Sphere
        Sphere(R) with R in meters.
    calib : Calibration, optional
        Separation offset (d_eff = d + d0).

    Returns
    -------
    np.ndarray
        Force in Newtons (negative for attraction).
    """
    if sphere.R <= 0.0:
        raise ValueError("Sphere radius R must be > 0.")
    d_eff = _effective_d(d, calib)
    return 2.0 * np.pi * float(sphere.R) * E_pp_ideal(d_eff)


def dF_dd_sphere_plane_pfa(d: Array, sphere: Sphere, calib: Calibration | None = None) -> Array:
    """
    Sphere–plane force gradient under PFA:
      dF/dd(d) ≈ 2π R * P_pp_ideal(d)

    Returns
    -------
    np.ndarray
        Force gradient in N/m (negative for attraction).
    """
    if sphere.R <= 0.0:
        raise ValueError("Sphere radius R must be > 0.")
    d_eff = _effective_d(d, calib)
    return 2.0 * np.pi * float(sphere.R) * P_pp_ideal(d_eff)
    