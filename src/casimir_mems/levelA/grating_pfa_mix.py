# src/casimir_mems/levelA/grating_pfa_mix.py
"""
Level A geometry baselines for sphere vs 1D rectangular trench / lamellar gratings.

This module contains two related "fast baseline" models:

1) PFA area-mixing (top + bottom only)
   - treats the target as two local planes:
       * top surface at separation d
       * trench bottom at separation d + h
   - weights by duty cycle f = w/p (fraction of area at trench bottom)
   - transparent screening baseline; matches common "mix" denominators used in papers

2) Bao-style PFA with sidewalls (A+ baseline)
   - implements a lamellar-profile PFA with explicit sidewall contribution
   - uses lengths l1 (top/ridge), l2 (bottom/trench), and sidewall remainder
   - analytic sidewall integral for model="ideal"; deterministic NumPy fallback otherwise

All functions are pure/deterministic and NumPy-only.
"""

from __future__ import annotations

import numpy as np

from ..types import Array, Sphere, RectTrenchGrating, Calibration
from .plane_plane import E_pp, P_pp


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def duty_cycle(g: RectTrenchGrating) -> float:
    """
    Return f = w/p, interpreted as the fraction of area at the trench bottom.

    NOTE:
    - This matches the original "mix" convention used in this repo's Level A.
    - For Bao-style PFA we use explicit l1/l2 fields instead (see bao_lengths()).
    """
    if g.p <= 0.0:
        raise ValueError("Grating period p must be > 0.")
    return float(g.w / g.p)


def bao_lengths(g: RectTrenchGrating) -> tuple[float, float, float]:
    """
    Return (l1, l2, p3_len) for Bao-style PFA.

    l1: top/ridge length per period (meters)
    l2: bottom/trench length per period (meters)
    p3_len: projected sidewall length on ONE side (meters)

    If RectTrenchGrating carries optional widths:
      - g.top_width_m -> l1
      - g.bottom_width_m -> l2
    Otherwise (backward compatible fallback):
      - l2 = g.w
      - l1 = g.p - l2

    Geometry sanity checks are performed:
      - p > 0
      - l1,l2 >= 0
      - l1 + l2 <= p (within tiny numeric slack)
      - p3_len >= 0
    """
    if g.p <= 0.0:
        raise ValueError("Grating period p must be > 0.")

    l2 = float(g.bottom_width_m if getattr(g, "bottom_width_m", None) is not None else g.w)
    l1 = float(g.top_width_m if getattr(g, "top_width_m", None) is not None else (g.p - l2))

    if l1 < 0.0 or l2 < 0.0:
        raise ValueError("Grating widths must be >= 0.")

    if (l1 + l2) > g.p * (1.0 + 1e-12):
        raise ValueError(f"Invalid lengths: l1+l2={l1 + l2:.6e} exceeds p={g.p:.6e}")

    p3_len = 0.5 * (g.p - l1 - l2)  # meters
    if p3_len < -1e-15:
        raise ValueError("Computed sidewall length p3 is negative.")
    if p3_len < 0.0:
        p3_len = 0.0  # numerical slack

    return l1, l2, p3_len


# -----------------------------------------------------------------------------
# PFA area-mixing: plane-plane effective energy/pressure
# -----------------------------------------------------------------------------

def E_pp_trench_mix(d: Array, g: RectTrenchGrating, *, model: str = "ideal", **kwargs) -> Array:
    """
    Effective plane-plane energy per area for a lamellar trench array (PFA mixing):

      E_eff(d) = (1 - f) * E_pp(d) + f * E_pp(d + h)

    where:
      f = w/p is interpreted as the fraction of area at the trench bottom.

    Parameters:
      d: separations (meters)
      g: RectTrenchGrating(p,w,h)
      model: plate model passed to plane_plane.E_pp
    """
    d = np.asarray(d, dtype=float)
    if np.any(d <= 0.0):
        raise ValueError("All separations must be > 0.")
    if g.h < 0.0:
        raise ValueError("Trench depth h must be >= 0.")

    f = duty_cycle(g)
    return (1.0 - f) * E_pp(d, model=model, **kwargs) + f * E_pp(d + g.h, model=model, **kwargs)


def P_pp_trench_mix(d: Array, g: RectTrenchGrating, *, model: str = "ideal", **kwargs) -> Array:
    """
    Effective plane-plane pressure for a lamellar trench array (PFA mixing):

      P_eff(d) = (1 - f) * P_pp(d) + f * P_pp(d + h)

    where:
      f = w/p is interpreted as the fraction of area at the trench bottom.
    """
    d = np.asarray(d, dtype=float)
    if np.any(d <= 0.0):
        raise ValueError("All separations must be > 0.")
    if g.h < 0.0:
        raise ValueError("Trench depth h must be >= 0.")

    f = duty_cycle(g)
    return (1.0 - f) * P_pp(d, model=model, **kwargs) + f * P_pp(d + g.h, model=model, **kwargs)


# -----------------------------------------------------------------------------
# Bao-style PFA with sidewalls: plane-plane effective pressure
# -----------------------------------------------------------------------------

def P_pp_trench_pfa_bao(d: Array, g: RectTrenchGrating, *, model: str = "ideal", **kwargs) -> Array:
    """
    Bao-style PFA effective pressure for a lamellar trench profile including sidewalls.

    Using per-period lengths:
      - l1: top/ridge length
      - l2: bottom/trench length
      - p3_len: sidewall projected length on one side
      - p: period

    Effective pressure:
      P_eff(d) = (l1/p) P(d) + (l2/p) P(d+h) + (2/p) ∫_{0}^{p3_len} P(d + h x / p3_len) dx

    Notes:
      - For model="ideal", the sidewall integral is evaluated analytically for speed and determinism.
      - For any non-ideal model added later, we fall back to a fixed-grid average along the wall.
    """
    d = np.asarray(d, dtype=float)
    if np.any(d <= 0.0):
        raise ValueError("All separations must be > 0.")
    if g.h < 0.0:
        raise ValueError("Trench depth h must be >= 0.")

    l1, l2, p3_len = bao_lengths(g)
    p = float(g.p)

    # top + bottom contributions
    P_top = P_pp(d, model=model, **kwargs)
    P_bot = P_pp(d + g.h, model=model, **kwargs)
    out = (l1 / p) * P_top + (l2 / p) * P_bot

    # sidewalls vanish if no wall length or no depth
    if p3_len <= 0.0 or g.h == 0.0:
        return out

    if model != "ideal":
        # Deterministic fixed-grid "average along wall" fallback:
        # ∫_0^{p3} P(d + h x/p3) dx = p3 * ∫_0^1 P(d + h u) du ≈ p3 * mean_u P(d + h u)
        u = np.linspace(0.0, 1.0, 33)  # fixed for reproducibility
        z = d[:, None] + g.h * u[None, :]
        P_wall = P_pp(z, model=model, **kwargs)  # broadcast: (N, 33)
        avg = np.mean(P_wall, axis=1)
        out += (2.0 / p) * (p3_len * avg)
        return out

    # Analytic sidewall integral for ideal plates:
    #
    # P(d) = -K / d^4
    # ∫_0^1 P(d + h u) du = -K * (1/(3h)) * (d^{-3} - (d+h)^{-3})
    #
    # We compute K pointwise from P(d): K = -P(d) * d^4
    Pd = P_pp(d, model="ideal")
    K = -Pd * (d ** 4)

    wall_avg_integral = -K * (1.0 / (3.0 * g.h)) * ((d ** -3) - ((d + g.h) ** -3))
    out += (2.0 / p) * (p3_len * wall_avg_integral)
    return out


# -----------------------------------------------------------------------------
# Sphere wrappers (PFA): force and force gradient
# -----------------------------------------------------------------------------

def F_sphere_trench_pfa_mix(
    d: Array,
    sphere: Sphere,
    g: RectTrenchGrating,
    *,
    model: str = "ideal",
    calib: Calibration | None = None,
    **kwargs
) -> Array:
    """
    Sphere vs trench-array force under PFA mixing:
      F(d) ≈ 2πR * E_eff(d)
    """
    if sphere.R <= 0.0:
        raise ValueError("Sphere radius R must be > 0.")
    d_eff = np.asarray(d, dtype=float) + (calib.d0 if calib else 0.0)
    return 2.0 * np.pi * sphere.R * E_pp_trench_mix(d_eff, g, model=model, **kwargs)


def dF_dd_sphere_trench_pfa_mix(
    d: Array,
    sphere: Sphere,
    g: RectTrenchGrating,
    *,
    model: str = "ideal",
    calib: Calibration | None = None,
    **kwargs
) -> Array:
    """
    Sphere vs trench-array force gradient under PFA mixing:
      dF/dd ≈ 2πR * P_eff(d)
    """
    if sphere.R <= 0.0:
        raise ValueError("Sphere radius R must be > 0.")
    d_eff = np.asarray(d, dtype=float) + (calib.d0 if calib else 0.0)
    return 2.0 * np.pi * sphere.R * P_pp_trench_mix(d_eff, g, model=model, **kwargs)


def dF_dd_sphere_trench_pfa_bao(
    d: Array,
    sphere: Sphere,
    g: RectTrenchGrating,
    *,
    model: str = "ideal",
    calib: Calibration | None = None,
    **kwargs
) -> Array:
    """
    Sphere vs trench-array force gradient under Bao-style PFA (includes sidewalls):
      dF/dd ≈ 2πR * P_eff_bao(d)
    """
    if sphere.R <= 0.0:
        raise ValueError("Sphere radius R must be > 0.")
    d_eff = np.asarray(d, dtype=float) + (calib.d0 if calib else 0.0)
    P_eff = P_pp_trench_pfa_bao(d_eff, g, model=model, **kwargs)
    return 2.0 * np.pi * sphere.R * P_eff