"""
casimir_mems.levelA.interface

Single entry point for experiments:
  sphere_target_curve(d, sphere, target, quantity="force_gradient")

Targets:
- Plane()
- RectTrenchGrating(p, w, h)

Quantities (Level A):
- "force"
- "force_gradient"
"""

from __future__ import annotations

from ..types import Array, Sphere, Plane, RectTrenchGrating, Calibration, Quantity
from .pfa import F_sphere_plane_pfa, dF_dd_sphere_plane_pfa
from .grating_pfa_mix import (
    F_sphere_trench_pfa_mix,
    dF_dd_sphere_trench_pfa_mix,
    dF_dd_sphere_trench_pfa_bao,   # add
)


def sphere_target_curve(
    d: Array,
    sphere: Sphere,
    target: object,
    quantity: Quantity = "force_gradient",
    *,
    model: str = "ideal",
    calib: Calibration | None = None,
    method: str = "pfa_mix",   # add
    **kwargs
) -> Array:
    """
    Compute a Level A curve for a sphere interacting with a target geometry.

    Parameters
    ----------
    d : array-like
        Separation(s) in meters.
    sphere : Sphere
        Sphere geometry.
    target : Plane or RectTrenchGrating
        Target geometry.
    quantity : {"force", "force_gradient"}
        Quantity to return.
    calib : Calibration, optional
        Separation offset d0 (d_eff = d + d0).

    Returns
    -------
    np.ndarray
        Curve values (N or N/m).
    """
    if isinstance(target, Plane):
        if quantity == "force":
            return F_sphere_plane_pfa(d, sphere, calib=calib)
        if quantity == "force_gradient":
            return dF_dd_sphere_plane_pfa(d, sphere, calib=calib)
        raise ValueError(f"Unsupported quantity: {quantity}")

    if target.__class__.__name__ == "RectTrenchGrating":
        g = target  # type: ignore
        if quantity == "force":
            if method != "pfa_mix":
                raise ValueError("Only method='pfa_mix' supports quantity='force' right now.")
            return F_sphere_trench_pfa_mix(d, sphere, g, model=model, calib=calib, **kwargs)

        if quantity == "force_gradient":
            if method == "pfa_mix":
                return dF_dd_sphere_trench_pfa_mix(d, sphere, g, model=model, calib=calib, **kwargs)
            if method == "pfa_bao":
                return dF_dd_sphere_trench_pfa_bao(d, sphere, g, model=model, calib=calib, **kwargs)
            raise ValueError(f"Unsupported trench method: {method!r}")
            
    raise TypeError(f"Unsupported target type: {type(target)}")