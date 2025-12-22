# src/casimir_mems/levelA/plane_plane.py

from __future__ import annotations

import numpy as np

from ..types import Array, MaterialModel

HBAR = 1.054_571_817e-34
C = 299_792_458.0
PI = np.pi


def E_pp_ideal(d: Array) -> Array:
    """
    Casimir energy per unit area for ideal parallel plates.

      E/A = - (pi^2 / 720) * (ħ c) / d^3

    d in meters, returns J/m^2.
    """
    d = np.asarray(d, dtype=float)
    if np.any(d <= 0.0):
        raise ValueError("All separations d must be > 0.")
    return -(PI**2 / 720.0) * (HBAR * C) / (d**3)


def P_pp_ideal(d: Array) -> Array:
    """
    Casimir pressure for ideal parallel plates.

      P = - d(E/A)/dd = - (pi^2 / 240) * (ħ c) / d^4

    Returns N/m^2 (Pa). Attraction is negative in this sign convention.
    """
    d = np.asarray(d, dtype=float)
    if np.any(d <= 0.0):
        raise ValueError("All separations d must be > 0.")
    return -(PI**2 / 240.0) * (HBAR * C) / (d**4)


def E_pp(d: Array, model: MaterialModel = "ideal", **kwargs) -> Array:
    """
    Dispatch: plane–plane baseline energy per area.

    Level A default is ideal only. Extra kwargs are accepted for forward
    compatibility with future material/environment extensions.
    """
    if model == "ideal":
        return E_pp_ideal(d)
    raise NotImplementedError("Level A: only model='ideal' is implemented.")


def P_pp(d: Array, model: MaterialModel = "ideal", **kwargs) -> Array:
    """
    Dispatch: plane–plane baseline pressure.

    Level A default is ideal only. Extra kwargs are accepted for forward
    compatibility with future material/environment extensions.
    """
    if model == "ideal":
        return P_pp_ideal(d)
    raise NotImplementedError("Level A: only model='ideal' is implemented.")