# src/casimir_mems/types.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

Array = np.ndarray

MaterialModel = Literal["ideal"]  # Level A: only ideal implemented initially
Quantity = Literal["energy", "pressure", "force", "force_gradient"]


@dataclass(frozen=True)
class Sphere:
    """Sphere geometry for PFA: radius R (meters)."""
    R: float


@dataclass(frozen=True)
class Plane:
    """Planar half-space target."""
    pass


@dataclass(frozen=True)
class Calibration:
    """
    Minimal experimental calibration knobs.
    d0: separation offset (meters). If provided, use d_eff = d + d0.
    """
    d0: float = 0.0


@dataclass(frozen=True)
class RectTrenchGrating:
    """
    1D lamellar grating / rectangular trench array.

    Required parameters (meters):
      p: period
      w: legacy width (kept for backward compatibility; in our usage this is l2 by default)
      h: trench depth

    Optional parameters (meters), for Bao-style PFA with sidewalls:
      top_width_m: l1 (ridge/top surface length)
      bottom_width_m: l2 (trench bottom length)

    Defaults:
      - bottom_width_m defaults to w
      - top_width_m defaults to p - bottom_width_m

    Notes:
      - For physical consistency, require p > 0, h >= 0, and widths in [0, p].
      - Sidewall remainder length is p3_total = p - l1 - l2 (must be >= 0).
    """
    p: float
    w: float
    h: float
    top_width_m: Optional[float] = None
    bottom_width_m: Optional[float] = None

    def l2(self) -> float:
        """Bottom length l2 (meters)."""
        return float(self.bottom_width_m if self.bottom_width_m is not None else self.w)

    def l1(self) -> float:
        """Top length l1 (meters)."""
        if self.top_width_m is not None:
            return float(self.top_width_m)
        return float(self.p - self.l2())

    def duty_cycle_bottom(self) -> float:
        """Return bottom duty cycle f2 = l2/p."""
        if self.p == 0.0:
            return 0.0
        return float(self.l2() / self.p)

    def sidewall_remainder(self) -> float:
        """
        Total remaining length assigned to sidewalls:
          p3_total = p - l1 - l2
        Must be >= 0 for a physical lamellar cell.
        """
        return float(self.p - self.l1() - self.l2())

    def validate(self) -> None:
        """Raise ValueError if geometry parameters are invalid."""
        if not (self.p > 0.0):
            raise ValueError("Grating period p must be > 0.")
        if not (self.h >= 0.0):
            raise ValueError("Grating depth h must be >= 0.")

        l1 = self.l1()
        l2 = self.l2()

        if l1 < 0.0 or l2 < 0.0:
            raise ValueError("Grating widths must be >= 0.")
        if l1 > self.p or l2 > self.p:
            raise ValueError("Grating widths must be <= period p.")

        rem = self.sidewall_remainder()
        if rem < -1e-18:
            raise ValueError(
                f"Invalid grating geometry: l1+l2 exceeds p (p={self.p:.3e}, l1={l1:.3e}, l2={l2:.3e})."
            )