"""
casimir_mems.levelB.sinusoid

Convenience wrapper for the Level B (B0) sinusoid backend:
- computes diagnostics (kd/ak/score/warn/refuse)
- computes η_B(d) via ideal DE
- returns a small, explicit bundle for experiment harnesses

This is intentionally minimal: the canonical math lives in
- validity.py (diagnostics)
- derivative_expansion.py (η_B)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import LEVELB_BACKEND_ID
from .validity import compute_validity_sinusoid, LevelBValidity
from .derivative_expansion import eta_levelB_DE_ideal


@dataclass(frozen=True)
class LevelBSinusoidResult:
    backend_id: str
    eta_levelB: np.ndarray
    validity: LevelBValidity


def compute_levelB_sinusoid_b0(
    d: np.ndarray,
    *,
    amplitude_m: float,
    period_m: float,
    kd_warn: float,
    kd_refuse: float,
    samples: int = 4096,
) -> LevelBSinusoidResult:
    """
    Compute Level B (B0) for a sinusoid in one call.

    Returns:
      backend_id
      eta_levelB
      validity (kd/ak/ak2/score + warned/refused masks)
    """
    validity = compute_validity_sinusoid(
        d,
        period=period_m,
        amplitude=amplitude_m,
        kd_warn=kd_warn,
        kd_refuse=kd_refuse,
    )
    etaB = eta_levelB_DE_ideal(
        d,
        a=amplitude_m,
        period=period_m,
        samples=samples,
    )
    return LevelBSinusoidResult(
        backend_id=LEVELB_BACKEND_ID,
        eta_levelB=etaB,
        validity=validity,
    )