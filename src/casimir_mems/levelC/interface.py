# FILE: src/casimir_mems/levelC/interface.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class LevelCResult:
    """
    Minimal Level C return type for the canonical C0 benchmark.

    eta_levelC: normalized observable η_C(d) (dimensionless), same shape as d
    n_modes:    integer mode count used
    converged:  per-point convergence flags (True if Δ < tol when increasing modes)
    """
    eta_levelC: np.ndarray
    n_modes: np.ndarray
    converged: np.ndarray


class LevelCBackend(Protocol):
    """
    Backend protocol for C0 sinusoid benchmark. This is intentionally narrow.

    Implementations must return η_C(d) plus convergence bookkeeping.
    """
    backend_id: str  # REQUIRED: stable identifier for CSV + plots
    def compute_sinusoid(
        self,
        d: np.ndarray,
        *,
        a: float,
        period: float,
        n_modes: int,
        tol: float,
    ) -> LevelCResult: ...


def compute_eta_levelC_sinusoid(
    d: np.ndarray,
    *,
    a: float,
    period: float,
    n_modes: int,
    tol: float,
    backend: LevelCBackend,
) -> LevelCResult:
    """
    Canonical Level C interface (C0 benchmark):

      Inputs:
        - d: separations (m)
        - a: sinusoid amplitude (m)
        - period: sinusoid period Λ (m)
        - n_modes: lateral Fourier modes used by Level C
        - tol: convergence tolerance for declaring converged
        - backend: Level C implementation

      Output:
        - LevelCResult containing η_C(d), n_modes per point, converged flags per point
    """
    d = np.asarray(d, dtype=float)
    if d.ndim != 1 or d.size < 2:
        raise ValueError("d must be a 1D array with at least 2 points.")
    if np.any(d <= 0.0):
        raise ValueError("All separations must be > 0.")
    if not (a > 0.0 and period > 0.0):
        raise ValueError("a and period must be > 0.")
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")

    return backend.compute_sinusoid(d, a=a, period=period, n_modes=n_modes, tol=tol)