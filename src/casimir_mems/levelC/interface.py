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
    Backend protocol for Level C. Intentionally narrow, but extensible.

    The canonical C0 geometry is sinusoid vs flat via compute_sinusoid().
    Additional geometry entrypoints (e.g. dualharm) may be implemented by
    selected backends; the harness must only call them when required by the case.
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

    def compute_dualharm(
        self,
        d: np.ndarray,
        *,
        a1: float,
        a2: float,
        period: float,
        phi2: float,
        n_modes: int,
        tol: float,
    ) -> LevelCResult:
        """
        Optional geometry entrypoint: dual-harmonic corrugation vs flat.

          H(x) = d + a1 cos(kx) + a2 cos(2kx + phi2)

        Backends that do not implement this MUST raise NotImplementedError.
        """
        raise NotImplementedError("compute_dualharm not implemented by this Level C backend")


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


def compute_eta_levelC_dualharm(
    d: np.ndarray,
    *,
    a1: float,
    a2: float,
    period: float,
    phi2: float,
    n_modes: int,
    tol: float,
    backend: LevelCBackend,
) -> LevelCResult:
    """
    Canonical Level C interface for dual-harmonic geometry (synthetic cases C12+).

      Inputs:
        - d: separations (m)
        - a1: fundamental harmonic amplitude (m)
        - a2: second harmonic amplitude (m)
        - period: fundamental period Λ (m) (second harmonic is Λ/2)
        - phi2: phase of the second harmonic (radians)
        - n_modes: lateral Fourier modes used by Level C
        - tol: convergence tolerance for declaring converged
        - backend: Level C implementation (must implement compute_dualharm)

      Output:
        - LevelCResult containing η_C(d), n_modes per point, converged flags per point
    """
    d = np.asarray(d, dtype=float)
    if d.ndim != 1 or d.size < 2:
        raise ValueError("d must be a 1D array with at least 2 points.")
    if np.any(d <= 0.0):
        raise ValueError("All separations must be > 0.")
    if not (a1 > 0.0 and period > 0.0):
        raise ValueError("a1 and period must be > 0.")
    if a2 < 0.0:
        raise ValueError("a2 must be >= 0.")
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")

    return backend.compute_dualharm(d, a1=a1, a2=a2, period=period, phi2=phi2, n_modes=n_modes, tol=tol)