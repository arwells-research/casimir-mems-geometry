# FILE: src/casimir_mems/levelC/convergence.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Any

from .interface import LevelCBackend, LevelCResult

def __init__(self, *, core: Any, sweep: ModeSweepConfig):
    self.core = core
    self.sweep = sweep
    
@dataclass(frozen=True)
class ModeSweepConfig:
    n_modes_start: int = 8
    n_modes_step: int = 8
    n_modes_max: int = 128
    tol: float = 1.0e-4


class ModeSweepBackend(LevelCBackend):
    """
    Convergence-enforcing wrapper.

    Calls the underlying backend multiple times with increasing n_modes and
    marks a point converged when |η(n) - η(n-Δ)| < tol.

    Supports:
      - compute_sinusoid (C0 canonical)
      - compute_dualharm (optional; requires core backend support)
    """
    backend_id = "mode_sweep"

    def __init__(self, *, core: LevelCBackend, sweep: ModeSweepConfig):
        self.core = core
        self.sweep = sweep

    def compute_sinusoid(
        self,
        d: np.ndarray,
        *,
        a: float,
        period: float,
        n_modes: int,  # ignored by sweep backend
        tol: float,    # ignored by sweep backend (uses self.sweep.tol)
        refused: Optional[np.ndarray] = None,
    ) -> LevelCResult:
        d = np.asarray(d, dtype=float)
        cfg = self.sweep

        if refused is None:
            refused_mask = np.zeros(d.shape, dtype=bool)
        else:
            refused_mask = np.asarray(refused, dtype=bool)
            if refused_mask.shape != d.shape:
                raise ValueError("refused mask must have same shape as d")

        if cfg.n_modes_start < 1:
            raise ValueError("ModeSweepConfig.n_modes_start must be >= 1")
        if cfg.n_modes_step < 1:
            raise ValueError("ModeSweepConfig.n_modes_step must be >= 1")
        if cfg.n_modes_max < cfg.n_modes_start:
            raise ValueError("ModeSweepConfig.n_modes_max must be >= n_modes_start")
        if cfg.tol <= 0:
            raise ValueError("ModeSweepConfig.tol must be > 0")

        # First evaluation at start modes
        res_prev = self.core.compute_sinusoid(d, a=a, period=period, n_modes=cfg.n_modes_start, tol=cfg.tol)
        eta_prev = np.asarray(res_prev.eta_levelC, dtype=float)

        converged = np.zeros(d.shape, dtype=bool)
        n_modes_at = np.full(d.shape, int(cfg.n_modes_start), dtype=int)

        # Trust boundary: refused points are never considered converged.
        converged[refused_mask] = False
        n_modes_at[refused_mask] = 0

        eta_last = eta_prev
        last_nm = int(cfg.n_modes_start)

        # Sweep upward
        for nm in range(cfg.n_modes_start + cfg.n_modes_step, cfg.n_modes_max + 1, cfg.n_modes_step):
            res = self.core.compute_sinusoid(d, a=a, period=period, n_modes=nm, tol=cfg.tol)
            eta = np.asarray(res.eta_levelC, dtype=float)

            delta = np.abs(eta - eta_prev)
            newly = (~converged) & (~refused_mask) & (delta < cfg.tol)

            n_modes_at[newly] = int(nm)
            converged |= newly

            eta_prev = eta
            eta_last = eta
            last_nm = int(nm)

            if np.all(converged | refused_mask):
                break

        # If not converged, record the max effort used (so n_modes shows the sweep happened)
        n_modes_at[(~converged) & (~refused_mask)] = last_nm

        return LevelCResult(
            eta_levelC=eta_last,
            n_modes=n_modes_at,
            converged=converged,
        )

    def compute_dualharm(
        self,
        d: np.ndarray,
        *,
        a1: float,
        a2: float,
        period: float,
        phi2: float,
        n_modes: int,  # ignored by sweep backend
        tol: float,    # ignored by sweep backend (uses self.sweep.tol)
        refused: Optional[np.ndarray] = None,
    ) -> LevelCResult:
        """
        Sweep wrapper for dual-harmonic profile.

        Requires that self.core implements compute_dualharm; otherwise
        NotImplementedError will propagate (which is correct).
        """
        d = np.asarray(d, dtype=float)
        cfg = self.sweep

        if refused is None:
            refused_mask = np.zeros(d.shape, dtype=bool)
        else:
            refused_mask = np.asarray(refused, dtype=bool)
            if refused_mask.shape != d.shape:
                raise ValueError("refused mask must have same shape as d")

        if cfg.n_modes_start < 1:
            raise ValueError("ModeSweepConfig.n_modes_start must be >= 1")
        if cfg.n_modes_step < 1:
            raise ValueError("ModeSweepConfig.n_modes_step must be >= 1")
        if cfg.n_modes_max < cfg.n_modes_start:
            raise ValueError("ModeSweepConfig.n_modes_max must be >= n_modes_start")
        if cfg.tol <= 0:
            raise ValueError("ModeSweepConfig.tol must be > 0")

        # First evaluation at start modes
        res_prev = self.core.compute_dualharm(
            d,
            a1=a1,
            a2=a2,
            period=period,
            phi2=phi2,
            n_modes=cfg.n_modes_start,
            tol=cfg.tol,
        )
        eta_prev = np.asarray(res_prev.eta_levelC, dtype=float)

        converged = np.zeros(d.shape, dtype=bool)
        n_modes_at = np.full(d.shape, int(cfg.n_modes_start), dtype=int)

        # Trust boundary: refused points are never considered converged.
        converged[refused_mask] = False
        n_modes_at[refused_mask] = 0

        eta_last = eta_prev
        last_nm = int(cfg.n_modes_start)

        # Sweep upward
        for nm in range(cfg.n_modes_start + cfg.n_modes_step, cfg.n_modes_max + 1, cfg.n_modes_step):
            res = self.core.compute_dualharm(
                d,
                a1=a1,
                a2=a2,
                period=period,
                phi2=phi2,
                n_modes=nm,
                tol=cfg.tol,
            )
            eta = np.asarray(res.eta_levelC, dtype=float)

            delta = np.abs(eta - eta_prev)
            newly = (~converged) & (~refused_mask) & (delta < cfg.tol)

            n_modes_at[newly] = int(nm)
            converged |= newly

            eta_prev = eta
            eta_last = eta
            last_nm = int(nm)

            if np.all(converged | refused_mask):
                break

        # If not converged, record the max effort used (so n_modes shows the sweep happened)
        n_modes_at[(~converged) & (~refused_mask)] = last_nm

        return LevelCResult(
            eta_levelC=eta_last,
            n_modes=n_modes_at,
            converged=converged,
        )