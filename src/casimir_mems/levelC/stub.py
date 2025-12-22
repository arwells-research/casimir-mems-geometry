# FILE: src/casimir_mems/levelC/stub.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .interface import LevelCResult, LevelCBackend


@dataclass(frozen=True)
class LevelCStubBackend(LevelCBackend):
    """
    Contract-only stub backend.

    Behavior:
      - returns eta_levelC == eta_levelB (provided at construction time)
      - reports a fixed n_modes (the requested n_modes)
      - reports converged=True only if n_modes >= converge_at_modes

    This is NOT physics. It exists to lock:
      - experiment script behavior
      - output CSV schema
      - plotting conventions
      - convergence bookkeeping pipeline
    """
    eta_levelB: np.ndarray
    converge_at_modes: int = 32

    def compute_sinusoid(
        self,
        d: np.ndarray,
        *,
        a: float,
        period: float,
        n_modes: int,
        tol: float,
    ) -> LevelCResult:
        """
        Stub Level C backend.

        Purpose:
        - Provide a mode-dependent, smoothly convergent η_C(d; n_modes) for exercising
          the harness (plots, CSV schema, refusal logic, convergence wrapper).
        - Do NOT claim convergence here. Convergence is a harness responsibility
          (e.g., ModeSweepBackend), not the stub.

        Notes:
        - a/period/tol are accepted to match the interface; they are not used here.
        """
        d = np.asarray(d, dtype=float)

        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")

        eta_b = np.asarray(self.eta_levelB, dtype=float)
        if eta_b.shape != d.shape:
            raise ValueError("Stub backend requires eta_levelB with same shape as d.")

        # Mode-dependent toy behavior: approaches η_B as n_modes -> ∞
        eta_c = eta_b * (1.0 + 1.0 / float(n_modes * n_modes))

        # Convergence is *not* decided by the stub.
        converged = np.zeros(d.shape, dtype=bool)
        nm = np.full(d.shape, int(n_modes), dtype=int)

        return LevelCResult(
            eta_levelC=eta_c,
            n_modes=nm,
            converged=converged,
        )