# FILE: src/casimir_mems/levelC/ideal_perturb_scattering.py
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .interface import LevelCBackend, LevelCResult


@dataclass(frozen=True)
class IdealPerturbScatteringBackend(LevelCBackend):
    """
    ideal_perturb_scattering_v0

    Deterministic, geometry-only "perturbative scattering-like" mode sum
    used as the first *real* Level C backend contract.

    IMPORTANT:
      - This is still ideal-metal and still geometry-first.
      - It is NOT a full scattering solver (no Lifshitz, no materials).
      - It exists to provide a stable, convergent, mode-resolved Level C curve
        whose convergence properties are meaningful and reproducible.

    Contract definition (sinusoid vs flat, normalized by Level B):

      k = 2π / period
      kd = k d
      q = a k

      S_N(kd) = q^2 * Σ_{m=1..N} exp(-2 m kd) / m^2

      η_C(d;N) = η_B(d) * (1 + α * S_N(kd))

    Notes:
      - S_N is monotone in N and convergent as N→∞.
      - For kd→∞, S_N→0.
      - For kd→0, S_N→ q^2 * (π^2/6) (as N→∞).
    """

    eta_levelB: np.ndarray
    alpha: float = 0.35
    # Audit-only: kept to reflect metadata intent / reproducibility knobs. Not used in the core.
    levelB_samples: int = field(default=0, compare=False)
    # audit
    call_count: int = field(default=0, compare=False)

    # stable identifier used in logs/CSV metadata
    backend_id: str = "ideal_perturb_scattering_v0"

    def compute_sinusoid(
        self,
        d: np.ndarray,
        *,
        a: float,
        period: float,
        n_modes: int,
        tol: float,
    ) -> LevelCResult:
        # NOTE: dataclass is frozen, but we still want a counter for audit.
        object.__setattr__(self, "call_count", int(self.call_count) + 1)

        d = np.asarray(d, dtype=float)
        if d.ndim != 1:
            raise ValueError("d must be a 1D array")
        if np.any(d <= 0.0):
            raise ValueError("d must be strictly positive")
        if float(period) <= 0.0:
            raise ValueError("period must be > 0")
        if float(a) <= 0.0:
            raise ValueError("amplitude a must be > 0")
        if int(n_modes) < 1:
            raise ValueError("n_modes must be >= 1")

        eta_b = np.asarray(self.eta_levelB, dtype=float)
        if eta_b.shape != d.shape:
            raise ValueError("IdealPerturbScatteringBackend requires eta_levelB with same shape as d.")

        k = (2.0 * np.pi) / float(period)
        kd = k * d  # (M,)
        q2 = float(a * k) ** 2

        # mode index (N,1) to broadcast against kd (1,M)
        m = np.arange(1, int(n_modes) + 1, dtype=float)[:, None]

        # S_N(kd) = q^2 * Σ exp(-2 m kd) / m^2
        # Use stable broadcasting; always finite for kd>=0.
        S = q2 * np.sum(np.exp(-2.0 * m * kd[None, :]) / (m * m), axis=0)

        eta_c = eta_b * (1.0 + float(self.alpha) * S)

        # Convergence is decided by ModeSweepBackend; core returns bookkeeping only.
        converged = np.zeros(d.shape, dtype=bool)
        nm = np.full(d.shape, int(n_modes), dtype=int)

        return LevelCResult(
            eta_levelC=eta_c,
            n_modes=nm,
            converged=converged,
        )