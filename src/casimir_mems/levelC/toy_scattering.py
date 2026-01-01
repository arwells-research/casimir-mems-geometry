from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math
import numpy as np

from .interface import LevelCBackend, LevelCResult


@dataclass(frozen=True)
class ToyScatteringBackend(LevelCBackend):
    """
    SYNTHETIC Level C backend (NOT physical scattering).

    PURPOSE (HARNESS ONLY):
      Exercise the Level C pipeline (mode sweep, CSV schema, refusal masking, plots)
      with a deterministic, kd-dependent, n_modes-dependent correction that converges
      under increasing mode count.

    NON-GOAL:
      This is NOT a scattering solver and MUST NOT be interpreted as physics.
      It exists only to validate convergence machinery and I/O contracts.

    Definition:
      kd = (2π/period) * d
      gamma(kd) = 0.15 + 0.25/(1+kd)
      S_N(kd) = sum_{m=1..N} exp(-gamma(kd) * m)
      eta_C = eta_B * (1 + alpha * S_N(kd))

    Replace with a real Level C backend (e.g., ideal-metal scattering) when available.
    """
    eta_levelB: np.ndarray
    alpha: float = 0.15
    call_count: int = field(default=0, compare=False)
    # Stable identifier for audit logs/CSV metadata. Keep as a field so callers can override
    # if desired, but default must remain stable for C0.
    backend_id: str = "toy_scattering_v0"

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
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")

        eta_b = np.asarray(self.eta_levelB, dtype=float)
        if eta_b.shape != d.shape:
            raise ValueError("ToyScatteringBackend requires eta_levelB with same shape as d.")

        k = (2.0 * np.pi) / float(period)
        kd = k * d

        m = np.arange(1, int(n_modes) + 1, dtype=float)[:, None]  # (N,1)
        if self.backend_id == "toy_scattering_v0":
            # v0 (original): slow-decay factor: small for small kd, approaches ~1 for large kd
            gamma = 0.15 + 0.25 / (1.0 + kd)   # in ~[0.18, 0.34] over your kd range
            S = np.sum(np.exp(-gamma[None, :] * m), axis=0)
        elif self.backend_id == "toy_scattering_v1":
            # v1 (alternate): still deterministic and convergent, but different from v0.
            # Slightly faster decay + mild oscillatory weighting to prove backend switching works.
            gamma = 0.18 + 0.30 / (1.0 + kd)
            base = np.exp(-gamma[None, :] * m)
            osc = 1.0 + 0.15 * np.cos(0.35 * m)
            S = np.sum(base * osc, axis=0)
        else:
            raise ValueError(f"Unknown ToyScattering backend_id={self.backend_id!r}")
        eta_c = eta_b * (1.0 + float(self.alpha) * S)

        # Convergence is decided by ModeSweepBackend; core returns metadata only.
        converged = np.zeros(d.shape, dtype=bool)
        nm = np.full(d.shape, int(n_modes), dtype=int)

        return LevelCResult(
            eta_levelC=eta_c,
            n_modes=nm,
            converged=converged,
        )