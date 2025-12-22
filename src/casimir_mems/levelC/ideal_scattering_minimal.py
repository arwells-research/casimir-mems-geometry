# FILE: src/casimir_mems/levelC/ideal_scattering_minimal.py
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from .interface import LevelCBackend, LevelCResult


def _bessel_I_orders_up_to(n_max: int, x: float) -> np.ndarray:
    """
    Return I_0(x)..I_n_max(x) for modified Bessel I_n (integer n>=0),
    using a stable power-series per order:

      I_n(x) = sum_{m=0..∞} (1/(m!(m+n)!)) * (x/2)^(2m+n)

    Implementation notes:
    - Avoid factorial overflow by computing the m=0 term using log-gamma:
        (x/2)^n / n! = exp(n*log(x/2) - lgamma(n+1))
    - Then advance via a stable recurrence in m:
        term_{m+1} = term_m * (x/2)^2 / ((m+1)(m+n+1))

    Notes:
    - If x==0: I_0(0)=1, I_n(0)=0 for n>0.
    """
    if n_max < 0:
        raise ValueError("n_max must be >= 0")

    out = np.zeros(n_max + 1, dtype=float)

    if x == 0.0:
        out[0] = 1.0
        return out

    half = 0.5 * float(x)
    half2 = half * half
    log_half = math.log(half)

    # Conservative but fast in practice for n_max <= ~512 in this repo.
    max_terms = 200
    tol = 1e-16

    for n in range(n_max + 1):
        # m=0 term in log domain: (half^n) / n! = exp(n*log(half) - lgamma(n+1))
        # This avoids huge intermediate integers from factorial(n).
        term = math.exp(n * log_half - math.lgamma(n + 1.0))
        s = term

        # term_{m+1} = term_m * half^2 / ((m+1)(m+n+1))
        for m in range(max_terms - 1):
            term *= half2 / ((m + 1) * (m + n + 1))
            s_new = s + term
            if abs(term) < tol * abs(s_new):
                s = s_new
                break
            s = s_new

        out[n] = float(s)

    return out

@dataclass(frozen=True)
class IdealScatteringMinimalBackend(LevelCBackend):
    """
    Minimal "real" (mode-mixing / log-det) Level C backend for ideal reflectors.

    This is NOT a full EM Lifshitz + scattering implementation. It is, however,
    genuinely "scattering-like" in the sense that it:
      - builds an explicit mode-coupling matrix in a truncated Fourier basis
      - forms a round-trip operator
      - evaluates a log-det convergence target under increasing mode count

    Geometry:
      - 1D periodic surface profile vs flat plate.
      - For analytic convenience of Fourier coefficients, we use:
            H(x) = d + a cos(k x)
        (sin vs cos is a lateral phase shift; the spectrum is the same.)

    Operator model (deliberately minimal / controlled approximation):
      - Use a scalar Dirichlet reflection model for both surfaces:
            R_plane = -I
            R_corr  = -C
        where C is a (2N+1)x(2N+1) coupling matrix induced by the corrugation.

      - Use a single representative decay scale κ0 = 1/d for the corrugation
        coupling coefficients. This is the main "minimal" approximation: we do
        NOT integrate over ξ, k_parallel, or Brillouin-zone kx.

      - Propagation for lateral mode index m in [-N..N]:
            κ_m = sqrt(κ0^2 + (m k)^2)
            P_m = exp(-2 κ_m d)

      - Corrugation coupling matrix from the Fourier series:
            exp( 2 κ0 a cos(kx) ) = Σ_{p=-∞..∞} I_p(2 κ0 a) e^{i p kx}
        So:
            C_{m,n} = I_{m-n}(2 κ0 a)

      - Round-trip operator:
            M = R_plane * P * R_corr * P
              = (-I) * P * (-C) * P
              = P * C * P
        (since the two minus signs cancel)

      - "Energy proxy" (dimensionless):
            E_N(d) = - log det( I - M )
        Flat reference (a=0 => C=I):
            E_flat,N(d) = - log det( I - P*I*P ) = - log det( I - P^2 )

      - Report the Level C normalized observable as:
            η_C(d) = E_N(d) / E_flat,N(d)
        This guarantees:
            a -> 0  =>  η_C(d) -> 1
        and provides a clean convergence target as N increases.

    Contract expectations for your repo:
      - Refusal masking and convergence bookkeeping are handled by ModeSweepBackend.
      - This core returns:
            eta_levelC (shape matches d)
            n_modes (filled with the passed n_modes)
            converged (all False; ModeSweepBackend decides convergence)

    Backend id:
      - backend_id defaults to "ideal_scattering_minimal_v0"
      - Keep stable once you freeze any benchmark using it.
    """
    call_count: int = field(default=0, compare=False)
    backend_id: str = "ideal_scattering_minimal_v0"

    def compute_sinusoid(
        self,
        d: np.ndarray,
        *,
        a: float,
        period: float,
        n_modes: int,
        tol: float,
    ) -> LevelCResult:
        # frozen=True => use object.__setattr__ for call_count
        object.__setattr__(self, "call_count", int(self.call_count) + 1)

        d = np.asarray(d, dtype=float)
        if d.ndim != 1:
            raise ValueError("d must be a 1D array")
        if np.any(d <= 0.0):
            raise ValueError("d must be strictly positive")
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        if period <= 0.0:
            raise ValueError("period must be > 0")
        if a <= 0.0:
            raise ValueError("a must be > 0")
        if tol <= 0.0:
            raise ValueError("tol must be > 0")

        k = 2.0 * math.pi / float(period)
        N = int(n_modes)
        m = np.arange(-N, N + 1, dtype=int)  # length = 2N+1
        L = m.size

        eta_out = np.empty_like(d, dtype=float)

        # Pre-allocate identity for logdet computations
        I = np.eye(L, dtype=float)

        for i, di in enumerate(d):
            # Representative evanescent scale (minimal approximation)
            kappa0 = 1.0 / float(di)

            # Propagation diagonal in truncated Fourier basis
            kappa_m = np.sqrt(kappa0 * kappa0 + (m.astype(float) * k) ** 2)
            P = np.exp(-2.0 * kappa_m * float(di))  # shape (L,)

            # Build C_{mn} = I_{m-n}(2 kappa0 a)
            # Need orders up to |m-n| <= 2N
            x = 2.0 * kappa0 * float(a)
            In = _bessel_I_orders_up_to(2 * N, x)  # I_0..I_{2N}

            # Difference matrix |m-n|
            dm = (m[:, None] - m[None, :]).astype(int)
            C = In[np.abs(dm)]  # integer p: I_-p = I_p

            # Round-trip: M = P*C*P (with P as diagonal)
            M = (P[:, None] * C) * P[None, :]

            # Energy proxy using slogdet for numerical robustness
            sign, logdet = np.linalg.slogdet(I - M)
            if sign <= 0:
                # Treat as approximation/numerical breakdown.
                eta_out[i] = float("nan")
                continue
            E = -float(logdet)

            # Flat reference: C=I => M_flat = P^2 on diagonal
            P2 = P * P
            if np.any(P2 >= 1.0):
                eta_out[i] = float("nan")
                continue
            E_flat = -float(np.sum(np.log1p(-P2)))

            # Normalized Level C observable
            eta_out[i] = E / E_flat if E_flat != 0.0 else float("nan")

        converged = np.zeros(d.shape, dtype=bool)
        nm = np.full(d.shape, int(n_modes), dtype=int)

        return LevelCResult(
            eta_levelC=eta_out,
            n_modes=nm,
            converged=converged,
        )