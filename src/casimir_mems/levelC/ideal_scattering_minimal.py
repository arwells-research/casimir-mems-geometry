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


def _toeplitz_from_coeffs(m: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Build Toeplitz C with C_{mn} = c_{m-n}, given:
      - m: modes in [-N..N] (int array length L=2N+1)
      - coeffs: complex or float array for c_p over p in [-2N..2N], length 4N+1
        indexed as coeffs[p + 2N]
    """
    N = (m.size - 1) // 2
    dm = (m[:, None] - m[None, :]).astype(int)  # in [-2N..2N]
    idx = dm + 2 * N
    return coeffs[idx]


def _coeffs_exp_xcos(n_max: int, x: float) -> np.ndarray:
    """
    Fourier coefficients c_p for exp( x cos(theta) ):

      exp(x cos θ) = Σ_{p=-∞..∞} I_p(x) e^{i p θ}

    Returns c_p for p in [-n_max..n_max] as a real array (since I_p(x) real),
    indexed by p+n_max.
    """
    In = _bessel_I_orders_up_to(n_max, x)  # I_0..I_n_max
    # c_0 = I_0, c_{±p} = I_p
    out = np.zeros(2 * n_max + 1, dtype=float)
    out[n_max] = In[0]
    for p in range(1, n_max + 1):
        out[n_max + p] = In[p]
        out[n_max - p] = In[p]
    return out


def _coeffs_exp_x1cos_plus_x2cos2(n_max: int, x1: float, x2: float, phi2: float) -> np.ndarray:
    r"""
    Fourier coefficients c_n for:

      f(θ) = exp( x1 cos θ + x2 cos(2θ + φ2) )

    Using product of Jacobi-Anger expansions:

      exp(x1 cos θ)          = Σ_p I_p(x1) e^{i p θ}
      exp(x2 cos(2θ + φ2))   = Σ_q I_q(x2) e^{i q (2θ + φ2)} = Σ_q I_q(x2) e^{i 2q θ} e^{i q φ2}

    Multiply and collect:
      c_n = Σ_q I_{n-2q}(x1) * I_q(x2) * e^{i q φ2}

    Returns c_n for n in [-n_max..n_max], complex array indexed by n+n_max.

    Notes:
    - If phi2=0, coefficients are real.
    - For real-valued f(θ), coefficients satisfy c_-n = conj(c_n).
    """
    # Need I_p up to |p| <= n_max + 2*|q|, but we truncate consistently.
    # We'll sum q over [-n_max..n_max]; terms outside will be negligible for typical x2.
    I1 = _bessel_I_orders_up_to(n_max + 2 * n_max, x1)  # up to 3 n_max (safe)
    I2 = _bessel_I_orders_up_to(n_max, x2)              # up to n_max

    out = np.zeros(2 * n_max + 1, dtype=np.complex128)

    # Helper: integer-order I_{-n}=I_n
    def I_from(arr: np.ndarray, n: int) -> float:
        nn = abs(int(n))
        if nn >= arr.size:
            return 0.0
        return float(arr[nn])

    for n in range(-n_max, n_max + 1):
        s = 0.0 + 0.0j
        for q in range(-n_max, n_max + 1):
            p = n - 2 * q
            s += I_from(I1, p) * I_from(I2, q) * np.exp(1j * float(q) * float(phi2))
        out[n + n_max] = s

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

      - For a sinusoid: exp( 2 κ0 a cos(kx) ) Fourier coefficients are I_p(2κ0 a),
        giving Toeplitz coupling C_{m,n} = I_{m-n}(2κ0 a).

      - For dual-harmonic: exp(2κ0[a1 cos θ + a2 cos(2θ+φ2)]) coefficients are
        computed via a controlled Bessel-convolution:
            c_n = Σ_q I_{n-2q}(x1) I_q(x2) e^{i q φ2}, with x1=2κ0 a1, x2=2κ0 a2.
        and C_{m,n} = c_{m-n}.

      - Round-trip operator:
            M = R_plane * P * R_corr * P = P * C * P

      - Energy proxy (dimensionless):
            E_N(d) = - log |det(I - M)|
        Flat reference (a=0 => C=I):
            E_flat,N(d) = - log det( I - P^2 )

      - Report:
            η_C(d) = E_N(d) / E_flat,N(d)

    Contract expectations for your repo:
      - Refusal masking and convergence bookkeeping are handled by ModeSweepBackend.
      - This core returns:
            eta_levelC (shape matches d)
            n_modes (filled with passed n_modes)
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
            kappa0 = 1.0 / float(di)
            kappa_m = np.sqrt(kappa0 * kappa0 + (m.astype(float) * k) ** 2)
            P = np.exp(-2.0 * kappa_m * float(di))  # shape (L,)

            # Build coupling coefficients for exp(2 κ0 a cos θ)
            x = 2.0 * kappa0 * float(a)
            coeffs = _coeffs_exp_xcos(2 * N, x)  # p in [-2N..2N]
            C = _toeplitz_from_coeffs(m, coeffs.astype(float))

            M = (P[:, None] * C) * P[None, :]

            sign, logabsdet = np.linalg.slogdet(I - M)
            if not np.isfinite(logabsdet):
                eta_out[i] = float("nan")
                continue
            # sign is complex with |sign|=1 for a non-singular matrix; keep as diagnostic if desired
            if not np.isfinite(sign.real) or not np.isfinite(sign.imag):
                eta_out[i] = float("nan")
                continue

            E = -float(logabsdet)

            P2 = P * P
            if np.any(P2 >= 1.0):
                eta_out[i] = float("nan")
                continue
            E_flat = -float(np.sum(np.log1p(-P2)))

            eta_out[i] = E / E_flat if E_flat != 0.0 else float("nan")

        converged = np.zeros(d.shape, dtype=bool)
        nm = np.full(d.shape, int(n_modes), dtype=int)

        return LevelCResult(
            eta_levelC=eta_out,
            n_modes=nm,
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
        n_modes: int,
        tol: float,
    ) -> LevelCResult:
        """
        Dual-harmonic 1D profile:
          H(x) = d + a1 cos(kx) + a2 cos(2kx + phi2)

        This is the minimal-backend extension that makes Level C a real
        geometry platform (spectrum-sensitive, phase-sensitive).
        """
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
        if a1 <= 0.0:
            raise ValueError("a1 must be > 0")
        if a2 < 0.0:
            raise ValueError("a2 must be >= 0")
        if tol <= 0.0:
            raise ValueError("tol must be > 0")

        k = 2.0 * math.pi / float(period)
        N = int(n_modes)
        m = np.arange(-N, N + 1, dtype=int)  # length = 2N+1
        L = m.size

        eta_out = np.empty_like(d, dtype=float)

        # Complex identity (coeffs can be complex when phi2 != 0)
        I = np.eye(L, dtype=np.complex128)

        for i, di in enumerate(d):
            kappa0 = 1.0 / float(di)
            kappa_m = np.sqrt(kappa0 * kappa0 + (m.astype(float) * k) ** 2)
            P = np.exp(-2.0 * kappa_m * float(di)).astype(np.complex128)

            # Build coupling coefficients for exp(2 κ0 [a1 cos θ + a2 cos(2θ+phi2)])
            x1 = 2.0 * kappa0 * float(a1)
            x2 = 2.0 * kappa0 * float(a2)
            coeffs = _coeffs_exp_x1cos_plus_x2cos2(2 * N, x1, x2, float(phi2))  # n in [-2N..2N]
            C = _toeplitz_from_coeffs(m, coeffs).astype(np.complex128)

            M = (P[:, None] * C) * P[None, :]

            sign, logdet = np.linalg.slogdet(I - M)
            # For complex, sign is complex; det phase lives there. We only care that logdet is finite.
            if not np.isfinite(logdet.real) or not np.isfinite(logdet.imag):
                eta_out[i] = float("nan")
                continue

            E = -float(logdet.real)

            # Flat reference: C=I => M_flat = P^2 on diagonal (still real)
            P2 = (P * P).real
            if np.any(P2 >= 1.0):
                eta_out[i] = float("nan")
                continue
            E_flat = -float(np.sum(np.log1p(-P2)))

            eta_out[i] = E / E_flat if E_flat != 0.0 else float("nan")

        converged = np.zeros(d.shape, dtype=bool)
        nm = np.full(d.shape, int(n_modes), dtype=int)

        return LevelCResult(
            eta_levelC=eta_out,
            n_modes=nm,
            converged=converged,
        )