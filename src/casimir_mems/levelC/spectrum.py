# src/casimir_mems/levelC/spectrum.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass(frozen=True)
class ProfileSpectrum:
    """
    1D periodic surface profile encoded as complex Fourier coefficients.

    Convention:
      h(x) = Re[ sum_{n=1..N} c_n * exp(i n k x) ]
    where:
      - k is the base wavevector (2π/λ)
      - c_n are complex coefficients encoding amplitude + phase.
        For a cosine term a cos(n k x + φ):
          c_n = a * exp(i φ)

    Notes:
      - We store only n>=1 terms (no DC offset).
      - Physical height is a real function; users should ensure symmetry as desired.
    """
    k_base: float
    coeffs: Dict[int, complex]

    def max_harmonic(self) -> int:
        return max(self.coeffs.keys()) if self.coeffs else 0

    def scaled_k(self, n: int) -> float:
        return float(n) * float(self.k_base)


def spectrum_sinusoid(a: float, k: float, phase: float = 0.0) -> ProfileSpectrum:
    """
    h(x) = a cos(k x + phase)
    """
    return ProfileSpectrum(k_base=float(k), coeffs={1: complex(a) * np.exp(1j * float(phase))})


def spectrum_dualharm(a1: float, a2: float, k: float, phi2: float = 0.0, phi1: float = 0.0) -> ProfileSpectrum:
    """
    h(x) = a1 cos(k x + phi1) + a2 cos(2 k x + phi2)
    """
    coeffs = {}
    if abs(a1) > 0.0:
        coeffs[1] = complex(a1) * np.exp(1j * float(phi1))
    if abs(a2) > 0.0:
        coeffs[2] = complex(a2) * np.exp(1j * float(phi2))
    return ProfileSpectrum(k_base=float(k), coeffs=coeffs)