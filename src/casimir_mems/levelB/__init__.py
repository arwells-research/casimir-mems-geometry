"""
casimir_mems.levelB

Level B (de_ideal_v0): Derivative Expansion (DE) bridge between Level A local averaging and
Level C explicitly nonlocal geometry methods.

This package intentionally implements an *ideal-conductor diagnostic* Level B
backend ("de_ideal_v0") used to:
- generate a controlled beyond-PFA geometric correction for smooth profiles,
- emit validity/refusal diagnostics (kd, a*k, score),
- and enforce explicit trust boundaries when DE assumptions are violated.

Level B is not a material-realistic solver; finite conductivity and temperature
are reserved for Level C backends (future IDs).
"""

from __future__ import annotations

# Frozen backend identifier (part of reproducibility contract)
LEVELB_BACKEND_ID: str = "de_ideal_v0"