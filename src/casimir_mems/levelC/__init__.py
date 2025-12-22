# FILE: src/casimir_mems/levelC/__init__.py
"""
Level C: nonlocal / scattering geometry (contract scaffolding).

This package deliberately starts as a *harness*:
- locks I/O contracts and audit schemas
- provides a backend interface
- supplies a stub backend that enforces convergence bookkeeping

A real Level C solver can later be plugged in without changing experiment scripts.
"""

from .interface import compute_eta_levelC_sinusoid  # noqa: F401