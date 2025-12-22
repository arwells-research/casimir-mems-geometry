# FILE: experiments/reproduce_levelC_c0_sinusoid.py
#!/usr/bin/env python3
"""
experiments/reproduce_levelC_c0_sinusoid.py

Canonical first Level C benchmark: C0 (single-mode sinusoid vs flat, ideal conductor).

Purpose:
- Lock the Level C *contract* (I/O + audit CSV schema + plotting conventions)
- Provide a geometry-only target that Level C must hit to prove it works
- Enable immediate scaffolding using a stub backend, without committing to a solver

Outputs (git-ignored):
- figures/derived/levelC_c0_overlay.png
- outputs/levelC_test_run.csv   (schema locked)

This benchmark is synthetic (parameter-defined), not digitized.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math

import numpy as np
import matplotlib.pyplot as plt
import yaml

from casimir_mems.levelA.plane_plane import P_pp_ideal
from casimir_mems.levelC.interface import LevelCResult
from casimir_mems.levelC.convergence import ModeSweepBackend, ModeSweepConfig
from casimir_mems.levelC.toy_scattering import ToyScatteringBackend
from casimir_mems.levelC.ideal_perturb_scattering import IdealPerturbScatteringBackend


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "levelc_c0_sinusoid"
FIG_DIR = REPO_ROOT / "figures" / "derived"
OUT_DIR = REPO_ROOT / "outputs"

# Ideal-metal EM DE coefficient used by this repository (Option 1 diagnostic Level B)
BETA_EM_IDEAL = (2.0 / 3.0) * (1.0 - 15.0 / (math.pi**2))


def banner(tag: str, **kv: object) -> None:
    items = " ".join([f"{k}={v}" for k, v in kv.items()])
    print(f"[{tag}] {items}".rstrip())


@dataclass(frozen=True)
class C0Config:
    period_m: float
    d_min_m: float
    d_max_m: float
    n_points: int
    amplitude_mode: str  # "absolute" or "relative_to_dmin"
    amplitude_m: float
    amplitude_factor: float
    kd_warn: float
    kd_refuse: float
    # Level C metadata-driven sweep + backend selection.
    # NOTE: "n_modes" is the per-call value passed into ModeSweepBackend; for sweep
    # runs this should match n_modes_start (the wrapper will increase internally).
    n_modes: int
    converge_at_modes: int
    converge_tol: float
    levelc_backend: str
    n_modes_start: int
    n_modes_step: int
    n_modes_max: int
    toy_alpha: float
    ideal_alpha: float
    ideal_levelB_samples: int


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> C0Config:
    meta = yaml.safe_load(path.read_text())

    geom = meta.get("geometry", {}) or {}
    thr = meta.get("thresholds", {}) or {}
    lvlc_root = meta.get("levelC", {}) or {}
    lvlc = (lvlc_root.get("convergence", {}) or {})
    levelc_backend = str(lvlc_root.get("backend", "unknown")).strip()

    toy = (lvlc_root.get("toy_scattering", {}) or {})
    ideal = (lvlc_root.get("ideal_perturb_scattering", {}) or {})

    period_m = float(geom["corrugation_period_m"])
    d_min_m = float(geom["d_min_m"])
    d_max_m = float(geom["d_max_m"])
    n_points = int(geom.get("n_points", 41))

    amp_cfg = geom.get("amplitude", {}) or {}
    amplitude_mode = str(amp_cfg.get("mode", "relative_to_dmin"))
    amplitude_m = float(amp_cfg.get("a_m", 0.0))
    amplitude_factor = float(amp_cfg.get("a_over_dmin", 0.2))

    kd_warn = float(thr.get("kd_warn", thr.get("kd_max", 0.1)))
    kd_refuse = float(thr.get("kd_refuse", float("inf")))

    # --- Level C convergence / sweep (metadata-driven) ---
    n_modes_start = int(lvlc.get("n_modes_start", lvlc.get("n_modes", 32)))
    n_modes_step = int(lvlc.get("n_modes_step", 8))
    n_modes_max = int(lvlc.get("n_modes_max", max(n_modes_start, 128)))
    converge_tol = float(lvlc.get("tol", 1.0e-4))

    # This remains the "first-call" n_modes passed to ModeSweepBackend; it will sweep upward.
    n_modes = int(lvlc.get("n_modes", n_modes_start))
    converge_at_modes = int(lvlc.get("converge_at_modes", n_modes))

    toy_alpha = float(toy.get("alpha", 0.15))
    ideal_alpha = float(ideal.get("alpha", 0.35))
    ideal_levelB_samples = int(ideal.get("levelB_samples", 2048))

    # Minimal validation: keep failures crisp and attributable to metadata contract.
    if n_modes_start < 1:
        raise ValueError("metadata levelC.convergence.n_modes_start must be >= 1")
    if n_modes_step < 1:
        raise ValueError("metadata levelC.convergence.n_modes_step must be >= 1")
    if n_modes_max < n_modes_start:
        raise ValueError("metadata levelC.convergence.n_modes_max must be >= n_modes_start")
    if converge_tol <= 0.0:
        raise ValueError("metadata levelC.convergence.tol must be > 0")

    if levelc_backend not in ("toy_scattering_v0", "ideal_perturb_scattering_v0"):
        raise ValueError(
            "Unsupported levelC.backend in C0 metadata: "
            f"{levelc_backend!r} (supported: toy_scattering_v0, ideal_perturb_scattering_v0)"
        )

    if ideal_levelB_samples < 256:
        raise ValueError("metadata levelC.ideal_perturb_scattering.levelB_samples must be >= 256")

    return C0Config(
        period_m=period_m,
        d_min_m=d_min_m,
        d_max_m=d_max_m,
        n_points=n_points,
        amplitude_mode=amplitude_mode,
        amplitude_m=amplitude_m,
        amplitude_factor=amplitude_factor,
        kd_warn=kd_warn,
        kd_refuse=kd_refuse,
        n_modes=n_modes,
        converge_at_modes=converge_at_modes,
        converge_tol=converge_tol,
        levelc_backend=levelc_backend,
        n_modes_start=n_modes_start,
        n_modes_step=n_modes_step,
        n_modes_max=n_modes_max,
        toy_alpha=toy_alpha,
        ideal_alpha=ideal_alpha,
        ideal_levelB_samples=ideal_levelB_samples,
    )


def build_separation_grid(cfg: C0Config) -> np.ndarray:
    if cfg.n_points < 5:
        raise ValueError("n_points too small; use >= 5.")
    if not (cfg.d_min_m > 0 and cfg.d_max_m > cfg.d_min_m):
        raise ValueError("Invalid separation range.")
    return np.linspace(cfg.d_min_m, cfg.d_max_m, cfg.n_points, dtype=float)


def resolve_amplitude(cfg: C0Config) -> float:
    if cfg.amplitude_mode == "absolute":
        a = cfg.amplitude_m
    elif cfg.amplitude_mode == "relative_to_dmin":
        a = cfg.amplitude_factor * cfg.d_min_m
    else:
        raise ValueError(f"Unknown amplitude.mode: {cfg.amplitude_mode!r}")

    if a <= 0:
        raise ValueError("Amplitude must be > 0.")

    # Ensure trough gap stays positive at d_min.
    if cfg.d_min_m - a <= 0:
        raise ValueError(f"Non-positive minimum gap: d_min={cfg.d_min_m:.3e}, a={a:.3e}")

    return a


def eta_levelA_local_avg(d: np.ndarray, *, a: float, period: float, samples: int = 4096) -> np.ndarray:
    """
    Level A (C0): local averaging of ideal-metal plane-plane pressure.
      η_A(d) = <P(d + a sin(kx))> / P(d)
    """
    if samples < 256:
        raise ValueError("samples too low; use >= 256.")
    k = 2.0 * math.pi / period
    x = np.linspace(0.0, 2.0 * math.pi / k, samples, endpoint=False)
    sinx = np.sin(k * x)

    P0 = P_pp_ideal(d)
    out = np.empty_like(d, dtype=float)
    for i, di in enumerate(d):
        H = di + a * sinx
        if np.any(H <= 0.0):
            raise ValueError(f"Non-positive local gap at d={di:.3e} (a={a:.3e})")
        Pav = float(np.mean(P_pp_ideal(H)))
        out[i] = Pav / float(P0[i])
    return out


def eta_levelB_DE_ideal(d: np.ndarray, *, a: float, period: float, samples: int = 4096) -> np.ndarray:
    """
    Level B (Option 1 diagnostic): ideal-metal EM DE correction.
      η_B(d) = < P(H) [1 + β_EM (∇H)^2] > / P(d)
    """
    if samples < 256:
        raise ValueError("samples too low; use >= 256.")
    k = 2.0 * math.pi / period
    x = np.linspace(0.0, 2.0 * math.pi / k, samples, endpoint=False)
    sinx = np.sin(k * x)
    cosx = np.cos(k * x)

    grad2 = (a * k * cosx) ** 2

    P0 = P_pp_ideal(d)
    out = np.empty_like(d, dtype=float)
    for i, di in enumerate(d):
        H = di + a * sinx
        if np.any(H <= 0.0):
            raise ValueError(f"Non-positive local gap at d={di:.3e} (a={a:.3e})")
        PH = P_pp_ideal(H)
        Pcorr = float(np.mean(PH * (1.0 + BETA_EM_IDEAL * grad2)))
        out[i] = Pcorr / float(P0[i])
    return out


def enforce_c0_contract(
    *,
    d: np.ndarray,
    kd: np.ndarray,
    refused: np.ndarray,
    resC: LevelCResult,
    sweep: ModeSweepConfig,
    kd_refuse: float,
) -> None:
    """
    C0 is a synthetic harness. This function enforces trust boundaries:

    - refused rows (kd > kd_refuse): must have converged=False and n_modes=0
    - non-refused rows: must have n_modes within the sweep bounds
    """
    d = np.asarray(d, dtype=float)
    kd = np.asarray(kd, dtype=float)
    refused = np.asarray(refused, dtype=bool)

    if not (d.shape == kd.shape == refused.shape == resC.converged.shape == resC.n_modes.shape):
        raise ValueError("C0 contract check: shape mismatch")

    # Refusal masking is a hard trust boundary.
    if np.any(resC.converged[refused]):
        bad = int(np.sum(resC.converged[refused]))
        raise RuntimeError(f"C0 contract violated: {bad} refused points reported as converged")
    if np.any(resC.n_modes[refused] != 0):
        bad = int(np.sum(resC.n_modes[refused] != 0))
        raise RuntimeError(f"C0 contract violated: {bad} refused points have n_modes != 0")

    # Non-refused points must show that a sweep occurred (bounded modes).
    ok = ~refused
    if np.any(ok):
        nm_ok = resC.n_modes[ok]
        if np.any(nm_ok < sweep.n_modes_start):
            raise RuntimeError(
                f"C0 contract violated: n_modes below sweep start ({sweep.n_modes_start}) on non-refused rows"
            )
        if np.any(nm_ok > sweep.n_modes_max):
            raise RuntimeError(
                f"C0 contract violated: n_modes above sweep max ({sweep.n_modes_max}) on non-refused rows"
            )
        # Optional sanity: kd ordering should match d ordering (grid is increasing)
        if not np.all(np.diff(d) > 0):
            raise RuntimeError("C0 contract violated: separation grid must be strictly increasing")
        if not np.all(np.diff(kd) > 0):
            raise RuntimeError("C0 contract violated: kd must be strictly increasing with d")

    # Refusal definition itself: refused iff kd > kd_refuse
    refused_expected = kd > float(kd_refuse)
    if not np.array_equal(refused, refused_expected):
        raise RuntimeError("C0 contract violated: refused mask does not match kd > kd_refuse")


def write_levelC_test_csv(
    path: Path,
    *,
    d: np.ndarray,
    kd: np.ndarray,
    etaA: np.ndarray,
    etaB: np.ndarray,
    resC: LevelCResult,
    levelc_backend: str,
    levelc_sweep: str | None = None,
) -> None:
    """
    outputs/levelC_test_run.csv schema (locked):

      separation_m      float
      kd_value          float
      eta_levelA        float
      eta_levelB        float
      eta_levelC        float
      n_modes           int
      converged         bool/int{0,1}
      levelC_backend    str
    """
    if not (d.shape == kd.shape == etaA.shape == etaB.shape == resC.eta_levelC.shape):
        raise ValueError("CSV write: shape mismatch among arrays.")
    if not (resC.n_modes.shape == d.shape and resC.converged.shape == d.shape):
        raise ValueError("CSV write: LevelC bookkeeping shape mismatch.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "separation_m",
                "kd_value",
                "eta_levelA",
                "eta_levelB",
                "eta_levelC",
                "n_modes",
                "converged",
                "levelC_backend",
            ]
        )
        for di, kdi, a0, b0, c0, nm, cv in zip(
            d, kd, etaA, etaB, resC.eta_levelC, resC.n_modes, resC.converged
        ):
            w.writerow(
                [
                    f"{di:.12e}",
                    f"{kdi:.12e}",
                    f"{a0:.12e}",
                    f"{b0:.12e}",
                    f"{c0:.12e}",
                    int(nm),
                    int(bool(cv)),
                    levelc_backend,
                ]
            )


def _build_levelc_core(cfg: C0Config, *, etaB: np.ndarray) -> object:
    """
    Instantiate the Level C *core* backend declared by metadata.
    Returns a LevelCBackend-compatible object with .compute_sinusoid().
    """
    if cfg.levelc_backend == "toy_scattering_v0":
        return ToyScatteringBackend(eta_levelB=etaB, alpha=cfg.toy_alpha)
    if cfg.levelc_backend == "ideal_perturb_scattering_v0":
        return IdealPerturbScatteringBackend(
            eta_levelB=etaB,
            alpha=cfg.ideal_alpha,
            levelB_samples=cfg.ideal_levelB_samples,
        )
    raise ValueError(f"Unsupported C0 levelC.backend={cfg.levelc_backend!r}")


def main() -> None:
    meta_path = DATA_DIR / "metadata.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    cfg = load_config(meta_path)
    d = build_separation_grid(cfg)
    a = resolve_amplitude(cfg)
    period = cfg.period_m
    tol = cfg.converge_tol
    n_modes = cfg.n_modes

    k = 2.0 * math.pi / cfg.period_m
    kd = k * d

    warned = kd > cfg.kd_warn
    refused = kd > cfg.kd_refuse

    etaA = eta_levelA_local_avg(d, a=a, period=cfg.period_m, samples=4096)
    etaB = eta_levelB_DE_ideal(d, a=a, period=cfg.period_m, samples=4096)

    core = _build_levelc_core(cfg, etaB=etaB)

    sweep = ModeSweepConfig(
        n_modes_start=cfg.n_modes_start,
        n_modes_step=cfg.n_modes_step,
        n_modes_max=cfg.n_modes_max,
        tol=cfg.converge_tol,
    )

    banner(
        "LEVELC_C0",
        period_m=f"{cfg.period_m:.3e}",
        d_min_m=f"{cfg.d_min_m:.3e}",
        d_max_m=f"{cfg.d_max_m:.3e}",
        kd_warn=cfg.kd_warn,
        kd_refuse=cfg.kd_refuse,
        levelC_backend=cfg.levelc_backend,
        sweep=f"{sweep.n_modes_start}..{sweep.n_modes_max} step {sweep.n_modes_step}",
        tol=cfg.converge_tol,
    )

    backend = ModeSweepBackend(core=core, sweep=sweep)
    resC = backend.compute_sinusoid(d, a=a, period=period, n_modes=n_modes, tol=tol, refused=refused)

    # Per-case core call-count banner for tools/repro.sh sentinels
    print(f"[LEVELC_C0] Level C core calls: {getattr(core, 'call_count', -1)}")

    # Hard contract check (synthetic harness)
    enforce_c0_contract(
        d=d,
        kd=kd,
        refused=refused,
        resC=resC,
        sweep=sweep,
        kd_refuse=cfg.kd_refuse,
    )

    ensure_dir(FIG_DIR)
    ensure_dir(OUT_DIR)

    out_csv = OUT_DIR / "levelC_test_run.csv"

    # Contract: CSV records the metadata-declared backend id (not inferred from object attributes).
    write_levelC_test_csv(
        out_csv,
        d=d,
        kd=kd,
        etaA=etaA,
        etaB=etaB,
        resC=resC,
        levelc_backend=str(cfg.levelc_backend),
    )

    # --- Level C C0 contract checks (must never silently change) ---
    rows = list(csv.DictReader(out_csv.open()))
    assert len(rows) == cfg.n_points, "C0 row count changed"

    # Refusal logic invariant
    refused_rows = [r for r in rows if float(r["kd_value"]) > cfg.kd_refuse]
    assert all(int(r["converged"]) == 0 for r in refused_rows), "Refused rows converged"

    # Convergence invariant (non-refused)
    ok_rows = [r for r in rows if float(r["kd_value"]) <= cfg.kd_refuse]
    assert any(int(r["converged"]) == 1 for r in ok_rows), "No converged points in C0"

    print("OK: Level C C0 contract checks passed.")

    # Plot: ladder A/B/C + warning/refusal shading + non-converged markers
    plt.figure()

    if np.any(warned):
        d_warn = float(d[np.argmax(warned)])
        plt.axvspan(d_warn, float(d[-1]), alpha=0.15, label=f"Warning: k d > {cfg.kd_warn:g}")

    if np.any(refused):
        d_ref = float(d[np.argmax(refused)])
        plt.axvspan(d_ref, float(d[-1]), alpha=0.30, label=f"Refusal: k d > {cfg.kd_refuse:g}")

    plt.plot(d, etaA, linestyle="--", label="Level A: local averaging (ideal)")
    plt.plot(d, etaB, linestyle="-", label=f"Level B: DE (ideal, beta={BETA_EM_IDEAL:.6g})")
    plt.plot(
        d,
        resC.eta_levelC,
        linestyle="-.",
        label=(
            f"Level C: backend={cfg.levelc_backend} "
            f"(sweep {sweep.n_modes_start}..{sweep.n_modes_max} step {sweep.n_modes_step}, tol={sweep.tol:.1e})"
        ),
    )

    # Highlight non-converged points (if any)
    if np.any(~resC.converged):
        idx = np.where(~resC.converged)[0]
        plt.plot(d[idx], resC.eta_levelC[idx], marker="x", linestyle="none", label="Level C: not converged")

    plt.xlabel("Separation d (m)")
    plt.ylabel("Normalized observable η(d)")
    plt.title("Level C C0 (synthetic harness): sinusoid vs flat — convergence + refusal test")
    plt.legend()
    plt.tight_layout()

    out_png = FIG_DIR / "levelC_c0_overlay.png"

    banner(
        "LEVELC_C0_OUT",
        fig=str(out_png.relative_to(REPO_ROOT)),
        csv=str(out_csv.relative_to(REPO_ROOT)),
    )

    plt.savefig(out_png, dpi=200)
    plt.close()

    print("Level C C0 harness complete.")
    print("Wrote:")
    print(f"  {out_png}")
    print(f"  {out_csv}")
    print("Params:")
    print(f"  period_m={cfg.period_m:.3e}  a_m={a:.3e}  d_min_m={cfg.d_min_m:.3e}  d_max_m={cfg.d_max_m:.3e}")
    print(f"  levelC_backend={cfg.levelc_backend}")
    print("kd range:", float(np.min(kd)), "to", float(np.max(kd)))
    print("Warned points:", int(np.sum(warned)), "of", int(d.size), f"(kd_warn={cfg.kd_warn:g})")
    print("Refused points:", int(np.sum(refused)), "of", int(d.size), f"(kd_refuse={cfg.kd_refuse:g})")
    n_total = int(d.size)
    n_ref = int(np.sum(refused))
    n_ok = n_total - n_ref
    n_conv = int(np.sum(resC.converged & ~refused))

    nm_min = int(np.min(resC.n_modes[~refused])) if n_ok > 0 else 0
    nm_max = int(np.max(resC.n_modes[~refused])) if n_ok > 0 else 0

    print(
        "Level C (mode sweep) converged points:",
        n_conv,
        "of",
        n_ok,
        f"(n_modes range {nm_min}..{nm_max})",
    )
    print("Level C core calls (toy backend):", getattr(core, "call_count", -1))


if __name__ == "__main__":
    main()