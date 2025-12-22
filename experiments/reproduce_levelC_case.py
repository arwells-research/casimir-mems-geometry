# FILE: experiments/reproduce_levelC_case.py
#!/usr/bin/env python3
"""
experiments/reproduce_levelC_case.py

Generic Level C case harness.

Given a case directory (data/raw/<case_id>/metadata.yaml), this script:
- computes Level A (local averaging) and Level B (ideal DE) baselines
- runs Level C via ModeSweepBackend using the metadata-selected core backend
- enforces refusal masking and sweep behavior invariants
- writes an audit CSV (schema locked) and overlay figure (git-ignored)

Outputs (git-ignored):
- figures/derived/<case_id>_overlay.png
- outputs/<case_id>_run.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
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
from casimir_mems.levelC.ideal_scattering_minimal import IdealScatteringMinimalBackend

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures" / "derived"
OUT_DIR = REPO_ROOT / "outputs"

# Ideal-metal EM DE coefficient used by this repository (Option 1 diagnostic Level B)
BETA_EM_IDEAL = (2.0 / 3.0) * (1.0 - 15.0 / (math.pi**2))


def banner(tag: str, **kv: object) -> None:
    items = " ".join([f"{k}={v}" for k, v in kv.items()])
    print(f"[{tag}] {items}".rstrip())


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    period_m: float
    d_min_m: float
    d_max_m: float
    n_points: int
    amplitude_mode: str  # "absolute" or "relative_to_dmin"
    amplitude_m: float
    amplitude_factor: float
    kd_warn: float
    kd_refuse: float
    levelc_backend: str
    n_modes_start: int
    n_modes_step: int
    n_modes_max: int
    tol: float
    # backend-specific knobs
    toy_alpha: float
    ideal_alpha: float
    ideal_levelB_samples: int


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_case_config(case_dir: Path) -> CaseConfig:
    meta_path = case_dir / "metadata.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    meta = yaml.safe_load(meta_path.read_text())

    geom = meta.get("geometry", {}) or {}
    thr = meta.get("thresholds", {}) or {}
    lvlc_root = meta.get("levelC", {}) or {}
    lvlc = (lvlc_root.get("convergence", {}) or {})
    levelc_backend = str(lvlc_root.get("backend", "unknown")).strip()

    toy = (lvlc_root.get("toy_scattering", {}) or {})
    ideal = (lvlc_root.get("ideal_perturb_scattering", {}) or {})

    case_id = str(meta.get("case_id", case_dir.name)).strip()
    if not case_id:
        raise ValueError("metadata.yaml missing case_id (and directory name is empty?)")

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

    n_modes_start = int(lvlc.get("n_modes_start", 8))
    n_modes_step = int(lvlc.get("n_modes_step", 8))
    n_modes_max = int(lvlc.get("n_modes_max", 128))
    tol = float(lvlc.get("tol", 1.0e-4))

    toy_alpha = float(toy.get("alpha", 0.15))
    ideal_alpha = float(ideal.get("alpha", 0.35))
    ideal_levelB_samples = int(ideal.get("levelB_samples", 2048))

    if levelc_backend not in ("toy_scattering_v0", "ideal_perturb_scattering_v0", "ideal_scattering_minimal_v0"):
        raise ValueError(
            f"Unsupported levelC.backend={levelc_backend!r} "
            "(supported: toy_scattering_v0, ideal_perturb_scattering_v0, ideal_scattering_minimal_v0)"
        )
    if n_modes_start < 1 or n_modes_step < 1 or n_modes_max < n_modes_start:
        raise ValueError("Invalid levelC.convergence sweep parameters (start/step/max)")
    if tol <= 0.0:
        raise ValueError("levelC.convergence.tol must be > 0")
    if ideal_levelB_samples < 256:
        raise ValueError("levelC.ideal_perturb_scattering.levelB_samples must be >= 256")

    return CaseConfig(
        case_id=case_id,
        period_m=period_m,
        d_min_m=d_min_m,
        d_max_m=d_max_m,
        n_points=n_points,
        amplitude_mode=amplitude_mode,
        amplitude_m=amplitude_m,
        amplitude_factor=amplitude_factor,
        kd_warn=kd_warn,
        kd_refuse=kd_refuse,
        levelc_backend=levelc_backend,
        n_modes_start=n_modes_start,
        n_modes_step=n_modes_step,
        n_modes_max=n_modes_max,
        tol=tol,
        toy_alpha=toy_alpha,
        ideal_alpha=ideal_alpha,
        ideal_levelB_samples=ideal_levelB_samples,
    )


def build_separation_grid(cfg: CaseConfig) -> np.ndarray:
    if cfg.n_points < 5:
        raise ValueError("n_points too small; use >= 5.")
    if not (cfg.d_min_m > 0 and cfg.d_max_m > cfg.d_min_m):
        raise ValueError("Invalid separation range.")
    return np.linspace(cfg.d_min_m, cfg.d_max_m, cfg.n_points, dtype=float)


def resolve_amplitude(cfg: CaseConfig) -> float:
    if cfg.amplitude_mode == "absolute":
        a = cfg.amplitude_m
    elif cfg.amplitude_mode == "relative_to_dmin":
        a = cfg.amplitude_factor * cfg.d_min_m
    else:
        raise ValueError(f"Unknown amplitude.mode: {cfg.amplitude_mode!r}")

    if a <= 0:
        raise ValueError("Amplitude must be > 0.")
    if cfg.d_min_m - a <= 0:
        raise ValueError(f"Non-positive minimum gap: d_min={cfg.d_min_m:.3e}, a={a:.3e}")
    return a


def eta_levelA_local_avg(d: np.ndarray, *, a: float, period: float, samples: int = 4096) -> np.ndarray:
    """
    Level A: local averaging of ideal-metal plane-plane pressure.
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


def write_case_csv(
    path: Path,
    *,
    d: np.ndarray,
    kd: np.ndarray,
    etaA: np.ndarray,
    etaB: np.ndarray,
    resC: LevelCResult,
    levelc_backend: str,
) -> None:
    """
    Audit CSV schema (locked, matches C0):

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


def _build_levelc_core(cfg: CaseConfig, *, etaB: np.ndarray) -> object:
    if cfg.levelc_backend == "toy_scattering_v0":
        return ToyScatteringBackend(eta_levelB=etaB, alpha=cfg.toy_alpha)
    if cfg.levelc_backend == "ideal_perturb_scattering_v0":
        return IdealPerturbScatteringBackend(
            eta_levelB=etaB,
            alpha=cfg.ideal_alpha,
            levelB_samples=cfg.ideal_levelB_samples,
        )
    if cfg.levelc_backend == "ideal_scattering_minimal_v0":
        return IdealScatteringMinimalBackend()        
    raise ValueError(f"Unsupported levelC.backend={cfg.levelc_backend!r}")

def _build_levelc_core_by_id(backend_id: str, cfg: CaseConfig, *, etaB: np.ndarray) -> object:
    if backend_id == "toy_scattering_v0":
        return ToyScatteringBackend(eta_levelB=etaB, alpha=cfg.toy_alpha)
    if backend_id == "ideal_perturb_scattering_v0":
        return IdealPerturbScatteringBackend(
            eta_levelB=etaB,
            alpha=cfg.ideal_alpha,
            levelB_samples=cfg.ideal_levelB_samples,
        )
    if backend_id == "ideal_scattering_minimal_v0":
        return IdealScatteringMinimalBackend()
    raise ValueError(f"Unsupported backend_id={backend_id!r}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True, help="Case directory under data/raw/ (or absolute path)")
    ap.add_argument(
        "--compare-backends",
        action="store_true",
        help="Diagnostic: plot Level C curves for toy/ideal_perturb/scattmin together (no CSV changes).",
    )
    args = ap.parse_args()

    case_dir_arg = Path(args.case_dir)
    case_dir = (REPO_ROOT / case_dir_arg).resolve() if not case_dir_arg.is_absolute() else case_dir_arg.resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case dir not found: {case_dir}")

    cfg = load_case_config(case_dir)
    d = build_separation_grid(cfg)
    a = resolve_amplitude(cfg)

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
        tol=cfg.tol,
    )

    banner(
        "LEVELC_CASE",
        case_id=cfg.case_id,
        period_m=f"{cfg.period_m:.3e}",
        d_min_m=f"{cfg.d_min_m:.3e}",
        d_max_m=f"{cfg.d_max_m:.3e}",
        kd_warn=cfg.kd_warn,
        kd_refuse=cfg.kd_refuse,
        levelC_backend=cfg.levelc_backend,
        sweep=f"{sweep.n_modes_start}..{sweep.n_modes_max} step {sweep.n_modes_step}",
        tol=f"{sweep.tol:.1e}",
    )

    backend = ModeSweepBackend(core=core, sweep=sweep)
    resC = backend.compute_sinusoid(
        d, a=a, period=cfg.period_m, n_modes=cfg.n_modes_start, tol=cfg.tol, refused=refused
    )

    case_tag = cfg.case_id.split("_", 2)[1].upper()  # "c1" -> "C1"
    print(f"[LEVELC_{case_tag}] Level C core calls: {getattr(core, 'call_count', -1)}")

    ensure_dir(FIG_DIR)
    ensure_dir(OUT_DIR)

    out_csv = OUT_DIR / f"{cfg.case_id}_run.csv"
    out_png = FIG_DIR / f"{cfg.case_id}_overlay.png"

    write_case_csv(
        out_csv,
        d=d,
        kd=kd,
        etaA=etaA,
        etaB=etaB,
        resC=resC,
        levelc_backend=str(cfg.levelc_backend),
    )

    # Plot
    plt.figure(figsize=(9.0, 6.0))

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
        label=f"Level C: {cfg.levelc_backend}",
    )
    # Optional diagnostic: compare Level C backends on the same geometry/grid.
    if args.compare_backends:
        compare_ids = [
            "toy_scattering_v0",
            "ideal_perturb_scattering_v0",
            "ideal_scattering_minimal_v0",
        ]

        for bid in compare_ids:
            core_cmp = _build_levelc_core_by_id(bid, cfg, etaB=etaB)
            backend_cmp = ModeSweepBackend(core=core_cmp, sweep=sweep)
            res_cmp = backend_cmp.compute_sinusoid(
                d, a=a, period=cfg.period_m, n_modes=cfg.n_modes_start, tol=cfg.tol, refused=refused
            )

            # Skip re-plot of the already-selected primary backend to avoid duplicates.
            if bid == cfg.levelc_backend:
                continue

            # Slightly de-emphasize comparison curves so the primary backend reads first.
            # Toy backend tends to be the loudest; fade it a bit more.
            cmp_alpha = 0.65
            cmp_lw = 2.0
            if bid == "toy_scattering_v0":
                cmp_alpha = 0.55

            plt.plot(
                d,
                res_cmp.eta_levelC,
                linestyle=":",
                linewidth=cmp_lw,
                alpha=cmp_alpha,
                label=f"Level C cmp: {bid}",
            )

        # Save an extra figure name (diagnostic-only)
        out_png_cmp = FIG_DIR / f"{cfg.case_id}_overlay_compare_backends.png"
        plt.tight_layout()
        plt.savefig(out_png_cmp, dpi=200, bbox_inches="tight")
        print(f"Diagnostic compare figure wrote:\n  {out_png_cmp}")

    if np.any(~resC.converged):
        idx = np.where(~resC.converged)[0]
        plt.plot(d[idx], resC.eta_levelC[idx], marker="x", linestyle="none", label="Level C: not converged")

    title1 = f"Level C case {cfg.case_id}: sinusoid vs flat"
    title2 = f"sweep {sweep.n_modes_start}..{sweep.n_modes_max} step {sweep.n_modes_step} | tol={sweep.tol:.1e}"
    plt.title(title1 + "\n" + title2, fontsize=11)
    plt.xlabel("Separation d (m)")
    plt.ylabel("Normalized observable η(d)")

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        fontsize=8,
        frameon=True,
    )

    plt.tight_layout()

    banner(
        "LEVELC_CASE_OUT",
        fig=str(out_png.relative_to(REPO_ROOT)),
        csv=str(out_csv.relative_to(REPO_ROOT)),
    )

    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print("Level C case harness complete.")
    print("Wrote:")
    print(f"  {out_png}")
    print(f"  {out_csv}")
    print("Level C core calls (backend):", getattr(core, "call_count", -1))


if __name__ == "__main__":
    main()