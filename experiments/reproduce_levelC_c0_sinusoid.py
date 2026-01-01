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
- figures/derived/levelc_c0_sinusoid_overlay.png
- outputs/levelc_c0_sinusoid_run.csv   (schema locked)

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

# --- Level B diagnostics (B0) ---
# We keep B0 in Level B package so experiments don’t duplicate math.
from casimir_mems.levelB.sinusoid import compute_levelB_sinusoid_b0


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "levelc_c0_sinusoid"
FIG_DIR = REPO_ROOT / "figures" / "derived"
OUT_DIR = REPO_ROOT / "outputs"


def banner(tag: str, **kv: object) -> None:
    items = " ".join([f"{k}={v}" for k, v in kv.items()])
    print(f"[{tag}] {items}".rstrip(), flush=True)


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

    n_modes = int(lvlc.get("n_modes", n_modes_start))
    converge_at_modes = int(lvlc.get("converge_at_modes", n_modes))

    toy_alpha = float(toy.get("alpha", 0.15))
    ideal_alpha = float(ideal.get("alpha", 0.35))
    ideal_levelB_samples = int(ideal.get("levelB_samples", 2048))

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
    if cfg.d_min_m - a <= 0:
        raise ValueError(f"Non-positive minimum gap: d_min={cfg.d_min_m:.3e}, a={a:.3e}")
    return a


def eta_levelA_local_avg(d: np.ndarray, *, a: float, period: float, samples: int = 4096) -> np.ndarray:
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


def enforce_c0_contract(
    *,
    d: np.ndarray,
    kd: np.ndarray,
    refused: np.ndarray,
    resC: LevelCResult,
    sweep: ModeSweepConfig,
    kd_refuse: float,
) -> None:
    d = np.asarray(d, dtype=float)
    kd = np.asarray(kd, dtype=float)
    refused = np.asarray(refused, dtype=bool)

    if not (d.shape == kd.shape == refused.shape == resC.converged.shape == resC.n_modes.shape):
        raise ValueError("C0 contract check: shape mismatch")

    if np.any(resC.converged[refused]):
        bad = int(np.sum(resC.converged[refused]))
        raise RuntimeError(f"C0 contract violated: {bad} refused points reported as converged")
    if np.any(resC.n_modes[refused] != 0):
        bad = int(np.sum(resC.n_modes[refused] != 0))
        raise RuntimeError(f"C0 contract violated: {bad} refused points have n_modes != 0")

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
        if not np.all(np.diff(d) > 0):
            raise RuntimeError("C0 contract violated: separation grid must be strictly increasing")
        if not np.all(np.diff(kd) > 0):
            raise RuntimeError("C0 contract violated: kd must be strictly increasing with d")

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
) -> None:
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

    # --- Level B diagnostics + η_B ---
    b0 = compute_levelB_sinusoid_b0(
        d,
        amplitude_m=a,
        period_m=cfg.period_m,
        kd_warn=cfg.kd_warn,
        kd_refuse=cfg.kd_refuse,
        samples=4096,
    )

    kd = b0.validity.kd
    warned = b0.validity.warned
    refused = b0.validity.refused

    etaA = eta_levelA_local_avg(d, a=a, period=cfg.period_m, samples=4096)
    etaB = b0.eta_levelB

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
        levelB_backend=b0.backend_id,
        levelC_backend=cfg.levelc_backend,
        sweep=f"{sweep.n_modes_start}..{sweep.n_modes_max} step {sweep.n_modes_step}",
        tol=f"{sweep.tol:.1e}",
    )

    core = _build_levelc_core(cfg, etaB=etaB)
    backend = ModeSweepBackend(core=core, sweep=sweep)
    resC = backend.compute_sinusoid(
        d, a=a, period=cfg.period_m, n_modes=cfg.n_modes, tol=cfg.converge_tol, refused=refused
    )

    # ---- REQUIRED SENTINEL FOR tools/repro.sh ----
    print(f"[LEVELC_C0] Level C core calls: {getattr(core, 'call_count', -1)}", flush=True)

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

    out_csv = OUT_DIR / "levelc_c0_sinusoid_run.csv"
    write_levelC_test_csv(
        out_csv,
        d=d,
        kd=kd,
        etaA=etaA,
        etaB=etaB,
        resC=resC,
        levelc_backend=str(cfg.levelc_backend),
    )

    # Contract checks (quick)
    rows = list(csv.DictReader(out_csv.open()))
    assert len(rows) == cfg.n_points, "C0 row count changed"

    refused_rows = [r for r in rows if float(r["kd_value"]) > cfg.kd_refuse]
    assert all(int(r["converged"]) == 0 for r in refused_rows), "Refused rows converged"

    ok_rows = [r for r in rows if float(r["kd_value"]) <= cfg.kd_refuse]
    assert any(int(r["converged"]) == 1 for r in ok_rows), "No converged points in C0"

    print("OK: Level C C0 contract checks passed.", flush=True)

    # Plot
    plt.figure()

    if np.any(warned):
        d_warn = float(d[np.argmax(warned)])
        plt.axvspan(d_warn, float(d[-1]), alpha=0.15, label=f"Warning: k d > {cfg.kd_warn:g}")

    if np.any(refused):
        d_ref = float(d[np.argmax(refused)])
        plt.axvspan(d_ref, float(d[-1]), alpha=0.30, label=f"Refusal: k d > {cfg.kd_refuse:g}")

    plt.plot(d, etaA, linestyle="--", label="Level A: local averaging (ideal)")
    plt.plot(d, etaB, linestyle="-", label="Level B: de_ideal_v0")
    plt.plot(
        d,
        resC.eta_levelC,
        linestyle="-.",
        label=f"Level C: backend={cfg.levelc_backend}",
    )

    if np.any(~resC.converged):
        idx = np.where(~resC.converged)[0]
        plt.plot(d[idx], resC.eta_levelC[idx], marker="x", linestyle="none", label="Level C: not converged")

    plt.xlabel("Separation d (m)")
    plt.ylabel("Normalized observable η(d)")
    plt.title("Level C C0 (synthetic harness): sinusoid vs flat — convergence + refusal test")
    plt.legend()
    plt.tight_layout()

    out_png = FIG_DIR / "levelc_c0_sinusoid_overlay.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    banner(
        "LEVELC_C0_OUT",
        fig=str(out_png.relative_to(REPO_ROOT)),
        csv=str(out_csv.relative_to(REPO_ROOT)),
    )


if __name__ == "__main__":
    main()
