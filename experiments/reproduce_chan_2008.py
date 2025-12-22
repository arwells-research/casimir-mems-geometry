#!/usr/bin/env python3
"""
experiments/reproduce_chan_2008.py

Reproduce Chan et al., PRL 101 (2008): gold sphere vs silicon rectangular trench array.

Outputs (written locally, git-ignored):
- figures/derived/chan_2008_overlay.png
- figures/derived/chan_2008_eta.png

Input contract:
- data/raw/chan_2008/metadata.yaml
- data/raw/chan_2008/digitized_curve.csv

CSV columns (required):
- separation_m
- force_gradient_SI   (N/m)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math

import numpy as np
import matplotlib.pyplot as plt
import yaml

from casimir_mems.types import Sphere, RectTrenchGrating, Calibration
from casimir_mems.levelA.interface import sphere_target_curve
from casimir_mems.levelA.diagnostics import deviation_factor

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "chan_2008"
SENTINELS_PATH = DATA_DIR / "sentinels.yaml"
FIG_DIR = REPO_ROOT / "figures" / "derived"
OUT_DIR = REPO_ROOT / "outputs"

def banner(tag: str, **kv: object) -> None:
    items = " ".join([f"{k}={v}" for k, v in kv.items()])
    print(f"[{tag}] {items}".rstrip())

@dataclass(frozen=True)
class Chan2008Config:
    sphere_R_m: float
    p_m: float
    w_m: float
    h_m: float
    d0_m: float
    y_kind: str
    y_column: str
    sign: str


def load_config(path: Path) -> Chan2008Config:
    cfg = yaml.safe_load(path.read_text())

    geom = cfg.get("geometry", {})
    cal = cfg.get("calibration", {})
    dc = cfg.get("data_convention", {})

    return Chan2008Config(
        sphere_R_m=float(geom["sphere_R_m"]),
        p_m=float(geom["trench_period_p_m"]),
        w_m=float(geom["trench_width_w_m"]),
        h_m=float(geom["trench_depth_h_m"]),
        d0_m=float(cal.get("d0_m", 0.0)),
        y_kind=str(dc.get("y_kind", "force_gradient")),
        y_column=str(dc.get("y_column", "force_gradient_SI")),
        sign=str(dc.get("sign", "signed")),
    )


def load_digitized_csv(path: Path, *, y_column: str) -> tuple[np.ndarray, np.ndarray]:
    # Supports optional comment lines starting with '#'
    d_list: list[float] = []
    y_list: list[float] = []

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = None

        for row in reader:
            if not row:
                continue
            if row[0].strip().startswith("#"):
                continue

            if header is None:
                header = [c.strip() for c in row]
                try:
                    d_idx = header.index("separation_m")
                    y_idx = header.index(y_column)
                except ValueError as e:
                    raise ValueError(
                        f"digitized_curve.csv must have headers: separation_m, {y_column}"
                    ) from e
                continue

            d_list.append(float(row[d_idx]))
            y_list.append(float(row[y_idx]))

    d = np.asarray(d_list, dtype=float)
    y = np.asarray(y_list, dtype=float)

    if d.ndim != 1 or y.ndim != 1 or d.size != y.size or d.size < 2:
        raise ValueError("Digitized data must be 1D arrays with >= 2 rows.")
    if np.any(d <= 0.0):
        raise ValueError("All separations must be > 0.")

    # Sort by separation (common for digitized data)
    order = np.argsort(d)
    return d[order], y[order]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_run_csv_chan(
    path: Path,
    d: np.ndarray,
    y_exp_signed: np.ndarray,
    y_base: np.ndarray,
    eta: np.ndarray,
) -> None:
    """
    Audit output for Chan 2008 reproduction.
    Columns:
      separation_m,
      exp_force_gradient_N_per_m,
      baseline_force_gradient_N_per_m,
      eta_exp_over_baseline
    """
    if not (d.size == y_exp_signed.size == y_base.size == eta.size):
        raise ValueError("Run CSV write: array sizes do not match.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "separation_m",
            "exp_force_gradient_N_per_m",
            "baseline_force_gradient_N_per_m",
            "eta_exp_over_baseline",
        ])
        for di, ye, yb, ei in zip(d, y_exp_signed, y_base, eta):
            w.writerow([f"{di:.12e}", f"{ye:.12e}", f"{yb:.12e}", f"{ei:.12e}"])

def interp_1d(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """
    Linear interpolation on a strictly increasing x-grid.
    If x0 exactly matches an entry, returns that value.
    Raises if x0 is outside the x range.
    """
    if x0 < float(x[0]) or x0 > float(x[-1]):
        raise ValueError(f"Sentinel x0={x0:.3e} outside digitized range [{x[0]:.3e}, {x[-1]:.3e}]")
    return float(np.interp(x0, x, y))


def enforce_sentinels(
    *,
    sentinels_path: Path,
    d: np.ndarray,
    y_curve: np.ndarray,
    curve_label: str,
) -> None:
    """
    Enforce sentinel points against a digitized curve.
    Sentinel schema (YAML list of dicts):
      - name: str (optional)
      - d_m: float
      - y: float
      - tol_abs: float (optional)
      - tol_rel: float (optional)
    """
    if not sentinels_path.exists():
        raise FileNotFoundError(f"Missing sentinels file: {sentinels_path}")

    raw = yaml.safe_load(sentinels_path.read_text())
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"Sentinels file must be a non-empty list: {sentinels_path}")

    failures: list[str] = []

    for s in raw:
        if not isinstance(s, dict):
            raise ValueError(f"Each sentinel must be a dict: {s!r}")
        name = str(s.get("name", "unnamed"))
        d_m = float(s["d_m"])
        y_exp = float(s["y"])
        tol_abs = float(s["tol_abs"]) if "tol_abs" in s else None
        tol_rel = float(s["tol_rel"]) if "tol_rel" in s else None

        y_at = interp_1d(d, y_curve, d_m)
        err = abs(y_at - y_exp)

        ok = False
        checks = []
        if tol_abs is not None:
            checks.append(err <= tol_abs)
        if tol_rel is not None:
            denom = max(abs(y_exp), 1e-30)
            checks.append(err <= tol_rel * denom)

        if checks:
            ok = all(checks)
        else:
            raise ValueError(f"Sentinel {name} must specify tol_abs and/or tol_rel")

        if not ok:
            failures.append(
                f"- {name}: d={d_m:.3e}, expected={y_exp:.6e}, got={y_at:.6e}, |err|={err:.3e}, "
                f"tol_abs={tol_abs}, tol_rel={tol_rel}"
            )

    if failures:
        msg = "\n".join(
            ["SENTINEL CHECK FAILED",
             f"  file: {sentinels_path}",
             f"  curve: {curve_label}",
             "  failures:"] + failures
        )
        raise RuntimeError(msg)

    print(f"OK: sentinels passed for {curve_label} ({sentinels_path.name})")

def main() -> None:
    meta_path = DATA_DIR / "metadata.yaml"
    csv_path = DATA_DIR / "digitized_curve.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing digitized curve: {csv_path}")

    cfg = load_config(meta_path)
    d, y_exp = load_digitized_csv(csv_path, y_column=cfg.y_column)

    banner(
        "CHAN2008",
        y_kind="force_gradient",
        y_column=str(cfg.y_column),
        sign=str(getattr(cfg, "sign", "unknown")),
        d_min_nm=f"{float(np.min(d))*1e9:.1f}",
        d_max_nm=f"{float(np.max(d))*1e9:.1f}",
        n_points=int(d.size),
    )

    # Geometry objects
    sphere = Sphere(R=cfg.sphere_R_m)
    grating = RectTrenchGrating(p=cfg.p_m, w=cfg.w_m, h=cfg.h_m)
    calib = Calibration(d0=cfg.d0_m)

    # Level A baseline: sphere–trench PFA-mix force gradient
    y_base = sphere_target_curve(
        d=d,
        sphere=sphere,
        target=grating,
        quantity="force_gradient",
        calib=calib,
    )

    # If CSV is magnitude-only, apply sign convention explicitly.
    # For Casimir attraction, the signed force gradient is negative in our convention.
    if cfg.sign == "magnitude":
        y_exp_signed = -np.abs(y_exp)
    elif cfg.sign == "signed":
        y_exp_signed = y_exp
    else:
        raise ValueError(f"Unsupported data_convention.sign: {cfg.sign!r}")

    enforce_sentinels(
        sentinels_path=SENTINELS_PATH,
        d=d,
        y_curve=y_exp_signed,
        curve_label="Chan 2008 experimental force gradient",
    )

    # Deviation factor η = exp / baseline
    eta = deviation_factor(y_exp_signed, y_base)

    ensure_dir(FIG_DIR)
    ensure_dir(OUT_DIR)

    # 1) Overlay plot
    plt.figure()
    plt.plot(d, y_exp_signed, marker="o", linestyle="none", label="Experiment (digitized)")
    plt.plot(d, y_base, linestyle="-", label="Level A baseline (PFA-mix)")
    plt.xlabel("Separation d (m)")
    plt.ylabel("Force gradient dF/dd (N/m)")
    plt.title("Chan et al. (2008): Force gradient vs separation")
    plt.legend()
    plt.tight_layout()
    out_overlay = FIG_DIR / "chan_2008_overlay.png"
    plt.savefig(out_overlay, dpi=200)
    plt.close()

    # 2) Deviation factor plot
    plt.figure()
    plt.plot(d, eta, marker="o", linestyle="-")
    plt.axhline(1.0, linewidth=1.0)
    plt.xlabel("Separation d (m)")
    plt.ylabel("Deviation factor η = (exp) / (baseline)")
    plt.title("Chan et al. (2008): Deviation factor vs separation")
    plt.tight_layout()
    out_eta = FIG_DIR / "chan_2008_eta.png"
    plt.savefig(out_eta, dpi=200)
    plt.close()

    run_csv = OUT_DIR / "chan_2008_run.csv"
    write_run_csv_chan(run_csv, d, y_exp_signed, y_base, eta)

    banner(
        "CHAN2008_OUT",
        fig_overlay=str(out_overlay.relative_to(REPO_ROOT)),
        fig_eta=str(out_eta.relative_to(REPO_ROOT)),
        csv=str(run_csv.relative_to(REPO_ROOT)),
    )

    print("Wrote:")
    print(f"  {FIG_DIR / 'chan_2008_overlay.png'}")
    print(f"  {FIG_DIR / 'chan_2008_eta.png'}")
    print(f"  {run_csv}")


if __name__ == "__main__":
    main()