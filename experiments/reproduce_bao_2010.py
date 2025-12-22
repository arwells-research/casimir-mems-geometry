#!/usr/bin/env python3
"""
experiments/reproduce_bao_2010.py

Reproduce Bao et al., PRL 105 (2010): sphere vs shallow rectangular corrugations.

This Bao anchor is stored as a *dimensionless ratio curve* (measured / PFA)
because it is the most robust, paper-aligned observable for Level A validation.

Inputs:
- data/raw/bao_2010/metadata.yaml
- data/raw/bao_2010/digitized_curve.csv

CSV columns (required):
- separation_m
- eta_measured_over_pfa   (dimensionless)

Outputs (written locally, git-ignored):
- figures/derived/bao_2010_eta.png             (primary: ratio)
- figures/derived/bao_2010_baseline_dFdd.png   (diagnostic: baseline comparison, mix vs bao)
- outputs/bao_2010_run.csv                     (audit table)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
import yaml

from casimir_mems.types import Sphere, RectTrenchGrating, Calibration
from casimir_mems.levelA.interface import sphere_target_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "bao_2010"
SENTINELS_PATH = DATA_DIR / "sentinels.yaml"
FIG_DIR = REPO_ROOT / "figures" / "derived"
OUT_DIR = REPO_ROOT / "outputs"

def banner(tag: str, **kv: object) -> None:
    items = " ".join([f"{k}={v}" for k, v in kv.items()])
    print(f"[{tag}] {items}".rstrip())

@dataclass(frozen=True)
class Bao2010Config:
    sphere_R_m: float
    p_m: float
    w_m: float          # l2 (bottom length) stored as trench_width_w_m
    l1_m: float         # l1 (top length) stored as trench_top_width_l1_m
    h_m: float
    d0_m: float


def load_config(path: Path) -> Bao2010Config:
    cfg = yaml.safe_load(path.read_text())
    geom = cfg.get("geometry", {})
    cal = cfg.get("calibration", {})

    return Bao2010Config(
        sphere_R_m=float(geom["sphere_R_m"]),
        p_m=float(geom["trench_period_p_m"]),
        w_m=float(geom["trench_width_w_m"]),
        l1_m=float(geom["trench_top_width_l1_m"]),
        h_m=float(geom["trench_depth_h_m"]),
        d0_m=float(cal.get("d0_m", 0.0)),
    )


def load_digitized_ratio_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a dimensionless ratio curve eta(d) = measured / PFA.

    Required headers:
    - separation_m
    - eta_measured_over_pfa
    """
    d_list: list[float] = []
    eta_list: list[float] = []

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
                    e_idx = header.index("eta_measured_over_pfa")
                except ValueError as e:
                    raise ValueError(
                        "digitized_curve.csv must have headers: separation_m, eta_measured_over_pfa"
                    ) from e
                continue

            d_list.append(float(row[d_idx]))
            eta_list.append(float(row[e_idx]))

    d = np.asarray(d_list, dtype=float)
    eta = np.asarray(eta_list, dtype=float)

    if d.ndim != 1 or eta.ndim != 1 or d.size != eta.size or d.size < 2:
        raise ValueError("Digitized ratio data must be 1D arrays with >= 2 rows.")
    if np.any(d <= 0.0):
        raise ValueError("All separations must be > 0.")
    if np.any(eta <= 0.0):
        raise ValueError("All eta values must be > 0 (dimensionless ratio).")

    order = np.argsort(d)
    return d[order], eta[order]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def interp_1d(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """Linear interpolation on a strictly increasing x-grid."""
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

        checks: list[bool] = []
        if tol_abs is not None:
            checks.append(err <= tol_abs)
        if tol_rel is not None:
            denom = max(abs(y_exp), 1e-30)
            checks.append(err <= tol_rel * denom)

        if not checks:
            raise ValueError(f"Sentinel {name} must specify tol_abs and/or tol_rel")

        if not all(checks):
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


def write_run_csv_bao_ratio(
    path: Path,
    d: np.ndarray,
    eta: np.ndarray,
    baseline_dFdd_mix: np.ndarray,
    baseline_dFdd_bao: np.ndarray,
) -> None:
    """
    Audit output for Bao 2010 reproduction (ratio mode).

    Columns:
      separation_m,
      eta_measured_over_pfa,
      baseline_force_gradient_mix_N_per_m,
      baseline_force_gradient_bao_N_per_m
    """
    if not (d.size == eta.size == baseline_dFdd_mix.size == baseline_dFdd_bao.size):
        raise ValueError("Run CSV write: array sizes do not match.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "separation_m",
            "eta_measured_over_pfa",
            "baseline_force_gradient_mix_N_per_m",
            "baseline_force_gradient_bao_N_per_m",
        ])
        for di, ei, y_mix, y_bao in zip(d, eta, baseline_dFdd_mix, baseline_dFdd_bao):
            w.writerow([f"{di:.12e}", f"{ei:.12e}", f"{y_mix:.12e}", f"{y_bao:.12e}"])


def main() -> None:
    meta_path = DATA_DIR / "metadata.yaml"
    csv_path = DATA_DIR / "digitized_curve.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing digitized curve: {csv_path}")

    cfg = load_config(meta_path)
    d, eta = load_digitized_ratio_csv(csv_path)

    banner(
        "BAO2010",
        y_kind="ratio_over_pfa",
        y_column="eta_measured_over_pfa",
        d_min_nm=f"{float(np.min(d))*1e9:.1f}",
        d_max_nm=f"{float(np.max(d))*1e9:.1f}",
        n_points=int(d.size),
        p_m=f"{cfg.p_m:.3e}",
        l1_m=f"{cfg.l1_m:.3e}",
        l2_m=f"{cfg.w_m:.3e}",
        h_m=f"{cfg.h_m:.3e}",
        R_m=f"{cfg.sphere_R_m:.3e}",
        d0_m=f"{cfg.d0_m:.3e}",
    )

    enforce_sentinels(
        sentinels_path=SENTINELS_PATH,
        d=d,
        y_curve=eta,
        curve_label="Bao 2010 digitized ratio eta (measured/PFA)",
    )

    sphere = Sphere(R=cfg.sphere_R_m)
    grating = RectTrenchGrating(
        p=cfg.p_m,
        w=cfg.w_m,                    # legacy, kept
        h=cfg.h_m,
        top_width_m=cfg.l1_m,          # l1
        bottom_width_m=cfg.w_m,        # l2
    )
    calib = Calibration(d0=cfg.d0_m)

    # Diagnostics only: compute both baseline definitions
    dFdd_base_mix = sphere_target_curve(
        d=d, sphere=sphere, target=grating,
        quantity="force_gradient", calib=calib,
        method="pfa_mix",
    )

    dFdd_base_bao = sphere_target_curve(
        d=d, sphere=sphere, target=grating,
        quantity="force_gradient", calib=calib,
        method="pfa_bao",
    )

    ensure_dir(FIG_DIR)
    ensure_dir(OUT_DIR)

    run_csv = OUT_DIR / "bao_2010_run.csv"
    write_run_csv_bao_ratio(run_csv, d, eta, dFdd_base_mix, dFdd_base_bao)

    out_eta_png = FIG_DIR / "bao_2010_eta.png"
    out_base_png = FIG_DIR / "bao_2010_baseline_dFdd.png"

    # Primary plot: eta vs separation
    plt.figure()
    plt.plot(d, eta, marker="o", linestyle="-", label="Digitized: η = measured / PFA")
    plt.axhline(1.0, linewidth=1.0, label="η = 1 (PFA)")
    plt.xlabel("Separation d (m)")
    plt.ylabel("Deviation factor η (dimensionless)")
    plt.title("Bao et al. (2010): Deviation factor vs separation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_eta_png, dpi=200)
    plt.close()

    # Diagnostic plot: baseline comparison (mix vs Bao-PFA)
    plt.figure()
    plt.plot(d, dFdd_base_mix, linestyle="-", label="Baseline dF/dd: PFA-mix (top+bottom)")
    plt.plot(d, dFdd_base_bao, linestyle="--", label="Baseline dF/dd: Bao-PFA (includes sidewalls)")
    plt.xlabel("Separation d (m)")
    plt.ylabel("Baseline dF/dd (N/m)")
    plt.title("Bao et al. (2010): Baseline force-gradient comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_base_png, dpi=200)
    plt.close()

    banner(
        "BAO2010_OUT",
        fig_eta=str(out_eta_png.relative_to(REPO_ROOT)),
        fig_baseline=str(out_base_png.relative_to(REPO_ROOT)),
        csv=str(run_csv.relative_to(REPO_ROOT)),
    )

    print("Bao 2010 ratio reproduction complete.")
    print("d range (nm):", float(np.min(d) * 1e9), "to", float(np.max(d) * 1e9))
    print("eta stats: min =", float(np.min(eta)), "max =", float(np.max(eta)), "median =", float(np.median(eta)))
    print("Wrote:")
    print(f"  {out_eta_png}")
    print(f"  {out_base_png}")
    print(f"  {run_csv}")    


if __name__ == "__main__":
    main()