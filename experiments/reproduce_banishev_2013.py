#!/usr/bin/env python3
"""
experiments/reproduce_banishev_2013.py

Banishev et al., PRL 110, 250403 (2013): sinusoidally corrugated plate vs sphere.

This benchmark uses the *dimensionless normalized force-gradient ratio*:

  eta(d) = F'_exp / F'_{PFA,flat}

Plot contract (Level B target):
- Black dots: digitized experimental eta(d)
- Blue dashed: Level A (local PFA averaging only)
- Red solid: Level B (Derivative Expansion correction included; ideal-metal DE here)
- Shading:
    - Light warning zone: k*d > kd_warn
    - Dark refusal zone:  k*d > kd_refuse

Inputs:
- data/raw/banishev_2013/metadata.yaml
- data/raw/banishev_2013/digitized_curve.csv
- data/raw/banishev_2013/sentinels.yaml

Outputs (written locally, git-ignored):
- figures/derived/banishev_2013_overlay.png
- outputs/banishev_2013_run.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math

import numpy as np
import matplotlib.pyplot as plt
import yaml

# Level A plane-plane model (ideal metal)
from casimir_mems.levelA.plane_plane import P_pp_ideal
from casimir_mems.levelB import LEVELB_BACKEND_ID
from casimir_mems.levelB.validity import compute_validity_sinusoid
from casimir_mems.levelB.derivative_expansion import eta_levelB_DE_ideal, BETA_EM_IDEAL

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "banishev_2013"
FIG_DIR = REPO_ROOT / "figures" / "derived"
OUT_DIR = REPO_ROOT / "outputs"

def banner(tag: str, **kv: object) -> None:
    items = " ".join([f"{k}={v}" for k, v in kv.items()])
    print(f"[{tag}] {items}".rstrip())


@dataclass(frozen=True)
class Banishev2013Config:
    sphere_R_m: float
    period_m: float
    amplitude_m: float
    kd_warn: float
    kd_refuse: float
    y_column: str


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> Banishev2013Config:
    meta = yaml.safe_load(path.read_text())

    geom = meta.get("geometry", {}) or {}
    thr = meta.get("thresholds", {}) or {}
    dc = meta.get("data_convention", {}) or {}

    # Back-compat: if older metadata used kd_max, treat it as kd_warn unless explicit warn/refuse provided.
    kd_warn = float(thr.get("kd_warn", thr.get("kd_max", 0.1)))
    kd_refuse = float(thr.get("kd_refuse", float("inf")))

    return Banishev2013Config(
        sphere_R_m=float(geom["sphere_R_m"]),
        period_m=float(geom["corrugation_period_m"]),
        amplitude_m=float(geom["corrugation_amplitude_m"]),
        kd_warn=kd_warn,
        kd_refuse=kd_refuse,
        y_column=str(dc.get("y_column", "eta_measured_over_pfa")),
    )


def load_digitized_csv(path: Path, *, y_column: str) -> tuple[np.ndarray, np.ndarray]:
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
    if np.any(y <= 0.0):
        raise ValueError("All eta values must be > 0 (dimensionless ratio).")

    order = np.argsort(d)
    return d[order], y[order]


def interp_1d(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    if x0 < float(x[0]) or x0 > float(x[-1]):
        raise ValueError(
            f"Sentinel x0={x0:.3e} outside digitized range [{x[0]:.3e}, {x[-1]:.3e}]"
        )
    return float(np.interp(x0, x, y))


def enforce_sentinels(*, sentinels_path: Path, d: np.ndarray, y_curve: np.ndarray, curve_label: str) -> None:
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


def eta_level_a_local_pfa_avg(d: np.ndarray, *, a: float, period: float, samples: int = 4096) -> np.ndarray:
    """
    Level A for corrugations:
      eta_A(d) = <P_pp(d + a sin(kx))> / P_pp(d)

    Sphere radius cancels in the ratio (PFA-flat baseline is 2πR P_pp(d)).
    """
    if samples < 256:
        raise ValueError("samples too low; use >= 256.")
    k = 2.0 * math.pi / period
    x = np.linspace(0.0, 2.0 * math.pi / k, samples, endpoint=False)  # one period
    sin = np.sin(k * x)

    P0 = P_pp_ideal(d)
    out = np.empty_like(d, dtype=float)
    for i, di in enumerate(d):
        H = di + a * sin
        if np.any(H <= 0.0):
            raise ValueError(f"Non-positive local gap encountered at d={di:.3e} with amplitude a={a:.3e}")
        Pav = float(np.mean(P_pp_ideal(H)))
        out[i] = Pav / float(P0[i])
    return out

def write_run_csv(
    path: Path,
    *,
    d: np.ndarray,
    eta_exp: np.ndarray,
    eta_a: np.ndarray,
    eta_b: np.ndarray,
    levelB_backend_id: str,
    kd: np.ndarray,
    ak: np.ndarray,
    ak2: np.ndarray,
    validity_score: np.ndarray,
    warned: np.ndarray,
    refused: np.ndarray,
) -> None:
    """
    Write the Banishev 2013 run-audit CSV.

    Contract-bound columns (B0):
      separation_m        float
      eta_exp             float
      eta_levelA          float
      eta_levelB          float
      levelB_backend_id   string  (frozen identifier, e.g. de_ideal_v0)
      kd                  float
      ak                  float   (slope amplitude proxy: a*k)
      ak2                 float   (ak^2)
      validity_score      float   (geometry confidence index in [0,1])
      warned              int     (kd > kd_warn)
      refused             int     (kd > kd_refuse)
    """
    if not (
        d.size
        == eta_exp.size
        == eta_a.size
        == eta_b.size
        == kd.size
        == ak.size
        == ak2.size
        == validity_score.size
        == warned.size
        == refused.size
    ):
        raise ValueError("Run CSV write: array sizes do not match.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "separation_m",
                "eta_exp",
                "eta_levelA",
                "eta_levelB",
                "levelB_backend_id",
                "kd",
                "ak",
                "ak2",
                "validity_score",
                "warned",
                "refused",
            ]
        )
        for di, ee, ea, eb, kdi, aki, ak2i, vsi, wn, rf in zip(
            d, eta_exp, eta_a, eta_b, kd, ak, ak2, validity_score, warned, refused
        ):
            w.writerow(
                [
                    f"{di:.12e}",
                    f"{ee:.12e}",
                    f"{ea:.12e}",
                    f"{eb:.12e}",
                    str(levelB_backend_id),
                    f"{kdi:.12e}",
                    f"{aki:.12e}",
                    f"{ak2i:.12e}",
                    f"{vsi:.12e}",
                    int(bool(wn)),
                    int(bool(rf)),
                ]
            )

def levelB_diagnostics_b0(
    d: np.ndarray,
    *,
    amplitude_m: float,
    period_m: float,
    kd_warn: float,
    kd_refuse: float,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Level B (B0) per-point diagnostics for the sinusoid geometry.

    Returns (all arrays same shape as d):
      backend_id, kd, ak, ak2, validity_score, warned, refused

    Notes:
    - This is purely geometric diagnostics + contract ID.
    - The *numerical* Level B curve eta_B is computed elsewhere (via casimir_mems.levelB).
    """
    from casimir_mems.levelB import LEVELB_BACKEND_ID
    from casimir_mems.levelB.validity import compute_validity_sinusoid

    diag = compute_validity_sinusoid(
        d,
        period=period_m,
        amplitude=amplitude_m,
        kd_warn=kd_warn,
        kd_refuse=kd_refuse,
    )

    return (
        LEVELB_BACKEND_ID,
        diag.kd,
        diag.ak,
        diag.ak2,
        diag.validity_score,
        diag.warned,
        diag.refused,
    )


def main() -> None:
    meta_path = DATA_DIR / "metadata.yaml"
    csv_path = DATA_DIR / "digitized_curve.csv"
    sent_path = DATA_DIR / "sentinels.yaml"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing digitized curve: {csv_path}")
    if not sent_path.exists():
        raise FileNotFoundError(f"Missing sentinels: {sent_path}")

    cfg = load_config(meta_path)
    d, eta_exp = load_digitized_csv(csv_path, y_column=cfg.y_column)

    banner(
        "BANISHEV2013",
        period_m=f"{cfg.period_m:.3e}",
        a_m=f"{cfg.amplitude_m:.3e}",
        kd_warn=cfg.kd_warn,
        kd_refuse=cfg.kd_refuse,
        y_column=cfg.y_column,
        n_points=int(d.size),
    )

    enforce_sentinels(
        sentinels_path=sent_path,
        d=d,
        y_curve=eta_exp,
        curve_label="Banishev 2013 digitized ratio eta (measured/PFA_flat)",
    )

    levelB_backend_id, kd, ak, ak2, validity_score, warned, refused = levelB_diagnostics_b0(
        d,
        amplitude_m=cfg.amplitude_m,
        period_m=cfg.period_m,
        kd_warn=cfg.kd_warn,
        kd_refuse=cfg.kd_refuse,
    )

    eta_a = eta_level_a_local_pfa_avg(d, a=cfg.amplitude_m, period=cfg.period_m, samples=4096)
    eta_b = eta_levelB_DE_ideal(d, a=cfg.amplitude_m, period=cfg.period_m, samples=4096)

    ensure_dir(FIG_DIR)
    ensure_dir(OUT_DIR)

    out_csv = OUT_DIR / "banishev_2013_run.csv"
    write_run_csv(
        path=out_csv,
        d=d,
        eta_exp=eta_exp,
        eta_a=eta_a,
        eta_b=eta_b,
        levelB_backend_id=levelB_backend_id,
        kd=kd,
        ak=ak,
        ak2=ak2,
        validity_score=validity_score,
        warned=warned,
        refused=refused,
    )

    # Plot
    plt.figure()

    # Light warning shading
    if np.any(warned):
        d_warn = float(d[np.argmax(warned)])
        plt.axvspan(
            d_warn,
            float(d[-1]),
            alpha=0.15,
            label=f"Warning: k d > {cfg.kd_warn:g}",
        )

    # Dark refusal shading (if any)
    if np.any(refused):
        d_ref = float(d[np.argmax(refused)])
        plt.axvspan(
            d_ref,
            float(d[-1]),
            alpha=0.30,
            label=f"Refusal: k d > {cfg.kd_refuse:g}",
        )

    plt.plot(d, eta_exp, marker="o", linestyle="none", label="Experiment (digitized)")
    plt.plot(d, eta_a, linestyle="--", label="Level A: local PFA averaging")
    plt.plot(d, eta_b, linestyle="-", label=f"Level B: {levelB_backend_id} (beta={BETA_EM_IDEAL:.6g})")

    plt.xlabel("Separation d (m)")
    plt.ylabel("Normalized force gradient η = F'exp / F'PFA_flat")
    plt.title("Banishev et al. (2013): Corrugation ratio vs separation")
    plt.legend()
    plt.tight_layout()

    out_png = FIG_DIR / "banishev_2013_overlay.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    banner(
        "BANISHEV2013_OUT",
        fig=str(out_png.relative_to(REPO_ROOT)),
        csv=str(out_csv.relative_to(REPO_ROOT)),
    )

    print("Wrote:")
    print(f"  {out_png}")
    print(f"  {out_csv}")
    print("beta_EM_ideal:", BETA_EM_IDEAL)
    print("kd range:", float(np.min(kd)), "to", float(np.max(kd)))
    print("Warned points:", int(np.sum(warned)), "of", int(d.size), f"(kd_warn={cfg.kd_warn:g})")
    print("Refused points:", int(np.sum(refused)), "of", int(d.size), f"(kd_refuse={cfg.kd_refuse:g})")


if __name__ == "__main__":
    main()