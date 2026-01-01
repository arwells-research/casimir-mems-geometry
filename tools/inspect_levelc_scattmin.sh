#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import pandas as pd
import numpy as np
from pathlib import Path

paths = [
  "outputs/levelc_c8_scattmin_canonical_run.csv",
  "outputs/levelc_c9_scattmin_a01_run.csv",
  "outputs/levelc_c10_scattmin_a03_run.csv",
  "outputs/levelc_c11_scattmin_sweep256_run.csv",
]

def summarize(p):
    df = pd.read_csv(p)
    x = df["eta_levelC"].to_numpy(dtype=float)
    nm = df["n_modes"].to_numpy(dtype=int)
    conv = df["converged"].to_numpy(dtype=int)
    finite = np.isfinite(x)
    return {
        "file": Path(p).name,
        "rows": len(df),
        "finite_frac": float(finite.mean()),
        "eta_min": float(np.nanmin(x)),
        "eta_max": float(np.nanmax(x)),
        "eta_med": float(np.nanmedian(x)),
        "modes_unique": sorted(set(int(v) for v in nm if v > 0))[:10],
        "conv_frac": float((conv==1).mean()),
    }

rows = [summarize(p) for p in paths]
for r in rows:
    print(
        f"{r['file']}: rows={r['rows']} finite={r['finite_frac']:.3f} "
        f"conv={r['conv_frac']:.3f} eta[min/med/max]={r['eta_min']:.3g}/{r['eta_med']:.3g}/{r['eta_max']:.3g} "
        f"modes={r['modes_unique']}"
    )
PY