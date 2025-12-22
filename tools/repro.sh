#!/usr/bin/env bash
set -euo pipefail

check_levelc_case() {
  local case_tag="$1" csv_path="$2" meta_path="$3"

  [[ -f "$csv_path" ]] || { echo "ERROR: missing csv: $csv_path" >&2; exit 1; }
  [[ -f "$meta_path" ]] || { echo "ERROR: missing metadata: $meta_path" >&2; exit 1; }
  echo "Checking: [$case_tag] CSV schema..."
  local hdr
  hdr="$(head -n 1 "$csv_path" | tr -d '\r')"
  local want
  want="separation_m,kd_value,eta_levelA,eta_levelB,eta_levelC,n_modes,converged,levelC_backend"
  [[ "$hdr" == "$want" ]] || {
    echo "ERROR: [$case_tag] CSV header mismatch" >&2
    echo "  got : $hdr" >&2
    echo "  want: $want" >&2
    exit 1
  }
  echo "OK: [$case_tag] CSV schema matches locked header."

  echo "Checking: [$case_tag] row invariants..."
  $PYTHON - <<PY
import csv
from pathlib import Path
import yaml

case_tag = "$case_tag"
csv_path = Path("$csv_path")
meta_path = Path("$meta_path")

rows = list(csv.DictReader(csv_path.open(newline="")))
meta = yaml.safe_load(meta_path.read_text())
geom = (meta.get("geometry") or {})
want_rows = int(geom.get("n_points", 41))
assert len(rows) == want_rows, f"Expected {want_rows} rows, got {len(rows)}"

lvlc = (meta.get("levelC") or {})
want_backend = str(lvlc.get("backend", "")).strip()
assert want_backend, "metadata.yaml missing levelC.backend"

thr = (meta.get("thresholds") or {})
kd_refuse = float(thr.get("kd_refuse", float("inf")))
assert kd_refuse < float("inf"), "metadata.yaml missing thresholds.kd_refuse"

# refused rows must be (n_modes=0, converged=0)
bad = []
for i, r in enumerate(rows):
    kd = float(r["kd_value"])
    nm = int(r["n_modes"])
    cv = int(r["converged"])
    if kd > kd_refuse:
        if cv != 0 or nm != 0:
            bad.append((i, kd, nm, cv))
if bad:
    raise SystemExit(f"Refusal invariant failed for kd>{kd_refuse:g}: examples={bad[:8]}")

# backend id must match in every row
bad_backend = [i for i, r in enumerate(rows) if (r.get("levelC_backend", "").strip() != want_backend)]
if bad_backend:
    raise SystemExit(f"levelC_backend sentinel mismatch: expected {want_backend!r}; bad rows: {bad_backend[:10]}")

# --------
# Sweep sentinel
# --------
# Accept any of:
# (A) nontrivial sweep: >=2 distinct nonzero n_modes among non-refused rows
# (B) uniform-step convergence: all non-refused rows converged at the same single n_modes
#     that is within [start, max] and aligned to step.
#
# NOTE: We intentionally DO NOT require a "trivial sweep at start" special-case,
# because it is a strict subset of (B) (nm_only == start and aligned).
nm_set = set()
nonref = []
for r in rows:
    kd = float(r["kd_value"])
    if kd <= kd_refuse:
        nm = int(r["n_modes"])
        cv = int(r["converged"])
        nonref.append((nm, cv))
        if nm > 0:
            nm_set.add(nm)

conv = (lvlc.get("convergence") or {})
n_modes_start = int(conv.get("n_modes_start", 0))
n_modes_step  = int(conv.get("n_modes_step", 0))
n_modes_max   = int(conv.get("n_modes_max", 0))
assert n_modes_start >= 1, "metadata.yaml missing levelC.convergence.n_modes_start"
assert n_modes_step >= 1, "metadata.yaml missing levelC.convergence.n_modes_step"
assert n_modes_max >= n_modes_start, "metadata.yaml missing/invalid levelC.convergence.n_modes_max"

if not nonref:
    raise SystemExit("Mode sweep sentinel failed: no non-refused rows to validate")

if len(nm_set) >= 2:
    print(f"OK: [{case_tag}] sweep sentinel: nontrivial sweep; distinct n_modes (non-refused) = {sorted(nm_set)[:10]} ...")
else:
    nm_only = next(iter(nm_set)) if nm_set else 0
    all_conv = all(cv == 1 for (_, cv) in nonref)
    aligned = (
        (nm_only >= n_modes_start)
        and (nm_only <= n_modes_max)
        and ((nm_only - n_modes_start) % n_modes_step == 0)
    )
    if all_conv and aligned and nm_only > 0:
        print(
            f"OK: [{case_tag}] sweep sentinel: uniform-step convergence at n_modes={nm_only} "
            f"(start={n_modes_start}, step={n_modes_step}, max={n_modes_max})."
        )
    else:
        raise SystemExit(
            "Mode sweep sentinel failed: expected >=2 distinct nonzero n_modes "
            "OR uniform-step convergence at a single aligned n_modes. "
            f"Got distinct n_modes={sorted(nm_set)}; examples(nonref)={nonref[:8]}"
        )
PY
}

check_levelc_core_calls() {
  local case_tag="$1" want_min_calls="$2"
  echo "Checking: [$case_tag] core call-count sentinel..."

  local line calls
  line="$(grep -F "[$case_tag] Level C core calls:" "$LOG_PATH" | tail -n 1 || true)"
  [[ -n "$line" ]] || { echo "ERROR: missing [$case_tag] core call-count line in $LOG_PATH" >&2; exit 1; }

  calls="$(echo "$line" | sed -E 's/.*: ([0-9-]+).*/\1/')"
  [[ -n "$calls" ]] || { echo "ERROR: could not parse call count from: $line" >&2; exit 1; }

  [[ "$calls" -ge "$want_min_calls" ]] || {
    echo "ERROR: [$case_tag] core call-count sentinel failed (calls=$calls; expected >=$want_min_calls)" >&2
    exit 1
  }
  echo "OK: [$case_tag] core call-count sentinel passed (calls=$calls)."
}

LOG_DIR="outputs"
LOG_PATH="$LOG_DIR/repro_last.log"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$repo_root/src"

echo "Repo root: $repo_root"
echo "Python: $PYTHON"
echo "PYTHONPATH: $PYTHONPATH"

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_PATH") 2>&1

# Dependency sanity (fast fail)
$PYTHON - <<'PY'
import importlib
for m in ("numpy","matplotlib","yaml"):
    importlib.import_module(m)
print("OK: deps present (numpy, matplotlib, pyyaml).")
PY

# -------------------------------
# Level A
# -------------------------------

echo "Running: Chan 2008 reproduction..."
$PYTHON experiments/reproduce_chan_2008.py

echo "Running: Bao 2010 reproduction..."
$PYTHON experiments/reproduce_bao_2010.py

# -------------------------------
# Level C harness (toy backend)
# -------------------------------

echo "Running: Level C canonical benchmark (C0 sinusoid) ..."
$PYTHON experiments/reproduce_levelC_c0_sinusoid.py
check_levelc_core_calls "LEVELC_C0" 2

echo "Running: Level C benchmark (C1 amplitude variant) ..."
$PYTHON experiments/reproduce_levelC_case.py --case-dir data/raw/levelc_c1_sinusoid_a01
check_levelc_core_calls "LEVELC_C1" 2

echo "Running: Level C benchmark (C2 amplitude stress) ..."
$PYTHON experiments/reproduce_levelC_case.py --case-dir data/raw/levelc_c2_sinusoid_a03
check_levelc_core_calls "LEVELC_C2" 2

echo "Running: Level C benchmark (C3 sweep stress) ..."
$PYTHON experiments/reproduce_levelC_case.py --case-dir data/raw/levelc_c3_sinusoid_sweep256
check_levelc_core_calls "LEVELC_C3" 2

# -------------------------------
# Level C real backend (ideal perturbative scattering)
# -------------------------------

echo "Running: Level C benchmark (C4 ideal-perturb canonical) ..."
$PYTHON experiments/reproduce_levelC_case.py --case-dir data/raw/levelc_c4_idealpert_canonical
check_levelc_core_calls "LEVELC_C4" 2

echo "Running: Level C benchmark (C5 ideal-perturb amplitude variant) ..."
$PYTHON experiments/reproduce_levelC_case.py --case-dir data/raw/levelc_c5_idealpert_a01
check_levelc_core_calls "LEVELC_C5" 2

echo "Running: Level C benchmark (C6 ideal-perturb amplitude stress) ..."
$PYTHON experiments/reproduce_levelC_case.py --case-dir data/raw/levelc_c6_idealpert_a03
check_levelc_core_calls "LEVELC_C6" 2

echo "Running: Level C benchmark (C7 ideal-perturb sweep stress) ..."
$PYTHON experiments/reproduce_levelC_case.py --case-dir data/raw/levelc_c7_idealpert_sweep256
check_levelc_core_calls "LEVELC_C7" 2

echo "Running: Level C benchmark (C8 scatt-min canonical) ..."
$PYTHON experiments/reproduce_levelC_case.py --case-dir data/raw/levelc_c8_scattmin_canonical
check_levelc_core_calls "LEVELC_C8" 2

echo "Running: Level C benchmark (C9 scatt-min amplitude variant) ..."
$PYTHON experiments/reproduce_levelC_case.py --case-dir data/raw/levelc_c9_scattmin_a01
check_levelc_core_calls "LEVELC_C9" 2

echo "Running: Level C benchmark (C10 scatt-min amplitude stress) ..."
$PYTHON experiments/reproduce_levelC_case.py --case-dir data/raw/levelc_c10_scattmin_a03
check_levelc_core_calls "LEVELC_C10" 2

echo "Running: Level C benchmark (C11 scatt-min sweep stress) ..."
$PYTHON experiments/reproduce_levelC_case.py --case-dir data/raw/levelc_c11_scattmin_sweep256
check_levelc_core_calls "LEVELC_C11" 2

# -------------------------------
# Level B
# -------------------------------

echo "Running: Banishev 2013 reproduction..."
$PYTHON experiments/reproduce_banishev_2013.py

# -------------------------------
# Artifact existence checks
# -------------------------------

echo "OK: Reproduction scripts completed."
echo "Figures: figures/derived/"
echo "Outputs: outputs/"

expected=(
  "figures/derived/chan_2008_overlay.png"
  "figures/derived/chan_2008_eta.png"
  "figures/derived/bao_2010_eta.png"
  "figures/derived/bao_2010_baseline_dFdd.png"
  "figures/derived/banishev_2013_overlay.png"

  "figures/derived/levelC_c0_overlay.png"
  "figures/derived/levelc_c1_sinusoid_a01_overlay.png"
  "figures/derived/levelc_c2_sinusoid_a03_overlay.png"
  "figures/derived/levelc_c3_sinusoid_sweep256_overlay.png"

  "figures/derived/levelc_c4_idealpert_canonical_overlay.png"
  "figures/derived/levelc_c5_idealpert_a01_overlay.png"
  "figures/derived/levelc_c6_idealpert_a03_overlay.png"
  "figures/derived/levelc_c7_idealpert_sweep256_overlay.png"

  "outputs/chan_2008_run.csv"
  "outputs/bao_2010_run.csv"
  "outputs/banishev_2013_run.csv"

  "outputs/levelC_test_run.csv"
  "outputs/levelc_c1_sinusoid_a01_run.csv"
  "outputs/levelc_c2_sinusoid_a03_run.csv"
  "outputs/levelc_c3_sinusoid_sweep256_run.csv"

  "outputs/levelc_c4_idealpert_canonical_run.csv"
  "outputs/levelc_c5_idealpert_a01_run.csv"
  "outputs/levelc_c6_idealpert_a03_run.csv"
  "outputs/levelc_c7_idealpert_sweep256_run.csv"

  "figures/derived/levelc_c8_scattmin_canonical_overlay.png"
  "figures/derived/levelc_c9_scattmin_a01_overlay.png"
  "figures/derived/levelc_c10_scattmin_a03_overlay.png"
  "figures/derived/levelc_c11_scattmin_sweep256_overlay.png"
  "outputs/levelc_c8_scattmin_canonical_run.csv"
  "outputs/levelc_c9_scattmin_a01_run.csv"
  "outputs/levelc_c10_scattmin_a03_run.csv"
  "outputs/levelc_c11_scattmin_sweep256_run.csv"  
)

for f in "${expected[@]}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing expected output: $f" >&2; exit 1; }
  echo "OK: found $f"
done

# -------------------------------
# Level C CSV schema + invariants
# -------------------------------

echo "Checking: outputs/levelC_test_run.csv schema..."
check_levelc_case "LEVELC_C0" "outputs/levelC_test_run.csv" "data/raw/levelc_c0_sinusoid/metadata.yaml"

echo "Checking: outputs/levelc_c1_sinusoid_a01_run.csv schema..."
check_levelc_case "LEVELC_C1" "outputs/levelc_c1_sinusoid_a01_run.csv" "data/raw/levelc_c1_sinusoid_a01/metadata.yaml"

echo "Checking: outputs/levelc_c2_sinusoid_a03_run.csv schema..."
check_levelc_case "LEVELC_C2" "outputs/levelc_c2_sinusoid_a03_run.csv" "data/raw/levelc_c2_sinusoid_a03/metadata.yaml"

echo "Checking: outputs/levelc_c3_sinusoid_sweep256_run.csv schema..."
check_levelc_case "LEVELC_C3" "outputs/levelc_c3_sinusoid_sweep256_run.csv" "data/raw/levelc_c3_sinusoid_sweep256/metadata.yaml"

echo "Checking: outputs/levelc_c4_idealpert_canonical_run.csv schema..."
check_levelc_case "LEVELC_C4" "outputs/levelc_c4_idealpert_canonical_run.csv" "data/raw/levelc_c4_idealpert_canonical/metadata.yaml"

echo "Checking: outputs/levelc_c5_idealpert_a01_run.csv schema..."
check_levelc_case "LEVELC_C5" "outputs/levelc_c5_idealpert_a01_run.csv" "data/raw/levelc_c5_idealpert_a01/metadata.yaml"

echo "Checking: outputs/levelc_c6_idealpert_a03_run.csv schema..."
check_levelc_case "LEVELC_C6" "outputs/levelc_c6_idealpert_a03_run.csv" "data/raw/levelc_c6_idealpert_a03/metadata.yaml"

echo "Checking: outputs/levelc_c7_idealpert_sweep256_run.csv schema..."
check_levelc_case "LEVELC_C7" "outputs/levelc_c7_idealpert_sweep256_run.csv" "data/raw/levelc_c7_idealpert_sweep256/metadata.yaml"

echo "Checking: outputs/levelc_c8_scattmin_canonical_run.csv schema..."
check_levelc_case "LEVELC_C8" "outputs/levelc_c8_scattmin_canonical_run.csv" "data/raw/levelc_c8_scattmin_canonical/metadata.yaml"

echo "Checking: outputs/levelc_c9_scattmin_a01_run.csv schema..."
check_levelc_case "LEVELC_C9" "outputs/levelc_c9_scattmin_a01_run.csv" "data/raw/levelc_c9_scattmin_a01/metadata.yaml"

echo "Checking: outputs/levelc_c10_scattmin_a03_run.csv schema..."
check_levelc_case "LEVELC_C10" "outputs/levelc_c10_scattmin_a03_run.csv" "data/raw/levelc_c10_scattmin_a03/metadata.yaml"

echo "Checking: outputs/levelc_c11_scattmin_sweep256_run.csv schema..."
check_levelc_case "LEVELC_C11" "outputs/levelc_c11_scattmin_sweep256_run.csv" "data/raw/levelc_c11_scattmin_sweep256/metadata.yaml"

echo "OK: all expected outputs verified."

# -------------------------------
# Stdout banner sentinels
# -------------------------------

echo "Checking: stdout banner sentinels..."
need_tags=(
  "[CHAN2008]"
  "[BAO2010]"
  "[BANISHEV2013]"
  "[LEVELC_C0]"
  "[LEVELC_C0_OUT]"
  "[LEVELC_CASE]"
  "[LEVELC_CASE_OUT]"
)
for t in "${need_tags[@]}"; do
  grep -Fq "$t" "$LOG_PATH" || {
    echo "ERROR: missing expected banner in stdout: $t" >&2
    echo "  log: $LOG_PATH" >&2
    exit 1
  }
done

# Case-ID sentinels (lock case selection + dispatch)
cids=(
  levelc_c1_sinusoid_a01
  levelc_c2_sinusoid_a03
  levelc_c3_sinusoid_sweep256
  levelc_c4_idealpert_canonical
  levelc_c5_idealpert_a01
  levelc_c6_idealpert_a03
  levelc_c7_idealpert_sweep256
  levelc_c8_scattmin_canonical
  levelc_c9_scattmin_a01
  levelc_c10_scattmin_a03
  levelc_c11_scattmin_sweep256
)
for cid in "${cids[@]}"; do
  grep -Fq "case_id=${cid}" "$LOG_PATH" || {
    echo "ERROR: missing expected Level C case_id in stdout: ${cid}" >&2
    echo "  log: $LOG_PATH" >&2
    exit 1
  }
done

echo "OK: stdout banner sentinels present. (log: $LOG_PATH)"