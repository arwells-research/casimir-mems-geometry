# Casimir/MEMS Geometry Benchmarks — Leveled Validation (A/B/C)

**Author:** A. R. Wells  
**Affiliation:** Dual-Frame Research Group  
**License:** MIT  
**Repository:** `arwells-research/casimir-mems-geometry`  
**Status:** Level A complete; Level B diagnostics complete; Level C milestones C1–C14 complete (C15 next)

---

## Overview

This repository provides a clean, reproducible Python pipeline for validating Casimir-force modeling against canonical **geometry-dependent** experimental benchmarks.

The project is organized as a leveled contract:

Level C progress through C14 establishes internal, synthetic geometry invariants;
C15 is reserved for confrontation with real experimental datasets.

- **Level A (Local PFA / area averaging)**: fast, robust baselines; correct limits; used for sanity checks and audit curves.
- **Level B (Derivative Expansion / DE)**: smooth-profile corrections to PFA, with explicit validity/refusal logic.
- **Level C (Scattering-inspired / explicitly nonlocal geometry methods)**: required when Level B validity is violated.

The goal is to build **engineer-grade trust** by enforcing explicit scope boundaries and refusing to over-interpret models outside their regime.

---

## What This Repository Does

✔ Stores digitized benchmark curves with a strict CSV+YAML contract  
✔ Generates overlays and run-audit CSVs for every benchmark  
✔ Enforces “sentinel” points to lock digitization integrity  
✔ Implements leveled theory curves (A/B) and shows refusal zones

🚫 Not a general-purpose Casimir solver  
🚫 No claim of validity outside each level’s contract  
🚫 No hidden parameter fitting inside reproduction scripts

---

## Repository Structure

NOTE: The tree below is shown in plain indented format to avoid nested fenced blocks.

This repository tracks **source + raw data**. Generated outputs (CSVs/PNGs) are produced locally and should be ignored (see `.gitignore`).

Directory naming convention (Level C):

- `levelc_c0_*` — canonical synthetic reference cases (infrastructure lock)
- `levelc_c1–c11_*` — sinusoidal and scattering-minimization baselines and stress tests
- `levelc_c12–c14_*` — dual-harmonic geometry-invariant cases (φ = 0°, π/2, π)
- External experiment folders (`chan_2008`, `bao_2010`, `banishev_2013`) are prepared
  datasets and are not yet executed as Level C cases (reserved for C15).
<!-- end list -->
  
    casimir-mems-geometry/
    ├── pyproject.toml
    ├── LICENSE
    ├── CITATION.cff
    ├── README.md
    ├── data/
    │   ├── raw/
    │   │   ├── chan_2008/
    │   │   │   ├── digitized_curve.csv
    │   │   │   ├── metadata.yaml
    │   │   │   └── sentinels.yaml
    │   │   ├── bao_2010/
    │   │   │   ├── digitized_curve.csv
    │   │   │   ├── metadata.yaml
    │   │   │   └── sentinels.yaml
    │   │   ├── banishev_2013/
    │   │   │   ├── digitized_curve.csv
    │   │   │   ├── metadata.yaml
    │   │   │   └── sentinels.yaml
    |   │   ├── levelc_c0_sinusoid/              # metadata.yaml
    |   │   ├── levelc_c1–c3_sinusoid_*/         # metadata.yaml
    |   │   ├── levelc_c4–c7_idealpert_*/        # metadata.yaml
    |   │   ├── levelc_c8–c11_scattmin_*/        # metadata.yaml
    |   │   └── levelc_c12–c14_dualharm_phi*/    # metadata.yaml
    │   └── derived/              # generated locally (ignored)
    ├── figures/
    │   └── derived/              # generated locally (ignored)
    ├── outputs/                  # generated locally (ignored)
    ├── experiments/
    │   ├── reproduce_chan_2008.py
    │   ├── reproduce_bao_2010.py
    │   ├── reproduce_banishev_2013.py
    │   └── reproduce_levelC_c0_sinusoid.py    
    ├── src/
    │   └── casimir_mems/
    │       ├── __init__.py
    │       ├── types.py
    │       ├── levelA/
    │       │   ├── __init__.py
    │       │   ├── plane_plane.py
    │       │   ├── pfa.py
    │       │   ├── grating_pfa_mix.py
    │       │   ├── interface.py
    │       │   └── diagnostics.py
    │       ├── levelB/
    │       │   ├── __init__.py
    │       │   ├── validity.py
    │       │   ├── derivative_expansion.py
    │       │   └── sinusoid.py
    │       └── levelC/
    │           ├── __init__.py
    │           ├── interface.py
    │           ├── convergence.py
    │           └── toy_scattering.py    
    ├── tests/
    └── tools/
        └── repro.sh

Generated locally (not committed):

    figures/derived/*.png
    outputs/*_run.csv
    data/derived/*

---

## Data Contract (Raw Benchmarks)

Each benchmark folder under `data/raw/<paper>/` contains:

1. `metadata.yaml`  
   - geometry parameters (SI units)
   - calibration offset (optional)
   - `data_convention` describing y-axis meaning and column mapping
   - refusal thresholds for Level B (per-benchmark)

2. `digitized_curve.csv`  
   - first column: `separation_m` (meters)
   - second column: a single y-column named by `metadata.yaml:data_convention:y_column`

3. `sentinels.yaml`  
   - a small set of locked reference points that must match by interpolation within tolerance
   - prevents silent digitization drift or accidental CSV edits

---

## Installation

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    pip install -e .[dev]

---

## Reproduce all benchmark figures

Run:

    ./tools/repro.sh

Artifacts are written locally to:

- `figures/derived/`
- `outputs/`

---

## Level A (PFA / Local Averaging)

Level A is intentionally conservative:

- correct asymptotic limits
- fast to evaluate
- used as the baseline and audit reference

Level A does **not** claim nonlocal geometric correlations.  
Its role is to establish a trusted geometric baseline against which
beyond-PFA effects can be diagnosed.

---

## Level B (Derivative Expansion)

Level B adds smooth-profile corrections to PFA via the Derivative Expansion (DE).
It explicitly includes **validity boundaries** and **refusal logic**.

When DE validity conditions are violated:
- plots must show a shaded warning/refusal region, and
- results beyond that boundary must be interpreted as *diagnostic only*.

The primary Level B benchmark is **Banishev et al. (2013)**, with the following
plot contract:

- **Black dots**: digitized experimental ratio η(d)
- **Blue dashed**: Level A result η_A(d)
- **Red solid**: Level B result η_B(d) (DE correction)
- **Grey band**: DE validity warning/refusal zone  
  (thresholds specified in `metadata.yaml`)

---

### Level B scope: Derivative Expansion as a diagnostic bridge

**Level B in this repository is intentionally implemented for *ideal conductors*.**

It is a *bridge* between:
- **Level A** (local PFA averaging), and
- **Level C** (full scattering / material-specific treatments),

not a destination in itself.

For the Banishev et al. (2013) experiment (Au at 300 K), discrepancies between
the Level B curve and the experimental data are **expected** and treated as
an *informative feature*, not a defect.

Interpretation of the ladder:

- **Level A** → isolates purely local geometric averaging  
  (typically overestimates corrugation enhancement).
- **Level B** → adds the leading beyond-PFA geometric correction in a controlled,
  ideal-metal environment.
- **Residual gap to experiment** → highlights finite conductivity, temperature,
  and other real-material effects that are **outside Level B by design** and
  are reserved for Level C.

This separation preserves the project’s purpose as a **geometry-first modeling
toolkit**, rather than evolving into a Lifshitz optical-data solver.

---

## Level C (Nonlocal / Scattering Geometry)

**Status:** Level C synthetic validation complete through **C14**.
C15 (external experimental confrontation) is the next milestone.

Level C addresses regimes where **explicit nonlocal geometric mode coupling**
invalidates both local averaging (Level A) and weakly nonlocal expansions
(Level B / Derivative Expansion).

Unlike Levels A and B, Level C explicitly resolves **lateral mode structure**
introduced by surface geometry and enforces convergence through a controlled
mode sweep.

---

### Entry criterion

Level C is required when one or more of the following conditions hold:

- the characteristic geometric wavevector satisfies `k d ≳ O(1)` over the
  region of interest,
- lateral mode coupling induced by surface structure is no longer perturbative
  in a local sense,
- or Level B validity thresholds are exceeded and geometry-only corrections
  fail to control the discrepancy with experiment.

This criterion is **geometric**, not material-specific.

---

### Level C backends

Level C supports multiple backends, each with a clearly defined role and scope.

| Backend ID | Purpose | Physical status |
|-----------|--------|-----------------|
| `toy_scattering_v0` | **Harness backend** used to validate Level C infrastructure (mode sweep, refusal masking, CSV schema, plotting, sentinels) | **Not physical** |
| `ideal_perturb_scattering_v0` | Ideal-metal, geometry-resolving **perturbative** scattering backend with explicit lateral mode summation | **Physical (idealized, perturbative)** |
| `ideal_scattering_minimal_v0` | Ideal-metal, **non-perturbative, mode-coupling** scattering backend using a minimal log-det formulation | **Physical (idealized, geometry-only)** |

**Contract freeze (v0).**  
The backends listed above define the **Level C v0 contract**.  
Any semantic or numerical change requires a **new backend ID**; existing backend
IDs and their benchmark cases are immutable.

The toy backend is retained permanently as a **contract and infrastructure test**.

The ideal-perturb backend defines the **weakly nonlocal / perturbative** Level C regime.

The ideal-scattering-minimal backend defines the **fully nonlocal geometric regime**, where
local gap expansions (Levels A/B) catastrophically fail but bounded, mode-coupled
scattering behavior remains well defined.

### How to add a new Level C backend

Level C backends are treated as audited, contract-bound components.
Adding a new backend is a structured process intended to prevent silent
numerical, convergence, or semantic regressions.

This section defines the required steps and invariants.

---

#### 1. Implement the backend class

Create a new backend file under:

src/casimir_mems/levelC/

The backend must:

- Subclass LevelCBackend
- Expose a stable backend identifier via a backend_id field
- Implement the required geometry method(s), e.g. compute_sinusoid
- Return a LevelCResult containing:
  - eta_levelC (numpy array)
  - n_modes (numpy integer array)
  - converged (boolean array)

The backend must NOT:

- Mutate global state
- Perform I/O
- Infer or short-circuit convergence internally
- Depend on Level A or Level B internals beyond declared inputs

Convergence decisions are handled exclusively by the ModeSweep controller.

Backend identifier conventions:

ideal_perturb_scattering_v0
ideal_perturb_scattering_v1
finite_cond_scattering_v0

Backend IDs are part of the reproducibility contract and MUST remain stable.

---

#### 2. Declare backend usage in metadata

Every Level C case must explicitly declare its backend in metadata.yaml.

Required fields:

levelC.backend  
levelC.convergence.n_modes_start  
levelC.convergence.n_modes_step  
levelC.convergence.n_modes_max  
levelC.convergence.tol  

Example (conceptual, not a literal code block):

levelC:
  backend: ideal_perturb_scattering_v0
  convergence:
    n_modes_start: 8
    n_modes_step: 8
    n_modes_max: 256
    tol: 1.0e-5

The backend is required to respect these parameters.

---

#### 3. Wire the backend into the experiment harness

Update the Level C factory logic (e.g. in reproduce_levelC_case.py) to:

- Map the backend ID string to the backend class
- Pass only documented constructor arguments
- Fail loudly on unknown backend IDs

The harness must emit banners identifying:

- The backend ID
- Sweep range and step
- Convergence tolerance

Required banners:

[LEVELC_C0] or [LEVELC_CASE]  
[LEVELC_C0_OUT] or [LEVELC_CASE_OUT]

These banners are treated as audit sentinels by tools/repro.sh.

---

#### 4. Add a new Level C benchmark ID if outputs change

If a backend produces outputs that differ numerically from any frozen
benchmark (e.g. C0), a new Level C benchmark ID MUST be introduced.

Examples:

C4: ideal-perturb canonical geometry  
C5: ideal-perturb amplitude variant  
C6: ideal-perturb stress test  

Frozen benchmarks must never change in-place.

---

#### 5. Verify with repro.sh

After adding the backend, tools/repro.sh must pass with:

- CSV schema validation
- Refusal masking invariants
- Mode sweep sentinel checks
- Core call-count sentinels
- Backend ID sentinels
- Banner presence checks

A backend is not considered valid until repro.sh passes cleanly.

---

#### 6. Update documentation

Finally, update the Level C README section to:

- Name the new backend
- State its mathematical intent and limitations
- Identify which benchmark IDs correspond to it

Documentation updates are required for any new backend ID.

---

### Ideal-perturb scattering backend (`ideal_perturb_scattering_v0`)

This backend implements a **geometry-resolving, ideal-metal scattering model**
with the following defining characteristics:

- **Exact lateral geometry resolution** via explicit mode summation
- **Ideal conductor** assumption (no finite conductivity, dissipation, or temperature)
- **Perturbative in surface amplitude**, but **nonlocal in lateral structure**
- **Controlled numerical convergence** via mode-count sweeping

The computed observable `η_C(d)` represents a **nonlocal geometric correction**
to the Level B result, obtained by summing lateral modes up to `n_modes`
and enforcing convergence under a declared tolerance.

---

### Minimal scattering backend (`ideal_scattering_minimal_v0`)

This backend implements a **minimal, non-perturbative scattering-like geometry model**
intended to capture **explicit lateral mode coupling** without relying on local
gap expansions or amplitude-squared perturbation theory.

It is deliberately **not** a full Lifshitz or electromagnetic scattering solver.
Instead, it provides a controlled geometric baseline that remains well behaved
in regimes where Levels A and B fail.


#### Defining characteristics

- **Explicit lateral mode coupling** via a truncated Fourier basis
- **Non-perturbative in surface amplitude**
- **Ideal conductor assumption**
- **Single representative evanescent scale** per separation
- **Log-determinant energy proxy** enforcing bounded response

#### Observable definition

For each separation \( d \), the backend computes a dimensionless geometric
observable:

\[
\eta_C(d) \equiv \frac{-\log\det\!\left(I - P(d)\,C(d)\,P(d)\right)}
{-\log\det\!\left(I - P(d)^2\right)},
\]

where:

- \( P(d) \) is a diagonal propagation operator encoding evanescent decay,
- \( C(d) \) is a geometry-induced lateral mode-coupling matrix derived from the
  Fourier spectrum of the surface profile.

This normalization guarantees:

- **Flat-surface limit:** \( a \to 0 \Rightarrow \eta_C(d) \to 1 \)
- **Bounded response:** no divergence as \( a/d \to O(1) \)

#### Regime behavior

The backend exhibits the following qualitative behavior across the benchmark suite:

- **Small amplitude (C9):** agreement with Levels A and B
- **Moderate amplitude (C8):** reduced enhancement relative to DE
- **Large amplitude (C10):** remains bounded while DE and local averaging diverge
  catastrophically

This behavior is **intentional** and reflects the breakdown of local gap-based
descriptions, not a numerical artifact.

#### What this backend demonstrates

- That **explicit geometric mode coupling alone** is sufficient to suppress
  unphysical divergences seen in local and perturbative models
- That a geometry-first scattering formulation can remain stable even when
  \( a \sim d \)
- That the transition from DE-valid to DE-invalid regimes is *structural*, not
  merely quantitative

#### What this backend does not include

- Frequency integration
- Material dispersion or dissipation
- Finite conductivity or temperature
- Full electromagnetic polarization structure

Those effects may be layered on in future backends, but **must not obscure
pure geometric behavior**, which is the organizing principle of Level C.

---

#### Reduction consistency

Backends are required to demonstrate **appropriate reduction behavior** consistent
with their declared physical scope:

- **Small-amplitude reduction**: as surface amplitude → 0, `η_C(d) → 1`
- **Long-wavelength consistency**: for sufficiently small `k d`, results must be
  compatible with weakly nonlocal behavior (e.g. Level B), within the limits of
  the backend’s construction
- **Sweep consistency**: results must converge under increasing `n_modes`
  according to the metadata-declared sweep parameters

For perturbative backends (e.g. `ideal_perturb_scattering_v0`), explicit reduction
to the Level B curve in the low-`k d` limit is required and enforced by the
C4–C7 benchmark suite.

---

### Canonical and benchmark cases

- **C0**  
  Canonical synthetic sinusoid benchmark using `toy_scattering_v0`.  
  **Frozen**. Any change requires a new benchmark ID.

- **C1–C3**  
  Synthetic stress cases (amplitude and sweep stress) using the toy backend.
  These validate Level C infrastructure and convergence logic.

- **C4–C7**  
  Canonical and stress cases using `ideal_perturb_scattering_v0`.  
  These form the **acceptance gate** for a real Level C backend.

- **C8–C11**  
  Canonical, amplitude-variant, and sweep-stress cases using
  `ideal_scattering_minimal_v0`.
  These cases are not acceptance gates for physical accuracy; they are
  invariants for boundedness, convergence, and reduction behavior.

  These benchmarks demonstrate:
  - bounded non-perturbative behavior at large amplitude,
  - correct reduction to unity at small amplitude,
  - stable convergence under mode-count sweeping,
  - explicit failure of local and DE-based models in the same regime.

- **C12–C14**  
  Dual-harmonic phase-sensitivity cases using `ideal_scattering_minimal_v0`.  
  These complete the internal Level C synthetic validation suite.
 
  **Scope boundary:**  
  Level C results reported here through **C14** are synthetic and internal.
  No Level C claim based on real experimental data is made until **C15**
  is completed and explicitly documented.

- **C15 (planned)**  
  Execution of Level C backends against real experimental datasets
  (`chan_2008`, `bao_2010`, `banishev_2013`) under locked protocols.

Cases **C12–C14** demonstrate that Level C resolves **geometry beyond local or
derivative-expansion descriptions**, even in a fully synthetic, idealized
setting.

These cases use a **dual-harmonic 1D surface profile** with identical amplitudes
and wavelength, differing **only in the relative phase** of the second harmonic:

- **C12**: φ₂ = 0  
- **C13**: φ₂ = π/2  
- **C14**: φ₂ = π  

Across all three cases:

- **Level A (local averaging)** and **Level B (ideal-metal DE)** curves are
  **identical by construction**.
- The same **refusal mask** applies (36/41 trusted points in all cases).
- Only **Level C** changes with φ₂.

On trusted points, the minimal log-det / mode-mixing backend
(`ideal_scattering_minimal_v0`) produces structured, phase-dependent shifts in
the normalized observable η_C(d):

- max |Δη_C|(φ₂ = π/2 − 0) ≈ **4.13×10⁻³**  
- max |Δη_C|(φ₂ = π − 0) ≈ **8.26×10⁻³** (≈ 2× larger)  
- max |Δη_C|(φ₂ = π − π/2) ≈ **4.13×10⁻³**

This behavior demonstrates that **explicit nonlocal geometric mode coupling**
alone is sufficient to produce **spectrum- and phase-sensitive responses** that
are invisible to local or weakly nonlocal models.

These cases therefore establish a **Level C geometry claim**. This claim is about **Level C capability**, not physical correctness of the underlying interaction model.:

> η_C(d) depends on the **spectral content and relative phase** of the surface
> profile, even when lower-level models are held fixed.

This claim is **synthetic, ideal-conductor, and not electromagnetically complete
by design**, but it is **bounded, reproducible, and contract-stable**, and it
marks the point at which Level C becomes a genuine **geometry-resolving
framework**, rather than an extension of sinusoidal benchmarks.

A Level C backend is considered valid only if **all declared cases reproduce
under `tools/repro.sh` with locked CSV schema, refusal masking, and convergence
sentinels**.

---

### Convergence and sweep contract

Each Level C case declares its convergence behavior in metadata:

```yaml
levelC:
  backend: <backend_id>
  convergence:
    n_modes_start: <int>
    n_modes_step: <int>
    n_modes_max: <int>
    tol: <float>
```

---

#### `ideal_perturb_scattering_v0` (physics-motivated backend)

- **Purpose:** first real Level C geometry backend
- **Status:** approximate, ideal-metal, amplitude² response
- **Captures:**
  - Known leading-order scattering asymptotics for sinusoidal corrugations
    in the ideal-conductor limit
  - Correct behavior in both limits:
    - long wavelength (`λ ≫ d`)
    - short wavelength (`λ ≪ d`)
  - Explicit mode saturation under increasing diffraction order
- **Does not yet include:**
  - full scattering-matrix frequency integration
  - finite conductivity, temperature, or material dispersion

This backend represents the **first physics-motivated implementation of Level C**
and serves as a bridge between analytic expansions (Levels A/B) and full
numerical scattering solvers.

**Scope note:** The mathematical form below applies **only** to the
`ideal_perturb_scattering_v0` backend and does not define a general
Level C requirement.

##### Mathematical contract (exact)

For a 1D sinusoidal corrugation of wavelength \(\lambda\) and amplitude \(a\),
with mean separation \(d\) and lateral wavevector \(k = 2\pi/\lambda\), this
backend defines the Level C normalized observable \(\eta_C(d)\) as:

\[
\eta_C(d) \equiv \eta_B(d)\,\bigl[1 + \Delta_C(d)\bigr].
\]

The correction \(\Delta_C(d)\) is a *controlled, amplitude-squared*, nonlocal
geometry term:

\[
\Delta_C(d) \equiv \alpha_0\,\Bigl(\frac{a}{d}\Bigr)^2\,W(kd)\,S_N(kd),
\]

where:

- \(\alpha_0\) is a dimensionless calibration constant (metadata parameter).
- \(W(kd)\) is a dimensionless crossover weight that enforces the correct
  asymptotic behavior:

\[
W(kd) \equiv \frac{kd}{1+kd}.
\]

- \(S_N(kd)\) is a monotone, bounded mode-saturation factor that converges as the
  diffraction-order cutoff \(N\) increases:

\[
S_N(kd) \equiv \sum_{m=1}^{N} \exp\!\bigl[-\gamma(kd)\,m\bigr],
\qquad
\gamma(kd) \equiv 0.25 + (kd)^{p}.
\]

The exponent \(p\) is a nonnegative metadata parameter (`decay_p`).

**Convergence contract:** for fixed \(d\), \(a\), and \(\lambda\), the sequence
\(\eta_C^{(N)}(d)\) produced by increasing \(N\) must be Cauchy under the mode
sweep tolerance (`levelC.convergence.tol`).

**Refusal contract:** when \(kd > kd\_refuse\) (from metadata thresholds), the
pipeline must hard-mark rows as refused: `converged=0` and `n_modes=0`.

---

### Benchmark summary (historical / minimal)

This section summarizes the original minimal benchmark ladder; the complete and
current benchmark set is defined in the **Canonical and benchmark cases**
section above.

- **C0** — canonical sinusoid, frozen reference (toy backend)
- **C1–C3** — stress and sweep variants (toy backend)
- **C4** — first sinusoid benchmark using `ideal_perturb_scattering_v0`

Each benchmark is fully specified by its case directory metadata and is
independently reproducible via `tools/repro.sh`.

---

## Why local and DE models fail at large amplitude  
### (and why Level C remains bounded)

The Level C stress cases **C8–C11** are intentionally constructed to expose a
regime where **local averaging (Level A)** and **Derivative Expansion (Level B)**
cease to be reliable geometric descriptions.

In these cases, the corrugation amplitude \(a\) is no longer small compared to
the mean separation \(d\). While the surfaces remain non-intersecting, the
geometry develops **deep troughs and steep gradients** that violate the core
assumptions underlying both local and perturbative models.

**How to read the stress-test plots (C8–C10):**
- The **horizontal axis** is separation \(d\).
- The **vertical axis** is the normalized observable \(\eta(d)\).
- **Level A and Level B curves** illustrate what local and DE-based models predict
  when extrapolated beyond their formal validity.
- The **Level C curve** shows the result of explicit nonlocal geometric coupling.
- Shaded regions mark geometric warning and refusal thresholds based on \(k d\).

### Failure modes of Levels A and B

- **Level A (local averaging)** samples the force or force gradient at shifted
  separations \(d + h(x)\) and averages the result. When \(a \sim d\), this
  procedure disproportionately weights near-contact regions. Because no
  saturation mechanism is present, the resulting enhancement grows rapidly and
  can become arbitrarily large.

- **Level B (Derivative Expansion)** improves upon local averaging by including
  gradient corrections, but it remains an **asymptotic expansion** valid only for
  smooth, slowly varying profiles. Once \(k d \gtrsim O(1)\) and surface slopes
  become large, the expansion is no longer controlled. The dramatic growth seen
  in C10 is therefore an expected structural failure, not a numerical instability.

These divergences are not bugs in the implementation. They are a direct
consequence of applying local or weakly nonlocal approximations outside their
domain of validity.

### Why Level C behaves differently

The **minimal scattering backend** (`ideal_scattering_minimal_v0`) does not rely
on local gap sampling or gradient expansions. Instead, it resolves the geometry
through **explicit lateral mode coupling** in a truncated Fourier basis and
combines contributions through a bounded log-determinant construction.

As a result:

- In **C9 (small amplitude)**, Level C agrees with Levels A and B, as expected.
- In **C8 (moderate amplitude)**, Level C predicts reduced enhancement relative
  to DE, indicating the onset of nonlocal saturation.
- In **C10 (large amplitude stress)**, Level C remains finite and well behaved
  while Levels A and B diverge catastrophically.

This behavior is intentional and diagnostic. It demonstrates that the explosive
growth seen in local and DE-based models at large amplitude is a **structural
artifact of their assumptions**, not an unavoidable consequence of geometry
itself.

### Scope and interpretation

The bounded response of Level C does **not** imply quantitative accuracy with
respect to experiment, nor does it replace full electromagnetic scattering
treatments. What it shows is that **explicit nonlocal geometric coupling alone**
is sufficient to suppress unphysical divergences arising from local descriptions.

Level C therefore serves a critical diagnostic role within the leveled framework:
it identifies when geometry-driven nonlocality becomes dominant and when local
or perturbative models must be abandoned, independent of material realism or
frequency-domain detail.

---

### What Level C is (by design)

- A geometry-resolving framework beyond PFA and DE
- A bridge from analytic expansions to controlled numerical treatments
- A tool for diagnosing *which geometric features* drive nonlocal behavior

---

### What Level C is not

- Not a general-purpose Casimir solver
- Not a Lifshitz optical-data management framework
- Not a guarantee of quantitative agreement with experiment
- Not a replacement for Levels A or B

Levels A and B remain required baselines and must continue to be shown
alongside any Level C result.

---

### Materials and temperature

Material properties (finite conductivity, temperature, dissipation) may be
introduced in Level C **only insofar as they do not obscure geometric effects**.

Geometry remains the primary organizing principle.

Material realism beyond this diagnostic role may be explored separately and
does not redefine Level C.

---

### Validation requirements

Any Level C implementation must demonstrate:

1. **Reduction consistency**  
   Recovery of the appropriate lower-level behavior (Level A or Level B)
   in the smooth or low-`k d` limit, consistent with the backend’s declared scope.

2. **Geometric sensitivity**  
   Explicit dependence on surface profile beyond local height statistics.

3. **Scope transparency**  
   Clear identification of geometric, numerical, and physical assumptions.

---

### Role within the leveled framework

The leveled structure is cumulative, not competitive:

- **Level A** establishes trusted local baselines.
- **Level B** identifies where geometry-driven nonlocality begins.
- **Level C** resolves that nonlocality explicitly.

Together, these levels provide a controlled pathway from intuition to
nonlocal geometric physics without overextending any single model.

---

## License

MIT. See `LICENSE`.
