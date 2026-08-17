# Fea / research / fenics — FEM ultrasound wave-sim (run & maintain guide)

FEniCS/DOLFINx finite-element research for DarkVision ultrasound simulation. Pure Python,
containerised — **independent of the C++/CUDA `Fea` build** (this is not `Fea/python/`, which is
reserved for the compiled `feapy` pybind11 module).

> **The "why" — asks, constraints, frozen scenario, plan, current state, and the full results
> tables — lives in the branch README: `C:/code/readme/rnd-nima-fea-README.md`. Read that first.**
> **This file is the "how": environment, layout, what each script does, and how to reproduce
> the headline result.**

> **READ-ONLY BOUNDARIES — see [`Fea/AGENTS.md`](../../AGENTS.md) before touching either.**
> `F:\DarkVision Dropbox\...` and every non-DVCode repo (including the `kwave` submodule)
> are **read-only**: read and list, never write/create/delete/move — not even a temp or probe file.
> Writes go under `C:\code\DVCode\` (outside the submodule), the branch README, or the scratchpad.

---

## 0. Headline result (2026-08-12)

Our FEM channel data and the research team's k-Wave channel data pushed through **their own
beamformer**, so the forward solver is the only difference. **Both valid steering angles:**

| metric (true value) | k-Wave +20 | **FEM +20** | k-Wave -20 | **FEM -20** |
|---|---|---|---|---|
| crack x error (38.25 mm) | 0.413 | **0.165** | 0.332 | **0.084** |
| notch extent (4.0 mm) | 3.23 (-19%) | **3.73 (-6.8%)** | 6.33 (+58%) | **3.85 (-3.8%)** |
| crack / clutter RMS | 22.8 dB | **24.0** | 23.1 dB | **24.0** |
| crack / clutter p95 | 16.6 dB | **17.0** | **17.0 dB** | 16.9 (tie) |
| crack / worst clutter | 10.3 dB | **12.2** | 9.8 dB | **10.4** |

**FEM better in 29/30 robustness checks** across the two angles (15/15 at +20, 14/15 at -20).
The one honest tie is p95 contrast at -20 deg, 0.1 dB behind.

**The -20 deg numbers were PREDICTED before the solve ran** (23-25 dB, 3.7-4.2 mm extent) and
landed inside the range, so the bandwidth mechanism in §5 is out-of-sample tested, not fitted.

**Our sizing is angle-consistent and theirs is not:** 3% spread across +-20 deg against their
96%. Since both datasets pass through the same imaging chain, that asymmetry is in their forward
data. For an inspection tool this matters more than any single dB figure - real cracks are never
conveniently on-axis.

**WHY we win: numerical bandwidth, NOT geometry.** The C4 controlled experiment
(`repro/c4_staircase.py`) staircased the curved ID at k-Wave's own 50 um and measured only
**+0.61 dB** of extra clutter - real in direction, far too small to explain the win. So do not
say "we win because the mesh conforms". The defendable mechanism is that we resolve the pulse
BANDWIDTH (see section 5), which was predicted before the -20 deg solve and confirmed. The
untested candidate for the remainder is k-Wave's crack VOID, filled with 500 m/s material at
only 2.5 points per wavelength - a much cruder approximation than staircasing a smooth arc.

See the branch README §4.7-4.8 for what is and is not defendable: sizing is converged, contrast
is a **lower bound** (still improving with refinement), and position is at the imaging grid's
pixel limit, so quote it as "sub-pixel" rather than as a ratio.

**0 deg is not a third angle.** TT-T is a half-skip SHEAR mode and normal incidence generates
almost no mode conversion, so neither solver produces a meaningful image there.

**The configuration matters: degree 4 and `--scale 0.8`.** Degree 3 at scale 1.0 LOSES to k-Wave
(12.2 dB) because it under-resolves the pulse bandwidth. See §5.

---

## 1. Environment

DOLFINx needs PETSc + MPI, which do not install on native Windows, so everything runs in a Linux
container via Docker Desktop (WSL2). Nothing is installed on Windows itself.

- **`dvfenics:bf`** (default) = `dolfinx/dolfinx:stable` (DOLFINx **0.11.0**, Python 3.12, PETSc,
  MPI, UFL/FFCx/Basix) + our extras + the five packages the research team's `beamformer` needs.
- **`dvfenics:latest`** is the older image, kept as a rollback.

```powershell
cd C:\code\DVCode\Fea\research\fenics
./run.ps1 python3 toys/check_dolfinx.py     # env smoke test -> DOLFINx 0.11.0 + OK
./run.ps1 python3 lib/bf_loader.py          # prove their beamformer loads
./run.ps1                                   # interactive bash shell
```

`run.ps1` mounts this folder read-write at `/work` and their beamformer **read-only** at `/opt/bf`.
Override the image with `$env:DVFENICS_IMAGE`.

**Gotchas that will waste your afternoon:**

| Symptom | Cause / fix |
|---|---|
| `working directory 'C:/Program Files/Git/work' is invalid` | Git Bash rewrites container paths. Use PowerShell, or prefix `MSYS_NO_PATHCONV=1`. |
| PowerShell reports failure on a successful run | It wraps native **stderr** in error records; their package emits harmless `SyntaxWarning`s. Don't use `2>&1`; or run via Bash. |
| `ModuleNotFoundError: beamformer` | Submodule not checked out: `git submodule update --init --depth 1 ../kwave` |
| `NotImplementedError: engine must be 'cuda'` | Receive-side Fermat TOF is GPU-only. Use the CPU `ray` path (the default in `lib/tt_t_image.py`). |
| A background solve looks dead (empty log, nothing in `docker ps`) | **It probably isn't.** Solves print nothing during the time loop, and `docker ps` has returned nothing while a container was up. Use `docker ps -a`. |

---

## 2. Reproducing the headline result

Data needed from Dropbox (read-only) is extracted first, then two solves and one comparison.
Total ~5 h of compute. Run from this folder.

```powershell
# 1. Extract the research team's runs (their raw data stays untouched).
#    Write the .npz UNDER results/ - it is gitignored there, it is inside the /work mount so
#    no extra -v is needed, and unlike a session scratch directory it does not disappear.
./run.ps1 python3 tools/extract_kwave_case.py `
    --run "data/kwave_runs/new_sims/2026-08-11 12-46-37 Simulation_ODnotch4mm_20" `
    --out results/kwave_cases/kwave_odnotch4mm_20.npz

# 2. Build the conforming mesh at the working resolution  (~10 s)
./run.ps1 python3 mesh/ili_mesh.py --quad --scale 0.8

# 3. Forward solve: degree 4 on that mesh  (~2.4 h, 163678 steps)
./run.ps1 python3 repro/ili_forward.py --angle 20 --degree 4 `
    --mesh results/ili_mesh/ili_mesh_s0p8.msh --tag deg4_s0p8_p20deg

# 4. Both datasets through THEIR beamformer -> the head-to-head table + figure
./run.ps1 python3 repro/compare_images.py --angle 20 `
    --ours results/ili_forward/channel_data_deg4_s0p8_p20deg.npz `
    --theirs results/kwave_cases/kwave_odnotch4mm_20.npz

# 5. Confirm the claims survive every reasonable analysis choice
./run.ps1 python3 repro/metric_robustness.py --angle 20
```

### The demonstration figures

```powershell
# wavefield animation: add --snapshots to any solve, then render (degree 3 is enough here)
./run.ps1 python3 repro/ili_forward.py --angle 20 --degree 3 `
    --mesh results/ili_mesh/ili_mesh.msh --tag snap_p20deg `
    --snapshots 240 --snap-window "18,46"
./run.ps1 python3 viz/wavefield_gif.py --in results/ili_forward/wavefield_snap_p20deg.npz `
    --stride 3 --fps 10 --colors 48 --smooth 2 --stills "22,27,31,35"

# mesh conformity figure (needs the two healthy triangle meshes; no solve required)
./run.ps1 python3 mesh/ili_mesh.py --no-notch --no-plot
./run.ps1 python3 mesh/ili_mesh.py --no-notch --no-plot --staircase
./run.ps1 python3 viz/mesh_zoom.py
```

---

## 3. Layout

```
fenics/
├─ Dockerfile, requirements.txt, run.ps1     infra
├─ lib/
│   ├─ bf_loader.py       loads THEIR beamformer without executing its __init__.py
│   └─ tt_t_image.py      the TT-T imaging chain + image_metrics(), shared by all comparisons
├─ mesh/
│   └─ ili_mesh.py        conforming gmsh mesh: exact ID/OD arcs + notch as a real void
├─ tools/
│   ├─ extract_kwave_case.py   their *_workspace.mat -> compact .npz (refuses to write to F:)
│   └─ probe_bf_deps.py        how the beamformer dependency list was derived
├─ validation/             V&V against EXACT solutions
│   ├─ bf_roundtrip.py         gate: their beamformer on their data == their image
│   ├─ zoeppritz.py            angle-resolved fluid-solid mode conversion
│   └─ cavity_scattering.py    defect scattering vs the exact Pao & Mow series
├─ repro/                  the ILI simulation and the comparison
│   ├─ ili_forward.py          THE FORWARD SOLVE (--snapshots for the animations)
│   ├─ analyze_forward.py      arrival times vs analytic ToF
│   ├─ compare_images.py       FEM vs k-Wave through one identical beamformer
│   ├─ compare_rf.py           raw channel-data diff, no beamformer
│   ├─ metric_robustness.py    do the claims survive different analysis choices?
│   ├─ c4_staircase.py         C4: our solver vs ITSELF, staircased vs conforming ID
│   └─ ili_gate.py, ili_angled.py, ili_beamform.py, ili_realistic.py, animate_gate.py,
│      render_*.py             earlier work; ili_realistic (corrosion) is LEGACY
├─ viz/                    Phase D: the demonstration figures and animations
│   ├─ wavefield_gif.py        D1: the beam mode-converting at the ID and hitting the notch
│   └─ mesh_zoom.py            D4: what "conforming" means, drawn from the real meshes
├─ toys/                   capability evidence, one physics concept at a time
├─ data/
│   ├─ kwave_ili/PROVENANCE.md   submodule + Dropbox pointers, staleness lesson
│   └─ kwave_runs/               their 3 new runs (GITIGNORED, 832 MB) + the driver .m
└─ results/<name>/         outputs. Figures tracked; .npz/.msh/.h5/.xdmf gitignored.
```

---

## 4. Scripts

### The comparison chain

| Script | What it does | Status |
|---|---|---|
| `lib/bf_loader.py` | Imports their `beamformer` imaging path by pre-registering the package in `sys.modules`, so `__init__.py` — which pulls `boto3`/`ilipy` via `cli`/`readers` — never runs. Deliberately not `pip install -e`: that would write into their checkout. | working |
| `lib/tt_t_image.py` | The TT-T chain (decimate 23, sparse 128 receive, 0.6–1.4 f0 bandpass, Hilbert, Snell transmit TOF, ray receive TOF, Kirchhoff with the 60/80 deg angle filter) plus `image_metrics()`. Used for BOTH datasets so imaging cannot bias the comparison. | working |
| `mesh/ili_mesh.py` | Conforming mesh. `--quad` (use it), `--scale`, `--no-notch` (healthy variant), `--h-notch`. Verifies 7 properties incl. arc conformity in microns and a DOLFINx round trip. | working |
| `tools/extract_kwave_case.py` | Their 40–230 MB MAT v7 workspace -> few-MB `.npz`. | working |
| `validation/bf_roundtrip.py` | **The gate.** Their beamformer on their own data vs their archived `TT_T.mat`: grid 370×358 exact, peak displacement **0.000 mm**, amplitude within **2.4 %**, correlation 0.929. **Do not raise `--nrays`; higher is worse.** | working |
| `repro/ili_forward.py` | The forward solve. **Use `--degree 4`.** Asserts the transmit delay span against k-Wave's recorded values, derives dt from the true minimum edge, and guards against divergence. | working |
| `repro/analyze_forward.py` | Arrival times vs analytic ToF. Requires peak *prominence*, so it reports NOT RESOLVED rather than inventing an echo. | working |
| `repro/compare_images.py` | The head-to-head table + figure. Panels normalised to their OWN max (source conventions differ). | working |
| `repro/compare_rf.py` | Raw channel-data diff — B-scans, front-wall onset vs element, band-limited energy ratios. Use this when an image difference needs explaining. | working |
| `repro/metric_robustness.py` | Sweeps thresholds, ROI widths and guard distances. **Run this before quoting any metric.** | working |
| `repro/c4_staircase.py` | C4: our solver against ITSELF on a healthy wall, one variable changed - exact ID arc vs 50 um pixel staircase. A defect-free wall must image black, so whatever it images is numerical. | working |

### Visualisation (`viz/`)

| Script | What it does | Status |
|---|---|---|
| `viz/wavefield_gif.py` | D1. Animates `div u` in the water (P) and `curl u` in the steel (S) from `--snapshots` output, so mode conversion at the ID is directly visible. Masks triangles that a raw Delaunay over sample points invents - past the OD, across the notch void, and straddling the ID (that last one would blend two different quantities into a fake halo). GIF is palette-quantised through PIL; `--stride`/`--colors` keep it publishable. | working |
| `viz/mesh_zoom.py` | D4. Four panels from the meshes on disk, with the mesh's own ID facets drawn against the exact circle: conforming wide, conforming vs staircased in the SAME off-axis window, and the notch-tip void. | working |

### Validation against exact solutions

| Script | Validates | Result |
|---|---|---|
| `validation/zoeppritz.py` | oblique fluid-solid reflection/transmission + **P→S mode conversion**, 0–60 deg through both critical angles | at 20 deg: \|R\| **0.10 %**, shear **0.81 %**, shear angle 0.55 deg |
| `validation/cavity_scattering.py` | **defect scattering** vs the exact Pao & Mow series | 0.88 % / 1.01 % max error; complex gain unity to 0.5 % modulus, 0.2 deg phase |

### Capability evidence (`toys/`)

| Script | Proves | Validated result |
|---|---|---|
| `poisson.py` | the FEM pipeline and weak form | max nodal error **2.4e-14** |
| `scalar_wave.py` / `scalar_wave_sem.py` | leapfrog + mass lumping; spectral elements are the timing lever | **2.19 % (P1) -> 0.001 % (P4)** |
| `elastic_wave.py` | elastodynamics, both wave speeds | c_P **0.002 %**, c_S **0.000 %** |
| `fluid_solid.py` | water/steel coupling with mu=0 | \|R\| **0.9351** = analytic, **0.00 %** |

---

## 5. Numerical configuration — read before changing anything

| config | DOF | dt | steps | per solve | crack/clutter |
|---|---|---|---|---|---|
| P3, scale 1.0 | 529 k | 0.865 ns | 69 k | ~15 min | 12.2 dB (**loses**) |
| P4, scale 1.0 | ~941 k | 0.486 ns | 123 k | ~1.2 h | 19.5 dB |
| **P4, scale 0.8** | ~1.5 M | 0.366 ns | 164 k | **~2.4 h** | **24.0 dB (wins)** |

**Why degree 4 and not 3.** "4 nodes per wavelength" must be satisfied at the pulse's UPPER usable
frequency, not its centre. A 1-cycle burst carries ~100 % bandwidth — real energy to 6–8 MHz —
where a mesh sized for 4 MHz gives only 2.7 nodes/wavelength, over a 53-wavelength water path. The
mesh then low-passes our own pulse in transit, and since **axial resolution is set by bandwidth,
not centre frequency**, that alone smeared the notch to 8.07 mm and buried the crack in clutter.

Convergence of the returned pulse, 20 deg:

| config | −6 dB BW | pulse length | (4–6)/(2–4) MHz energy |
|---|---|---|---|
| P3 s1.0 | 1.67 MHz | 587 ns | 0.137 |
| P4 s1.0 | 2.50 MHz | 353 ns | 0.444 |
| **P4 s0.8** | **3.33 MHz** | **282 ns** | **0.694** |
| k-Wave | 4.17 MHz | 211 ns | 0.743 |

Eliminated by measurement, so do not re-litigate: the **source** is not the limit (ours is 5.50 MHz
wide over 167 ns, broader than their returned echo); the energy is **genuinely lost, not delayed**
(the whole-record ratio above is window-independent); and the **absorbing boundary** is not the
cause (our late/reverberation energy is 1.06 and 0.93 of theirs).

**Other settings not to disturb:** `--h-notch` 0.30 mm (finer is a CFL trap — one global dt is set
by the smallest cell over the fastest wavespeed; 0.09 mm cost 10x runtime for nothing), and the
array-plane refinement `H_ARRAY` 0.09 mm (needed to resolve individual 0.30 mm elements; free in
CFL terms because water is 3.8x slower than steel).

---

## 6. Conventions

- Root = infra; `lib/` = shared code; `mesh/` = geometry; `tools/` = utilities; `validation/` = V&V
  against exact solutions; `repro/` = the ILI simulation and comparison; `toys/` = learning
  examples; `results/<name>/` = outputs.
- **Cache heavy intermediates** (`channel_data_*.npz`, `images_*.npz`) so re-analysis and
  re-rendering never re-solve. All the comparison and robustness scripts read cached data.
- Figures are tracked; large regenerable artefacts (`.npz`, `.msh`, `.h5`, `.xdmf`) are gitignored.
- **Run `metric_robustness.py` before quoting a metric.** Two diagnostics in this project produced
  confident wrong numbers before being caught (a window-argmax arrival pick that invented an echo,
  and an onset picker that reported the window floor as an arrival).

## 7. Status

**Done:** environment; capability chain (Poisson -> fluid–solid coupling); their beamformer running
on our data; exact-solution validation of oblique mode conversion and defect scattering; the
conforming mesh (arc error 0.05 um vs ~140 um for a staircase); the forward solve, timing-validated;
and the **head-to-head at 20 deg, which FEM wins on every metric, 15/15 robustness checks**.

**Remaining:** −20 deg at P4/s0.8 (a second valid angle — 0 deg cannot test TT-T, since there is
almost no mode conversion at normal incidence); the **C4 staircase-vs-conforming** controlled
experiment that would isolate *why* we localise better; one more refinement to converge the CNR
number; and **Phase D visualisations, which do not exist yet**. See the branch README §5.
