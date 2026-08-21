# src/backend — FEM ultrasound wave-sim (run & maintain guide)

FEniCS/DOLFINx finite-element research for DarkVision ultrasound simulation. Pure Python,
containerised. This is **the software**; the published evidence and the scripts that build it
live in `presentation/` (its own README).

> **The "why" — asks, constraints, frozen scenario, plan, current state, and the full results
> tables — lives in the branch README: `F:/code/readme/rnd-nima-FEA-README.md`. Read that first.**
> **This file is the "how": environment, layout, what each script does, and how to reproduce
> the headline result.**

> **READ-ONLY BOUNDARY — see [`AGENTS.md`](../../AGENTS.md).** The `src/kwave` submodule
> (`darkvisiontech/Research`) is the research team's: read and list, never write, commit or push.
> Writes go under `F:\code\FEA\` (outside the submodule), the branch README, or the scratchpad.

---

## 0. Headline result

Our FEM channel data and the research team's k-Wave channel data pushed through **their own
beamformer**, so the forward solver is the only difference. **Both valid steering angles, on the
adopted `faithfulbf` imaging chain** (see `presentation/data/compare/NAMING.md`):

| metric (true value) | k-Wave +20 | **FEM +20** | k-Wave -20 | **FEM -20** |
|---|---|---|---|---|
| crack x error (38.25 mm) | 0.413 | **0.165** | 0.332 | **0.084** |
| notch extent (4.0 mm) | 3.48 (-13.0%) | **3.73 (-6.8%)** | 3.85 (-3.8%) | 3.85 (-3.8%, tie) |
| crack / clutter RMS | 24.03 dB | **26.46** | 24.47 dB | **26.26** |
| crack / clutter p95 | 17.90 dB | **18.90** | 18.41 dB | **18.88** |
| crack / worst clutter | 11.57 dB | **14.12** | 11.83 dB | **13.47** |

**FEM better in 25/30 robustness checks** across the two angles (14/15 at +20, 11/15 at -20).
Lead with what is unanimous: **contrast 10/10** at both angles and every guard distance, and
position 8/8. Sizing is 15/20 - a +20 deg win and an exact TIE at -20 deg. Say the tie out loud.

**The -20 deg numbers were PREDICTED before the solve ran** (23-25 dB, 3.7-4.2 mm extent) and
landed inside the range on the chain in use at the time, so the bandwidth mechanism in §5 is
out-of-sample tested, not fitted.

**Our sizing is angle-consistent and theirs is not:** 3% spread across +-20 deg against their
11%. Since both datasets pass through the same imaging chain, that asymmetry is in their forward
data. For an inspection tool this matters more than any single dB figure - real cracks are never
conveniently on-axis. **The 96% figure quoted before 2026-08-19 was inflated by our own imaging
defect - do not use it.** The advantage is real but 3.7x, not 30x.

**WHY we win is OPEN, and it is not geometry.** C4 (`presentation/scripts/c4_staircase.py`)
staircased the curved ID at k-Wave's own 50 um: **+0.61 dB**. C5 filled our notch with their
500 m/s material: **0.07 dB**. Both eliminated, so do not say "we win because the mesh conforms".
Numerical bandwidth explains **our own convergence** from losing to winning - predicted before
the -20 deg solve and confirmed - but on raw bandwidth k-Wave is still ahead of us, so it does
not explain exceeding them. "We measure better and cannot yet attribute it" is the honest position.

See the branch README §4.7-4.12 for what is and is not defendable: sizing is converged, contrast
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
cd F:\code\FEA
./run.ps1 python3 lib/bf_loader.py          # env smoke test: DOLFINx + their beamformer
./run.ps1                                   # interactive bash shell
```

The real launcher is `docker/run.ps1`; the repo root carries a forwarder so every documented
command copy-pastes from the root. Mounts: `src/backend` -> `/work` rw, `data/results` ->
`/work/results` rw, `presentation` -> `/work/presentation` rw, `data/raw` -> `/raw` **ro**, their
beamformer -> `/opt/bf` **ro**. So a container-side path stays `results/ili_mesh/...` even though
on the host the outputs sit outside the source tree. `lib/paths.py` is the single place that
resolves either layout. `./run.ps1 -PrintArgs` emits the whole contract as JSON (Cracken, the GUI
at `src/gui/`, reads it). Override the image with `$env:DVFENICS_IMAGE`; `-Gpu` selects
`dvfenics:gpu` and passes the device through.

**Gotchas that will waste your afternoon:**

| Symptom | Cause / fix |
|---|---|
| `working directory 'C:/Program Files/Git/work' is invalid` | Git Bash rewrites container paths. Use PowerShell, or prefix `MSYS_NO_PATHCONV=1`. |
| PowerShell reports failure on a successful run | It wraps native **stderr** in error records; their package emits harmless `SyntaxWarning`s. Don't use `2>&1`; or run via Bash. |
| `ModuleNotFoundError: beamformer` | Submodule not checked out: `git submodule update --init --depth 1 src/kwave` |
| `NotImplementedError: engine must be 'cuda'` | Receive-side Fermat TOF is GPU-only. Use the CPU `ray` path (the default in `lib/tt_t_image.py`). |
| A background solve looks dead (empty log, nothing in `docker ps`) | **It probably isn't.** Solves print nothing during the time loop, and `docker ps` has returned nothing while a container was up. Use `docker ps -a`. |

---

## 2. Reproducing the headline result

Their raw runs are extracted first, then two solves and one comparison. **Run every command from
the repo root** (`F:\code\FEA`) — the forwarder there is what makes these paths work. On the GPU
a production angle is 7.7 min; on a CPU core it is 2.4 h.

The extracted k-Wave cases and the published channel data are already committed under
`presentation/data/`, so steps 1 and 3 only need re-running if you are regenerating them.

```powershell
# 1. Extract the research team's runs (their raw data stays untouched).
./run.ps1 python3 tools/extract_kwave_case.py `
    --run "presentation/data/k-wave/raw/derrell/2026-08-11 12-46-37 Simulation_ODnotch4mm_20" `
    --out presentation/data/k-wave/kwave_cases/kwave_odnotch4mm_20.npz

# 2. Build the conforming mesh at the working resolution  (~10 s)
./run.ps1 python3 mesh/ili_mesh.py --quad --scale 0.8

# 3. Forward solve: degree 4 on that mesh  (163678 steps; 7.7 min on the GPU, 2.4 h on a core)
./run.ps1 -Gpu python3 repro/ili_forward.py --angle 20 --degree 4 --gpu --abc-legacy `
    --mesh results/ili_mesh/ili_mesh_s0p8.msh --tag deg4_s0p8_p20deg

# 4. Both datasets through THEIR beamformer -> the head-to-head table + figure
./run.ps1 python3 repro/compare_images.py --angle 20 --no-overlay `
    --ours presentation/data/ili_forward/channel_data_deg4_s0p8_p20deg.npz `
    --theirs presentation/data/k-wave/kwave_cases/kwave_odnotch4mm_20.npz

# 5. Confirm the claims survive every reasonable analysis choice
./run.ps1 python3 presentation/scripts/metric_robustness.py --angle 20
```

`--abc-legacy` is the boundary treatment every published figure uses; it is not the flag default.
Imaging always passes `--no-overlay` — no annotated figure is published any more.

### The demonstration figures

These build the published record, so they live in `presentation/scripts/` — see
`presentation/README.md` for the full rebuild list.

```powershell
# wavefield animation: add --snapshots to any solve, then render (degree 3 is enough here)
./run.ps1 -Gpu python3 repro/ili_forward.py --angle 20 --degree 3 --gpu --abc-legacy `
    --mesh results/ili_mesh/ili_mesh_tri.msh --tag snap_p20deg `
    --snapshots 240 --snap-window "18,46"
./run.ps1 python3 viz/wavefield_gif.py --in results/ili_forward/wavefield_snap_p20deg.npz `
    --stride 3 --fps 10 --colors 48 --smooth 2 --stills "22,27,31,35"

# mesh conformity figure (needs the two healthy triangle meshes; no solve required)
./run.ps1 python3 mesh/ili_mesh.py --no-notch --no-plot
./run.ps1 python3 mesh/ili_mesh.py --no-notch --no-plot --staircase
./run.ps1 python3 presentation/scripts/mesh_zoom.py --cracked results/ili_mesh/ili_mesh_s0p8.msh
```

---

## 3. Layout

```
src/backend/                 THE SOFTWARE. Mounted at /work, so it is the working directory.
├─ lib/
│   ├─ paths.py           WHERE THINGS LIVE. One definition; env vars in-container, walk-out
│   │                     to the repo root on the host. Never rebuild a results path by hand.
│   ├─ bf_loader.py       loads THEIR beamformer without executing its __init__.py
│   └─ tt_t_image.py      the TT-T imaging chain + image_metrics(), shared by all comparisons
├─ mesh/
│   └─ ili_mesh.py        conforming gmsh mesh: exact ID/OD arcs + notch as a real void
├─ tools/
│   ├─ extract_kwave_case.py   their *_workspace.mat -> compact .npz
│   ├─ gpu_probe.py, gpu_gate.py   is the GPU path faster, and does it still arrive on time
│   ├─ mpi_scaling.py, matrix_free_probe.py, cfl_limit.py   cost and scaling probes
│   ├─ scenario_dump.py        the frozen scenario as JSON (the GUI reads it)
│   ├─ publication_backup.py   snapshot the published record
│   └─ probe_bf_deps.py        how the beamformer dependency list was derived
├─ validation/             V&V against EXACT solutions
│   ├─ bf_roundtrip.py         gate: their beamformer on their data == their image
│   ├─ zoeppritz.py            angle-resolved fluid-solid mode conversion
│   └─ cavity_scattering.py    defect scattering vs the exact Pao & Mow series
├─ repro/                  the ILI simulation and the comparison
│   ├─ ili_forward.py          THE FORWARD SOLVE (--gpu, --snapshots for the animations)
│   ├─ analyze_forward.py      arrival times vs analytic ToF
│   ├─ compare_images.py       FEM vs k-Wave through one identical beamformer
│   └─ ili_gate.py, ili_angled.py, ili_beamform.py, ili_realistic.py, animate_gate.py,
│      analyze_gate.py, render_*.py    earlier work; ili_realistic (corrosion) is LEGACY
├─ viz/
│   └─ wavefield_gif.py        D1: the beam mode-converting at the ID and hitting the notch
├─ tests/                  assert-based self-checks
└─ results/                empty mountpoint on the host; data/results is mounted here.

  ../../data/results/      routine run output. NOT version-controlled, and voluminous.
  ../../presentation/      THE EVIDENCE: its own data, scripts and documents. Own README.
  ../kwave/                submodule: their k-Wave driver .m + their beamformer (READ-ONLY)
```

**The split that matters:** if the GUI or a routine run needs a script it belongs here; if it
exists only to produce publication material it belongs in `presentation/scripts/`. That is why
`metric_robustness.py`, `c4_staircase.py`, `compare_rf.py`, `mesh_zoom.py`,
`artifact_reduction.py`, `bandwidth_convergence.py`, `baseline_subtract.py` and
`build_artifacts.py` are not in this tree. `validation/` stays here on purpose: its figures are
publication material, but what it does is check the solver, so it has to re-run whenever the
solver changes.

---

## 4. Scripts

### The comparison chain

| Script | What it does | Status |
|---|---|---|
| `lib/bf_loader.py` | Imports their `beamformer` imaging path by pre-registering the package in `sys.modules`, so `__init__.py` — which pulls `boto3`/`ilipy` via `cli`/`readers` — never runs. Deliberately not `pip install -e`: that would write into their checkout. | working |
| `lib/tt_t_image.py` | The TT-T chain (decimate 23, sparse 128 receive, 0.6–1.4 f0 bandpass, Hilbert, Snell transmit TOF, ray receive TOF, Kirchhoff with the 60/80 deg angle filter) plus `image_metrics()`. Used for BOTH datasets so imaging cannot bias the comparison. | working |
| `mesh/ili_mesh.py` | Conforming mesh. `--quad` (use it), `--scale`, `--no-notch` (healthy variant), `--h-notch`. Verifies 7 properties incl. arc conformity in microns and a DOLFINx round trip. | working |
| `tools/extract_kwave_case.py` | Their 40–230 MB MAT v7 workspace -> few-MB `.npz`. | working |
| `validation/bf_roundtrip.py` | **The gate.** Their beamformer on their own data vs their archived `TT_T.mat`: grid 370×358 exact, peak displacement **0.000 mm**, amplitude within **2.4 %**, correlation 0.929. **Do not raise `--nrays`; higher is worse.** | **passed once, cannot re-run** — its reference `TT_T.mat` and 0 deg case went with Dropbox. State it as such. Partial substitute: our chain reproduces k-Wave's published metrics to 3 s.f. |
| `repro/ili_forward.py` | The forward solve. **Use `--degree 4`.** Asserts the transmit delay span against k-Wave's recorded values, derives dt from the true minimum edge, and guards against divergence. | working |
| `repro/analyze_forward.py` | Arrival times vs analytic ToF. Requires peak *prominence*, so it reports NOT RESOLVED rather than inventing an echo. | working |
| `repro/compare_images.py` | The head-to-head table + figure. Panels normalised to their OWN max (source conventions differ). | working |
| `presentation/scripts/compare_rf.py` | Raw channel-data diff — B-scans, front-wall onset vs element, band-limited energy ratios. Use this when an image difference needs explaining. | working |
| `presentation/scripts/metric_robustness.py` | Sweeps thresholds, ROI widths and guard distances. **Run this before quoting any metric.** | working |
| `presentation/scripts/c4_staircase.py` | C4: our solver against ITSELF on a healthy wall, one variable changed - exact ID arc vs 50 um pixel staircase. A defect-free wall must image black, so whatever it images is numerical. | working |

### Visualisation (`viz/`)

| Script | What it does | Status |
|---|---|---|
| `viz/wavefield_gif.py` | D1. Animates `div u` in the water (P) and `curl u` in the steel (S) from `--snapshots` output, so mode conversion at the ID is directly visible. Masks triangles that a raw Delaunay over sample points invents - past the OD, across the notch void, and straddling the ID (that last one would blend two different quantities into a fake halo). GIF is palette-quantised through PIL; `--stride`/`--colors` keep it publishable. | working |
| `presentation/scripts/mesh_zoom.py` | D4. Four panels from the meshes on disk, with the mesh's own ID facets drawn against the exact circle: conforming wide, conforming vs staircased in the SAME off-axis window, and the notch-tip void. | working |

### Validation against exact solutions

| Script | Validates | Result |
|---|---|---|
| `validation/zoeppritz.py` | oblique fluid-solid reflection/transmission + **P→S mode conversion**, 0–60 deg through both critical angles | at 20 deg: \|R\| **0.10 %**, shear **0.81 %**, shear angle 0.55 deg |
| `validation/cavity_scattering.py` | **defect scattering** vs the exact Pao & Mow series | 0.88 % / 1.01 % max error; complex gain unity to 0.5 % modulus, 0.2 deg phase |

---

## 5. Numerical configuration — read before changing anything

| config | DOF | dt | steps | per solve (CPU / GPU) | crack/clutter |
|---|---|---|---|---|---|
| P3, scale 1.0 | 529 k | 0.865 ns | 69 k | ~15 min | 12.2 dB (**loses**) |
| P4, scale 1.0 | ~941 k | 0.486 ns | 123 k | ~1.2 h | 19.5 dB |
| **P4, scale 0.8** | ~1.5 M | 0.366 ns | 164 k | **~2.4 h / 7.7 min** | **26.5 dB (wins)** |

The two upper rows are legacy-chain numbers and cannot be regenerated (their channel data is
gone); they show the refinement trend only. The bottom row is on the adopted chain. The GPU
figure is the time loop only — meshing and assembly stay on the CPU — and it is validated on
ARRIVAL TIME by `tools/gpu_gate.py`, not on a norm (`presentation/data/perf/gpu_gate.txt`).

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

- `lib/` = shared code; `mesh/` = geometry; `tools/` = utilities and probes; `validation/` = V&V
  against exact solutions; `repro/` = the ILI simulation and comparison; `viz/` = animations.
  Infra is in `docker/`; published material is in `presentation/`.
- **Resolve output paths through `lib/paths.py`** (`RESULTS`, `RAW`, `PRESENTATION`, `PRES_DATA`),
  never by rebuilding one from a script's own location. That assumption broke when data was
  separated from source.
- **Cache heavy intermediates** (`channel_data_*.npz`, `images_*.npz`) so re-analysis and
  re-rendering never re-solve. All the comparison and robustness scripts read cached data.
- Figures are tracked; large regenerable artefacts (`.npz`, `.msh`, `.h5`, `.xdmf`) are gitignored.
- **Run `metric_robustness.py` before quoting a metric.** Two diagnostics in this project produced
  confident wrong numbers before being caught (a window-argmax arrival pick that invented an echo,
  and an onset picker that reported the window floor as an arrival).

## 7. Status

**Done:** environment; capability chain (Poisson -> fluid–solid coupling); their beamformer running
on our data; exact-solution validation of oblique mode conversion and defect scattering; the
conforming mesh (arc error 0.05 um vs ~140 um for a staircase); the forward solve,
timing-validated; the head-to-head at **both** valid angles (25/30 robustness checks); the
controlled experiments C4 (staircase, +0.61 dB) and C5 (crack fill, 0.07 dB), both negative; the
healthy-wall baseline; the wavefield animation and the mesh-conformity figure; and the GPU time
loop, ~19x and gated on arrival time.

**Remaining:** convergence / GCI error bars (B4 — the last piece of promised rigour); a third and
fourth steering angle, since sizing is a +20 deg win and a tie at -20; and a defect geometry a
grid represents badly, which is the one structural advantage we assert and have never
demonstrated. The residual edge artefact is characterised rather than open: six candidate causes
eliminated by measurement, and part of it is a grating lobe inherent to the array and present in
both solvers. **0 deg is not a third angle** — normal incidence produces almost no mode
conversion, so neither solver images anything meaningful there. See the branch README §5.
