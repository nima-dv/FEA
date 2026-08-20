# presentation — the evidence for the R&D challenge

A sub-project inside FEA. Everything here exists to support one claim: that an open-source
finite-element ultrasound simulation is **accurate**, **valid**, and **measurably more accurate
than** the MATLAB k-Wave ILI crack simulation the research team uses today.

It is deliberately separate from the rest of the repository. `data/` in the project root is
routine run output — regenerable, voluminous, and not evidence. What is in here is evidence,
and the folder boundary is what keeps a stray run from overwriting a published figure.

## The headline, +20° steering

Both datasets pushed through the research team's own beamformer with byte-identical settings,
so **the forward solver is the only difference**:

| metric (true value) | k-Wave | ours |
|---|---|---|
| crack depth (4.0 mm) | 3.48 mm, −13% | **3.73 mm, −6.8%** |
| crack position error | 0.413 mm | **0.165 mm** |
| crack above clutter (RMS) | 24.0 dB | **26.5 dB** |
| crack above worst-case clutter | 11.6 dB | **14.1 dB** |

One production angle is **7.7 minutes** on a consumer GPU against 2.4 hours on a CPU core,
with no licence either way.

## Layout

```
presentation/
  data/              the published record - 101 files, ~450 MB, all version-controlled
    compare/         head-to-head figures and the beamformed image caches
    ili_forward/     channel data from every published solve
    k-wave/          the research team's runs, extracted to a few MB each
    viz/             animations, wavefield stills, the mesh and artifact figures
    zoeppritz/       mode conversion against the exact fluid-solid solution
    cavity_scattering/  defect scattering against the exact Pao & Mow series
    perf/            the GPU acceptance gate's output
    ili_mesh/        the production mesh figure
  docs/
    lessons.md       the physics and numerics of the whole project, written to be learned from
    PITCH.md         the pitch outline
    brief.html       the decision brief      (generated)
    dossier.html     the technical dossier   (generated)
  scripts/           everything that builds the above, and nothing else
```

## What is in here, and what is not

The rule: **if the GUI or a routine simulation run needs a script, it belongs to the software
and lives in `src/backend`. If it exists only to produce publication material, it lives here.**

| here | in `src/backend` |
|---|---|
| `build_artifacts.py`, `_brief.py`, `_dossier.py` | `mesh/ili_mesh.py` |
| `artifact_reduction.py`, `bandwidth_convergence.py`, `mesh_zoom.py` | `repro/ili_forward.py` |
| `baseline_subtract.py`, `compare_rf.py` | `repro/compare_images.py` |
| `metric_robustness.py`, `c4_staircase.py` | `viz/wavefield_gif.py` |

`validation/` stays in the backend, and that one is a judgement call worth recording. Its
figures appear in the dossier, so by output it looks like presentation material — but what it
does is check the solver against exact solutions, and that has to be re-runnable whenever the
solver changes. It belongs with the solver regardless of who reads the plots.

## Rebuilding

Every command runs from the repository root and goes through the container, which mounts this
folder at `/work/presentation`:

```powershell
# the documents
./run.ps1 python3 presentation/scripts/build_artifacts.py

# the figures they embed
./run.ps1 python3 presentation/scripts/artifact_reduction.py --angle 20
./run.ps1 python3 presentation/scripts/mesh_zoom.py
./run.ps1 python3 presentation/scripts/bandwidth_convergence.py
./run.ps1 python3 presentation/scripts/baseline_subtract.py --angle 20 `
    --cracked presentation/data/ili_forward/channel_data_deg4_s0p8_p20deg.npz `
    --healthy presentation/data/ili_forward/channel_data_deg4_s0p8_healthy_p20deg.npz

# do the published claims survive different analysis choices?
./run.ps1 python3 presentation/scripts/metric_robustness.py --angle 20
```

`build_artifacts.py` names every figure it expects and **prints a placeholder warning** if one
is missing rather than emitting a page with a silent hole in it.

## Three rules that hold everywhere in here

**No annotated figures.** No wall arcs, no marker over the crack. Being told where to look
disqualifies a judgement of whether the defect is detectable, and a marker covers the very
feature a figure exists to show. Every comparison figure is a `--no-overlay` render.

**Every figure regenerates from committed code.** That is the difference between evidence and
a screenshot. If a number appears in the brief or the dossier, the script that produced it is
in `scripts/` and the data it read is in `data/`.

**Nothing in here is written by the GUI.** The app writes to `data/results` in the project
root, its output filenames carry a `gui_` prefix, and its runner refuses outright to write to
any git-tracked path. Three independent reasons a routine run cannot land in the record.

## Adding something

Put the script in `scripts/`, the data it produces in the matching `data/` subdirectory,
register any new figure in `build_artifacts.py`'s `IMG` dictionary, and commit it. There is no
allowlist to update — the folder boundary is the policy.

If a figure is expensive to produce, say so in its docstring with the measured cost. A future
reader deciding whether to regenerate it needs the number, not an adjective.
