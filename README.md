# FEA — open-source FEM ultrasound crack simulation

An R&D benchmark and a tool. The benchmark asks whether an open-source finite-element
ultrasound simulation (FEniCS/DOLFINx, in Docker) can match — and measurably beat — the MATLAB
**k-Wave** ILI crack-detection simulation. The tool is **Cracken**, a desktop app that drives the
same simulation so an engineer can run it without a terminal.

The benchmark's answer, at +20° steering, both datasets pushed through the research team's own
beamformer with byte-identical settings so **the forward solver is the only difference**:

| metric (true value) | k-Wave | ours |
|---|---|---|
| crack depth (4.0 mm) | 3.48 mm, −13% | **3.73 mm, −6.8%** |
| crack position error | 0.413 mm | **0.165 mm** |
| crack above clutter (RMS) | 24.0 dB | **26.5 dB** |

One production angle is **7.7 minutes** on a consumer GPU against 2.4 hours on a CPU core,
with no licence either way.

## Layout

```
run.ps1              forwarder, so every documented ./run.ps1 command works from the root
docker/              Dockerfile, Dockerfile.gpu, requirements.txt, the real run.ps1
src/
  backend/           THE SOFTWARE. lib/ mesh/ repro/ tools/ validation/ viz/ tests/
  gui/               Cracken, the desktop app that drives it (PySide6)
  kwave/             READ-ONLY submodule: the research team's repo and beamformer
presentation/        THE EVIDENCE. Its own data, scripts and documents - see its README
data/                routine run output. Nothing here is version-controlled.
```

The split that matters is **`presentation/` against `data/`**. `presentation/` holds the
published record — 123 tracked files, all cited by the brief or the dossier. `data/` holds
whatever the last run produced, and none of it is version-controlled: a single solve writes
~42 MB of channel data and a snapshot run 476–966 MB, there is going to be a lot of it, and it
regenerates. A folder boundary does that job better than any ignore rule, because it cannot be
defeated by a new filename that happens to match a pattern.

`src/kwave/` is their repository, mounted read-only. Nothing in this project writes to it.

Inside the container `data/results` is mounted back at `/work/results` and `presentation/` at
`/work/presentation`, so a documented command still reads `--mesh results/ili_mesh/...`. The
host layout is what matters; the container view is deliberately kept stable.
`src/backend/lib/paths.py` is the single place that resolves either.

## Running a simulation

```powershell
./run.ps1 python3 mesh/ili_mesh.py --scale 0.8 --quad
./run.ps1 -Gpu python3 repro/ili_forward.py --degree 4 --gpu --angle 20 --abc-legacy `
    --mesh results/ili_mesh/ili_mesh_s0p8.msh --tag deg4_s0p8_p20deg
./run.ps1 python3 repro/compare_images.py --angle 20 --no-overlay `
    --ours results/ili_forward/channel_data_deg4_s0p8_p20deg.npz
```

`./run.ps1 -PrintArgs` emits the container contract as JSON — the GUI reads it, so the mount
and image decisions have exactly one definition.

## Running the app

```powershell
.\.venv-gui\Scripts\python.exe src\gui\main.py
```

This launches **Cracken**, the desktop app.

Simulate (with a live cross-section and the equation being solved), Results, Export, Inspect
(interactive mesh, wavefield scrubber, image probe), Settings. Every parameter carries a
plain-language explanation of what it does to the result and what it costs.

Three things the app will not do: overwrite a git-tracked file, draw a marker over the crack,
or compare against k-Wave — that comparison is complete, and any further one is a manual CLI
job with `compare_images.py --theirs`.

## Where things are documented

- **`presentation/docs/lessons.md`** — the physics, the numerics and the tooling, written to be
  learned from: the weak form, mass lumping, CFL, λ-driven mesh sizing, absorbing boundaries,
  grating lobes, how the container is put together, how the GPU path works, and the
  experimental method the project runs on
- **`presentation/README.md`** — what the evidence is and how to rebuild every piece of it
- **`src/backend/README.md`** — what every script in the software does and why
- **`src/gui/README.md`** — Cracken's architecture and its non-negotiables
- **`presentation/docs/brief.html`** and **`dossier.html`** — the decision brief and the
  technical dossier, rebuilt with `presentation/scripts/build_artifacts.py`
