# FEA — open-source FEM ultrasound crack simulation

An R&D benchmark: can an open-source finite-element ultrasound simulation
(FEniCS/DOLFINx, in Docker) match — and measurably beat — the MATLAB **k-Wave** ILI
crack-detection simulation the research team uses today?

Current answer at +20° steering, both datasets pushed through the research team's own
beamformer with byte-identical settings, so **the forward solver is the only difference**:

| metric (true value) | k-Wave | ours |
|---|---|---|
| crack depth (4.0 mm) | 3.48 mm, −13% | **3.73 mm, −6.8%** |
| crack position error | 0.413 mm | **0.165 mm** |
| crack above clutter (RMS) | 24.0 dB | **26.5 dB** |

One production angle is **7.7 minutes** on a consumer GPU, against 2.4 hours on a CPU
core, with no licence either way.

## Layout

```
run.ps1              forwarder, so every documented ./run.ps1 command works from the root
docker/              Dockerfile, Dockerfile.gpu, requirements.txt, the real run.ps1
docs/                lessons.md (concepts, equations, method), PITCH.md
src/
  backend/           the science. lib/ mesh/ repro/ tools/ validation/ viz/ tests/
  gui/               the desktop app that drives the backend (PySide6)
  kwave/             READ-ONLY submodule: the research team's repo and beamformer
data/
  raw/               their raw k-Wave workspaces (gitignored bulk, ~423 MB)
  results/           our outputs: meshes, channel data, images, figures, built pages
```

Source and data are separate, and **nothing writes into `src/kwave/`** — it is their
repository, mounted read-only.

Inside the container `data/results` is mounted back at `/work/results`, so a documented
command still reads `--mesh results/ili_mesh/...`. The host layout is what matters; the
container view is deliberately kept stable. `src/backend/lib/paths.py` is the single place
that resolves either.

## Running

```powershell
./run.ps1 python3 mesh/ili_mesh.py --scale 0.8 --quad
./run.ps1 -Gpu python3 repro/ili_forward.py --degree 4 --gpu --angle 20 --abc-legacy `
    --mesh results/ili_mesh/ili_mesh_s0p8.msh --tag deg4_s0p8_p20deg
./run.ps1 python3 repro/compare_images.py --angle 20 `
    --theirs results/kwave_cases/kwave_odnotch4mm_20.npz `
    --ours results/ili_forward/channel_data_deg4_s0p8_p20deg.npz
```

`./run.ps1 -PrintArgs` emits the container contract as JSON — the GUI reads it so the mount
and image decisions have exactly one definition.

## Where things are documented

- `docs/lessons.md` — the physics and numerics: weak form, mass lumping, CFL, λ-driven mesh
  sizing, absorbing boundaries, grating lobes, and the experimental method the project runs on
- `src/backend/README.md` — what every script does and why
- `data/results/artifacts/` — the built brief and dossier (rebuild with
  `viz/build_artifacts.py`; the HTML itself is gitignored, the figures are not)
