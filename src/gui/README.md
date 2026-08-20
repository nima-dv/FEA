# GUI — work order and contract

A desktop app that drives the existing backend. **The backend is finished**; nothing here
adds physics. The app builds an argument list, runs the same Docker command we run by hand,
and shows what comes back.

## Non-negotiables

1. **No physics in the GUI.** Anything derived and displayed (nodes per wavelength, estimated
   dt) is labelled an *estimate* until the backend prints the real number. If the app ever
   decides a value the solver should decide, GUI runs and published runs can diverge.
2. **Every job records its argv**, so it reproduces from a terminal without the app.
3. **No job may overwrite a git-tracked file.** `data/results` holds the published record for
   the R&D challenge — the k-Wave +20° baseline and everything scored against it. A default
   GUI run would otherwise regenerate exactly those filenames. Two defences: the
   `gui_` tag prefix in `model/spec.py`, and a hard check in the runner.
4. **No annotated figures.** Imaging always passes `--no-overlay`.
5. **One container contract.** `docker/run.ps1 -PrintArgs` emits image, mounts and env as
   JSON. Read it once; never hardcode a mount.

## Layout and ownership

```
src/gui/
  main.py            entry point + main window shell            [W3]
  core/
    docker.py        read -PrintArgs, build docker argv, run/kill   [W1]
    runner.py        queue, states, signals, cancel, guards        [W1]
    logparse.py      progress out of backend stdout                [W1]
    manifest.py      per-run JSON: params, argv, timings, outputs  [W1]
  model/
    spec.py          THE CONTRACT - already written, do not change [done]
    scenario.py      read-only scenario facts via scenario_dump    [W2]
    derived.py       nodes/lambda, dt, steps, runtime, disk        [W2]
    guards.py        blocking and warning rules                    [W2]
    presets.py       named configurations + diff vs published      [W2]
  views/
    simulate.py      parameter form + consequences                 [W3]
    queue.py         job cards, progress, log tail, cancel         [W4]
    results.py       run gallery, figures, metrics, compare slider [W4]
    export.py        asset picker + export bundle                  [W4]
  widgets/
    crosssection.py  live QPainter schematic                       [W3]
    consequences.py  derived-quantity panel                        [W3]
    jobcard.py       one queued job                                [W4]
    compareslider.py two images, one draggable handle              [W4]
  theme/dark.qss     the stylesheet                                [W3]
  tests/             assert-based self-checks, one per module      all
```

Run anything with the host venv: `.venv-gui/Scripts/python.exe src/gui/main.py`.
PySide6 6.11 is installed there. Qt is the only host dependency — figures are rendered by
the backend and displayed as files, so there is no host-side matplotlib.

## Palette — DarkVision black and orange

Pin these tokens; do not invent shades.

| token | value | use |
|---|---|---|
| `--bg` | `#0B0B0C` | window |
| `--surface` | `#16181B` | panels, cards |
| `--surface-hi` | `#1D2024` | inputs, elevated |
| `--rule` | `#2A2E33` | borders, separators |
| `--ink` | `#E8E8EA` | primary text |
| `--ink-soft` | `#9AA0A6` | labels, captions |
| `--accent` | `#FF7A1A` | primary action, focus, progress |
| `--accent-hi` | `#FFA24D` | hover |
| `--accent-lo` | `#C25A0F` | pressed |
| `--accent-wash` | `#FF7A1A` at 12% | chips, selected rows |
| `--ok` | `#3FB950` | succeeded |
| `--warn` | `#E3B341` | warning |
| `--fail` | `#F0553A` | failed |
| `--idle` | `#6E7681` | queued, disabled |

Orange is for interaction and emphasis only. Numbers are monospace and right-aligned so
columns compare by eye. **Never re-tint a result image** — the heatmaps come from the backend
with scientific colormaps and their colours carry meaning.

## Phase 1 scope

Parameter form, live cross-section, consequences panel, dry-run, job queue with progress and
cancel, status bar, and a results gallery good enough to look at what a run produced.
Deferred by explicit decision: swappable beamformer, mesh editor for arbitrary anomalies,
editable geometry and materials (those are backend constants, not flags).

## Parameters, and the one that needed a decision

`model/spec.py` is authoritative. Artifact reduction is a three-way choice with **fixed**
specs, because exposing its knobs would invite tuning a boundary treatment against an image:

| option | what it does | flags |
|---|---|---|
| **None** (default) | no workaround, and what every published figure uses | `--abc-legacy` |
| Sponge layer | shear-matched boundary + graded damping in the dead margins | `--sponge-mm 8.0 --sponge-db 40.0` |
| Widened domain | moves the boundary out of reach instead of approximating it | mesh `--x-min -45 --x-max 120`, plus `--abc-legacy` |

40 dB is an optimum, not a maximum — 60 and 200 dB both measured worse. The widened extent is
capped by geometry: past x = −47.46 mm the pipe arc rises above the array plane.

## Guardrails (`model/guards.py`)

**Block** — domain extent outside `[-47.46, +123.96]` mm; projected disk above free space;
target output path is git-tracked; comparison requested with no k-Wave case on disk.
**Warn** — nodes per wavelength below ~3.5 (name the `--scale` that fixes it); snapshots on
(700–970 MB per run); CPU selected for a full-length solve (2.4 h against 7.7 min).

## Progress parsing (`core/logparse.py`)

The solver already prints what is needed:

```
  step 140000/182457 (77%)  4.1 ms/step  elapsed 9.6 min  ETA 2.9 min
```

Mesh prints `[1]`…`[7]` stage markers; `compare_images.py` prints one block per dataset.
Parse, don't reformat: if a line does not match, show it verbatim rather than dropping it.
