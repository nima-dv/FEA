# Fea - Agent Guidelines

## ACCESS RESTRICTIONS - READ FIRST, NO EXCEPTIONS

This project reads data owned by other teams. Two locations are **strictly read-only**.
These are not preferences. Violating them corrupts other people's data.

### 1. Dropbox: `F:\DarkVision Dropbox\...` is READ-ONLY

The research team's k-Wave result archive (~580 GB) lives at
`F:\DarkVision Dropbox\RnD\Data\4_Simulations\pipe\polarWave\`.

**Permitted:** read files, list directories.

**FORBIDDEN - all of it:** create, write, append, delete, move, rename, edit, `mkdir`,
`touch`, redirect output into it, write temp/scratch/lock/output files, or write anywhere
else under `F:\DarkVision Dropbox\`.

**Do not "test" whether it is writable.** A probe file is a write. There is no permitted
write, including one you intend to delete afterwards.

Every tool that reads from there must write its output somewhere else.
`tools/extract_kwave_case.py` hard-refuses an `--out` path on `F:` for exactly this reason;
keep that guard.

### 2. Non-DVCode git repos are READ-ONLY - including the `kwave-reference` submodule

`Fea/research/kwave-reference` is a submodule of `darkvisiontech/Research`, which is owned
by the research team. It is the oracle we compare against and it holds their beamformer.

**Permitted:** `git fetch`, `git submodule update`, read any file.

**FORBIDDEN:** editing tracked files, `git commit`, `git push`, `git tag`, force-updating
refs, `pip install -e` into it (that writes an egg-link/`.egg-info` inside their checkout -
this is why `lib/bf_loader.py` exists instead), or writing generated files anywhere in the
submodule tree.

A technical guardrail blocks pushes. **It is local git config and does NOT survive a fresh
clone** - re-apply it after cloning:

```bash
git -C Fea/research/kwave-reference config remote.origin.pushurl \
    "no-push://READ-ONLY-MIRROR/darkvisiontech-Research/we-never-push-here"
```

Verify: `git -C Fea/research/kwave-reference push --dry-run origin HEAD` must fail with
`remote helper 'no-push'`. The fetch URL is untouched, so
`git submodule update --remote Fea/research/kwave-reference` still works.

### 3. Where writes ARE allowed

- Anywhere under `C:\code\DVCode\` (this repo), except the submodule tree above.
- `C:\code\readme\rnd-nima-fea-README.md` (the branch's living design doc).
- The session scratchpad.

Nowhere else. Configuration changes to the developer's machine (Docker images, git global
config, PATH, installed packages) require explicit approval first.

### 4. Delegating to subagents

A subagent does not inherit this file's contents automatically. When spawning one for any
task that touches Dropbox or the submodule, **restate sections 1-3 verbatim in its prompt**.
An agent that has not been told will happily write a temp file next to the data it is
reading.

---

## The FEniCS research project (`research/fenics/`)

FEM ultrasound wave simulation benchmarked against the research team's MATLAB k-Wave ILI
crack-detection sim. Pure Python in Docker, independent of the C++/CUDA `Fea` build.

- **How to run, layout, per-script reference:** `research/fenics/README.md`.
- **Why - asks, constraints, frozen scenario, results, plan:**
  `C:\code\readme\rnd-nima-fea-README.md`. Read it before changing anything numerical;
  several settings look wrong and are deliberate (`--degree 4`, `H_NOTCH` 0.30 mm).

Conventions: cache heavy intermediates so re-analysis never re-solves; run
`repro/metric_robustness.py` before quoting any metric; ASCII only in source, per the
repo-root `AGENTS.md`.

## C++/CUDA `Fea`

See the repo-root [AGENTS.md](../AGENTS.md) for coding standards and build instructions.
