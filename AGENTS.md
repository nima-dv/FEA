# FEA - Agent Guidelines

## ACCESS RESTRICTION - READ FIRST

### `src/kwave/` is READ-ONLY

It is a submodule of `darkvisiontech/Research`, owned by the research team. It holds both the
k-Wave simulation we benchmark against (the oracle) and their `beamformer` package.

**Permitted:** `git fetch`, `git submodule update`, read any file.

**FORBIDDEN:** editing tracked files, `git commit`, `git push`, `git tag`, force-updating refs,
writing generated files anywhere in the tree, and `pip install -e` (that writes an
`.egg-info` inside their checkout - which is why `lib/bf_loader.py` imports it in place).

A push guardrail is applied, but it is **local config and does NOT survive a fresh clone**:

```bash
git -C src/kwave config remote.origin.pushurl \
    "no-push://READ-ONLY-MIRROR/darkvisiontech-Research/we-never-push-here"
```

Verify: `git -C src/kwave push --dry-run origin HEAD` must fail with `remote helper 'no-push'`.
The fetch URL is untouched, so `git submodule update --remote src/kwave` still works.

**A subagent does not inherit this file.** Restate this section verbatim in its prompt.

### `data/derrell/` is bulk input, never committed

~423 MB of the research team's raw k-Wave output. Each `_workspace.mat` is ~122 MB, over
GitHub's hard 100 MB per-file limit, so a push carrying one is **rejected outright**.
Gitignored. Mounted read-only at `/data` in the container.

---

## The project (`src/fenics/`)

FEM ultrasound wave simulation benchmarked against the research team's MATLAB k-Wave ILI
crack-detection sim. Pure Python in Docker.

- **How to run, layout, per-script reference:** `src/fenics/README.md`.
- **Why - asks, constraints, frozen scenario, results, plan:**
  `F:\code\readme\rnd-nima-FEA-README.md`. Read it before changing anything numerical;
  several settings look wrong and are deliberate (`--degree 4`, `H_NOTCH` 0.30 mm).
- **Theory - equations, the two methods, how the comparison is made fair:** `lessons.md`.
  Concepts only; keep measured numbers out of it so it cannot drift from the README.
  Update it whenever the understanding changes.
- **Pitch talking points, slide by slide:** `PITCH.md`.

**No default FEM configuration is fixed yet.** `legacy boundary` is what every published figure
uses; `--abc-legacy` and `--sponge-mm` variants are experiments, not defaults. Do not quietly
adopt one.

Conventions: ASCII only in source. Cache heavy intermediates so re-analysis never re-solves.
Run `repro/metric_robustness.py` before quoting any metric. Keep markdown light - current
state only, no changelog, no dead text.
