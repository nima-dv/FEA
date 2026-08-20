r"""Progress out of backend stdout. Parse, never reformat.

The backend already prints everything the UI needs, so nothing here computes progress from a
model of the solver - a model would drift the moment the solver's print does. Rule from the
work order: a line that does not match is returned as None and MUST be shown VERBATIM by the
caller. Dropping unmatched lines would hide the divergence guard, the sponge summary and
every warning the scripts emit.

The three producers, with their exact prints:
  repro/ili_forward.py   "  step 140000/182457 (77%)  4.1 ms/step  elapsed 9.6 min  ETA 2.9 min"
                         "  step 100/182457  4.12 ms/step"            (--probe, no ETA)
                         "4.100 ms/step over 182455 steps"            (final, authoritative)
  mesh/ili_mesh.py       "[1] standoff identity ...".."[7] DOLFINx ..."
  repro/compare_images.py  one block per dataset: "[FEM] file.npz (256, 4096) peak |p| 1.2e-3"
                           then "    imaged (300, 400) in 4.2s", then the HEAD-TO-HEAD table.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MESH_STAGES = 7          # ili_mesh.py prints [1]..[7]; it is a fixed sequence, not a loop.


@dataclass(frozen=True)
class Progress:
    fraction: float                      # 0..1, monotone within a stage
    detail: str                          # short label for the job card
    ms_per_step: float | None = None
    eta_seconds: float | None = None


# "  step 140000/182457 (77%)  4.1 ms/step  elapsed 9.6 min  ETA 2.9 min"
_STEP_ETA = re.compile(
    r"^\s*step\s+(\d+)/(\d+)\s+\(\s*\d+%\)\s+([\d.]+)\s*ms/step"
    r"\s+elapsed\s+[\d.]+\s*min\s+ETA\s+([\d.]+)\s*min")
# --probe prints the same counter without percent or ETA.
_STEP_BARE = re.compile(r"^\s*step\s+(\d+)/(\d+)\s+([\d.]+)\s*ms/step\s*$")
# The end-of-loop summary. This ms/step is the one worth recording: it averages the whole
# loop, whereas the periodic figure includes assembly time in its first interval.
_STEP_DONE = re.compile(r"^\s*([\d.]+)\s*ms/step over\s+(\d+)\s+steps")
# Mesh stage markers. Only 1..9 so "[FEM]" and "[k-Wave]" cannot match.
_MESH_MARK = re.compile(r"^\[(\d)\]\s*(.*)")
# Sizes for the runtime history: mesh cell count and solver DOF count.
_CELLS = re.compile(r"^\[2\]\s+cells\s+(\d+)")
_DOFS = re.compile(r"degree-\d+ vector DOF\s*=\s*(\d+)")

# compare_images.py never announces how many datasets it will image, so a real fraction is
# not available. These are ordered milestones instead - coarse, monotone, and honest about
# being coarse. ponytail: replace with n/total if the script ever prints a dataset count.
_IMAGE_MARKS: tuple[tuple[re.Pattern[str], float, str], ...] = (
    (re.compile(r"^\[([^\]]+)\]\s+\S+\.npz"), 0.15, "loaded {0}"),
    (re.compile(r"^\s+imaged\s+\(([^)]*)\)\s+in\s+([\d.]+)s"), 0.55, "beamformed {0}"),
    (re.compile(r"^HEAD-TO-HEAD"), 0.80, "metrics"),
    (re.compile(r"^\s*wrote\s+(\S+)"), 1.00, "wrote {0}"),
)


def parse_line(stage, line: str) -> Progress | None:
    """One stdout line -> Progress, or None if it carries no progress (show it verbatim).

    `stage` is a model.spec.Stage; compared by its .value so this module does not import the
    contract just to read an enum (and so a fake stage in a test still works).
    """
    kind = getattr(stage, "value", stage)

    if kind == "forward":
        m = _STEP_ETA.match(line)
        if m:
            n, total, ms, eta_min = m.groups()
            return Progress(int(n) / max(int(total), 1), f"step {n}/{total}",
                            float(ms), float(eta_min) * 60.0)
        m = _STEP_BARE.match(line)
        if m:
            n, total, ms = m.groups()
            return Progress(int(n) / max(int(total), 1), f"step {n}/{total}", float(ms))
        m = _STEP_DONE.match(line)
        if m:
            # Time loop done, but the file write and the receiver resample still follow, so
            # this is 1.0 of the SOLVE, not of the job. The job card should keep showing
            # output until the process exits.
            return Progress(1.0, "time loop done", float(m.group(1)))
        return None

    if kind == "mesh":
        m = _MESH_MARK.match(line)
        if m:
            i = int(m.group(1))
            if 1 <= i <= MESH_STAGES:
                return Progress(i / MESH_STAGES, f"[{i}] " + m.group(2)[:60].strip())
        return None

    if kind in ("image", "figures"):
        for pat, frac, tmpl in _IMAGE_MARKS:
            m = pat.match(line)
            if m:
                return Progress(frac, tmpl.format(*m.groups()) if m.groups() else tmpl)
        return None

    return None


def parse_size(line: str) -> int | None:
    """Cell count (mesh) or DOF count (solver) if this line announces one.

    Lives here so both regexes sit next to the prints they mirror; manifest.history() needs
    the number to key past runtimes on problem size.
    """
    for pat in (_CELLS, _DOFS):
        m = pat.search(line)
        if m:
            return int(m.group(1))
    return None


def demo() -> None:
    """Self-check with lines copied out of the backend scripts, character for character."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.spec import Stage                                    # noqa: E402

    p = parse_line(Stage.FORWARD,
                   "  step 140000/182457 (77%)  4.1 ms/step  elapsed 9.6 min  ETA 2.9 min")
    assert p and abs(p.fraction - 140000 / 182457) < 1e-9, p
    assert p.ms_per_step == 4.1 and abs(p.eta_seconds - 174.0) < 1e-6, p
    assert p.detail == "step 140000/182457"

    p = parse_line(Stage.FORWARD, "  step 100/182457  4.12 ms/step")
    assert p and p.ms_per_step == 4.12 and p.eta_seconds is None, p

    p = parse_line(Stage.FORWARD, "4.100 ms/step over 182455 steps")
    assert p and p.fraction == 1.0 and p.ms_per_step == 4.1, p

    # Divergence guard, sponge notes, warnings: no progress, so verbatim display.
    assert parse_line(Stage.FORWARD, "solution diverging at step 500: max|u| = 1.2e+30") is None
    assert parse_line(Stage.FORWARD, "  step 140000/182457 (77%)") is None, "partial line"

    m = parse_line(Stage.MESH, "[1] standoff identity      R_ID + z_c = 193.675 + (-0.174) = 20")
    assert m and abs(m.fraction - 1 / 7) < 1e-12 and m.detail.startswith("[1] standoff"), m
    assert parse_line(Stage.MESH, "[7] DOLFINx 0.9.0 round trip: ok").fraction == 1.0
    assert parse_line(Stage.MESH, "[8] not a stage") is None
    assert parse_line(Stage.MESH, "    wall 9.525 mm | OD on axis z = 203.026 mm") is None
    # A mesh marker must not be confused with a compare_images dataset label.
    assert parse_line(Stage.MESH, "[k-Wave] file.npz (256, 4096)  peak |p| 1.2e-03") is None

    i = parse_line(Stage.IMAGE, "[k-Wave] kwave_p20deg.npz  (256, 4096)  peak |p| 1.234e-03")
    assert i and i.detail == "loaded k-Wave" and i.fraction == 0.15, i
    i2 = parse_line(Stage.IMAGE, "    imaged (301, 401) in 4.2s")
    assert i2 and i2.fraction == 0.55 and "301, 401" in i2.detail, i2
    assert parse_line(Stage.IMAGE, "HEAD-TO-HEAD, steering +20 deg, identical beamformer, "
                                   "same grid").fraction == 0.80
    done = parse_line(Stage.IMAGE, "wrote /work/results/compare/images_20_gui_x_nooverlay.npz")
    assert done and done.fraction == 1.0 and "images_20" in done.detail, done
    assert parse_line(Stage.IMAGE, "Higher dB = crack stands further above the clutter") is None
    # Milestones must not go backwards within a run.
    assert [x.fraction for x in (i, i2, done)] == sorted(x.fraction for x in (i, i2, done))

    assert parse_size("[2] cells 128744 (fluid 91000, steel 37744, fill 0) | ...") == 128744
    assert parse_size("      degree-4 vector DOF = 2094218") == 2094218
    assert parse_size("    dashpot dofs 512 something") is None, "dashpot count is not a size"
    print("logparse.demo: ok")


if __name__ == "__main__":
    demo()
