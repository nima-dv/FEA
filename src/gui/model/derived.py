r"""Derived quantities for the consequences panel. Pure functions, no Qt, no physics decisions.

EVERYTHING HERE IS AN ESTIMATE, and says so. The solver decides dt from the mesh it actually
read; the mesher decides how many cells the size field produced. The GUI cannot know either
before the run, and if it ever pretends to, a GUI run and a published run can diverge on
paper while agreeing on disk - which is the one failure mode this project cannot afford.
So each result carries `is_estimate` and a note naming what the real number depends on.

THE THREE FORMULAS, AND WHERE THEY COME FROM (docs/lessons.md sections 4-5)
  resolution  N = p * lambda / h      lambda = c / f_upper, NOT c / f0
  time step   dt = CFL * h / (c p^2)  minimised over regions, h the SHORTEST EDGE
  cost        one step is a sparse matvec: memory-bandwidth bound, so ms/step ~ unknowns

CALIBRATION AGAINST THE PRODUCTION RUN (degree 4, scale 0.8, quad, 45,711 cells, 1.47 M
unknowns, 60 us): real dt 0.3666 ns, 163,680 steps, 7.7 min on the GPU, ~2.4 h on one CPU
core. Every constant below with "measured" on it was fitted to that run or to the files it
left in data/results. A formula on paper that lands 2x from the measured number is not a
useful estimate, so the two calibration factors are explicit and named rather than hidden.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

try:                                  # inside the app: `from model import derived`
    from . import scenario as scen
    from .spec import ArtifactReduction, Device, RunConfig
except ImportError:                   # run directly: python src/gui/model/derived.py
    import scenario as scen           # type: ignore[no-redef]
    from spec import ArtifactReduction, Device, RunConfig   # type: ignore[no-redef]

# --- resolution ------------------------------------------------------------------------
# The warning threshold, set just BELOW the published record rather than at a round number.
# The production mesh measures 3.46 nodes per wavelength in steel shear and 3.35 in water at
# 8 MHz, so a 3.5 floor would flag the baseline configuration itself - and a guardrail that
# fires on the published record is one users learn to click through. Anything at or above the
# record is by definition as resolved as every published figure.
NODES_MIN = 3.3

# --- time step -------------------------------------------------------------------------
# MEASURED, and the single biggest correction in this module. A size field asking for 0.24 mm
# cells produces cells whose SHORTEST EDGE is 0.111 mm: gmsh grades, and a quad's short side
# is not its target size. The solver takes dt from that shortest edge (see the domain.h()
# note in repro/ili_forward.py), so a target-size dt is ~2.15x too optimistic - the kind of
# error that looks fine in a 300-step probe and diverges an hour later.
#   check: 0.30 * (0.30*0.8/2.154) mm / (5700 * 4^2) = 0.3666 ns = the measured dt exactly.
MESH_EDGE_FACTOR = 2.154

# --- cell count ------------------------------------------------------------------------
REF_CELLS, REF_SCALE = 45711, 0.8      # measured: the production quad mesh
REF_DOF = 1_469_690                    # measured: tools/gpu_probe.py, degree 4 on that mesh
WIDE_CELL_FACTOR = 1.42                # measured: wavefield_snap_w165 / _base file sizes at
                                       # equal frame count. Less than the 1.77 width ratio
                                       # because the added margin is coarse steel and water.
TRI_CELL_FACTOR = 2.0                  # recombination merges 2 triangles into 1 quad

# --- cost ------------------------------------------------------------------------------
# MEASURED at the production configuration (1.47 M unknowns), which is what these are
# normalised by. GPU: 163,680 x 2.820 ms = 7.69 min, matching the 7.7 min on record.
# CPU: 163,680 x 52 ms = 2.36 h, matching the 2.4 h on record. tools/gpu_probe.py quotes
# 2.500 and 58.75 ms/step, but that probe times the bare matvec loop - these two include
# the source assembly and recording the solve actually does every step.
MS_PER_STEP_REF = {Device.GPU: 2.820, Device.CPU: 52.0}

# --- disk ------------------------------------------------------------------------------
# All measured off data/results. Channel data is resampled onto k-Wave's 380 MHz grid, so it
# scales with record length and not with the mesh: 44.0 MB for 60 us.
BYTES_CHANNEL_PER_S = 44.0e6 / 60.0e-6
# Snapshots: div and curl, float32, on a DG2 space ~= 8 samples per cell (measured 7.8).
BYTES_SNAP_PER_CELL_FRAME = 8 * 2 * 4
BYTES_MESH_PER_CELL = 165              # .msh + .h5 + .xdmf together, measured 7.5 MB


@dataclass(frozen=True)
class Quantity:
    """One number for the consequences panel. `is_estimate` drives the ~ prefix in the UI."""
    name: str
    value: float
    unit: str
    is_estimate: bool = True
    note: str = ""


@dataclass(frozen=True)
class Resolution:
    """Nodes per wavelength per material, plus which case binds."""
    cases: tuple[tuple[str, float], ...]
    binding: str
    value: float
    f_upper: float

    def as_quantity(self) -> Quantity:
        return Quantity("nodes per wavelength", self.value, "nodes", True,
                        f"binding case: {self.binding} at {self.f_upper/1e6:.1f} MHz")


def target_h(config: RunConfig, sc: scen.Scenario) -> dict[str, float]:
    """Target cell size per region, mm, AFTER --scale. What the size field asks for.

    Every size in the mesh script is multiplied by --scale, including the array face and the
    notch override, so one factor covers all of them.
    """
    h = {"water": sc.h_water, "steel": sc.h_steel, "array": sc.h_array,
         "notch": config.h_notch if config.h_notch is not None else sc.h_notch}
    if config.staircase_id:
        # The 50 um pixel risers are forced GEOMETRY: they exist whatever the size field
        # says, so they enter the CFL limit even though nobody asked for cells that small.
        # This is the C4 trap - rasterising the ID arc costs a 6x smaller time step.
        h["stair"] = sc.h_stair
    return {k: v * config.scale for k, v in h.items()}


def nodes_per_wavelength(config: RunConfig, sc: scen.Scenario) -> Resolution:
    """N = p * lambda / h per material, at the pulse's UPPER usable frequency.

    Sized at 2*f0 = 8 MHz, not at f0: a 1-cycle burst is ~100% bandwidth, and a mesh that
    resolves only the centre frequency low-passes its own pulse in transit, which costs
    depth resolution directly (docs/lessons.md, "the bandwidth trap").

    STEEL SHEAR IS THE CRITERION, WHICH SURPRISES PEOPLE. Steel is the fast material, so the
    instinct is that it needs the coarsest cells - but the SHEAR wave is what images the
    crack, its speed is 3100 m/s against water's 1500, and the mesh gives steel exactly 2x
    the cell size (H_STEEL = 2 * H_WATER). Those two ratios nearly cancel, so steel shear
    lands within a few percent of water and both sit at the threshold: 3.46 in steel shear
    against 3.35 in water at the production settings. Steel COMPRESSIONAL, the number people
    reach for first, is never the criterion - its wavelength is 1.8x longer.
    Two regions are deliberately excluded: the array face (sized to resolve 0.3 mm elements,
    a geometry requirement) and the notch (exact faces at any cell size - conformity comes
    from the boundary, which is the whole geometric advantage over a grid).
    """
    h = target_h(config, sc)
    lam = sc.f_upper
    cases = (("water", config.degree * (sc.c_f / lam) * 1e3 / h["water"]),
             ("steel shear", config.degree * (sc.c_s / lam) * 1e3 / h["steel"]),
             ("steel compressional", config.degree * (sc.c_p / lam) * 1e3 / h["steel"]))
    binding, value = min(cases, key=lambda kv: kv[1])
    return Resolution(cases, binding, value, sc.f_upper)


def scale_for_nodes(config: RunConfig, sc: scen.Scenario,
                    target: float = NODES_MIN) -> float:
    """The --scale that would put the binding case at `target` nodes per wavelength.

    N ~ 1/h ~ 1/scale, so this is one proportion. Rounded DOWN to 0.05 because a scale that
    lands 0.01 short of the threshold would re-trigger the warning it was meant to clear.
    """
    r = nodes_per_wavelength(config, sc)
    return math.floor(config.scale * r.value / target * 20.0) / 20.0


def estimated_dt(config: RunConfig, sc: scen.Scenario) -> Quantity:
    """dt = CFL * h / (c p^2), minimised over regions. AN ESTIMATE, and here is why.

    The real dt comes from the SHORTEST ACTUAL EDGE in the mesh over the fastest wave speed
    in that cell, and the shortest actual edge is about 2.15x smaller than the target size
    the size field asked for (gmsh grades; a quad's short side is not its target). That
    factor is applied here as MESH_EDGE_FACTOR, measured on the production mesh - without it
    this function is optimistic by the same 2.15x and every step count and runtime built on
    it is wrong by the same factor. The number the run reports may still differ: it depends
    on the mesh gmsh actually produced, which nothing outside gmsh can predict.
    """
    h = target_h(config, sc)
    c = {"water": sc.c_f, "array": sc.c_f, "steel": sc.c_p,
         # The notch sits in steel, so its CFL limit is set by the fast compressional speed -
         # that is why refining the crack tip is a time-step trap and not free accuracy.
         "notch": sc.c_p, "stair": sc.c_p}
    dts = {k: config.cfl * (v * 1e-3 / MESH_EDGE_FACTOR) / (c[k] * config.degree ** 2)
           for k, v in h.items()}
    where = min(dts, key=lambda k: dts[k])
    return Quantity("time step", dts[where], "s", True,
                    f"binding region: {where}; real dt comes from the shortest mesh edge")


def estimated_steps(config: RunConfig, sc: scen.Scenario) -> Quantity:
    """ceil(t_end / dt). Inherits the dt estimate's uncertainty, nothing more."""
    dt = estimated_dt(config, sc)
    return Quantity("time steps", math.ceil(config.t_end / dt.value), "steps", True,
                    f"at ~{dt.value*1e9:.4f} ns")


def estimated_cells(config: RunConfig) -> Quantity:
    """Cell count, scaled off the measured production mesh.

    Cells go as 1/scale^2 because every target size carries the same --scale factor and the
    domain area is fixed. No scenario argument: this needs only the knobs, and keeping it
    scenario-free is what lets estimated_disk() be called from a form with nothing loaded.
    """
    n = REF_CELLS * (REF_SCALE / config.scale) ** 2
    if config.artifact_reduction is ArtifactReduction.WIDE_DOMAIN:
        n *= WIDE_CELL_FACTOR
    if not config.quad:
        n *= TRI_CELL_FACTOR
    if config.staircase_id:
        # 50 um cells in a band along the whole ID arc, on top of everything else.
        # ponytail: crude flat factor - the only staircase mesh on disk is the healthy tri
        # one at scale 1.0, so there is nothing to fit against at production settings.
        n *= 1.5
    return Quantity("cells", round(n), "cells", True, "measured 45,711 at scale 0.8, quad")


def estimated_dof(config: RunConfig) -> Quantity:
    """Unknowns = 2 displacement components on a degree-p node layout.

    A quad carries p^2 nodes once sharing is accounted for, a triangle about half that.
    Check: 45,711 * 4^2 * 2 = 1.463 M against the 1.4697 M gpu_probe reported. Cost scales
    with THIS, not with cells, which is why degree shows up in the runtime estimate twice -
    once through the step count and once through the work per step.
    """
    per_cell = config.degree ** 2 * (2 if config.quad else 1)
    return Quantity("unknowns", round(estimated_cells(config).value * per_cell), "dof", True)


def ms_per_step(config: RunConfig, history: list[dict] | None = None) -> Quantity:
    """Cost of one time step, from past runs on this device if we have any.

    The kernel is memory-bandwidth bound - one sparse matvec, ~2 flops per 12 bytes fetched -
    so ms/step is proportional to unknowns, and a single-parameter model through the origin
    fits it. `history` entries are manifest dicts: {"device": "gpu"|"cpu", "dof" or "cells",
    "ms_per_step"}. Least squares through the origin, so one past run is already enough and
    more runs just tighten the slope.
    """
    dof = estimated_dof(config).value
    rows = []
    for h in history or []:
        dev = h.get("device")
        dev = dev.value if isinstance(dev, Device) else str(dev).lower()
        ms = h.get("ms_per_step")
        d = h.get("dof") or (h["cells"] * config.degree ** 2 * 2 if h.get("cells") else None)
        if dev == config.device.value and ms and d:
            rows.append((float(d), float(ms)))
    if rows:
        slope = sum(d * ms for d, ms in rows) / sum(d * d for d, _ in rows)
        return Quantity("ms per step", slope * dof, "ms", True,
                        f"fitted on {len(rows)} past run(s) on this device")
    slope = MS_PER_STEP_REF[config.device] / REF_DOF
    return Quantity("ms per step", slope * dof, "ms", True,
                    f"measured anchor: {MS_PER_STEP_REF[config.device]:.3f} ms/step at "
                    f"1.47 M unknowns on {config.device.value.upper()}")


def estimated_runtime(config: RunConfig, sc: scen.Scenario,
                      history: list[dict] | None = None) -> Quantity:
    """Wall-clock for the FORWARD stage: steps x ms/step.

    Meshing (a minute or two) and imaging (seconds) are not in here - they are separate jobs
    with their own cards in the queue, and lumping them would hide which one is slow. The
    measured anchors already absorb assembly and matrix upload, so no extra constant term.
    """
    steps = estimated_steps(config, sc).value
    ms = ms_per_step(config, history)
    return Quantity("solve time", steps * ms.value / 1e3, "s", True, ms.note)


def estimated_disk(config: RunConfig) -> Quantity:
    """Bytes one full pipeline writes: channel data + optional snapshots + mesh.

    Snapshots dominate by an order of magnitude, which is why they are opt-in: two float32
    fields on a DG2 space, every frame, is 700-970 MB at 240 frames.
    """
    channel = BYTES_CHANNEL_PER_S * config.t_end
    cells = estimated_cells(config).value
    snaps = config.snapshots * cells * BYTES_SNAP_PER_CELL_FRAME
    mesh = cells * BYTES_MESH_PER_CELL
    return Quantity("disk", channel + snaps + mesh, "B", True,
                    f"channel {channel/1e6:.0f} MB + snapshots {snaps/1e6:.0f} MB "
                    f"+ mesh {mesh/1e6:.0f} MB")


def all_estimates(config: RunConfig, sc: scen.Scenario,
                  history: list[dict] | None = None) -> tuple[Quantity, ...]:
    """Everything the consequences panel shows, in reading order."""
    return (nodes_per_wavelength(config, sc).as_quantity(),
            estimated_dt(config, sc), estimated_steps(config, sc),
            estimated_cells(config), estimated_dof(config),
            estimated_runtime(config, sc, history), estimated_disk(config))


def consequences(config: RunConfig, history: list[dict] | None = None) -> dict:
    """Flat mapping for widgets/consequences.py, which asks by key rather than by dataclass.

    Loads the scenario from cache, never the container: this runs on every keystroke in the
    form. Units are the ones that widget's formatters expect - seconds, MB, plain counts.
    """
    sc = scen.load()
    r = nodes_per_wavelength(config, sc)
    return {"nodes_per_wavelength": r.value, "binding": r.binding,
            "dt": estimated_dt(config, sc).value,
            "steps": estimated_steps(config, sc).value,
            "runtime_s": estimated_runtime(config, sc, history).value,
            "disk_mb": estimated_disk(config).value / 1e6,
            "cells": estimated_cells(config).value, "dof": estimated_dof(config).value}


def demo() -> None:
    """Self-check against the MEASURED production run, which is the only real test here."""
    sc = scen.load()
    c = RunConfig()                                    # defaults = the published +20 deg run

    r = nodes_per_wavelength(c, sc)
    assert 3.2 < r.value < 3.7, r                      # measured: ~3.5 at 8 MHz
    assert r.binding in ("water", "steel shear"), r.binding
    assert dict(r.cases)["steel compressional"] > dict(r.cases)["steel shear"]
    # halving the cell size must double the resolution: the formula is linear in scale
    fine = nodes_per_wavelength(RunConfig(scale=0.4), sc)
    assert abs(fine.value - 2 * r.value) < 1e-9
    assert scale_for_nodes(RunConfig(scale=1.6), sc) < 1.6, "coarser mesh must ask for finer"

    dt = estimated_dt(c, sc)
    assert abs(dt.value - 0.3666e-9) / 0.3666e-9 < 0.02, f"{dt.value*1e9:.4f} ns vs 0.3666"
    assert "notch" in dt.note, dt.note                 # the notch binds, as the mesh warns

    n = estimated_steps(c, sc)
    assert abs(n.value - 163680) / 163680 < 0.02, n     # measured 163,680 steps to 60 us
    assert estimated_steps(RunConfig(degree=2), sc).value < n.value / 3, "p^2 in the CFL"

    assert abs(estimated_cells(c).value - 45711) < 1
    assert abs(estimated_dof(c).value - 1_469_690) / 1_469_690 < 0.01
    assert estimated_cells(RunConfig(scale=0.4)).value > 3.9 * 45711   # 1/scale^2

    gpu = estimated_runtime(c, sc)
    assert abs(gpu.value / 60 - 7.7) < 0.5, f"{gpu.value/60:.2f} min vs 7.7 measured"
    cpu = estimated_runtime(RunConfig(device=Device.CPU), sc)
    assert abs(cpu.value / 3600 - 2.4) < 0.2, f"{cpu.value/3600:.2f} h vs 2.4 measured"
    # a past run on this device must override the anchor, and it must be device-matched
    hist = [{"device": "gpu", "dof": 1_469_690, "ms_per_step": 5.64}]
    assert abs(estimated_runtime(c, sc, hist).value - 2 * gpu.value) < 1.0
    assert abs(estimated_runtime(RunConfig(device=Device.CPU), sc, hist).value
               - cpu.value) < 1e-6, "cpu must not be fitted on gpu timings"

    d = estimated_disk(c).value
    assert 44e6 < d < 60e6, d                          # channel 44 MB + a few MB of mesh
    snap = estimated_disk(RunConfig(snapshots=240)).value
    assert 700e6 < snap < 970e6, snap                  # measured 682-966 MB on disk
    wide = estimated_disk(RunConfig(snapshots=240,
                                    artifact_reduction=ArtifactReduction.WIDE_DOMAIN)).value
    # measured on disk: wavefield_snap_w165 966 MB + channel 44 MB + mesh ~11 MB
    assert abs(wide - 1021e6) / 1021e6 < 0.05, wide

    assert len(all_estimates(c, sc)) == 7
    # the widget contract: every key views/consequences.py renders, in its unit
    q = consequences(c)
    for k in ("nodes_per_wavelength", "dt", "steps", "runtime_s", "disk_mb", "binding"):
        assert k in q, k
    assert abs(q["disk_mb"] - estimated_disk(c).value / 1e6) < 1e-6
    print(f"derived.demo: ok  N={r.value:.2f} ({r.binding})  dt={dt.value*1e9:.4f} ns  "
          f"steps={n.value}  gpu={gpu.value/60:.2f} min  cpu={cpu.value/3600:.2f} h")


if __name__ == "__main__":
    demo()
