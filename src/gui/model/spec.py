r"""THE CONTRACT between the GUI and the backend. Every other GUI module reads this one.

Nothing here computes physics. The GUI's job is to build an argument list and hand it to the
same Docker commands we run by hand, so that a GUI run and a published run are the same run.
The moment this module starts deciding what dt should be, GUI results and published results
can diverge, and the project's central claim - that every figure reproduces from committed
code - stops being true.

TWO RULES THAT ARE NOT NEGOTIABLE
  1. Every job records the exact argv it ran, so it reproduces from a terminal without the app.
  2. No job may overwrite a git-tracked file. data/results holds the published record for the
     R&D challenge - the k-Wave +20 deg baseline and everything scored against it - and a GUI
     run with default parameters would otherwise regenerate exactly those filenames on top of
     them. Hence GUI_TAG_PREFIX and `guard_not_tracked` below.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace

# GUI runs are tagged so they cannot collide with the published record even by accident.
GUI_TAG_PREFIX = "gui_"


class Stage(enum.Enum):
    """One pipeline step = one queued job = one container."""
    MESH = "mesh"
    FORWARD = "forward"
    IMAGE = "image"
    FIGURES = "figures"


class Device(enum.Enum):
    CPU = "cpu"
    GPU = "gpu"


class Notch(enum.Enum):
    """What the defect is, in the mesh."""
    PRESENT = "present"        # the 4 mm x 1 mm slot: the production case
    ABSENT = "absent"          # healthy wall. Every visible feature is then numerical.
    FILLED = "filled"          # slot filled with a soft solid instead of void
    STAIRCASE = "staircase"    # slot rasterised onto a 50 um grid, to imitate a voxel model


class ArtifactReduction(enum.Enum):
    """How to treat the domain edges.

    NONE is the default and applies NO workaround: the plain absorbing boundary, standard
    domain. It is also what every published figure uses, which is the other reason it is the
    default - the app should reproduce the baseline unless asked not to.

    The other two carry FIXED specs, deliberately. Both were measured, and exposing their
    knobs would invite tuning a boundary treatment against an image, which is how you talk
    yourself into an artefact.
    """
    NONE = "none"
    SPONGE = "sponge"          # shear-matched boundary + graded sponge in the dead margins
    WIDE_DOMAIN = "wide"       # move the boundary out of reach instead of approximating it


# Fixed specs for the two treatments. See docs/lessons.md section 6 for why each number.
SPONGE_MM = 8.0        # the dead margin outboard of the aperture at the standard extent
SPONGE_DB = 40.0       # round-trip attenuation. An OPTIMUM, not a maximum: 60 and 200 dB
                       # both measured WORSE, because a steeper ramp reflects at its own onset.
WIDE_X_MIN = -45.0     # widened extent. Bounded by geometry, not taste: past x = -47.46 the
WIDE_X_MAX = 120.0     # pipe arc rises above the flat array plane. Max usable width 171.4 mm.


@dataclass(frozen=True)
class RunConfig:
    """Everything the user can set. Defaults ARE the published +20 deg configuration."""
    # --- beam and imaging
    angle: float = 20.0
    chain: str = "faithfulbf"          # imaging chain preset; the published one

    # --- mesh
    scale: float = 0.8                 # multiplies every target cell size; smaller = finer
    quad: bool = True                  # quadrilaterals: makes row-sum lumping exact
    h_notch: float | None = None       # override the cell size at the notch, mm
    notch: Notch = Notch.PRESENT

    # --- solver
    degree: int = 4                    # polynomial order
    cfl: float = 0.30
    t_end: float = 60.0e-6

    # --- edges
    artifact_reduction: ArtifactReduction = ArtifactReduction.NONE

    # --- compute
    device: Device = Device.GPU
    snapshots: int = 0                 # 0 = off. 240 writes ~700-970 MB, so it is opt-in.
    snap_window: tuple[float, float] | None = (18.0, 46.0)   # us

    # --- comparison
    compare_kwave: bool = True         # only meaningful while the scenario is the benchmark one

    def tag(self) -> str:
        """Filename tag. Mirrors the convention already on disk, prefixed for safety."""
        sign = "p" if self.angle >= 0 else "m"
        bits = [f"deg{self.degree}", f"s{self.scale:.2f}".replace(".", "p").rstrip("0")]
        if self.notch is not Notch.PRESENT:
            bits.append(self.notch.value)
        if self.artifact_reduction is ArtifactReduction.SPONGE:
            bits.append("sponge")
        elif self.artifact_reduction is ArtifactReduction.WIDE_DOMAIN:
            bits.append("w165")
        bits.append(f"{sign}{abs(self.angle):.0f}deg")
        return GUI_TAG_PREFIX + "_".join(bits)

    def mesh_name(self) -> str:
        """The .msh the mesh stage will produce. Must match mesh/ili_mesh.py's own naming."""
        n = "ili_mesh"
        if self.scale != 1.0:
            n += f"_s{self.scale:.1f}".replace(".", "p")
        if self.notch is Notch.ABSENT:
            n = n.replace("ili_mesh", "ili_mesh_healthy")
        if self.artifact_reduction is ArtifactReduction.WIDE_DOMAIN:
            n += "_w165"
        if not self.quad:
            n += "_tri"
        return n + ".msh"


@dataclass(frozen=True)
class JobSpec:
    """One container invocation."""
    stage: Stage
    config: RunConfig
    argv: list[str]                     # the command INSIDE the container
    gpu: bool = False                   # selects dvfenics:gpu and passes the device through
    outputs: list[str] = field(default_factory=list)   # expected files, relative to results
    label: str = ""


def _mesh_argv(c: RunConfig) -> list[str]:
    a = ["python3", "mesh/ili_mesh.py", "--scale", f"{c.scale}", "--no-plot"]
    if c.quad:
        a.append("--quad")
    if c.h_notch is not None:
        a += ["--h-notch", f"{c.h_notch}"]
    if c.notch is Notch.ABSENT:
        a.append("--no-notch")
    elif c.notch is Notch.FILLED:
        a.append("--notch-fill")
    elif c.notch is Notch.STAIRCASE:
        a.append("--staircase")
    if c.artifact_reduction is ArtifactReduction.WIDE_DOMAIN:
        a += ["--x-min", f"{WIDE_X_MIN}", "--x-max", f"{WIDE_X_MAX}"]
    return a


def _forward_argv(c: RunConfig) -> list[str]:
    a = ["python3", "-u", "repro/ili_forward.py",
         "--angle", f"{c.angle}", "--degree", f"{c.degree}", "--cfl", f"{c.cfl}",
         "--t-end", f"{c.t_end}", "--tag", c.tag(),
         "--mesh", f"results/ili_mesh/{c.mesh_name()}"]
    if c.device is Device.GPU:
        a.append("--gpu")
    # NONE and WIDE_DOMAIN both keep the plain boundary; SPONGE switches to the shear-matched
    # dashpot (i.e. drops --abc-legacy) and adds the graded layer.
    if c.artifact_reduction is ArtifactReduction.SPONGE:
        a += ["--sponge-mm", f"{SPONGE_MM}", "--sponge-db", f"{SPONGE_DB}"]
    else:
        a.append("--abc-legacy")
    if c.snapshots:
        a += ["--snapshots", f"{c.snapshots}"]
        if c.snap_window:
            a.append(f"--snap-window={c.snap_window[0]},{c.snap_window[1]}")
    return a


def _image_argv(c: RunConfig, kwave_case: str | None) -> list[str]:
    a = ["python3", "-u", "repro/compare_images.py", "--angle", f"{c.angle}",
         "--chain", c.chain, "--tag", c.tag(), "--no-overlay",
         "--ours", f"results/ili_forward/channel_data_{c.tag()}.npz"]
    if c.compare_kwave and kwave_case:
        a += ["--theirs", kwave_case]
    return a


def plan(c: RunConfig, kwave_case: str | None = None,
         stages: tuple[Stage, ...] = tuple(Stage)) -> list[JobSpec]:
    """The pipeline for one configuration, in order.

    `--no-overlay` is not optional: no published figure carries wall arcs or a marker over
    the crack, because being told where to look disqualifies a judgement of detectability.
    """
    gpu = c.device is Device.GPU
    out: list[JobSpec] = []
    if Stage.MESH in stages:
        out.append(JobSpec(Stage.MESH, c, _mesh_argv(c), gpu=False,
                           outputs=[f"ili_mesh/{c.mesh_name()}"], label="mesh"))
    if Stage.FORWARD in stages:
        out.append(JobSpec(Stage.FORWARD, c, _forward_argv(c), gpu=gpu,
                           outputs=[f"ili_forward/channel_data_{c.tag()}.npz"],
                           label=f"solve {c.angle:+.0f} deg"))
    if Stage.IMAGE in stages:
        out.append(JobSpec(Stage.IMAGE, c, _image_argv(c, kwave_case), gpu=False,
                           outputs=[f"compare/images_{c.angle:.0f}_{c.tag()}_nooverlay.npz"],
                           label="image"))
    return out


def demo() -> None:
    """Self-check: the defaults must reproduce the published solve, modulo the safety tag."""
    c = RunConfig()
    fwd = _forward_argv(c)
    for expect in ("--angle", "20.0", "--degree", "4", "--gpu", "--abc-legacy"):
        assert expect in fwd, (expect, fwd)
    assert "--sponge-mm" not in fwd, "NONE must apply no workaround"
    assert c.tag().startswith(GUI_TAG_PREFIX), "GUI runs must not collide with the record"
    assert c.mesh_name() == "ili_mesh_s0p8.msh", c.mesh_name()

    w = replace(c, artifact_reduction=ArtifactReduction.WIDE_DOMAIN)
    assert "--x-min" in _mesh_argv(w) and w.mesh_name() == "ili_mesh_s0p8_w165.msh"
    assert "--abc-legacy" in _forward_argv(w), "widening keeps the plain boundary"

    s = replace(c, artifact_reduction=ArtifactReduction.SPONGE)
    assert "--sponge-mm" in _forward_argv(s) and "--abc-legacy" not in _forward_argv(s)

    h = replace(c, notch=Notch.ABSENT)
    assert "--no-notch" in _mesh_argv(h) and "healthy" in h.mesh_name()

    assert [j.stage for j in plan(c)][:3] == [Stage.MESH, Stage.FORWARD, Stage.IMAGE]
    print("spec.demo: ok")


if __name__ == "__main__":
    demo()
