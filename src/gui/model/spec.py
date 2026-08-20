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
    # NOTE there is no STAIRCASE member. --staircase rasterises the INNER WALL ARC, not the
    # notch, and it composes with --no-notch - the C4 experiment ran it on a healthy wall, so
    # the mesh on disk is ili_mesh_healthy_stair_tri. It is a separate axis: see
    # RunConfig.staircase_id.


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
    # Rasterise the inner wall onto a 50 um pixel staircase, imitating how a Cartesian-grid
    # code represents a curved boundary. This is the C4 experiment: the ONE variable that
    # isolates "our mesh conforms" from everything else. Orthogonal to `notch`.
    staircase_id: bool = False

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

    def tag(self) -> str:
        """Filename tag. Mirrors the convention already on disk, prefixed for safety."""
        sign = "p" if self.angle >= 0 else "m"
        # :g, matching mesh_stem. The old form was f"s{scale:.2f}" then rstrip("0"), which
        # turned scale 1.0 into the truncated "s1p" - and would have collided 1.0 with 1.00.
        bits = [f"deg{self.degree}", f"s{self.scale:g}".replace(".", "p")]
        if self.notch is not Notch.PRESENT:
            # "healthy", not "absent": the record on disk is channel_data_..._healthy_p20deg,
            # and a GUI healthy run should sort next to it rather than under another word for
            # the same thing. mesh_stem() already says healthy.
            bits.append("healthy" if self.notch is Notch.ABSENT else self.notch.value)
        if self.staircase_id:
            bits.append("stair")
        if self.artifact_reduction is ArtifactReduction.SPONGE:
            bits.append("sponge")
        elif self.artifact_reduction is ArtifactReduction.WIDE_DOMAIN:
            bits.append("w165")
        bits.append(f"{sign}{abs(self.angle):.0f}deg")
        return GUI_TAG_PREFIX + "_".join(bits)

    def mesh_stem(self) -> str:
        """Mirror mesh/ili_mesh.py's own stem construction, in the same order.

        This is duplicated logic and that is a liability, so it is pinned by demo() against
        the real filenames sitting in data/results/ili_mesh. Getting it wrong is not cosmetic:
        the forward stage passes this name to --mesh, so a mismatch means either "file not
        found" or, far worse, solving on the wrong mesh.

        The backend puts every variant-producing flag in the stem precisely so one variant
        cannot overwrite another. Note `:g` formatting, not a fixed precision - `--scale 0.65`
        is `_s0p65`, and rounding it to `_s0p7` would point at a different mesh.
        """
        stem = "ili_mesh" if self.notch is not Notch.ABSENT else "ili_mesh_healthy"
        if abs(self.scale - 1.0) > 1e-9:
            stem += f"_s{self.scale:g}".replace(".", "p")
        if self.artifact_reduction is ArtifactReduction.WIDE_DOMAIN:
            stem += f"_w{WIDE_X_MAX - WIDE_X_MIN:.0f}"
        if self.notch is Notch.FILLED:
            stem += "_fill"
        if self.staircase_id:
            stem += "_stair"
        if not self.quad:
            stem += "_tri"
        if self.h_notch is not None:
            stem += f"_hn{self.h_notch:g}".replace(".", "p")
        return stem

    def mesh_name(self) -> str:
        return self.mesh_stem() + ".msh"


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
    if c.staircase_id:
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


def _figures_argv(c: RunConfig) -> list[str]:
    """The wavefield animation. Only meaningful if the solve wrote snapshots."""
    return ["python3", "-u", "viz/wavefield_gif.py",
            "--in", f"results/ili_forward/wavefield_{c.tag()}.npz",
            "--out", f"results/viz/{c.tag()}.gif",
            "--stride", "8", "--fps", "7", "--colors", "24", "--smooth", "2"]


def _image_argv(c: RunConfig) -> list[str]:
    """Image OUR result. No k-Wave comparison: that is deliberately not a GUI concept.

    The published head-to-head is complete and archived, and any further comparison will be
    run by hand with compare_images.py --theirs, which is untouched. Building the app around a
    comparison it will rarely make meant every screen, guard and manifest had to keep
    reasoning about whether a reference case happened to be on disk.
    """
    return ["python3", "-u", "repro/compare_images.py", "--angle", f"{c.angle}",
            "--chain", c.chain, "--tag", c.tag(), "--no-overlay",
            "--ours", f"results/ili_forward/channel_data_{c.tag()}.npz"]


def plan(c: RunConfig,
         stages: tuple[Stage, ...] = tuple(Stage)) -> list[JobSpec]:
    """The pipeline for one configuration, in order.

    `--no-overlay` is not optional: no published figure carries wall arcs or a marker over
    the crack, because being told where to look disqualifies a judgement of detectability.
    """
    gpu = c.device is Device.GPU
    out: list[JobSpec] = []
    # outputs must list EVERY file a stage writes, not just the interesting one: the runner's
    # tracked-file guard and the manifest's output collection both see only what is declared.
    if Stage.MESH in stages:
        stem = c.mesh_stem()
        out.append(JobSpec(Stage.MESH, c, _mesh_argv(c), gpu=False,
                           outputs=[f"ili_mesh/{stem}{e}" for e in (".msh", ".xdmf", ".h5")],
                           label="mesh"))
    if Stage.FORWARD in stages:
        fwd_out = [f"ili_forward/channel_data_{c.tag()}.npz"]
        if c.snapshots:
            # The snapshot dump is 700-970 MB and is the FIGURES stage's own input. Omitting
            # it meant no manifest ever recorded the largest file the run produced, and the
            # animation stage pointed at something nothing had declared.
            fwd_out.append(f"ili_forward/wavefield_{c.tag()}.npz")
        out.append(JobSpec(Stage.FORWARD, c, _forward_argv(c), gpu=gpu,
                           outputs=fwd_out, label=f"solve {c.angle:+.0f} deg"))
    if Stage.IMAGE in stages:
        sign = "p" if c.angle >= 0 else "m"
        out.append(JobSpec(
            Stage.IMAGE, c, _image_argv(c), gpu=False,
            outputs=[f"compare/images_{c.angle:.0f}_{c.tag()}_nooverlay.npz",
                     f"compare/compare_{sign}{abs(c.angle):.0f}deg_{c.tag()}_nooverlay.png"],
            label="image"))
    if Stage.FIGURES in stages and c.snapshots:
        out.append(JobSpec(Stage.FIGURES, c, _figures_argv(c), gpu=False,
                           outputs=[f"viz/{c.tag()}.gif"], label="animation"))
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

    # Pin mesh_stem() against filenames that actually exist, because it duplicates logic that
    # lives in mesh/ili_mesh.py and silent drift there means solving on the wrong mesh.
    cases = {
        "ili_mesh_s0p8": dict(),
        "ili_mesh": dict(scale=1.0),
        "ili_mesh_healthy_s0p8": dict(notch=Notch.ABSENT),
        "ili_mesh_s0p8_w165": dict(artifact_reduction=ArtifactReduction.WIDE_DOMAIN),
        "ili_mesh_w165_tri": dict(scale=1.0, quad=False,
                                  artifact_reduction=ArtifactReduction.WIDE_DOMAIN),
        "ili_mesh_healthy_tri": dict(scale=1.0, quad=False, notch=Notch.ABSENT),
        "ili_mesh_healthy_stair_tri": dict(scale=1.0, quad=False, notch=Notch.ABSENT,
                                           staircase_id=True),
    }
    for want, kw in cases.items():
        got = replace(c, **kw).mesh_stem()
        assert got == want, f"mesh_stem: got {got}, want {want} for {kw}"
    # :g not :.1f - rounding 0.65 to 0p7 would point at a different mesh
    assert replace(c, scale=0.65).mesh_stem() == "ili_mesh_s0p65"
    assert replace(c, h_notch=0.15).mesh_stem() == "ili_mesh_s0p8_hn0p15"

    figs = plan(replace(c, snapshots=240), stages=(Stage.FIGURES,))
    assert figs and any("wavefield_gif.py" in a for a in figs[0].argv)
    assert not plan(c, stages=(Stage.FIGURES,)), "no snapshots -> no animation stage"
    mesh_job = plan(c, stages=(Stage.MESH,))[0]
    assert len(mesh_job.outputs) == 3, mesh_job.outputs
    snap_fwd = plan(replace(c, snapshots=240), stages=(Stage.FORWARD,))[0]
    assert any("wavefield_" in o for o in snap_fwd.outputs), snap_fwd.outputs
    assert len(plan(c, stages=(Stage.FORWARD,))[0].outputs) == 1
    assert "healthy" in replace(c, notch=Notch.ABSENT).tag()
    img = plan(c, stages=(Stage.IMAGE,))[0]
    assert "--theirs" not in img.argv, "the GUI does not do k-Wave comparisons"
    assert "--no-overlay" in img.argv, "no annotated figure, ever"
    assert replace(c, scale=1.0).tag() == "gui_deg4_s1_p20deg", replace(c, scale=1.0).tag()
    assert c.tag() == "gui_deg4_s0p8_p20deg", c.tag()

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
