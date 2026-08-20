r"""Every user-facing string for every parameter, in one reviewable place.

WHY THIS IS A DATA FILE AND NOT SCATTERED setToolTip CALLS
  The form opened with code names for labels - "scale", "degree", "h_notch" - and three
  tooltips between fifteen fields. Someone who did not write the solver could not tell what to
  change or what a value meant. Copy that explains physics has to be correct, so it lives
  where it can be read and corrected in one pass rather than hunted through widget code.

WHAT EACH FIELD CARRIES
  label   what the user reads. Plain language, no code names.
  flag    the CLI flag, shown as secondary text so the dry-run command stays legible.
  unit    appended to the widget, not baked into the label.
  hint    ONE line, always visible under the field. The thing you most need to know.
  detail  the fuller answer, on hover and in the "?" expander. May be several sentences.
  choices for discrete parameters: value -> (label, one-line hint).
  warn    True if choosing anything but the default deserves visible caution.

RULE FOR THE HINTS
  Say what the parameter does to THE RESULT, and when you would change it. "Multiplies every
  target cell size" is what it does to the input, which the user can already see. "0.8 is what
  we publish; halving it costs about 4x the time" is what they actually need.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Help:
    label: str
    hint: str
    detail: str = ""
    flag: str = ""
    unit: str = ""
    choices: dict[str, tuple[str, str]] = field(default_factory=dict)
    warn: bool = False


HELP: dict[str, Help] = {
    # ------------------------------------------------------------------ beam
    "angle": Help(
        label="Beam angle", flag="--angle", unit=" deg",
        hint="Steering of the transmitted beam. Plus or minus 20 degrees is where this "
             "inspection works.",
        detail="The crack is imaged by a shear wave that converts from pressure at the inner "
               "wall and skips once off the outer wall. How much shear you generate depends "
               "on the incidence angle: near 20 degrees the converted wave is almost pure "
               "shear at about 45 degrees in the steel, which is what reaches the crack.\n\n"
               "0 degrees is not a useful test. At normal incidence there is almost no mode "
               "conversion, so neither this solver nor k-Wave produces a meaningful image - "
               "an empty result there is physics, not a bug.\n\n"
               "Sign convention: positive tilts the beam toward increasing x. The clutter "
               "pattern mirrors when you flip the sign, which is a useful check that a "
               "feature is beam-related rather than a property of the wall."),
    "chain": Help(
        label="Imaging chain", flag="--chain",
        hint="How the channel data is turned into an image. Leave on the published one.",
        detail="Both options run the research team's own beamformer; they differ in the "
               "settings passed to it.",
        choices={
            "faithfulbf": ("Published (faithfulbf)",
                           "Matches the research team's own script. This is what every "
                           "published figure uses."),
            "legacy": ("Legacy (pre-2026-08-19)",
                       "Frozen. Exists only to reproduce figures made before the chain was "
                       "corrected - it under-samples and omits the migration anti-alias, "
                       "which costs about 2.5 dB of clutter."),
        }),

    # ------------------------------------------------------------------ mesh
    "scale": Help(
        label="Mesh fineness", flag="--scale",
        hint="Multiplies every cell size. 0.8 is what we publish; smaller is finer and "
             "slower.",
        detail="Cell sizes are chosen per material from the wavelength, then all scaled by "
               "this one number - so it moves resolution everywhere at once without "
               "disturbing the balance between water and steel.\n\n"
               "Cost is steep and worth knowing before you turn it: halving the scale is "
               "roughly 4x the cells and, because the time step falls with the smallest cell, "
               "roughly 8x the run. Refinement has improved every metric monotonically so "
               "far, so this is the honest way to buy accuracy - it is simply not free.\n\n"
               "At 0.8 the tightest region is the water at about 3.35 nodes per wavelength "
               "at the pulse's upper edge. Below about 3 the mesh starts low-passing the "
               "pulse in transit, which costs depth resolution."),
    "degree": Help(
        label="Element order", flag="--degree",
        hint="Accuracy within each cell. 4 is published; 3 is for a quick look.",
        detail="Higher order resolves a wave with fewer cells, and on this problem it is "
               "usually a better buy than refining the mesh - the operator stays on the same "
               "mesh, so the time step does not fall as sharply as it would under "
               "refinement.\n\n"
               "The time step does still fall, as roughly 1/order-squared, so order 5 is not "
               "free either. Orders below 3 are not useful here: the pulse is short and "
               "low-order elements disperse it.",
        choices={
            "3": ("3 - quick look", "About 4x faster. Used for the wavefield animations."),
            "4": ("4 - published", "Every published number uses this."),
            "5": ("5 - convergence check", "For error bars, not for production."),
        }),
    "h_notch": Help(
        label="Cell size at the crack", flag="--h-notch", unit=" mm", warn=True,
        hint="Leave on auto. This sets the global time step, and refining it is the most "
             "expensive mistake available here.",
        detail="There is one time step for the whole model, and it is set by the smallest "
               "cell divided by the fastest wave speed. The crack sits in steel, where the "
               "pressure wave runs at 5700 m/s, so the smallest cell near the crack governs "
               "the entire simulation.\n\n"
               "Measured: refining the tip to 0.09 mm gave a 0.34 ns step and 178,000 steps, "
               "against 3.4 ns and 18,000 steps for ordinary steel cells. That is roughly a "
               "10x penalty for no accuracy gain, because the crack's shape is captured by "
               "the mesh conforming to its boundary, not by cells being small.\n\n"
               "Auto uses 0.30 mm before scaling, which exists to resolve the tip corners "
               "geometrically and nothing more."),
    "quad": Help(
        label="Quadrilateral cells", flag="--quad",
        hint="Leave on. It is what makes the fast time loop exact rather than approximate.",
        detail="On quadrilaterals the interpolation nodes coincide with a quadrature rule, so "
               "the mass matrix comes out exactly diagonal and the explicit time step needs "
               "no linear solve at all - just a division. Verified diagonal to 7e-17.\n\n"
               "On triangles that only holds for special low-order families; at order 4 the "
               "same lumping produces negative masses, which is unstable. Triangles are kept "
               "for the meshes that need them, not as a preference."),
    "staircase_id": Help(
        label="Staircase the inner wall", flag="--staircase", warn=True,
        hint="An experiment, not a setting: deliberately rasterises the curved wall onto a "
             "50 micron grid.",
        detail="This imitates what a Cartesian-grid code has no choice but to do, so that "
               "the ONE difference between two of our own runs is how the curved boundary is "
               "represented. It is how 'our mesh conforms' becomes a measurement instead of "
               "an argument.\n\n"
               "Result so far: 0.61 dB on a flat-bottomed axial notch, which is nearly "
               "grid-aligned anyway. The interesting case is a geometry a grid represents "
               "badly - a hole, or an off-axis crack - and that has not been run yet."),
    "notch": Help(
        label="Defect", flag="--no-notch / --notch-fill",
        hint="What is in the wall. The healthy wall is how you tell numerical artifacts from "
             "real echoes.",
        detail="",
        choices={
            "present": ("Crack (4 mm x 1 mm slot)",
                        "The production case. A traction-free void, which is the physically "
                        "correct condition for an open crack."),
            "absent": ("Healthy wall - no defect",
                       "Nothing is there, so EVERY feature in the image is numerical. This "
                       "is the cleanest possible measurement of an artifact floor."),
            "filled": ("Crack filled with soft solid",
                       "Tests whether fill material matters. It barely does: an impedance "
                       "step this large still reflects about 99% either way (0.07 dB)."),
        }),

    # ------------------------------------------------------------------ domain
    "artifact_reduction": Help(
        label="Artifact reduction", flag="--sponge-mm / --x-min",
        hint="How the domain edges are treated. None is the default and what we publish.",
        detail="Three independent tests now say the side boundaries are not the source of "
               "the residual edge artifact, so expect little from these. Specs are fixed "
               "deliberately: tuning a boundary treatment against an image is how you talk "
               "yourself into an artifact.",
        choices={
            "none": ("None - no workaround",
                     "The plain absorbing boundary at the standard domain size. What every "
                     "published figure uses."),
            "sponge": ("Absorbing layer (sponge)",
                       "Graded damping across the 8 mm dead margins at 40 dB round trip, "
                       "with the boundary matched to shear. Takes about 1 dB off the diffuse "
                       "edge clutter but shifts measured crack size by 0.12 mm, and the "
                       "direction flips with beam angle - which is why it is not published."),
            "wide": ("Widened domain (1.8x)",
                     "Moves the boundary out of reach instead of approximating it: margins "
                     "8 mm to 45 mm, so side-wall echoes arrive after the crack echo rather "
                     "than during it. Best measured configuration, and it costs about 50% "
                     "more time. Capped by geometry - the pipe arc rises above the array "
                     "plane past 1.84x."),
        }),
    "t_end": Help(
        label="Record length", flag="--t-end", unit=" us",
        hint="How long to simulate. 60 us captures the crack echo with margin; shorter is "
             "proportionally faster.",
        detail="The crack echo arrives at 33-40 us and the first side-wall return at about "
               "29.5 us, so anything below roughly 45 us starts cutting into the "
               "measurement. Short records are for plumbing tests, not results.\n\n"
               "Cost is exactly linear: the step count is the record length divided by the "
               "time step."),
    "cfl": Help(
        label="Time-step safety factor", flag="--cfl", warn=True,
        hint="0.30 in both this solver and k-Wave. Raising it is not a speed-up worth having.",
        detail="The explicit time step must stay below a stability limit set by cell size "
               "and wave speed; this factor is how far below. At 0.30 we sit at about 16% of "
               "the true limit, so there is real headroom.\n\n"
               "But stability is not accuracy. Time-stepping error grows with the square of "
               "the step, so a larger step buys speed by adding dispersion to the arrival "
               "time - and arrival time is the measurement. The value matches k-Wave's on "
               "purpose, so neither side is flattered."),

    # ------------------------------------------------------------------ compute
    "device": Help(
        label="Compute", flag="--gpu",
        hint="GPU is about 19x faster and validated to agree with the CPU.",
        detail="Only the time loop moves to the GPU; meshing and assembly stay on the CPU "
               "either way. The loop is memory-bandwidth bound, which is why the speed-up "
               "tracks memory bandwidth rather than core counts, and why double precision "
               "costs little.\n\n"
               "Validated on the quantity that matters: a full-length GPU run agrees with "
               "the stored CPU record to 1.7e-12 of a sample in arrival time, with no "
               "systematic lead or lag, and every imaging metric is identical. Bit-identical "
               "is impossible - the sparse library sums in a different order - so the gate "
               "scores timing, not a vector norm.",
        choices={
            "gpu": ("GPU", "About 7.7 minutes for a production run."),
            "cpu": ("CPU (one core)", "About 2.4 hours for the same run."),
        }),
    "snapshots": Help(
        label="Wavefield snapshots", flag="--snapshots", warn=True,
        hint="Frames saved for the animation. 240 frames writes 700-970 MB and slows the "
             "solve.",
        detail="Only needed if you want a wavefield animation or the interactive scrubber; "
               "the images and every metric come from the channel data, which is always "
               "written. 0 turns it off.\n\n"
               "The cost is disk and IO rather than arithmetic - two interpolations per "
               "saved frame - but a full run's dump is comfortably the largest file the "
               "pipeline produces."),
    "snap_window": Help(
        label="Snapshot window", flag="--snap-window", unit=" us",
        hint="Which slice of time to save frames from. 18-46 us covers conversion, the skip, "
             "and the crack echo.",
        detail="Before about 18 us the beam is still in the water and there is nothing to "
               "see; after about 46 us the interesting arrivals are past. Narrowing the "
               "window is the cheapest way to cut the snapshot file size without losing the "
               "part worth watching."),
}


def get(name: str) -> Help:
    """Never raise on a missing entry - a field with no copy is a gap, not a crash."""
    return HELP.get(name) or Help(label=name.replace("_", " "), hint="")


def demo() -> None:
    """Every field in the contract must have copy, and the copy must be usable."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataclasses import fields
    from model.spec import RunConfig

    have = set(HELP)
    want = {f.name for f in fields(RunConfig)}
    missing = want - have
    assert not missing, f"parameters with no help text: {sorted(missing)}"
    extra = have - want
    assert not extra, f"help for parameters that do not exist: {sorted(extra)}"

    for name, h in HELP.items():
        assert h.label and h.label[0].isupper(), f"{name}: label must read as a label"
        assert "_" not in h.label, f"{name}: {h.label!r} is still a code name"
        assert h.hint, f"{name}: no hint"
        assert len(h.hint) <= 120, f"{name}: hint is {len(h.hint)} chars, must fit one line"
        assert not h.hint.endswith(".."), name
        for val, (lab, hnt) in h.choices.items():
            assert lab and hnt, f"{name}.{val}: incomplete choice copy"

    # The three parameters most likely to be turned without understanding them.
    for name in ("h_notch", "cfl", "staircase_id"):
        assert HELP[name].warn, f"{name} should be flagged as needing caution"
        assert len(HELP[name].detail) > 200, f"{name}: detail must actually explain the trap"

    print(f"help.demo: ok ({len(HELP)} parameters, "
          f"{sum(len(h.choices) for h in HELP.values())} documented choices)")


if __name__ == "__main__":
    demo()
