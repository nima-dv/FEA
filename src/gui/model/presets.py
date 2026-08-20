r"""Named configurations, and what differs from the published record.

The two "Published" entries are not conveniences - they are the reference the UI marks every
other configuration against. `RunConfig()`'s defaults ARE the published +20 deg run, so the
+20 preset is the bare default and the -20 preset flips one field. If either ever needs more
than that, the defaults in spec.py have drifted away from the record and that is the bug.

The rest are the experiments that actually exist in data/results (plus the two controlled
geometry variants the mesh script supports but nobody has solved yet), so a user can
reproduce a published run without retyping seven fields and getting one of them wrong.

ONE TAGGING MISMATCH, deliberately NOT worked around here - spec.py is the contract and W2
does not edit it: Notch.ABSENT tags the channel data "absent", while the record on disk
spells the same thing "healthy" (channel_data_deg4_s0p8_healthy_p20deg.npz). mesh_stem()
already says "healthy", so only the solve output disagrees. Harmless - every GUI tag carries
the gui_ prefix so nothing can collide either way - but a GUI run of the healthy case will
not sort next to the published one.
"""
from __future__ import annotations

from dataclasses import fields, replace

try:                                  # inside the app: `from model import presets`
    from .spec import (GUI_TAG_PREFIX, ArtifactReduction, Device, Notch, RunConfig)
except ImportError:                   # run directly: python src/gui/model/presets.py
    from spec import (GUI_TAG_PREFIX, ArtifactReduction, Device, Notch,   # type: ignore
                      RunConfig)

PUBLISHED_P20 = RunConfig()                          # defaults are the record, by design
PUBLISHED_M20 = replace(PUBLISHED_P20, angle=-20.0)

# The published channel data on disk, for the demo to check the tags against. Without the
# gui_ prefix, which is exactly what the prefix is for.
PUBLISHED_TAGS = {20.0: "deg4_s0p8_p20deg", -20.0: "deg4_s0p8_m20deg"}

PRESETS: dict[str, RunConfig] = {
    # --- the record ---------------------------------------------------------------------
    "Published +20 deg": PUBLISHED_P20,
    "Published -20 deg": PUBLISHED_M20,
    # --- experiments with results already on disk ----------------------------------------
    # Defect-free wall: every visible feature in the image is then numerical, which is what
    # makes it the clutter floor the notch response is measured against.
    "Healthy wall (clutter floor)": replace(PUBLISHED_P20, notch=Notch.ABSENT),
    # Both boundary treatments came back negative against the plain absorbing boundary. They
    # are here to be re-run, not because they are recommended.
    "Sponge layer +20 deg": replace(PUBLISHED_P20,
                                    artifact_reduction=ArtifactReduction.SPONGE),
    "Widened domain +20 deg": replace(PUBLISHED_P20,
                                      artifact_reduction=ArtifactReduction.WIDE_DOMAIN),
    # 240 frames over the window where the wall echo and the notch response arrive.
    "Wavefield snapshots +20 deg": replace(PUBLISHED_P20, snapshots=240),
    # The GPU acceptance gate ran 3 us on both devices; a 3 us CPU solve is minutes, not
    # hours, which is why the CPU warning is keyed to record length and not to the device.
    "CPU gate (3 us)": replace(PUBLISHED_P20, device=Device.CPU, t_end=3.0e-6),
    # --- designed but not yet solved -----------------------------------------------------
    # C4: the ID arc rasterised onto k-Wave's own 50 um grid, everything else identical, so a
    # difference in the image is attributable to geometry representation and nothing else.
    # Kept at production settings on purpose - "everything else identical" is the experiment.
    # The only staircase mesh on disk is ili_mesh_healthy_stair_tri (scale 1.0, tri, healthy
    # wall), which was never solved, so this one has no published counterpart yet.
    "Staircase ID (C4)": replace(PUBLISHED_P20, staircase_id=True),
    # C5: the crack void filled with k-Wave's "outside" material instead of left traction
    # free, to see whether that substitution is what costs them.
    "Notch filled (C5)": replace(PUBLISHED_P20, notch=Notch.FILLED),
}


def published_for(angle: float) -> RunConfig:
    """The published run this configuration should be marked against.

    Keyed on the SIGN of the steering angle, because +20 and -20 are two separate published
    solves; an angle of 15 deg is a variation on the +20 one.
    """
    return PUBLISHED_P20 if angle >= 0 else PUBLISHED_M20


def diff_vs_published(config: RunConfig) -> dict[str, tuple[object, object]]:
    """{field: (this value, published value)} for every field that differs.

    The UI puts a marker on each field this names, so it has to be exact: a false positive
    tells the user their run deviates from the record when it does not, and a false negative
    is worse - it lets a deviation reach a figure caption unnoticed.
    """
    ref = published_for(config.angle)
    return {f.name: (getattr(config, f.name), getattr(ref, f.name))
            for f in fields(RunConfig)
            if getattr(config, f.name) != getattr(ref, f.name)}


def demo() -> None:
    # The presets must reproduce the record: strip the safety prefix and the tag has to be
    # the filename that is actually on disk today.
    for angle, want in PUBLISHED_TAGS.items():
        c = published_for(angle)
        assert c.angle == angle, c
        assert c.tag() == GUI_TAG_PREFIX + want, (c.tag(), want)
        assert diff_vs_published(c) == {}, "a published preset cannot differ from itself"
    assert PUBLISHED_P20.mesh_name() == "ili_mesh_s0p8.msh"      # the mesh on disk

    # every preset must be a real variation, and be marked against the right published run
    for name, c in PRESETS.items():
        d = diff_vs_published(c)
        assert (d == {}) == name.startswith("Published"), (name, d)
        assert "angle" not in d or abs(c.angle) != 20.0, (name, d)

    # the diff is per-field and exact: three changed fields, three markers, no more
    c = replace(PUBLISHED_P20, scale=0.6, degree=3, snapshots=240)
    assert set(diff_vs_published(c)) == {"scale", "degree", "snapshots"}, diff_vs_published(c)
    assert diff_vs_published(c)["scale"] == (0.6, 0.8)
    # a -20 deg variation is marked against the -20 record, so the angle is NOT a difference
    assert "angle" not in diff_vs_published(replace(PUBLISHED_M20, cfl=0.25))

    # if the published data is on disk, the tag really does name it (host layout only)
    from pathlib import Path
    res = Path(__file__).resolve().parents[3] / "data" / "results" / "ili_forward"
    if res.is_dir():
        for want in PUBLISHED_TAGS.values():
            assert (res / f"channel_data_{want}.npz").exists(), want
    print(f"presets.demo: ok ({len(PRESETS)} presets)")


if __name__ == "__main__":
    demo()
