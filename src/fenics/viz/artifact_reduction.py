r"""
The artifact-reduction figure: what each treatment actually removed.

WHY THIS SCRIPT EXISTS. The stock comparison figures cannot show artifact reduction, for two
reasons that are easy to miss:

  1. Every panel is normalised to ITS OWN maximum, so a panel with less clutter looks
     identical to one with more - the colour scale silently rescales the difference away.
  2. The absolute amplitude is not comparable between imaging chains at all: the migration
     anti-alias kernel integrates the data twice, so the adopted chain's raw pixel values are
     ~3500x smaller than the previous chain's. Comparing raw levels is meaningless.

Both are fixed by normalising every panel to ITS OWN CRACK PEAK and then sharing one colour
scale. That is exactly the basis every metric in this project uses (dB re crack peak), so the
picture and the numbers finally measure the same thing.

The right-hand panel is the honest part: a 0.8 dB change is real but sub-visual in a heat map,
so it is also plotted as a curve, where it is unambiguous.

RUN
  ./run.ps1 python3 viz/artifact_reduction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent))
from lib.tt_t_image import FROZEN  # noqa: E402

CMP = Path(__file__).resolve().parents[1] / "results" / "compare"
OUT = Path(__file__).resolve().parents[1] / "results" / "viz" / "artifact_reduction.png"

# (npz tag, key inside that npz, label). Order = the order the treatments were applied.
ARMS = [
    ("_legacybf", "FEM_img", "Starting point"),
    ("", "FEM_img", "+ imaging anti-alias"),
    ("_boundary", "FEM shear-matched + sponge_img", "+ boundary and sponge"),
]
FLOOR = -40.0  # dB re crack peak, matching the project's standard display range


def wall_masks(x, z):
    """in-wall mask, and the crack ROI used by every metric in this project."""
    X, Z = np.meshgrid(x, z, indexing="ij")
    R = np.hypot(X - FROZEN["x_c"] * 1e3, Z - FROZEN["z_c"] * 1e3)
    r_id, r_od = FROZEN["r_id"] * 1e3, FROZEN["r_od"] * 1e3
    in_wall = (R >= r_id) & (R <= r_od)
    t_x = FROZEN["notch_x"] * 1e3
    t_d = FROZEN["notch_depth"] * 1e3
    crack = in_wall & (np.abs(X - t_x) <= 1.5) & (R >= r_od - t_d - 1.0)
    return in_wall, crack, t_x


def load(tag, key):
    d = np.load(CMP / f"images_20{tag}.npz")
    if key not in d.files:
        raise SystemExit(f"images_20{tag}.npz has no '{key}'. keys: {d.files}")
    return np.nan_to_num(d[key]), d["x"], d["z"]


def main() -> None:
    panels = []
    for tag, key, label in ARMS:
        img, x, z = load(tag, key)
        in_wall, crack, t_x = wall_masks(x, z)
        pk = img[crack].max()
        db = 20 * np.log10(np.maximum(img / pk, 1e-9))       # dB re THIS panel's crack peak
        panels.append(dict(label=label, db=db, x=x, z=z, in_wall=in_wall, t_x=t_x))

    # LAYOUT. The wall is a thin arc - about 95 mm across and 18 mm deep - so at TRUE
    # aspect any panel of it is wide and short. Side-by-side panels therefore had to be
    # stretched vertically to fill the width, which distorted the arc. Stacking instead
    # keeps 1 mm in x equal to 1 mm in z, and has the better property that before and
    # after sit at the same x, so the eye compares the same column of wall.
    X0, X1, Z0, Z1 = 13.0, 86.0, 14.5, 30.5     # the imaged wall, cropped to the beam
    fig = plt.figure(figsize=(9.8, 8.0), dpi=150)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.15], hspace=0.28,
                          left=0.085, right=0.90, bottom=0.075, top=0.905)

    heat_axes, im = [], None
    for i, p in enumerate((panels[0], panels[-1])):
        ax = fig.add_subplot(gs[i, 0])
        shown = np.where(p["in_wall"], p["db"], np.nan)
        im = ax.pcolormesh(p["x"], p["z"], shown.T, cmap="inferno",
                           vmin=FLOOR, vmax=0.0, shading="auto", rasterized=True)
        ax.set_title(p["label"] if i == 0 else "After both treatments",
                     fontsize=11, pad=5)
        ax.set_ylabel("z [mm]", fontsize=9)
        ax.set_xlim(X0, X1)
        ax.set_ylim(Z0, Z1)
        ax.set_box_aspect((Z1 - Z0) / (X1 - X0))   # true scale at full width - see note
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.set_xticklabels([])
        heat_axes.append(ax)

    cb = fig.colorbar(im, ax=heat_axes, fraction=0.020, pad=0.012, aspect=34)
    cb.set_label("dB re each panel's own crack peak", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    # --- the curve carries the quantitative claim --------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    for p, style in zip(panels, ["-", "--", "-"]):
        lin = np.where(p["in_wall"], 10 ** (p["db"] / 20.0), np.nan)
        keep = p["in_wall"].any(axis=1) & (np.abs(p["x"] - p["t_x"]) > 3.0)
        with np.errstate(invalid="ignore"):
            prof = 20 * np.log10(np.sqrt(np.nanmean(lin ** 2, axis=1)))
        keep &= np.nan_to_num(prof, nan=-999.0) > -60.0    # drop unimaged columns
        ax.plot(p["x"][keep], prof[keep], style, lw=1.7, label=p["label"])
    ax.set_xlabel("x [mm]", fontsize=9)
    ax.set_ylabel("clutter [dB re crack peak]", fontsize=9)
    ax.set_title("Clutter along the wall, crack column excluded - lower is cleaner",
                 fontsize=11, pad=5)
    ax.set_xlim(heat_axes[0].get_xlim())          # share the x axis with the wall images
    ax.set_ylim(-34, -13)
    ax.grid(alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=3,
              borderaxespad=0.2, columnspacing=1.6)

    fig.suptitle("Artifact reduction at +20 deg - one shared scale, each panel referenced to "
                 "its own crack peak", fontsize=11.5, y=0.985)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e3:.0f} kB)")

    # Report the numbers the figure is claiming, so the caption cannot drift from it.
    print("\nclutter RMS over the whole wall, crack column excluded [dB re crack peak]:")
    base = None
    for p in panels:
        lin = np.where(p["in_wall"], 10 ** (p["db"] / 20.0), np.nan)
        with np.errstate(invalid="ignore"):
            col = 20 * np.log10(np.sqrt(np.nanmean(lin ** 2, axis=1)))
        keep = (np.abs(p["x"] - p["t_x"]) > 3.0) & (np.nan_to_num(col, nan=-999.0) > -60.0)
        v = 20 * np.log10(np.sqrt(np.nanmean(lin[keep, :] ** 2)))
        base = v if base is None else base
        print(f"  {p['label']:<24} {v:7.2f}   ({v - base:+.2f} vs start)")


if __name__ == "__main__":
    main()
