r"""
Is the FEM-vs-k-Wave geometry advantage ROBUST, or an artefact of one threshold?

The 20 deg head-to-head reported that we locate the notch to 0.08 mm against k-Wave's
0.41 mm, and size it to -3.8% against their -19.3%. Both numbers came from ONE choice of
analysis parameters: a 25%-of-peak threshold for the extent, one peak pick for position,
and one guard distance for the clutter region.

A claim that only holds at one threshold is not a result. This re-analyses the SAVED images
(no re-solve) across every reasonable choice and reports whether the ordering survives.

Reads results/compare/images_<angle>.npz written by repro/compare_images.py.

RUN
  ./run.ps1 python3 repro/metric_robustness.py --angle 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.tt_t_image import (EDGE_Z_BAND, FROZEN,  # noqa: E402
                            edge_clutter)

OUT = Path(__file__).resolve().parents[1] / "results" / "compare"
NOTCH_X, NOTCH_DEPTH = 38.25, 4.0
# Steering angle of the images being analysed. main() sets it from --angle; it only feeds
# lib.tt_t_image.edge_clutter, whose band mirrors for negative steering.
ANGLE_DEG = 20.0
TRUE_TIP_Z, TRUE_OD_Z = 25.525, 29.525          # notch spans this radial band on axis


def measure(img, x, z, thresh, guard, halfwidth, col_halfwidth):
    """Position + extent + clutter for one choice of analysis parameters."""
    X, Z = np.meshgrid(x, z, indexing="ij")
    R = np.hypot(X - FROZEN["x_c"] * 1e3, Z - FROZEN["z_c"] * 1e3)
    r_id, r_od = FROZEN["r_id"] * 1e3, FROZEN["r_od"] * 1e3
    in_wall = (R >= r_id) & (R <= r_od)
    crack_roi = in_wall & (np.abs(X - NOTCH_X) <= halfwidth) & (R >= r_od - NOTCH_DEPTH - 1.0)
    clutter = img[in_wall & (np.abs(X - NOTCH_X) > guard)]
    a = np.nan_to_num(img)
    pk = a[crack_roi].max()
    i = np.unravel_index(np.where(crack_roi, a, -np.inf).argmax(), a.shape)
    col = a[np.abs(x - NOTCH_X) <= col_halfwidth, :].max(axis=0)
    hit = np.where(col > thresh * pk)[0]
    ext = (z[hit.max()] - z[hit.min()]) if hit.size else np.nan
    rms = float(np.sqrt((clutter ** 2).mean()))
    out = dict(x=float(x[i[0]]), z=float(z[i[1]]), ext=float(ext), pk=float(pk),
               cnr=float(20 * np.log10(pk / rms)))
    # Same pinned edge-clutter numbers as compare_images, from the same function.
    out.update(edge_clutter(img, x, z, float(pk), angle_deg=ANGLE_DEG))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--angle", type=int, default=20)
    ap.add_argument("--tag", default="", help="read images_<angle><tag>.npz, so a "
                                             "variant comparison can be swept too")
    args = ap.parse_args()

    global ANGLE_DEG
    ANGLE_DEG = float(args.angle)
    d = np.load(OUT / f"images_{args.angle}{args.tag}.npz")
    x, z = d["x"], d["z"]
    labels = [k[:-4] for k in d.files if k.endswith("_img")]
    imgs = {l: d[f"{l}_img"] for l in labels}
    print(f"angle {args.angle} deg, grid {imgs[labels[0]].shape}, "
          f"dx {x[1]-x[0]:.4f} mm, dz {z[1]-z[0]:.4f} mm")
    print(f"datasets: {labels}\n")
    # The tallies below count the literal label "FEM", so they only mean anything for a
    # TWO-dataset comparison. On an N-way run the per-row winner column is still right but
    # the tally is not, so refuse it rather than print a confident wrong number.
    two_way = len(labels) == 2

    # --- 1. extent vs threshold -------------------------------------------------------
    print("NOTCH EXTENT [mm] vs threshold (true = 4.00; radial band 25.53-29.53 mm)")
    print(f"{'threshold':<12}" + "".join(f"{l:>12}" for l in labels) + f"{'winner':>10}")
    ext_win = []
    for th in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        vals = {l: measure(imgs[l], x, z, th, 6.0, 1.5, 0.4)["ext"] for l in labels}
        err = {l: abs(vals[l] - NOTCH_DEPTH) for l in labels}
        w = min(err, key=err.get)
        ext_win.append(w)
        print(f"{th:<12.2f}" + "".join(f"{vals[l]:12.2f}" for l in labels) + f"{w:>10}")
    print(f"  -> FEM closer to truth at {ext_win.count('FEM')}/{len(ext_win)} thresholds\n"
          if two_way else "  -> tally omitted: N-way run, read the winner column\n")

    # --- 2. position vs ROI half-width -------------------------------------------------
    print("CRACK POSITION error [mm] vs crack-ROI half-width (true x = 38.25)")
    print(f"{'halfwidth':<12}" + "".join(f"{l+' dx':>12}" for l in labels) + f"{'winner':>10}")
    pos_win = []
    for hw in (1.0, 1.5, 2.0, 3.0):
        vals = {l: measure(imgs[l], x, z, 0.25, 6.0, hw, 0.4) for l in labels}
        err = {l: abs(vals[l]["x"] - NOTCH_X) for l in labels}
        w = min(err, key=err.get)
        pos_win.append(w)
        print(f"{hw:<12.1f}" + "".join(f"{err[l]:12.3f}" for l in labels) + f"{w:>10}")
    print(f"  -> FEM closer at {pos_win.count('FEM')}/{len(pos_win)} ROI widths\n"
          if two_way else "  -> tally omitted: N-way run, read the winner column\n")

    # --- 3. CNR vs guard distance ------------------------------------------------------
    print("CRACK/CLUTTER RMS [dB] vs guard distance around the notch")
    print(f"{'guard mm':<12}" + "".join(f"{l:>12}" for l in labels) + f"{'winner':>10}")
    cnr_win = []
    for g in (3.0, 4.5, 6.0, 8.0, 12.0):
        vals = {l: measure(imgs[l], x, z, 0.25, g, 1.5, 0.4)["cnr"] for l in labels}
        w = max(vals, key=vals.get)
        cnr_win.append(w)
        print(f"{g:<12.1f}" + "".join(f"{vals[l]:12.1f}" for l in labels) + f"{w:>10}")
    print(f"  -> k-Wave higher at {cnr_win.count('k-Wave')}/{len(cnr_win)} guard distances\n"
          if two_way else "  -> tally omitted: N-way run, read the winner column\n")

    # --- 4. edge clutter, both pinned definitions --------------------------------------
    # Not a sweep: these are FIXED definitions (see lib/tt_t_image.edge_clutter). They are
    # printed here so the robustness run and the head-to-head quote the same numbers.
    print(f"EDGE CLUTTER [dB re each image's own crack peak], lower = cleaner")
    print(f"{'definition':<34}" + "".join(f"{l:>12}" for l in labels)
          + f"{'FEM-kWave':>12}")
    edge = {l: measure(imgs[l], x, z, 0.25, 6.0, 1.5, 0.4) for l in labels}
    xb = edge[labels[0]]["edge_x_band"]      # mirrored already if the steering is negative
    for key, name in (("edge_p95_db", f"x{xb[0]:.1f}-{xb[1]:.1f} "
                                      f"z{EDGE_Z_BAND[0]:.0f}-{EDGE_Z_BAND[1]:.0f} p95 "
                                      f"PINNED"),
                      ("edge_rms_db", f"x{xb[0]:.1f}-{xb[1]:.1f} "
                                      f"all-z RMS (variant)")):
        row = "".join(f"{edge[l][key]:12.2f}" for l in labels)
        exc = (f"{edge['FEM'][key] - edge['k-Wave'][key]:+12.2f}"
               if two_way and "FEM" in edge and "k-Wave" in edge else f"{'-':>12}")
        print(f"{name:<34}{row}{exc}")
    print("  -> positive excess = FEM dirtier. The two definitions DISAGREE in sign;\n"
          "     see lib/tt_t_image.edge_clutter for every definition measured.\n")

    # --- verdict -----------------------------------------------------------------------
    n = len(ext_win) + len(pos_win)
    fem = ext_win.count("FEM") + pos_win.count("FEM")
    print("=" * 72)
    if two_way:
        print(f"GEOMETRY (position + extent): FEM better in {fem}/{n} parameter choices")
        print(f"CONTRAST (CNR):               k-Wave better in {cnr_win.count('k-Wave')}/"
              f"{len(cnr_win)} parameter choices")
    else:
        print(f"N-WAY RUN ({len(labels)} datasets): no tally. These counts compare exactly two")
        print("datasets; read the per-row winner column above instead.")
    print("=" * 72)
    print("A claim that flips with the threshold is not a result. Both orderings must be\n"
          "unanimous, or near it, before either is quoted.")


if __name__ == "__main__":
    main()
