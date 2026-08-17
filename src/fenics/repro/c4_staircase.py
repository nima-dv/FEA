r"""
C4: does the clutter come FROM the staircase? The controlled experiment.

Everything else in this project compares our solver with k-Wave, which differ in many ways at
once. This compares OUR SOLVER WITH ITSELF, changing exactly one thing: whether the fluid/steel
ID interface is an exact circular arc or a 50 um pixel staircase (k-Wave's own grid spacing).
Same code, same element order, same target cell sizes, same source, same beamformer.

WHY A HEALTHY WALL
  A defect-free wall must image as BLACK. There is no scatterer, so every bit of energy in the
  image is numerical. That removes the need to separate "crack response" from "clutter" and
  makes the measurement a straight comparison of two clutter floors.

WHAT THIS CAN AND CANNOT CONCLUDE
  It CAN establish that staircasing a curved fluid/solid interface generates grid-scale
  scattering in an otherwise identical solver - i.e. the mechanism is real and is in the
  geometry representation, not in k-Wave's time-stepping.
  It CANNOT quantify k-Wave's clutter, for two reasons worth stating in any writeup:
    * k-Wave applies a ~2.5-pixel soft-interface blend which MITIGATES the staircase; our arm
      is unblended, so it is an upper bound on the effect at that pixel size.
    * the two arms do not cost the same. The staircase forces 28 um cells, so its explicit dt
      is 7x smaller for the same 60 us. It is given MORE compute and still resolves less -
      which is a stronger statement than a cost-matched comparison, but it is not cost-matched.

RUN
  ./run.ps1 python3 repro/c4_staircase.py \
      --conforming results/ili_forward/channel_data_c4_conforming_p20deg.npz \
      --staircase  results/ili_forward/channel_data_c4_staircase_p20deg.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.bf_loader import load_beamformer                                   # noqa: E402
from lib.tt_t_image import tt_t_image, FROZEN                              # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "compare"


def wall_stats(img, x, z):
    """Clutter statistics inside the pipe wall. No crack ROI: there is no crack."""
    X, Z = np.meshgrid(x, z, indexing="ij")
    R = np.hypot(X - FROZEN["x_c"] * 1e3, Z - FROZEN["z_c"] * 1e3)
    in_wall = (R >= FROZEN["r_id"] * 1e3) & (R <= FROZEN["r_od"] * 1e3)
    v = np.nan_to_num(img)[in_wall]
    return dict(rms=float(np.sqrt((v ** 2).mean())), p95=float(np.percentile(v, 95)),
                worst=float(v.max()), n=int(in_wall.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conforming", required=True)
    ap.add_argument("--staircase", required=True)
    ap.add_argument("--angle", type=float, default=20.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    bf = load_beamformer()
    res = {}
    for label, path in (("conforming", args.conforming), ("staircase", args.staircase)):
        d = np.load(path, allow_pickle=True)
        ch, dt = d["channel_data"], float(d["dt"])
        print(f"\n[{label}] {Path(path).name}  {ch.shape}  peak |p| {np.abs(ch).max():.4g}  "
              f"solver dt {float(d['dt_solver'])*1e9:.4f} ns")
        img, x, z = tt_t_image(bf, ch, dt, args.angle, dict(FROZEN), verbose=True)
        res[label] = dict(img=img, x=x, z=z, s=wall_stats(img, x, z),
                          dt=float(d["dt_solver"]))

    a, b = res["conforming"]["s"], res["staircase"]["s"]
    print("\n" + "=" * 78)
    print(f"C4: STAIRCASED vs CONFORMING ID, healthy wall, steering {args.angle:+.0f} deg")
    print("=" * 78)
    print(f"{'wall clutter':<26}{'conforming':>18}{'staircase':>18}{'ratio [dB]':>14}")
    print("-" * 78)
    for k, name in (("rms", "RMS"), ("p95", "p95"), ("worst", "worst pixel")):
        db = 20 * np.log10(max(b[k], 1e-30) / max(a[k], 1e-30))
        print(f"{name:<26}{a[k]:18.4g}{b[k]:18.4g}{db:+14.2f}")
    print("-" * 78)
    print(f"{'solver dt [ns]':<26}{res['conforming']['dt']*1e9:18.4f}"
          f"{res['staircase']['dt']*1e9:18.4f}"
          f"{'':>14}")
    print(f"{'wall pixels measured':<26}{a['n']:18d}{b['n']:18d}")
    print("-" * 78)
    db_rms = 20 * np.log10(max(b["rms"], 1e-30) / max(a["rms"], 1e-30))
    if db_rms > 1.0:
        print(f"CONCLUSION: staircasing the ID raises the numerical clutter floor by "
              f"{db_rms:+.2f} dB\n"
              f"in an otherwise IDENTICAL solver. The mechanism is the geometry "
              f"representation.")
    elif db_rms < -1.0:
        print(f"UNEXPECTED: the staircased arm is QUIETER by {-db_rms:.2f} dB. Do not report "
              f"the\nconforming-mesh advantage as a clutter mechanism until this is explained.")
    else:
        print(f"INCONCLUSIVE: {db_rms:+.2f} dB is within noise. At this pixel size the "
              f"staircase does\nnot measurably change the clutter floor; say so rather than "
              f"asserting the mechanism.")
    print("Caveat to quote with this number: k-Wave blends its interface over ~2.5 pixels,\n"
          "this arm does not, so it is an UPPER BOUND on the effect at 50 um.")

    # --- figure: the side-by-side that makes the point visually --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    vmax = max(res["conforming"]["img"].max(), res["staircase"]["img"].max())
    for ax, l in zip(axes, ("conforming", "staircase")):
        r = res[l]
        db = 20 * np.log10(np.maximum(r["img"], vmax * 1e-4) / vmax)
        im = ax.imshow(db.T, origin="lower", aspect="equal", cmap="inferno", vmin=-40, vmax=0,
                       extent=[r["x"][0], r["x"][-1], r["z"][0], r["z"][-1]])
        th = np.linspace(-0.25, 0.25, 200)
        for rr in (FROZEN["r_id"] * 1e3, FROZEN["r_od"] * 1e3):
            ax.plot(rr * np.sin(th) + FROZEN["x_c"] * 1e3,
                    rr * np.cos(th) + FROZEN["z_c"] * 1e3, c="cyan", lw=0.8)
        ax.set_xlim(r["x"][0], r["x"][-1]); ax.set_ylim(0, 40)
        ax.set_title(f"ID {l}   (wall clutter RMS {r['s']['rms']:.3g})", fontsize=10)
        ax.set_xlabel("x [mm]")
    axes[0].set_ylabel("depth z [mm]")
    # SHARED normalisation here, unlike compare_images.py. There the two panels came from
    # different solvers with different source conventions, so only within-image ratios were
    # comparable. Here both panels come from the SAME solver and the same unit traction, so the
    # absolute levels ARE comparable - and a shared scale is the whole point of the figure.
    fig.colorbar(im, ax=axes.tolist(), label="dB re the brighter panel's max", shrink=0.85)
    fig.suptitle(f"C4: healthy wall, one variable changed - exact ID arc vs 50 um pixel "
                 f"staircase ({args.angle:+.0f} deg, shared colour scale)", fontsize=11)
    p = OUT / "c4_staircase_vs_conforming.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"\nwrote {p}")
    np.savez_compressed(OUT / "c4_images.npz",
                        conforming=res["conforming"]["img"],
                        staircase=res["staircase"]["img"],
                        x=res["conforming"]["x"], z=res["conforming"]["z"])


if __name__ == "__main__":
    main()
