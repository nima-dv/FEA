r"""
FEM vs k-Wave: the head-to-head. Both datasets, one identical beamformer.

Loads channel data from OUR forward solve and from THEIR k-Wave run for the same steering
angle and the same geometry, pushes both through the SAME imaging chain
(lib/tt_t_image.py = their beamformer), and reports the metrics that matter.

WHY THIS IS THE LOAD-BEARING COMPARISON
  Because image formation is byte-identical for both inputs, any difference in the output
  images is attributable to the FORWARD SOLVER - not to imaging choices, apodisation,
  travel-time models or display scaling. That is the only way to make "more accurate than
  k-Wave" a measurement rather than an argument.

THE HEADLINE METRIC
  A defect-free region of homogeneous steel must image as BLACK. Whatever it does image is
  numerical. So crack-peak / defect-free-wall-clutter is the number that separates a solver
  whose geometry is exact (conforming mesh, 0.05 um wall error) from one that rasterises it
  onto a 50 um Cartesian grid. Digitised from their published figures, k-Wave's worst-case
  clutter sits only 4.3-10.6 dB below the crack.

RUN
  ./run.ps1 python3 repro/compare_images.py --angle 0 \
      --ours   results/ili_forward/channel_data_p0deg.npz \
      --theirs <scratch>/kwave_cases/kwave_odnotch4mm_0.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.bf_loader import load_beamformer                      # noqa: E402
from lib.tt_t_image import (tt_t_image, image_metrics, params_from_kwave,  # noqa: E402
                            FROZEN)

OUT = Path(__file__).resolve().parents[1] / "results" / "compare"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--angle", type=float, required=True)
    ap.add_argument("--ours", default=None, help="our FEM channel_data .npz")
    ap.add_argument("--theirs", default=None, help="their extracted k-Wave .npz")
    ap.add_argument("--ours-healthy", default=None, help="our healthy-wall FEM .npz")
    ap.add_argument("--fem", action="append", default=[], metavar="LABEL=PATH",
                    help="an extra FEM dataset as LABEL=PATH. Repeat for an N-way comparison "
                         "(e.g. baseline vs a boundary-condition variant). The label appears "
                         "on the panel and as the table column heading.")
    ap.add_argument("--no-overlay", action="store_true",
                    help="draw the raw image only - no wall arcs, no true-notch marker")
    ap.add_argument("--tag", default="", help="suffix for the output figure, so a variant "
                                              "run does not clobber the canonical one")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    bf = load_beamformer()
    results = {}

    def run(label, path, params):
        d = np.load(path, allow_pickle=True)
        ch, dt = d["channel_data"], float(d["dt"])
        ang = float(d["angle"]) if "angle" in d else float(d["polar_incidence_angle"])
        if abs(ang - args.angle) > 1e-6:
            raise SystemExit(f"{label}: file steering {ang} != requested {args.angle}")
        print(f"\n[{label}] {Path(path).name}  {ch.shape}  peak |p| {np.abs(ch).max():.4g}")
        t0 = time.time()
        img, x_ax, z_ax = tt_t_image(bf, ch, dt, ang, params, verbose=True)
        m = image_metrics(img, x_ax, z_ax, params)
        print(f"    imaged {img.shape} in {time.time()-t0:.1f}s")
        results[label] = dict(img=img, x=x_ax, z=z_ax, m=m)
        return m

    if args.theirs:
        d = np.load(args.theirs, allow_pickle=True)
        run("k-Wave", args.theirs, params_from_kwave(d))
    if args.ours:
        run("FEM", args.ours, dict(FROZEN))
    if args.ours_healthy:
        run("FEM healthy", args.ours_healthy, dict(FROZEN))
    for spec in args.fem:
        if "=" not in spec:
            raise SystemExit(f"--fem wants LABEL=PATH, got {spec!r}")
        lab, pth = spec.split("=", 1)
        if lab in results:
            raise SystemExit(f"--fem label {lab!r} is already used")
        run(lab, pth, dict(FROZEN))

    if not results:
        raise SystemExit("nothing to do - pass --ours and/or --theirs")

    # --- comparison table ---------------------------------------------------------------
    # Headings quote the scenario, not literals, so they cannot go stale if the notch,
    # the wall or the steering angle changes.
    t_x = FROZEN["notch_x"] * 1e3
    t_d = FROZEN["notch_depth"] * 1e3
    t_od = (FROZEN["r_od"] + FROZEN["z_c"]) * 1e3
    keys = [("crack_peak", "crack peak (arb)", "{:.4g}"),
            ("crack_x_mm", f"crack x [mm]  (true {t_x:.2f})", "{:.2f}"),
            ("crack_z_mm", f"crack z [mm]  (OD {t_od:.2f})", "{:.2f}"),
            ("notch_extent_mm", f"notch extent [mm]  (true {t_d:.1f})", "{:.2f}"),
            ("clutter_rms", "wall clutter RMS", "{:.4g}"),
            ("clutter_p95", "wall clutter p95", "{:.4g}"),
            ("clutter_max", "wall clutter worst", "{:.4g}"),
            ("cnr_rms_db", "crack / clutter RMS   [dB]", "{:.1f}"),
            ("cnr_p95_db", "crack / clutter p95   [dB]", "{:.1f}"),
            ("cnr_worst_db", "crack / worst clutter [dB]", "{:.1f}")]
    labels = list(results)
    w = 22
    print("\n" + "=" * (34 + w * len(labels)))
    print(f"HEAD-TO-HEAD, steering {args.angle:+.0f} deg, identical beamformer")
    print("=" * (34 + w * len(labels)))
    print(f"{'metric':<34}" + "".join(f"{l:>{w}}" for l in labels))
    print("-" * (34 + w * len(labels)))
    for k, name, fmt in keys:
        row = "".join(f"{fmt.format(results[l]['m'][k]):>{w}}" for l in labels)
        print(f"{name:<34}{row}")
    print("-" * (34 + w * len(labels)))
    print("Higher dB = crack stands further above the numerical clutter floor.")
    print("A defect-free steel wall should image as black, so clutter is numerical.")

    # --- figures ------------------------------------------------------------------------
    # Normalise EACH panel to its OWN max, not to a shared max. The absolute amplitude
    # scales are not comparable between the two solvers: k-Wave drives a velocity source of
    # magnitude 2e-6 while our solver applies a unit traction, so a common normalisation
    # makes one panel look washed out for reasons that have nothing to do with image
    # quality. What IS comparable, and what the metrics table reports, is the ratio of
    # crack response to clutter WITHIN each image.
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.2), squeeze=False)
    for ax, l in zip(axes[0], labels):
        r = results[l]
        vmax = r["img"].max()
        db = 20 * np.log10(np.maximum(r["img"], vmax * 1e-4) / vmax)
        im = ax.imshow(db.T, origin="lower", aspect="equal", cmap="inferno",
                       vmin=-40, vmax=0,
                       extent=[r["x"][0], r["x"][-1], r["z"][0], r["z"][-1]])
        # --no-overlay draws the raw image only. Asked for by the research team: the wall
        # arcs and the lime true-notch marker tell the viewer where to look, so a reviewer
        # cannot judge unaided detectability with them on.
        if not args.no_overlay:
            th = np.linspace(-0.25, 0.25, 200)
            for rr, c in ((FROZEN["r_id"] * 1e3, "cyan"), (FROZEN["r_od"] * 1e3, "cyan")):
                ax.plot(rr * np.sin(th) + FROZEN["x_c"] * 1e3,
                        rr * np.cos(th) + FROZEN["z_c"] * 1e3, c=c, lw=0.8)
            z_od = (FROZEN["r_od"] + FROZEN["z_c"]) * 1e3
            ax.plot([t_x, t_x], [z_od - t_d, z_od], color="lime", lw=1.6)
        ax.set_xlim(r["x"][0], r["x"][-1]); ax.set_ylim(0, 40)
        ax.set_title(f"{l}   (crack/clutter {r['m']['cnr_rms_db']:.1f} dB RMS)",
                     fontsize=10)
        ax.set_xlabel("x [mm]")
    axes[0][0].set_ylabel("depth z [mm]")
    fig.colorbar(im, ax=axes[0].tolist(), label="dB re each panel's own max", shrink=0.85)
    note = ("" if abs(args.angle) > 1 else
            "\nNOTE: at 0 deg the TT-T (half-skip SHEAR) mode is barely generated - there is "
            "almost no mode conversion at normal incidence - so neither image is meaningful.")
    fig.suptitle(f"TT-T image, steering {args.angle:+.0f} deg - identical beamformer, "
                 f"forward solver is the only difference{note}", fontsize=11)
    suffix = f"_{args.tag}" if args.tag else ("_nooverlay" if args.no_overlay else "")
    p = OUT / (f"compare_{args.angle:+.0f}deg".replace("+", "p").replace("-", "m")
               + f"{suffix}.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"\nwrote {p}")
    np.savez_compressed(OUT / f"images_{int(args.angle)}{suffix}.npz",
                        **{f"{l}_img": results[l]["img"] for l in labels},
                        x=results[labels[0]]["x"], z=results[labels[0]]["z"])


if __name__ == "__main__":
    main()
