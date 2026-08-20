r"""
Crack minus healthy: isolate the defect response by removing the wall coherently.

The head-to-head reports crack-to-clutter WITHIN one image, which always leaves the question
of how much of the bright spot is really the crack and how much is wall structure that happens
to sit there. Subtracting a defect-free run of the SAME geometry answers it directly: the wall
echoes are common to both and cancel coherently in the channel domain, so what survives is the
crack and nothing else.

The two runs come from the same mesh generator with the same sizing fields, source and element
order, differing by the notch void. They are NOT, however, bit-for-bit identical elsewhere, and
that sets a floor on what the subtraction can resolve:

  * removing the notch removes the cells around it, so the mesh differs everywhere the sizing
    field reached, not only at the notch;
  * that changes the smallest cell and hence the global explicit time step (0.367 ns cracked
    against 0.435 ns healthy), so the two records are resampled onto k-Wave's 380 MHz base from
    different sample grids.

Measured consequence: the pre-arrival difference floor is ~2.3% of the difference peak, against
~0.002% for k-Wave's own two runs, which shared a grid exactly. The crack response is ~10% of
the cracked peak, so the subtraction still has ~13 dB of headroom over its own noise - usable,
and the numbers below bear that out - but this is a coherent-cancellation measurement with a
real floor, not an exact null. Report it that way.

k-Wave has no defect-free run at these settings, so this is a self-consistency measurement of
our solver, not a head-to-head. It is also the single cheapest thing to ask the research team
for: a no-crack run at the odnotch4mm settings would make it one.

WHAT TO EXPECT, AND WHAT WOULD BE ALARMING
  * The difference should be a COMPACT feature at the notch, and the rest of the wall should go
    dark. A difference image that still shows wall structure means the two runs were not
    actually identical apart from the notch.
  * The difference amplitude is SMALL relative to the wall echoes - a few percent - because the
    notch is a 1 mm feature in a 9.5 mm wall. Small is expected; it is the CONTRAST after
    subtraction that matters.

RUN
  ./run.ps1 python3 repro/baseline_subtract.py \
      --cracked results/ili_forward/channel_data_deg4_s0p8_p20deg.npz \
      --healthy results/ili_forward/channel_data_deg4_s0p8_healthy_p20deg.npz --angle 20
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
from lib.tt_t_image import (tt_t_image, image_metrics, CHAINS,   # noqa: E402
                            FROZEN)
from lib.paths import RESULTS

OUT = RESULTS / "compare"
NOTCH_X = 38.25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cracked", required=True)
    ap.add_argument("--healthy", required=True)
    ap.add_argument("--angle", type=float, default=20.0)
    ap.add_argument("--chain", default="faithfulbf", choices=sorted(CHAINS),
                    help="imaging chain preset (lib/tt_t_image.CHAINS). Default is the "
                         "published baseline; 'legacy' reproduces the _legacybf figure.")
    ap.add_argument("--tag", default="", help="suffix for the output figure. Without "
                    "it a variant run SILENTLY OVERWRITES the canonical figure that "
                    "the artifact pages embed, leaving their caption describing a "
                    "figure that is no longer there. This has happened; use a tag.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    A = np.load(args.cracked, allow_pickle=True)
    B = np.load(args.healthy, allow_pickle=True)
    ca, cb = A["channel_data"], B["channel_data"]
    dta, dtb = float(A["dt"]), float(B["dt"])
    if ca.shape != cb.shape or abs(dta - dtb) > 1e-18:
        raise SystemExit(f"not comparable: {ca.shape}@{dta} vs {cb.shape}@{dtb}")
    # Solver dt may legitimately differ (the healthy mesh has no notch cells, so its stable
    # step is larger); both are resampled onto k-Wave's 380 MHz base, which is what must match.
    print(f"cracked solver dt {float(A['dt_solver'])*1e9:.4f} ns, "
          f"healthy {float(B['dt_solver'])*1e9:.4f} ns -> both resampled to "
          f"{dta*1e9:.4f} ns, {ca.shape}")

    diff = ca - cb
    print(f"peak |p|: cracked {np.abs(ca).max():.4g}, healthy {np.abs(cb).max():.4g}, "
          f"difference {np.abs(diff).max():.4g} "
          f"({100*np.abs(diff).max()/np.abs(ca).max():.2f}% of cracked)")
    # Pre-arrival window: nothing physical has arrived yet, so whatever the difference holds
    # there is pure numerical noise and sets the floor this measurement can resolve.
    n_pre = int(15e-6 / dta)
    pre = np.abs(diff[:n_pre]).max()
    frac = 100 * pre / max(np.abs(diff).max(), 1e-30)
    print(f"pre-arrival (t < 15 us) difference floor {pre:.4g} = {frac:.3f}% of the difference "
          f"peak -> {20*np.log10(100/max(frac,1e-9)):.1f} dB of headroom")
    print("  (a floor, not a bug: the healthy mesh has no notch cells, so its stable dt differs"
          "\n   and the two records reach the 380 MHz base from different sample grids)")

    bf = load_beamformer()
    res = {}
    for label, ch in (("cracked", ca), ("healthy", cb), ("difference", diff)):
        img, x, z = tt_t_image(bf, ch, dta, args.angle, dict(FROZEN), verbose=False,
                               chain=args.chain)
        m = image_metrics(img, x, z, dict(FROZEN), angle_deg=args.angle)
        res[label] = dict(img=img, x=x, z=z, m=m)
        print(f"  {label:<11} notch-ROI peak {m['crack_peak']:.4g}  "
              f"at x {m['crack_x_mm']:.2f} mm  extent {m['notch_extent_mm']:.2f} mm  "
              f"wall clutter RMS {m['clutter_rms']:.4g}")

    d, h, c = res["difference"]["m"], res["healthy"]["m"], res["cracked"]["m"]
    print("\n" + "=" * 76)
    print(f"BASELINE SUBTRACTION, steering {args.angle:+.0f} deg")
    print("=" * 76)
    det = 20 * np.log10(max(c["crack_peak"], 1e-30) / max(h["crack_peak"], 1e-30))
    print(f"crack response vs a DEFECT-FREE wall at the same ROI : {det:+.2f} dB")
    print(f"  cracked notch-ROI peak {c['crack_peak']:.4g} against healthy "
          f"{h['crack_peak']:.4g}")
    print(f"in the HEALTHY image the notch ROI is {h['cnr_worst_db']:+.1f} dB relative to the "
          f"worst wall clutter,")
    print(f"i.e. there is no feature there at all - the 'peak' it finds is ordinary clutter.")
    print(f"\nafter subtraction: peak at x {d['crack_x_mm']:.2f} mm (true {NOTCH_X}), "
          f"extent {d['notch_extent_mm']:.2f} mm (true 4.0),")
    print(f"  crack/clutter {d['cnr_rms_db']:.1f} dB RMS, {d['cnr_worst_db']:.1f} dB vs worst")
    print("=" * 76)
    print("The wall is common to both runs and cancels; what survives is the defect. This is a\n"
          "self-consistency check on OUR solver - k-Wave has no defect-free run at these\n"
          "settings, which is the cheapest single thing to ask the research team for.")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))
    vmax = res["cracked"]["img"].max()          # ONE scale: the panels must be comparable
    for ax, l in zip(axes, ("cracked", "healthy", "difference")):
        r = res[l]
        db = 20 * np.log10(np.maximum(r["img"], vmax * 1e-4) / vmax)
        im = ax.imshow(db.T, origin="lower", aspect="equal", cmap="inferno", vmin=-40, vmax=0,
                       extent=[r["x"][0], r["x"][-1], r["z"][0], r["z"][-1]])
        # NOTHING IS DRAWN ON THE IMAGE. No wall arcs, no notch marker. A marker over the
        # crack tells the viewer where to look, which disqualifies any judgement of whether
        # the defect is detectable, and it covers the very feature the figure exists to show.
        # This figure used to draw both unconditionally, which made it the last annotated
        # image still reaching a published page.
        ax.set_xlim(r["x"][0], r["x"][-1]); ax.set_ylim(0, 40)
        ax.set_title(f"{l}   (notch-ROI peak {r['m']['crack_peak']:.3g})", fontsize=10)
        ax.set_xlabel("x [mm]")
    axes[0].set_ylabel("depth z [mm]")
    fig.colorbar(im, ax=axes.tolist(), label="dB re the CRACKED panel's max", shrink=0.85)
    fig.suptitle(f"Crack minus healthy, {args.angle:+.0f} deg - the wall is common to both runs "
                 f"and cancels coherently", fontsize=11)
    suffix = f"_{args.tag}" if args.tag else ""
    p = OUT / f"baseline_subtract_{int(args.angle)}deg{suffix}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
