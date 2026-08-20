r"""
Verify the forward solve against analytic time-of-flight - the FIRST thing to check.

The forward solve is only trustworthy if its echoes arrive when geometry says they must.
Everything downstream (imaging, the k-Wave comparison, the whole "timing >> amplitude"
claim) rests on this. Amplitudes come later; arrival times come first.

ANALYTIC ToF ON THE BEAM AXIS, normal incidence (0 deg steering)
  The array sits 20 mm of water from the ID. The wall is 9.525 mm of steel. The notch is
  4.0 mm deep, cut inward from the OD, so its TIP is 9.525 - 4.0 = 5.525 mm into the wall.

    front wall (ID)      2 * 20.000 / 1500          = 26.667 us
    notch tip (via P)    26.667 + 2 * 5.525 / 5700  = 28.606 us
    back wall (OD, P)    26.667 + 2 * 9.525 / 5700  = 30.009 us
    back wall (OD, S)    26.667 + 2 * 9.525 / 3100  = 32.812 us

  The notch tip echo arrives ~1.40 us BEFORE the back wall - that gap is its depth
  signature, and reproducing it is the point of the whole simulation.

  Caveat that must be stated: the pipe is CURVED, so only the on-axis ray is exactly
  normal. Off-axis elements see a slightly longer path, which broadens each arrival. We
  therefore pick on the CENTRE elements only, and quote the analytic value as the target
  for those.

RUN
  ./run.ps1 python3 repro/analyze_forward.py --case results/ili_forward/channel_data_p0deg.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from lib.paths import RESULTS

C_P, C_S, C_F = 5700.0, 3100.0, 1500.0
STANDOFF, WALL, NOTCH_H = 0.020, 0.009525, 0.004

TOF = {
    "front wall (ID)":   2 * STANDOFF / C_F,
    "notch tip (P)":     2 * STANDOFF / C_F + 2 * (WALL - NOTCH_H) / C_P,
    "back wall (OD, P)": 2 * STANDOFF / C_F + 2 * WALL / C_P,
    "back wall (OD, S)": 2 * STANDOFF / C_F + 2 * WALL / C_S,
}

OUT = RESULTS / "ili_forward"

# A peak must stand this far above its surrounding troughs, as a fraction of the largest
# envelope value, to count as an echo rather than a bump on a decaying tail.
PROMINENCE_FRAC = 0.02


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--n-centre", type=int, default=16,
                    help="how many central elements to average for the A-scan")
    args = ap.parse_args()

    d = np.load(args.case, allow_pickle=True)
    ch = d["channel_data"]
    dt = float(d["dt"])
    angle = float(d["angle"])
    nt, ne = ch.shape
    t = np.arange(nt) * dt
    t_us = t * 1e6

    print(f"case {Path(args.case).name}: {ch.shape}, dt {dt:.6e} s, steering {angle:+.1f} deg")
    print(f"       peak |p| {np.abs(ch).max():.4g}, "
          f"record length {t_us[-1]:.2f} us")
    if not np.isfinite(ch).all():
        raise SystemExit("channel data contains non-finite values - the solve diverged")

    # Centre elements: their rays are closest to normal incidence on the curved wall.
    c0 = ne // 2 - args.n_centre // 2
    a = ch[:, c0:c0 + args.n_centre].mean(axis=1)
    env = np.abs(hilbert(a))

    # --- pick arrivals in windows around each analytic prediction --------------------
    # A window maximum is NOT an echo. On a decaying ringdown tail the argmax of a window
    # is just the leftmost sample, which yields a confident-looking but meaningless
    # "arrival". We therefore require a genuine local maximum with PROMINENCE - the peak
    # must stand above the surrounding trough by a set fraction of its own height. Echoes
    # that fail this are reported as NOT RESOLVED, which is a real and useful result: at
    # 0 deg the notch scatters weakly and mode conversion is zero, so neither the tip
    # echo nor a shear back-wall echo should be visible.
    from scipy.signal import find_peaks
    pk, props = find_peaks(env, prominence=PROMINENCE_FRAC * env.max())
    print(f"\nprominent peaks found: {pk.size} "
          f"(prominence > {PROMINENCE_FRAC:.0%} of max envelope)")

    print(f"\n{'echo':<20} {'analytic':>10} {'measured':>10} {'error':>9}   {'rel.amp':>8}")
    print("-" * 66)
    rows = []
    for name, tof in TOF.items():
        # +-1.0 us window: wide enough to catch a mistimed echo, narrow enough not to grab
        # a neighbour (the closest analytic pair is 1.40 us apart).
        cand = pk[(t[pk] > tof - 1.0e-6) & (t[pk] < tof + 1.0e-6)]
        if t[-1] < tof - 1.0e-6:
            print(f"{name:<20} {tof*1e6:10.3f}   (outside record)")
            continue
        if cand.size == 0:
            print(f"{name:<20} {tof*1e6:10.3f}   NOT RESOLVED "
                  f"(no prominent peak in +-1 us)")
            continue
        i = int(cand[env[cand].argmax()])
        meas = t[i]
        err = (meas - tof) / tof * 100
        rows.append((name, tof, meas, err, env[i]))
        print(f"{name:<20} {tof*1e6:10.3f} {meas*1e6:10.3f} {err:+8.2f}%   "
              f"{env[i]/env.max():8.3f}")

    # --- plot ------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(t_us, a, lw=0.6, color="0.35")
    ax[0].plot(t_us, env, lw=1.2, color="C3", label="envelope")
    ax[0].set_ylabel("pressure [arb]")
    ax[0].set_title(f"A-scan, mean of {args.n_centre} central elements "
                    f"(steering {angle:+.0f} deg)")
    ax[1].semilogy(t_us, np.maximum(env, env.max() * 1e-6), lw=1.0, color="C0")
    ax[1].set_ylabel("envelope [log]")
    ax[1].set_xlabel("time [us]")
    for axi in ax:
        for k, (name, tof, meas, err, amp) in enumerate(zip(
                [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
                [r[3] for r in rows], [r[4] for r in rows])):
            axi.axvline(tof * 1e6, color=f"C{k+1}", ls="--", lw=1.0,
                        label=name if axi is ax[1] else None)
            axi.axvline(meas * 1e6, color=f"C{k+1}", ls=":", lw=1.4)
    ax[1].legend(fontsize=7, loc="lower right")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_xlim(20, min(45, t_us[-1]))
    fig.suptitle("dashed = analytic ToF, dotted = measured peak", fontsize=9, y=0.995)
    fig.tight_layout()
    p = OUT / f"ascan_{Path(args.case).stem.replace('channel_data_','')}.png"
    fig.savefig(p, dpi=130)
    print(f"\nwrote {p}")

    # --- B-scan across the aperture ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    E = np.abs(hilbert(ch, axis=0))
    vmax = np.percentile(E, 99.9)
    ax.imshow(E, aspect="auto", origin="upper", cmap="inferno", vmin=0, vmax=vmax,
              extent=[0, ne, t_us[-1], 0])
    for k, (name, tof) in enumerate(TOF.items()):
        ax.axhline(tof * 1e6, color="cyan", ls="--", lw=0.8)
        ax.text(2, tof * 1e6 - 0.3, name, color="cyan", fontsize=7)
    ax.set_ylim(min(45, t_us[-1]), 20)
    ax.set_xlabel("element"); ax.set_ylabel("time [us]")
    ax.set_title(f"B-scan (envelope), steering {angle:+.0f} deg")
    fig.tight_layout()
    p = OUT / f"bscan_{Path(args.case).stem.replace('channel_data_','')}.png"
    fig.savefig(p, dpi=130)
    print(f"wrote {p}")

    worst = max((abs(r[3]) for r in rows), default=float("nan"))
    print(f"\n{len(rows)}/{len(TOF)} echoes resolved; worst timing error of those "
          f"{worst:.2f}%  ->  {'PASS' if worst < 2.0 else 'INVESTIGATE'}")
    print("NOTE: 'PASS' refers only to the RESOLVED echoes. An unresolved echo is not a\n"
          "      failure per se - at 0 deg the notch tip and the shear back-wall echo are\n"
          "      both expected to be absent (weak normal-incidence scattering, and zero\n"
          "      mode conversion at normal incidence). Notch detection is a 20 deg test.")


if __name__ == "__main__":
    main()
