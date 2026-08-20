r"""
Compare our FEM channel data with k-Wave's DIRECTLY, before any beamforming.

Why: the beamformed comparison at 20 deg showed our crack response buried in clutter
(crack/worst-clutter -0.1 dB) while k-Wave's stands 10.3 dB clear. That could be the
forward solve, the source model, the time-zero convention, or the recorded quantity - and
the beamformer hides which. Both datasets are (22801, 256) on the SAME 380 MHz time base
and the same geometry, so they can be diffed sample by sample.

What to look for, in order of how badly it would break things:
  1. Does the ID (front-wall) echo arrive at the same TIME in both? A time offset means our
     source time-zero or delay law is wrong, which biases everything downstream.
  2. Does the ID echo show the same DELAY-vs-ELEMENT slope? At 20 deg the steered wavefront
     hits the wall obliquely; a wrong slope (or a mirrored one) means the beam is steering
     to the wrong angle, so the beamformer looks along the wrong path.
  3. Do the later, weaker arrivals (wall reverberation, notch scattering) have comparable
     amplitude RELATIVE to the ID echo? Absolute scale is not comparable - they drive a
     2e-6 velocity source, we apply a unit traction - but the ratio of late to early energy
     is, and it is what determines clutter.

RUN
  ./run.ps1 python3 presentation/scripts/compare_rf.py --ours presentation/data/ili_forward/channel_data_p20deg.npz \
      --theirs <scratch>/k-wave/kwave_odnotch4mm_20.npz --angle 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from lib.paths import PRES_DATA, PRESENTATION

OUT = PRES_DATA / "compare"


def onset(env: np.ndarray, t: np.ndarray, frac: float = 0.2) -> float:
    """First time the envelope exceeds `frac` of its own max, or NaN if it never does.

    The `or NaN` matters. The first version returned t[0] whenever the threshold was never
    crossed (np.argmax on an all-False array returns 0), so every element whose echo lay
    outside the search window silently reported the window's lower bound. Over half the
    aperture did exactly that, and the resulting "median difference 0.000 us" was an
    average of non-measurements that looked like perfect agreement.
    """
    m = env.max()
    if m <= 0:
        return np.nan
    hit = env > frac * m
    return t[np.argmax(hit)] if hit.any() else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--theirs", required=True)
    ap.add_argument("--angle", type=float, required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    A = np.load(args.ours, allow_pickle=True)
    B = np.load(args.theirs, allow_pickle=True)
    ca, cb = A["channel_data"], B["channel_data"]
    dta, dtb = float(A["dt"]), float(B["dt"])
    if ca.shape != cb.shape or abs(dta - dtb) > 1e-15:
        raise SystemExit(f"not comparable: {ca.shape}@{dta:.4e} vs {cb.shape}@{dtb:.4e}")
    nt, ne = ca.shape
    t_us = np.arange(nt) * dta * 1e6
    print(f"both {ca.shape}, dt {dta:.6e} s   steering {args.angle:+.0f} deg")
    print(f"peak |p|:  FEM {np.abs(ca).max():.4g}   k-Wave {np.abs(cb).max():.4g}   "
          f"(scales differ by source convention; ratios are what matter)")

    Ea, Eb = np.abs(hilbert(ca, axis=0)), np.abs(hilbert(cb, axis=0))

    # --- 1 & 2: ID echo onset per element, and its slope across the aperture ------------
    # Search window must span the whole steered arc. At 20 deg the front-wall echo sweeps
    # from ~21 us at element 0 to ~41 us at element 255, so a 20-30 us window (the first
    # attempt) misses it entirely beyond about element 105 - which is how the broken onset
    # picker above produced a fake perfect agreement.
    w = (t_us > 18) & (t_us < 48)
    ta = np.array([onset(Ea[w, e], t_us[w]) for e in range(ne)])
    tb = np.array([onset(Eb[w, e], t_us[w]) for e in range(ne)])
    ok = np.isfinite(ta) & np.isfinite(tb)
    el = np.arange(ne)
    pa = np.polyfit(el[ok], ta[ok], 1)
    pb = np.polyfit(el[ok], tb[ok], 1)
    print(f"\nID-echo onset across the aperture (linear fit):")
    print(f"  FEM     intercept {pa[1]:8.3f} us   slope {pa[0]*1e3:+8.3f} ns/element")
    print(f"  k-Wave  intercept {pb[1]:8.3f} us   slope {pb[0]*1e3:+8.3f} ns/element")
    print(f"  DIFF    intercept {pa[1]-pb[1]:+8.3f} us   slope "
          f"{(pa[0]-pb[0])*1e3:+8.3f} ns/element")
    print(f"  median per-element onset difference {np.median(ta[ok]-tb[ok]):+.3f} us")
    if abs(pa[0] - pb[0]) * 1e3 > 5:
        print("  !! SLOPE MISMATCH > 5 ns/element: the beam is steering to a different "
              "angle than k-Wave's. Check the delay law sign/convention.")
    if abs(np.median(ta[ok] - tb[ok])) > 0.2:
        print("  !! ONSET OFFSET > 0.2 us: time-zero convention differs.")

    # --- 3: late/early energy ratio (this is what becomes clutter) ----------------------
    def band(E, lo, hi):
        m = (t_us >= lo) & (t_us < hi)
        return float(np.sqrt((E[m] ** 2).mean()))
    print(f"\nRMS envelope by time window, normalised to each dataset's ID-echo window:")
    print(f"{'window [us]':<16}{'FEM':>12}{'k-Wave':>12}{'FEM/kWave':>12}")
    ref_a, ref_b = band(Ea, 24, 32), band(Eb, 24, 32)
    for lo, hi, label in ((0, 20, "pre-arrival"), (24, 32, "ID+wall"),
                          (32, 40, "late wall"), (40, 60, "reverberation")):
        ra, rb = band(Ea, lo, hi) / ref_a, band(Eb, lo, hi) / ref_b
        print(f"{f'{lo}-{hi}':<16}{ra:12.4g}{rb:12.4g}{ra/rb:12.3f}")
    print("A FEM/kWave ratio >> 1 in the late windows means our solve puts relatively more\n"
          "energy into late arrivals, which is exactly what shows up as image clutter.")

    # --- figures ------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for j, (E, lab) in enumerate(((Ea, "FEM"), (Eb, "k-Wave"))):
        v = np.percentile(E, 99.8)
        ax[0, j].imshow(E, aspect="auto", origin="upper", cmap="inferno", vmin=0, vmax=v,
                        extent=[0, ne, t_us[-1], 0])
        ax[0, j].set_ylim(45, 20); ax[0, j].set_title(f"{lab} B-scan (envelope)")
        ax[0, j].set_xlabel("element"); ax[0, j].set_ylabel("time [us]")
    ax[1, 0].plot(el[ok], ta[ok], ".", ms=3, label="FEM")
    ax[1, 0].plot(el[ok], tb[ok], ".", ms=3, label="k-Wave")
    ax[1, 0].set_xlabel("element"); ax[1, 0].set_ylabel("ID onset [us]")
    ax[1, 0].set_title("front-wall onset vs element (steering signature)")
    ax[1, 0].legend(fontsize=8)
    c = ne // 2
    ax[1, 1].semilogy(t_us, Ea[:, c] / Ea[:, c].max(), lw=0.8, label="FEM")
    ax[1, 1].semilogy(t_us, Eb[:, c] / Eb[:, c].max(), lw=0.8, label="k-Wave")
    ax[1, 1].set_xlim(20, 60); ax[1, 1].set_ylim(1e-4, 2)
    ax[1, 1].set_xlabel("time [us]"); ax[1, 1].set_ylabel("envelope / own max")
    ax[1, 1].set_title(f"centre element {c}, self-normalised")
    ax[1, 1].legend(fontsize=8)
    fig.suptitle(f"Raw channel data, steering {args.angle:+.0f} deg - no beamformer",
                 fontsize=11)
    fig.tight_layout()
    p = OUT / f"rf_compare_{int(args.angle)}deg.png"
    fig.savefig(p, dpi=135)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
