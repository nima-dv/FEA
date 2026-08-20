r"""
The mechanism figure: numerical bandwidth, computed from the actual channel data.

This is the figure behind the central numerical lesson of the project - that a mesh sized for
the CENTRE frequency low-passes a short pulse in transit, and axial resolution is set by
BANDWIDTH, not centre frequency. It also makes visible the fact that is easy to gloss over:

    k-Wave's returned pulse is BROADER than ours. It is ahead of us on this axis and still
    sizes the notch worse. So bandwidth explains why OUR refinement helps - it does NOT
    explain why we beat them.

Left panel is computed live from the four channel-data records, so it cannot drift from the
result. Right panel is the measured image metrics from those same runs (see METRICS below,
sourced from repro/compare_images.py output) plotted against refinement, with k-Wave as a
reference line.

RUN
  ./run.ps1 python3 presentation/scripts/bandwidth_convergence.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lib.paths import PRES_DATA, PRESENTATION

FW = PRES_DATA / "ili_forward"
KW = PRES_DATA / "k-wave"
OUT = PRES_DATA / "viz"

# Datasets: label -> (file, colour). All are the SAME scenario at +20 deg, so the only
# difference in the spectra is numerical resolution.
RUNS = [
    # The P3 s1.0 curve comes from the SNAPSHOT solve: same scenario, same mesh, same angle,
    # degree 3 - asking for snapshots does not change the physics, so it is the P3 s1.0 run.
    ("FEM P3, scale 1.0", FW / "channel_data_snap_p20deg.npz", "#c1443c"),
    ("FEM P4, scale 1.0", FW / "channel_data_deg4_p20deg.npz", "#e08214"),
    ("FEM P4, scale 0.8", FW / "channel_data_deg4_s0p8_p20deg.npz", "#1f6fb4"),
    ("k-Wave (50 um grid)", KW / "kwave_odnotch4mm_20.npz", "#111111"),
]

# Image metrics from repro/compare_images.py at +20 deg. P3 s1.0 and P4 s0.8 are MEASURED -
# re-beamformed from channel data on disk. P4 s1.0 is TRANSCRIBED from branch README 4.7: no
# channel data for it survives and re-solving costs ~1.2 h for one interior ladder point. The
# figure labels it, because a mixed-provenance plot that does not say so is a trap.
METRICS = [
    #  label,        notch extent [mm], crack/clutter RMS [dB], measured?
    ("P3\ns1.0", 8.07, 12.2, True),
    ("P4\ns1.0", 3.85, 19.5, False),
    ("P4\ns0.8", 3.73, 24.0, True),
]
KWAVE_EXTENT, KWAVE_CNR = 3.23, 22.8
TRUE_EXTENT = 4.0


def mean_spectrum(ch: np.ndarray, dt: float, half_us: float = 0.6):
    """Element-averaged power spectrum of the RETURNED ECHO.

    Each element is windowed around ITS OWN envelope peak before transforming. That matters
    at 20 deg, where the steering delay law spreads the front-wall arrival over 19.1 us: a
    single common window would chop most elements' echoes in half.

    A WHOLE-RECORD spectrum was tried first and is wrong for this purpose. It reported P3 as
    the WIDEST-band FEM run (6.18 MHz) - backwards, and impossible to reconcile with P3
    smearing the notch to 8.07 mm. The 60 us record is mostly late reverberation, and on the
    coarse mesh that tail is dominated by broadband dispersion noise, which fills the 4-6 MHz
    band with energy that is not usable pulse bandwidth. Measure the pulse, not the noise.
    """
    from scipy.signal import hilbert
    nt, ne = ch.shape
    t = np.arange(nt) * dt
    env = np.abs(hilbert(ch, axis=0))
    search = (t > 20e-6) & (t < 45e-6)                       # the returned-echo region
    idx = np.where(search)[0]
    half = int(half_us * 1e-6 / dt)
    win = np.hanning(2 * half)                               # taper: no edge discontinuity
    segs = []
    for e in range(ne):
        i0 = idx[env[idx, e].argmax()]
        a, b = i0 - half, i0 + half
        if a < 0 or b > nt:
            continue
        segs.append(ch[a:b, e] * win)
    seg = np.asarray(segs).T                                 # (2*half, n_ok)
    f = np.fft.rfftfreq(seg.shape[0], dt) * 1e-6             # MHz
    p = (np.abs(np.fft.rfft(seg, axis=0)) ** 2).mean(axis=1)
    return f, p


def band_energy(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(p[m].sum())


def minus6db_bw(f, p):
    """Width of the band where the spectrum is within 6 dB of its peak."""
    pk = p.max()
    m = p >= pk / 4.0                                        # -6 dB in power = factor 4
    return float(f[m].max() - f[m].min())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmax", type=float, default=10.0, help="plot limit [MHz]")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 5.6))

    print(f"{'run':<22}{'-6dB BW':>10}{'(4-6)/(2-4)':>14}")
    rows = []
    for label, path, col in RUNS:
        if not path.exists():
            print(f"  SKIP {label}: {path} not found")
            continue
        d = np.load(path, allow_pickle=True)
        ch, dt = d["channel_data"], float(d["dt"])
        f, p = mean_spectrum(ch, dt)
        bw = minus6db_bw(f, p)
        ratio = band_energy(f, p, 4, 6) / max(band_energy(f, p, 2, 4), 1e-30)
        rows.append((label, bw, ratio))
        print(f"{label:<22}{bw:>8.2f} MHz{ratio:>14.3f}")
        keep = f <= args.fmax
        axL.plot(f[keep], 10 * np.log10(p[keep] / p.max()), color=col, lw=1.7, label=label)

    axL.axvspan(4, 6, color="0.85", zorder=0)
    axL.text(5.0, -3, "4-6 MHz\n(the band that\ngets lost)", ha="center", va="top",
             fontsize=8, color="0.35")
    axL.axvline(4.0, color="0.5", ls=":", lw=1)
    axL.text(4.05, -46, "  f0 = 4 MHz", fontsize=8, color="0.4", rotation=90, va="bottom")
    axL.set_xlim(0, args.fmax); axL.set_ylim(-50, 2)
    axL.set_xlabel("frequency [MHz]"); axL.set_ylabel("power, dB re each record's own peak")
    axL.set_title("Returned-pulse spectrum, +20 deg - per-element echo window\n"
                  "element-averaged, computed from the channel data (not transcribed)",
                  fontsize=10)
    axL.legend(fontsize=8.5, loc="upper right")
    axL.grid(alpha=0.25)

    # --- right: does closing that gap actually move the image metrics? -------------------
    x = np.arange(len(METRICS))
    ext = [m[1] for m in METRICS]
    cnr = [m[2] for m in METRICS]
    axR.plot(x, ext, "o-", color="#1f6fb4", lw=2, ms=8, label="FEM notch extent")
    axR.axhline(TRUE_EXTENT, color="0.2", ls="-", lw=1.4)
    axR.text(0.04, TRUE_EXTENT + 0.15, "TRUE 4.0 mm", fontsize=8.5, color="0.2")
    axR.axhline(KWAVE_EXTENT, color="#c1443c", ls="--", lw=1.6)
    axR.text(0.04, KWAVE_EXTENT - 0.45, "k-Wave 3.23 mm (-19%)", fontsize=8.5, color="#c1443c")
    axR.set_xticks(x)
    axR.set_xticklabels([m[0] if m[3] else m[0] + "\n(from README)" for m in METRICS],
                        fontsize=9)
    axR.set_ylabel("imaged notch extent [mm]", color="#1f6fb4")
    axR.set_ylim(0, 9)
    axR.set_xlabel("refinement  ->")

    ax2 = axR.twinx()
    ax2.plot(x, cnr, "s--", color="#2e8b57", lw=2, ms=7, label="FEM crack/clutter")
    ax2.axhline(KWAVE_CNR, color="#2e8b57", ls=":", lw=1.4, alpha=0.7)
    ax2.text(1.35, KWAVE_CNR + 0.4, "k-Wave 22.8 dB", fontsize=8.5, color="#2e8b57")
    ax2.set_ylabel("crack / clutter RMS [dB]", color="#2e8b57")
    ax2.set_ylim(8, 28)
    axR.set_title("Closing the bandwidth gap moves BOTH image metrics\n"
                  "and was predicted to, before the second angle was solved", fontsize=10)
    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="center right")
    axR.grid(alpha=0.25)

    fig.suptitle("Numerical bandwidth: what limited US - and note k-Wave is AHEAD of us on it",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / "bandwidth_convergence.png"
    fig.savefig(p, dpi=170)
    print(f"\nwrote {p}")
    if rows:
        best = max(rows, key=lambda r: r[1])
        print(f"widest returned bandwidth: {best[0]} at {best[1]:.2f} MHz")
        print("If that is k-Wave, the figure is telling the truth: they are ahead on bandwidth\n"
              "and still size the notch worse, so bandwidth is not why we beat them.")


if __name__ == "__main__":
    main()
