r"""
Animate the ILI wavefield: the 20 deg beam mode-converting at the ID and skipping to the notch.

This is D1, the demonstration animation. It renders the snapshots written by
`repro/ili_forward.py --snapshots N` (sample values plus their coordinates, so nothing here
needs FEniCS or a mesh reader).

SAMPLING IS NOT A FREE CHOICE
  The first version of this animation sampled one value per cell and the wavefronts came out
  BEADED - visible aliasing, not physics. A 0.25 mm mesh against a 0.375 mm water wavelength
  gives only 1.5 samples per wavelength that way, below Nyquist. `--snap-degree 2` (the
  default in the solver) gives 9 samples per quad, ~4.5 per wavelength, and the crests render
  as continuous bands. If a rendered wavefront ever looks dashed, suspect the sampling before
  suspecting the solve.

WHAT IS PLOTTED, AND WHY IT IS TWO DIFFERENT FIELDS
  By Helmholtz decomposition a P wave is curl-free and an S wave is divergence-free, so
  div(u) and curl(u) separate the two wave types cleanly. The water carries no shear at all
  (mu = 0 there, by construction), so:
      water -> div(u)   the incident and reflected PRESSURE beam
      steel -> curl(u)  the mode-converted SHEAR wave, which is what actually images the notch
  Showing both in one frame is the only way to see mode conversion happen: the beam arrives
  as P in the water and leaves as S at ~45 deg in the steel. Each region is normalised by its
  OWN robust maximum - the two fields have different units and wildly different magnitudes,
  and a shared scale renders one of them invisible.

HONEST LABELLING
  The animation is illustrative, not metric-bearing, and is deliberately produced at degree 3
  / scale 1.0 (~20 min) rather than the degree-4 / scale-0.8 configuration used for every
  published number (~2.4 h). The physics on show is identical; the quantitative claims come
  from `repro/compare_images.py`. The frame stamp says which configuration it is.

RUN
  ./run.ps1 python3 viz/wavefield_gif.py --in results/ili_forward/wavefield_snap_p20deg.npz
  ./run.ps1 python3 viz/wavefield_gif.py --in ... --stills 22,28,33,38   # key frames as PNG
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
# (GIF assembly is done through PIL below - see the note at the save call)

# Geometry, in mm, matching mesh/ili_mesh.py. Used only to mask the parts of the
# triangulation that are not physical domain and to draw the outlines.
X_C, Z_C = 38.25, -173.675
R_ID, R_OD = 193.675, 203.200
NOTCH_X, NOTCH_W, NOTCH_DEPTH = 38.25, 1.0, 4.0
Z_TIP = Z_C + R_OD - NOTCH_DEPTH                      # 25.525 mm

OUT = Path(__file__).resolve().parents[1] / "results" / "viz"


def build_triangulation(x_mm, z_mm, steel: np.ndarray, has_notch: bool):
    """Delaunay over the cell centroids, with the non-physical triangles masked out.

    Three things get masked:
      1. Triangles beyond the curved OD. The centroids are a point cloud, so a raw Delaunay
         triangulation fills in anything concave and bulges past the boundary.
      2. Triangles bridging the notch VOID, for the same reason.
      3. Triangles STRADDLING the water/steel interface. This one is not cosmetic: we plot
         div(u) in the water and curl(u) in the steel, so a gouraud-shaded triangle with
         vertices on both sides would blend two different physical quantities and paint a
         halo along the ID that is pure rendering artefact. Dropping those triangles leaves a
         one-cell gap exactly where the ID arc is drawn in black, so nothing is lost visually.

    Masked analytically from the known geometry rather than by an edge-length heuristic, which
    would also eat legitimate triangles in the coarse fluid region.
    """
    tri = mtri.Triangulation(x_mm, z_mm)
    tx = x_mm[tri.triangles].mean(axis=1)
    tz = z_mm[tri.triangles].mean(axis=1)
    bad = np.hypot(tx - X_C, tz - Z_C) > R_OD          # outside the pipe OD
    if has_notch:
        bad |= (np.abs(tx - NOTCH_X) <= NOTCH_W / 2) & (tz >= Z_TIP)
    ns = steel[tri.triangles].sum(axis=1)
    bad |= (ns > 0) & (ns < 3)                         # straddles the ID
    tri.set_mask(bad)
    return tri, int(bad.sum())


def smoothing_edges(tri, x_mm, z_mm, steel: np.ndarray):
    """Edge list for neighbour averaging, with the edges that must NOT be averaged removed.

    Two exclusions, both load-bearing:
      * edges crossing the water/steel interface - the two sides hold different physical
        quantities (div u vs curl u), so averaging across the ID is meaningless;
      * unusually long edges, which are the ones bridging the notch void.
    """
    e = tri.edges
    same = steel[e[:, 0]] == steel[e[:, 1]]
    ln = np.hypot(x_mm[e[:, 0]] - x_mm[e[:, 1]], z_mm[e[:, 0]] - z_mm[e[:, 1]])
    return e[same & (ln < 3.0 * np.median(ln))]


def smooth_field(v: np.ndarray, e: np.ndarray, n: int, iters: int) -> np.ndarray:
    """`iters` passes of self-plus-neighbour averaging over the sample cloud.

    Cosmetic, and declared as such in the figure caption whenever it is used. It exists
    because the samples are DISCONTINUOUS across cells, so a triangulation over the sample
    cloud shows a faint checkerboard where neighbouring cells disagree - grain at sample
    spacing (~0.08 mm), far finer than the 0.375 mm wave it sits on. Averaging suppresses that
    without touching the wavefronts. It is NOT a fix for under-sampling: if the wavefronts
    themselves look beaded, raise --snap-degree instead, because no amount of smoothing
    recovers information that was never sampled.
    """
    cnt = np.bincount(e[:, 0], minlength=n) + np.bincount(e[:, 1], minlength=n)
    for _ in range(iters):
        s = (np.bincount(e[:, 0], weights=v[e[:, 1]], minlength=n)
             + np.bincount(e[:, 1], weights=v[e[:, 0]], minlength=n))
        v = np.where(cnt > 0, (v + s) / (1 + cnt), v)
    return v


def draw_geometry(ax, has_notch: bool) -> None:
    """ID arc, OD arc and the notch outline, so the wave is seen against the structure."""
    th = np.linspace(np.deg2rad(-30), np.deg2rad(30), 400)
    for R, style in ((R_ID, dict(lw=1.1, color="k")), (R_OD, dict(lw=1.1, color="k"))):
        ax.plot(X_C + R * np.sin(th), Z_C + R * np.cos(th), **style, zorder=5)
    if has_notch:
        xl, xr = NOTCH_X - NOTCH_W / 2, NOTCH_X + NOTCH_W / 2
        zl = Z_C + np.sqrt(max(R_OD**2 - (xl - X_C) ** 2, 0.0))
        zr = Z_C + np.sqrt(max(R_OD**2 - (xr - X_C) ** 2, 0.0))
        ax.plot([xl, xl, xr, xr], [zl, Z_TIP, Z_TIP, zr], color="k", lw=1.3, zorder=6)
    ax.axhline(0.0, color="0.35", lw=1.6, zorder=5)     # the 256-element array plane


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="wavefield_*.npz from ili_forward")
    ap.add_argument("--out", default=None, help="output .gif (default: derived from --in)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--clip", type=float, default=99.5,
                    help="percentile for the colour limit; below 100 so a single hot cell "
                         "cannot flatten the whole field (default 99.5)")
    ap.add_argument("--smooth", type=int, default=0,
                    help="cosmetic neighbour-averaging passes over the sample cloud "
                         "(default 0 = off). Suppresses the faint checkerboard from "
                         "cell-to-cell sample discontinuity. Declared in the caption when "
                         "used. Not a substitute for --snap-degree.")
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth frame (default 1). Halves file size per doubling.")
    ap.add_argument("--colors", type=int, default=96,
                    help="GIF palette size (default 96). The field is smooth, so a small "
                         "palette costs nothing visually and dominates the file size.")
    ap.add_argument("--stills", default=None, metavar="T1,T2,...",
                    help="also write PNG stills at these times [us]")
    ap.add_argument("--vlim", default=None, metavar="WATER,STEEL",
                    help="force the two colour limits instead of deriving them from this "
                         "file's own p99.5. REQUIRED for a side-by-side pair: with per-file "
                         "limits the same wave renders at a different brightness in each "
                         "panel, and a viewer would read the normalisation as physics. Run "
                         "one animation first, then pass the limits it printed to the other.")
    ap.add_argument("--xlim", default=None, metavar="X0,X1",
                    help="crop to this x window [mm]. Needed to animate two domains of "
                         "DIFFERENT width over the same window: at a fixed figure size a "
                         "wider domain would otherwise render at a different mm-per-pixel, "
                         "and its colour scale would be a percentile over a different region. "
                         "The crop is applied to the sample cloud before both, so a cropped "
                         "wide run and an uncropped narrow run are pixel-comparable.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    d = np.load(args.inp)
    x, z = d["x"] * 1e3, d["z"] * 1e3
    div, curl, t_us = d["div"], d["curl"], d["t"] * 1e6
    steel = d["steel"].astype(bool)
    if args.xlim:
        x0, x1 = (float(v) for v in args.xlim.split(","))
        keep = (x >= x0) & (x <= x1)
        if not keep.any():
            raise SystemExit(f"--xlim {x0},{x1} keeps no samples; data spans "
                             f"{x.min():.1f} to {x.max():.1f} mm")
        print(f"xlim crop: {keep.sum()} of {x.size} samples kept, "
              f"x {x0:.1f} to {x1:.1f} mm (data spans {x.min():.1f} to {x.max():.1f})")
        x, z, steel = x[keep], z[keep], steel[keep]
        div, curl = div[:, keep], curl[:, keep]
    deg = int(d["degree"]) if "degree" in d.files else 0
    ang = float(d["angle"]) if "angle" in d.files else 0.0
    nf = div.shape[0]
    # Cracked or healthy mesh? On the cracked mesh the notch is a real VOID, so no cell
    # centroid lies inside it. Margins keep boundary-hugging cells out of the test.
    in_notch = (np.abs(x - NOTCH_X) <= NOTCH_W / 2 - 0.05) & (z >= Z_TIP + 0.05)
    has_notch = not bool(in_notch.any())
    print(f"{nf} frames, {x.size} cells, t {t_us[0]:.2f}-{t_us[-1]:.2f} us, "
          f"steel {steel.sum()} cells, notch {'present' if has_notch else 'absent'}")

    # One robust scale per region, held FIXED across the animation so brightness changes
    # mean physics rather than autoscaling.
    s_water = np.percentile(np.abs(div[:, ~steel]), args.clip)
    s_steel = np.percentile(np.abs(curl[:, steel]), args.clip)
    print(f"colour limits: water |div u| {s_water:.4g}, steel |curl u| {s_steel:.4g} "
          f"(each = p{args.clip} of its own region)")
    if args.vlim:
        v_w, v_s = (float(v) for v in args.vlim.split(","))
        print(f"  OVERRIDDEN by --vlim: water {v_w:.4g} ({v_w/s_water:.3f}x this file's own), "
              f"steel {v_s:.4g} ({v_s/s_steel:.3f}x) - shared scale for a side-by-side pair")
        s_water, s_steel = v_w, v_s

    tri, nmask = build_triangulation(x, z, steel, has_notch)
    print(f"triangulation: {tri.triangles.shape[0]} triangles, {nmask} masked "
          f"(outside the OD + notch void + straddling the ID)")
    sm_e = smoothing_edges(tri, x, z, steel) if args.smooth else None
    if args.smooth:
        print(f"smoothing: {args.smooth} pass(es) over {sm_e.shape[0]} intra-region edges "
              f"(cosmetic; stated in the caption)")

    def frame_field(i: int) -> np.ndarray:
        f = np.where(steel, curl[i] / max(s_steel, 1e-30), div[i] / max(s_water, 1e-30))
        if args.smooth:
            f = smooth_field(f, sm_e, x.size, args.smooth)
        return np.clip(f, -1.0, 1.0)

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    coll = ax.tripcolor(tri, frame_field(0), shading="gouraud", cmap="RdBu_r",
                        vmin=-1, vmax=1, rasterized=True)
    draw_geometry(ax, has_notch)
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(0, z.max())
    ax.set_aspect("equal"); ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
    cb = fig.colorbar(coll, ax=ax, pad=0.01, shrink=0.9)
    cb.set_label("normalised   water: div u (P)   |   steel: curl u (S)")
    ttl = ax.set_title("")
    sub = (f"FEniCS/DOLFINx, degree {deg}, steering {ang:+.0f} deg  -  illustrative; "
           f"published numbers use degree 4 / scale 0.8"
           + (f"  |  {args.smooth} cosmetic smoothing pass(es)" if args.smooth else ""))
    fig.text(0.012, 0.015, sub, fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    def update(i: int):
        coll.set_array(frame_field(i))
        ttl.set_text(f"ILI wavefield   t = {t_us[i]:6.2f} us      "
                     f"P beam in water (blue/red) mode-converts to ~45 deg SHEAR in the steel")
        return coll, ttl

    out = Path(args.out) if args.out else OUT / f"{Path(args.inp).stem}.gif"
    # Assemble the GIF through PIL with an adaptive palette rather than via PillowWriter.
    # A noisy truecolour wavefield converted frame-by-frame to GIF does not compress: the
    # first attempt at this animation came out 67 MB, too big to publish. Quantising to a
    # shared adaptive palette and letting PIL optimise the frame deltas gets the same 240
    # frames to a few MB with no visible change at this colour depth.
    from PIL import Image
    frames = []
    for i in range(0, nf, max(1, args.stride)):
        update(i)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(buf).convert(
            "P", palette=Image.ADAPTIVE, colors=args.colors))
    frames[0].save(out, save_all=True, append_images=frames[1:], optimize=True,
                   duration=int(1000 / args.fps), loop=0, disposal=2)
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB, {len(frames)} frames "
          f"@ {args.fps} fps, {args.colors} colours)")

    if args.stills:
        for ts in (float(v) for v in args.stills.split(",")):
            i = int(np.abs(t_us - ts).argmin())
            update(i)
            p = OUT / f"{out.stem}_t{t_us[i]:.1f}us.png".replace(".", "p", 1)
            fig.savefig(p, dpi=160)
            print(f"  still t={t_us[i]:.2f} us -> {p}")


if __name__ == "__main__":
    main()
