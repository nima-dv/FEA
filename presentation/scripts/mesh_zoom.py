r"""
D4: what "conforming" actually means, at the scale where it matters.

Three zooms, drawn from the real meshes on disk (not sketches):

  1. the ID interface on the conforming mesh - element edges lie ON the exact circle
  2. the SAME interface on the C4 staircase mesh - 50 um pixel steps, k-Wave's grid spacing
  3. the notch tip on the cracked mesh - a real void with two exact sharp corners, which is a
     traction-free surface and therefore the exactly correct steel/air condition

Panels 1 and 2 are the visual statement of the headline conformity number (chord sagitta
0.05 um against ~140 um for a voxel grid at the same cell size). Panel 3 is why the notch does
not need refinement to be represented exactly: conformity comes from the boundary, not from
cell size.

RUN
  ./run.ps1 python3 presentation/scripts/mesh_zoom.py
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from lib.paths import PRES_DATA, PRESENTATION

MESH = PRES_DATA / "ili_mesh"
OUT = PRES_DATA / "viz"

R_ID, R_OD = 193.675, 203.200
X_C, Z_C = 38.25, -173.675
NOTCH_X, NOTCH_W, NOTCH_DEPTH = 38.25, 1.0, 4.0
Z_TIP = Z_C + R_OD - NOTCH_DEPTH


TAG_ID = 11                                              # facet tag from mesh/ili_mesh.py


def read_edges(path: Path):
    """Cell edges and the ID-interface facets of a 2-D gmsh mesh, in mm.

    The ID facets are returned separately so the figure can draw the interface the MESH
    actually has, in contrast to the exact circle. Without that overlay the staircase is
    only inferable from the cell pattern, which is exactly the thing the figure is meant to
    show rather than imply. meshio is already a dependency.
    """
    import meshio
    m = meshio.read(str(path))
    pts = m.points[:, :2] * 1e3
    segs, id_segs = set(), []
    phys = m.cell_data.get("gmsh:physical", [None] * len(m.cells))
    for blk, tags in zip(m.cells, phys):
        if blk.type == "triangle":
            loops = [(0, 1), (1, 2), (2, 0)]
        elif blk.type == "quad":
            loops = [(0, 1), (1, 2), (2, 3), (3, 0)]
        elif blk.type == "line":
            if tags is not None:
                id_segs.append(blk.data[np.asarray(tags) == TAG_ID])
            continue
        else:
            continue
        for cell in blk.data:
            for a, b in loops:
                segs.add((min(cell[a], cell[b]), max(cell[a], cell[b])))
    idl = np.concatenate(id_segs) if id_segs else np.empty((0, 2), dtype=int)
    return pts, np.array(sorted(segs)), idl


def panel(ax, path: Path, xc: float, zc: float, half: float, title: str,
          arcs=(), notch=False):
    pts, e, idl = read_edges(path)
    # Keep only edges with an endpoint in view; drawing 75k cells would be slow and invisible.
    inview = (np.abs(pts[:, 0] - xc) <= half * 1.6) & (np.abs(pts[:, 1] - zc) <= half * 1.6)
    keep = inview[e[:, 0]] | inview[e[:, 1]]
    ax.add_collection(LineCollection(pts[e[keep]], colors="0.62", linewidths=0.4))
    if idl.size:
        k = inview[idl[:, 0]] | inview[idl[:, 1]]
        ax.add_collection(LineCollection(pts[idl[k]], colors="#1f4fd8", linewidths=2.0,
                                         zorder=3, label="mesh ID facets"))
    for R, col in arcs:
        # The EXACT circle, plotted independently of the mesh. In panel 1 it should be
        # invisible under the element edges; in panel 2 it visibly cuts across the steps.
        th = np.linspace(-0.3, 0.3, 20000)
        ax.plot(X_C + R * np.sin(th), Z_C + R * np.cos(th), col, lw=1.2, alpha=0.95,
                label="exact circle", zorder=4, ls="--")
    if notch:
        xl, xr = NOTCH_X - NOTCH_W / 2, NOTCH_X + NOTCH_W / 2
        ax.plot([xl, xl, xr, xr],
                [Z_C + math.sqrt(R_OD**2 - (xl - X_C)**2), Z_TIP, Z_TIP,
                 Z_C + math.sqrt(R_OD**2 - (xr - X_C)**2)],
                "r", lw=1.4, label="notch (CAD)", zorder=4)
    ax.set_xlim(xc - half, xc + half); ax.set_ylim(zc - half, zc + half)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("x [mm]")
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(h, l, fontsize=7, loc="lower right", framealpha=0.92)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conforming", default=str(MESH / "ili_mesh_healthy_tri.msh"))
    ap.add_argument("--staircase", default=str(MESH / "ili_mesh_healthy_stair_tri.msh"))
    ap.add_argument("--cracked", default=str(MESH / "ili_mesh.msh"))
    ap.add_argument("--half", type=float, default=0.9, help="zoom half-width [mm]")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    z_id = Z_C + R_ID                                    # ID on the beam axis = 20.000 mm
    # WHERE to zoom for the staircase matters, and getting it wrong makes the figure lie by
    # omission. At the arc APEX (x = 38.25) the quantised level is flat for +-3.1 mm - z only
    # falls by half a pixel over that distance - so a tight zoom there shows NO steps at all
    # and the staircase looks harmless. The staircase error is worst where the wall is STEEP,
    # so panels 2 and 3 sit off-axis at x = 5 mm, where the surface drops 50 um every 0.29 mm.
    x_off = 5.0
    z_off = Z_C + math.sqrt(R_ID ** 2 - (x_off - X_C) ** 2)
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.0))
    panel(axes[0], Path(args.conforming), X_C, z_id - 6.0, 9.0,
          "1. CONFORMING ID, 18 mm wide\nthe mesh follows the true curvature",
          arcs=((R_ID, "lime"),))
    panel(axes[1], Path(args.conforming), x_off, z_off, 0.8,
          f"2. CONFORMING, zoomed x = {x_off:.0f} mm (1.6 mm wide)\nedges lie ON the arc: "
          f"sagitta 0.05 um",
          arcs=((R_ID, "lime"),))
    panel(axes[2], Path(args.staircase), x_off, z_off, 0.8,
          f"3. STAIRCASED, same window\n50 um pixel steps (k-Wave's grid); arc cuts across, "
          f"35 um error",
          arcs=((R_ID, "lime"),))
    panel(axes[3], Path(args.cracked), NOTCH_X, Z_TIP + 0.3, args.half,
          "4. NOTCH TIP, cracked mesh\na real void: corners exact, traction-free",
          arcs=(), notch=True)
    axes[0].set_ylabel("z [mm]")
    fig.suptitle("What 'conforming' means at the interface - drawn from the actual meshes",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = OUT / "mesh_zoom.png"
    fig.savefig(p, dpi=170)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
