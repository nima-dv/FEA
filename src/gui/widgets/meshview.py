r"""Mesh inspector: the cells of a real .msh, coloured by size or quality, against the exact wall.

NO meshio ON THE GUI VENV
The backend has meshio (viz/mesh_zoom.py imports it), but `.venv-gui` does not - checked with
the venv interpreter, `import meshio` fails. Rather than quietly adding a dependency to the
host venv, this module parses the one format that is actually on disk: gmsh ASCII 4.1, which
is what `mesh/ili_mesh.py` writes. `read_msh` handles nodes, triangles, quads and tagged
facets and nothing else - see `_MSH_NOTE`. If meshio ever lands in the venv this parser can be
deleted; it is ~60 lines, not an abstraction.

WHY THE FACET OVERLAY IS THE POINT
Following viz/mesh_zoom.py: the mesh's OWN inner/outer wall facets are drawn on top of the
exact circle, computed independently from the geometry. On a conforming mesh the circle
disappears under the facets; on the staircase mesh (`*_stair_tri.msh`) it visibly cuts across
50 um steps. Without both curves present, conformity has to be inferred from the cell pattern,
which is exactly what the reader should not have to do.

WHY size AND quality, AND NOTHING DERIVED
Cell size drives the CFL step and nodes-per-wavelength; shape quality drives conditioning.
Both are measured off the coordinates in the file. This view computes no physics - it does not
convert size into a nodes/wavelength claim, because that is the solver's number to print
(src/gui/README.md, non-negotiable 1).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

_GUI_ROOT = Path(__file__).resolve().parents[1]
if str(_GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_GUI_ROOT))

from matplotlib.collections import LineCollection, PolyCollection              # noqa: E402
from PySide6.QtCore import Qt                                                  # noqa: E402
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,      # noqa: E402
                               QVBoxLayout, QWidget)

from widgets.mplcanvas import ACCENT, INK_SOFT, RULE, MplCanvas, Task                # noqa: E402

# Geometry in mm. The source of truth is mesh/ili_mesh.py in the backend, mirrored by
# viz/mesh_zoom.py; restated here because the backend modules import lib.paths and are not
# importable from the GUI venv. Only used to draw the exact circle the mesh is compared with.
X_C, Z_C = 38.25, -173.675
R_ID, R_OD = 193.675, 203.200
TAG_ID, TAG_OD = 11, 12                       # physical facet tags from mesh/ili_mesh.py

_MSH_NOTE = "gmsh ASCII 4.1: $Entities curve->physical map, $Nodes, $Elements types 1/2/3"

# Above this many cells the per-cell outline is dropped: 64k stroked polygons make every pan
# and zoom re-stroke 64k paths, and the outline is invisible at full-domain zoom anyway.
EDGE_LIMIT = 20000
MESH_DIR = "ili_mesh"


@dataclass
class Mesh:
    """One 2-D mesh, in mm. `cells` is one (m, k) index block per element type present."""
    path: Path
    pts: np.ndarray                      # (n, 2) mm
    cells: list[np.ndarray]              # (m, 3) triangles and/or (m, 4) quads
    facets: dict[int, np.ndarray]        # physical tag -> (m, 2) node indices

    @property
    def n_cells(self) -> int:
        return int(sum(c.shape[0] for c in self.cells))


def meshes(root: Path) -> list[Path]:
    """Every mesh on disk, newest first - the picker's contents. Read-only, always."""
    d = root / MESH_DIR
    if not d.is_dir():
        return []
    return sorted(d.glob("*.msh"), key=lambda p: p.stat().st_mtime, reverse=True)


# ---- the parser -------------------------------------------------------------------------

def _section(txt: str, name: str) -> str:
    i = txt.index("$" + name + "\n") + len(name) + 2
    return txt[i:txt.index("$End" + name)]


def _nums(line: str) -> list[float]:
    return [float(v) for v in line.split()]


def read_msh(path: Path, report: Callable[[str], None] | None = None) -> Mesh:
    """Nodes (mm), cells and physically-tagged facets out of a gmsh ASCII 4.1 file.

    In MSH 4.1 an element carries its GEOMETRIC entity tag, not its physical tag, so the
    physical group of a facet is only knowable via $Entities - that indirection is why this
    reads the entity table first. Getting it wrong would silently draw the wrong arc.
    """
    if report:
        report("reading %s" % path.name)
    txt = path.read_text(encoding="ascii", errors="replace")
    if "4.1" not in _section(txt, "MeshFormat").split("\n")[0]:
        raise ValueError("%s is not gmsh ASCII 4.1 (%s)" % (path.name, _MSH_NOTE))

    # $Entities: points, then curves. A curve line is
    #   tag  minx miny minz maxx maxy maxz  numPhys phys...  numBounding bnd...
    ent = _section(txt, "Entities").strip().split("\n")
    n_pt, n_cur = (int(v) for v in ent[0].split()[:2])
    phys_of_curve: dict[int, int] = {}
    for line in ent[1 + n_pt:1 + n_pt + n_cur]:
        f = line.split()
        n_phys = int(f[7])
        if n_phys:
            phys_of_curve[int(f[0])] = int(f[8])

    if report:
        report("nodes")
    nd = _section(txt, "Nodes").strip().split("\n")
    n_blk, n_node = (int(v) for v in nd[0].split()[:2])
    tags = np.empty(n_node, dtype=np.int64)
    xyz = np.empty((n_node, 3))
    i, at = 1, 0
    for _ in range(n_blk):
        cnt = int(nd[i].split()[3])
        i += 1
        tags[at:at + cnt] = np.array(nd[i:i + cnt], dtype=np.int64)
        i += cnt
        xyz[at:at + cnt] = np.array(" ".join(nd[i:i + cnt]).split(),
                                    dtype=float).reshape(cnt, 3)
        i += cnt
        at += cnt
    # Node tags are 1-based and need not be contiguous, so map tag -> row once.
    lut = np.zeros(int(tags.max()) + 1, dtype=np.int64)
    lut[tags] = np.arange(n_node)

    if report:
        report("elements")
    el = _section(txt, "Elements").strip().split("\n")
    n_blk = int(el[0].split()[0])
    tri: list[np.ndarray] = []
    quad: list[np.ndarray] = []
    facets: dict[int, list[np.ndarray]] = {}
    i = 1
    for _ in range(n_blk):
        _dim, ent_tag, etype, cnt = (int(v) for v in el[i].split()[:4])
        i += 1
        k = {1: 2, 2: 3, 3: 4}.get(etype)
        if k is None:                     # higher-order or 3-D: not drawn, not needed
            i += cnt
            continue
        blk = np.array(" ".join(el[i:i + cnt]).split(),
                       dtype=np.int64).reshape(cnt, k + 1)[:, 1:]
        i += cnt
        conn = lut[blk]
        if etype == 1:
            tag = phys_of_curve.get(ent_tag)
            if tag is not None:
                facets.setdefault(tag, []).append(conn)
        elif etype == 2:
            tri.append(conn)
        else:
            quad.append(conn)

    cells = [np.concatenate(b) for b in (tri, quad) if b]
    return Mesh(path=path, pts=xyz[:, :2] * 1e3, cells=cells,
                facets={t: np.concatenate(v) for t, v in facets.items()})


# ---- per-cell measures ------------------------------------------------------------------

def cell_metrics(pts: np.ndarray, cell: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                             np.ndarray]:
    """(edge lengths (m,k), area (m,), shape quality in (0,1]) for one block of cells.

    Quality is the isoperimetric ratio 2*sqrt(pi*A)/P normalised by the regular k-gon's own
    value, so 1.0 means equilateral triangle / square and one formula covers both element
    types. It penalises stretch AND skew, unlike a min/max edge ratio, which calls a badly
    sheared parallelogram perfect.
    """
    v = pts[cell]                                        # (m, k, 2)
    d = np.roll(v, -1, axis=1) - v
    ln = np.hypot(d[:, :, 0], d[:, :, 1])
    x, y = v[:, :, 0], v[:, :, 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, 1))
    k = cell.shape[1]
    ideal = math.sqrt(math.pi / (k * math.tan(math.pi / k)))
    per = ln.sum(axis=1)
    q = np.where(per > 0, 2.0 * np.sqrt(np.pi * area) / np.maximum(per, 1e-30) / ideal, 0.0)
    return ln, area, np.clip(q, 0.0, 1.0)


class MeshView(QWidget):
    """Pick a mesh, colour its cells, and see the wall it claims to resolve."""

    COLOR_BY = (("cell size (mean edge) [mm]", "size"), ("cell quality [1 = regular]", "qual"))

    def __init__(self, root: Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from views.results import results_root
        self.root = root or results_root()
        self.mesh: Mesh | None = None
        self._task: Task | None = None
        self._values: list[np.ndarray] = []           # per cell block, matching self.mesh
        self._centroids: np.ndarray = np.empty((0, 2))
        self._edges: np.ndarray = np.empty(0)         # every cell edge length, cell-aligned
        self._cell_of_edge: np.ndarray = np.empty(0, dtype=int)
        self._cb: Any = None

        self.picker = QComboBox()
        for p in meshes(self.root):
            self.picker.addItem(p.name, p)
        self.color_by = QComboBox()
        for label, key in self.COLOR_BY:
            self.color_by.addItem(label, key)
        self.walls = QCheckBox("wall facets vs exact circle")
        self.walls.setChecked(True)
        self.status = QLabel("")
        self.status.setStyleSheet("color:%s;" % INK_SOFT)
        self.canvas = MplCanvas()

        bar = QHBoxLayout()
        bar.setContentsMargins(8, 6, 8, 0)
        bar.addWidget(QLabel("mesh"))
        bar.addWidget(self.picker, 2)
        bar.addWidget(QLabel("colour"))
        bar.addWidget(self.color_by, 1)
        bar.addWidget(self.walls)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.status)
        lay.addWidget(self.canvas, 1)

        self.picker.currentIndexChanged.connect(lambda _i: self.load_current())
        self.color_by.currentIndexChanged.connect(self._draw)
        self.walls.toggled.connect(self._draw)
        for name in ("xlim_changed", "ylim_changed"):
            self.canvas.ax.callbacks.connect(name, lambda _ax: self._update_stats())
        self.canvas.set_readout(lambda x, z, ax: "x %8.3f mm   z %8.3f mm" % (x, z))

        if self.picker.count():
            self.load_current()
        else:
            self.canvas.message("no meshes found under\n%s" % (self.root / MESH_DIR))

    # ---- loading -------------------------------------------------------------------

    def load_current(self) -> None:
        path = self.picker.currentData()
        if path is None:
            return
        self.status.setText("loading %s ..." % path.name)
        # Off-thread even though a 5 MB mesh parses in well under a second: it is the same
        # Task the wavefield tab needs, and a 165 mm wide mesh is 5x the smallest one.
        self._task = Task(lambda report: read_msh(path, report), self)
        self._task.progress.connect(lambda s: self.status.setText("%s: %s" % (path.name, s)))
        self._task.loaded.connect(self._on_loaded)
        self._task.failed.connect(self._on_failed)
        self._task.start()

    def _on_failed(self, msg: str) -> None:
        self.mesh = None
        self.status.setText(msg)
        self.canvas.message("could not read this mesh\n%s" % msg)

    def _on_loaded(self, mesh: Mesh) -> None:
        self.mesh = mesh
        cen, ed, owner, size, qual = [], [], [], [], []
        off = 0                     # cells are stored per element type; the stats are global
        for cell in mesh.cells:
            ln, _area, q = cell_metrics(mesh.pts, cell)
            cen.append(mesh.pts[cell].mean(axis=1))
            size.append(ln.mean(axis=1))
            qual.append(q)
            ed.append(ln.ravel())
            owner.append(np.repeat(np.arange(cell.shape[0]) + off, ln.shape[1]))
            off += cell.shape[0]
        self._centroids = np.concatenate(cen)
        self._edges = np.concatenate(ed)
        self._cell_of_edge = np.concatenate(owner)
        self._size, self._qual = size, qual
        self._draw(first=True)

    # ---- drawing -------------------------------------------------------------------

    def _draw(self, *_a: Any, first: bool = False) -> None:
        m = self.mesh
        if m is None:
            return
        key = self.color_by.currentData()
        self._values = self._size if key == "size" else self._qual
        allv = np.concatenate(self._values) if self._values else np.zeros(1)
        # Robust limits: one sliver cell must not flatten the whole colour range.
        lo, hi = (float(np.percentile(allv, 1)), float(np.percentile(allv, 99)))
        if self._cb is not None:        # a cleared Axes does not take its colorbar with it
            self._cb.remove()
            self._cb = None
        self.canvas.clear()
        ax = self.canvas.ax
        first_coll = None
        for cell, val in zip(m.cells, self._values):
            coll = PolyCollection(
                m.pts[cell], array=val, cmap="viridis", clim=(lo, max(hi, lo + 1e-12)),
                linewidths=0.15 if m.n_cells <= EDGE_LIMIT else 0.0,
                edgecolors=RULE if m.n_cells <= EDGE_LIMIT else "none")
            ax.add_collection(coll)
            if first_coll is None:
                first_coll = coll
        if self.walls.isChecked():
            self._draw_walls(ax, m)
        ax.set_xlim(m.pts[:, 0].min(), m.pts[:, 0].max())
        ax.set_ylim(m.pts[:, 1].min(), m.pts[:, 1].max())
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        self.canvas.true_aspect(True)
        if first_coll is not None:
            self._cb = self.canvas.colorbar(first_coll, self.color_by.currentText())
        self.canvas.canvas.draw_idle()
        self._update_stats()

    def _draw_walls(self, ax: Any, m: Mesh) -> None:
        """The mesh's own ID/OD facets, then the exact circle underneath them.

        Same statement as viz/mesh_zoom.py: on a conforming mesh the dashed circle vanishes
        under the facets; on a staircase mesh it cuts across the steps.
        """
        for tag, col in ((TAG_ID, ACCENT), (TAG_OD, "#FFA24D")):
            seg = m.facets.get(tag)
            if seg is not None and seg.size:
                ax.add_collection(LineCollection(m.pts[seg], colors=col, linewidths=1.4,
                                                 zorder=4))
        th = np.linspace(np.deg2rad(-30), np.deg2rad(30), 4000)
        for R in (R_ID, R_OD):
            ax.plot(X_C + R * np.sin(th), Z_C + R * np.cos(th), color="#3FB950", lw=0.9,
                    ls="--", zorder=5)

    def _update_stats(self) -> None:
        """Cell count and edge-length spread for what is currently on screen."""
        m = self.mesh
        if m is None or not self._values:
            return
        ax = self.canvas.ax
        (x0, x1), (z0, z1) = ax.get_xlim(), ax.get_ylim()
        c = self._centroids
        inview = (c[:, 0] >= x0) & (c[:, 0] <= x1) & (c[:, 1] >= z0) & (c[:, 1] <= z1)
        n = int(inview.sum())
        if n == 0:
            self.status.setText("%s   %d cells total   nothing in view"
                                % (m.path.name, m.n_cells))
            return
        e = self._edges[inview[self._cell_of_edge]]
        self.status.setText(
            "%s   %d of %d cells in view   edge min %.4f  mean %.4f  max %.4f mm"
            % (m.path.name, n, m.n_cells, e.min(), e.mean(), e.max()))


def demo() -> None:
    """Self-check against a real mesh on disk: parser, units, tags, metrics and the widget."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from views.results import results_root
    app = QApplication.instance() or QApplication([])

    root = results_root()
    found = meshes(root)
    assert found, "no .msh under %s" % (root / MESH_DIR)

    conforming = next((p for p in found if p.name == "ili_mesh_healthy_tri.msh"), found[0])
    m = read_msh(conforming)
    assert m.pts.shape[1] == 2 and m.n_cells > 1000, (m.pts.shape, m.n_cells)
    assert m.pts[:, 0].max() > 80.0, "coordinates must be mm, not m"
    # The load-bearing check: the facets tagged 11 really are the ID arc. If the entity ->
    # physical mapping were wrong this would be off by millimetres, not microns.
    idf = m.facets.get(TAG_ID)
    assert idf is not None and idf.size, "no ID facets: the $Entities mapping is wrong"
    r = np.hypot(m.pts[idf][:, :, 0] - X_C, m.pts[idf][:, :, 1] - Z_C)
    assert np.abs(r - R_ID).max() < 0.01, np.abs(r - R_ID).max()

    ln, area, q = cell_metrics(m.pts, m.cells[0])
    assert (area > 0).all() and ln.min() > 0
    assert 0.0 < q.min() and q.max() <= 1.0 and q.mean() > 0.5, (q.min(), q.max(), q.mean())
    # A staircase mesh must NOT read as conforming: its ID facets are axis-aligned steps, so
    # they deviate from the exact circle by tens of microns.
    stair = next((p for p in found if "stair" in p.name), None)
    if stair is not None:
        ms = read_msh(stair)
        rs = np.hypot(ms.pts[ms.facets[TAG_ID]][:, :, 0] - X_C,
                      ms.pts[ms.facets[TAG_ID]][:, :, 1] - Z_C)
        assert np.abs(rs - R_ID).max() > 0.01, "staircase mesh should not lie on the circle"

    w = MeshView(root)
    w.resize(900, 560)
    w.show()
    assert w._task is not None
    w._task.wait(60000)
    app.processEvents()
    assert w.mesh is not None, "the threaded load produced nothing"
    assert "cells in view" in w.status.text(), w.status.text()
    n_before = len(w.canvas.ax.collections)
    assert n_before >= 1
    w.color_by.setCurrentIndex(1)                       # quality
    app.processEvents()
    assert w._values is w._qual
    w.walls.setChecked(False)
    app.processEvents()
    assert len(w.canvas.ax.collections) < n_before, "wall overlay did not come off"
    # Zooming must change the in-view statistics, not just the picture.
    before = w.status.text()
    w.canvas.ax.set_xlim(30.0, 45.0)
    w.canvas.ax.set_ylim(18.0, 26.0)
    app.processEvents()
    assert w.status.text() != before, "in-view stats did not follow the zoom"
    print("meshview.demo: ok, %s, %d cells, %d ID facets"
          % (conforming.name, m.n_cells, idf.shape[0]))


if __name__ == "__main__":
    demo()
