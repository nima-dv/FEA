r"""Wavefield scrubber: one snapshot record, frame by frame, on a colour scale that never moves.

WHAT IS PLOTTED, AND WHY IT IS TWO FIELDS
Straight from viz/wavefield_gif.py: by Helmholtz decomposition a P wave is curl-free and an S
wave is divergence-free, and the water carries no shear at all (mu = 0 there by construction).
So the water shows div u - the incident and reflected pressure beam - and the steel shows
curl u, the mode-converted shear wave that actually images the notch. One frame containing both
is the only way to watch mode conversion happen.

THE COLOUR LIMITS ARE COMPUTED ONCE AND HELD FIXED
p99.5 of |div u| over the water and of |curl u| over the steel, over the WHOLE record, then
never touched again. This is the load-bearing decision in the whole tab: with per-frame
autoscaling the arrival always fills the colour range, so a weak late echo looks exactly as
bright as the incident beam and the viewer reads normalisation as physics. With fixed limits,
brightness means amplitude. Each region keeps its own limit because div u and curl u have
different units and wildly different magnitudes - a shared scale renders one of them invisible.
(p99.5 rather than the max, so one hot cell cannot flatten the field.)

WHY IT IS LOADED WHOLE, ON A THREAD, AND NOT MEMORY-MAPPED
Checked with zipfile: every member of these npz files is DEFLATE-compressed (compress_type 8).
np.load's mmap_mode only works on an uncompressed .npy, and does not apply to npz at all, so
memory-mapping is not available - the choice is between decompressing and not seeing the data.
So: load once on a worker thread with per-array progress. The solver already writes float32,
and `div` alone is 252 MB of it in the 476 MB file, so div+curl are ~0.5 GB resident for that
record and ~2 GB for the 970 MB one.
ponytail: whole-record resident. If a 970 MB snapshot ever OOMs a machine, load every Nth
frame (a stride argument on load_record) - the colour limits would then come from the strided
subset, which is a real change to what p99.5 means and must be said in the status line.

SNAPSHOTS MAY NOT EXIST
`--snapshots N` is off by default and costs 700-970 MB a run, so a fresh checkout has none.
That is a normal state and shows as an empty-state message, not an error.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

_GUI_ROOT = Path(__file__).resolve().parents[1]
if str(_GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_GUI_ROOT))

import matplotlib                                                              # noqa: E402
import matplotlib.tri as mtri                                                  # noqa: E402
from PySide6.QtCore import Qt, QTimer                                          # noqa: E402
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,      # noqa: E402
                               QProgressBar, QPushButton, QSlider, QVBoxLayout, QWidget)

from widgets.mplcanvas import INK_SOFT, SURFACE, MplCanvas, Task              # noqa: E402

# Geometry in mm, mirroring viz/wavefield_gif.py (which mirrors mesh/ili_mesh.py). Used only
# to mask non-physical triangles and to draw the outlines.
X_C, Z_C = 38.25, -173.675
R_ID, R_OD = 193.675, 203.200
NOTCH_X, NOTCH_W, NOTCH_DEPTH = 38.25, 1.0, 4.0
Z_TIP = Z_C + R_OD - NOTCH_DEPTH                    # 25.525 mm

CLIP = 99.5                                         # same percentile the backend animation uses
SNAP_DIR = "ili_forward"
FPS = 12                                            # see the note on _tick
OUTLINE = "#101214"                                 # geometry lines; see draw_geometry
SMOOTH_PASSES = 2                                   # what the published stills used, when on


def snapshots(root: Path) -> list[Path]:
    """Every wavefield snapshot record on disk, newest first. Read-only, always."""
    d = root / SNAP_DIR
    if not d.is_dir():
        return []
    return sorted(d.glob("wavefield_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)


def build_triangulation(x_mm: np.ndarray, z_mm: np.ndarray, steel: np.ndarray,
                        has_notch: bool) -> tuple[mtri.Triangulation, int]:
    """Delaunay over the sample points with the non-physical triangles masked out.

    Ported from viz/wavefield_gif.build_triangulation, deliberately unchanged. The samples are
    a point cloud, so a raw Delaunay invents triangles in three places:
      1. past the curved OD, because the triangulation fills anything concave;
      2. across the notch VOID, for the same reason;
      3. straddling the water/steel interface - and that one is not cosmetic. Since the water
         plots div u and the steel plots curl u, a gouraud triangle with vertices on both
         sides blends two different physical quantities and paints a halo along the ID that is
         pure rendering artefact. Dropping those triangles leaves a one-cell gap exactly where
         the ID arc is drawn, so nothing is lost visually.
    Masked analytically from the known geometry, not by an edge-length heuristic, which would
    also eat legitimate triangles in the coarse fluid region.
    """
    tri = mtri.Triangulation(x_mm, z_mm)
    tx = x_mm[tri.triangles].mean(axis=1)
    tz = z_mm[tri.triangles].mean(axis=1)
    bad = np.hypot(tx - X_C, tz - Z_C) > R_OD
    if has_notch:
        bad |= (np.abs(tx - NOTCH_X) <= NOTCH_W / 2) & (tz >= Z_TIP)
    ns = steel[tri.triangles].sum(axis=1)
    bad |= (ns > 0) & (ns < 3)
    tri.set_mask(bad)
    return tri, int(bad.sum())


@dataclass
class Grid:
    """A fixed sample-cloud -> pixel operator, built once per record.

    WHY THIS EXISTS: drawing a frame as `tripcolor(..., shading="gouraud")` over this record's
    520k triangles measured 1.3 s per frame. A slider that costs 1.3 s a step is not a
    scrubber, and every one of those seconds is spent blocking the GUI thread. But the
    triangulation and the display grid never change between frames, so which triangle covers a
    pixel - and with what barycentric weights - is fixed too. Precompute that once and a frame
    becomes one gather plus one dot product, a few ms.

    The masking is unchanged: the trifinder is built on the MASKED triangulation, so a pixel
    over a masked triangle (past the OD, across the notch void, straddling the ID) returns -1
    and stays blank. Same three exclusions, same reasons, just resolved per pixel.
    """
    extent: tuple[float, float, float, float]        # x0, x1, z0, z1 [mm]
    shape: tuple[int, int]                           # nz, nx
    ok: np.ndarray                                   # (npix,) bool - pixel is inside a triangle
    idx: np.ndarray                                  # (nok, 3) node indices
    w: np.ndarray                                    # (nok, 3) barycentric weights

    def apply(self, vals: np.ndarray) -> np.ndarray:
        out = np.full(self.ok.size, np.nan, dtype=np.float32)
        out[self.ok] = (vals[self.idx] * self.w).sum(axis=1)
        return np.ma.masked_invalid(out.reshape(self.shape))


def build_grid(tri: mtri.Triangulation, x: np.ndarray, z: np.ndarray,
               nx: int = 1100) -> Grid:
    """Pixel -> (triangle nodes, weights). nx is chosen so a pixel is finer than the sample
    spacing, which is what keeps this from being a downsample."""
    x0, x1, z0, z1 = float(x.min()), float(x.max()), 0.0, float(z.max())
    dx = (x1 - x0) / (nx - 1)
    nz = max(int(round((z1 - z0) / dx)) + 1, 2)
    gx, gz = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(z0, z1, nz))
    # The trifinder returns -1 for a pixel in a MASKED triangle as well as for one outside the
    # hull (verified against this record), which is what carries the masking through unchanged.
    ti = tri.get_trifinder()(gx.ravel(), gz.ravel())
    ok = ti >= 0
    t = tri.triangles[ti[ok]]
    ax_, az_ = x[t[:, 0]], z[t[:, 0]]
    v0x, v0z = x[t[:, 1]] - ax_, z[t[:, 1]] - az_
    v1x, v1z = x[t[:, 2]] - ax_, z[t[:, 2]] - az_
    v2x, v2z = gx.ravel()[ok] - ax_, gz.ravel()[ok] - az_
    den = v0x * v1z - v1x * v0z
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    lb = (v2x * v1z - v1x * v2z) / den
    lc = (v0x * v2z - v2x * v0z) / den
    w = np.stack([1.0 - lb - lc, lb, lc], axis=1).astype(np.float32)
    return Grid(extent=(x0, x1, z0, z1), shape=(nz, nx), ok=ok, idx=t.astype(np.int32), w=w)


def smoothing_edges(tri: mtri.Triangulation, x: np.ndarray, z: np.ndarray,
                    steel: np.ndarray) -> np.ndarray:
    """Edge list for neighbour averaging, with the edges that must NOT be averaged removed.

    Ported from viz/wavefield_gif.smoothing_edges. Two exclusions, both load-bearing:
      * edges crossing the water/steel interface - the two sides hold different physical
        quantities (div u vs curl u), so averaging across the ID is meaningless;
      * unusually long edges, which are the ones bridging the notch void.
    """
    e = tri.edges
    same = steel[e[:, 0]] == steel[e[:, 1]]
    ln = np.hypot(x[e[:, 0]] - x[e[:, 1]], z[e[:, 0]] - z[e[:, 1]])
    return e[same & (ln < 3.0 * np.median(ln))]


def smooth_field(v: np.ndarray, e: np.ndarray, cnt: np.ndarray, iters: int) -> np.ndarray:
    """`iters` passes of self-plus-neighbour averaging over the sample cloud.

    Ported from viz/wavefield_gif.smooth_field, and cosmetic exactly as it is there: the
    samples are DISCONTINUOUS across cells, so a triangulation over the sample cloud shows a
    faint checkerboard where neighbouring cells disagree - grain at sample spacing (~0.08 mm),
    far finer than the 0.375 mm wave it sits on. The published stills use 2 passes and SAY so
    in their caption; this tab does the same in its title, and defaults to off.
    It is NOT a fix for under-sampling: if the wavefronts themselves look beaded, the snapshot
    needs a higher --snap-degree, because no averaging recovers information never sampled.
    """
    n = cnt.size
    for _ in range(iters):
        sm = (np.bincount(e[:, 0], weights=v[e[:, 1]], minlength=n)
              + np.bincount(e[:, 1], weights=v[e[:, 0]], minlength=n))
        v = np.where(cnt > 0, (v + sm) / (1 + cnt), v)
    return v


@dataclass
class Record:
    """One snapshot file, resident in RAM. Frames are normalised on access, not stored twice."""
    path: Path
    x: np.ndarray                  # (nc,) mm
    z: np.ndarray
    steel: np.ndarray              # (nc,) bool
    div: np.ndarray                # (nf, nc) float32
    curl: np.ndarray
    t_us: np.ndarray               # (nf,)
    degree: int
    angle: float
    has_notch: bool
    v_water: float                 # the two FIXED colour limits
    v_steel: float
    tri: mtri.Triangulation
    n_masked: int
    grid: Grid
    edges: np.ndarray              # intra-region edges, for the optional smoothing
    degree_of: np.ndarray          # neighbour count per sample, precomputed with them

    @property
    def n_frames(self) -> int:
        return int(self.div.shape[0])

    def frame(self, i: int, smooth: int = 0) -> np.ndarray:
        """Frame i per SAMPLE, in [-1, 1]: curl over the steel limit in the steel,
        div over the water limit everywhere else."""
        f = np.where(self.steel, self.curl[i] / max(self.v_steel, 1e-30),
                     self.div[i] / max(self.v_water, 1e-30))
        if smooth:
            f = smooth_field(f, self.edges, self.degree_of, smooth)
        return np.clip(f, -1.0, 1.0)

    def image(self, i: int, smooth: int = 0) -> np.ndarray:
        """Frame i as a masked 2-D image, ready for set_data."""
        return self.grid.apply(self.frame(i, smooth))


def load_record(path: Path, report: Callable[[str], None] | None = None) -> Record:
    """Read one wavefield_*.npz. Blocking and slow by nature - always call it via Task."""
    def say(msg: str) -> None:
        if report:
            report(msg)

    mb = path.stat().st_size / 1e6
    say("opening %s (%.0f MB)" % (path.name, mb))
    d = np.load(path)                     # NpzFile: each member decompresses on first access
    say("coordinates")
    x, z = d["x"] * 1e3, d["z"] * 1e3
    steel = d["steel"].astype(bool)
    t_us = d["t"] * 1e6
    degree = int(d["degree"]) if "degree" in d.files else 0
    angle = float(d["angle"]) if "angle" in d.files else 0.0
    say("div u (pressure, water) - %.0f MB compressed" % mb)
    div = d["div"].astype(np.float32)
    say("curl u (shear, steel)")
    curl = d["curl"].astype(np.float32)

    # Cracked or healthy mesh? On the cracked mesh the notch is a real void, so no sample
    # lies inside it. Margins keep boundary-hugging cells out of the test. (wavefield_gif.py)
    in_notch = (np.abs(x - NOTCH_X) <= NOTCH_W / 2 - 0.05) & (z >= Z_TIP + 0.05)
    has_notch = not bool(in_notch.any())

    say("colour limits over the whole record")
    v_water = float(np.percentile(np.abs(div[:, ~steel]), CLIP))
    v_steel = float(np.percentile(np.abs(curl[:, steel]), CLIP))
    say("triangulating %d samples" % x.size)
    tri, n_masked = build_triangulation(x, z, steel, has_notch)
    say("building the pixel operator")
    grid = build_grid(tri, x, z)
    edges = smoothing_edges(tri, x, z, steel)
    cnt = (np.bincount(edges[:, 0], minlength=x.size)
           + np.bincount(edges[:, 1], minlength=x.size))
    say("ready")
    return Record(path=path, x=x, z=z, steel=steel, div=div, curl=curl, t_us=t_us,
                  degree=degree, angle=angle, has_notch=has_notch, v_water=v_water,
                  v_steel=v_steel, tri=tri, n_masked=n_masked, grid=grid, edges=edges,
                  degree_of=cnt)


def draw_geometry(ax: Any, has_notch: bool) -> None:
    """ID arc, OD arc, notch outline and the array plane, so the wave is seen against the
    structure. This is the wavefield, not a beamformed image - the geometry is the subject
    here, and the same outlines are drawn in the published animation.

    Drawn DARK, not in the app's ink: RdBu_r is white in the middle, so most of the field is
    pale and a light outline disappears into it. The backend draws these in black for the same
    reason - the outline colour follows the colormap, not the widget theme."""
    th = np.linspace(np.deg2rad(-30), np.deg2rad(30), 400)
    for R in (R_ID, R_OD):
        ax.plot(X_C + R * np.sin(th), Z_C + R * np.cos(th), color=OUTLINE, lw=1.0, zorder=5)
    if has_notch:
        xl, xr = NOTCH_X - NOTCH_W / 2, NOTCH_X + NOTCH_W / 2
        zl = Z_C + np.sqrt(max(R_OD ** 2 - (xl - X_C) ** 2, 0.0))
        zr = Z_C + np.sqrt(max(R_OD ** 2 - (xr - X_C) ** 2, 0.0))
        ax.plot([xl, xl, xr, xr], [zl, Z_TIP, Z_TIP, zr], color=OUTLINE, lw=1.2, zorder=6)
    ax.axhline(0.0, color=OUTLINE, lw=1.2, alpha=0.6, zorder=5)  # the 256-element array plane


class WavefieldView(QWidget):
    """Pick a snapshot record, scrub time, play it back."""

    def __init__(self, root: Path | None = None, autoload: bool = True,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from views.results import results_root
        self.root = root or results_root()
        self.record: Record | None = None
        self._task: Task | None = None
        self._coll: Any = None
        self._cb: Any = None
        self._steps = 0

        self.picker = QComboBox()
        for p in snapshots(self.root):
            self.picker.addItem("%s  (%.0f MB)" % (p.name, p.stat().st_size / 1e6), p)
        self.bar = QProgressBar()
        self.bar.setMaximumHeight(10)
        self.bar.setTextVisible(False)
        self.bar.hide()
        self.status = QLabel("")
        self.status.setStyleSheet("color:%s;" % INK_SOFT)
        self.play = QPushButton("play")
        self.play.setCheckable(True)
        self.play.setEnabled(False)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.time = QLabel("")
        self.time.setStyleSheet("color:%s;" % INK_SOFT)
        self.time.setMinimumWidth(120)
        # Off by default: the raw samples are what the solver wrote. The published stills turn
        # this on and declare it in their caption; so does the title here, because a smoothed
        # frame that does not say so is a cosmetic edit passing as data.
        self.smooth = QCheckBox("cosmetic smoothing (%d passes)" % SMOOTH_PASSES)
        self.canvas = MplCanvas()

        bar = QHBoxLayout()
        bar.setContentsMargins(8, 6, 8, 0)
        bar.addWidget(QLabel("snapshot"))
        bar.addWidget(self.picker, 1)
        bar.addWidget(self.smooth)
        row = QHBoxLayout()
        row.setContentsMargins(8, 0, 8, 6)
        row.addWidget(self.play)
        row.addWidget(self.slider, 1)
        row.addWidget(self.time)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        lay.addWidget(self.canvas, 1)
        lay.addLayout(row)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.play.toggled.connect(self._on_play)
        self.slider.valueChanged.connect(self._show_frame)
        self.picker.currentIndexChanged.connect(lambda _i: self.load_current())
        self.smooth.toggled.connect(lambda _on: self._show_frame(self.slider.value()))
        self.canvas.set_readout(self._readout)

        if not self.picker.count():
            self.canvas.message("no snapshot runs found under\n%s\n\n"
                                "snapshots are off by default (700-970 MB a run);\n"
                                "run the forward solve with --snapshots N to make one"
                                % (self.root / SNAP_DIR))
        elif autoload:
            self.load_current()
        else:
            # views/inspector.py builds all three tabs at once and defers this one: half a
            # gigabyte should decompress when the user opens the tab, not when the app does.
            self.canvas.message("select a snapshot to load\n"
                                "(%d on disk, 480-970 MB each)" % self.picker.count())

    # ---- loading -------------------------------------------------------------------

    def load_current(self) -> None:
        path = self.picker.currentData()
        if path is None:
            return
        self.play.setChecked(False)
        self.play.setEnabled(False)
        self.slider.setEnabled(False)
        self._steps = 0
        self.bar.setRange(0, 9)             # the number of `say(...)` stages in load_record
        self.bar.setValue(0)
        self.bar.show()
        self.status.setText("loading %s ..." % path.name)
        self._task = Task(lambda report: load_record(path, report), self)
        self._task.progress.connect(self._on_progress)
        self._task.loaded.connect(self._on_loaded)
        self._task.failed.connect(self._on_failed)
        self._task.start()

    def _on_progress(self, msg: str) -> None:
        self._steps += 1
        self.bar.setValue(self._steps)
        self.status.setText(msg)

    def _on_failed(self, msg: str) -> None:
        self.bar.hide()
        self.record = None
        self.status.setText(msg)
        self.canvas.message("could not read this snapshot\n%s" % msg)

    def _on_loaded(self, rec: Record) -> None:
        self.bar.hide()
        self.record = rec
        self.status.setText(
            "%s   %d frames   %d samples   %d triangles (%d masked)   %.0f MB resident   "
            "fixed limits: water |div u| %.4g, steel |curl u| %.4g (p%.1f)"
            % (rec.path.name, rec.n_frames, rec.x.size, rec.tri.triangles.shape[0],
               rec.n_masked, (rec.div.nbytes + rec.curl.nbytes) / 1e6,
               rec.v_water, rec.v_steel, CLIP))
        if self._cb is not None:
            self._cb.remove()
            self._cb = None
        self.canvas.clear()
        ax = self.canvas.ax
        # vmin/vmax pinned to the pre-normalised range: set_data on later frames must not be
        # allowed to rescale, or the "fixed limits" promise above is quietly broken.
        cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad=SURFACE)
        self._coll = ax.imshow(rec.image(0, self._passes()), origin="lower",
                               extent=rec.grid.extent,
                               cmap=cmap, vmin=-1.0, vmax=1.0, interpolation="nearest")
        draw_geometry(ax, rec.has_notch)
        ax.set_xlim(rec.x.min(), rec.x.max())
        ax.set_ylim(0.0, rec.z.max())
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        self.canvas.true_aspect(True)
        self._cb = self.canvas.colorbar(
            self._coll, "normalised   water: div u (P)   |   steel: curl u (S)")
        self._title = ax.set_title("")
        self.slider.setRange(0, rec.n_frames - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.play.setEnabled(True)
        self._show_frame(0)

    # ---- playback ------------------------------------------------------------------

    def _on_play(self, on: bool) -> None:
        self.play.setText("pause" if on else "play")
        if on and self.record is not None:
            self._timer.start(int(1000 / FPS))
        else:
            self._timer.stop()

    def _tick(self) -> None:
        """Advance one frame, wrapping. FPS is deliberately modest: a gouraud tripcolor over
        ~260k triangles takes tens of ms to rasterise, and a timer faster than the redraw
        just queues events. ponytail: flat shading or frame decimation if playback drags."""
        if self.record is None:
            return
        self.slider.setValue((self.slider.value() + 1) % self.record.n_frames)

    def _passes(self) -> int:
        return SMOOTH_PASSES if self.smooth.isChecked() else 0

    def _show_frame(self, i: int) -> None:
        rec = self.record
        if rec is None or self._coll is None:
            return
        self._coll.set_data(rec.image(i, self._passes()))
        self.time.setText("t %7.2f us   %d/%d" % (rec.t_us[i], i + 1, rec.n_frames))
        # The title, not the status line: it is the part that survives a screenshot.
        self._title.set_text(
            "t = %.2f us   degree %d, steering %+.0f deg   -   illustrative; published "
            "numbers come from repro/compare_images.py%s"
            % (rec.t_us[i], rec.degree, rec.angle,
               "  |  %d cosmetic smoothing pass(es)" % self._passes()
               if self._passes() else ""))
        self.canvas.canvas.draw_idle()

    def _readout(self, x: float, z: float, _ax: Any) -> str:
        """Nearest sample: which region it is in, and that region's value at this frame."""
        rec = self.record
        if rec is None:
            return "x %8.3f mm   z %8.3f mm" % (x, z)
        j = int(np.argmin((rec.x - x) ** 2 + (rec.z - z) ** 2))
        i = self.slider.value()
        if bool(rec.steel[j]):
            what, val = "steel  curl u", float(rec.curl[i, j])
        else:
            what, val = "water  div u ", float(rec.div[i, j])
        return "x %8.3f mm   z %8.3f mm   %s %+.4g" % (x, z, what, val)


def demo() -> None:
    """Self-check against a real snapshot if one is on disk; the empty state if not."""
    import os
    import time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from views.results import results_root
    app = QApplication.instance() or QApplication([])

    root = results_root()
    found = snapshots(root)
    w = WavefieldView(root)
    w.resize(1000, 520)
    w.show()
    app.processEvents()

    if not found:
        # The degraded state is the thing under test when there is no data.
        assert not w.play.isEnabled() and not w.slider.isEnabled()
        assert any("no snapshot runs found" in t.get_text() for t in w.canvas.ax.texts)
        print("wavefieldview.demo: ok, no snapshots on disk, empty state shown")
        return

    smallest = min(found, key=lambda p: p.stat().st_size)
    w.picker.setCurrentIndex([w.picker.itemData(i) for i in range(w.picker.count())]
                             .index(smallest))
    app.processEvents()
    assert w._task is not None
    t0 = time.time()
    w._task.wait(600000)
    app.processEvents()
    load_s = time.time() - t0
    rec = w.record
    assert rec is not None, w.status.text()

    # The masking is the hard-won part: every one of the three classes must actually fire.
    tri = rec.tri
    tx = rec.x[tri.triangles].mean(axis=1)
    tz = rec.z[tri.triangles].mean(axis=1)
    mask = tri.mask
    assert mask is not None and rec.n_masked > 0
    assert mask[np.hypot(tx - X_C, tz - Z_C) > R_OD].all(), "triangles past the OD survived"
    ns = rec.steel[tri.triangles].sum(axis=1)
    assert mask[(ns > 0) & (ns < 3)].all(), "ID-straddling triangles survived: fake halo"
    if rec.has_notch:
        inside = (np.abs(tx - NOTCH_X) <= NOTCH_W / 2) & (tz >= Z_TIP)
        assert not inside.any() or mask[inside].all(), "notch void bridged"

    # Fixed limits: the same numbers must hold for every frame, and the artist must not
    # autoscale when a new frame is pushed in.
    assert rec.v_water > 0 and rec.v_steel > 0
    assert w._coll.get_clim() == (-1.0, 1.0)
    f0 = rec.frame(0)
    w.slider.setValue(rec.n_frames // 2)
    app.processEvents()
    assert w._coll.get_clim() == (-1.0, 1.0), "colour limits moved between frames"
    fm = rec.frame(rec.n_frames // 2)
    assert np.abs(fm).max() <= 1.0 and np.abs(f0).max() <= 1.0
    assert not np.array_equal(f0, fm), "scrubbing produced the same frame twice"
    # Both regions must carry signal, and the shear in the steel must peak LATER than the
    # first snapshot - the beam starts as pressure in the water and converts at the ID.
    w_rms = float(np.sqrt((rec.div[0, ~rec.steel] ** 2).mean()))
    s_peak = np.abs(rec.curl[:, rec.steel]).max(axis=1)
    assert w_rms > 0 and s_peak.max() > 0
    assert int(s_peak.argmax()) > 0, "no mode conversion after the first frame?"
    assert "t " in w.time.text() and "%d" % rec.n_frames in w.time.text()

    # Playback runs on a timer and must advance the frame without touching the limits.
    before = w.slider.value()
    w.play.setChecked(True)
    t1 = time.time()
    while time.time() - t1 < 1.0:
        app.processEvents()
    w.play.setChecked(False)
    assert w.slider.value() != before, "play did not advance"
    assert not w._timer.isActive()

    r = w._readout(float(rec.x[0]), float(rec.z[0]), w.canvas.ax)
    assert ("water" in r) or ("steel" in r), r

    # The pixel operator must reproduce the sample cloud, not smear across the masked gaps.
    img = rec.image(0)
    assert img.shape == rec.grid.shape and np.ma.is_masked(img), img.shape
    assert float(np.abs(img).max()) <= 1.0 + 1e-5    # float32 barycentric rounding
    assert 0.3 < rec.grid.ok.mean() < 0.98, rec.grid.ok.mean()   # some pixels in, some out
    # Smoothing is cosmetic, must be declared, and must actually reduce sample-scale grain.
    e = rec.edges
    raw, sm = rec.frame(60, 0), rec.frame(60, SMOOTH_PASSES)
    grain = lambda f: float(np.abs(f[e[:, 0]] - f[e[:, 1]]).mean())
    assert grain(sm) < 0.6 * grain(raw), (grain(raw), grain(sm))
    assert not w.smooth.isChecked(), "smoothing must default to off"
    w.smooth.setChecked(True)
    app.processEvents()
    assert "cosmetic smoothing" in w._title.get_text(), w._title.get_text()
    w.smooth.setChecked(False)
    app.processEvents()
    assert "cosmetic" not in w._title.get_text()
    assert "illustrative" in w._title.get_text()

    t2 = time.time()
    for i in range(10):
        rec.image(i)
    per_frame_ms = 100 * (time.time() - t2)
    assert per_frame_ms < 200, ("a frame step costs %.0f ms; scrubbing would stutter"
                                % per_frame_ms)
    print("wavefieldview.demo: ok, %s, %d frames, %d samples, load %.1f s, "
          "frame %.0f ms (was 1300 ms as gouraud tripcolor), water %.4g steel %.4g"
          % (rec.path.name, rec.n_frames, rec.x.size, load_s, per_frame_ms,
             rec.v_water, rec.v_steel))


if __name__ == "__main__":
    demo()
