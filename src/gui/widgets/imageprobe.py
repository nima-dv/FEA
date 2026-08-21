r"""Beamformed-image probe: one panel of an images_*.npz, on the project's own dB scale.

THE SCALE IS COPIED, NOT CHOSEN
repro/compare_images.py renders every published panel as

    vmax = img.max()
    db   = 20 * log10(maximum(img, vmax * 1e-4) / vmax)
    imshow(db.T, origin="lower", aspect="equal", cmap="inferno", vmin=-40, vmax=0,
           extent=[x[0], x[-1], z[0], z[-1]])

and this module reproduces that expression character for character, including the 1e-4 floor
that keeps log10(0) out of the array. dB relative to THAT PANEL'S OWN maximum is what makes two
panels comparable when their absolute amplitudes differ by 8x, and inferno plus the -40 dB
floor is the convention the whole record is read against. Re-tinting or re-flooring would make
a GUI panel and a published panel disagree about what the same data looks like, which is the
one thing this view must never do.

WHATEVER LABELS THE FILE HOLDS
The picker is built from the `<label>_img` keys actually present - one, two or five. Nothing
here expects a particular pair: the published files on disk happen to carry `k-Wave` and `FEM`,
but the k-Wave comparison is no longer part of the GUI contract, so assuming it would break on
the first file written without it.

NO OVERLAYS ON THE IMAGE
No wall arcs, no marker on the notch. compare_images.py writes an annotated figure and a
_nooverlay twin precisely because a reviewer who has been shown where to look cannot judge
detectability. The crosshair is a different thing: it is transient, it follows the cursor, and
it is gone the moment the mouse leaves. Same for the profile line, which the user places.

WHY THESE LOAD ON THE GUI THREAD
An images_*.npz is ~0.5 MB - two 370x358 float32 panels and their axes. Reading one is
sub-millisecond, so the thread machinery the wavefield tab needs would be pure ceremony here.
The rule is "do not block the event loop", not "always use a thread".
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_GUI_ROOT = Path(__file__).resolve().parents[1]
if str(_GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_GUI_ROOT))

from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QVBoxLayout,    # noqa: E402
                               QWidget)

from widgets.mplcanvas import ACCENT, INK, INK_SOFT, MplCanvas, style_axes     # noqa: E402

DB_FLOOR = -40.0                 # compare_images.py: vmin=-40
FLOOR_FRAC = 1e-4                # ... and the 20*log10 argument floor that goes with it
COMPARE_DIR = "compare"


def image_sets(root: Path) -> list[Path]:
    """Every beamformed image file on disk, newest first. Read-only, always."""
    d = root / COMPARE_DIR
    if not d.is_dir():
        return []
    return sorted(d.glob("images_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)


def to_db(img: np.ndarray) -> np.ndarray:
    """Amplitude -> dB relative to this panel's own maximum. See the module docstring."""
    vmax = float(img.max())
    if vmax <= 0.0:
        return np.full(img.shape, DB_FLOOR, dtype=float)
    return 20.0 * np.log10(np.maximum(img, vmax * FLOOR_FRAC) / vmax)


def load_images(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """({label: dB panel (nx, nz)}, x [mm], z [mm]) from one images_*.npz."""
    d = np.load(path)
    panels = {k[:-4]: to_db(d[k]) for k in d.files if k.endswith("_img")}
    if not panels:
        raise ValueError("%s holds no <label>_img array (found %s)" % (path.name, d.files))
    return panels, d["x"], d["z"]


class ImageProbe(QWidget):
    """Pick a file and a panel, read dB under the cursor, drag a line profile."""

    def __init__(self, root: Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from views.results import results_root
        # data/results only - the GUI's own run output. presentation/ is the published record
        # and is never read by this app (see views/results.py module docstring).
        self.root = root or results_root()
        self.panels: dict[str, np.ndarray] = {}
        self.x = np.empty(0)
        self.z = np.empty(0)
        self._im: Any = None
        self._cb: Any = None
        self._vline: Any = None
        self._hline: Any = None
        self._profile_line: Any = None       # the draggable one, on the image
        self._profile: Any = None            # the curve, in the axes below
        self._jz = 0
        self._dragging = False

        self.picker = QComboBox()
        for p in image_sets(self.root):
            self.picker.addItem(p.name, p)
        self.panel_pick = QComboBox()
        self.status = QLabel("")
        self.status.setStyleSheet("color:%s;" % INK_SOFT)
        self.canvas = MplCanvas()

        # Two axes on one figure: image on top, profile beneath, x shared so a feature and its
        # profile peak line up by construction rather than by eye.
        fig = self.canvas.fig
        self.canvas.ax.remove()
        gs = fig.add_gridspec(2, 1, height_ratios=(3.0, 1.0))
        self.canvas.ax = fig.add_subplot(gs[0])
        self.prof_ax = fig.add_subplot(gs[1], sharex=self.canvas.ax)
        style_axes(self.canvas.ax)
        style_axes(self.prof_ax)

        bar = QHBoxLayout()
        bar.setContentsMargins(8, 6, 8, 0)
        bar.addWidget(QLabel("images"))
        bar.addWidget(self.picker, 2)
        bar.addWidget(QLabel("panel"))
        bar.addWidget(self.panel_pick, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.status)
        lay.addWidget(self.canvas, 1)

        self.picker.currentIndexChanged.connect(lambda _i: self.load_current())
        self.panel_pick.currentIndexChanged.connect(lambda _i: self._draw())
        self.canvas.set_readout(self._readout)
        self.canvas.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.canvas.mpl_connect("motion_notify_event", self._on_motion)

        if self.picker.count():
            self.load_current()
        else:
            self.canvas.message("no images_*.npz found under\n%s"
                                % (self.root / COMPARE_DIR))

    # ---- loading -------------------------------------------------------------------

    def load_current(self) -> None:
        path = self.picker.currentData()
        if path is None:
            return
        try:
            self.panels, self.x, self.z = load_images(path)
        except (ValueError, OSError) as exc:
            self.panels = {}
            self.status.setText("%s: %s" % (type(exc).__name__, exc))
            self.canvas.message("could not read this file\n%s" % exc)
            return
        # Rebuild the panel picker without letting its signal redraw a half-built state.
        self.panel_pick.blockSignals(True)
        self.panel_pick.clear()
        for label in self.panels:
            self.panel_pick.addItem(label, label)
        self.panel_pick.blockSignals(False)
        self.panel_pick.setEnabled(len(self.panels) > 1)
        self._jz = int(self.z.size // 2)
        self._draw()

    # ---- drawing -------------------------------------------------------------------

    def _db(self) -> np.ndarray | None:
        label = self.panel_pick.currentData()
        return self.panels.get(label) if label else None

    def _draw(self) -> None:
        db = self._db()
        if db is None:
            return
        if self._cb is not None:
            self._cb.remove()
            self._cb = None
        self.canvas.clear()
        self.prof_ax.clear()
        style_axes(self.prof_ax)
        ax = self.canvas.ax
        ext = [float(self.x[0]), float(self.x[-1]), float(self.z[0]), float(self.z[-1])]
        # db.T, origin lower, equal aspect, inferno, -40..0: compare_images.py's own call.
        self._im = ax.imshow(db.T, origin="lower", aspect="equal", cmap="inferno",
                             vmin=DB_FLOOR, vmax=0.0, extent=ext)
        ax.set_ylabel("z [mm]")
        self._cb = self.canvas.colorbar(self._im, "dB re panel max")
        self._vline = ax.axvline(float(self.x[0]), color=INK, lw=0.7, alpha=0.7)
        self._hline = ax.axhline(float(self.z[0]), color=INK, lw=0.7, alpha=0.7)
        self._profile_line = ax.axhline(float(self.z[self._jz]), color=ACCENT, lw=1.1)
        self.prof_ax.set_xlabel("x [mm]")
        self.prof_ax.set_ylabel("dB")
        self.prof_ax.set_ylim(DB_FLOOR, 2.0)
        self._profile, = self.prof_ax.plot(self.x, db[:, self._jz], color=ACCENT, lw=1.0)
        self._update_profile()
        self.canvas.canvas.draw_idle()

    def _update_profile(self) -> None:
        db = self._db()
        if db is None or self._profile is None:
            return
        z0 = float(self.z[self._jz])
        self._profile.set_data(self.x, db[:, self._jz])
        self._profile_line.set_ydata([z0, z0])
        self.status.setText(
            "%s   panel %s   %d x %d   x %.2f..%.2f mm   z %.2f..%.2f mm   "
            "profile at z = %.2f mm   0 dB = this panel's own max"
            % (self.picker.currentText(), self.panel_pick.currentText(),
               db.shape[0], db.shape[1], self.x[0], self.x[-1], self.z[0], self.z[-1], z0))
        self.canvas.canvas.draw_idle()

    # ---- interaction ---------------------------------------------------------------

    def _at(self, x: float, z: float) -> tuple[int, int]:
        return (int(np.abs(self.x - x).argmin()), int(np.abs(self.z - z).argmin()))

    def _readout(self, x: float, z: float, ax: Any) -> str:
        db = self._db()
        if db is None or ax is not self.canvas.ax:
            return "x %8.3f   %8.3f" % (x, z)
        i, j = self._at(x, z)
        return "x %7.2f mm   z %7.2f mm   %+7.2f dB" % (self.x[i], self.z[j], db[i, j])

    def _busy(self) -> bool:
        """True while the navigation toolbar owns the mouse (pan or zoom armed)."""
        tb = self.canvas.toolbar
        return bool(tb is not None and getattr(tb, "mode", ""))

    def _on_press(self, event: Any) -> None:
        if event.inaxes is not self.canvas.ax or self._busy() or event.button != 1:
            return
        self._dragging = True
        self._move_profile(event.ydata)

    def _on_release(self, _event: Any) -> None:
        self._dragging = False

    def _on_motion(self, event: Any) -> None:
        db = self._db()
        if db is None or event.inaxes is not self.canvas.ax:
            return
        # Crosshair follows the cursor - transient and user-driven, not an annotation.
        self._vline.set_xdata([event.xdata, event.xdata])
        self._hline.set_ydata([event.ydata, event.ydata])
        if self._dragging:
            self._move_profile(event.ydata)
        else:
            self.canvas.canvas.draw_idle()

    def _move_profile(self, z: float | None) -> None:
        if z is None:
            return
        self._jz = self._at(0.0, z)[1]
        self._update_profile()


def demo() -> None:
    """Self-check against a synthetic images_*.npz: the dB scale, the readout, the profile.

    Built in a temp tree, never read from presentation/: this widget's default root is
    data/results (see __init__), and its self-check must not depend on the published record
    either - a synthetic fixture is also more precise, since every expected number is known
    in advance rather than reverse-engineered from a real file.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import shutil
    import tempfile
    from types import SimpleNamespace
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    root = Path(tempfile.mkdtemp(prefix="fea_imageprobe_"))
    try:
        (root / COMPARE_DIR).mkdir(parents=True)
        nx, nz = 90, 220
        x = np.linspace(0.0, 50.0, nx)
        z = np.linspace(0.0, 30.0, nz)
        xv, zv = np.meshgrid(x, z, indexing="ij")
        fem = 1.0 + np.sin(xv / 6.0) ** 2 + np.cos(zv / 4.0) ** 2          # strictly positive
        kwave = 1.0 + np.cos(xv / 5.0) ** 2 + np.sin(zv / 3.0) ** 2
        path = root / COMPARE_DIR / "images_sample.npz"
        np.savez(path, **{"FEM_img": fem, "k-Wave_img": kwave, "x": x, "z": z})

        found = image_sets(root)
        assert found == [path], found

        raw = np.load(path)
        panels, xr, zr = load_images(path)
        assert len(panels) == 2 and all(v.ndim == 2 for v in panels.values())
        label = next(iter(panels))
        db, img = panels[label], raw[label + "_img"]
        # The scale must BE compare_images.py's, not merely resemble it.
        vmax = float(img.max())
        want = 20.0 * np.log10(np.maximum(img, vmax * FLOOR_FRAC) / vmax)
        assert np.allclose(db, want), np.abs(db - want).max()
        assert abs(db.max()) < 1e-9, "the peak of a panel must sit at exactly 0 dB"
        # The array floor is the 1e-4 amplitude clamp (-80 dB); -40 is where the COLOUR scale
        # bottoms out. Conflating the two would silently change what the image shows.
        assert db.min() >= 20.0 * np.log10(FLOOR_FRAC) - 1e-6, db.min()
        assert db.shape == (xr.size, zr.size), (db.shape, xr.size, zr.size)

        w = ImageProbe(root)
        w.resize(900, 700)
        w.show()
        app.processEvents()
        assert w._im is not None, w.status.text()
        assert w._im.get_cmap().name == "inferno"
        assert w._im.get_clim() == (DB_FLOOR, 0.0)
        assert tuple(w._im.get_extent()) == (float(xr[0]), float(xr[-1]), float(zr[0]),
                                             float(zr[-1]))
        assert w.canvas.ax.get_aspect() == 1.0, "a beamformed image must be shown to true aspect"
        # No annotation on the image: only the crosshair pair and the user's profile line.
        assert not w.canvas.ax.patches and not w.canvas.ax.collections
        assert len(w.canvas.ax.lines) == 3, [l.get_color() for l in w.canvas.ax.lines]
        assert not w.canvas.ax.texts

        # Readout: the number under the cursor must be the number in the array.
        i, j = 40, 60
        r = w._readout(float(xr[i]), float(zr[j]), w.canvas.ax)
        assert "%+7.2f dB" % w.panels[w.panel_pick.currentData()][i, j] in r, r
        # ... and the readout must not claim dB while the cursor is in the profile axes.
        assert "dB" not in w._readout(1.0, 2.0, w.prof_ax)

        # Dragging the profile line moves both the line and the curve beneath it.
        before = w._profile.get_ydata().copy()
        w._on_press(SimpleNamespace(inaxes=w.canvas.ax, button=1, xdata=float(xr[i]),
                                    ydata=float(zr[5])))
        assert w._dragging and w._jz == 5
        w._on_motion(SimpleNamespace(inaxes=w.canvas.ax, button=1, xdata=float(xr[i]),
                                     ydata=float(zr[200])))
        w._on_release(None)
        assert w._jz == 200 and not w._dragging
        assert not np.array_equal(w._profile.get_ydata(), before), "the profile did not follow"
        assert w._profile_line.get_ydata()[0] == float(zr[200])
        assert np.allclose(w._profile.get_ydata(), w.panels[w.panel_pick.currentData()][:, 200])
        assert "profile at z" in w.status.text()

        w.panel_pick.setCurrentIndex(1)
        app.processEvents()
        assert w.panel_pick.currentData() != label
        assert abs(w._im.get_array().max()) < 1e-6, "each panel is 0 dB at its own peak"
        print("imageprobe.demo: ok, synthetic fixture, panels %s, %dx%d"
              % (list(panels), db.shape[0], db.shape[1]))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    demo()
