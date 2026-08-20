r"""Matplotlib embedded in Qt, styled once, plus the one thread helper the Inspect tabs share.

WHY MATPLOTLIB AT ALL
The backend renders every published figure with matplotlib, so an inspector built on the same
library inherits the same colormaps, the same dB mapping and the same aspect handling by
construction. A PyVista/VTK viewer would be a second rendering stack to keep in agreement with
the first, for a problem that is 2-D anyway.

WHY THE PROVENANCE STAMP LIVES ON THE FIGURE
This view is for exploration; publication stays backend-rendered. The navigation toolbar can
save the canvas to a PNG, so a warning that existed only as a Qt label would not travel with
that file - the person who receives the PNG is exactly the person who needs to be told. So the
stamp is drawn into the figure, where a screenshot and a saved PNG both carry it.

WHY A QThread FOR LOADING
A wavefield snapshot is 480-970 MB of deflate-compressed npz. Decompressing that on the GUI
thread freezes the window for tens of seconds, which reads as a hung app. `Task` runs one
callable off-thread and reports progress as text; all three tabs use it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

_GUI_ROOT = Path(__file__).resolve().parents[1]
if str(_GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_GUI_ROOT))

import matplotlib                                                              # noqa: E402
matplotlib.use("QtAgg")
from matplotlib.axes import Axes                                               # noqa: E402
from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg,              # noqa: E402
                                               NavigationToolbar2QT)
from matplotlib.figure import Figure                                           # noqa: E402

from PySide6.QtCore import QSize, Qt, QThread, Signal                          # noqa: E402
from PySide6.QtGui import QColor, QFont, QPalette                              # noqa: E402
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget        # noqa: E402

from theme import PALETTE                                                      # noqa: E402

# One source for the shades; theme/__init__.py owns the values (README's table).
BG = PALETTE["bg"]                  # figure face  #0B0B0C
SURFACE = PALETTE["surface"]        # axes face    #16181B
INK = PALETTE["ink"]                # text         #E8E8EA
INK_SOFT = PALETTE["ink_soft"]
RULE = PALETTE["rule"]              # spines       #2A2E33
ACCENT = PALETTE["accent"]
MONO = "Consolas, DejaVu Sans Mono, monospace"

STAMP = "exploration view - not the published figure"


class Task(QThread):
    """Run `fn(report)` on a worker thread; `report(str)` pushes a progress line back.

    Deliberately not QThreadPool/QRunnable: there is exactly one load in flight per tab, and
    a QThread with two signals is less machinery than a pool plus a signal-carrying runnable.
    """

    progress = Signal(str)
    loaded = Signal(object)         # not "finished" - QThread already has that signal
    failed = Signal(str)

    def __init__(self, fn: Callable[[Callable[[str], None]], Any],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:          # noqa: D102 - QThread entry point
        try:
            self.loaded.emit(self._fn(self.progress.emit))
        except Exception as exc:    # a bad file must cost a message, not the app
            self.failed.emit("%s: %s" % (type(exc).__name__, exc))


def style_axes(ax: Axes) -> None:
    """Dark palette on one Axes. Public because tabs add profile/colorbar axes of their own."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SOFT, which="both", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(RULE)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.title.set_fontsize(9.5)


class MplCanvas(QWidget):
    """Figure + slim toolbar + coordinate readout, on the app palette."""

    def __init__(self, toolbar: bool = True, readout: bool = True,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Set a dark QPalette BEFORE the toolbar exists: NavigationToolbar2QT decides at
        # construction whether to invert its icons, by looking at its own palette value. With
        # only a stylesheet applied the palette still reads light and the icons come out black
        # on black. Children inherit this palette, so setting it here is enough.
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(BG))
        pal.setColor(QPalette.ColorRole.Base, QColor(BG))
        pal.setColor(QPalette.ColorRole.Button, QColor(BG))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(INK))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(INK))
        pal.setColor(QPalette.ColorRole.Text, QColor(INK))
        self.setPalette(pal)

        self.fig = Figure(figsize=(7.0, 4.4), dpi=100, facecolor=BG, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setStyleSheet("background-color: %s;" % BG)
        self.ax: Axes = self.fig.add_subplot(111)
        style_axes(self.ax)
        self.fig.text(0.995, 0.004, STAMP, ha="right", va="bottom", fontsize=6.5,
                      color=INK_SOFT, alpha=0.8, zorder=100)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.toolbar: NavigationToolbar2QT | None = None
        if toolbar:
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.toolbar.setIconSize(QSize(16, 16))
            self.toolbar.setStyleSheet(
                "QToolBar { background:%s; border:0; border-bottom:1px solid %s; padding:1px; }"
                "QToolButton { color:%s; padding:2px; }"
                "QToolButton:hover { background:%s; }"
                "QLabel { color:%s; }" % (SURFACE, RULE, INK, PALETTE["surface_hi"], INK_SOFT))
            lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)

        self.readout = QLabel("")
        self.readout.setFont(QFont(MONO.split(",")[0], 8))
        self.readout.setStyleSheet("color:%s; padding:1px 6px;" % INK_SOFT)
        self.readout.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.readout)

        self._fmt: Callable[[float, float, Axes], str] | None = None
        if readout:
            self.canvas.mpl_connect("motion_notify_event", self._on_move)

    # ---- helpers the tabs use ------------------------------------------------------

    def set_readout(self, fmt: Callable[[float, float, Axes], str] | None) -> None:
        """Install a formatter: fmt(x, y, axes) -> one line. None restores the default."""
        self._fmt = fmt

    def true_aspect(self, on: bool = True) -> None:
        """Equal mm per pixel in x and z. A pipe wall drawn to the wrong aspect makes a 20 deg
        beam look like a 30 deg one, so geometry views must never autoscale the aspect."""
        self.ax.set_aspect("equal" if on else "auto", adjustable="box")

    def colorbar(self, mappable: Any, label: str = "", ax: Axes | None = None) -> Any:
        cb = self.fig.colorbar(mappable, ax=ax or self.ax, pad=0.015, shrink=0.92)
        cb.set_label(label, color=INK, fontsize=8)
        cb.ax.tick_params(colors=INK_SOFT, labelsize=7.5)
        cb.outline.set_edgecolor(RULE)
        return cb

    def clear(self) -> None:
        """Reset the main axes, keeping the styling and the provenance stamp."""
        self.ax.clear()
        style_axes(self.ax)

    def message(self, text: str) -> None:
        """Replace the plot with a centred message (empty state, load error)."""
        self.clear()
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, text, ha="center", va="center", color=INK_SOFT,
                     fontsize=10, transform=self.ax.transAxes, wrap=True)
        self.canvas.draw_idle()

    def _on_move(self, event: Any) -> None:
        if event.inaxes is None or event.xdata is None:
            self.readout.setText("")
            return
        if self._fmt is not None:
            self.readout.setText(self._fmt(event.xdata, event.ydata, event.inaxes))
        else:
            self.readout.setText("x %8.3f   y %8.3f" % (event.xdata, event.ydata))


def demo() -> None:
    """Self-check: the canvas builds, styles, stamps, and reports coordinates."""
    import os
    from types import SimpleNamespace
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    c = MplCanvas()
    c.ax.plot([0, 1], [0, 1], color=ACCENT)
    c.true_aspect(True)
    c.canvas.draw()
    app.processEvents()

    assert c.fig.get_facecolor()[:3] == matplotlib.colors.to_rgb(BG), c.fig.get_facecolor()
    assert c.ax.get_facecolor()[:3] == matplotlib.colors.to_rgb(SURFACE)
    assert c.ax.get_aspect() == 1.0
    texts = [t.get_text() for t in c.fig.texts]
    assert STAMP in texts, "the provenance stamp must be on the figure, not just the widget"
    assert c.toolbar is not None and c.toolbar.iconSize().width() == 16

    # Readout, default and custom formatter.
    c._on_move(SimpleNamespace(inaxes=c.ax, xdata=1.5, ydata=-2.25))
    assert "1.500" in c.readout.text(), c.readout.text()
    c.set_readout(lambda x, y, ax: "probe %.1f" % x)
    c._on_move(SimpleNamespace(inaxes=c.ax, xdata=3.0, ydata=0.0))
    assert c.readout.text() == "probe 3.0"
    c._on_move(SimpleNamespace(inaxes=None, xdata=None, ydata=None))
    assert c.readout.text() == ""

    c.message("nothing loaded")
    assert not c.ax.axison, "the empty state must not show a bare pair of axes"

    # Task: the worker really runs off the GUI thread and reports both outcomes.
    seen: list[Any] = []
    t = Task(lambda report: (report("half"), 42)[1])
    t.progress.connect(seen.append)
    t.loaded.connect(seen.append)
    t.start()
    t.wait(5000)
    app.processEvents()
    assert "half" in seen and 42 in seen, seen
    bad = Task(lambda report: 1 / 0)
    errs: list[str] = []
    bad.failed.connect(errs.append)
    bad.start()
    bad.wait(5000)
    app.processEvents()
    assert errs and "ZeroDivisionError" in errs[0], errs
    print("mplcanvas.demo: ok")


if __name__ == "__main__":
    demo()
