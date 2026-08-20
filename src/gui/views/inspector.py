r"""The Inspect view: three interactive tabs over data that is already on disk.

WHAT THIS IS NOT
It is not a figure factory. Published figures are rendered by the backend, from the container,
with the argv recorded - that is what makes them reproducible. This view exists so that a
number in a published panel can be interrogated: what is the dB at that pixel, what does the
mesh actually do at the ID, where in time does the shear wave arrive. Every canvas carries a
permanent stamp saying so (widgets/mplcanvas.STAMP), and there is one in the header too,
because the failure mode being guarded against is a screenshot pasted into a report.

NAMED inspector.py, NOT inspect.py
A module called `inspect.py` in this directory shadows the standard library's `inspect` for any
script run directly from inside views/ - Python puts the script's own directory first on
sys.path, and dataclasses, typing and PySide6 all import `inspect` internally. The failure lands
several frames deep in someone else's import and names no cause. Hence the -or.

DISCOVERY IS BORROWED, NOT REBUILT
`views/results.py` already resolves the results root and discovers runs from two sources
(GUI manifests and the published record). This module imports both rather than globbing again,
so the Inspect run list and the Results gallery can never disagree about what exists.

THE SHARED SELECTOR ONLY POINTS AT WHAT A RUN ACTUALLY HAS
Picking a run aims the wavefield and image tabs at that run's own files. It does NOT touch the
mesh tab: a published entry names no mesh, and silently loading an unrelated one would be worse
than leaving the mesh picker where the user put it. A run with no snapshot leaves that tab
alone too - snapshots are optional and usually absent.

TABS ARE BUILT EAGERLY, THE BIG LOAD IS DEFERRED
All three widgets are cheap to construct; only the wavefield record is expensive (~0.5-2 GB).
That one loads when its tab is first opened, so selecting Inspect in the rail stays instant.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

_GUI_ROOT = Path(__file__).resolve().parents[1]
if str(_GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_GUI_ROOT))

from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QTabWidget,     # noqa: E402
                               QVBoxLayout, QWidget)

from views.results import discover_runs, presentation_root, results_root                          # noqa: E402
from widgets.imageprobe import ImageProbe                                     # noqa: E402
from widgets.mplcanvas import INK_SOFT, STAMP                                 # noqa: E402
from widgets.meshview import MeshView                                         # noqa: E402
from widgets.wavefieldview import WavefieldView                               # noqa: E402

# compare_p20deg_widedomain_nooverlay.png  ->  images_20_widedomain_nooverlay.npz
_FIG_RE = re.compile(r"^compare_(p|m)(\d+)deg(.*)\.png$")


def images_for(entry: Any, root: Path) -> Path | None:
    """The images_*.npz behind one run's comparison figure, if it is on disk.

    A manifest may list the npz among its outputs; the published record does not, but its
    figure name determines it, because compare_images.py writes the pair together. Derived
    from the name rather than guessed at, and only returned when the file exists.
    """
    for p in getattr(entry, "npz", []) or []:
        if p.name.startswith("images_"):
            return p
    for fig in getattr(entry, "figures", []) or []:
        m = _FIG_RE.match(fig.name)
        if not m:
            continue
        sign, deg, rest = m.group(1), m.group(2), m.group(3)
        cand = root / "compare" / ("images_%s%s%s.npz"
                                  % ("-" if sign == "m" else "", deg, rest))
        if cand.exists():
            return cand
    return None


def wavefield_for(entry: Any) -> Path | None:
    """The wavefield snapshot this run recorded, if any. Usually there is none."""
    for p in getattr(entry, "npz", []) or []:
        if p.name.startswith("wavefield"):
            return p
    return None


def _select(combo: QComboBox, path: Path | None) -> bool:
    """Point a picker at `path` if it is in its list. Returns whether it moved."""
    if path is None:
        return False
    for i in range(combo.count()):
        if combo.itemData(i) == path:
            if combo.currentIndex() != i:
                combo.setCurrentIndex(i)
            return True
    return False


class InspectView(QWidget):
    """Mesh / Wavefield / Image tabs, with one run selector over the top."""

    def __init__(self, root: Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Image caches and animations are published data and live in presentation/.
        self.root = root or presentation_root()
        self.runs = discover_runs(self.root)

        self.run_pick = QComboBox()
        self.run_pick.addItem("(all files on disk)", None)
        for e in self.runs:
            self.run_pick.addItem(e.title, e)
        self.note = QLabel(STAMP + " - published figures are rendered by the backend")
        self.note.setStyleSheet("color:%s;" % INK_SOFT)

        self.mesh = MeshView(self.root)
        self.wavefield = WavefieldView(self.root, autoload=False)
        self.probe = ImageProbe(self.root)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.mesh, "Mesh")
        self.tabs.addTab(self.wavefield, "Wavefield")
        self.tabs.addTab(self.probe, "Image")

        head = QHBoxLayout()
        head.setContentsMargins(10, 8, 10, 0)
        head.addWidget(QLabel("run"))
        head.addWidget(self.run_pick, 2)
        head.addStretch(1)
        head.addWidget(self.note)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(head)
        lay.addWidget(self.tabs, 1)

        self.run_pick.currentIndexChanged.connect(self._on_run)
        self.tabs.currentChanged.connect(self._on_tab)

    # ---- wiring --------------------------------------------------------------------

    def _on_run(self, _i: int) -> None:
        entry = self.run_pick.currentData()
        if entry is None:
            return
        moved = []
        if _select(self.probe.picker, images_for(entry, self.root)):
            moved.append("image")
        wf = wavefield_for(entry)
        if _select(self.wavefield.picker, wf):
            moved.append("wavefield")
            # Only load it if that tab is the one being looked at; otherwise the deferral in
            # __init__ would be undone by the act of choosing a run.
            if self.tabs.currentWidget() is self.wavefield:
                self.wavefield.load_current()
        self.probe.status.setToolTip("run selector aimed: %s" % (", ".join(moved) or "nothing"))

    def _on_tab(self, _i: int) -> None:
        w = self.tabs.currentWidget()
        wf = self.wavefield
        if w is wf and wf.record is None and wf._task is None:
            wf.load_current()

    def set_run(self, title: str) -> bool:
        """Select a run by its title - the hook a test or another view can use."""
        for i in range(self.run_pick.count()):
            if self.run_pick.itemText(i) == title:
                self.run_pick.setCurrentIndex(i)
                return True
        return False


def demo() -> None:
    """Self-check: three tabs, real runs in the selector, and the selector really aims them."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    import theme
    app = QApplication.instance() or QApplication([])
    theme.apply(app)

    root = results_root()
    v = InspectView(root)
    v.resize(1150, 760)
    v.show()
    app.processEvents()

    assert v.tabs.count() == 3 and [v.tabs.tabText(i) for i in range(3)] == \
        ["Mesh", "Wavefield", "Image"]
    assert STAMP in v.note.text(), "the header must say this is not the published figure"
    assert v.runs, "no runs discovered - views/results.discover_runs found nothing"

    # The big load must NOT have happened just because the view was built.
    assert v.wavefield.record is None and v.wavefield._task is None

    # Name mapping: every published comparison figure that has an npz must resolve to it.
    hits = [(e.title, images_for(e, root)) for e in v.runs]
    assert any(p is not None for _t, p in hits), hits[:4]
    for title, p in hits:
        if p is not None:
            assert p.exists() and p.name.startswith("images_"), (title, p)

    # Aim the run selector at a run that has an image set, and check the probe followed.
    idx = next(i for i in range(1, v.run_pick.count())
               if images_for(v.run_pick.itemData(i), root) is not None)
    want = images_for(v.run_pick.itemData(idx), root)
    v.run_pick.setCurrentIndex(idx)
    app.processEvents()
    assert v.probe.picker.currentData() == want, (v.probe.picker.currentData(), want)
    assert v.probe._im is not None and v.probe._im.get_clim() == (-40.0, 0.0)

    # Opening the wavefield tab is what starts the load (if there is anything to load).
    v.tabs.setCurrentIndex(1)
    app.processEvents()
    if v.wavefield.picker.count():
        assert v.wavefield._task is not None, "opening the tab did not start the load"
        v.wavefield._task.wait(600000)
        app.processEvents()
        assert v.wavefield.record is not None, v.wavefield.status.text()
        assert v.wavefield.slider.isEnabled()
    else:
        assert not v.wavefield.slider.isEnabled()

    v.mesh._task.wait(60000)
    app.processEvents()
    assert v.mesh.mesh is not None, v.mesh.status.text()
    print("inspector.demo: ok, %d runs, tabs %s, mesh %s, images %s, snapshot %s"
          % (len(v.runs), [v.tabs.tabText(i) for i in range(3)],
             v.mesh.mesh.path.name, v.probe.picker.currentText(),
             v.wavefield.record.path.name if v.wavefield.record else "none"))


if __name__ == "__main__":
    demo()
