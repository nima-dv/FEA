r"""DV-FEA entry point and window shell.

The shell knows the five sections and nothing about what is inside them. Views are built
LAZILY, on first selection, inside a try/except ImportError: four people are writing this app
at the same time, and a missing views/queue.py must cost a placeholder panel, not a dead app.
That is also why nothing here imports another stream's module at the top of the file.

Run: .venv-gui/Scripts/python.exe src/gui/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):     # direct run: make src/gui the import root
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QStackedWidget, QVBoxLayout,
                               QWidget)

import theme

TITLE = "DV-FEA  Crack Simulation"

# (rail label, module, preferred class names). The module may not exist yet; see _build_view.
VIEW_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Simulate", "views.simulate", ("SimulateView",)),
    ("Queue", "views.queue", ("QueueView", "Queue")),
    ("Results", "views.results", ("ResultsView", "Results")),
    ("Export", "views.export", ("ExportView", "Export")),
    ("Settings", "views.settings", ("SettingsView", "Settings")),
)

# Status bar fields, in order: (key in the probe dict, caption).
STATUS_FIELDS: tuple[tuple[str, str], ...] = (
    ("daemon", "docker"),
    ("gpu", "gpu"),
    ("images", "images"),
    ("disk_free", "free disk"),
    ("running", "running"),
)


def _placeholder(label: str, module: str, why: str) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title = QLabel(label + " - not built yet")
    title.setObjectName("title")
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    sub = QLabel("%s\n%s" % (module, why))
    sub.setProperty("role", "caption")
    sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(title)
    lay.addWidget(sub)
    return w


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(TITLE)
        self.setMinimumSize(1100, 720)
        self._views: dict[str, QWidget] = {}
        self._index: dict[str, int] = {}

        central = QWidget()
        row = QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        rail = QFrame()
        rail.setObjectName("rail")
        rail.setFixedWidth(150)
        rl = QVBoxLayout(rail)
        rl.setContentsMargins(0, 8, 0, 8)
        brand = QLabel("  DV-FEA")
        brand.setObjectName("title")
        rl.addWidget(brand)
        self.rail = QListWidget()
        self.rail.setObjectName("railList")
        rl.addWidget(self.rail, 1)
        row.addWidget(rail, 0)

        self.stack = QStackedWidget()
        row.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        for label, module, _classes in VIEW_SPECS:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, label)
            item.setToolTip(module)
            self.rail.addItem(item)
            # A placeholder holds every slot, so the stack index of a section never moves when
            # its real view arrives.
            self._index[label] = self.stack.addWidget(
                _placeholder(label, module, "select it to load"))

        self.rail.currentRowChanged.connect(self._on_rail)
        self._build_status_bar()
        self._restore_geometry()
        self.rail.setCurrentRow(0)          # loads Simulate

    # ---- views ---------------------------------------------------------------------

    def register_view(self, name: str, widget: QWidget) -> None:
        """Drop a widget into a section, replacing whatever is there.

        The hook other workstreams use: main.py never needs editing to adopt a real view, and
        a view can be swapped at runtime (a test fixture, a stub) without touching the shell.
        """
        idx = self._index.get(name)
        if idx is None:
            self.rail.addItem(QListWidgetItem(name))
            self._index[name] = self.stack.addWidget(widget)
        else:
            old = self.stack.widget(idx)
            self.stack.insertWidget(idx, widget)
            self.stack.removeWidget(old)
            old.deleteLater()
        self._views[name] = widget
        if self.rail.currentItem() is not None and self.rail.currentItem().text() == name:
            self.stack.setCurrentIndex(self._index[name])

    def _build_view(self, label: str, module: str, classes: tuple[str, ...]) -> None:
        """Import and instantiate one section's view, or leave a placeholder saying why not."""
        from PySide6.QtWidgets import QWidget as _QW
        try:
            mod = __import__(module, fromlist=["_"])
        except ImportError as exc:
            self.register_view(label, _placeholder(label, module, "ImportError: %s" % exc))
            return
        cls = next((getattr(mod, n) for n in classes if hasattr(mod, n)), None)
        if cls is None:
            # Class names are another stream's choice, so fall back to the first QWidget the
            # module itself defines rather than dictating a name across workstreams.
            cls = next((o for o in vars(mod).values()
                        if isinstance(o, type) and issubclass(o, _QW)
                        and o.__module__ == mod.__name__), None)
        if cls is None:
            self.register_view(label, _placeholder(label, module, "no QWidget subclass found"))
            return
        try:
            widget = cls()
        except Exception as exc:            # a view that cannot construct must not kill the app
            self.register_view(label, _placeholder(label, module,
                                                   "%s: %s" % (type(exc).__name__, exc)))
            return
        self.register_view(label, widget)
        self._wire(widget)

    def _wire(self, widget: QWidget) -> None:
        """Connect a freshly built view to the runner, if both ends exist.

        The form emits a RunConfig; the shell decides who executes it. Keeping that wire here
        is what lets views/simulate.py be tested with no Docker on the machine.
        """
        for sig_name, method in (("run_requested", "submit"), ("queue_requested", "enqueue")):
            sig = getattr(widget, sig_name, None)
            if sig is None:
                continue
            runner = self._runner()
            fn = getattr(runner, method, None) if runner is not None else None
            if fn is not None:
                sig.connect(fn)
            else:
                sig.connect(lambda cfg, stages, m=method:
                            self.statusBar().showMessage(
                                "core.runner.%s not available - %s (%d stages) not submitted"
                                % (m, cfg.tag(), len(stages)), 6000))

    def _runner(self):
        """The single shared runner instance, or None while core/runner.py is unwritten."""
        if not hasattr(self, "_runner_obj"):
            self._runner_obj = None
            try:
                from core import runner as runner_mod
            except ImportError:
                return None
            for name in ("Runner", "JobRunner", "Queue"):
                cls = getattr(runner_mod, name, None)
                if isinstance(cls, type):
                    try:
                        self._runner_obj = cls()
                    except Exception:
                        self._runner_obj = None
                    break
        return self._runner_obj

    def _on_rail(self, row: int) -> None:
        if row < 0:
            return
        label, module, classes = VIEW_SPECS[row] if row < len(VIEW_SPECS) else (None, None, None)
        if label is not None and label not in self._views:
            self._build_view(label, module, classes)
        self.stack.setCurrentIndex(self._index[self.rail.item(row).text()])

    # ---- status bar ----------------------------------------------------------------

    def _build_status_bar(self) -> None:
        self._status: dict[str, QLabel] = {}
        bar = self.statusBar()
        for key, caption in STATUS_FIELDS:
            cap = QLabel(caption)
            cap.setProperty("role", "caption")
            val = QLabel("unknown")
            val.setProperty("state", "idle")
            bar.addPermanentWidget(cap)
            bar.addPermanentWidget(val)
            self._status[key] = val
        self._refresh_status()
        # 5 s: the daemon, the disk and the job count all move while the app is open, and a
        # probe is a couple of cheap subprocess calls.
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(5000)

    def _refresh_status(self) -> None:
        probe: dict = {}
        try:
            from core import docker as docker_mod
            fn = getattr(docker_mod, "probe", None)
            if callable(fn):
                probe = dict(fn() or {})
        except ImportError:
            probe = {}
        except Exception as exc:
            probe = {"daemon": "probe failed: %s" % exc}
        for key, _caption in STATUS_FIELDS:
            v = probe.get(key)
            lbl = self._status[key]
            lbl.setText("unknown" if v is None else str(v))
            state = "idle"
            if v is not None:
                bad = str(v).lower() in ("false", "down", "no", "none", "0 b", "unavailable")
                state = "fail" if bad else "ok"
            lbl.setProperty("state", state)
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

    # ---- geometry ------------------------------------------------------------------

    def _settings(self) -> QSettings:
        return QSettings("DarkVision", "DV-FEA")

    def _restore_geometry(self) -> None:
        g = self._settings().value("geometry")
        if g is not None:
            self.restoreGeometry(g)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self._settings().setValue("geometry", self.saveGeometry())
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("DV-FEA")
    theme.apply(app)
    win = MainWindow()
    win.show()
    return app.exec()


def demo() -> None:
    """Self-check: the shell opens, every section is selectable, and a view can be swapped in."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    theme.apply(app)

    win = MainWindow()
    win.show()
    app.processEvents()

    assert win.rail.count() == len(VIEW_SPECS)
    assert win.windowTitle() == TITLE
    # Every section must survive selection with the other streams' modules missing.
    for row in range(win.rail.count()):
        win.rail.setCurrentRow(row)
        app.processEvents()
        assert win.stack.currentWidget() is not None, row
    assert "Simulate" in win._views, "the form is this stream's own view; it must load"

    # register_view is the parallel-work hook: it must replace, not append.
    n = win.stack.count()
    probe = QLabel("stand-in")
    win.register_view("Queue", probe)
    assert win.stack.count() == n and win._views["Queue"] is probe

    win.rail.setCurrentRow(0)
    app.processEvents()
    assert len(win._status) == len(STATUS_FIELDS)
    assert all(lbl.text() for lbl in win._status.values()), "status fields must never be blank"
    print("main.demo: ok, %d sections" % win.rail.count())


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        sys.exit(main())
