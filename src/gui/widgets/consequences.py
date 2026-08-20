r"""What the current parameters will cost: nodes per wavelength, dt, steps, runtime, disk.

Every number here is an ESTIMATE and is labelled one, per README non-negotiable #1. The
solver decides dt; this panel only says roughly what to expect before you spend GPU minutes
on it. The arithmetic itself lives in model/derived.py (W2) - this widget renders it and
shows em-dashes when that module is not present, so the shell is usable without it.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):     # direct run: make src/gui the import root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy, QWidget

from model.spec import RunConfig

DASH = "\u2014"      # em-dash: "not computed", which is not the same as zero


def _fmt_dt(v: float) -> str:
    return "%.1f ns" % (v * 1e9) if v < 1e-6 else "%.3f us" % (v * 1e6)


def _fmt_runtime(v: float) -> str:
    return "%.1f min" % (v / 60.0) if v < 3600 else "%.2f h" % (v / 3600.0)


def _fmt_disk(v: float) -> str:
    return "%.0f MB" % v if v < 1024 else "%.2f GB" % (v / 1024.0)


# (key in the derived mapping, row label, formatter). Order is the order on screen.
ROWS: tuple[tuple[str, str, object], ...] = (
    ("nodes_per_wavelength", "nodes / wavelength", lambda v: "%.2f" % v),
    ("dt", "time step dt", _fmt_dt),
    ("steps", "steps", lambda v: "{:,}".format(int(v))),
    ("runtime_s", "runtime", _fmt_runtime),
    ("disk_mb", "disk", _fmt_disk),
)


def _derived_for(cfg: RunConfig) -> dict:
    """Ask model/derived.py for the numbers. Absent module -> empty, not a crash.

    Expected contract: a callable taking a RunConfig and returning a mapping with the keys in
    ROWS, plus an optional 'binding' naming the material that sets nodes/wavelength (steel
    shear at 4 MHz is the short wavelength that binds, and naming it stops anyone reading the
    number against the P wave).
    """
    try:
        from model import derived
    except ImportError:
        return {}
    for name in ("consequences", "derive", "estimate"):
        fn = getattr(derived, name, None)
        if callable(fn):
            try:
                return dict(fn(cfg))
            except Exception as exc:              # a half-built module must not kill the UI
                return {"_error": "%s: %s" % (type(exc).__name__, exc)}
    return {}


class Consequences(QFrame):
    """Compact read-only table. Values monospace and right-aligned so runs compare by eye."""

    def __init__(self, config: RunConfig | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("role", "card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        grid = QGridLayout(self)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)

        head = QLabel("Consequences")
        head.setObjectName("title")
        grid.addWidget(head, 0, 0, 1, 3)

        self._values: dict[str, QLabel] = {}
        for r, (key, label, _) in enumerate(ROWS, start=1):
            cap = QLabel(label)
            cap.setProperty("role", "caption")
            val = QLabel(DASH)
            val.setProperty("role", "num")
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            est = QLabel("est.")
            est.setProperty("role", "caption")
            grid.addWidget(cap, r, 0)
            grid.addWidget(val, r, 1)
            grid.addWidget(est, r, 2)
            self._values[key] = val
        grid.setColumnStretch(1, 1)

        self._note = QLabel("")
        self._note.setProperty("role", "caption")
        self._note.setWordWrap(True)
        grid.addWidget(self._note, len(ROWS) + 1, 0, 1, 3)

        self.set_config(config or RunConfig())

    def set_config(self, cfg: RunConfig) -> None:
        d = _derived_for(cfg)
        for key, _, fmt in ROWS:
            v = d.get(key)
            self._values[key].setText(DASH if v is None else fmt(v))
        if "_error" in d:
            self._note.setText("model.derived failed: " + d["_error"])
        elif not d:
            self._note.setText("model.derived not available - estimates unavailable")
        else:
            binding = d.get("binding", "steel shear at 4 MHz")
            self._note.setText("nodes / wavelength binds on " + str(binding)
                               + ". Estimates, until the solver prints its own numbers.")


def demo() -> None:
    """Self-check: renders with model.derived missing, and formats what it is given."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    w = Consequences()
    w.show()
    app.processEvents()
    # With no W2 module, every value must be an em-dash rather than a fabricated number.
    have_derived = bool(_derived_for(RunConfig()))
    if not have_derived:
        assert all(lbl.text() == DASH for lbl in w._values.values()), "invented a value"
        assert "not available" in w._note.text()

    assert _fmt_dt(4.2e-9) == "4.2 ns" and _fmt_dt(1.5e-6) == "1.500 us"
    assert _fmt_runtime(462.0) == "7.7 min" and _fmt_runtime(8640.0) == "2.40 h"
    assert _fmt_disk(970.0) == "970 MB" and _fmt_disk(2048.0) == "2.00 GB"
    print("consequences.demo: ok (model.derived present: %s)" % have_derived)


if __name__ == "__main__":
    demo()
