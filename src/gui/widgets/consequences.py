r"""What the current parameters will cost: nodes per wavelength, dt, steps, runtime, disk.

Every number here is an ESTIMATE and is labelled one, per README non-negotiable #1. The solver
decides dt; this panel only says roughly what to expect before spending GPU minutes on it.

The arithmetic lives in model/derived.py (W2). This widget only renders what that module
returns - a tuple of derived.Quantity(name, value, unit, is_estimate, note) - and shows
em-dashes when the module is absent, so the shell is usable before it lands. Formatting by
UNIT rather than by row name is what keeps that contract loose: W2 can add a quantity and it
appears here with no change on this side.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):     # direct run: make src/gui the import root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy, QWidget

from model.spec import RunConfig

DASH = "\u2014"     # em-dash: "not computed", which is not the same as zero

# Shown before model.derived answers, so the panel has a shape from the first paint.
PLACEHOLDER_ROWS = ("nodes per wavelength", "time step", "time steps", "solve time", "disk")


def fmt(value: float, unit: str) -> str:
    """One number, in the unit a human would read it in. Magnitude decides, not the caller."""
    if unit == "s":
        if value < 1e-6:
            return "%.1f ns" % (value * 1e9)
        if value < 1e-3:
            return "%.3f us" % (value * 1e6)
        if value < 90.0:
            return "%.2f s" % value
        if value < 3600.0:
            return "%.1f min" % (value / 60.0)
        return "%.2f h" % (value / 3600.0)
    if unit == "B":
        for div, suf in ((1e9, "GB"), (1e6, "MB"), (1e3, "kB")):
            if value >= div:
                return "%.2f %s" % (value / div, suf)
        return "%.0f B" % value
    if unit in ("steps", "cells", "dof"):
        return "{:,}".format(int(round(value)))
    if unit == "nodes":
        return "%.2f" % value
    return ("%.3g %s" % (value, unit)).strip()


def _estimates(cfg: RunConfig):
    """model.derived's quantities for this config, or () if that module is not available.

    Scenario facts come from model/scenario.py's cache, which never touches Docker, so this is
    cheap enough to call on every keystroke.
    """
    try:
        from model import derived, scenario
    except ImportError:
        return ()
    try:
        return tuple(derived.all_estimates(cfg, scenario.load()))
    except Exception as exc:            # a half-built module must not kill the UI
        return (exc,)


class Consequences(QFrame):
    """Compact read-only table. Values monospace and right-aligned so runs compare by eye."""

    def __init__(self, config: RunConfig | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("role", "card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(10, 8, 10, 8)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(3)

        head = QLabel("Consequences")
        head.setObjectName("title")
        self._grid.addWidget(head, 0, 0, 1, 3)
        self._grid.setColumnStretch(1, 1)

        self._rows: dict[str, tuple[QLabel, QLabel]] = {}   # name -> (value, est. marker)
        self._note = QLabel("")
        self._note.setProperty("role", "caption")
        self._note.setWordWrap(True)
        self._note.setMinimumWidth(1)

        self._build_rows(PLACEHOLDER_ROWS)
        self.set_config(config or RunConfig())

    def _build_rows(self, names: tuple[str, ...]) -> None:
        """(Re)build the row set. Only runs when the set of quantities changes."""
        while self._grid.count() > 1:                 # keep the header, drop the rest
            item = self._grid.takeAt(1)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self._rows = {}
        for r, name in enumerate(names, start=1):
            cap = QLabel(name)
            cap.setProperty("role", "caption")
            val = QLabel(DASH)
            val.setProperty("role", "num")
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            est = QLabel("")
            est.setProperty("role", "caption")
            self._grid.addWidget(cap, r, 0)
            self._grid.addWidget(val, r, 1)
            self._grid.addWidget(est, r, 2)
            self._rows[name] = (val, est)
        self._grid.addWidget(self._note, len(names) + 1, 0, 1, 3)

    def set_config(self, cfg: RunConfig) -> None:
        qs = _estimates(cfg)
        if len(qs) == 1 and isinstance(qs[0], Exception):
            self._note.setText("model.derived failed: %s: %s"
                               % (type(qs[0]).__name__, qs[0]))
            for val, est in self._rows.values():
                val.setText(DASH)
                est.setText("")
            return
        if not qs:
            self._note.setText("model.derived not available - estimates unavailable")
            return

        names = tuple(q.name for q in qs)
        if names != tuple(self._rows):
            self._build_rows(names)
        notes = []
        for q in qs:
            val, est = self._rows[q.name]
            val.setText(fmt(q.value, q.unit))
            est.setText("est." if q.is_estimate else "")
            val.setToolTip(q.note or "")
            if q.note:
                notes.append("%s: %s" % (q.name, q.note))
        # The binding-material note is the one that changes how you read the first row, so it
        # is on screen rather than only in a tooltip.
        self._note.setText(notes[0] if notes else "")
        self._note.setToolTip("\n".join(notes))


def demo() -> None:
    """Self-check: formatting by unit, and a render whether or not model.derived exists."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    assert fmt(0.3666e-9, "s") == "0.4 ns"
    assert fmt(462.0, "s") == "7.7 min" and fmt(8640.0, "s") == "2.40 h"
    assert fmt(970e6, "B") == "970.00 MB" and fmt(2.1e9, "B") == "2.10 GB"
    assert fmt(163680, "steps") == "163,680"
    assert fmt(3.4567, "nodes") == "3.46"

    w = Consequences()
    w.show()
    app.processEvents()
    qs = _estimates(RunConfig())
    if qs and not isinstance(qs[0], Exception):
        # With W2 present every row must carry a real number - a silent em-dash would read as
        # "cheap" on a run that is anything but.
        assert tuple(w._rows) == tuple(q.name for q in qs)
        assert all(v.text() != DASH for v, _ in w._rows.values()), "row left blank"
    else:
        assert all(v.text() == DASH for v, _ in w._rows.values()), "invented a value"
    print("consequences.demo: ok (%d quantities)" % len(qs))


if __name__ == "__main__":
    demo()
