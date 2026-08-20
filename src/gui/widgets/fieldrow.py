r"""One parameter, fully explained. The unit of layout in the parameter form.

WHY THIS EXISTS
  The form used to be a QFormLayout of code names - "scale", "degree", "h at notch" - with
  three tooltips between fifteen fields. A user who did not write the solver could not tell
  what to change or what a value meant. Every string that fixes that is already written and
  reviewed in model/help.py; this widget is the only thing that renders it, so the copy has
  exactly one consumer and the form has no strings of its own.

WHAT ONE ROW LOOKS LIKE
    Cell size at the crack   [ auto      mm ]  (?)  *
    --h-notch                Leave on auto. This sets the global time step, and refining
                             it is the most expensive mistake available here.
                             time step ~0.367 ns -> 163,680 steps (est.)
                             caution: 0.09 mm is 10x the steps for no accuracy gain

  label      plain language, from help.label, right-aligned in a fixed-width column so
             every row in every group lines up on the same edge.
  flag       the CLI flag, small and dim, under the label. It is what makes the dry-run
             command legible - you can read the command and find the field that produced it.
  hint       ALWAYS VISIBLE, --ink-soft, one line. A tooltip that has to be discovered is
             not an explanation; the whole failure this widget exists to fix was information
             that was technically present.
  (?)        toggles the fuller `detail` inline. Not a modal: a dialog you must dismiss to
             look at the field it describes is worse than no help at all.
  *          the orange "differs from the published baseline" dot, hover names the value.
  extra      two more optional lines the form fills in - the derived consequence of the
             current value, and any caution or guardrail finding attached to this field.

The widget is passed in rather than created here: the form owns the typed mapping between a
Qt widget and a RunConfig field, and this row owns nothing but presentation.
"""
from __future__ import annotations

import enum
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):     # direct run: make src/gui the import root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel, QSizePolicy,
                               QToolButton, QWidget)

# Both siblings are optional on purpose: the four workstreams land in any order and a missing
# module must degrade to a plainer row, never to a dead window.
try:
    from model import help as helptext
except ImportError:                                        # pragma: no cover - degraded path
    helptext = None                                        # type: ignore[assignment]

try:
    from theme import PALETTE
except ImportError:                                        # pragma: no cover - degraded path
    PALETTE = {"ink_soft": "#9AA0A6", "rule": "#2A2E33", "accent": "#FF7A1A",
               "accent_hi": "#FFA24D", "warn": "#E3B341", "fail": "#F0553A",
               "mono": '"Consolas", monospace'}

# One number, so labels line up across groups. QFormLayout aligns within a form only, and the
# form is now five groups deep; a fixed column is the cheapest way to get one shared edge.
LABEL_W = 122

_QMARK_CSS = ("QToolButton {{ color: {ink_soft}; background: transparent;"
              " border: 1px solid {rule}; border-radius: 8px; padding: 0px;"
              " min-width: 16px; max-width: 16px; min-height: 16px; max-height: 16px; }}"
              "QToolButton:hover {{ color: {accent_hi}; border-color: {accent_hi}; }}"
              "QToolButton:checked {{ color: {accent}; border-color: {accent}; }}")


class WrapLabel(QLabel):
    """A word-wrapped caption that takes its height from its WIDTH, not from a guess.

    Two Qt defaults fight word wrap inside a scroll area, and both had to go:

      * minimumSizeHint() of a wrapped label is the height it needs at its MINIMUM width, and
        the form sets that minimum to 1 px so a paragraph wraps instead of widening the
        column. Left alone the two together demand thousands of pixels of minimum height.
      * QSizePolicy.Minimum takes the minimum height from sizeHint() instead, which for a
        wrapped label is the same tall guess.

    Either one turns a QScrollArea into a page of blank space under the last field with a
    scrollbar that goes nowhere. Reporting no minimum and a shrinkable policy leaves the
    height to heightForWidth, which was the right number all along.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setMinimumWidth(1)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def minimumSizeHint(self) -> QSize:      # noqa: N802 - Qt name
        return QSize(0, 0)


def help_for(name: str):
    """model.help entry for a field, or a blank one. Never raises: a field with no copy is a
    gap in the copy, not a reason for the form to fail to construct."""
    if helptext is None:
        return SimpleNamespace(label=name.replace("_", " ").capitalize(), hint="", detail="",
                               flag="", unit="", choices={}, warn=False)
    return helptext.get(name)


def choice_key(value: object) -> str:
    """The model.help.choices key for a typed config value.

    Every discrete parameter in the contract keys on its own string form - Device.GPU is
    "gpu", degree 4 is "4" - so one function covers all five instead of a table per field.
    """
    return value.value if isinstance(value, enum.Enum) else str(value)


def shrink(box: QComboBox) -> None:
    """Let a combo be narrower than its longest item.

    A QComboBox sizes its MINIMUM to the widest string in it by default, so one long label -
    "Wavefield snapshots +20 deg", "Crack (4 mm x 1 mm slot)" - sets the minimum width of the
    whole form column, and at 1100 px the hints get clipped instead of wrapped. The popup
    still sizes to its contents, so nothing becomes unreadable.
    """
    box.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(10)


def build_combo(name: str, values: tuple, parent: QWidget | None = None) -> QComboBox:
    """A combo showing model.help's friendly labels, carrying the typed values as item data.

    The data is the real config value, so the form never parses a display string back into an
    enum - that round trip is where a UI silently starts disagreeing with the contract.
    """
    h = help_for(name)
    box = QComboBox(parent)
    shrink(box)
    for v in values:
        key = choice_key(v)
        label, hint = h.choices.get(key, (key, ""))
        box.addItem(label, v)
        box.setItemData(box.count() - 1, hint, Qt.ItemDataRole.ToolTipRole)
    return box


class FieldRow(QWidget):
    """Label, widget, flag, hint, "?" detail, dot, consequence, caution - for one parameter."""

    def __init__(self, name: str, widget: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.name = name
        self.widget = widget
        h = help_for(name)
        self._choices: dict[str, tuple[str, str]] = dict(h.choices)
        self._base_hint = h.hint
        self.warn = bool(h.warn)

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 2, 0, 6)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(1)
        grid.setColumnMinimumWidth(0, LABEL_W)
        grid.setColumnStretch(1, 1)
        # Reserved even when the "?" is hidden (a field whose copy is all in its choices), so
        # a row without one does not grow its input past every other row's right edge.
        grid.setColumnMinimumWidth(2, 16)

        self.label = QLabel(h.label or name)
        self.label.setFixedWidth(LABEL_W)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.label.setToolTip(h.detail or h.hint)

        # unit belongs to the widget, not to the label: "Record length" reads as a name,
        # "Record length (us)" reads as a name someone has already started editing.
        if h.unit:
            setter = getattr(widget, "setSuffix", None)
            if callable(setter):
                setter(h.unit)
            else:
                widget = self._with_unit(widget, h.unit)
        widget.setToolTip(h.detail or h.hint)

        self.qmark = QToolButton()
        self.qmark.setText("?")
        self.qmark.setCheckable(True)
        self.qmark.setAutoRaise(True)
        self.qmark.setCursor(Qt.CursorShape.PointingHandCursor)
        self.qmark.setToolTip("what this parameter does")
        self.qmark.setStyleSheet(_QMARK_CSS.format(**PALETTE))
        self.qmark.setFocusPolicy(Qt.FocusPolicy.NoFocus)   # keeps Tab on the fields only
        self.qmark.setVisible(bool(h.detail))

        self.dot = QLabel()
        self.dot.setObjectName("dot")
        self.dot.setProperty("on", "false")

        self.flag = QLabel(h.flag)
        self.flag.setFixedWidth(LABEL_W)
        self.flag.setWordWrap(True)     # "--no-notch / --notch-fill" does not fit on one line
        self.flag.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        self.flag.setStyleSheet("color: %s; font-size: 10px; font-family: %s;"
                                % (PALETTE["ink_soft"], PALETTE["mono"]))
        self.flag.setToolTip("the flag this field writes into the command")
        self.flag.setVisible(bool(h.flag))

        self.hint = self._caption(h.hint)
        self.hint.setVisible(bool(h.hint))
        self.consequence = self._caption("")
        self.consequence.setStyleSheet("color: %s; font-family: %s;"
                                       % (PALETTE["ink_soft"], PALETTE["mono"]))
        self.consequence.setVisible(False)
        self.alert = self._caption("")
        self.alert.setVisible(False)
        self.detail = self._caption(h.detail)
        self.detail.setContentsMargins(0, 4, 0, 4)
        self.detail.setVisible(False)
        self.qmark.toggled.connect(self.detail.setVisible)

        grid.addWidget(self.label, 0, 0)
        grid.addWidget(widget, 0, 1)
        grid.addWidget(self.qmark, 0, 2)
        grid.addWidget(self.dot, 0, 3)
        grid.addWidget(self.flag, 1, 0)
        grid.addWidget(self.hint, 1, 1, 1, 3)
        grid.addWidget(self.consequence, 2, 1, 1, 3)
        grid.addWidget(self.alert, 3, 1, 1, 3)
        grid.addWidget(self.detail, 4, 1, 1, 3)

        # Per-value hints: the copy for "3 - quick look" is different from the copy for
        # "5 - convergence check", so the line under the field has to follow the selection.
        if isinstance(self.widget, QComboBox) and self._choices:
            self.widget.currentIndexChanged.connect(self._sync_choice_hint)
            self._sync_choice_hint()

    # ---- construction helpers -------------------------------------------------------

    def _caption(self, text: str) -> QLabel:
        lab = WrapLabel(text)
        lab.setProperty("role", "caption")
        return lab

    def _with_unit(self, widget: QWidget, unit: str) -> QWidget:
        """Unit as a trailing label, for widgets with no setSuffix of their own."""
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(widget, 1)
        tail = QLabel(unit.strip())
        tail.setProperty("role", "caption")
        lay.addWidget(tail, 0)
        return box

    # ---- state ----------------------------------------------------------------------

    def _sync_choice_hint(self, *_a) -> None:
        key = choice_key(self.widget.currentData())
        hint = self._choices.get(key, ("", ""))[1]
        self.set_hint(hint or self._base_hint)

    def set_hint(self, text: str) -> None:
        self.hint.setText(text)
        self.hint.setVisible(bool(text))

    def set_consequence(self, text: str) -> None:
        """The derived effect of the CURRENT value, next to the value. The side panel is the
        summary; this is the answer to "what did I just do"."""
        self.consequence.setText(text)
        self.consequence.setVisible(bool(text))

    def consequence_text(self) -> str:
        return self.consequence.text()

    def set_alert(self, text: str, state: str = "warn") -> None:
        """A caution or a guardrail finding, in --warn or --fail. Empty text clears it."""
        self.alert.setText(text)
        self.alert.setProperty("state", state)
        self.alert.setVisible(bool(text))
        # Dynamic properties only reach the painted widget after a restyle.
        self.alert.style().unpolish(self.alert)
        self.alert.style().polish(self.alert)

    def alert_text(self) -> str:
        return self.alert.text()

    def set_dot(self, on: bool, tip: str = "") -> None:
        self.dot.setProperty("on", "true" if on else "false")
        self.dot.setToolTip(tip or "differs from the published baseline")
        self.dot.style().unpolish(self.dot)
        self.dot.style().polish(self.dot)

    def dot_on(self) -> bool:
        return self.dot.property("on") == "true"


def demo() -> None:
    """Self-check: the copy reaches the screen, and it is on screen rather than in a tooltip."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QDoubleSpinBox
    app = QApplication.instance() or QApplication([])

    spin = QDoubleSpinBox()
    row = FieldRow("h_notch", spin)
    row.show()
    app.processEvents()

    h = help_for("h_notch")
    assert row.label.text() == h.label and "_" not in row.label.text()
    assert row.flag.text() == "--h-notch" and row.flag.isVisible()
    assert row.hint.text() == h.hint and row.hint.isVisible(), "the hint must not be hidden"
    assert spin.suffix() == " mm", spin.suffix()
    assert row.warn is True, "h_notch is a caution field in help.py"

    # The "?" reveals the long copy inline. It must start hidden, or every row is a wall.
    assert not row.detail.isVisible() and row.detail.text() == h.detail
    row.qmark.click()
    app.processEvents()
    assert row.detail.isVisible(), "the ? must reveal the detail"

    row.set_consequence("time step ~0.367 ns")
    assert row.consequence.isVisible() and "0.367" in row.consequence_text()
    row.set_alert("BLOCK: nope", "fail")
    assert row.alert.isVisible() and row.alert.property("state") == "fail"
    row.set_alert("")
    assert not row.alert.isVisible()
    row.set_dot(True, "published: auto")
    assert row.dot_on() and "auto" in row.dot.toolTip()

    # A discrete field shows friendly labels, carries typed data, and swaps hint per value.
    combo = build_combo("degree", (3, 4, 5))
    crow = FieldRow("degree", combo)
    crow.show()
    app.processEvents()
    assert combo.count() == 3 and combo.itemText(1).startswith("4 - published")
    combo.setCurrentIndex(0)
    assert combo.currentData() == 3
    assert crow.hint.text() == help_for("degree").choices["3"][1], crow.hint.text()
    combo.setCurrentIndex(2)
    assert crow.hint.text() == help_for("degree").choices["5"][1]

    # A missing help entry must produce a plain row, not an exception.
    plain = FieldRow("no_such_parameter", QDoubleSpinBox())
    assert plain.label.text() and not plain.flag.isVisible()
    print("fieldrow.demo: ok")


if __name__ == "__main__":
    demo()
