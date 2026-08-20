r"""One queued job, as a card: stage, parameters, progress, ETA, state, cancel, log tail.

Fed PLAIN DATA (`JobView`), never a live Runner. Two reasons: the card is then testable with
core/runner.py absent, and a widget that reaches into the runner ends up parsing log lines in
the paint path, which is how a UI thread gets blocked by a 2.4 h solve.

THE LOG TAIL SHOWS EVERYTHING, VERBATIM
core/logparse.py returns None for any line it does not recognise. That is deliberate - the
solver's tracebacks, CUDA warnings and "out of memory" lines are exactly the ones no progress
regex matches. If the card only showed lines the parser understood, a failed run would look
like a run that simply stopped. So `append_log` filters nothing and reformats nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Mapping

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QProgressBar, QPlainTextEdit,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

# Only the tokens this widget has to decide itself; everything else comes from theme/dark.qss.
INK = "#E8E8EA"
INK_SOFT = "#9AA0A6"
RULE = "#2A2E33"
SURFACE = "#16181B"
ACCENT = "#FF7A1A"
STATE_COLOR = {
    "queued": "#6E7681",      # --idle
    "running": "#FF7A1A",     # --accent
    "succeeded": "#3FB950",   # --ok
    "failed": "#F0553A",      # --fail
    "cancelled": "#E3B341",   # --warn: cancelled is a choice, not a fault
}
TERMINAL = ("succeeded", "failed", "cancelled")


@dataclass
class JobView:
    """A snapshot of one job. Whatever the runner's internals are, this is the wire format."""
    job_id: str = ""
    stage: str = "forward"                  # mesh | forward | image | figures
    state: str = "queued"                   # queued | running | succeeded | failed | cancelled
    summary: str = ""                       # one-line parameter summary
    percent: float | None = None            # 0-100; None while running = indeterminate
    step: int | None = None
    total_steps: int | None = None
    ms_per_step: float | None = None
    eta_s: float | None = None
    elapsed_s: float | None = None
    log: list[str] = field(default_factory=list)

    @classmethod
    def from_obj(cls, obj: Any) -> "JobView":
        """Accept a dict, another dataclass, or anything with matching attributes.

        The runner lands in a different workstream, so its job object is not knowable here.
        Pulling by field name keeps this card usable whichever shape arrives - and anything
        missing simply stays at its default rather than raising in a signal handler.
        """
        if isinstance(obj, JobView):
            return obj
        names = {f.name for f in fields(cls)}
        if isinstance(obj, Mapping):
            got = {k: v for k, v in obj.items() if k in names}
        else:
            got = {n: getattr(obj, n) for n in names if hasattr(obj, n)}
        if "stage" in got and hasattr(got["stage"], "value"):   # model.spec.Stage enum
            got["stage"] = got["stage"].value
        if "state" in got and hasattr(got["state"], "value"):
            got["state"] = got["state"].value
        return cls(**got)


def fmt_dur(s: float | None) -> str:
    """Human duration. Minutes for a solve, hours for a CPU solve, dashes for unknown."""
    if s is None or s < 0:
        return "--"
    if s < 90:
        return f"{s:.0f} s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"


def _chip(text: str, fg: str, bg: str) -> QLabel:
    lab = QLabel(text)
    lab.setStyleSheet(f"color:{fg}; background:{bg}; border-radius:3px; padding:1px 7px;")
    lab.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return lab


def _mono(text: str = "") -> QLabel:
    lab = QLabel(text)
    f = QFont("Consolas")
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPointSizeF(max(7.5, QFont().pointSizeF() - 0.5))
    lab.setFont(f)
    lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    lab.setStyleSheet(f"color:{INK_SOFT};")
    return lab


class JobCard(QFrame):
    """Displays one JobView. Emits cancel_requested(job_id); it never cancels anything."""

    cancel_requested = Signal(str)
    LOG_LINES = 800          # bounded: a snapshotting solve prints tens of thousands of lines

    def __init__(self, view: JobView | Mapping | Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.view = JobView.from_obj(view)
        self.setObjectName("jobCard")
        self.setStyleSheet(
            f"#jobCard {{ background:{SURFACE}; border:1px solid {RULE}; border-radius:6px; }}")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(8)
        self._stage = _chip("", ACCENT, "rgba(255,122,26,0.12)")     # --accent-wash
        self._summary = QLabel()
        self._summary.setStyleSheet(f"color:{INK};")
        self._summary.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._state = _chip("", INK, "transparent")
        self._cancel = QPushButton("Cancel")
        self._cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel.clicked.connect(lambda: self.cancel_requested.emit(self.view.job_id))
        top.addWidget(self._stage)
        top.addWidget(self._summary, 1)
        top.addWidget(self._state)
        top.addWidget(self._cancel)
        outer.addLayout(top)

        self._bar = QProgressBar()
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._bar.setStyleSheet(
            f"QProgressBar {{ background:{RULE}; border:none; border-radius:3px; }}"
            f"QProgressBar::chunk {{ background:{ACCENT}; border-radius:3px; }}")
        outer.addWidget(self._bar)

        mid = QHBoxLayout()
        mid.setSpacing(12)
        self._log_toggle = QPushButton("log")
        self._log_toggle.setCheckable(True)
        self._log_toggle.setFlat(True)
        self._log_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._log_toggle.toggled.connect(self._on_toggle)
        self._numbers = _mono()
        mid.addWidget(self._log_toggle)
        mid.addStretch(1)
        mid.addWidget(self._numbers)
        outer.addLayout(mid)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(self.LOG_LINES)
        self._log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        lf = QFont("Consolas")
        lf.setStyleHint(QFont.StyleHint.Monospace)
        self._log.setFont(lf)
        self._log.setFixedHeight(9 * QFontMetrics(lf).height())
        self._log.setVisible(False)
        outer.addWidget(self._log)

        # Detach the incoming list before replaying it: append_log writes into self.view.log,
        # so iterating the same object would append to the list being iterated.
        initial, self.view.log = self.view.log, []
        if initial:
            self.append_log(initial)
        self.refresh()

    # ---- updates -----------------------------------------------------------------------
    def update_view(self, view: JobView | Mapping | Any) -> None:
        """Replace the snapshot. Log lines already shown are kept, not re-appended."""
        new = JobView.from_obj(view)
        new.log = self.view.log
        self.view = new
        self.refresh()

    def update_progress(self, parsed: Mapping[str, Any] | None) -> None:
        """Apply one parsed progress record from core.logparse; None is a no-op.

        Key names are read tolerantly (`ms_per_step` or `ms`, `eta_s` or `eta`) because the
        parser's record shape is another stream's decision and a KeyError in a signal handler
        would take the whole queue pane down.
        """
        if not parsed:
            return
        g = parsed.get
        v = self.view
        v.percent = _first(g("percent"), g("pct"), g("frac_pct"), default=v.percent)
        v.step = _first(g("step"), default=v.step)
        v.total_steps = _first(g("total"), g("total_steps"), g("steps"), default=v.total_steps)
        v.ms_per_step = _first(g("ms_per_step"), g("ms"), default=v.ms_per_step)
        v.eta_s = _first(g("eta_s"), g("eta"), default=v.eta_s)
        v.elapsed_s = _first(g("elapsed_s"), g("elapsed"), default=v.elapsed_s)
        if v.percent is None and v.step and v.total_steps:
            v.percent = 100.0 * v.step / v.total_steps
        self.refresh()

    def append_log(self, lines: str | Iterable[str]) -> None:
        """Verbatim. No filtering, no reformatting - see the module docstring."""
        if isinstance(lines, str):
            lines = [lines]
        for ln in lines:
            text = ln.rstrip("\r\n")
            self.view.log.append(text)
            self._log.appendPlainText(text)
        del self.view.log[:-self.LOG_LINES]

    def log_text(self) -> str:
        return self._log.toPlainText()

    def set_log_visible(self, on: bool) -> None:
        self._log_toggle.setChecked(on)

    def _on_toggle(self, on: bool) -> None:
        self._log.setVisible(on)
        self._log_toggle.setText("log" if not on else "log (hide)")
        if on:                                  # jump to the tail: that is where the error is
            self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    # ---- render ------------------------------------------------------------------------
    def refresh(self) -> None:
        v = self.view
        self._stage.setText(v.stage)
        fm = QFontMetrics(self._summary.font())
        self._summary.setText(fm.elidedText(v.summary, Qt.TextElideMode.ElideRight,
                                            max(120, self._summary.width())))
        self._summary.setToolTip(v.summary)

        col = STATE_COLOR.get(v.state, INK_SOFT)
        self._state.setText(v.state)
        self._state.setStyleSheet(
            f"color:{col}; background:transparent; border:1px solid {col};"
            f"border-radius:3px; padding:1px 7px;")
        self._cancel.setVisible(v.state == "running")

        running = v.state == "running"
        if v.percent is not None:
            self._bar.setRange(0, 1000)
            self._bar.setValue(int(round(min(100.0, max(0.0, v.percent)) * 10)))
        elif running:
            self._bar.setRange(0, 0)            # busy sweep: meshing prints stages, not percent
        else:
            self._bar.setRange(0, 1000)
            self._bar.setValue(1000 if v.state == "succeeded" else 0)

        bits = []
        if v.step and v.total_steps:
            bits.append(f"step {v.step}/{v.total_steps}")
        if v.ms_per_step is not None:
            bits.append(f"{v.ms_per_step:.1f} ms/step")
        if v.elapsed_s is not None:
            bits.append(f"elapsed {fmt_dur(v.elapsed_s)}")
        if running and v.eta_s is not None:
            bits.append(f"ETA {fmt_dur(v.eta_s)}")
        self._numbers.setText("   ".join(bits))


def _first(*vals: Any, default: Any = None) -> Any:
    return next((v for v in vals if v is not None), default)


def demo() -> None:
    """Self-check: state colours, cancel visibility, and that unmatched log lines survive."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    card = JobCard({"job_id": "j1", "stage": "forward", "state": "queued",
                    "summary": "deg 4  scale 0.80  +20 deg  GPU  none"})
    assert not card._cancel.isVisibleTo(card), "no cancel button on a queued job"

    card.update_view(JobView(job_id="j1", stage="forward", state="running",
                             summary="deg 4  scale 0.80  +20 deg  GPU  none"))
    assert card._cancel.isVisibleTo(card), "running jobs must be cancellable"
    assert card._bar.maximum() == 0, "no percent yet -> indeterminate bar"

    card.update_progress({"step": 140000, "total": 182457, "ms_per_step": 4.1,
                          "elapsed_s": 576.0, "eta_s": 174.0})
    assert abs(card.view.percent - 76.7) < 0.2, card.view.percent
    assert card._bar.value() == int(round(card.view.percent * 10))
    nums = card._numbers.text()
    assert "4.1 ms/step" in nums and "ETA 2.9 min" in nums, nums

    # The whole point: a line no parser recognises must still reach the user.
    ugly = "RuntimeError: CUDA out of memory (tried to allocate 2.14 GiB)"
    card.update_progress(None)                       # logparse said "I do not know this line"
    card.append_log([ugly, "  step 140001/182457 (77%)  4.1 ms/step"])
    assert ugly in card.log_text(), "unmatched lines must be shown verbatim"
    card.set_log_visible(True)
    assert card._log.isVisibleTo(card)

    card.update_view(JobView(job_id="j1", stage="forward", state="failed"))
    assert STATE_COLOR["failed"] in card._state.styleSheet()
    assert not card._cancel.isVisibleTo(card)
    assert ugly in card.log_text(), "a state change must not clear the log"

    cancelled: list[str] = []
    run = JobCard(JobView(job_id="j2", stage="mesh", state="running"))
    run.cancel_requested.connect(cancelled.append)
    run._cancel.click()
    assert cancelled == ["j2"], cancelled

    assert fmt_dur(None) == "--" and fmt_dur(8640) == "2.40 h" and fmt_dur(45) == "45 s"

    # A card built WITH a backlog must replay it once, not loop on its own list.
    pre = JobCard(JobView(job_id="j3", stage="image", state="succeeded",
                          log=["[1] gmsh", "[7] wrote mesh"]))
    assert pre.log_text().splitlines() == ["[1] gmsh", "[7] wrote mesh"], pre.log_text()
    print("jobcard.demo: ok")
    del app


if __name__ == "__main__":
    demo()
