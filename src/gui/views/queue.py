r"""The queue pane: a scrollable stack of job cards, newest first, with a summary header.

WHY THE RUNNER IS OPTIONAL
core/runner.py is another workstream's file and may not exist yet. This pane therefore never
imports it at module level: it takes plain `JobView` snapshots, and `attach_runner` connects
to whatever signals the runner turns out to expose. With the runner absent the pane still
constructs, still renders, and `fake_jobs()` gives it something to show - which is the only
way to develop and test the queue without launching a 2.4 h solve.

The signal-name table in `attach_runner` is deliberate slack, not indecision: the runner's
signal names are not settled, and a missing connection must degrade to "no live updates"
rather than to an exception inside a Qt slot. Once core/runner.py is on disk, delete the
aliases it does not use.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

# Direct execution puts src/gui/views on sys.path; the app puts src/gui there. Normalise, so
# `python src/gui/views/queue.py` and `python src/gui/main.py` resolve `widgets.*` the same way.
_GUI_ROOT = Path(__file__).resolve().parents[1]
if str(_GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_GUI_ROOT))

from PySide6.QtCore import Qt, Signal                                    # noqa: E402
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QScrollArea, QSizePolicy,  # noqa: E402
                               QVBoxLayout, QWidget)

from widgets.jobcard import (INK_SOFT, STATE_COLOR, TERMINAL, JobCard,   # noqa: E402
                            JobView)


class QueueView(QWidget):
    """A list of JobCards keyed by job_id, plus a one-line summary of the whole queue."""

    cancel_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cards: dict[str, JobCard] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        head = QHBoxLayout()
        self._summary = QLabel()
        self._summary.setStyleSheet(f"color:{INK_SOFT};")
        head.addWidget(QLabel("<b>Queue</b>"))
        head.addStretch(1)
        head.addWidget(self._summary)
        outer.addLayout(head)

        self._body = QWidget()
        self._stack = QVBoxLayout(self._body)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.setSpacing(8)
        self._stack.addStretch(1)          # keeps cards top-aligned as the list empties

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setWidget(self._body)
        self._scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        outer.addWidget(self._scroll, 1)

        self._empty = QLabel("nothing queued")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet(f"color:{INK_SOFT};")
        outer.addWidget(self._empty)
        self._refresh_summary()

    # ---- data in -----------------------------------------------------------------------
    def upsert(self, view: JobView | Mapping | Any) -> JobCard:
        """Add the job, or update it if its job_id is already on screen. Newest first."""
        v = JobView.from_obj(view)
        card = self._cards.get(v.job_id)
        if card is None:
            card = JobCard(v)
            card.cancel_requested.connect(self.cancel_requested)
            self._stack.insertWidget(0, card)          # index 0 = top = newest
            self._cards[v.job_id] = card
        else:
            card.update_view(v)
        self._refresh_summary()
        return card

    def set_jobs(self, views: Iterable[Any]) -> None:
        """Replace the list wholesale. Given oldest-first, the pane still shows newest first."""
        for card in self._cards.values():
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        for v in views:
            self.upsert(v)

    def append_log(self, job_id: str, lines: str | Iterable[str]) -> None:
        card = self._cards.get(job_id)
        if card is not None:
            card.append_log(lines)

    def update_progress(self, job_id: str, parsed: Mapping[str, Any] | None) -> None:
        card = self._cards.get(job_id)
        if card is not None:
            card.update_progress(parsed)

    def card(self, job_id: str) -> JobCard | None:
        return self._cards.get(job_id)

    def counts(self) -> dict[str, int]:
        c = {k: 0 for k in STATE_COLOR}
        for card in self._cards.values():
            c[card.view.state] = c.get(card.view.state, 0) + 1
        return c

    def _refresh_summary(self) -> None:
        c = self.counts()
        parts = [f"{c.get('queued', 0)} queued", f"{c.get('running', 0)} running",
                 f"{c.get('succeeded', 0)} done"]
        for state in ("failed", "cancelled"):
            if c.get(state):
                parts.append(f"{c[state]} {state}")
        self._summary.setText("   ".join(parts))
        self._empty.setVisible(not self._cards)
        self._scroll.setVisible(bool(self._cards))

    # ---- runner wiring -----------------------------------------------------------------
    def attach_runner(self, runner: Any = None) -> list[str]:
        """Connect to core.runner.Runner. Returns the signal names actually connected.

        `runner=None` means "look for a shared instance on the main window", because main.py
        owns the single Runner and builds this view with no arguments. Every step is guarded:
        with core/runner.py absent, or a signal renamed, the pane loses live updates and
        nothing else.
        """
        if runner is None:
            runner = _shared_runner(self)
        if runner is None:
            return []
        self._runner = runner
        wired: list[str] = []
        for name, slot in (("queue_changed", self.sync_from_runner),
                           ("job_started", self._slot_id),
                           ("job_finished", self._slot_id),
                           ("job_progress", self._slot_progress),
                           ("job_log", self.append_log)):
            sig = getattr(runner, name, None)
            if sig is not None and hasattr(sig, "connect"):
                sig.connect(slot)
                wired.append(name)
        if callable(getattr(runner, "cancel", None)):
            self.cancel_requested.connect(runner.cancel)
        self.sync_from_runner()
        return wired

    def showEvent(self, ev) -> None:  # noqa: ANN001
        """Best-effort self-attach the first time the pane is shown.

        main.py builds views with no arguments and wires only run_requested/queue_requested,
        so without this the queue pane would sit empty next to a running solve. An explicit
        `attach_runner(runner)` from the shell is still the preferred wiring; this is the
        fallback that makes the pane work without editing another stream's file.
        """
        if not getattr(self, "_runner", None):
            self.attach_runner()
        super().showEvent(ev)

    def sync_from_runner(self, *_: Any) -> None:
        """Rebuild every card's state from runner.jobs().

        queue_changed carries no payload by design (the runner hands out ids, not Job objects,
        so a view cannot mutate queue state), so the pane re-reads the list. Cards are upserted
        rather than recreated, which is what keeps each card's log tail and progress alive.
        """
        runner = getattr(self, "_runner", None)
        jobs = getattr(runner, "jobs", None)
        if not callable(jobs):
            return
        for job in jobs():
            self.upsert(view_from_job(job))

    def _slot_id(self, job_id: str, *_: Any) -> None:
        """job_started / job_finished carry only the id; the details come from the runner."""
        runner = getattr(self, "_runner", None)
        job = runner.job(job_id) if runner is not None and hasattr(runner, "job") else None
        self.upsert(view_from_job(job) if job is not None
                    else JobView(job_id=job_id, state="running"))

    def _slot_progress(self, job_id: str, progress: Any) -> None:
        """job_progress carries a logparse.Progress (fraction 0..1, detail, ms/step, ETA)."""
        if progress is None:                     # a line the parser did not understand
            return
        if isinstance(progress, Mapping):
            self.update_progress(job_id, progress)
            return
        frac = getattr(progress, "fraction", None)
        self.update_progress(job_id, {
            "percent": None if frac is None else 100.0 * float(frac),
            "detail": getattr(progress, "detail", "") or "",
            "ms_per_step": getattr(progress, "ms_per_step", None),
            "eta_s": (lambda m: None if m is None else 60.0 * float(m))(
                getattr(progress, "eta_seconds", None)),
        })


def _shared_runner(widget: QWidget) -> Any:
    """The app's single Runner, if the shell has one. Never constructs a second one.

    A second Runner would mean two queues fighting over the same GPU and the same output
    filenames, so this only ever borrows an existing instance.
    """
    win = widget.window()
    for name in ("runner", "_runner_obj"):
        got = getattr(win, name, None)
        if got is not None and hasattr(got, "jobs"):
            return got
    factory = getattr(win, "_runner", None)      # main.py's lazy accessor
    if callable(factory):
        try:
            return factory()
        except Exception:
            return None
    return None


def summarize_config(cfg: Any) -> str:
    """One line of parameters, in the order a reader scans for them."""
    if cfg is None:
        return ""
    g = lambda n, d=None: getattr(cfg, n, d)              # noqa: E731 - local shorthand
    bits = []
    if g("degree") is not None:
        bits.append(f"deg {g('degree')}")
    if g("scale") is not None:
        bits.append(f"scale {g('scale'):.2f}")
    if g("angle") is not None:
        bits.append(f"{g('angle'):+.0f} deg")
    if g("cfl") is not None:
        bits.append(f"cfl {g('cfl'):.2f}")
    for name in ("device", "notch", "artifact_reduction"):
        v = g(name)
        if v is not None:
            bits.append(str(getattr(v, "value", v)))
    if g("snapshots"):
        bits.append(f"{g('snapshots')} snapshots")
    return "  ".join(bits)


def view_from_job(job: Any) -> JobView:
    """core.runner.Job -> JobView. Reads only the public fields the runner documents."""
    spec = getattr(job, "spec", None)
    cfg = getattr(spec, "config", None)
    state = getattr(getattr(job, "state", None), "value", str(getattr(job, "state", "queued")))
    stage = getattr(getattr(spec, "stage", None), "value", "") or "job"
    started, ended = getattr(job, "started", 0.0) or 0.0, getattr(job, "ended", None)
    elapsed = None
    if started:
        elapsed = (ended if ended else time.time()) - started
    label = getattr(spec, "label", "") or ""
    summary = "   ".join(x for x in (label, summarize_config(cfg)) if x)
    return JobView(job_id=getattr(job, "job_id", ""), stage=stage, state=state,
                   summary=summary, elapsed_s=elapsed,
                   ms_per_step=getattr(job, "ms_per_step", None),
                   percent=100.0 if state == "succeeded" else None)


def fake_jobs() -> list[JobView]:
    """Development fixture. Mirrors real solver output, including a line no parser matches."""
    return [
        JobView("g1", "mesh", "succeeded", "scale 0.80  quad  notch present",
                percent=100.0, elapsed_s=41.0,
                log=["[1] geometry", "[7] wrote results/ili_mesh/ili_mesh_s0p8.msh"]),
        JobView("g2", "forward", "running", "deg 4  cfl 0.30  +20 deg  GPU  abc-legacy",
                percent=76.7, step=140000, total_steps=182457, ms_per_step=4.1,
                elapsed_s=576.0, eta_s=174.0,
                log=["  step 140000/182457 (77%)  4.1 ms/step  elapsed 9.6 min  ETA 2.9 min"]),
        JobView("g3", "image", "queued", "chain faithfulbf  +20 deg  --no-overlay"),
        JobView("g4", "forward", "failed", "deg 4  cfl 0.30  -20 deg  GPU",
                percent=12.0, elapsed_s=88.0,
                log=["RuntimeError: CUDA out of memory (tried to allocate 2.14 GiB)"]),
    ]


def demo() -> None:
    """Self-check: ordering, header counts, cancel plumbing, and a runner-less attach."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    q = QueueView()
    q.resize(700, 600)
    assert q.counts() == {k: 0 for k in STATE_COLOR}
    assert "0 queued" in q._summary.text()

    q.set_jobs(fake_jobs())
    assert len(q._cards) == 4
    # Newest first: the last job handed in sits at the top of the stack.
    assert q._stack.itemAt(0).widget().view.job_id == "g4"
    assert q._stack.itemAt(3).widget().view.job_id == "g1"
    s = q._summary.text()
    assert s.startswith("1 queued   1 running   1 done") and "1 failed" in s, s

    # An update must not duplicate the card, and must not clear its log.
    q.append_log("g2", "PETSc: using device 0")
    q.upsert(JobView("g2", "forward", "succeeded", "deg 4  cfl 0.30  +20 deg  GPU",
                     percent=100.0, elapsed_s=462.0))
    assert len(q._cards) == 4
    assert "PETSc: using device 0" in q.card("g2").log_text()
    assert "2 done" in q._summary.text(), q._summary.text()

    q.update_progress("g3", {"step": 10, "total": 100, "ms_per_step": 3.9})
    assert abs(q.card("g3").view.percent - 10.0) < 1e-9

    seen: list[str] = []
    q.cancel_requested.connect(seen.append)
    q.upsert(JobView("g5", "forward", "running", "cancel me"))
    q.card("g5")._cancel.click()
    assert seen == ["g5"], seen

    # core.runner is absent in this workstream; attaching must be a no-op, not a crash.
    assert q.attach_runner() == []
    for state in TERMINAL:
        assert state in STATE_COLOR

    q.grab()
    print("queue.demo: ok")
    del app


if __name__ == "__main__":
    demo()
