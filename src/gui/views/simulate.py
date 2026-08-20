r"""The parameter form: everything the user can set, what it means, and what it will cost.

Two columns. Left is the form: one FieldRow per parameter, each carrying its plain-language
label, its flag, an always-visible one-line hint, a "?" for the fuller explanation, and the
derived consequence of the value currently in it. Right is the summary: the schematic above,
the consequences panel below, then the guardrails and the dry-run command - and under all of
that, behind a draggable splitter, the job queue.

THE QUEUE IS PART OF THIS PAGE (2026-08-20). It used to be its own rail section, which meant
submitting a run navigated away from the form that described it - so checking a parameter
against a running job was two clicks each way. Same widget (views/queue.QueueView), embedded.
This view still owns no runner: main.py calls attach_runner() here and it forwards.

Every user-facing string comes from model/help.py and nowhere else. This module has no copy of
its own, so the physics claims stay in one reviewable file instead of spreading across widget
code where nobody proofreads them.

This view NEVER runs anything. It emits run_requested / queue_requested with a RunConfig and
the stages to run; main.py wires those to core/runner.py. That split is what keeps the form
testable with no Docker on the machine, and it is why "Dry run" is local: it only prints the
command that would be issued.

Field defaults come from model/spec.py and nowhere else, so the form opens on the published
+20 deg configuration.

THE k-WAVE COMPARISON IS NOT A GUI CONCEPT (2026-08-20). Enough comparison data exists for the
publication and any further comparison is done by hand from the CLI, where
`compare_images.py --theirs` still works. So the form offers no comparison field and the image
stage beamforms our own channel data only.
"""
from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    # Direct run. This has to happen BEFORE any other import: Python puts the script's own
    # directory first on sys.path, and views/ contains inspect.py - which shadows the standard
    # library's `inspect`, so `import dataclasses` fails with a circular-import error four
    # frames deep in PySide6. Drop views/ and make src/gui the import root instead.
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
    sys.path.insert(0, os.path.dirname(_HERE))

from dataclasses import fields, replace
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, 
                            QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, 
                            QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTabWidget, 
                            QVBoxLayout, QWidget)

from model.spec import (ArtifactReduction, Device, Notch, RunConfig, Stage, WIDE_X_MAX,
                        WIDE_X_MIN, plan)
from widgets.consequences import Consequences, fmt
from widgets.crosssection import CrossSection
from widgets.fieldrow import (FieldRow, WrapLabel, build_combo, choice_key, help_for,
                              shrink)

CUSTOM = "(custom)"

REPO = Path(__file__).resolve().parents[3]

# Mesh fineness as a named choice, because "0.8" means nothing to someone who has not read the
# meshing section. model/help.py has no `choices` for --scale (it is continuous), so only the
# NAMES live here; what the numbers cost is still help.py's hint and detail.
FINENESS: tuple[tuple[float, str], ...] = (
    (1.00, "Coarse (1.0)"),
    (0.80, "Production (0.8)"),
    (0.65, "Fine (0.65)"),
)

# What each pipeline stage does, for the checkboxes in the action row.
STAGE_TIPS = {Stage.MESH: "build the mesh this configuration names",
              Stage.FORWARD: "run the solver",
              # No comparison: the image stage beamforms our own channel data and nothing
              # else. A k-Wave comparison is a manual CLI job now.
              Stage.IMAGE: "beamform our own channel data into an image",
              Stage.FIGURES: "wavefield animation"}


def _scenario_rows(sc) -> tuple[tuple[str, str], ...]:
    """The read-only facts, formatted. `source` is on screen so a fallback is never mistaken
    for the container's own answer."""
    return (
        ("pipe", "ID r %.3f / OD r %.3f mm, wall %.3f mm" % (sc.r_id, sc.r_od, sc.wall)),
        ("standoff", "%.1f mm water on the beam axis" % sc.standoff),
        ("array", "%d el @ %.2f mm = %.1f mm aperture at z = 0"
                  % (sc.n_elem, sc.pitch, sc.aperture)),
        ("pulse", "%d-cycle toneburst, f0 %.1f MHz (resolve %.1f MHz)"
                  % (sc.n_cycle, sc.f0 / 1e6, sc.f_upper / 1e6)),
        ("steel", "c_P %.0f, c_S %.0f m/s, rho %.0f kg/m3" % (sc.c_p, sc.c_s, sc.rho_s)),
        ("water", "c %.0f m/s, rho %.0f kg/m3" % (sc.c_f, sc.rho_f)),
        ("notch", "%.1f x %.1f mm slot at x = %.2f mm, from the OD inward"
                  % (sc.notch_depth, sc.notch_width, sc.notch_x)),
        ("domain", "%+.1f .. %+.1f mm, limits %+.2f .. %+.2f"
                   % (sc.x_min, sc.x_max, sc.x_limit_lo, sc.x_limit_hi)),
        ("source", sc.source),
    )


def _optional(mod: str):
    """Import a sibling module that another workstream owns, or None. The four streams land in
    any order, so nothing here may hard-depend on a module that is not written yet."""
    try:
        return __import__(mod, fromlist=["_"])
    except ImportError:
        return None


def _free_bytes() -> int | None:
    """Free space where results land. shutil, not the docker probe: the guard needs a number
    on every keystroke and a subprocess per keystroke is not that."""
    import shutil
    try:
        return shutil.disk_usage(REPO / "data" / "results").free
    except OSError:
        return None


def _show_value(value: object) -> str:
    """A config value as the "differs from published" dot should say it - the help.py key
    rather than the repr, so the hover reads "published: present", not "Notch.PRESENT"."""
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "on" if value else "off"
    return choice_key(value)


def _consequence_lines(cfg: RunConfig) -> dict[str, str]:
    """The derived effect of each value, keyed by field, for the line under that field.

    All of it is model/derived.py's arithmetic and all of it is labelled an estimate, per
    README non-negotiable #1. The crack cell size is the one that earns this feature: there is
    ONE time step for the whole model and the smallest cell sets it, so a "harmless"
    refinement at the crack multiplies the step count everywhere - and nothing on screen used
    to say so.
    """
    derived = _optional("model.derived")
    scenario = _optional("model.scenario")
    if derived is None or scenario is None:
        return {}
    try:
        sc = scenario.load()
        h = derived.target_h(cfg, sc)
        res = derived.nodes_per_wavelength(cfg, sc)
        dt = derived.estimated_dt(cfg, sc).value
        steps = derived.estimated_steps(cfg, sc).value
        dof = derived.estimated_dof(cfg).value
        runtime = derived.estimated_runtime(cfg, sc).value
        disk = derived.estimated_disk(cfg).value
    except Exception:            # a half-built or failing model module must not kill the form
        return {}
    step_line = "time step ~%.3f ns -> %s steps (est.)" % (dt * 1e9, fmt(steps, "steps"))
    return {
        "scale": "cells: water %.3f mm, steel %.3f mm | %s nodes per wavelength in %s (est.)"
                 % (h["water"], h["steel"], fmt(res.value, "nodes"), res.binding),
        "degree": "%s unknowns | solve ~%s on %s (est.)"
                  % (fmt(dof, "dof"), fmt(runtime, "s"), cfg.device.value.upper()),
        "h_notch": "crack cells %.3f mm | %s" % (h["notch"], step_line),
        "cfl": step_line,
        "t_end": "%s steps at ~%.3f ns (est.)" % (fmt(steps, "steps"), dt * 1e9),
        "snapshots": "projected disk ~%s for the whole pipeline (est.)" % fmt(disk, "B"),
        "device": "solve ~%s (est.)" % fmt(runtime, "s"),
        "quad": "%s cells, %s unknowns (est.)"
                % (fmt(derived.estimated_cells(cfg).value, "cells"), fmt(dof, "dof")),
    }


class _Group(QGroupBox):
    """A collapsible group of FieldRows.

    Qt's checkable group box is the collapse toggle - no custom header, no animation. The title
    carries [-] or [+] too, because a 9 px checkbox indicator is not enough signal to tell an
    empty group from a collapsed one at a glance.
    """

    def __init__(self, title: str, parent: QWidget | None = None,
                 open_: bool = True) -> None:
        super().__init__(parent)
        self._title = title
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.body = QWidget(self)
        self.box = QVBoxLayout(self.body)
        self.box.setContentsMargins(0, 0, 0, 0)
        self.box.setSpacing(2)
        outer.addWidget(self.body)
        self.toggled.connect(self.body.setVisible)
        self.toggled.connect(self._sync_title)
        self.setChecked(open_)
        self.body.setVisible(open_)
        self._sync_title(open_)

    def _sync_title(self, on: bool) -> None:
        self.setTitle(("[-] " if on else "[+] ") + self._title)

    def add(self, w: QWidget) -> None:
        self.box.addWidget(w)


class _Fineness(QWidget):
    """Mesh fineness as a named choice, with the raw number revealed only for Custom.

    A bare spinbox invites arbitrary values on the one parameter whose cost is roughly cubic
    in itself (4x the cells and a proportionally smaller time step). Naming the three that are
    actually used makes the published setting the obvious one, and Custom keeps the door open
    for the exact scale a guardrail names.
    """

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.combo = QComboBox()
        shrink(self.combo)
        for value, label in FINENESS:
            self.combo.addItem(label, value)
        self.combo.addItem("Custom...", None)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.20, 2.00)
        self.spin.setDecimals(2)
        self.spin.setSingleStep(0.05)
        self.spin.setVisible(False)
        # Capped: a spinbox asks for room for its widest possible value plus its buttons, and
        # two of those side by side set the minimum width of the whole form column - which is
        # what clips the hints at 1100 px instead of wrapping them.
        self.spin.setMaximumWidth(72)
        lay.addWidget(self.combo, 1)
        lay.addWidget(self.spin, 0)
        self.combo.currentIndexChanged.connect(self._on_combo)
        self.spin.valueChanged.connect(lambda _v: self.changed.emit())
        self.combo.setCurrentIndex(1)                      # Production (0.8) is the default

    def _on_combo(self, _i: int) -> None:
        custom = self.combo.currentData() is None
        self.spin.setVisible(custom)
        if not custom:
            self.spin.blockSignals(True)      # the combo already IS the change; don't double it
            self.spin.setValue(float(self.combo.currentData()))
            self.spin.blockSignals(False)
        self.changed.emit()

    def tab_widgets(self) -> tuple[QWidget, ...]:
        return (self.combo, self.spin)

    def value(self) -> float:
        data = self.combo.currentData()
        return self.spin.value() if data is None else float(data)

    def setValue(self, v: float) -> None:      # noqa: N802 - matches QDoubleSpinBox
        for i, (value, _label) in enumerate(FINENESS):
            if abs(value - v) < 1e-9:
                self.combo.setCurrentIndex(i)
                return
        self.spin.setValue(v)
        self.combo.setCurrentIndex(self.combo.count() - 1)


class _Window(QWidget):
    """The snapshot window: an optional (start, end) pair in microseconds.

    The contract allows None - save frames over the whole record - so the checkbox is not
    decoration: without it the form could not round-trip a config that has no window.
    """

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.on = QCheckBox("limit")
        self.lo = QDoubleSpinBox()
        self.hi = QDoubleSpinBox()
        for s in (self.lo, self.hi):
            s.setRange(0.0, 500.0)
            s.setDecimals(1)
            s.setSingleStep(2.0)
            # No per-spin suffix: FieldRow already puts help.py's unit once at the end of the
            # row, and " us" inside each box only eats the digits.
            s.setMaximumWidth(78)             # see _Fineness.spin: keep the column narrow
            s.valueChanged.connect(lambda _v: self.changed.emit())
        lay.addWidget(self.on, 0)
        lay.addWidget(self.lo, 1)
        lay.addWidget(self.hi, 1)
        self.on.toggled.connect(self._on_toggle)
        self.on.setChecked(True)

    def _on_toggle(self, on: bool) -> None:
        self.lo.setEnabled(on)
        self.hi.setEnabled(on)
        self.changed.emit()

    def tab_widgets(self) -> tuple[QWidget, ...]:
        return (self.on, self.lo, self.hi)

    def value(self) -> tuple[float, float] | None:
        return (self.lo.value(), self.hi.value()) if self.on.isChecked() else None

    def setValue(self, v: tuple[float, float] | None) -> None:   # noqa: N802 - Qt-ish name
        self.on.setChecked(v is not None)
        if v is not None:
            self.lo.setValue(v[0])
            self.hi.setValue(v[1])


class SimulateView(QWidget):
    """The form. Owns no runner and no Docker; it only describes a run."""

    run_requested = Signal(object, object)      # (RunConfig, tuple[Stage, ...])
    queue_requested = Signal(object, object)
    config_changed = Signal(object)             # (RunConfig) - for the status bar / other views

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._base = RunConfig()          # carries the fields this form does not expose
        self._loading = False            # suppress the change cascade while populating widgets
        self._rows: dict[str, FieldRow] = {}
        self._presets: dict[str, RunConfig] = {}
        self._last_preset: str | None = None
        self._tab: list[QWidget] = []

        self.cross = CrossSection()
        self.consequences = Consequences()
        # Optional, and lazily so: the equation panel pulls in matplotlib, and a form that
        # cannot open because a plotting library is missing would be a poor trade.
        try:
            from widgets.equation import EquationPanel
            self.equation = EquationPanel()
        except Exception:
            self.equation = None
        # The queue lives on THIS page: setting a run up and watching it run are one screen, so
        # Run no longer navigates anywhere. Same lazy-and-degrade rule as the equation panel -
        # views/queue.py is another stream's file and the form must open without it.
        try:
            from views.queue import QueueView
            self.queue = QueueView()
        except Exception:
            self.queue = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_form_column())
        split.addWidget(self._build_result_column())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 4)
        # Explicit, because stretch factors alone let the schematic's own size hint squeeze the
        # form down to its minimum at 1100 px - which is where the hints start clipping.
        split.setSizes([480, 620])
        root.addWidget(split, 1)
        root.addWidget(self._build_action_row(), 0)

        # Tab follows reading order. Creation order almost does, but the composite widgets
        # build their children before the row that hosts them, so it is stated not assumed.
        for a, b in zip(self._tab, self._tab[1:]):
            self.setTabOrder(a, b)

        self._load_scenario_facts()
        self.set_config(self._base)

    # ---- construction --------------------------------------------------------------

    def _row(self, group: _Group, name: str, widget: QWidget) -> FieldRow:
        """One parameter: its widget, its copy from model/help.py, and its change wire."""
        row = FieldRow(name, widget)
        self._rows[name] = row
        group.add(row)
        for sig in ("valueChanged", "currentIndexChanged", "toggled", "changed"):
            s = getattr(widget, sig, None)
            if s is not None and hasattr(s, "connect"):
                s.connect(self._on_change)
                break
        tabs = getattr(widget, "tab_widgets", None)
        self._tab.extend(tabs() if callable(tabs) else [widget])
        return row

    def _build_form_column(self) -> QWidget:
        col = QWidget()
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # --- preset, on the same label column as every field below it
        from widgets.fieldrow import LABEL_W
        head = QWidget()
        hl = QHBoxLayout(head)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)
        cap = QLabel("Preset")
        cap.setFixedWidth(LABEL_W)
        cap.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.preset = QComboBox()
        shrink(self.preset)
        # Populate BEFORE connecting: addItems fires currentTextChanged, and a preset applied
        # while the rest of the form does not exist yet is an AttributeError, not a preset.
        self.preset.addItems(self._preset_names() + [CUSTOM])
        self.preset.currentTextChanged.connect(self._on_preset)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("put every field back to the last named preset")
        self.btn_reset.setAutoDefault(False)
        self.btn_reset.clicked.connect(self._on_reset)
        hl.addWidget(cap, 0)
        hl.addWidget(self.preset, 1)
        hl.addWidget(self.btn_reset, 0)
        lay.addWidget(head)
        self._tab += [self.preset, self.btn_reset]

        # --- Beam
        g = _Group("Beam")
        self.angle = QDoubleSpinBox()
        self.angle.setRange(-60.0, 60.0)
        self.angle.setDecimals(1)
        self.angle.setSingleStep(5.0)
        self.chain = build_combo("chain", ("faithfulbf", "legacy"))
        self._row(g, "angle", self.angle)
        self._row(g, "chain", self.chain)
        lay.addWidget(g)

        # --- Mesh
        g = _Group("Mesh")
        self.fineness = _Fineness()
        self.degree = build_combo("degree", (3, 4, 5))
        self.h_notch = QDoubleSpinBox()
        self.h_notch.setRange(0.0, 2.0)
        self.h_notch.setDecimals(2)
        self.h_notch.setSingleStep(0.05)
        # 0.0 means "let the mesher decide", i.e. h_notch = None in the config.
        self.h_notch.setSpecialValueText("auto")
        self.notch = build_combo("notch", tuple(Notch))
        self.quad = self._flag_box("quad")
        # A separate axis from `notch`: --staircase rasterises the INNER WALL ARC, and the C4
        # experiment ran it on a healthy wall. See RunConfig.staircase_id.
        self.staircase = self._flag_box("staircase_id")
        self._row(g, "scale", self.fineness)
        self._row(g, "degree", self.degree)
        self._row(g, "h_notch", self.h_notch)
        self._row(g, "notch", self.notch)
        self._row(g, "quad", self.quad)
        self._row(g, "staircase_id", self.staircase)
        lay.addWidget(g)

        # --- Domain: three fixed treatments, NONE default. The knobs behind each are
        # deliberately not exposed - tuning a boundary treatment against an image is how you
        # talk yourself into an artefact.
        g = _Group("Domain")
        self.artifact = build_combo("artifact_reduction", tuple(ArtifactReduction))
        self._row(g, "artifact_reduction", self.artifact)
        lay.addWidget(g)

        # --- Compute
        g = _Group("Compute")
        self.device = build_combo("device", (Device.GPU, Device.CPU))
        self.snapshots = QSpinBox()
        self.snapshots.setRange(0, 600)
        self.snapshots.setSingleStep(60)
        self.snapshots.setSpecialValueText("off")
        self._row(g, "device", self.device)
        self._row(g, "snapshots", self.snapshots)
        lay.addWidget(g)

        # --- Advanced: real parameters, rarely touched, so collapsed rather than absent.
        # Leaving them out was worse: t_end and cfl appear in every command the app prints,
        # and a value you can read in the dry run but cannot find in the form is its own kind
        # of confusing.
        g = _Group("Advanced", open_=False)
        self.t_end = QDoubleSpinBox()
        self.t_end.setRange(1.0, 200.0)
        self.t_end.setDecimals(1)
        self.t_end.setSingleStep(5.0)
        self.cfl = QDoubleSpinBox()
        self.cfl.setRange(0.05, 1.00)
        self.cfl.setDecimals(2)
        self.cfl.setSingleStep(0.05)
        self.snap_window = _Window()
        self._row(g, "t_end", self.t_end)
        self._row(g, "cfl", self.cfl)
        self._row(g, "snap_window", self.snap_window)
        lay.addWidget(g)

        # --- Scenario: read-only. These are backend constants, not parameters.
        g = _Group("Scenario (read-only)", open_=False)
        facts = QWidget()
        self._scenario_form = QFormLayout(facts)
        self._scenario_form.setContentsMargins(0, 0, 0, 0)
        self._scenario_form.setSpacing(4)
        self._scenario_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        g.add(facts)
        lay.addWidget(g)

        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # No horizontal scrolling: the column takes the viewport width, which is what makes the
        # wrapped hints and details wrap instead of widening the form.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(col)
        scroll.setMinimumWidth(470)      # below this the label column crushes the inputs
        return scroll

    def _flag_box(self, name: str) -> QCheckBox:
        """A boolean field. Its text lives in the row's label column with every other label,
        so the box itself is bare - repeating the label beside it breaks the value column."""
        box = QCheckBox("")
        box.setAccessibleName(help_for(name).label)
        return box

    def _build_result_column(self) -> QWidget:
        """The summary column, with the job queue under it.

        A splitter, not a fixed split: a job card with its log expanded needs room that only
        the user knows they want, and the alternative (a scroll area inside a scroll area)
        is worse. The queue keeps a floor tall enough for its header plus one card, so
        dragging it down to a sliver still shows "N queued / N running / N done".
        """
        top = self._build_summary_column()
        if self.queue is None:
            return top
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self.queue)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([460, 240])
        self.queue.setMinimumHeight(150)
        split.setCollapsible(1, False)      # a queue you cannot see is a queue you forget
        return split

    def _build_summary_column(self) -> QWidget:
        col = QWidget()
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        card = QFrame()
        card.setProperty("role", "card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(1, 1, 1, 1)
        if self.equation is not None:
            tabs = QTabWidget()
            tabs.addTab(self.cross, "Geometry")
            tabs.addTab(self.equation, "Equation")
            tabs.setToolTip("Geometry: the domain you are about to solve on.\n"
                            "Equation: the system that will be solved, with the terms that "
                            "are not in force dimmed.")
            cl.addWidget(tabs)
        else:
            cl.addWidget(self.cross)
        lay.addWidget(card, 0)          # the schematic sizes itself: extra height is letterbox
        lay.addWidget(self.consequences, 0)

        # Two labels, not one: a BLOCK and a WARN are not the same news and must not be the
        # same colour. Both are on screen; the Run button's tooltip is a copy, not the source.
        self.guard_block = WrapLabel("")
        self.guard_warn = WrapLabel("")
        for lab, state in ((self.guard_block, "fail"), (self.guard_warn, "warn")):
            lab.setProperty("state", state)
            lab.setVisible(False)
            lay.addWidget(lab, 0)

        self.dry = QPlainTextEdit()
        self.dry.setObjectName("log")
        self.dry.setReadOnly(True)
        self.dry.setPlaceholderText("Dry run prints the exact command here, and copies it.")
        self.dry.setMinimumHeight(90)
        lay.addWidget(self.dry, 1)
        return col

    def _build_action_row(self) -> QWidget:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        self.stage_boxes: dict[Stage, QCheckBox] = {}
        for st in (Stage.MESH, Stage.FORWARD, Stage.IMAGE, Stage.FIGURES):
            cb = QCheckBox(st.value)
            cb.setChecked(True)
            cb.setToolTip(STAGE_TIPS[st])
            cb.toggled.connect(self._on_change)
            self.stage_boxes[st] = cb
            lay.addWidget(cb)
            self._tab.append(cb)
        lay.addStretch(1)

        self.btn_dry = QPushButton("Dry run")
        self.btn_queue = QPushButton("Add to queue")
        self.btn_run = QPushButton("Run")
        self.btn_run.setObjectName("primary")
        self.btn_dry.clicked.connect(self._on_dry_run)
        self.btn_queue.clicked.connect(
            lambda: self.queue_requested.emit(self.config(), self.stages()))
        self.btn_run.clicked.connect(
            lambda: self.run_requested.emit(self.config(), self.stages()))
        for b in (self.btn_dry, self.btn_queue, self.btn_run):
            # Enter inside a spinbox must not start a run: nothing here is a default button.
            b.setAutoDefault(False)
            b.setDefault(False)
            lay.addWidget(b)
            self._tab.append(b)
        return row

    def _load_scenario_facts(self) -> None:
        """Fill the read-only group, and hand the same facts to the schematic.

        scenario.load() reads a cache and never touches Docker, so this is safe during window
        construction; refreshing from the container is an explicit action, not a startup cost.
        """
        scenario = _optional("model.scenario")
        rows: tuple[tuple[str, str], ...] = (("scenario", "model.scenario not available"),)
        if scenario is not None:
            try:
                sc = scenario.load()
                self.cross.set_facts(sc)
                rows = _scenario_rows(sc)
            except Exception as exc:              # a broken cache is not worth a dead window
                rows = (("error", "model.scenario failed: %s" % exc),)
        for name, text in rows:
            val = WrapLabel(str(text))
            val.setProperty("role", "caption")
            self._scenario_form.addRow(name, val)

    def _preset_names(self) -> list[str]:
        presets = _optional("model.presets")
        table = getattr(presets, "PRESETS", None) if presets else None
        self._presets = dict(table) if isinstance(table, dict) else {}
        return list(self._presets) or ["published baseline (+20 deg)"]

    # ---- state ---------------------------------------------------------------------

    def config(self) -> RunConfig:
        """The form as a RunConfig. Fields the form does not expose keep their base values."""
        return replace(
            self._base,
            angle=self.angle.value(),
            chain=self.chain.currentData(),
            scale=round(self.fineness.value(), 4),
            degree=self.degree.currentData(),
            h_notch=None if self.h_notch.value() == 0.0 else round(self.h_notch.value(), 4),
            quad=self.quad.isChecked(),
            staircase_id=self.staircase.isChecked(),
            notch=self.notch.currentData(),
            artifact_reduction=self.artifact.currentData(),
            device=self.device.currentData(),
            snapshots=self.snapshots.value(),
            # The UI is microseconds because that is the unit every arrival time in this
            # problem is quoted in; the contract stores seconds. Convert at the edge, here.
            t_end=self.t_end.value() / 1e6,
            cfl=round(self.cfl.value(), 4),
            snap_window=self.snap_window.value(),
        )

    def set_config(self, cfg: RunConfig) -> None:
        self._loading = True
        try:
            self._base = cfg
            self.angle.setValue(cfg.angle)
            self.chain.setCurrentIndex(max(0, self.chain.findData(cfg.chain)))
            self.fineness.setValue(cfg.scale)
            self.degree.setCurrentIndex(max(0, self.degree.findData(cfg.degree)))
            self.h_notch.setValue(0.0 if cfg.h_notch is None else cfg.h_notch)
            self.quad.setChecked(cfg.quad)
            self.staircase.setChecked(cfg.staircase_id)
            self.notch.setCurrentIndex(max(0, self.notch.findData(cfg.notch)))
            self.artifact.setCurrentIndex(
                max(0, self.artifact.findData(cfg.artifact_reduction)))
            self.device.setCurrentIndex(max(0, self.device.findData(cfg.device)))
            self.snapshots.setValue(cfg.snapshots)
            self.t_end.setValue(round(cfg.t_end * 1e6, 3))     # s -> us, display only
            self.cfl.setValue(cfg.cfl)
            self.snap_window.setValue(cfg.snap_window)
            match = next((n for n, c in self._presets.items() if c == cfg), None)
            if match:
                self._last_preset = match
            self.preset.setCurrentText(match or CUSTOM)
        finally:
            self._loading = False
        self._on_change()

    def stages(self) -> tuple[Stage, ...]:
        return tuple(st for st, cb in self.stage_boxes.items() if cb.isChecked())

    def attach_runner(self, runner: object = None) -> list[str]:
        """main.py's wiring hook, forwarded to the embedded queue.

        The queue used to be its own rail section and main.py attached the runner to it
        directly. Now that it lives here, this view is the only thing main.py can see - but
        the runner still belongs to the shell, and this view still never starts a job.
        """
        if self.queue is None:
            return []
        return self.queue.attach_runner(runner)

    # ---- reactions -----------------------------------------------------------------

    def _sync_preset_label(self, cfg: RunConfig) -> None:
        """A combo still reading "Published +20 deg" over edited fields is a lie, so any
        divergence flips it to (custom). Reset is then the way back."""
        name = self.preset.currentText()
        if name != CUSTOM:
            known = self._presets.get(name)
            if known is not None and known != cfg:
                self._loading = True
                self.preset.setCurrentText(CUSTOM)
                self._loading = False
        self.btn_reset.setEnabled(self.preset.currentText() == CUSTOM
                                  and self._last_preset is not None)
        self.btn_reset.setToolTip("back to %s" % self._last_preset if self._last_preset
                                  else "no named preset to go back to")

    def _on_preset(self, name: str) -> None:
        if self._loading or name == CUSTOM:
            return
        cfg = self._presets.get(name)
        if isinstance(cfg, RunConfig):
            self._last_preset = name
            self.set_config(cfg)

    def _on_reset(self) -> None:
        cfg = self._presets.get(self._last_preset or "")
        if isinstance(cfg, RunConfig):
            self.set_config(cfg)

    def _on_change(self, *_args) -> None:
        if self._loading:
            return
        cfg = self.config()
        # The animation stage exists only if the solve wrote snapshots to animate, and the
        # snapshot window means nothing without snapshots either.
        fig = self.stage_boxes.get(Stage.FIGURES)
        if fig is not None:
            fig.setEnabled(cfg.snapshots > 0)
            fig.setToolTip("needs snapshots > 0" if not cfg.snapshots
                           else STAGE_TIPS[Stage.FIGURES])
        self.snap_window.setEnabled(cfg.snapshots > 0)
        self.cross.set_config(cfg)
        self.consequences.set_config(cfg)
        lines = _consequence_lines(cfg)
        for name, row in self._rows.items():
            row.set_consequence(lines.get(name, ""))
        self._update_dots(cfg)
        self._update_guards(cfg)
        self._sync_preset_label(cfg)
        self.config_changed.emit(cfg)

    def _update_dots(self, cfg: RunConfig) -> None:
        presets = _optional("model.presets")
        fn = getattr(presets, "diff_vs_published", None) if presets else None
        diff: dict = {}
        if callable(fn):
            try:
                d = fn(cfg)
                diff = d if isinstance(d, dict) else {k: () for k in d}
            except Exception:
                diff = {}                  # a broken diff must not paint dots at random
        for name, row in self._rows.items():
            pair = diff.get(name)
            tip = ""
            if isinstance(pair, tuple) and len(pair) == 2:
                tip = "differs from the published baseline - published: %s (this run: %s)" % (
                    _show_value(pair[1]), _show_value(pair[0]))
            row.set_dot(name in diff, tip)

    def _update_guards(self, cfg: RunConfig) -> None:
        """Run is disabled by a BLOCK, and the reason is on screen next to it - never a
        silently dead button. Each finding also lands on the field it names, because a message
        about nodes per wavelength is only actionable beside the control that sets them.
        """
        guards = _optional("model.guards")
        scenario = _optional("model.scenario")
        blocks: list[str] = []
        warns: list[str] = []
        per_field: dict[str, tuple[str, str]] = {}
        if guards is not None and scenario is not None:
            try:
                # `tracked` is left empty on purpose: resolving it means a git call per output
                # path on every keystroke, and the runner does that check hard before it starts
                # a container. This panel is the early warning, not the last line of defence.
                ctx = guards.Context(free_bytes=_free_bytes())
                for f in guards.check(cfg, scenario.load(), ctx) or ():
                    level = str(getattr(f, "level", getattr(f, "severity", "warn"))).upper()
                    msg = str(getattr(f, "message", f))
                    block = "BLOCK" in level
                    (blocks if block else warns).append(msg)
                    # First finding per field wins: check() sorts blocks first, and the block
                    # is the one that has to be readable.
                    per_field.setdefault(str(getattr(f, "field", "")),
                                         ("BLOCK: " + msg if block else msg,
                                          "fail" if block else "warn"))
            except Exception as exc:
                warns.append("model.guards failed: %s" % exc)
        self.btn_run.setEnabled(not blocks)
        self.btn_queue.setEnabled(not blocks)
        self.guard_block.setText("\n".join("BLOCK: " + b for b in blocks))
        self.guard_block.setVisible(bool(blocks))
        self.guard_warn.setText("\n".join("warning: " + w for w in warns))
        self.guard_warn.setVisible(bool(warns))
        self.btn_run.setToolTip(self.guard_block.text() if blocks
                                else "Run the selected stages now")
        default = RunConfig()
        for name, row in self._rows.items():
            found = per_field.get(name)
            if found:
                row.set_alert(*found)
            elif row.warn and getattr(cfg, name) != getattr(default, name):
                # A `warn: True` parameter sitting on its default is unremarkable; moved off
                # it, it deserves to say so even when no guardrail fires. These three are the
                # ones people turn without reading the detail first.
                row.set_alert("caution: not the published value - read the ? before trusting "
                              "this run", "warn")
            else:
                row.set_alert("")

    def guard_text(self) -> str:
        """Everything the guardrails are saying, as one string. For the self-check."""
        return "\n".join(t for t in (self.guard_block.text(), self.guard_warn.text()) if t)

    def _on_dry_run(self) -> None:
        text = self.dry_run_text()
        self.dry.setPlainText(text)
        from PySide6.QtGui import QGuiApplication
        cb = QGuiApplication.clipboard()
        if cb is not None:
            cb.setText(text)

    def dry_run_text(self) -> str:
        """The exact commands this configuration would issue, one per stage.

        core/docker.py owns the host side of the command - the image, the mounts, the env -
        and reads them from docker/run.ps1 -PrintArgs, so nothing is hardcoded here. Until it
        exists, show the in-container argv, which is the half this view is responsible for.
        """
        cfg = self.config()
        jobs = plan(cfg, self.stages())
        docker = _optional("core.docker")
        contract = None
        docker_note = ""
        if docker is not None:
            try:
                contract = docker.container_contract()
            except Exception as exc:
                # No Docker on this machine is a normal state for a dry run: say so and still
                # print the half of the command this view is responsible for.
                contract = None
                docker_note = "# container contract unavailable: %s" % exc
        lines = ["# tag %s   mesh %s" % (cfg.tag(), cfg.mesh_name()),
                 "# the image stage beamforms our own channel data only"]
        if docker_note:
            lines.append(docker_note)
        for job in jobs:
            lines.append("# %s" % job.label)
            if contract is not None:
                argv = docker.build_argv(job, contract, "gui_dryrun_" + job.stage.value)
                lines.append("  " + " ".join(argv))
            else:
                lines.append("  # in-container argv only")
                lines.append("  " + " ".join(job.argv))
        return "\n".join(lines)


def demo() -> None:
    """Self-check: every parameter is on screen, the form round-trips, and a BLOCK kills Run."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    v = SimulateView()
    v.resize(1100, 720)
    v.show()
    app.processEvents()

    # Every field in the contract has a row, so nothing is tunable-but-invisible.
    want = {f.name for f in fields(RunConfig)}
    assert set(v._rows) == want, sorted(set(v._rows) ^ want)
    # The k-Wave comparison is not a GUI concept: it must not come back as a field.
    assert not any("compare" in n or "kwave" in n for n in v._rows), sorted(v._rows)

    # Every row must SHOW its copy, not hide it in a tooltip. That is the whole point.
    for name, row in v._rows.items():
        h = help_for(name)
        assert row.label.text() == h.label, (name, row.label.text())
        assert row.hint.text(), name
        assert row.flag.text() == h.flag, name

    # Defaults must be the published configuration, untouched by the widgets' own ranges.
    assert v.config() == RunConfig(), (v.config(), RunConfig())
    # Nothing may block a default configuration: the form has to be live on startup.
    assert v.btn_run.isEnabled() and v.btn_queue.isEnabled(), v.guard_text()
    assert not v.guard_block.isVisible(), v.guard_block.text()

    for cfg in (
        replace(RunConfig(), angle=-35.0, scale=0.5, degree=3, h_notch=0.15, quad=False,
                notch=Notch.FILLED, staircase_id=True,
                artifact_reduction=ArtifactReduction.SPONGE,
                device=Device.CPU, snapshots=240, chain="legacy"),
        replace(RunConfig(), artifact_reduction=ArtifactReduction.WIDE_DOMAIN),
        replace(RunConfig(), t_end=3.0e-6, cfl=0.25, snap_window=None),
        replace(RunConfig(), t_end=45.5e-6, snap_window=(20.0, 40.0)),
        RunConfig(),
    ):
        v.set_config(cfg)
        assert v.config() == cfg, (v.config(), cfg)

    # Mesh fineness: the named choices hide the number, Custom reveals it.
    v.set_config(replace(RunConfig(), scale=0.71))
    assert v.fineness.spin.isVisible() and v.config().scale == 0.71
    v.set_config(RunConfig())
    assert not v.fineness.spin.isVisible(), "0.8 is a named choice, not a custom value"

    # The widened domain must reach the schematic, not just the config.
    v.set_config(replace(RunConfig(), artifact_reduction=ArtifactReduction.WIDE_DOMAIN))
    assert v.cross.extent() == (WIDE_X_MIN, WIDE_X_MAX)
    v.set_config(RunConfig())

    # THE CFL TRAP, which is why the consequences are inline: refining the crack cells
    # multiplies the step count for the whole model, and nothing used to say so on screen.
    before = v._rows["h_notch"].consequence_text()
    v.set_config(replace(RunConfig(), h_notch=0.09))
    after = v._rows["h_notch"].consequence_text()
    if before:                     # only meaningful with model.derived present
        assert "steps" in before and after != before, (before, after)
        assert v._rows["h_notch"].alert_text().startswith("caution"), "warn: True, off default"
    v.set_config(RunConfig())
    assert not v._rows["h_notch"].alert_text(), "the default must not be flagged"

    # Record length is microseconds on screen and seconds in the contract.
    assert abs(v.t_end.value() - 60.0) < 1e-9, v.t_end.value()
    v.t_end.setValue(30.0)
    assert abs(v.config().t_end - 30.0e-6) < 1e-15, v.config().t_end
    v.set_config(RunConfig())

    # Presets: divergence reads as (custom), and the way back is a button.
    if v._presets:
        first = next(iter(v._presets))
        v.preset.setCurrentText(first)
        assert v.preset.currentText() == first
        v.angle.setValue(v.angle.value() + 5.0)
        assert v.preset.currentText() == CUSTOM, v.preset.currentText()
        assert v.btn_reset.isEnabled()
        v.btn_reset.click()
        assert v.config() == v._presets[first], "reset must restore the preset exactly"
    v.set_config(RunConfig())

    txt = v.dry_run_text()
    for expect in ("ili_mesh.py", "ili_forward.py", "compare_images.py", "--no-overlay"):
        assert expect in txt, expect
    assert "--theirs" not in txt, "the GUI must never request a k-Wave comparison"
    assert v.stages() == (Stage.MESH, Stage.FORWARD, Stage.IMAGE, Stage.FIGURES), v.stages()
    v.stage_boxes[Stage.MESH].setChecked(False)
    assert Stage.MESH not in v.stages() and "ili_mesh.py" not in v.dry_run_text()
    v.stage_boxes[Stage.MESH].setChecked(True)

    # A BLOCK must disable Run AND put its reason where the user can read it - never a button
    # that is dead for reasons the app kept to itself. Forced through the disk rule, the one
    # this form can provoke without touching the filesystem.
    real_free = globals()["_free_bytes"]
    globals()["_free_bytes"] = lambda: 10 ** 6
    try:
        v._on_change()
        assert not v.btn_run.isEnabled() and not v.btn_queue.isEnabled()
        assert v.guard_block.isVisible() and v.guard_block.text().startswith("BLOCK")
        assert v.btn_run.toolTip() == v.guard_block.text()
        assert v._rows["snapshots"].alert_text().startswith("BLOCK"), "on the field it names"
    finally:
        globals()["_free_bytes"] = real_free
    v._on_change()
    assert v.btn_run.isEnabled(), v.guard_text()

    # A WARN must be readable, must reach its own field, and must disable nothing.
    v.set_config(replace(RunConfig(), scale=2.0))
    assert v.guard_warn.isVisible() and v.btn_run.isEnabled()
    assert "nodes per wavelength" in v.guard_warn.text(), v.guard_warn.text()
    assert v._rows["scale"].alert_text(), "the fineness row must carry its own warning"
    v.set_config(RunConfig())

    # Advanced opens collapsed, and a collapsed group says so in its title.
    adv = v._rows["cfl"].parentWidget()
    assert not adv.isVisible(), "Advanced opens collapsed"
    assert adv.parentWidget().title().startswith("[+]"), adv.parentWidget().title()

    got = []
    v.run_requested.connect(lambda c, s: got.append((c, s)))
    v.btn_run.click()
    assert got and got[0][0] == v.config(), "run_requested did not carry the config"

    # The queue is on this page now, and attach_runner must reach it. Checked against a stub:
    # the real Runner would be a second queue fighting for the GPU, and no demo may start one.
    assert v.queue is not None, "views.queue failed to import - the embedded queue is missing"
    assert v.queue.isVisibleTo(v), "the queue must be on screen, not merely constructed"
    assert v.queue.minimumHeight() > 0, "a collapsed-to-nothing queue is a forgotten queue"
    assert "0 queued" in v.queue._summary.text(), v.queue._summary.text()

    class _StubRunner(QObject):
        queue_changed = Signal()

        def jobs(self):
            return []

    stub = _StubRunner()
    assert v.attach_runner(stub) == ["queue_changed"], "attach_runner did not reach the queue"
    from views.queue import fake_jobs
    v.queue.set_jobs(fake_jobs())
    assert "1 running" in v.queue._summary.text(), v.queue._summary.text()
    print("simulate.demo: ok (%d parameters, %d with a detail expander)"
          % (len(v._rows), sum(1 for n in v._rows if help_for(n).detail)))


if __name__ == "__main__":
    demo()
