r"""The parameter form: everything the user can set, and what it will cost.

Two columns. Left is the form, grouped so the whole configuration is legible without
scrolling past anything that matters. Right is the consequence of the form: the schematic
above, the derived numbers below. Change a field and both move.

This view NEVER runs anything. It emits run_requested / queue_requested with a RunConfig and
the stages to run; main.py wires those to core/runner.py (W1). That split is what keeps the
form testable with no Docker on the machine, and it is why "Dry run" is local: it only prints
the command that would be issued.

Field defaults come from model/spec.py and nowhere else, so the form opens on the published
+20 deg configuration.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):     # direct run: make src/gui the import root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPlainTextEdit, QPushButton, QRadioButton, QScrollArea,
                               QSizePolicy, QSpinBox, QSplitter, QVBoxLayout, QWidget)

from model.spec import (ArtifactReduction, Device, Notch, RunConfig, SPONGE_DB, SPONGE_MM,
                        Stage, WIDE_X_MAX, WIDE_X_MIN, plan)
from widgets.consequences import Consequences
from widgets.crosssection import DEFAULT_FACTS, CrossSection

# Read-only scenario facts, mirrored from repro/ili_forward.py (materials) and mesh/ili_mesh.py
# (geometry). Geometry and materials are backend CONSTANTS, not flags - README phase-1 scope
# defers editing them - so this panel exists to state them, not to change them. model/scenario.py
# (W2) replaces these with the backend's own dump when it lands.
SCENARIO_FALLBACK: tuple[tuple[str, str], ...] = (
    ("pipe", "ID r 193.675 / OD r 203.200 mm, wall 9.525 mm"),
    ("standoff", "20.0 mm water on the beam axis"),
    ("array", "256 el @ 0.30 mm = 76.5 mm aperture at z = 0"),
    ("pulse", "1-cycle toneburst, f0 4.0 MHz"),
    ("steel", "c_P 5700, c_S 3100 m/s, rho 7850 kg/m3"),
    ("water", "c 1500 m/s, rho 1000 kg/m3"),
    ("notch", "4.0 x 1.0 mm slot at x = 38.25 mm, from the OD inward"),
)

_ARTIFACT_CHOICES: tuple[tuple[ArtifactReduction, str, str], ...] = (
    (ArtifactReduction.NONE, "None",
     "no workaround; what every published figure uses  (--abc-legacy)"),
    (ArtifactReduction.SPONGE, "Sponge layer",
     "shear-matched boundary + graded damping  (--sponge-mm %.1f --sponge-db %.1f)"
     % (SPONGE_MM, SPONGE_DB)),
    (ArtifactReduction.WIDE_DOMAIN, "Widened domain",
     "boundary moved out of reach  (mesh --x-min %.0f --x-max %.0f)" % (WIDE_X_MIN, WIDE_X_MAX)),
)

CUSTOM = "(custom)"


def _optional(mod: str):
    """Import a sibling module that another workstream owns, or None. The four streams land in
    any order, so nothing here may hard-depend on a module that is not written yet."""
    try:
        return __import__(mod, fromlist=["_"])
    except ImportError:
        return None


def _first_callable(mod, *names):
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    return None


class _Group(QGroupBox):
    """Collapsible group. Qt's checkable group box is the collapse toggle - no custom header,
    no animation. Hiding the body rather than the box keeps the title clickable."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.body = QWidget(self)
        self.form = QFormLayout(self.body)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(6)
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        outer.addWidget(self.body)
        self.toggled.connect(self.body.setVisible)

    def row(self, label: str, widget: QWidget, dot: QWidget | None = None) -> None:
        if dot is None:
            self.form.addRow(label, widget)
            return
        line = QWidget()
        lay = QHBoxLayout(line)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(widget, 1)
        lay.addWidget(dot, 0)
        self.form.addRow(label, line)


class SimulateView(QWidget):
    """The form. Owns no runner and no Docker; it only describes a run."""

    run_requested = Signal(object, object)      # (RunConfig, tuple[Stage, ...])
    queue_requested = Signal(object, object)
    config_changed = Signal(object)             # (RunConfig) - for the status bar / other views

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._base = RunConfig()          # carries the fields this form does not expose
        self._loading = False            # suppress the change cascade while populating widgets
        self._dots: dict[str, QLabel] = {}

        self.cross = CrossSection()
        self.consequences = Consequences()

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_form_column())
        split.addWidget(self._build_result_column())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 4)
        root.addWidget(split, 1)
        root.addWidget(self._build_action_row(), 0)

        self._load_scenario_facts()
        self.set_config(self._base)

    # ---- construction --------------------------------------------------------------

    def _dot(self, field: str) -> QLabel:
        """The marker that says "this differs from the published baseline"."""
        d = QLabel()
        d.setObjectName("dot")
        d.setProperty("on", "false")
        d.setToolTip("differs from the published baseline")
        self._dots[field] = d
        return d

    def _build_form_column(self) -> QWidget:
        col = QWidget()
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.preset = QComboBox()
        self.preset.addItems(self._preset_names() + [CUSTOM])
        self.preset.currentTextChanged.connect(self._on_preset)
        head = QFormLayout()
        head.addRow("Preset", self.preset)
        lay.addLayout(head)

        # --- Beam
        g = _Group("Beam")
        self.angle = QDoubleSpinBox()
        self.angle.setRange(-60.0, 60.0)
        self.angle.setDecimals(1)
        self.angle.setSingleStep(5.0)
        self.angle.setSuffix(" deg")
        self.chain = QComboBox()
        # faithfulbf is the published baseline; legacy is frozen and kept only to reproduce
        # figures published before 2026-08-19 (see AGENTS.md).
        self.chain.addItems(["faithfulbf", "legacy"])
        g.row("angle", self.angle, self._dot("angle"))
        g.row("imaging chain", self.chain, self._dot("chain"))
        lay.addWidget(g)

        # --- Mesh
        g = _Group("Mesh")
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.20, 2.00)
        self.scale.setDecimals(2)
        self.scale.setSingleStep(0.05)
        self.scale.setToolTip("multiplies every target cell size; smaller is finer")
        self.degree = QSpinBox()
        self.degree.setRange(1, 6)
        self.h_notch = QDoubleSpinBox()
        self.h_notch.setRange(0.0, 2.0)
        self.h_notch.setDecimals(2)
        self.h_notch.setSingleStep(0.05)
        # 0.0 means "let the mesher decide", i.e. h_notch = None in the config.
        self.h_notch.setSpecialValueText("auto")
        self.h_notch.setSuffix(" mm")
        self.quad = QCheckBox("quadrilaterals (exact row-sum lumping)")
        # A separate axis from `notch`: --staircase rasterises the INNER WALL ARC, and the C4
        # experiment ran it on a healthy wall. See RunConfig.staircase_id.
        self.staircase = QCheckBox("staircase the ID arc onto 50 um pixels")
        self.notch = QComboBox()
        for n in Notch:
            self.notch.addItem(n.value, n)
        g.row("scale", self.scale, self._dot("scale"))
        g.row("degree", self.degree, self._dot("degree"))
        g.row("h at notch", self.h_notch, self._dot("h_notch"))
        g.row("elements", self.quad, self._dot("quad"))
        g.row("notch", self.notch, self._dot("notch"))
        g.row("ID arc", self.staircase, self._dot("staircase_id"))
        lay.addWidget(g)

        # --- Domain: three fixed treatments, NONE default. The knobs are deliberately not
        # exposed - tuning a boundary treatment against an image is how you invent an artefact.
        g = _Group("Domain")
        self.artifact_group = QButtonGroup(self)
        for i, (opt, title, why) in enumerate(_ARTIFACT_CHOICES):
            rb = QRadioButton(title)
            rb.setToolTip(why)
            self.artifact_group.addButton(rb, i)
            note = QLabel(why)
            note.setProperty("role", "caption")
            note.setWordWrap(True)
            note.setMinimumWidth(1)     # a wrapped label must not set the column's width
            g.row("" if i else "artifact reduction", rb,
                  self._dot("artifact_reduction") if i == 0 else None)
            g.form.addRow("", note)
        self.artifact_group.idToggled.connect(lambda _i, on: on and self._on_change())
        lay.addWidget(g)

        # --- Compute
        g = _Group("Compute")
        self.device_group = QButtonGroup(self)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        for i, dev in enumerate((Device.GPU, Device.CPU)):
            rb = QRadioButton(dev.value.upper())
            self.device_group.addButton(rb, i)
            rl.addWidget(rb)
        rl.addStretch(1)
        self.device_group.idToggled.connect(lambda _i, on: on and self._on_change())
        self.snapshots = QSpinBox()
        self.snapshots.setRange(0, 600)
        self.snapshots.setSingleStep(60)
        self.snapshots.setSpecialValueText("off")
        self.snapshots.setToolTip("240 snapshots writes 700-970 MB per run")
        g.row("device", row, self._dot("device"))
        g.row("snapshots", self.snapshots, self._dot("snapshots"))
        lay.addWidget(g)

        # --- Scenario: read-only. These are backend constants, not parameters.
        g = _Group("Scenario (read-only)")
        self._scenario_form = g.form
        lay.addWidget(g)
        self._scenario_group = g

        lay.addStretch(1)
        for w in (self.angle, self.scale, self.h_notch):
            w.valueChanged.connect(self._on_change)
        for w in (self.degree, self.snapshots):
            w.valueChanged.connect(self._on_change)
        for w in (self.chain, self.notch):
            w.currentIndexChanged.connect(self._on_change)
        for w in (self.quad, self.staircase):
            w.toggled.connect(self._on_change)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # No horizontal scrolling: the column takes the viewport width, which is what makes the
        # wrapped explanatory captions wrap instead of widening the form.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(col)
        return scroll

    def _build_result_column(self) -> QWidget:
        col = QWidget()
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        card = QFrame()
        card.setProperty("role", "card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(1, 1, 1, 1)
        cl.addWidget(self.cross)
        lay.addWidget(card, 0)          # the schematic sizes itself: extra height is letterbox
        lay.addWidget(self.consequences, 0)

        self.guard_label = QLabel("")
        self.guard_label.setWordWrap(True)
        self.guard_label.setProperty("state", "warn")
        lay.addWidget(self.guard_label, 0)

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
            cb.toggled.connect(self._on_change)
            self.stage_boxes[st] = cb
            lay.addWidget(cb)
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
            lay.addWidget(b)
        return row

    def _load_scenario_facts(self) -> None:
        """Fill the read-only group, and hand the same facts to the schematic."""
        scenario = _optional("model.scenario")
        rows = SCENARIO_FALLBACK
        if scenario is not None:
            fn = _first_callable(scenario, "facts", "scenario", "load", "read")
            if fn is not None:
                try:
                    facts = fn()
                    self.cross.set_facts(facts)
                    described = _first_callable(scenario, "describe", "summary")
                    if described is not None:
                        rows = tuple(described())
                except Exception as exc:          # a scenario dump that fails is not fatal
                    rows = SCENARIO_FALLBACK + (("error", "%s" % exc),)
        for name, text in rows:
            val = QLabel(str(text))
            val.setProperty("role", "caption")
            val.setWordWrap(True)
            val.setMinimumWidth(1)
            self._scenario_form.addRow(name, val)

    def _preset_names(self) -> list[str]:
        presets = _optional("model.presets")
        table = getattr(presets, "PRESETS", None) if presets else None
        self._presets = dict(table) if isinstance(table, dict) else {}
        return list(self._presets) or ["published baseline (+20 deg)"]

    # ---- state ---------------------------------------------------------------------

    def config(self) -> RunConfig:
        """The form as a RunConfig. Fields the form does not expose keep their base values."""
        art = _ARTIFACT_CHOICES[max(0, self.artifact_group.checkedId())][0]
        dev = (Device.GPU, Device.CPU)[max(0, self.device_group.checkedId())]
        return replace(
            self._base,
            angle=self.angle.value(),
            chain=self.chain.currentText(),
            scale=round(self.scale.value(), 4),
            degree=self.degree.value(),
            h_notch=None if self.h_notch.value() == 0.0 else self.h_notch.value(),
            quad=self.quad.isChecked(),
            staircase_id=self.staircase.isChecked(),
            notch=self.notch.currentData(),
            artifact_reduction=art,
            device=dev,
            snapshots=self.snapshots.value(),
        )

    def set_config(self, cfg: RunConfig) -> None:
        self._loading = True
        try:
            self._base = cfg
            self.angle.setValue(cfg.angle)
            self.chain.setCurrentText(cfg.chain)
            self.scale.setValue(cfg.scale)
            self.degree.setValue(cfg.degree)
            self.h_notch.setValue(0.0 if cfg.h_notch is None else cfg.h_notch)
            self.quad.setChecked(cfg.quad)
            self.staircase.setChecked(cfg.staircase_id)
            self.notch.setCurrentIndex(list(Notch).index(cfg.notch))
            self.artifact_group.button(
                [c[0] for c in _ARTIFACT_CHOICES].index(cfg.artifact_reduction)).setChecked(True)
            self.device_group.button(0 if cfg.device is Device.GPU else 1).setChecked(True)
            self.snapshots.setValue(cfg.snapshots)
        finally:
            self._loading = False
        self._on_change()

    def stages(self) -> tuple[Stage, ...]:
        return tuple(st for st, cb in self.stage_boxes.items() if cb.isChecked())

    # ---- reactions -----------------------------------------------------------------

    def _on_preset(self, name: str) -> None:
        if self._loading or name == CUSTOM:
            return
        cfg = self._presets.get(name)
        if isinstance(cfg, RunConfig):
            self.set_config(cfg)

    def _on_change(self, *_args) -> None:
        if self._loading:
            return
        cfg = self.config()
        # The animation stage exists only if the solve wrote snapshots to animate.
        fig = self.stage_boxes.get(Stage.FIGURES)
        if fig is not None:
            fig.setEnabled(cfg.snapshots > 0)
            fig.setToolTip("needs snapshots > 0" if not cfg.snapshots else "wavefield animation")
        self.cross.set_config(cfg)
        self.consequences.set_config(cfg)
        self._update_dots(cfg)
        self._update_guards(cfg)
        self.config_changed.emit(cfg)

    def _update_dots(self, cfg: RunConfig) -> None:
        presets = _optional("model.presets")
        fn = getattr(presets, "diff_vs_published", None) if presets else None
        differing: set[str] = set()
        if callable(fn):
            try:
                d = fn(cfg)
                differing = set(d.keys()) if isinstance(d, dict) else set(d)
            except Exception:
                differing = set()          # a broken diff must not paint dots at random
        for field, dot in self._dots.items():
            dot.setProperty("on", "true" if field in differing else "false")
            dot.style().unpolish(dot)      # dynamic properties need a restyle to take effect
            dot.style().polish(dot)

    def _update_guards(self, cfg: RunConfig) -> None:
        """Run is disabled by a BLOCK, and the reason is on screen next to it - never a
        silently dead button."""
        guards = _optional("model.guards")
        fn = _first_callable(guards, "check", "evaluate", "check_config") if guards else None
        blocks: list[str] = []
        warns: list[str] = []
        if fn is not None:
            try:
                for item in fn(cfg) or ():
                    level = str(getattr(item, "level", item[0] if isinstance(item, tuple)
                                        else "warn"))
                    msg = str(getattr(item, "message", item[-1] if isinstance(item, tuple)
                                      else item))
                    (blocks if "BLOCK" in level.upper() else warns).append(msg)
            except Exception as exc:
                warns.append("model.guards failed: %s" % exc)
        self.btn_run.setEnabled(not blocks)
        self.btn_queue.setEnabled(not blocks)
        text = "  ".join("BLOCK: " + b for b in blocks) or "  ".join(warns)
        self.guard_label.setText(text)
        self.guard_label.setProperty("state", "fail" if blocks else "warn")
        self.guard_label.style().unpolish(self.guard_label)
        self.guard_label.style().polish(self.guard_label)
        self.btn_run.setToolTip(text if blocks else "Run the selected stages now")

    def _on_dry_run(self) -> None:
        text = self.dry_run_text()
        self.dry.setPlainText(text)
        from PySide6.QtGui import QGuiApplication
        cb = QGuiApplication.clipboard()
        if cb is not None:
            cb.setText(text)

    def dry_run_text(self) -> str:
        """The exact commands this configuration would issue, one per stage.

        core/docker.py (W1) owns the host side of the command - the image, the mounts, the env -
        and reads them from docker/run.ps1 -PrintArgs, so nothing is hardcoded here. Until it
        exists, show the in-container argv, which is the half this view is responsible for.
        """
        cfg = self.config()
        jobs = plan(cfg, kwave_case=None, stages=self.stages())
        docker = _optional("core.docker")
        fn = _first_callable(docker, "command_for", "docker_argv", "argv_for") if docker else None
        lines = ["# %s   tag %s" % (cfg.tag(), cfg.mesh_name())]
        for job in jobs:
            if fn is not None:
                try:
                    argv = fn(job)
                    lines.append(" ".join(argv) if isinstance(argv, (list, tuple)) else str(argv))
                    continue
                except Exception as exc:
                    lines.append("# core.docker failed: %s" % exc)
            lines.append("# %-7s (in-container argv; core.docker not available)" % job.stage.value)
            lines.append("  " + " ".join(job.argv))
        return "\n".join(lines)


def demo() -> None:
    """Self-check: the form round-trips a RunConfig and the dry run names every stage."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    v = SimulateView()
    v.resize(1100, 700)
    v.show()
    app.processEvents()

    # Defaults must be the published configuration, untouched by the widgets' own ranges.
    assert v.config() == RunConfig(), (v.config(), RunConfig())

    for cfg in (
        replace(RunConfig(), angle=-35.0, scale=0.5, degree=3, h_notch=0.15, quad=False,
                notch=Notch.FILLED, staircase_id=True,
                artifact_reduction=ArtifactReduction.SPONGE,
                device=Device.CPU, snapshots=240, chain="legacy"),
        replace(RunConfig(), artifact_reduction=ArtifactReduction.WIDE_DOMAIN),
        RunConfig(),
    ):
        v.set_config(cfg)
        assert v.config() == cfg, (v.config(), cfg)

    # The widened domain must reach the schematic, not just the config.
    v.set_config(replace(RunConfig(), artifact_reduction=ArtifactReduction.WIDE_DOMAIN))
    assert v.cross.extent() == (WIDE_X_MIN, WIDE_X_MAX)
    v.set_config(RunConfig())

    txt = v.dry_run_text()
    for expect in ("ili_mesh.py", "ili_forward.py", "compare_images.py", "--no-overlay"):
        assert expect in txt, expect
    assert v.stages() == (Stage.MESH, Stage.FORWARD, Stage.IMAGE, Stage.FIGURES), v.stages()
    v.stage_boxes[Stage.MESH].setChecked(False)
    assert Stage.MESH not in v.stages() and "ili_mesh.py" not in v.dry_run_text()

    # No guards module yet: Run must be live, not disabled by a rule nobody wrote.
    assert v.btn_run.isEnabled()

    got = []
    v.run_requested.connect(lambda c, s: got.append((c, s)))
    v.btn_run.click()
    assert got and got[0][0] == v.config(), "run_requested did not carry the config"
    print("simulate.demo: ok")


if __name__ == "__main__":
    demo()
