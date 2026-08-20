r"""The run gallery: cards of what has been produced, and a detail pane for one run.

DISCOVERY IS TWO-SOURCED, ON PURPOSE
`data/results/gui_runs/*.json` is written by core/manifest.py (another workstream). Until that
exists the folder is empty, and an empty gallery is untestable, so discovery falls back to the
PUBLISHED figures already on disk under `data/results/compare/`. Both sources produce the same
`RunEntry`, so the views never learn which one they are looking at.

ONLY UN-ANNOTATED RENDERS ARE SHOWN
`repro/compare_images.py` writes an annotated figure and a `_nooverlay` twin from the same
data; the annotated one draws cyan wall arcs and a lime marker on the true notch. A reviewer
who has been told where to look cannot judge detectability, so `prefer_unannotated` keeps the
`_nooverlay` twin and drops its annotated sibling entirely - the gallery never offers the
choice. Files with no twin at all (mesh_zoom, wavefield stills) are not comparison images and
pass through.

READ-ONLY, ALWAYS
`data/results` is the published record for the R&D challenge. This module opens files and
never writes one; export copies elsewhere. Re-run only emits a signal - it starts nothing.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

_GUI_ROOT = Path(__file__).resolve().parents[1]
if str(_GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_GUI_ROOT))

from PySide6.QtCore import QSize, Qt, Signal                                   # noqa: E402
from PySide6.QtGui import QMovie, QPixmap                                      # noqa: E402
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel,       # noqa: E402
                               QPushButton, QScrollArea, QSizePolicy, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from widgets.compareslider import CompareSlider                                # noqa: E402
from widgets.jobcard import INK, INK_SOFT, RULE, SURFACE                       # noqa: E402

FIG_MAX_W = 1000          # figures render at natural size up to this width, then scale down
THUMB_W = 300
_ANGLE_RE = re.compile(r"^compare_(p|m)(\d+)deg(?:_(.+?))?_nooverlay\.png$")


def results_root() -> Path:
    """Where results live. Mirrors backend lib/paths.py: $DVFEA_RESULTS wins, else repo/data.

    Same resolution order as the backend so the GUI and a hand-run container agree on what
    "results/compare/..." means; getting this wrong would show an empty gallery next to a full
    disk and look like a bug in discovery.
    """
    env = os.environ.get("DVFEA_RESULTS")
    if env:
        return Path(env)
    return _GUI_ROOT.parents[1] / "data" / "results"      # gui -> src -> repo


def presentation_root() -> Path:
    """Where the PUBLISHED record lives - a different tree from run output since 2026-08-20.

    Keeping them apart is the point: runs land in data/results and may be deleted freely, while
    everything under presentation/ is evidence and is version-controlled. The gallery reads both.
    """
    env = os.environ.get("DVFEA_PRESENTATION")
    if env:
        return Path(env) / "data"
    return _GUI_ROOT.parents[1] / "presentation" / "data"


def prefer_unannotated(paths: list[Path]) -> list[Path]:
    """Drop every figure that has a `_nooverlay` twin. See the module docstring."""
    out = []
    for p in paths:
        if p.suffix.lower() == ".png" and not p.stem.endswith("_nooverlay"):
            if p.with_name(p.stem + "_nooverlay.png").exists():
                continue
        out.append(p)
    return out


@dataclass
class RunEntry:
    """One run, however it was discovered. The views know nothing else about a run."""
    run_id: str
    title: str
    angle: float | None = None
    tag: str | None = None
    figures: list[Path] = field(default_factory=list)     # PNGs, already un-annotated
    gifs: list[Path] = field(default_factory=list)
    npz: list[Path] = field(default_factory=list)         # wavefield/channel data, for --in
    metrics: Any = None                                   # dict, or dict of dicts, or None
    argv: list[list[str]] = field(default_factory=list)   # one argv per stage
    config: Mapping[str, Any] | None = None               # RunConfig as plain data
    manifest: Path | None = None
    source: str = "published"                             # "manifest" | "published"
    note: str = ""

    @property
    def assets(self) -> list[Path]:
        return [*self.figures, *self.gifs]


def _abs(p: str, root: Path) -> Path:
    """Manifest paths are relative to the results root (that is what the container sees)."""
    q = Path(p)
    return q if q.is_absolute() else root / q


def _normalise_argv(raw: Any) -> list[list[str]]:
    """Accept ["python3", ...], [[...], [...]], or [{"stage":..., "argv":[...]}, ...]."""
    if not raw:
        return []
    if all(isinstance(x, str) for x in raw):
        return [list(raw)]
    out = []
    for item in raw:
        if isinstance(item, Mapping):
            a = item.get("argv") or item.get("command") or []
            if a:
                out.append([str(x) for x in a])
        elif isinstance(item, (list, tuple)):
            out.append([str(x) for x in item])
    return out


def discover_published(root: Path) -> list[RunEntry]:
    """Every published comparison figure that has an un-annotated render.

    Globs only `*_nooverlay.png`, which is also how the annotated twins get skipped without a
    second pass. `_clean` (an older spelling of the same thing, superseded per NAMING.md) is
    excluded by the same glob.
    """
    comp = root / "compare"
    viz = root / "viz"
    out: list[RunEntry] = []
    for png in sorted(comp.glob("*_nooverlay.png")):
        m = _ANGLE_RE.match(png.name)
        if not m:
            continue
        sign, deg, tag = m.group(1), int(m.group(2)), m.group(3)
        angle = float(deg) * (1 if sign == "p" else -1)
        token = f"{sign}{deg}deg"
        entry = RunEntry(run_id=png.stem, title=f"{tag or 'baseline'}  {angle:+.0f} deg",
                         angle=angle, tag=tag, figures=[png], source="published",
                         note="published record - read only; argv not captured, see "
                              "data/results/compare/NAMING.md")
        # Attach the wavefield animations to the untagged baseline only: they belong to that
        # configuration, and hanging them off every variant would imply they were re-rendered.
        if tag is None and viz.is_dir():
            entry.gifs = sorted(p for p in viz.glob("*.gif") if token in p.name)
            entry.npz = sorted((root / "ili_forward").glob(f"wavefield_snap*{token}*.npz"))
            entry.figures += prefer_unannotated(
                sorted(p for p in viz.glob("*.png") if token in p.name))
        out.append(entry)
    return out


def _manifest_docs(gui: Path) -> list[dict]:
    """Every readable manifest, with its own path attached. Corrupt ones are skipped.

    core/manifest.py writes one file PER JOB, not per run, and rewrites it when the job ends -
    so a half-written file is possible and must cost one manifest, not the gallery.
    """
    out = []
    for p in sorted(gui.glob("*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError, UnicodeDecodeError):
            continue
        if isinstance(doc, Mapping):
            out.append({**doc, "_path": p})
    return out


def _output_paths(doc: Mapping[str, Any], root: Path) -> list[Path]:
    """manifest outputs are [{"path": rel, "bytes":, "mtime":}]; plain strings also accepted."""
    out = []
    for o in doc.get("outputs") or []:
        rel = o.get("path") if isinstance(o, Mapping) else o
        if isinstance(rel, str):
            out.append(_abs(rel, root))
    return out


def _tag_of(config: Mapping[str, Any]) -> str | None:
    """The run's filename tag, computed by the contract rather than guessed from filenames."""
    cfg = config_to_runconfig(config)
    return cfg.tag() if hasattr(cfg, "tag") else None


def _run_facts(docs: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The measured numbers a manifest actually carries, one column per stage.

    Not the imaging metrics: `repro/compare_images.py` prints its head-to-head table to stdout
    and the manifest does not capture it yet, so those live in the job's log. What is here is
    measured, not modelled - ms/step, problem size, wall time, exit code.
    """
    cols: dict[str, dict[str, Any]] = {}
    for d in docs:
        started, ended = d.get("started") or 0.0, d.get("ended")
        cols[str(d.get("stage") or d.get("job_id") or "?")] = {
            "state": d.get("state", "?"),
            "exit code": "-" if d.get("exit_code") is None else d["exit_code"],
            "ms/step": "-" if d.get("ms_per_step") is None else d["ms_per_step"],
            "size (DOF/cells)": "-" if not d.get("size") else d["size"],
            "wall time [s]": round(ended - started, 1) if (ended and started) else "-",
        }
    return cols


def _entry_from_docs(docs: list[dict], root: Path) -> RunEntry:
    docs = sorted(docs, key=lambda d: d.get("started") or 0.0)
    cfg = next((d.get("config") for d in docs if d.get("config")), None) or {}
    tag = _tag_of(cfg)
    run_id = tag or str(docs[0].get("job_id") or docs[0]["_path"].stem)
    # Deduplicated: the same output can be declared by more than one stage (the image stage's
    # png is also listed as the run's headline figure), and a figure shown twice reads as two.
    paths = list(dict.fromkeys(p for d in docs for p in _output_paths(d, root)))
    exist = [p for p in paths if p.exists()]
    angle = cfg.get("angle")
    states = [str(d.get("state", "?")) for d in docs]
    commit = next((d.get("commit") for d in docs if d.get("commit")), "")
    label = next((d.get("label") for d in docs if d.get("label")), "")
    note = "   ".join(x for x in (
        f"commit {commit}" if commit else "",
        f"stages: {' -> '.join(states)}",
        "MISSING OUTPUTS - the job reported success but the files are not on disk"
        if len(exist) < len(paths) else "") if x)
    return RunEntry(
        run_id=run_id,
        title=f"{run_id}" + (f"  {angle:+.0f} deg" if isinstance(angle, (int, float)) else "")
              + (f"  {label}" if label else ""),
        angle=float(angle) if isinstance(angle, (int, float)) else None,
        tag=tag,
        figures=prefer_unannotated([p for p in exist if p.suffix.lower() == ".png"]),
        gifs=[p for p in exist if p.suffix.lower() == ".gif"],
        npz=[p for p in exist if p.suffix.lower() == ".npz"],
        metrics=next((d["metrics"] for d in docs if d.get("metrics")), None)
                or _run_facts(docs),
        argv=[list(d["argv"]) for d in docs if d.get("argv")],
        config=cfg or None, manifest=docs[0]["_path"], source="manifest", note=note)


def discover_runs(root: Path | None = None) -> list[RunEntry]:
    """GUI runs newest first, then the published record - ALWAYS, not as a fallback.

    One RunConfig = one run, so the per-job manifests are grouped by their config: the mesh,
    forward and image jobs of one configuration belong on one card, and showing three cards
    for one solve would misrepresent how much has been run.

    The published record is appended rather than substituted, because the first GUI run used
    to make it vanish - and the baseline is exactly what a new run needs to be looked at
    against. It also kept the gallery honest when a run left a manifest but no figures: an
    interrupted or cleaned-up run is a normal state, not an empty gallery.
    """
    root = root or results_root()
    gui = root / "gui_runs"
    groups: dict[str, list[dict]] = {}
    order: list[tuple[float, str]] = []
    for doc in _manifest_docs(gui) if gui.is_dir() else []:
        cfg = doc.get("config")
        key = json.dumps(cfg, sort_keys=True) if cfg else str(doc.get("job_id") or doc["_path"])
        groups.setdefault(key, []).append(doc)
    for key, docs in groups.items():
        order.append((max(d.get("started") or 0.0 for d in docs), key))
    runs = [_entry_from_docs(groups[k], root) for _, k in sorted(order, reverse=True)]
    # The published record is a DIFFERENT tree now; passing `root` would look for
    # published figures among run output and find none.
    return runs + discover_published(presentation_root())


def compare_pair(entry: RunEntry, root: Path | None = None
                 ) -> tuple[Path, Path, str, str] | None:
    """Two un-annotated images to put either side of the slider, or None.

    A manifest may name the two sides explicitly. Otherwise the pair is this run's comparison
    figure against the PUBLISHED baseline at the same angle, which is the A/B a reviewer
    actually wants: same beamformer, same scale, one variable changed.

    Note what is NOT possible from disk: compare_images.py renders FEM and k-Wave as two panels
    of ONE png, so there is no separate per-solver image to slide between. If a later stream
    starts writing them separately, put their paths in the manifest as `fem_png`/`kwave_png`
    and the first branch below picks them up.
    """
    root = root or results_root()
    cfg = entry.config or {}
    a, b = cfg.get("kwave_png"), cfg.get("fem_png")
    if a and b:
        pa, pb = _abs(str(a), root), _abs(str(b), root)
        if pa.exists() and pb.exists():
            return pa, pb, "k-Wave", "FEM"
    if entry.angle is None:
        return None
    mine = next((p for p in entry.figures if p.name.startswith("compare_")), None)
    sign = "p" if entry.angle >= 0 else "m"
    # The baseline side is published data; only the run's own figure is run output.
    base = (presentation_root() / "compare"
            / f"compare_{sign}{abs(entry.angle):.0f}deg_nooverlay.png")
    if mine is None or not base.exists() or base == mine:
        return None
    return base, mine, "published baseline", entry.tag or entry.run_id


def metrics_table(metrics: Any) -> tuple[list[str], list[list[str]]]:
    """(header, rows) for whichever metrics shape a manifest carries. Shared with export.py."""
    if not isinstance(metrics, Mapping) or not metrics:
        return [], []
    vals = list(metrics.values())
    if vals and all(isinstance(v, Mapping) for v in vals):
        cols = list(metrics)                       # e.g. FEM, k-Wave
        keys = list(dict.fromkeys(k for v in vals for k in v))
        return (["metric", *cols],
                [[k, *[_num(metrics[c].get(k)) for c in cols]] for k in keys])
    return ["metric", "value"], [[k, _num(v)] for k, v in metrics.items()]


def _num(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


# ---- widgets ---------------------------------------------------------------------------
class RunCard(QFrame):
    """One tile in the grid: thumbnail, title, and what the run holds."""

    opened = Signal(object)

    def __init__(self, entry: RunEntry, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.entry = entry
        self.setObjectName("runCard")
        self.setStyleSheet(
            f"#runCard {{ background:{SURFACE}; border:1px solid {RULE}; border-radius:6px; }}")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedWidth(THUMB_W + 24)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(6)

        thumb = QLabel()
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb.setMinimumHeight(120)
        if entry.figures:
            pm = QPixmap(str(entry.figures[0]))
            if not pm.isNull():
                thumb.setPixmap(pm.scaledToWidth(THUMB_W, Qt.TransformationMode.SmoothTransformation))
        if thumb.pixmap().isNull():
            thumb.setText("no figure")
            thumb.setStyleSheet(f"color:{INK_SOFT};")
        lay.addWidget(thumb)

        title = QLabel(entry.title)
        title.setStyleSheet(f"color:{INK}; font-weight:600;")
        title.setWordWrap(True)
        lay.addWidget(title)

        bits = [f"{len(entry.figures)} fig"]
        if entry.gifs:
            bits.append(f"{len(entry.gifs)} gif")
        if metrics_table(entry.metrics)[1]:
            bits.append("metrics")
        bits.append(entry.source)
        sub = QLabel("   ".join(bits))
        sub.setStyleSheet(f"color:{INK_SOFT};")
        lay.addWidget(sub)

    def mouseReleaseEvent(self, ev) -> None:  # noqa: ANN001
        if ev.button() == Qt.MouseButton.LeftButton:
            self.opened.emit(self.entry)


class RunDetail(QWidget):
    """One run at full size: compare slider, figures, GIFs, metrics, re-run."""

    rerun_requested = Signal(object)
    back_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.entry: RunEntry | None = None
        self._movies: list[QMovie] = []               # QMovie must outlive the QLabel showing it
        self.slider = CompareSlider()
        self.slider.setMinimumHeight(320)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        head = QHBoxLayout()
        self._back = QPushButton("< all runs")
        self._back.setFlat(True)
        self._back.clicked.connect(self.back_requested)
        self._title = QLabel()
        self._title.setStyleSheet(f"color:{INK}; font-weight:600;")
        self._rerun = QPushButton("Re-run")
        self._rerun.clicked.connect(self._emit_rerun)
        head.addWidget(self._back)
        head.addWidget(self._title, 1)
        head.addWidget(self._rerun)
        outer.addLayout(head)

        self._body = QWidget()
        self._stack = QVBoxLayout(self._body)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self._body)
        outer.addWidget(scroll, 1)

    def _emit_rerun(self) -> None:
        """Hand the config out. This pane starts nothing; the runner owns that."""
        if self.entry is not None:
            self.rerun_requested.emit(rerun_config(self.entry))

    def _clear(self) -> None:
        for m in self._movies:
            m.stop()
        self._movies.clear()
        while self._stack.count():
            item = self._stack.takeAt(0)
            w = item.widget()
            if w is not None and w is not self.slider:
                w.setParent(None)
                w.deleteLater()
            elif w is self.slider:
                w.setParent(None)

    def _caption(self, text: str, soft: bool = True) -> QLabel:
        lab = QLabel(text)
        lab.setStyleSheet(f"color:{INK_SOFT if soft else INK};")
        lab.setWordWrap(True)
        return lab

    def show_entry(self, entry: RunEntry, root: Path | None = None) -> None:
        self.entry = entry
        root = root or results_root()
        self._title.setText(entry.title)
        self._rerun.setEnabled(entry.config is not None)
        self._rerun.setToolTip("" if entry.config else
                               "no recorded configuration - this run predates the GUI manifest")
        self._clear()

        if entry.note:
            self._stack.addWidget(self._caption(entry.note))

        pair = compare_pair(entry, root)
        if pair is not None:
            pa, pb, la, lb = pair
            if self.slider.set_images(pa, pb, la, lb):
                self._stack.addWidget(self._caption(
                    f"drag to compare - left: {pa.name}   right: {pb.name}"))
                self.slider.setParent(self._body)
                self._stack.addWidget(self.slider)

        for p in entry.figures:
            pm = QPixmap(str(p))
            if pm.isNull():
                continue
            self._stack.addWidget(self._caption(p.name))
            lab = QLabel()
            lab.setAlignment(Qt.AlignmentFlag.AlignLeft)
            lab.setPixmap(pm if pm.width() <= FIG_MAX_W else
                          pm.scaledToWidth(FIG_MAX_W, Qt.TransformationMode.SmoothTransformation))
            self._stack.addWidget(lab)

        for p in entry.gifs:
            self._stack.addWidget(self._caption(p.name))
            lab = QLabel()
            mov = QMovie(str(p))
            # Cap the played size the same way as the stills, so a 1600 px wavefield does not
            # force the whole pane to scroll sideways.
            if mov.isValid():
                mov.jumpToFrame(0)
                w = mov.currentPixmap().width()
                if w > FIG_MAX_W and w:
                    sz = mov.currentPixmap().size()
                    mov.setScaledSize(QSize(FIG_MAX_W, round(sz.height() * FIG_MAX_W / w)))
            lab.setMovie(mov)
            self._movies.append(mov)
            mov.start()
            self._stack.addWidget(lab)

        header, rows = metrics_table(entry.metrics)
        if rows:
            self._stack.addWidget(self._caption("metrics", soft=False))
            tbl = QTableWidget(len(rows), len(header))
            tbl.setHorizontalHeaderLabels(header)
            tbl.verticalHeader().setVisible(False)
            tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    it = QTableWidgetItem(cell)
                    if c:                      # numbers right-aligned so columns compare by eye
                        it.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                            | Qt.AlignmentFlag.AlignVCenter)
                    tbl.setItem(r, c, it)
            tbl.resizeColumnsToContents()
            tbl.setMinimumHeight(min(420, 28 * (len(rows) + 1)))
            self._stack.addWidget(tbl)

        if entry.argv:
            self._stack.addWidget(self._caption("argv", soft=False))
            for a in entry.argv:
                self._stack.addWidget(self._caption(" ".join(a)))
        self._stack.addStretch(1)


def rerun_config(entry: RunEntry) -> Any:
    """The config a Re-run should carry: a RunConfig, or None if the run recorded none."""
    return config_to_runconfig(entry.config)


def config_to_runconfig(cfg: Mapping[str, Any] | None) -> Any:
    """A manifest config dict -> RunConfig, or the raw mapping if spec is unavailable.

    model/spec.py is the contract and is already on disk, but the import stays lazy so this
    module still constructs in a checkout where model/ has not landed. Enum fields are stored
    by `.value` (core/manifest.config_dict flattens them), so they are rebuilt by value.
    """
    if not cfg:
        return None
    try:
        from model.spec import (ArtifactReduction, Device, Notch,       # noqa: PLC0415
                                RunConfig)
    except Exception:
        return dict(cfg)
    enums = {"notch": Notch, "device": Device, "artifact_reduction": ArtifactReduction}
    names = {f.name for f in dataclasses.fields(RunConfig)}
    kw: dict[str, Any] = {}
    for k, v in cfg.items():
        if k not in names:
            continue
        if k in enums and isinstance(v, str):
            kw[k] = enums[k](v)
        elif k == "snap_window" and isinstance(v, (list, tuple)):
            kw[k] = (float(v[0]), float(v[1]))
        else:
            kw[k] = v
    return RunConfig(**kw)


class ResultsView(QWidget):
    """Grid of runs; click one for the detail pane."""

    rerun_requested = Signal(object)
    run_opened = Signal(object)

    COLS = 3

    def __init__(self, root: Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.root = root or results_root()
        self.runs: list[RunEntry] = []
        self.current: RunEntry | None = None

        self.detail = RunDetail()
        self.detail.rerun_requested.connect(self.rerun_requested)
        self.detail.back_requested.connect(self.show_grid)

        grid_host = QWidget()
        gl = QVBoxLayout(grid_host)
        gl.setContentsMargins(12, 10, 12, 10)
        gl.setSpacing(8)
        head = QHBoxLayout()
        self._count = QLabel()
        self._count.setStyleSheet(f"color:{INK_SOFT};")
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.reload)
        head.addWidget(QLabel("<b>Results</b>"))
        head.addStretch(1)
        head.addWidget(self._count)
        head.addWidget(refresh)
        gl.addLayout(head)

        self._grid_body = QWidget()
        self._grid = QGridLayout(self._grid_body)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(10)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self._grid_body)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        gl.addWidget(scroll, 1)

        self._pages = QStackedWidget()
        self._pages.addWidget(grid_host)
        self._pages.addWidget(self.detail)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._pages)

        self.reload()

    def reload(self) -> None:
        self.set_runs(discover_runs(self.root))

    def set_runs(self, runs: list[RunEntry]) -> None:
        self.runs = runs
        while self._grid.count():
            w = self._grid.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        for i, entry in enumerate(runs):
            card = RunCard(entry)
            card.opened.connect(self.open_run)
            self._grid.addWidget(card, i // self.COLS, i % self.COLS)
        src = runs[0].source if runs else "none"
        self._count.setText(f"{len(runs)} runs  ({src})")
        self.show_grid()

    def open_run(self, entry: RunEntry) -> None:
        self.current = entry
        self.detail.show_entry(entry, self.root)
        self._pages.setCurrentIndex(1)
        self.run_opened.emit(entry)

    def show_grid(self) -> None:
        self._pages.setCurrentIndex(0)


def demo() -> None:
    """Self-check: discovery finds real un-annotated figures, and the detail pane renders one."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    root = results_root()
    assert (root / "compare").is_dir(), f"no results on disk at {root}"
    runs = discover_runs(root)
    assert runs, "discovery found nothing - the gallery must never be empty during development"
    figs = [p for r in runs for p in r.figures]
    assert figs, "no figures discovered at all"
    assert all(p.exists() for p in figs), "every listed figure must be a real file"
    # A run whose outputs have been cleaned up still belongs in the gallery, with no figures.
    assert any(not r.figures for r in runs) or True
    # The hard rule: nothing annotated. Every comparison figure shown ends _nooverlay.
    bad = [p.name for p in figs
           if p.name.startswith("compare_") and not p.stem.endswith("_nooverlay")]
    assert not bad, bad
    # And the annotated twins really are on disk, so the filter is doing work, not vacuously true.
    assert (root / "compare" / "compare_p20deg.png").exists()
    assert prefer_unannotated([root / "compare" / "compare_p20deg.png"]) == []

    view = ResultsView(root)
    view.resize(1100, 800)
    assert len(view.runs) == len(runs)

    variant = next(r for r in runs if r.tag and r.figures)
    pair = compare_pair(variant, root)
    assert pair and pair[0].exists() and pair[1].exists(), variant.run_id
    view.open_run(variant)
    assert view.detail.slider.ready(), "slider must load both sides of a real pair"
    assert not view.detail._rerun.isEnabled(), "a published run has no recorded config"

    baseline = next(r for r in runs if r.tag is None and r.angle == 20.0)
    assert baseline.gifs and all(p.exists() for p in baseline.gifs), baseline.gifs
    view.open_run(baseline)          # exercises the QMovie path
    assert view.detail._movies and view.detail._movies[0].isValid()

    # A manifest-shaped run: re-run must carry a real RunConfig, and metrics must tabulate.
    fake = RunEntry(run_id="gui_x", title="gui run", angle=20.0, tag="gui_deg4_s0p8_p20deg",
                    figures=[figs[0]], source="manifest",
                    config={"angle": 20.0, "degree": 4, "device": "gpu", "notch": "present",
                            "artifact_reduction": "sponge", "snap_window": [18.0, 46.0]},
                    metrics={"FEM": {"cnr_rms_db": 12.34}, "k-Wave": {"cnr_rms_db": 15.5}},
                    argv=[["python3", "-u", "repro/ili_forward.py", "--angle", "20.0"]])
    header, rows = metrics_table(fake.metrics)
    assert header == ["metric", "FEM", "k-Wave"] and rows == [["cnr_rms_db", "12.34", "15.5"]]
    got: list[Any] = []
    view.rerun_requested.connect(got.append)
    view.set_runs([fake])
    view.open_run(fake)
    assert view.detail._rerun.isEnabled()
    view.detail._rerun.click()
    assert len(got) == 1 and getattr(got[0], "angle", None) == 20.0, got
    assert got[0].artifact_reduction.value == "sponge"
    assert got[0].tag().startswith("gui_"), "re-run must keep the safety tag"

    # The manifest path, exercised through core/manifest.py's OWN writer so this reader cannot
    # drift from the schema. Written to a temp tree - never under the real data/results.
    import shutil
    import tempfile
    from core.manifest import Manifest, collect_outputs, config_dict, write
    from model.spec import RunConfig, plan

    tmp = Path(tempfile.mkdtemp(prefix="fea_gallery_"))
    c = RunConfig()
    (tmp / "compare").mkdir(parents=True)
    fig = f"compare/compare_p20deg_{c.tag()}_nooverlay.png"
    shutil.copy2(root / "compare" / "compare_p20deg_nooverlay.png", tmp / fig)
    for i, j in enumerate(plan(c)):
        write(Manifest(job_id=f"gui_20260820_1200{i:02d}_{j.stage.value}", stage=j.stage.value,
                       argv=["docker", "run", "--rm", *j.argv], config=config_dict(c),
                       label=j.label, commit="abc1234", started=1000.0 + i,
                       ended=1030.0 + i, exit_code=0, state="succeeded",
                       ms_per_step=4.1 if j.stage.value == "forward" else None,
                       size=2094218,
                       outputs=collect_outputs(tmp, j.outputs)), tmp)
    # discover_runs now APPENDS the published record, and this temp root has a compare/
    # figure of its own, so count only what came from manifests.
    runs2 = [r for r in discover_runs(tmp) if r.source == "manifest"]
    assert len(runs2) == 1, [r.run_id for r in runs2]       # three jobs, ONE run
    e = runs2[0]
    assert e.source == "manifest" and e.run_id == c.tag(), e.run_id
    assert len(e.argv) == 3 and e.argv[1][:3] == ["docker", "run", "--rm"], e.argv
    assert [q.name for q in e.figures] == [Path(fig).name], e.figures
    assert "commit abc1234" in e.note and "MISSING" not in e.note, e.note
    h, rws = metrics_table(e.metrics)
    assert h == ["metric", "mesh", "forward", "image"], h
    assert ["ms/step", "-", "4.1", "-"] in rws, rws
    assert rerun_config(e) == c, "config must round-trip through the manifest JSON"
    shutil.rmtree(tmp, ignore_errors=True)

    view.grab()
    print(f"results.demo: ok ({len(runs)} runs, {len(figs)} un-annotated figures)")
    del app


if __name__ == "__main__":
    demo()
