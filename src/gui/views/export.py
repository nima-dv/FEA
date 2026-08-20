r"""Export pane: pick assets from one or more runs, copy them out with their provenance.

COPY, NEVER MOVE
`data/results` is the published record for the R&D challenge - the k-Wave +20 deg baseline and
everything scored against it. Every path here is a source; the only writes go to the
destination the user picks, and `export_bundle` refuses a destination inside the results tree
so a stray "export into results/" cannot rewrite the record.

A BUNDLE IS NOT JUST PICTURES
`commands.txt` carries the argv that produced each figure, because a figure without its
command is an assertion rather than a result. `metrics.md` carries the numbers in the same
folder as the images they describe.

POWERSHELL AND THE COMMA (recorded here because it cost an afternoon)
`--xlim -8,85` cannot be passed to a script from PowerShell as two tokens: PowerShell splits
the argument on the comma and then reads the leading `-` of `-8` as the start of a parameter
name, so the script sees `--xlim` with no value. It has to be one quoted token,
`'--xlim=-8,85'`. The app never hits this - it builds an argv list and hands it straight to
docker, with no shell in between - but anything copied out of `commands.txt` and pasted into a
PowerShell prompt does, so the file is written in the `--flag=value` form.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

_GUI_ROOT = Path(__file__).resolve().parents[1]
if str(_GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_GUI_ROOT))

from PySide6.QtCore import Qt, Signal                                          # noqa: E402
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog,         # noqa: E402
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSpinBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from views.results import (RunEntry, discover_runs, metrics_table,             # noqa: E402
                          results_root)
from widgets.jobcard import INK_SOFT                                          # noqa: E402

CHECKED = Qt.CheckState.Checked
UNCHECKED = Qt.CheckState.Unchecked
ROLE = Qt.ItemDataRole.UserRole

# The real flags of src/backend/viz/wavefield_gif.py, with THAT SCRIPT's defaults and help
# text - so what the dialog shows is what the script does when a field is left alone. Note that
# model/spec.py's FIGURES stage picks its own values for the published animation (stride 8,
# fps 7, colors 24, smooth 2); this dialog is for re-rendering by hand, not for editing that.
# Kept as data so the dialog cannot drift from the script: if a flag changes there, this list
# is the one place to correct, and `--clip/--smooth/--stills` are deliberately not exposed
# (cosmetic smoothing has to be declared in a caption, so it is not a GUI convenience).
GIF_FLAGS: tuple[tuple[str, str, Any, str], ...] = (
    ("stride", "--stride", 1,
     "use every Nth frame (default 1). Halves file size per doubling."),
    ("fps", "--fps", 20, "frames per second (default 20)."),
    ("colors", "--colors", 96,
     "GIF palette size (default 96). The field is smooth, so a small palette costs nothing "
     "visually and dominates the file size."),
    ("xlim", "--xlim", "",
     "X0,X1 - crop to this x window [mm]. Needed to animate two domains of DIFFERENT width "
     "over the same window, so a cropped wide run and an uncropped narrow run are "
     "pixel-comparable."),
    ("vlim", "--vlim", "",
     "WATER,STEEL - force the two colour limits instead of deriving them from this file's own "
     "p99.5. REQUIRED for a side-by-side pair: with per-file limits the same wave renders at a "
     "different brightness in each panel, and a viewer would read the normalisation as physics."),
)


class GifDialog(QDialog):
    """The animation flags, spelled as `wavefield_gif.py` spells them."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Wavefield GIF options")
        form = QFormLayout(self)
        self._fields: dict[str, QSpinBox | QLineEdit] = {}
        for name, flag, default, help_text in GIF_FLAGS:
            if isinstance(default, int):
                w: QSpinBox | QLineEdit = QSpinBox()
                w.setRange(1, 100000 if name != "colors" else 256)
                w.setValue(default)
            else:
                w = QLineEdit()
                w.setPlaceholderText("derived from the file" if name == "vlim" else "no crop")
            w.setToolTip(help_text)
            lab = QLabel(flag)
            lab.setToolTip(help_text)
            form.addRow(lab, w)
            self._fields[name] = w
        note = QLabel("Comma pairs are written as --flag=value: PowerShell would otherwise "
                      "split on the comma and read the leading minus as a parameter.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{INK_SOFT};")
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def flags(self) -> list[str]:
        """argv fragment. Non-default numbers and any set string flag, nothing else."""
        out: list[str] = []
        for name, flag, default, _ in GIF_FLAGS:
            w = self._fields[name]
            if isinstance(w, QSpinBox):
                if w.value() != default:
                    out += [flag, str(w.value())]
            else:
                v = w.text().strip()
                if v:
                    # `=` form on purpose - see the PowerShell note in the module docstring.
                    out.append(f"{flag}={v}")
        return out

    def set_values(self, **kw: Any) -> None:
        """Used by the self-check; also lets a preset seed the dialog later."""
        for k, v in kw.items():
            w = self._fields[k]
            if isinstance(w, QSpinBox):
                w.setValue(int(v))
            else:
                w.setText(str(v))


def gif_argv(npz: Path, flags: Sequence[str], root: Path | None = None) -> list[str]:
    """The command INSIDE the container, so it matches what docker/run.ps1 mounts."""
    root = root or results_root()
    try:
        rel = f"results/{npz.relative_to(root).as_posix()}"
    except ValueError:
        rel = str(npz)
    return ["python3", "viz/wavefield_gif.py", "--in", rel, *flags]


def _fmt_argv(argv: Sequence[str]) -> str:
    """One pasteable line. Only quotes what needs it; keeps `--flag=value` pairs intact."""
    return " ".join(a if a and not any(c in a for c in ' "\t') else f'"{a}"' for a in argv)


def commands_text(runs: Sequence[RunEntry], gif: Sequence[list[str]] = ()) -> str:
    lines = ["# Commands that produced this bundle. Run from the repo root:",
             "#   docker/run.ps1 <command>            (add -Gpu for a GPU solve)",
             "# Comma-valued flags must stay ONE token in PowerShell: '--xlim=-8,85'.", ""]
    for r in runs:
        lines.append(f"## {r.run_id}")
        if r.argv:
            lines += [_fmt_argv(a) for a in r.argv]
        else:
            lines.append("# argv not recorded for this run (predates the GUI manifest); "
                         "see data/results/compare/NAMING.md for how it was made")
        lines.append("")
    for a in gif:
        lines += ["## wavefield animation", _fmt_argv(a), ""]
    return "\n".join(lines)


def metrics_text(runs: Sequence[RunEntry]) -> str:
    out = ["# Metrics", ""]
    for r in runs:
        out.append(f"## {r.run_id}")
        header, rows = metrics_table(r.metrics)
        if rows:
            out += ["| " + " | ".join(header) + " |",
                    "|" + "|".join(["---"] * len(header)) + "|"]
            out += ["| " + " | ".join(row) + " |" for row in rows]
        else:
            out.append("No metrics recorded. `repro/compare_images.py` prints its table to "
                       "stdout, so the numbers live in this run's log until core/manifest.py "
                       "captures them.")
        out.append("")
    return "\n".join(out)


def export_bundle(runs: Sequence[RunEntry], files: Iterable[Path], dest: Path,
                  gif: Sequence[list[str]] = (), per_run: bool | None = None) -> list[Path]:
    """Copy `files` plus commands.txt and metrics.md into `dest`. Returns what was written."""
    dest = Path(dest).resolve()
    root = results_root().resolve()
    # A destination inside the published record would let an export overwrite the very files it
    # is exporting. Blocked here rather than in the UI so the check cannot be bypassed.
    if dest == root or root in dest.parents:
        raise ValueError(f"refusing to export into the published record: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    files = list(files)
    if per_run is None:
        per_run = len(runs) > 1              # avoids name collisions between runs
    # First run wins a shared path: two runs pointing at the same file is degenerate, and a
    # deterministic owner beats a silently dict-ordered one.
    owner: dict[Path, RunEntry] = {}
    for r in runs:
        for p in r.assets + r.npz:
            owner.setdefault(p, r)

    written: list[Path] = []
    for src in files:
        if not src.exists():
            continue
        out_dir = dest
        if per_run and src in owner:
            out_dir = dest / owner[src].run_id
            out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / src.name
        shutil.copy2(src, target)            # COPY: the source is the published record
        written.append(target)

    for name, text in (("commands.txt", commands_text(runs, gif)),
                       ("metrics.md", metrics_text(runs))):
        p = dest / name
        p.write_text(text, encoding="ascii", errors="replace")
        written.append(p)
    return written


class ExportView(QWidget):
    """Checkbox tree of assets, a destination, and one button that copies."""

    exported = Signal(str)
    gif_requested = Signal(list)

    def __init__(self, runs: Sequence[RunEntry] | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.runs: list[RunEntry] = []
        self._gif = GifDialog(self)
        self._gif_flags: list[str] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(QLabel("<b>Export</b>"))
        head.addStretch(1)
        self._status = QLabel()
        self._status.setStyleSheet(f"color:{INK_SOFT};")
        head.addWidget(self._status)
        outer.addLayout(head)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["asset", "kind"])
        self.tree.setColumnWidth(0, 420)
        outer.addWidget(self.tree, 1)

        row = QHBoxLayout()
        self.dest = QLineEdit()
        self.dest.setPlaceholderText("destination folder (outside data/results)")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        gifbtn = QPushButton("GIF options...")
        gifbtn.clicked.connect(self._open_gif)
        self._export_btn = QPushButton("Export bundle")
        self._export_btn.clicked.connect(self.do_export)
        row.addWidget(self.dest, 1)
        row.addWidget(browse)
        row.addWidget(gifbtn)
        row.addWidget(self._export_btn)
        outer.addLayout(row)

        self.set_runs(list(runs) if runs is not None else [])

    # ---- tree --------------------------------------------------------------------------
    def set_runs(self, runs: Sequence[RunEntry]) -> None:
        self.runs = list(runs)
        self.tree.clear()
        for r in self.runs:
            top = QTreeWidgetItem([r.title or r.run_id, r.source])
            top.setFlags(top.flags() | Qt.ItemFlag.ItemIsUserCheckable
                         | Qt.ItemFlag.ItemIsAutoTristate)
            top.setCheckState(0, CHECKED)
            self.tree.addTopLevelItem(top)
            groups = (("PNG", r.figures), ("GIF", r.gifs), ("NPZ", r.npz))
            for kind, paths in groups:
                if not paths:
                    continue
                grp = QTreeWidgetItem([kind, f"{len(paths)}"])
                grp.setFlags(grp.flags() | Qt.ItemFlag.ItemIsUserCheckable
                             | Qt.ItemFlag.ItemIsAutoTristate)
                top.addChild(grp)
                for p in paths:
                    leaf = QTreeWidgetItem([p.name, kind.lower()])
                    leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    # NPZ is raw data, not a figure: opt-in, because one file is ~120 MB.
                    leaf.setCheckState(0, UNCHECKED if kind == "NPZ" else CHECKED)
                    leaf.setData(0, ROLE, str(p))
                    leaf.setToolTip(0, str(p))
                    grp.addChild(leaf)
            for label, present in (("metrics.md", bool(metrics_table(r.metrics)[1])),
                                   ("commands.txt", bool(r.argv))):
                it = QTreeWidgetItem([label, "always written"
                                      if present else "written, no data recorded"])
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                it.setDisabled(True)
                top.addChild(it)
            top.setExpanded(True)
        self._status.setText(f"{len(self.runs)} runs, {len(self.selected_files())} files selected")

    def selected_files(self) -> list[Path]:
        out: list[Path] = []
        it = self.tree.topLevelItemCount()
        for i in range(it):
            self._collect(self.tree.topLevelItem(i), out)
        return out

    def _collect(self, item: QTreeWidgetItem, out: list[Path]) -> None:
        data = item.data(0, ROLE)
        if data and item.checkState(0) == CHECKED:
            out.append(Path(data))
        for i in range(item.childCount()):
            self._collect(item.child(i), out)

    def selected_runs(self) -> list[RunEntry]:
        """Runs with at least one checked asset; export writes provenance only for those."""
        chosen = set(self.selected_files())
        return [r for r in self.runs if chosen & set(r.assets + r.npz)] or self.runs

    # ---- actions -----------------------------------------------------------------------
    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Export to")
        if d:
            self.dest.setText(d)

    def _open_gif(self) -> None:
        if self._gif.exec():
            self._gif_flags = self._gif.flags()
            for a in self.gif_commands():
                self.gif_requested.emit(a)          # W1 may run it; this pane never does

    def gif_commands(self) -> list[list[str]]:
        """One command per selected wavefield snapshot file."""
        return [gif_argv(p, self._gif_flags)
                for r in self.selected_runs() for p in r.npz
                if p.name.startswith("wavefield")]

    def do_export(self) -> list[Path]:
        dest = self.dest.text().strip()
        if not dest:
            self._status.setText("pick a destination folder first")
            return []
        try:
            written = export_bundle(self.selected_runs(), self.selected_files(), Path(dest),
                                    self.gif_commands())
        except (OSError, ValueError) as exc:
            self._status.setText(str(exc))
            return []
        self._status.setText(f"copied {len(written)} files to {dest}")
        self.exported.emit(dest)
        return written


def demo() -> None:
    """Self-check: real assets get copied, provenance is written, the record is protected."""
    import os
    import tempfile
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    root = results_root()
    runs = discover_runs(root)
    assert runs, "nothing to export - discovery found no runs"
    baseline = next(r for r in runs if r.tag is None and r.angle == 20.0)

    pane = ExportView([baseline])
    pane.resize(900, 600)
    files = pane.selected_files()
    assert files and all(p.exists() for p in files), files
    assert not any(p.suffix == ".npz" for p in files), "raw data must be opt-in"
    assert all(not (p.name.startswith("compare_") and not p.stem.endswith("_nooverlay"))
               for p in files), "only un-annotated figures may leave the app"

    dest = Path(tempfile.mkdtemp(prefix="fea_export_"))
    pane.dest.setText(str(dest))
    written = pane.do_export()
    assert (dest / "commands.txt").exists() and (dest / "metrics.md").exists()
    assert len(written) == len(files) + 2, (len(written), len(files))
    for p in files:                       # copy, not move: the record is still there
        assert p.exists() and (dest / p.name).exists()
        assert (dest / p.name).stat().st_size == p.stat().st_size
    txt = (dest / "commands.txt").read_text()
    assert "argv not recorded" in txt, txt

    # The guard that matters: never write into the published record.
    try:
        export_bundle([baseline], files, root / "compare" / "exported")
    except ValueError as exc:
        assert "refusing" in str(exc)
    else:
        raise AssertionError("export into data/results must be refused")

    # GIF flags: defaults produce nothing, and comma pairs come out as one token.
    d = GifDialog()
    assert d.flags() == []
    d.set_values(stride=2, fps=12, xlim="-8,85", vlim="1.2e-9,3.4e-10")
    assert d.flags() == ["--stride", "2", "--fps", "12",
                         "--xlim=-8,85", "--vlim=1.2e-9,3.4e-10"], d.flags()
    pane._gif_flags = d.flags()
    cmds = pane.gif_commands()
    assert cmds and cmds[0][:3] == ["python3", "viz/wavefield_gif.py", "--in"], cmds
    assert cmds[0][3].startswith("results/ili_forward/wavefield_snap"), cmds[0]
    assert "--xlim=-8,85" in cmds[0]

    # A manifest-shaped run carries argv and metrics, so both files gain real content.
    rich = RunEntry(run_id="gui_deg4_s0p8_p20deg", title="gui run", angle=20.0,
                    figures=[next(r for r in runs if r.tag).figures[0]], source="manifest",
                    metrics={"FEM": {"cnr_rms_db": 12.3}, "k-Wave": {"cnr_rms_db": 15.5}},
                    argv=[["python3", "-u", "repro/ili_forward.py", "--angle", "20.0",
                           "--snap-window=18.0,46.0"]])
    pane.set_runs([rich, baseline])
    d2 = Path(tempfile.mkdtemp(prefix="fea_export2_"))
    pane.dest.setText(str(d2))
    pane.do_export()
    assert (d2 / rich.run_id).is_dir(), "several runs must not collide in one folder"
    cmd = (d2 / "commands.txt").read_text()
    assert "--snap-window=18.0,46.0" in cmd, cmd
    assert "| cnr_rms_db | 12.3 | 15.5 |" in (d2 / "metrics.md").read_text()

    shutil.rmtree(dest, ignore_errors=True)
    shutil.rmtree(d2, ignore_errors=True)
    print("export.demo: ok")
    del app


if __name__ == "__main__":
    demo()
