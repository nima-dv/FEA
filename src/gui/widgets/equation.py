r"""The equation actually being solved, rebuilt whenever a parameter changes.

WHY THIS IS WORTH A PANEL
  A parameter form hides the thing it is configuring. "Artifact reduction: sponge" is a word;
  what it MEANS is that a damping term appears in the momentum balance and the absorbing
  boundary condition changes shape. Showing the equation makes the form self-explanatory in a
  way no tooltip does, and it separates the parameters that change the PHYSICS from the ones
  that only change the numerics - which is a distinction users of this app need and cannot
  otherwise see.

HOW IT WORKS
  Every row is (latex, active, note). Rows that are not in force for the current configuration
  are dimmed and annotated rather than hidden, because "there is no damping term right now" is
  information, and a row that vanishes teaches nothing. Rendering is matplotlib mathtext -
  no LaTeX installation, and it is the same library the backend renders published figures
  with.

WHAT IT IS NOT
  Not a derivation, and not a substitute for presentation/docs/lessons.md. It states the system being
  solved, which boundary conditions are live, and the stability limit with the current numbers
  substituted in.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg      # noqa: E402
from matplotlib.figure import Figure                                 # noqa: E402
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget      # noqa: E402

from model.spec import ArtifactReduction, Notch, RunConfig           # noqa: E402

try:
    from theme import PALETTE
except Exception:                                    # standalone use
    PALETTE = {"bg": "#0B0B0C", "surface": "#16181B", "ink": "#E8E8EA",
               "ink_soft": "#9AA0A6", "accent": "#FF7A1A", "rule": "#2A2E33"}


@dataclass(frozen=True)
class Row:
    latex: str
    active: bool = True
    note: str = ""
    heading: bool = False


def _fmt_dt(cfg: RunConfig) -> tuple[str, str]:
    """Live time step and step count, or em-dashes if the estimator is unavailable."""
    try:
        from model import derived, scenario
        sc = scenario.load()
        dt = derived.estimated_dt(cfg, sc)
        steps = derived.estimated_steps(cfg, sc)
        val = getattr(dt, "value", dt)
        n = getattr(steps, "value", steps)
        return f"{val * 1e9:.4f}\\ \\mathrm{{ns}}", f"{int(n):,}"
    except Exception:
        return "\\mathrm{-}", "-"


def rows_for(cfg: RunConfig) -> list[Row]:
    """The system for this configuration. Order is strong form, materials, BCs, discrete."""
    sponge = cfg.artifact_reduction is ArtifactReduction.SPONGE
    dt_s, steps_s = _fmt_dt(cfg)
    out: list[Row] = [Row("\\mathrm{Momentum\\ balance\\ in\\ }\\Omega", heading=True)]

    # The damping term only exists for the sponge. Showing both forms, one dimmed, is the
    # whole point: it makes "sponge" mean something.
    out.append(Row(
        r"$\rho\,\ddot{\mathbf{u}} = \nabla\!\cdot\!\sigma$",
        active=not sponge,
        note="" if not sponge else "replaced by the damped form below"))
    out.append(Row(
        r"$\rho\,\ddot{\mathbf{u}} + C(\mathbf{x})\,\dot{\mathbf{u}}"
        r" = \nabla\!\cdot\!\sigma$",
        active=sponge,
        note="graded sponge, 40 dB round trip over the 8 mm margins"
             if sponge else "no damping term: artifact reduction is not set to sponge"))
    out.append(Row(
        r"$\sigma = \lambda\,(\nabla\!\cdot\!\mathbf{u})\,\mathbf{I}"
        r" + 2\mu\,\varepsilon(\mathbf{u}),\qquad"
        r"\varepsilon = \frac{1}{2}(\nabla\mathbf{u} + \nabla\mathbf{u}^{T})$"))

    out.append(Row("\\mathrm{Regions}", heading=True))
    out.append(Row(r"$\mathrm{steel:}\ \ \mu > 0$"
                   r"$\quad c_P = \sqrt{(\lambda + 2\mu)/\rho},\ \ c_S = \sqrt{\mu/\rho}$",
                   note="5700 and 3100 m/s"))
    out.append(Row(r"$\mathrm{water:}\ \ \mu = 0"
                   r"\quad\Rightarrow\quad \sigma = -p\,\mathbf{I},"
                   r"\ \ p = -\lambda_f\,\nabla\!\cdot\!\mathbf{u}$",
                   note="1500 m/s; the same equation, not a separate acoustic solver"))
    out.append(Row(r"$\mathrm{crack\ fill:}\ \ \mu = 0,\ \ \lambda_{fill}$",
                   active=cfg.notch is Notch.FILLED,
                   note="" if cfg.notch is Notch.FILLED else "no fill region in this run"))

    out.append(Row("\\mathrm{Boundary\\ conditions}", heading=True))
    out.append(Row(
        r"$\Gamma_{array}:\ \ \sigma\!\cdot\!\mathbf{n}"
        r" = -g(t - \tau_i)\,\mathbf{n}$",
        note=f"the source; delays tau_i set by the {cfg.angle:+.0f} deg steering"))
    out.append(Row(
        r"$\Gamma_{ID},\ \Gamma_{OD},\ \Gamma_{crack}:\ \ "
        r"\sigma\!\cdot\!\mathbf{n} = \mathbf{0}$",
        note="traction-free. An open crack needs no material model - this IS the crack"
             if cfg.notch is not Notch.ABSENT else
             "healthy wall: no crack boundary in this run"))
    # The two absorbing forms. NONE is the published one and is the cruder of the two.
    out.append(Row(
        r"$\Gamma_{abc}:\ \ \sigma\!\cdot\!\mathbf{n}"
        r" = -\rho\,c_P\,\dot{\mathbf{u}}$",
        active=not sponge,
        note="one wave speed for both modes; over-damps shear by ~30% of amplitude, "
             "and is what every published figure uses"))
    out.append(Row(
        r"$\Gamma_{abc}:\ \ \sigma\!\cdot\!\mathbf{n} = -\rho\left[c_P(\dot{\mathbf{u}}"
        r"\!\cdot\!\mathbf{n})\mathbf{n} + c_S\,\dot{\mathbf{u}}_t\right]$",
        active=sponge,
        note="shear-matched: exact at normal incidence, leakier as the angle grows"
             if sponge else "not in force"))

    out.append(Row("\\mathrm{Discrete\\ system}", heading=True))
    out.append(Row(
        r"$\mathbf{M}\,\ddot{u} + \mathbf{C}\,\dot{u} + \mathbf{K}u = f(t)$" if sponge
        else r"$\mathbf{M}\,\ddot{u} + \mathbf{K}u = f(t)$",
        note=f"degree {cfg.degree} elements"))
    out.append(Row(
        r"$u^{n+1} = 2u^{n} - u^{n-1}"
        r" - \Delta t^{2}\,\mathbf{M}^{-1}\left(\mathbf{K}u^{n} - f^{n}\right)$",
        note="explicit leapfrog; M is diagonal, so this is a division and not a solve"
             if cfg.quad else
             "WARNING: on triangles the lumped mass is only approximately diagonal"))
    out.append(Row(
        r"$\Delta t \leq C\,\frac{h_{min}}{c_{max}\,p^{2}}"
        rf"\qquad \Delta t \approx {dt_s},\quad {steps_s}\ \mathrm{{steps}}$",
        note=f"C = {cfg.cfl:.2f}, same as k-Wave. One global step, set by the smallest cell "
             f"over the fastest wave"))
    return out


class EquationPanel(QWidget):
    """A figure that re-renders on every configuration change."""

    def __init__(self, cfg: RunConfig | None = None, parent=None) -> None:
        super().__init__(parent)
        self._fig = Figure(figsize=(6.4, 7.6), dpi=100)
        self._fig.patch.set_facecolor(PALETTE.get("surface", "#16181B"))
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._canvas)
        self.rows: list[Row] = []
        self.set_config(cfg or RunConfig())

    def set_config(self, cfg: RunConfig) -> None:
        self.rows = rows_for(cfg)
        self._draw()

    def _draw(self) -> None:
        self._fig.clear()
        ink = PALETTE.get("ink", "#E8E8EA")
        soft = PALETTE.get("ink_soft", "#9AA0A6")
        accent = PALETTE.get("accent", "#FF7A1A")
        # Dimmed rows must stay legible - they are information, not decoration. This is a
        # deliberate mid-grey rather than a low alpha, which would fringe the mathtext glyphs.
        off = "#5A5F66"

        y = 0.985
        for r in self.rows:
            if r.heading:
                y -= 0.018
                self._fig.text(0.03, y, r.latex.replace("\\mathrm{", "").replace("}", "")
                               .replace("\\ ", " "), color=accent, fontsize=9.5,
                               va="top", family="monospace")
                y -= 0.030
                continue
            self._fig.text(0.045, y, r.latex, color=ink if r.active else off,
                           fontsize=12.5, va="top")
            y -= 0.036
            if r.note:
                self._fig.text(0.055, y, r.note, color=soft if r.active else off,
                               fontsize=7.8, va="top", wrap=True)
                y -= 0.028
            y -= 0.008
        self._canvas.draw_idle()


def demo() -> None:
    """The equation must actually change with the parameters, not just be displayed."""
    import os
    from dataclasses import replace
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    base = RunConfig()
    p = EquationPanel(base)
    active = lambda rows: [r.latex for r in rows if r.active and not r.heading]   # noqa: E731

    # 1. Default: undamped balance live, sponge form dimmed.
    a0 = active(p.rows)
    assert any("C(\\mathbf{x})" not in s and "\\ddot{\\mathbf{u}} = \\nabla" in s for s in a0), a0
    assert not any("C(\\mathbf{x})" in s for s in a0), "no damping term should be live"

    # 2. Sponge: the damping term appears AND the absorbing condition changes shape.
    p.set_config(replace(base, artifact_reduction=ArtifactReduction.SPONGE))
    a1 = active(p.rows)
    assert any("C(\\mathbf{x})" in s for s in a1), "sponge must add the damping term"
    assert any("c_S" in s and "Gamma_{abc}" in s for s in a1), "and change the ABC"
    assert any("\\mathbf{C}" in s for s in a1), "discrete system must gain C"

    # 3. Healthy wall: the crack boundary is no longer in force.
    p.set_config(replace(base, notch=Notch.ABSENT))
    note = next(r.note for r in p.rows if "Gamma_{crack}" in r.latex)
    assert "healthy" in note, note

    # 4. Filled crack: the fill region becomes live.
    p.set_config(replace(base, notch=Notch.FILLED))
    assert any("lambda_{fill}" in r.latex and r.active for r in p.rows)

    # 5. Triangles: the lumped-mass caveat must appear, because it is a real one.
    p.set_config(replace(base, quad=False))
    assert any("only approximately diagonal" in r.note for r in p.rows)

    # 6. The CFL line carries live numbers, and they move with the parameters.
    p.set_config(base)
    cfl_row = next(r for r in p.rows if "\\Delta t \\leq" in r.latex)
    p.set_config(replace(base, degree=3))
    cfl_row3 = next(r for r in p.rows if "\\Delta t \\leq" in r.latex)
    assert cfl_row.latex != cfl_row3.latex, "step count must change with element order"

    p.resize(640, 760)
    p._draw()
    # draw_idle() defers, and a deferred draw never parses the mathtext - which is how a
    # \frac that mathtext does not implement passed this check once already. Force it.
    for cfg in (base,
                replace(base, artifact_reduction=ArtifactReduction.SPONGE),
                replace(base, notch=Notch.FILLED),
                replace(base, quad=False, degree=3)):
        p.set_config(cfg)
        p._canvas.draw()          # raises on any unsupported symbol in any row
    print(f"equation.demo: ok ({len(p.rows)} rows, "
          f"{sum(1 for r in p.rows if not r.heading and not r.active)} dimmed)")


if __name__ == "__main__":
    demo()
