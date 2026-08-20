r"""The live schematic: pipe wall, array, notch, domain extent, steered beam.

Pure QPainter. The app has no host-side matplotlib on purpose - figures come from the
backend as files - and this drawing has to repaint on every spinbox tick, which rules out
rendering a figure anyway.

The one rule that makes this widget trustworthy: TRUE ASPECT. A 9.525 mm wall over a 93 mm
domain is a 1:10 sliver, and stretching it to fill the widget would make the wall look like
a slab and the 4 mm notch look like a canyon. So the world-to-widget map is a single
isotropic scale and the widget letterboxes.

The frame is FIXED to the geometric limits of the domain (x = -47.46 .. +123.96 mm, where the
pipe arc rises to the array plane), not to the selected extent. That is what makes the
widened domain read as widened: the dashed rectangle grows inside a frame that does not move.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):     # direct run: make src/gui the import root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from model.spec import ArtifactReduction, Notch, RunConfig, WIDE_X_MAX, WIDE_X_MIN
from theme import PALETTE

# Mirrors src/backend/mesh/ili_mesh.py. Duplicated rather than imported because that module
# imports gmsh at top level and gmsh is not in the GUI venv. model/scenario.py (W2) reads the
# real numbers out of the backend's scenario dump; pass them in via set_facts() and this
# fallback stops mattering. Any drift shows up as a schematic that disagrees with the mesh.
DEFAULT_FACTS: dict[str, float] = {
    "r_id": 193.675, "r_od": 203.200,        # wall 9.525 mm
    "x_c": 38.25, "z_c": -173.675,           # pipe centre; r_id + z_c = 20.0 mm standoff
    "array_x0": 0.0, "array_x1": 76.5,       # 256 elements at 0.30 mm pitch
    "n_elem": 256, "pitch": 0.30,
    "notch_x": 38.25, "notch_depth": 4.0, "notch_width": 1.0,
    "x_min": -8.0, "x_max": 85.0,            # standard extent, 93 mm
    "x_limit_lo": -47.46, "x_limit_hi": 123.96,
    "standoff": 20.0,
}

_PAD_MM = 4.0          # breathing room around the geometric limits, in world units
_ARC_SAMPLES = 240     # polyline sampling of the arcs; 240 keeps the sagitta sub-pixel
_CAPTION_PX = 18       # a text band under the drawing, in pixels: it must not letterbox


@dataclass(frozen=True)
class _View:
    """Isotropic world(mm) -> widget(px) map. One scale for both axes, by construction."""
    scale: float
    ox: float
    oz: float

    def pt(self, x_mm: float, z_mm: float) -> QPointF:
        # z is up in the world and down in the widget, hence the negated term.
        return QPointF(self.ox + x_mm * self.scale, self.oz - z_mm * self.scale)


def _arc_z(x: float, r: float, x_c: float, z_c: float) -> float | None:
    """Height of the arc of radius r above the array plane at x, or None if off the arc."""
    dx = x - x_c
    d = r * r - dx * dx
    return None if d <= 0.0 else z_c + math.sqrt(d)


class CrossSection(QWidget):
    """Schematic of the scenario for one RunConfig. Read-only; it never edits the config."""

    def __init__(self, config: RunConfig | None = None,
                 facts: dict | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = config or RunConfig()
        self._facts = dict(DEFAULT_FACTS)
        if facts:
            self.set_facts(facts)
        self.setMinimumSize(360, 100)
        # heightForWidth, not Expanding-in-both: at true aspect a taller widget only adds
        # letterbox, so the widget asks for exactly the height its width implies.
        pol = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        pol.setHeightForWidth(True)
        self.setSizePolicy(pol)
        self.setToolTip("Scenario cross-section. True aspect: the wall really is this thin.")

    # ---- inputs -------------------------------------------------------------------

    def set_config(self, config: RunConfig) -> None:
        self._cfg = config
        self.update()

    def set_facts(self, facts) -> None:
        """Accepts a dict or anything with matching attributes, so model/scenario.py (W2) can
        hand over its own facts object without this widget knowing its type."""
        if not isinstance(facts, dict):
            obj = facts
            facts = {k: getattr(obj, k) for k in DEFAULT_FACTS if hasattr(obj, k)}
            # model.scenario states the aperture WIDTH; this widget needs its far edge.
            if hasattr(obj, "aperture"):
                facts["array_x1"] = facts.get("array_x0", 0.0) + obj.aperture
        self._facts.update({k: v for k, v in facts.items() if k in DEFAULT_FACTS})
        self.update()

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt name
        return True

    def heightForWidth(self, w: int) -> int:  # noqa: N802 - Qt name
        r = self.world_rect()
        return int(round(w * r.height() / r.width())) + _CAPTION_PX

    def sizeHint(self):  # noqa: N802 - Qt name
        from PySide6.QtCore import QSize
        return QSize(560, self.heightForWidth(560))

    # ---- derived ------------------------------------------------------------------

    def extent(self) -> tuple[float, float]:
        """The lateral domain the mesh will actually build, per spec.py."""
        if self._cfg.artifact_reduction is ArtifactReduction.WIDE_DOMAIN:
            return WIDE_X_MIN, WIDE_X_MAX
        return self._facts["x_min"], self._facts["x_max"]

    def world_rect(self) -> QRectF:
        """The fixed world window, in mm, with z UP. Independent of the config on purpose."""
        f = self._facts
        z_top = f["z_c"] + f["r_od"] + _PAD_MM          # just above the OD apex
        return QRectF(f["x_limit_lo"] - _PAD_MM, -_PAD_MM,
                      (f["x_limit_hi"] + _PAD_MM) - (f["x_limit_lo"] - _PAD_MM),
                      z_top + _PAD_MM)

    def view(self) -> _View:
        w = self.world_rect()
        avail_h = max(20.0, self.height() - _CAPTION_PX)
        scale = min(self.width() / w.width(), avail_h / w.height())
        # Centre the letterboxed drawing. oz is the widget y of world z = 0.
        ox = (self.width() - w.width() * scale) / 2.0 - w.x() * scale
        oz = (avail_h + w.height() * scale) / 2.0 - (-w.y()) * scale
        return _View(scale, ox, oz)

    # ---- paint --------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor(PALETTE["bg"]))
        if self.width() < 40 or self.height() < 30:
            return
        v = self.view()
        f = self._facts
        self._paint_water(p, v)
        self._paint_wall(p, v)
        self._paint_notch(p, v)
        self._paint_array(p, v)
        self._paint_domain(p, v)
        self._paint_beam(p, v)
        self._paint_captions(p, v, f)

    def _pen(self, token: str, width: float = 1.0, dash: list[float] | None = None) -> QPen:
        pen = QPen(QColor(PALETTE[token]))
        pen.setWidthF(width)
        pen.setCosmetic(True)                 # thin rules stay thin at any widget size
        if dash:
            pen.setStyle(Qt.PenStyle.CustomDashLine)
            pen.setDashPattern(dash)
        return pen

    def _arc_path(self, v: _View, r: float, x0: float, x1: float) -> QPainterPath:
        f = self._facts
        path = QPainterPath()
        for i in range(_ARC_SAMPLES + 1):
            x = x0 + (x1 - x0) * i / _ARC_SAMPLES
            z = _arc_z(x, r, f["x_c"], f["z_c"])
            if z is None:
                continue
            pt = v.pt(x, z)
            if path.elementCount() == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        return path

    def _paint_water(self, p: QPainter, v: _View) -> None:
        """Couplant: array plane up to the ID arc. Barely tinted - it is context, not subject."""
        f, w = self._facts, self.world_rect()
        path = self._arc_path(v, f["r_id"], w.left(), w.right())
        path.lineTo(v.pt(w.right(), 0.0))
        path.lineTo(v.pt(w.left(), 0.0))
        path.closeSubpath()
        p.fillPath(path, QColor(PALETTE["surface"]))

    def _paint_wall(self, p: QPainter, v: _View) -> None:
        """Steel between the arcs, in a muted fill. The subject of the drawing."""
        f, w = self._facts, self.world_rect()
        path = self._arc_path(v, f["r_id"], w.left(), w.right())
        back = self._arc_path(v, f["r_od"], w.right(), w.left())
        path.connectPath(back)
        path.closeSubpath()
        # Steel is the brightest neutral in the palette, water the next: three legible steps
        # (bg, surface, rule) instead of inventing a grey.
        p.fillPath(path, QColor(PALETTE["rule"]))
        p.setPen(self._pen("ink_soft", 1.0))
        p.drawPath(self._arc_path(v, f["r_id"], w.left(), w.right()))
        p.drawPath(self._arc_path(v, f["r_od"], w.left(), w.right()))

    def _paint_notch(self, p: QPainter, v: _View) -> None:
        """The 4 x 1 mm slot: a void, so it is punched back to the window colour."""
        f = self._facts
        if self._cfg.notch is Notch.ABSENT:
            return
        xl = f["notch_x"] - f["notch_width"] / 2.0
        xr = f["notch_x"] + f["notch_width"] / 2.0
        z_tip = f["z_c"] + f["r_od"] - f["notch_depth"]
        path = QPainterPath()
        path.moveTo(v.pt(xl, z_tip))
        path.lineTo(v.pt(xl, _arc_z(xl, f["r_od"], f["x_c"], f["z_c"]) or z_tip))
        path.lineTo(v.pt(xr, _arc_z(xr, f["r_od"], f["x_c"], f["z_c"]) or z_tip))
        path.lineTo(v.pt(xr, z_tip))
        path.closeSubpath()
        # FILLED means a soft solid, not a void: shade it instead of cutting it out.
        void = self._cfg.notch is not Notch.FILLED
        p.fillPath(path, QColor(PALETTE["bg"] if void else PALETTE["surface"]))
        p.setPen(self._pen("ink_soft", 1.0))
        p.drawPath(path)
        # Leader line up out of the wall, so the label never sits on the 1 mm slot.
        top = v.pt(f["notch_x"], f["z_c"] + f["r_od"])
        p.setPen(self._pen("rule", 1.0))
        p.drawLine(top, QPointF(top.x(), top.y() - 12))
        self._text(p, QPointF(top.x() + 4, top.y() - 10), "ink_soft",
                   "notch %.1f x %.1f mm" % (f["notch_depth"], f["notch_width"]))

    def _paint_array(self, p: QPainter, v: _View) -> None:
        """The flat transducer plane at z = 0, with the aperture marked."""
        f, w = self._facts, self.world_rect()
        p.setPen(self._pen("rule", 1.0, dash=[6, 4]))
        p.drawLine(v.pt(w.left(), 0.0), v.pt(w.right(), 0.0))
        a0, a1 = f["array_x0"], f["array_x1"]
        p.setPen(self._pen("ink", 3.0))
        p.drawLine(v.pt(a0, 0.0), v.pt(a1, 0.0))
        p.setPen(self._pen("ink_soft", 1.0))
        for x in (a0, a1):                        # end ticks: the aperture edges are exact
            p.drawLine(v.pt(x, -1.6), v.pt(x, 1.6))
        self._text(p, v.pt(a0, 0.0) + QPointF(2, 14), "ink_soft",
                   "%d el @ %.2f mm = %.1f mm aperture"
                   % (f["n_elem"], f["pitch"], a1 - a0))

    def _paint_domain(self, p: QPainter, v: _View) -> None:
        """Dashed lateral extent. Orange when the config moved it - that edge is the change."""
        f = self._facts
        x0, x1 = self.extent()
        z_top = f["z_c"] + f["r_od"] + 1.0
        wide = self._cfg.artifact_reduction is ArtifactReduction.WIDE_DOMAIN
        if wide:
            # Ghost of the standard extent, so the widening is a visible delta, not a claim.
            p.setPen(self._pen("rule", 1.0, dash=[2, 4]))
            p.drawRect(QRectF(v.pt(f["x_min"], z_top), v.pt(f["x_max"], 0.0)))
        p.setPen(self._pen("accent" if wide else "rule", 1.4, dash=[7, 5]))
        p.drawRect(QRectF(v.pt(x0, z_top), v.pt(x1, 0.0)))
        tok = "accent" if wide else "ink_soft"
        for x, align in ((x0, 3.0), (x1, -46.0)):
            self._text(p, v.pt(x, z_top) + QPointF(align, -4), tok, "%+.1f" % x)

    def _paint_beam(self, p: QPainter, v: _View) -> None:
        """Plane-wave propagation direction from the aperture centre.

        Positive angle tilts toward +x. The delay law itself is the research team's - see
        repro/ili_forward.element_delays - so this arrow states the convention, not the law.
        """
        f = self._facts
        xc = (f["array_x0"] + f["array_x1"]) / 2.0
        a = math.radians(self._cfg.angle)
        length = f["standoff"] + 6.0
        tip = QPointF(xc + length * math.sin(a), length * math.cos(a))
        p0, p1 = v.pt(xc, 0.0), v.pt(tip.x(), tip.y())
        p.setPen(self._pen("rule", 1.0, dash=[2, 3]))
        p.drawLine(p0, v.pt(xc, length))                      # the normal, for reference
        p.setPen(self._pen("accent", 1.8))
        p.drawLine(p0, p1)
        # Arrowhead in widget space: 7 px long, 4 px half-width, pointing along the beam.
        ux, uy = p1.x() - p0.x(), p1.y() - p0.y()
        n = math.hypot(ux, uy) or 1.0
        ux, uy = ux / n, uy / n
        base = QPointF(p1.x() - 9 * ux, p1.y() - 9 * uy)
        head = QPolygonF([p1, QPointF(base.x() - 4.5 * uy, base.y() + 4.5 * ux),
                          QPointF(base.x() + 4.5 * uy, base.y() - 4.5 * ux)])
        p.setBrush(QColor(PALETTE["accent"]))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(head)
        p.setBrush(Qt.BrushStyle.NoBrush)
        self._text(p, QPointF(p1.x() + 6, p1.y() + 4), "accent",
                   "%+.1f deg" % self._cfg.angle)

    def _paint_captions(self, p: QPainter, v: _View, f: dict) -> None:
        x0, x1 = self.extent()
        self._text(p, QPointF(8, self.height() - 8), "ink_soft",
                   "extent %+.1f .. %+.1f mm  (%.1f mm)   wall %.3f mm   standoff %.1f mm"
                   % (x0, x1, x1 - x0, f["r_od"] - f["r_id"], f["r_id"] + f["z_c"]),
                   mono=True)

    def _text(self, p: QPainter, at: QPointF, token: str, s: str, mono: bool = False) -> None:
        font = QFont("Consolas" if mono else "Segoe UI", 8)
        p.setFont(font)
        p.setPen(self._pen(token, 1.0))
        p.drawText(at, s)


def demo() -> None:
    """Self-check: the map is isotropic, the frame is config-independent, the extent tracks."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from dataclasses import replace

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    w = CrossSection()
    w.resize(800, 220)
    w.show()
    app.processEvents()

    v = w.view()
    world = w.world_rect()
    # Isotropy is the whole point: one scale, so a 9.525 mm wall stays 1:10 against 93 mm.
    assert v.scale > 0
    assert abs((v.pt(10, 0).x() - v.pt(0, 0).x()) - (v.pt(0, 0).y() - v.pt(0, 10).y())) < 1e-9, \
        "aspect is not preserved"
    # Letterboxed, not cropped: the whole world window lands inside the widget.
    for x in (world.left(), world.right()):
        for z in (0.0, world.height() - _PAD_MM):
            pt = v.pt(x, z)
            assert -0.5 <= pt.x() <= w.width() + 0.5, pt
            assert -0.5 <= pt.y() <= w.height() + 0.5, pt

    assert w.extent() == (DEFAULT_FACTS["x_min"], DEFAULT_FACTS["x_max"])
    frame_before = w.world_rect()
    w.set_config(replace(RunConfig(), artifact_reduction=ArtifactReduction.WIDE_DOMAIN))
    assert w.extent() == (WIDE_X_MIN, WIDE_X_MAX), "widened domain not picked up"
    assert w.world_rect() == frame_before, "the frame must not move when the extent does"

    for cfg in (RunConfig(), replace(RunConfig(), angle=-35.0),
                replace(RunConfig(), notch=Notch.ABSENT),
                replace(RunConfig(), notch=Notch.FILLED)):
        w.set_config(cfg)
        w.repaint()          # every branch of the notch drawing must survive a repaint
    app.processEvents()
    print("crosssection.demo: ok, scale %.3f px/mm" % v.scale)


if __name__ == "__main__":
    demo()
