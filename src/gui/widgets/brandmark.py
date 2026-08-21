r"""BrandMark: the "Cracken" wordmark - orange face, layered black/orange shadow beneath.

Painted, not styled: QSS has no text-shadow, and a QGraphicsDropShadowEffect gives only one
flat-colour blur. A wordmark wants more than that - a soft black drop shadow AND a warmer,
closer glow under the letters, which is two different colours at two different offsets, not
one. So this widget builds the text as a QPainterPath and layers fills under it by hand.

Every colour is a PALETTE token from theme/__init__.py, restated here for the same reason
every other widget module restates its own tokens (see widgets/jobcard.py): the values are
pinned by src/gui/README.md and must not drift, so they are copied verbatim, not invented.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPainterPath
from PySide6.QtWidgets import QSizePolicy, QWidget

ACCENT = "#FF7A1A"
ACCENT_HI = "#FFA24D"
ACCENT_LO = "#C25A0F"

# Shadow passes, back to front: (dy, colour, alpha). Black first and furthest down for depth;
# then a closer, warmer pass in accent-lo so the shadow itself reads as black AND orange, not
# just a dark smear. Several near-identical offsets fake the softness a real blur would give,
# without pulling in QGraphicsBlurEffect for four letters of text.
_SHADOW_PASSES: tuple[tuple[float, str, int], ...] = (
    (5.0, "#000000", 55),
    (4.0, "#000000", 50),
    (3.0, "#000000", 65),
    (2.0, ACCENT_LO, 90),
    (1.0, ACCENT_LO, 70),
)

_PAD_X = 14.0             # left inset, and the slack kept on the right so nothing clips
_PAD_Y = 6.0              # vertical slack kept top and bottom
_MIN_PT, _MAX_PT = 14.0, 34.0


class BrandMark(QWidget):
    """One wordmark, painted, sized to FILL whatever width it is given - a fixed point size
    reads differently on every machine's default font, so "big enough" means "as big as the
    rail allows", found each time by measurement rather than guessed as one constant offset.
    """

    def __init__(self, text: str = "Cracken", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = text
        self.setFixedHeight(56)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _font(self) -> QFont:
        """The largest bold size, up to a cap, that fits this widget's current box."""
        avail_w = max(self.width() - 2 * _PAD_X, 10.0)
        avail_h = max(self.height() - 2 * _PAD_Y, 10.0)
        size = _MAX_PT
        while size > _MIN_PT:
            f = QFont(self.font())
            f.setBold(True)
            f.setPointSizeF(size)
            fm = QFontMetrics(f)
            if fm.horizontalAdvance(self._text) <= avail_w and fm.height() <= avail_h:
                break
            size -= 1.0
        f = QFont(self.font())
        f.setBold(True)
        f.setPointSizeF(size)
        return f

    def text_path(self) -> tuple[QPainterPath, QFontMetrics]:
        """The glyph outline, positioned once, so painting and any future sizing agree."""
        f = self._font()
        fm = QFontMetrics(f)
        baseline = (self.height() + fm.ascent() - fm.descent()) / 2.0
        path = QPainterPath()
        path.addText(_PAD_X, baseline, f, self._text)
        return path, fm

    def sizeHint(self):  # noqa: ANN201 - Qt override
        from PySide6.QtCore import QSize
        return QSize(int(2 * _PAD_X + 120), self.height())

    def paintEvent(self, ev) -> None:  # noqa: ANN001 - Qt event type
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path, _fm = self.text_path()

        for dy, color, alpha in _SHADOW_PASSES:
            p.save()
            p.translate(0.0, dy)
            c = QColor(color)
            c.setAlpha(alpha)
            p.fillPath(path, c)
            p.restore()

        bounds = path.boundingRect()
        grad = QLinearGradient(0.0, bounds.top(), 0.0, bounds.bottom())
        grad.setColorAt(0.0, QColor(ACCENT_HI))
        grad.setColorAt(1.0, QColor(ACCENT))
        p.fillPath(path, grad)
        p.end()


def demo() -> None:
    """Self-check: the glyph path is real and sized plausibly, and painting does not crash."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    w = BrandMark("Cracken")
    w.resize(150, 44)
    path, fm = w.text_path()
    assert not path.isEmpty(), "no glyphs - the font failed to render anything"
    b = path.boundingRect()
    assert b.width() > fm.horizontalAdvance("C"), "path narrower than one letter - broken text"
    assert b.height() > 0

    w.show()
    w.grab()          # forces a real paintEvent; catches painter/gradient blowups
    print("brandmark.demo: ok, path %dx%d" % (b.width(), b.height()))
    del app


if __name__ == "__main__":
    demo()
