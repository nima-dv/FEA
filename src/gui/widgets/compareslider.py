r"""Two images, one draggable handle: the k-Wave baseline against ours, side by side.

This is the most persuasive interaction in the app, and it only stays persuasive if it is
scrupulous about three things:

  * BOTH images are drawn into the SAME target rect. A slider that fit each image to its own
    rect would put the notch at a different pixel column in each half, and the eye would read
    the misregistration as a difference between the solvers.
  * Neither image is stretched. Each keeps its own aspect ratio inside the shared rect and is
    centred there, so a 3:2 render never gets squashed to fit a 2:1 one.
  * Neither image is re-tinted. The backend's colormap carries meaning (dB re each panel's own
    max), so the pixmaps are blitted untouched - only the clip rect changes.

Scaled pixmaps are cached per target size because a drag repaints on every mouse move, and
rescaling a 2000 px figure per frame is the difference between smooth and sticky.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

ACCENT = "#FF7A1A"          # the only palette token this widget decides for itself
NUDGE = 0.02                # one arrow key press, in fraction of the image width


@dataclass
class _Side:
    """One half of the comparison: a pixmap, its label, and the cache of scaled copies."""
    label: str = ""
    pixmap: QPixmap | None = None
    _scaled: QPixmap | None = None
    _for: QSize | None = None

    def scaled(self, box: QSize) -> QPixmap | None:
        if self.pixmap is None or self.pixmap.isNull() or box.isEmpty():
            return None
        if self._for != box:
            # KeepAspectRatio, not IgnoreAspectRatio: no stretching, ever.
            self._scaled = self.pixmap.scaled(box, Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation)
            self._for = QSize(box)
        return self._scaled


class CompareSlider(QWidget):
    """Reveal image A left of the handle and image B right of it."""

    split_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._a = _Side()
        self._b = _Side()
        self._split = 0.5
        self._dragging = False
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.SplitHCursor)
        # Focusable so the arrow keys work; a mouse-only handle is unusable at fine offsets
        # and impossible for anyone driving the app from the keyboard.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ---- content -----------------------------------------------------------------------
    def set_images(self, path_a: str | Path | None, path_b: str | Path | None,
                   label_a: str = "A", label_b: str = "B") -> bool:
        """Load both sides. Returns True only if both pixmaps loaded."""
        self._a = _Side(label_a, QPixmap(str(path_a)) if path_a else None)
        self._b = _Side(label_b, QPixmap(str(path_b)) if path_b else None)
        self.update()
        return self.ready()

    def ready(self) -> bool:
        return all(s.pixmap is not None and not s.pixmap.isNull() for s in (self._a, self._b))

    def split(self) -> float:
        return self._split

    def set_split(self, frac: float) -> None:
        frac = min(1.0, max(0.0, float(frac)))
        if frac != self._split:
            self._split = frac
            self.split_changed.emit(frac)
            self.update()

    # ---- geometry ----------------------------------------------------------------------
    def fit_rect(self) -> QRect:
        """The shared target rect, sized by A's aspect ratio (B's if A is missing)."""
        box = self.contentsRect()
        src = next((s.pixmap for s in (self._a, self._b)
                    if s.pixmap is not None and not s.pixmap.isNull()), None)
        if src is None or box.isEmpty():
            return box
        sz = src.size().scaled(box.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRect(box.x() + (box.width() - sz.width()) // 2,
                     box.y() + (box.height() - sz.height()) // 2,
                     sz.width(), sz.height())

    def split_rects(self) -> tuple[QRect, QRect]:
        """(left, right) halves of the fit rect. Left is empty at 0.0, right empty at 1.0."""
        r = self.fit_rect()
        w = round(r.width() * self._split)
        return (QRect(r.x(), r.y(), w, r.height()),
                QRect(r.x() + w, r.y(), r.width() - w, r.height()))

    def handle_x(self) -> int:
        r = self.fit_rect()
        return r.x() + round(r.width() * self._split)

    # ---- interaction -------------------------------------------------------------------
    def _split_from_x(self, x: int) -> None:
        r = self.fit_rect()
        if r.width() > 0:
            self.set_split((x - r.x()) / r.width())

    def mousePressEvent(self, ev) -> None:  # noqa: ANN001 - Qt event type
        if ev.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._split_from_x(int(ev.position().x()))

    def mouseMoveEvent(self, ev) -> None:  # noqa: ANN001
        if self._dragging:
            self._split_from_x(int(ev.position().x()))

    def mouseReleaseEvent(self, ev) -> None:  # noqa: ANN001
        self._dragging = False

    def keyPressEvent(self, ev) -> None:  # noqa: ANN001
        k = ev.key()
        step = NUDGE * (5 if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1)
        if k in (Qt.Key.Key_Left, Qt.Key.Key_Down):
            self.set_split(self._split - step)
        elif k in (Qt.Key.Key_Right, Qt.Key.Key_Up):
            self.set_split(self._split + step)
        elif k == Qt.Key.Key_Home:
            self.set_split(0.0)
        elif k == Qt.Key.Key_End:
            self.set_split(1.0)
        else:
            super().keyPressEvent(ev)

    # ---- painting ----------------------------------------------------------------------
    def _blit(self, p: QPainter, side: _Side, rect: QRect, clip: QRect) -> None:
        pm = side.scaled(rect.size())
        if pm is None or clip.isEmpty():
            return
        # Centre each image in the SHARED rect, so both are registered on the same centre
        # even when their aspect ratios differ slightly.
        at = QRect(rect.x() + (rect.width() - pm.width()) // 2,
                   rect.y() + (rect.height() - pm.height()) // 2,
                   pm.width(), pm.height())
        p.save()
        p.setClipRect(clip)
        p.drawPixmap(at, pm)
        p.restore()

    def _pill(self, p: QPainter, text: str, rect: QRect, right: bool) -> None:
        if not text:
            return
        f = QFont(self.font())
        f.setPointSizeF(max(7.5, f.pointSizeF() - 0.5))
        p.setFont(f)
        fm = QFontMetrics(f)
        w, h = fm.horizontalAdvance(text) + 16, fm.height() + 8
        x = rect.right() - w - 8 if right else rect.left() + 8
        box = QRect(x, rect.top() + 8, w, h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(11, 11, 12, 200))          # scrim over the figure, not a re-tint
        p.drawRoundedRect(box, 3, 3)
        p.setPen(QColor("#E8E8EA"))
        p.drawText(box, int(Qt.AlignmentFlag.AlignCenter), text)

    def paintEvent(self, ev) -> None:  # noqa: ANN001
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().window())
        rect = self.fit_rect()
        if rect.isEmpty():
            return
        left, right = self.split_rects()
        self._blit(p, self._a, rect, left)
        self._blit(p, self._b, rect, right)

        if self.ready():
            self._pill(p, self._a.label, left if left.width() > 60 else rect, False)
            self._pill(p, self._b.label, right if right.width() > 60 else rect, True)

        x = self.handle_x()
        if rect.x() < x < rect.right():
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(ACCENT))
            p.drawRect(QRect(x - 1, rect.y(), 2, rect.height()))       # thin, 2 px
            cy = rect.y() + rect.height() // 2
            p.drawEllipse(QRect(x - 7, cy - 7, 14, 14))               # grab knob
        p.end()


def demo() -> None:
    """Self-check: split geometry at 0 / 50 / 100 %, against two real figures on disk."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    root = Path(__file__).resolve().parents[3] / "data" / "results" / "compare"
    a = root / "compare_p20deg_nooverlay.png"
    b = root / "compare_p20deg_widedomain_nooverlay.png"
    assert a.exists() and b.exists(), f"expected published figures under {root}"

    app = QApplication.instance() or QApplication([])
    w = CompareSlider()
    assert w.set_images(a, b, "baseline", "wide domain"), "both pixmaps must load"
    w.resize(800, 400)

    r = w.fit_rect()
    assert r.width() > 0 and r.height() > 0, r
    # Aspect preserved: the fit rect matches the source ratio to within a pixel of rounding.
    src = QPixmap(str(a)).size()
    assert abs(r.width() / r.height() - src.width() / src.height()) < 0.02, (r, src)

    w.set_split(0.0)
    lo, hi = w.split_rects()
    assert lo.width() == 0 and hi.width() == r.width(), (lo, hi)
    w.set_split(0.5)
    lo, hi = w.split_rects()
    assert abs(lo.width() - hi.width()) <= 1 and lo.width() + hi.width() == r.width()
    w.set_split(1.0)
    lo, hi = w.split_rects()
    assert lo.width() == r.width() and hi.width() == 0, (lo, hi)

    # keyboard nudge, and the clamp at both ends
    w.set_split(0.5)
    w.keyPressEvent(_key(Qt.Key.Key_Left))
    assert abs(w.split() - (0.5 - NUDGE)) < 1e-9, w.split()
    w.set_split(0.0)
    w.keyPressEvent(_key(Qt.Key.Key_Left))
    assert w.split() == 0.0

    w.grab()          # forces a full paintEvent; catches painter/geometry blowups
    print("compareslider.demo: ok")
    del app


def _key(key: Qt.Key):  # noqa: ANN201 - demo helper only
    from PySide6.QtGui import QKeyEvent
    return QKeyEvent(QKeyEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)


if __name__ == "__main__":
    demo()
