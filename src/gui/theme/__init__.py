r"""One place where a shade is defined.

QSS has no variables, so `dark.qss` is written with `$token` placeholders and substituted
here at load time. That keeps the palette in the README's table shape - one row, one value,
used everywhere - instead of the same hex string drifting between eight selectors.
"""
from __future__ import annotations

from pathlib import Path
from string import Template

# The DarkVision palette. Values are pinned by src/gui/README.md; do not invent shades.
# Keys use underscores because they are Template identifiers.
PALETTE: dict[str, str] = {
    "bg": "#0B0B0C",            # window
    "surface": "#16181B",       # panels, cards
    "surface_hi": "#1D2024",    # inputs, elevated
    "rule": "#2A2E33",          # borders, separators
    "ink": "#E8E8EA",           # primary text
    "ink_soft": "#9AA0A6",      # labels, captions
    "accent": "#FF7A1A",        # primary action, focus, progress
    "accent_hi": "#FFA24D",     # hover
    "accent_lo": "#C25A0F",     # pressed
    "ok": "#3FB950",
    "warn": "#E3B341",
    "fail": "#F0553A",
    "idle": "#6E7681",
    # QSS has no alpha function, so the 12% wash is spelled out as rgba once.
    "accent_wash": "rgba(255, 122, 26, 0.12)",
    "mono": '"Consolas", "DejaVu Sans Mono", monospace',
}

QSS_PATH = Path(__file__).with_name("dark.qss")


def qss() -> str:
    """The stylesheet with the palette substituted in."""
    return Template(QSS_PATH.read_text(encoding="ascii")).substitute(PALETTE)


def apply(app) -> None:
    """Style a QApplication. Takes the app rather than importing Qt here, so the palette
    dict stays importable by anything (tests, docs) without a Qt dependency."""
    app.setStyleSheet(qss())


def demo() -> None:
    text = qss()
    assert "$" not in text, "every placeholder must resolve"
    assert PALETTE["accent"] in text, "accent unused: the stylesheet is not wired up"
    for tok in PALETTE:
        assert f"${tok}" in QSS_PATH.read_text(encoding="ascii"), f"{tok} defined but unused"
    print("theme.demo: ok, %d chars" % len(text))


if __name__ == "__main__":
    demo()
