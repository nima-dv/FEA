r"""
Build the two deliverable web pages from the figures and results on disk.

Two audiences, two documents:
  dossier.html  - the technical explainer: the physics, the method, every result and every
                  caveat. Written so someone who did not run the work can defend it.
  brief.html    - the decision brief: evidence, limiting factors, a recommendation, and
                  costed next steps.

Images are inlined as data URIs because the artifact host blocks external requests. Keep the
total under 16 MB; `du -b` the asset list if it starts creeping.

Both pages carry a SLIDE MAP at the end, so the PowerPoint can be assembled without going
back to source.

RUN
  ./run.ps1 python3 viz/build_artifacts.py
"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "results" / "artifacts"

# C5 is the notch-fill experiment. Set to a dict when it lands and re-run this script; the
# pages then swap the "pending" block for the result without any other edit.
C5 = ("<strong>C5 came back a second null, and more strongly than predicted.</strong> "
      "Filling the crack with k-Wave's material changed the imaged extent by nothing at all "
      "(<span class=\"n\">3.73</span> mm either way), the position by nothing "
      "(<span class=\"n\">38.09</span> mm), and the crack response by "
      "<span class=\"n\">0.07</span> dB. Predicted &quot;under 1 dB&quot;; delivered "
      "essentially zero, for the reason given in advance &mdash; a filled crack still reflects "
      "<span class=\"n\">98.9%</span> of what a free surface would, so almost no energy "
      "enters it to be got wrong. <strong>Both candidate explanations are now eliminated by "
      "experiment.</strong> We measure better and cannot yet attribute it; what remains "
      "untested sits inside k-Wave where we cannot instrument it &mdash; single precision, "
      "the absorbing layer, and its interface blending.")


# Figures that were not regenerated for this run. A pitch scoped to ONE scenario does not
# pay for the -20 deg, C4 and bandwidth-ladder solves, so rather than hard-failing we render
# a visible placeholder and list what is absent. A silent gap would be worse than an obvious
# one: the caption around a missing figure still claims a result.
MISSING: list[str] = []

_PLACEHOLDER = (
    "data:image/svg+xml;base64," + base64.b64encode(
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 200">'
        b'<rect width="800" height="200" fill="#e8ecf1" stroke="#b9c2ce" '
        b'stroke-dasharray="8 6" stroke-width="2"/><text x="400" y="95" '
        b'text-anchor="middle" font-family="system-ui,sans-serif" font-size="20" '
        b'fill="#6b7684">figure not generated for this run</text><text x="400" y="125" '
        b'text-anchor="middle" font-family="system-ui,sans-serif" font-size="15" '
        b'fill="#8b95a4">out of scope: single-scenario pitch</text></svg>').decode()
)


def uri(rel: str) -> str:
    p = RES / rel
    if not p.exists():
        MISSING.append(rel)
        return _PLACEHOLDER
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


# STANDING RULES for this asset set, so a later edit does not quietly drop them:
#   1. EVERY page carries the wavefield ANIMATION. A moving picture of the beam converting at
#      the inner wall and skipping to the crack is the single most convincing asset we have,
#      and it belongs in the decision brief as much as in the technical dossier. Use the
#      web-sized render (wavefield_web.gif) so the pages stay openable.
#   2. EVERY annotated comparison ships with its un-annotated twin - see the _clean keys below.
IMG = {
    "gif": "viz/wavefield_web.gif",
    "wave": "viz/wavefield_snap_p20deg_t27p0us.png",
    "mesh": "viz/mesh_zoom.png",
    "bw": "viz/bandwidth_convergence.png",
    "p20": "compare/compare_p20deg.png",
    "m20": "compare/compare_m20deg.png",
    # EVERY annotated comparison must ship with its un-annotated twin. Requested by the
    # research team: the wall arcs and the lime true-notch marker tell the reader where to
    # look, so nobody can judge unaided detectability with them on. Generate with
    # `repro/compare_images.py ... --no-overlay`. Keep this pairing for any figure added later.
    "p20_clean": "compare/compare_p20deg_nooverlay.png",
    "m20_clean": "compare/compare_m20deg_nooverlay.png",
    "base": "compare/baseline_subtract_20deg.png",
    # Artifact-reduction programme. "p20_legacy_clean" is the SAME data before the imaging
    # operator anti-aliasing was enabled - the before/after pair for section 05. The healthy
    # one is the defect-free wall under the shear-matched + sponge boundary variant, i.e. the
    # experiment that came back negative. Both un-annotated, per the twin rule.
    # Purpose-built by viz/artifact_reduction.py. The stock comparison figures CANNOT show
    # artifact reduction: each of their panels is normalised to its own maximum, and absolute
    # levels are not comparable between imaging chains. This one puts every panel on one
    # shared scale referenced to its own crack peak, and carries the sub-visual part as a
    # curve. Do not substitute a stock compare_*.png here.
    "artred": "viz/artifact_reduction.png",
    # Was toys/fluid_solid.py's mode_conversion.png; toys/ is gone. validation/zoeppritz.py
    # covers the same physics better - angle-resolved against the exact fluid-solid system
    # through both critical angles, rather than one normal-incidence coefficient.
    "zoep": "zoeppritz/amplitude.png",
    "pao": "cavity_scattering/dscf_vs_exact.png",
}

CSS = """
:root{
  --paper:#f6f7f9; --surface:#ffffff; --sunk:#eef1f5;
  --ink:#14181f; --ink-soft:#4a5462; --ink-faint:#6b7684;
  --rule:#d9dee6; --rule-soft:#e8ecf1;
  --fem:#1f6fb4; --kwave:#b4472f; --good:#2e7d5b; --warn:#9a6b0f; --dead:#7a8494;
  --display:Georgia,"Iowan Old Style","Times New Roman",serif;
  --body:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"DejaVu Sans Mono",monospace;
  --measure:70ch; --band:900px; --wide:1180px;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0f1319; --surface:#161b23; --sunk:#1c222b;
    --ink:#e7ebf1; --ink-soft:#9aa6b5; --ink-faint:#7c8899;
    --rule:#2a323d; --rule-soft:#222a34;
    --fem:#63a8e0; --kwave:#e37a5e; --good:#4fb185; --warn:#d6a63c; --dead:#8b95a4;
  }
}
:root[data-theme="dark"]{
  --paper:#0f1319; --surface:#161b23; --sunk:#1c222b;
  --ink:#e7ebf1; --ink-soft:#9aa6b5; --ink-faint:#7c8899;
  --rule:#2a323d; --rule-soft:#222a34;
  --fem:#63a8e0; --kwave:#e37a5e; --good:#4fb185; --warn:#d6a63c; --dead:#8b95a4;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--body); font-size:16.5px; line-height:1.72;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:var(--wide); margin:0 auto; padding:0 24px 96px}
/* ONE content width, no exceptions: prose, figures, tables, equations and captions all
   sit on --band, so no element steps in or out as the reader scrolls. If you add a block,
   give it --band too. --measure survives only as the reading-length reference. */
.col{max-width:var(--band)}
header.mast{border-bottom:1px solid var(--rule); padding:44px 0 26px; margin-bottom:40px}
.eyebrow{
  font-family:var(--mono); font-size:11.5px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--ink-faint); margin:0 0 14px;
}
h1{
  font-family:var(--display); font-weight:600; font-size:clamp(30px,4.4vw,46px);
  line-height:1.12; letter-spacing:-.01em; margin:0 0 16px; text-wrap:balance;
}
h2{
  font-family:var(--display); font-weight:600; font-size:26px; line-height:1.22;
  margin:56px 0 8px; text-wrap:balance; scroll-margin-top:16px;
}
h2 .num{
  font-family:var(--mono); font-size:13px; font-weight:400; color:var(--fem);
  letter-spacing:.06em; display:block; margin-bottom:6px;
}
h3{font-family:var(--body); font-weight:650; font-size:16.5px; margin:30px 0 6px; letter-spacing:.005em}
p{margin:0 0 15px}
.lede{font-size:19px; line-height:1.55; color:var(--ink-soft)}
a{color:var(--fem)}
strong{font-weight:650}
code,.n{font-family:var(--mono); font-variant-numeric:tabular-nums}
code{font-size:.9em; background:var(--sunk); padding:1px 5px; border-radius:3px}
ul,ol{margin:0 0 15px; padding-left:22px}
li{margin:0 0 7px}
hr{border:0; border-top:1px solid var(--rule-soft); margin:40px 0}

/* verdict / callout blocks */
.verdict{
  background:var(--surface); border:1px solid var(--rule);
  border-left:3px solid var(--fem); padding:22px 24px; margin:26px 0 34px;
}
.verdict.warn{border-left-color:var(--warn)}
.verdict.good{border-left-color:var(--good)}
.verdict.dead{border-left-color:var(--dead)}
.verdict h3{margin-top:0}
.verdict p:last-child{margin-bottom:0}
.kicker{
  font-family:var(--mono); font-size:11px; letter-spacing:.13em; text-transform:uppercase;
  color:var(--ink-faint); display:block; margin-bottom:8px;
}
.fine{ font-size:13.5px; color:var(--ink-faint); }
.ledger{max-width:var(--band)}

/* tables */
.tw{
  overflow-x:auto; max-width:var(--band); margin:28px 0 34px;
  border:1px solid var(--rule); background:var(--surface);
}
.eq{
  max-width:var(--band);
  font-family:var(--mono); font-size:15px; text-align:center;
  background:var(--sunk); border:1px solid var(--rule-soft);
  padding:14px 18px; margin:18px 0; overflow-x:auto; line-height:2.1;
}
.eq .lbl{display:block; font-family:var(--body); font-size:12.5px; color:var(--ink-faint);
  text-transform:uppercase; letter-spacing:.08em; margin-top:6px}
table{border-collapse:collapse; width:100%; font-size:14.5px}
caption{
  caption-side:top; text-align:left; padding:13px 16px 11px; color:var(--ink-soft);
  font-size:13.5px; border-bottom:1px solid var(--rule-soft);
}
th,td{padding:9px 14px; text-align:left; border-bottom:1px solid var(--rule-soft)}
thead th{
  font-family:var(--mono); font-size:11.5px; letter-spacing:.07em; text-transform:uppercase;
  color:var(--ink-faint); font-weight:400; vertical-align:bottom;
}
thead th{border-bottom:1px solid var(--rule)}
tbody tr:last-child td{border-bottom:0}
tbody tr:nth-child(even) td{background:var(--sunk)}
td.num,th.num{font-family:var(--mono); font-variant-numeric:tabular-nums; text-align:right}
.fem{color:var(--fem); font-weight:650}
.kw{color:var(--kwave); font-weight:650}
.tie{color:var(--ink-soft)}
.row-label{color:var(--ink-soft)}

/* figures */
figure{margin:28px 0 34px; max-width:var(--band)}
figure img{width:100%; height:auto; display:block; background:var(--surface); border:1px solid var(--rule)}
figcaption{font-size:13.5px; color:var(--ink-soft); margin-top:11px; max-width:var(--band)}
figcaption b{color:var(--ink); font-weight:650}

/* claim ledger */
.ledger{display:grid; gap:10px; margin:20px 0 30px}
.claim{
  display:grid; grid-template-columns:88px 1fr; gap:16px; align-items:start;
  background:var(--surface); border:1px solid var(--rule); padding:14px 16px;
}
.tag{
  font-family:var(--mono); font-size:10.5px; letter-spacing:.09em; text-transform:uppercase;
  padding:3px 7px; text-align:center; border:1px solid currentColor; white-space:nowrap;
}
.tag.yes{color:var(--good)} .tag.no{color:var(--kwave)} .tag.part{color:var(--warn)}
.tag.dead{color:var(--dead)}
.claim p{margin:0; font-size:14.5px}
.claim p+p{margin-top:5px; color:var(--ink-soft); font-size:13.5px}

/* slide map */
.slides{counter-reset:s; display:grid; gap:0; border:1px solid var(--rule); background:var(--surface)}
.slide{display:grid; grid-template-columns:44px 1fr; border-bottom:1px solid var(--rule-soft); padding:13px 16px}
.slide:last-child{border-bottom:0}
.slide::before{
  counter-increment:s; content:counter(s); font-family:var(--mono); font-size:12px;
  color:var(--ink-faint); padding-top:2px;
}
.slide b{display:block; font-size:15px; margin-bottom:3px}
.slide span{font-size:13.5px; color:var(--ink-soft)}
.slide em{font-family:var(--mono); font-style:normal; font-size:12px; color:var(--fem)}
footer{margin-top:64px; padding-top:22px; border-top:1px solid var(--rule); font-size:13.5px; color:var(--ink-faint)}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
"""


def page(title: str, desc: str, body: str) -> str:
    return (f'<title>{title}</title>\n<meta name="description" content="{desc}">\n'
            f"<style>{CSS}</style>\n<div class=\"wrap\">\n{body}\n</div>\n")


def write(name: str, html: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(html, encoding="utf-8")
    print(f"wrote {p}  ({p.stat().st_size/1048576:.2f} MB)")


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _dossier, _brief                                          # noqa: E402
    img = {k: uri(v) for k, v in IMG.items()}
    write("dossier.html", page(
        "FEM vs k-Wave: crack simulation dossier",
        "Technical dossier: the physics, the method, every result and every caveat of "
        "benchmarking an open-source FEM ultrasound simulation against MATLAB k-Wave.",
        _dossier.body(img, C5)))
    write("brief.html", page(
        "Should we keep going? Crack simulation decision brief",
        "Decision brief: evidence, limiting factors, cost and recommended next steps for the "
        "open-source crack-simulation benchmark.",
        _brief.body({k: img[k] for k in ("gif", "p20", "p20_clean", "m20", "m20_clean", "base", "mesh", "bw",
                        "artred")}, C5)))
    if MISSING:
        print("\nPLACEHOLDERS RENDERED - these figures do not exist on disk:")
        for m in MISSING:
            print(f"  - {m}")
        print("Either run the solve that produces them, or cut the section that cites them "
              "before this goes to anyone.")


if __name__ == "__main__":
    main()
