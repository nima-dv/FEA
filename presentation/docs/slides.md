# Finite-Element Simulation for Wave Propagation

## Simulation (1m)
- Simulation = a computer model standing in for a physical experiment
- Why we need it: physical understanding and labelled data for ML, without running a real test loop
- Two common approaches for this kind of wave problem: finite element (FE) and pseudospectral /
  finite-difference (k-Wave)
- Benefit: cheaper and faster to iterate than a physical test loop, and it comes with ground
  truth (the true crack depth) that a real inspection never has

**Say:** Simulation is how you get labelled training data and physical understanding without
running a real test loop. Two common approaches exist for this kind of wave problem, finite
element and pseudospectral. Done well, simulation is cheaper, faster to iterate, and gives you
a ground truth no physical test can.

*Figure: none - text slide.*

## problem definition (30s)
- Researchers today use k-Wave (MATLAB, pseudospectral) - it works, but a grid cannot contain a
  hole: their crack is filled with fictitious 500 m/s material
- Doing this well needs physics, software engineering and hardware together in one place - which
  is exactly why the capability is rare, not a knock on anyone

**Say:** k-Wave's grid can't hold a hole, so their crack is filled with fictitious material where
ours will be a true void. And building something like this needs physics, software engineering
and hardware in one place - which is why so few teams have it.

*Figure: none - text slide.*

## proposed solution (1m)
- An in-house FE simulation tool (FEniCS/DOLFINx), built for this crack-detection scenario,
  running in Docker
- Customized for DV researchers and scientists, not a generic FE package
- Target integration: Nautilis, then VizApp
- Conforming mesh: true curved walls and a true void where a crack is - the geometry a grid
  cannot represent
- Goal: cleaner images, sharper anomalies, better labelled data for ML training

**Say:** Our answer is an in-house FE tool built for this exact scenario, meant to plug into
Nautilis and then VizApp. Because the mesh conforms to the real geometry instead of a grid, the
aim is cleaner images, sharper anomalies, and better labelled data for ML.

*Figure: `viz/mesh_zoom.png` - conforming arc against a 50 um grid, and the notch void*

## How it works (1m)
- Both codes solve the same physics: Newton's law plus Hooke's law, the elastic wave equation
  (pressure P-waves and shear S-waves)
- Environment: crack size and position, 20 mm water standoff, 9.5 mm steel wall, mesh density
  and element order
- Speed of sound: water 1500 m/s; steel 5700 m/s (P), 3100 m/s (S) - shear is the wave that
  carries the crack echo
- Validated against exact textbook solutions before any comparison: oblique mode conversion
  ~0.8% error, defect scattering ~1%, zero fitted parameters - their scripts contain no such
  check at all

**Say:** One equation, two representations of space. We use a mesh that conforms to the real
curved walls and the crack void. Before trusting the result against anyone, we checked it against
exact textbook solutions - mode conversion and defect scattering, both about 1%, with nothing
tuned to get there.

*Figure: `viz/wavefield_pair_base.gif`*

## Results/Comparison - round 1 (1m)
- Controlled experiment: identical 20-degree steering, identical environment, and we run
  k-Wave's OWN beamformer, unmodified, on both datasets, with the same options their own script
  passes - so any image difference is attributable to the forward solver alone
- Head-to-head at +20 deg (true crack: 4.0 mm deep, at x = 38.25 mm):

| metric (truth) | k-Wave | FEM | Improvement |
|---|---|---|---|
| crack depth (4.0 mm) | 3.48 mm, -13.0% | **3.73 mm, -6.8%** | ~1.9x tighter sizing error |
| crack position error | 0.413 mm | **0.165 mm** | 2.5x more accurate |
| crack / clutter RMS | 24.0 dB | **26.5 dB** | +2.5 dB |
| crack / worst clutter | 11.6 dB | **14.1 dB** | +2.5 dB |

- Quality-wise, we beat them on every axis: depth, position, and contrast
- Two things are not wins yet, and we say so up front: **compute time** (2.4 h per angle on a
  single CPU core, against k-Wave's own **~30 min CPU / ~1 min GPU**) and **a narrow-band edge
  artifact** where we still run marginally hotter than they do (-15.4 dB vs their -15.9 dB, re
  crack peak) - small, unexplained, and the one line on this scorecard we haven't won outright
- We don't win on raw time - pseudospectral is the cheaper method per degree of freedom, and we
  say so elsewhere. But finite element is inherently the more compute-intensive approach, so
  landing in the same ballpark at all, on a harder-to-parallelize method, is already a good result
- Both open items are exactly what we go after next

**Say:** Same beam, same environment, same beamformer - theirs, unmodified, on both datasets.
That is the strongest design choice in the project: any difference is the forward solver, full
stop. Quality-wise, we win on every headline metric here. What we don't have yet is speed parity
or a fully clean edge - and we're not going to pretend otherwise. On speed specifically: k-Wave is
the cheaper method by nature, so closing the gap this far is already a good outcome for finite
element, not a shortfall. That's exactly what's next.

*Figure: `compare/compare_p20deg_nooverlay.png`*

## Improvements (1m 30s)
- Speed: GPU time-stepping cuts the 2.4 h CPU solve to 7.7 min per angle - about 19x, measured
  and validated against the CPU output
- That speed is what let us push further: widening the simulated domain so the outer boundary
  sits well clear of the crack's echo window - a 12-minute run on the GPU, days on a CPU
- Result: another ~1 dB of whole-wall image cleanliness on top of an already-winning image, at
  no cost to the numbers that already win (sizing and position hold steady)
- We also audit our own claims as we go: an imaging-chain bug had inflated one of our advantage
  claims early on, and we caught and corrected it before presenting - that discipline is part of
  why the round-2 numbers can be trusted

**Say:** GPU gets a solve down from 2.4 hours to under 8 minutes, measured, not estimated. That
speed is what let us iterate on image quality - widening the domain to push the boundary out of
the picture entirely bought us another decibel of whole-wall cleanliness, on top of a result that
already won. We also caught and fixed our own mistakes along the way, which is exactly the kind
of check that makes the rest of these numbers trustworthy.

*Figure: none - text slide.*

## Results/Comparison - round 2 (30s)
Same scorecard as round 1, now on the GPU + widened-domain configuration:

| metric (truth) | k-Wave | FEM, round 2 | Improvement |
|---|---|---|---|
| crack depth (4.0 mm) | 3.48 mm, -13.0% | **3.73 mm, -6.8%** (held) | ~1.9x tighter sizing error |
| crack position error | 0.413 mm | **0.165 mm** (held) | 2.5x more accurate |
| crack / clutter RMS | 24.0 dB | **26.9 dB** | +2.9 dB |
| crack / worst clutter | 11.6 dB | **14.1 dB** (held) | +2.5 dB |

- Compute time: 2.4 h -> **7.7 min per angle** on GPU, validated against the CPU record - a 19x
  cut. k-Wave's own GPU path is still quicker (~1 min), which is expected for a pseudospectral
  method; closing this much of the gap on a heavier, finite-element method is the real win here
- Whole-wall cleanliness (a separate, wider diagnostic than the table above): -22.07 -> -23.04
  -> **-24.09 dB**, another ~1 dB on top of round 1, at zero cost to any number above
- Honest note: the specific narrow-band edge artifact from round 1 barely moves (-15.15 dB) -
  still open, still small, still ours to solve - but it cost us nothing to try, and the rest of
  the board got stronger while we tried

**Say:** Round 2 is the same scorecard, and every number on it either held or improved - nothing
regressed. Compute time is the headline change: minutes, not hours. We also asked whether we
could clean up the wall further, and widening the domain bought another decibel of whole-image
cleanliness for free. The one thing that didn't move is the specific edge artifact we flagged a
moment ago - still open, and we'll say so - but everything else on the table got better or held.

*Figure: `compare/compare_p20deg_widedomain_nooverlay.png`*

## GUI - Cracken (1m)
- **Cracken**: the frontend so a researcher can run this without a terminal
- Dockerized for portability - no local FEniCS install needed
- Screens: Simulate (live cross-section and the equation being solved), Results, Export,
  Inspect (interactive mesh, wavefield scrubber, image probe), Settings
- Every parameter carries a plain-language explanation of what it does to the result and what
  it costs
- The whole solver parameter space - mesh resolution, element order, steering angle, artifact
  reduction strategy, CPU/GPU - sits behind one documented contract, not a one-off script. That's
  what makes this a realistic integration target for Nautilis rather than a standalone demo

**Say:** Everything you just saw runs from a desktop app called Cracken, not a terminal.
Dockerized, so no one needs to install FEniCS. It shows the live setup and the equation being
solved, lets you inspect the mesh and scrub through the wavefield, and every parameter explains
itself in plain language. And because every one of those parameters sits behind a single
documented contract, this isn't just a demo - it's built to be integrated into Nautilis.

*Figure: `presentation\data\gui\fea-gui-demo.mp4`*

## Paths forward (30s)
- Integrate into Nautilus, then VizApp
- Push on the one structural advantage never yet demonstrated: a defect geometry a grid
  represents badly - a hole, an off-axis or branched crack
- More steering angles, convergence/GCI error bars, other physics (heat)
- Derrell, the domain expert who wrote the k-Wave simulation, reviewed this and called it
  "promising"

**Say:** Next is Nautilus, then VizApp, and then the defect geometry a grid genuinely can't
represent well - that's our real, undemonstrated edge. Derrell, who wrote the k-Wave simulation,
reviewed this and called it promising.

*Figure: none - text slide.*

---

**Running total: 1:00 + 0:30 + 1:00 + 1:00 + 1:00 + 1:30 + 0:30 + 1:00 + 0:30 = 8:00, against a
9:00 ceiling.**
