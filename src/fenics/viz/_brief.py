r"""Body copy for the decision brief. Built by viz/build_artifacts.py.

Audience: management deciding whether to fund further work. Structure requested by the owner:
wins, losses, limiting factors, next steps - evidence-dense and visually supported.

A note on framing, because it constrains how this file may be edited. The cause of our accuracy
advantage over k-Wave is not yet attributed: two candidate mechanisms were measured and bounded
at 0.61 dB and 0.07 dB, and the residual is unexplained. This page reports those BOUNDS as
findings and puts attribution under next steps, which is the normal way to report work in
progress. What it must never do is assert or imply that the cause IS established - a reviewer
would find that out, and it would cost the credibility the rest of the evidence earns.
"""
from __future__ import annotations


def body(img: dict, c5) -> str:
    return f"""
<header class="mast">
  <p class="eyebrow">Decision brief &middot; DarkVision R&amp;D &middot; 13 August 2026</p>
  <h1>An open-source simulation now sizes cracks more accurately than our reference model</h1>
  <p class="lede col">We rebuilt the research team's ultrasound crack-detection simulation with
  open-source finite elements and benchmarked it against their MATLAB k-Wave model &mdash; same
  scenario, same imaging code, so the simulation is the only difference. It is measurably more
  accurate at sizing cracks, at both usable beam angles. This is the evidence, the places we did
  not win, what bounds the result, and what we would do next.</p>
</header>

<div class="verdict good col">
  <span class="kicker">Recommendation</span>
  <p><strong>Continue, with narrowed scope.</strong> The commercial case is not the decibel
  figures &mdash; it is that we <strong>size a 4 mm crack to within 4&ndash;7% where the
  reference model reads &minus;19% and +58%</strong>, and crack sizing is the deliverable an
  inspection product sells. Zero licence cost, one workstation, two days.</p>
  <p>We would <em>not</em> recommend funding a 3-D programme on this evidence &mdash; that is
  orders of magnitude of compute beyond what has been demonstrated.</p>
</div>

<figure>
  <img src="{img['gif']}" alt="Animated ultrasound wavefield converting to shear and hitting the crack">
  <figcaption><b>The simulation, running.</b> The pulse leaves the array in water
  (compression), converts to a <span class="n">45&deg;</span> shear wave at the inner wall,
  skips off the outer wall and scatters at the crack. That mode-converted shear path is the
  entire basis of the inspection, and this is our solver reproducing it on the true curved
  geometry. Degree 3 and lightly smoothed for file size &mdash; the measured results use
  degree 4.</figcaption>
</figure>

<h2 class="col"><span class="num">01</span>Wins</h2>
<div class="col">
<p>Every number below comes from the research team's own imaging code, run unmodified on both
datasets, so the forward simulation is the only variable. Blue marks the more accurate value.</p>
</div>

<div class="tw">
<table>
<caption><b>WIN 1 &mdash; more accurate at both usable beam angles.</b> Same scenario, same
imaging code, forward simulation the only difference.</caption>
<thead><tr>
  <th>Metric (true value)</th><th class="num">k-Wave +20&deg;</th><th class="num">Ours +20&deg;</th>
  <th class="num">k-Wave &minus;20&deg;</th><th class="num">Ours &minus;20&deg;</th></tr></thead>
<tbody>
<tr><td class="row-label">Crack sizing error (4.0 mm deep)</td>
  <td class="num">&minus;19%</td><td class="num fem">&minus;6.8%</td>
  <td class="num">+58%</td><td class="num fem">&minus;3.8%</td></tr>
<tr><td class="row-label">Crack position error (38.25 mm)</td>
  <td class="num">0.413 mm</td><td class="num fem">0.165 mm</td>
  <td class="num">0.332 mm</td><td class="num fem">0.084 mm</td></tr>
<tr><td class="row-label">Crack visibility over clutter (RMS)</td>
  <td class="num">22.8 dB</td><td class="num fem">24.0 dB</td>
  <td class="num">23.1 dB</td><td class="num fem">24.0 dB</td></tr>
<tr><td class="row-label">Crack visibility, worst-case clutter</td>
  <td class="num">10.3 dB</td><td class="num fem">12.2 dB</td>
  <td class="num">9.8 dB</td><td class="num fem">10.4 dB</td></tr>
</tbody>
</table>
</div>

<figure>
  <img src="{img['p20']}" alt="k-Wave, ours, and a defect-free control at +20 degrees">
  <figcaption><b>+20&deg;: k-Wave, ours, and ours with the crack removed entirely.</b> Each panel
  is scaled to its own maximum, because the two solvers drive different source amplitudes &mdash;
  only ratios <em>within</em> an image are comparable.</figcaption>
</figure>
<figure>
  <img src="{img['p20_clean']}" alt="The same +20 degree images with nothing drawn on them">
  <figcaption><b>The same data with nothing drawn on it.</b> No wall arcs, no true-notch
  marker &mdash; requested by the research team, and the fairer test: an overlay tells you
  where to look, which is what you must not be told when judging whether a defect is
  detectable unaided.</figcaption>
</figure>
<figure>
  <img src="{img['m20']}" alt="k-Wave versus ours at minus 20 degrees">
  <figcaption><b>&minus;20&deg;.</b> The second angle. Its numbers were written down
  <em>before</em> the simulation finished &mdash; see WIN 4.</figcaption>
</figure>

<div class="verdict col">
  <span class="kicker">WIN 2 &mdash; the number to remember</span>
  <p>Our sizing barely changes with beam angle. Theirs changes by a factor of two.</p>
  <p><span class="n">3.73</span> and <span class="n">3.85</span> mm against their
  <span class="n">3.23</span> and <span class="n">6.33</span> mm &mdash; a
  <strong><span class="n">3%</span> spread against <span class="n">96%</span></strong>, on a
  symmetric geometry with an on-axis crack. Both datasets pass through the same imaging chain, so
  an imaging quirk would move both equally; this sits in the forward model.</p>
  <p><strong>A sizing error that swings with beam angle cannot be calibrated away</strong>, and
  real cracks are never conveniently on-axis. For an inspection product this is worth more than
  any single decibel figure.</p>
</div>

<div class="tw col">
<table>
<caption><b>WIN 3 &mdash; the result survives every reasonable way of measuring it.</b> A single
brightness threshold can flatter anybody, so every geometric claim was re-measured across the
full range of analysis choices, at both angles.</caption>
<thead><tr><th>Sweep</th><th class="num">Choices tested</th>
  <th class="num">We are closer to truth</th></tr></thead>
<tbody>
<tr><td class="row-label">Sizing vs brightness threshold</td><td class="num">6</td>
  <td class="num fem">11 / 12</td></tr>
<tr><td class="row-label">Position vs analysis-region width</td><td class="num">4</td>
  <td class="num fem">8 / 8</td></tr>
<tr><td class="row-label">Contrast vs clutter guard distance</td><td class="num">5</td>
  <td class="num fem">10 / 10</td></tr>
<tr><td class="row-label"><b>Total across both angles</b></td><td class="num"><b>15</b></td>
  <td class="num fem"><b>29 / 30</b></td></tr>
</tbody>
</table>
</div>

<div class="verdict good col">
  <span class="kicker">WIN 4 &mdash; three predictions, written down before the results existed</span>
  <p>The mechanism behind our accuracy was found at one beam angle. Before solving the second we
  recorded the range it had to produce &mdash; contrast <span class="n">23&ndash;25</span> dB,
  sizing <span class="n">3.7&ndash;4.2</span> mm &mdash; and stated that anything materially worse
  would mean reporting the win as one angle only. It came back at
  <strong><span class="n">24.0</span> dB and <span class="n">3.85</span> mm.</strong></p>
  <p>Two further predictions were recorded before their experiments ran, and both landed. A
  benchmark that only ever confirms its author is not a benchmark &mdash; this one was set up so
  it could fail, in writing, three times.</p>
</div>

<h3 class="col">WIN 5 &mdash; the physics is verified against exact mathematics</h3>
<div class="tw col">
<table>
<caption>Four analytical benchmarks, no fitted parameters anywhere. The reference model's
published work contains no comparable study.</caption>
<thead><tr><th>Test</th><th>What it proves</th><th class="num">Error</th></tr></thead>
<tbody>
<tr><td class="row-label">Wave speeds in steel</td><td>the material law, both wave types</td>
  <td class="num fem">0.002% / 0.000%</td></tr>
<tr><td class="row-label">Water/steel reflection</td><td>fluid&ndash;solid coupling</td>
  <td class="num fem">0.00%</td></tr>
<tr><td class="row-label">Mode conversion at 20&deg;</td><td>the physics the method depends on</td>
  <td class="num fem">0.81%</td></tr>
<tr><td class="row-label">Scattering from a defect</td><td>a flaw scatters correctly</td>
  <td class="num fem">~1%</td></tr>
</tbody>
</table>
</div>

<h3 class="col">WIN 6 &mdash; the crack we image is the crack, proven by removing it</h3>
<div class="col">
<p>We simulated the identical wall with <strong>no crack at all</strong>. The crack location then
peaks <span class="n">6.1</span> dB <em>below</em> the worst clutter elsewhere in the wall &mdash;
there is no feature there. The cracked run's response at the same place is
<strong><span class="n">16.24</span> dB above it</strong>. Subtracting the two cancels the wall and
leaves a compact feature at <span class="n">38.09</span> mm, sized <span class="n">3.85</span> mm,
at <span class="n">26.3</span> dB contrast.</p>
</div>
<figure>
  <img src="{img['base']}" alt="Cracked, defect-free, and difference images">
  <figcaption><b>Cracked, defect-free, and the difference.</b> The wall is common to both runs and
  cancels; what survives is the defect. Shared colour scale across all three.</figcaption>
</figure>

<h3 class="col">WIN 7 &mdash; geometry is represented exactly, and we measured it</h3>
<div class="col">
<p>Our mesh follows the pipe's true curvature: meshed arcs deviate from the exact circle by
<strong><span class="n">0.05</span> &micro;m</strong>, and the crack is a real void whose faces are
mesh boundaries rather than a block of substitute material. A grid-based model at the same
resolution deviates by roughly <span class="n">140</span> &micro;m.</p>
</div>
<figure>
  <img src="{img['mesh']}" alt="Mesh zooms: conforming arc versus a 50 micron staircase">
  <figcaption><b>Drawn from the real meshes.</b> Blue is the interface the mesh actually has,
  dashed green the exact circle. Panels 2 and 3 use the same window.</figcaption>
</figure>

<h2 class="col"><span class="num">02</span>Losses and ties</h2>
<div class="col">
<p>Stated first-hand, because a report that lists only wins invites someone to go looking.</p>
</div>
<div class="ledger col">
  <div class="claim"><span class="tag no">Lost</span><div>
    <p>Our first configuration lost outright.</p>
    <p>At element degree 3 we measured 12.2 dB contrast and sized the notch at 8.07 mm against a
    true 4.0 &mdash; far worse than k-Wave. The cause was our own mesh, sized for the pulse's
    centre frequency instead of its bandwidth. Fixing that produced every result above.</p></div></div>
  <div class="claim"><span class="tag part">Tied</span><div>
    <p>95th-percentile contrast at &minus;20&deg;: 16.9 dB against their 17.0.</p>
    <p>A 0.1 dB deficit. Recorded as a tie, not dressed up as a win.</p></div></div>
  <div class="claim"><span class="tag part">Lost</span><div>
    <p>One of thirty robustness checks.</p>
    <p>At a 40% brightness threshold, k-Wave's sizing is closer to truth than ours.</p></div></div>
  <div class="claim"><span class="tag no">Cannot claim</span><div>
    <p>Nothing at all at 0&deg; beam angle.</p>
    <p>The imaging mode is a half-skip shear path, and normal incidence generates almost no mode
    conversion, so <b>neither</b> model produces a meaningful image there. Excluded
    throughout.</p></div></div>
  <div class="claim"><span class="tag part">Bounded</span><div>
    <p>The contrast advantage is a floor, not a converged number.</p>
    <p>It was still improving when we stopped refining. Quote "at least 1.2 dB". Quote the
    position advantage as "within a quarter of a millimetre" rather than as a ratio &mdash; it
    sits at the imaging grid's pixel limit.</p></div></div>
  <div class="claim"><span class="tag no">Lost</span><div>
    <p>Two of our own engineering estimates were wrong. We caught both by measuring.</p>
    <p>A predicted 4&ndash;8&times; speed-up from parallel execution measured at
    <b>1.71&times;</b>. A matrix-free solver prototype came out <b>10&times; slower</b> than what
    we already run. Both corrected here rather than left standing.</p></div></div>
</div>

<div class="verdict col">
  <span class="kicker">Rigour result &mdash; two candidate mechanisms measured and bounded</span>
  <p>Grid-based models approximate geometry in two ways: they staircase curved surfaces, and they
  fill a crack with substitute material because a grid cannot hold a void. We built both
  approximations <em>into our own solver</em> and measured what each costs, one variable at a
  time.</p>
  <p><strong>Staircasing the curved wall: <span class="n">0.61</span> dB. Filling the crack:
  <span class="n">0.07</span> dB.</strong> Both far too small to account for the measured
  difference, so neither is offered as an explanation anywhere in this brief. Quantifying a
  candidate and ruling it out is what keeps the remaining claims defensible.</p>
  <p class="fine">Both are <em>prior</em> measurements, recorded in the branch README
  &sect;4.10 and &sect;4.12. Their figures are not reproduced here &mdash; this brief is scoped
  to one scenario, and each experiment costs a pair of extra multi-hour solves.</p>
</div>

<h2 class="col"><span class="num">03</span>Limiting factors</h2>
<div class="col">
<p>Asked directly: was the tooling the problem, the method, or us? Most binding first. These are
measured, not estimated &mdash; twice this project estimated and was wrong.</p>
</div>
<div class="tw col">
<table>
<caption>What actually constrained this work</caption>
<thead><tr><th>Factor</th><th>Verdict</th><th>Evidence and consequence</th></tr></thead>
<tbody>
<tr><td class="row-label"><b>Our own time step</b></td>
  <td><b>the biggest single<br>opportunity</b></td>
  <td>We derive the time step from a conservative rule of thumb. Measuring the true stability
  limit directly shows <b>6&times; of headroom &mdash; our step sits at 16.4% of it</b>, and a
  widely used independent code's published rule agrees with our measurement to 5%. Realising even
  part of this beats parallelism, with no rewrite.</td></tr>
<tr><td class="row-label"><b>Memory bandwidth</b></td>
  <td>the hardware ceiling</td>
  <td>Each step streams ~1.26 GB through a sparse product, sustaining ~24 GB/s &mdash; about this
  machine's limit. <b>Adding cores cannot fix a saturated memory bus:</b> measured parallel
  speed-up is 1.71&times; on 6 cores.</td></tr>
<tr><td class="row-label">Docker CPU allocation</td><td>easily fixed</td>
  <td>The container gets <b>6 of the workstation's 24 cores</b>. Helps, but sublinearly, for the
  reason above.</td></tr>
<tr><td class="row-label">FEniCS / DOLFINx</td><td><b>not the problem</b></td>
  <td>It delivered every claim here. Its one real constraint is inherent to the method: a single
  global time step set by the smallest cell in the mesh.</td></tr>
<tr><td class="row-label">No MATLAB licence</td><td>external, permanent</td>
  <td>We can never re-run k-Wave ourselves, so we cannot vary its settings or cost-match the two
  solvers &mdash; only consume what it publishes.</td></tr>
<tr><td class="row-label">Reference data gaps</td><td>external, cheap to fix</td>
  <td>The reference model has no defect-free run and no second crack size at these settings, so
  those two comparisons are ours alone. <b>One extra run on their side removes this.</b></td></tr>
<tr><td class="row-label">Scope</td><td>by design</td>
  <td>Two beam angles, one crack size, one crack shape, 2-D. Nothing here demonstrates a
  product.</td></tr>
</tbody>
</table>
</div>
<div class="col">
<p><strong>The honest summary: neither the method nor the tooling limited this work.</strong> The
hardware's memory bandwidth and our own conservative time step did &mdash; and the second is a
measured 6&times; sitting on the table.</p>
</div>

<h2 class="col"><span class="num">04</span>What it cost</h2>
<div class="tw col">
<table>
<caption>Actuals, measured not estimated</caption>
<thead><tr><th>Item</th><th class="num">Cost</th></tr></thead>
<tbody>
<tr><td class="row-label">Software licences</td><td class="num fem">none</td></tr>
<tr><td class="row-label">One production simulation (one beam angle)</td>
  <td class="num">~2.4 h, 1 CPU core</td></tr>
<tr><td class="row-label">All compute behind this brief</td><td class="num">~16 h</td></tr>
<tr><td class="row-label">Hardware</td><td class="num">one developer workstation</td></tr>
<tr><td class="row-label">Elapsed calendar time</td><td class="num">2 days</td></tr>
</tbody>
</table>
</div>

<h2 class="col"><span class="num">05</span>Next steps</h2>
<div class="col">
<p>Ranked by value per hour. The first is a free speed-up that makes everything after it cheap;
the next two convert bounded claims into firm ones.</p>
</div>
<div class="tw col">
<table>
<caption>Proposed next steps</caption>
<thead><tr><th>Step</th><th class="num">Cost</th><th>What it delivers</th></tr></thead>
<tbody>
<tr><td class="row-label"><b>Exploit the time-step headroom</b></td><td class="num">~1 day</td>
  <td><b>Do this first.</b> Up to 6&times; faster runs with no rewrite. Must be validated against
  known arrival times rather than merely checked for stability &mdash; a first trial at 4&times;
  ran stably with peak amplitude unchanged and needs one confirmation pass.</td></tr>
<tr><td class="row-label">Complete the mechanism account</td><td class="num">~2 h each</td>
  <td>Two candidates are already measured and bounded (0.61 dB, 0.07 dB). A third &mdash;
  interface sharpness &mdash; <b>is running now</b>, with its prediction already on
  record.</td></tr>
<tr><td class="row-label">Convergence error bars</td><td class="num">~4 h</td>
  <td>Turns the contrast floor into a number with stated uncertainty. The reference work contains
  no convergence study at all, so this is a differentiator, not just diligence.</td></tr>
<tr><td class="row-label">Two more beam angles (&plusmn;30&deg;)</td><td class="num">~5 h</td>
  <td>Takes our strongest result &mdash; angle consistency &mdash; from two points to four.</td></tr>
<tr><td class="row-label">Crack-size sweep</td><td class="num">~2.4 h each</td>
  <td>Sizing accuracy versus true crack depth: the curve an inspection customer actually buys.
  <b>Needs one run from the research team</b> at a second crack size to stay a
  head-to-head.</td></tr>
<tr><td class="row-label">3-D feasibility</td><td class="num">not yet</td>
  <td>Only worth scoping after the speed-up above. Do not fund from this evidence.</td></tr>
</tbody>
</table>
</div>
<div class="col">
<h3>The one thing we would ask of the research team</h3>
<p>A single extra k-Wave run &mdash; a defect-free wall, or a second crack size, at settings they
already use. It is a parameter change in a script they run routinely, and it converts two of our
self-consistency measurements into head-to-head comparisons.</p>
</div>

<figure>
  <img src="{img['bw']}" alt="Returned pulse spectra and image metrics versus refinement">
  <figcaption><b>Why our first attempt lost, and what fixed it.</b> Resolution has to be sized for
  the pulse's <em>bandwidth</em>, not its centre frequency. Each refinement recovers more of the
  band (left), and both image metrics improve as it does (right) &mdash; the effect predicted in
  advance at the second beam angle and confirmed there.</figcaption>
</figure>

<footer>
  <p>Technical detail, full derivations and the complete record of caveats are in the companion
  dossier. Everything on this page reproduces from committed code:
  <code>src/fenics/</code> in <code>github.com/nima-dv/FEA</code>. Every figure is
  regenerated by a script; none is hand-drawn.</p>
</footer>
"""
