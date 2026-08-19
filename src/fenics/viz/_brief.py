r"""Body copy for the decision brief. Built by viz/build_artifacts.py.

Audience: someone deciding whether this project continues. They will not read a derivation,
but they will ask "what is it, why is it better, what does it cost".

Scope rules for this page, deliberate:
  * ONE beam angle (+20 deg) carries the results. The second angle exists and is in the
    dossier; a brief that reports both spends its length on hedging.
  * Every annotated comparison is followed by its UN-ANNOTATED twin, so the reader can judge
    unaided detectability. Requested by the research team.
  * The wavefield ANIMATION is on this page, not only in the dossier. It is the most
    convincing single asset in the project.
  * Current state only. No account of superseded configurations or corrected numbers - that
    history belongs in git and the branch README, not in front of an audience.
"""
from __future__ import annotations


def body(img: dict, c5) -> str:
    return f"""
<header class="mast">
  <p class="eyebrow">Decision brief &middot; DarkVision R&amp;D &middot; ILI crack sizing</p>
  <h1>An open-source finite-element simulation sizes cracks more accurately than our
  reference model</h1>
  <p class="lede col">We rebuilt the research team's ultrasound crack-detection simulation using
  open-source finite elements and benchmarked it against their MATLAB k-Wave model &mdash; same
  scenario, same imaging code, so the forward simulation is the only difference. It sizes a 4 mm
  crack to about half the error, and separates the crack from background clutter by a further
  2.4 dB.</p>
</header>

<div class="verdict good col">
  <span class="kicker">Recommendation</span>
  <p><strong>Continue, with narrow scope.</strong> The commercial case is not decibels &mdash; it
  is that crack <em>sizing</em> is what an inspection product sells, and we size to
  <span class="n">&minus;6.8%</span> where the reference model reads
  <span class="n">&minus;13%</span>. No licence cost, one workstation core, 2.4 hours a run.</p>
  <p>We would <em>not</em> fund a 3-D programme on this evidence. That is orders of magnitude
  beyond what has been demonstrated.</p>
</div>

<figure>
  <img src="{img['gif']}" alt="Animated ultrasound wavefield converting to shear and striking the crack">
  <figcaption><b>The simulation, running.</b> The pulse leaves the array as a compression wave in
  water, converts to a <span class="n">45&deg;</span> shear wave at the inner pipe wall, skips off
  the outer wall and scatters at the crack. That mode-converted shear path is the entire basis of
  the inspection. Rendered at element degree 3 and smoothed for file size; the measured results use
  degree 4.</figcaption>
</figure>

<h2 class="col"><span class="num">01</span>The reference model</h2>
<div class="col">
<p>The research team's simulation is <b>MATLAB with k-Wave</b>, a well-regarded acoustics package.
It lays a <b>uniform Cartesian grid</b> over the water gap and pipe wall &mdash;
<span class="n">50</span> micrometre pixels &mdash; and steps the wave forward in time, taking
spatial derivatives by Fourier transform. The method is genuinely strong: in smooth uniform
material it is more accurate per unit of computation than finite elements, and it needs only about
two grid points per wavelength.</p>
<p>The simulated inspection is the real one. A <span class="n">9.5</span> mm steel wall, a
<span class="n">20</span> mm water standoff, a <span class="n">256</span>-element array on a
<span class="n">0.30</span> mm pitch firing a <span class="n">4</span> MHz single-cycle pulse
steered <span class="n">20&deg;</span> off normal, and a <span class="n">4.0</span> mm deep notch
in the far (outer) wall. The output is one time trace per array element, which is then beamformed
into an image.</p>
<p><b>Its one structural limitation is geometry.</b> A grid cannot hold a hole. The crack has to be
painted onto pixels and filled with a fictitious soft material &mdash;
<span class="n">500</span> m/s, about <span class="n">2.5</span> points per wavelength at 4 MHz
&mdash; because there is no way to give a pixel a free surface.</p>
</div>

<h2 class="col"><span class="num">02</span>The alternative: finite elements</h2>
<div class="col">
<p>We built the same experiment in <b>FEniCS / DOLFINx</b>, an open-source finite-element
framework, in Python, in a container, on one CPU core. Both codes solve identical physics &mdash;
Newton's second law for a continuous solid, closed with Hooke's law:</p>
</div>
<div class="eq col">
&rho;&nbsp;&part;<sup>2</sup><b>u</b>/&part;t<sup>2</sup> = &nabla;&sdot;&sigma;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&sigma; = &lambda;(&nabla;&sdot;<b>u</b>)<b>I</b> + 2&mu;&nbsp;&epsilon;(<b>u</b>)
<span class="lbl">momentum balance, and the stress law it needs</span>
</div>
<div class="col">
<p>Two wave speeds fall out of that one equation, which is why a single model covers the whole
problem: a compression wave at
<span class="n">c<sub>P</sub></span> = &radic;((&lambda;+2&mu;)/&rho;), which is
<span class="n">5700</span> m/s in steel, and a shear wave at
<span class="n">c<sub>S</sub></span> = &radic;(&mu;/&rho;), <span class="n">3100</span> m/s. Water
is the <em>same</em> equation with <span class="n">&mu; = 0</span>: no shear stiffness, therefore
no shear wave, and compression at <span class="n">1500</span> m/s. The sensors record pressure,
which is <span class="n">p = &minus;&lambda;<sub>water</sub>&nbsp;&nabla;&sdot;<b>u</b></span>.</p>

<h3>What "finite element" actually does</h3>
<p>A second derivative cannot be applied to a piecewise polynomial &mdash; it does not have one.
So the equation is multiplied by a test function, integrated over the domain, and one derivative is
moved onto the test function by integration by parts. That leaves only first derivatives, which
piecewise polynomials do have, and turns the physics into linear algebra:</p>
</div>
<div class="eq col">
<b>M</b>&nbsp;<b>&uuml;</b> + <b>K</b>&nbsp;<b>u</b> = <b>0</b>
<span class="lbl">mass matrix, stiffness matrix, one vector of unknowns</span>
</div>
<div class="col">
<p>The domain is divided into <span class="n">45,711</span> cells; inside each the solution is a
polynomial of degree <span class="n">4</span>, giving <span class="n">1.47</span> million unknowns.
Putting the polynomial nodes at special (Gauss&ndash;Lobatto) points makes <b>M</b> diagonal, so a
time step is one sparse matrix&ndash;vector product with no system to solve. That is why a
1.5-million-unknown wave problem runs in <span class="n">2.4</span> hours on a single core.</p>
<p>The trade is honest in both directions. Finite elements need <em>more</em> unknowns than k-Wave
for the same accuracy in smooth material. What they buy is <b>geometry</b>.</p>
</div>

<h2 class="col"><span class="num">03</span>Meshing: the structural difference</h2>
<div class="col">
<p>k-Wave's grid is fixed and uniform: every pixel
<span class="n">50</span>&nbsp;&times;&nbsp;<span class="n">50</span> micrometres, everywhere,
whatever the geometry does. Ours is <b>unstructured</b> &mdash; cell edges are placed <em>on</em>
the true curved walls and on the crack faces, at any orientation, and cell <em>size</em> varies by
region.</p>
<p>Size comes from the wavelength: &lambda; = c/f, so the <em>slowest</em> material needs the
finest cells &mdash; water is meshed finer than steel, even though the crack is in the steel. Sizes
grade rather than step between regions, because an abrupt change in element size scatters a wave
much as a change in material does.</p>
</div>
<figure>
  <img src="{img['mesh']}" alt="Mesh detail at the curved inner wall and at the notch">
  <figcaption><b>Left: the inner wall. Right: the notch.</b> Cell edges sit on the true arc, so the
  wall is represented to <span class="n">0.05</span> micrometres. The same wall on a
  <span class="n">50</span> micrometre grid is a staircase carrying about
  <span class="n">140</span> micrometres of error &mdash; roughly
  <span class="n">700&times;</span> worse. The notch is cut as a genuine void: its faces are mesh
  boundaries with the physically correct traction-free condition, which is what steel against air
  is. No filler material.</figcaption>
</figure>
<div class="col">
<h3>What that is worth &mdash; today, and later</h3>
<p><b>Today, for this crack: very little &mdash; and we tested that rather than assuming it.</b>
Inside our own solver we staircased the wall onto k-Wave's grid, and separately filled the crack
with their fictitious material. Staircasing moved contrast by <span class="n">0.61</span> dB;
filling the crack moved the crack response by <span class="n">0.07</span> dB. Neither accounts for
the advantage we measure. For a simple on-axis rectangular notch, conformity is a
<em>capability</em> we hold rather than the reason we are ahead.</p>
<p><b>Where it should matter is geometry a grid represents badly</b> &mdash; a through or
side-drilled hole, an off-axis or branched crack, a weld cap, corrosion pitting. A grid must
approximate each as pixels and cannot give any of them a free surface, while an unstructured mesh
puts cell edges on the real boundary whatever its shape, and refines locally where the geometry is
tight instead of everywhere. That is a reasoned expectation, not a result: <b>we have simulated one
notch.</b> It is first on the next-steps list precisely because it is our weakest-evidenced claim
and the one most likely to become decisive.</p>
</div>

<h2 class="col"><span class="num">04</span>Results at +20&deg;</h2>
<div class="col">
<p>Both datasets pass through the research team's own imaging code, with the settings their own
script uses, so the forward simulation is the only variable. Ground truth is known because we chose
the notch: <span class="n">4.0</span> mm deep, at <span class="n">x = 38.25</span> mm.</p>
</div>
<figure>
  <img src="{img['p20']}" alt="k-Wave and FEM images at +20 degrees, annotated">
  <figcaption><b>Annotated.</b> k-Wave left, ours right. Grey arcs mark the true pipe walls and the
  lime marker the true notch, so the reader knows where to look.</figcaption>
</figure>
<figure>
  <img src="{img['p20_clean']}" alt="The same two images with no annotation overlay">
  <figcaption><b>The same two images, unannotated.</b> No wall arcs, no notch marker &mdash; what
  the images look like to someone judging detectability unaided, which is the fair way to look at
  them. Identical data to the figure above.</figcaption>
</figure>
<div class="tw col">
<table>
<caption><b>Head-to-head at +20&deg;.</b> Blue marks the better value.</caption>
<thead><tr><th>Metric (true value)</th><th class="num">k-Wave</th><th class="num">Ours</th></tr></thead>
<tbody>
<tr><td class="row-label">Crack depth (4.0 mm)</td>
  <td class="num">3.48 mm (&minus;13%)</td><td class="num fem">3.73 mm (&minus;6.8%)</td></tr>
<tr><td class="row-label">Crack position error (38.25 mm)</td>
  <td class="num">0.413 mm</td><td class="num fem">0.165 mm</td></tr>
<tr><td class="row-label">Crack visibility over clutter (RMS)</td>
  <td class="num">24.0 dB</td><td class="num fem">26.5 dB</td></tr>
<tr><td class="row-label">Crack visibility, worst-case clutter</td>
  <td class="num">11.6 dB</td><td class="num fem">14.1 dB</td></tr>
</tbody>
</table>
</div>
<figure>
  <img src="{img['base']}" alt="Cracked wall, defect-free wall, and the difference">
  <figcaption><b>The crack is a crack, not wall structure.</b> The same simulation on a defect-free
  wall puts no feature in the crack region at all; the cracked run puts
  <span class="n">+18.4</span> dB there. Subtracting the two isolates the crack and darkens the
  wall.</figcaption>
</figure>

<h2 class="col"><span class="num">05</span>Wins</h2>
<div class="ledger col">
  <div class="claim"><span class="tag yes">Win</span><div>
    <p><b>Sizing &mdash; about half the error.</b></p>
    <p><span class="n">&minus;6.8%</span> against <span class="n">&minus;13%</span> on a 4 mm
    notch. This is the commercially decisive number: an inspection call is a depth, and depth
    drives the fitness-for-service decision.</p></div></div>
  <div class="claim"><span class="tag yes">Win</span><div>
    <p><b>Contrast &mdash; <span class="n">+2.4</span> dB, unanimous.</b></p>
    <p>Better at every clutter-guard distance tested, at both beam angles. It is also a
    <em>floor</em> rather than a converged value &mdash; still improving when we stopped
    refining.</p></div></div>
  <div class="claim"><span class="tag yes">Win</span><div>
    <p><b>Position &mdash; inside one image pixel.</b></p>
    <p><span class="n">0.165</span> mm against <span class="n">0.413</span> mm. Quote it as
    "sub-pixel", not as a ratio: the image pixel is <span class="n">0.248</span> mm, so the ratio
    flatters us.</p></div></div>
  <div class="claim"><span class="tag yes">Win</span><div>
    <p><b>The physics is verified against exact answers.</b></p>
    <p>Mode conversion at the steel interface against the exact fluid&ndash;solid solution:
    <span class="n">0.8%</span> error at the production angle, shear angle right to
    <span class="n">0.55&deg;</span>. Scattering from a defect against an exact series solution:
    about <span class="n">1%</span>. Arrival-time error <span class="n">0.001%</span>. No fitted
    parameters anywhere. The reference work contains no convergence, stability or dispersion study
    to compare against.</p></div></div>
  <div class="claim"><span class="tag yes">Win</span><div>
    <p><b>It survives being measured differently.</b></p>
    <p>Every geometric claim was re-measured across six brightness thresholds, four
    analysis-region widths and five clutter distances. We are better in
    <span class="n">14</span> of <span class="n">15</span>. A result that flips with the threshold
    is not a result.</p></div></div>
  <div class="claim"><span class="tag yes">Win</span><div>
    <p><b>Cost and licensing.</b></p>
    <p>No licences. One CPU core, <span class="n">2.4</span> hours per beam angle, on a developer
    workstation. Every figure here regenerates from committed code.</p></div></div>
</div>

<h2 class="col"><span class="num">06</span>What we are not claiming</h2>
<div class="ledger col">
  <div class="claim"><span class="tag part">Bounded</span><div>
    <p>Sizing is a <span class="n">+20&deg;</span> win, not an every-angle win.</p>
    <p>At <span class="n">&minus;20&deg;</span> both models size the notch at
    <span class="n">3.85</span> mm &mdash; an exact tie. Contrast stays ahead at both
    angles.</p></div></div>
  <div class="claim"><span class="tag part">Bounded</span><div>
    <p>Two beam angles is <b>n = 2</b>.</p>
    <p>Enough to show the result is not a single-angle accident; not enough to characterise angle
    dependence. <span class="n">0&deg;</span> does not count &mdash; normal incidence produces
    almost no mode conversion, so <em>neither</em> model images anything meaningful
    there.</p></div></div>
  <div class="claim"><span class="tag no">Open</span><div>
    <p><b>Why</b> we are more accurate is not settled.</p>
    <p>The two obvious explanations were tested inside our own solver and both eliminated
    (<span class="n">0.61</span> dB and <span class="n">0.07</span> dB). "We measure better and
    cannot yet fully attribute it" is the honest position, and a stronger one than an explanation
    that collapses under questioning.</p></div></div>
  <div class="claim"><span class="tag no">Open</span><div>
    <p>Both images carry clutter; neither wall images black.</p>
    <p>Ours is the cleaner of the two, but a residual edge artefact remains unexplained after five
    candidate causes were tested and eliminated. Part of it is a true grating lobe, set by the array
    pitch and steering angle, and present in <em>both</em> models.</p></div></div>
  <div class="claim"><span class="tag dead">Not attempted</span><div>
    <p>3-D, complex defect shapes, and speed.</p>
    <p>All 2-D, one notch geometry. The unstructured-mesh advantage argued in section 03 is
    <b>untested</b>. And do not promise parallel speed-up: estimated at 4&ndash;8&times;, measured
    at <span class="n">1.7&times;</span>, because the bottleneck is memory bandwidth rather than
    cores.</p></div></div>
</div>

<h2 class="col"><span class="num">07</span>Cost and next steps</h2>
<div class="tw col">
<table>
<caption>Measured, not estimated</caption>
<thead><tr><th>Item</th><th class="num">Cost</th></tr></thead>
<tbody>
<tr><td class="row-label">Software licences</td><td class="num fem">none</td></tr>
<tr><td class="row-label">One production simulation, one beam angle</td>
  <td class="num">~2.4 h, 1 CPU core</td></tr>
<tr><td class="row-label">Hardware</td><td class="num">one developer workstation</td></tr>
</tbody>
</table>
</div>
<div class="tw col">
<table>
<caption>Proposed next steps, cheapest first</caption>
<thead><tr><th>Step</th><th class="num">Cost</th><th>What it delivers</th></tr></thead>
<tbody>
<tr><td class="row-label"><b>A defect geometry a grid handles badly</b></td>
  <td class="num">~2.4 h</td>
  <td><b>Do this first.</b> A hole, or an off-axis or branched crack. It is the one place the
  method difference should be decisive rather than incremental, and it is currently our
  weakest-evidenced claim.</td></tr>
<tr><td class="row-label">A third and fourth beam angle</td><td class="num">~2.4 h each</td>
  <td>Takes angle behaviour from two points to four and settles whether the sizing win is general
  or specific to <span class="n">+20&deg;</span>.</td></tr>
<tr><td class="row-label">Convergence error bars</td><td class="num">~1 day</td>
  <td>Turns the contrast floor into a number with stated uncertainty. The reference work has none,
  so this is a differentiator rather than diligence.</td></tr>
<tr><td class="row-label">Crack-size sweep</td><td class="num">~2.4 h each</td>
  <td>Sizing accuracy versus true depth &mdash; the curve a customer actually buys.
  <b>Needs one run from the research team</b> at a second crack size to stay a
  head-to-head.</td></tr>
<tr><td class="row-label">3-D feasibility</td><td class="num">not yet</td>
  <td>Scope only after a solver speed-up. Do not fund from this evidence.</td></tr>
</tbody>
</table>
</div>
<div class="col">
<h3>The one thing we would ask of the research team</h3>
<p>A single extra k-Wave run at settings they already use &mdash; a defect-free wall, or a second
crack size. It is a parameter change in a script they run routinely, and it converts two of our
self-consistency measurements into head-to-head comparisons.</p>
</div>

<footer>
  <p>Full derivations, the second beam angle and the complete record of caveats are in the companion
  dossier. Everything on this page reproduces from committed code: <code>src/fenics/</code> in
  <code>github.com/nima-dv/FEA</code>. Every figure is regenerated by a script; none is
  hand-drawn.</p>
</footer>
"""
