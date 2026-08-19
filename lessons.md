# Lessons: the theory behind this project

Plain-language explanation of what is being simulated, the equations behind it, how the two
approaches differ, and how the comparison is made fair. Written to be understood and presented,
not to be rigorous.

Scope of this file: **concepts, equations and method that stay true regardless of results.**
Current status, the head-to-head numbers and open questions live in the branch README
(`F:\code\readme\rnd-nima-FEA-README.md`); slide-by-slide talking points live in `PITCH.md`.
Keep it that way - no scoreboard, no history, no changelog here. A few measured numbers do
appear below, but only where the number *is* the lesson - how fast dispersion falls with element
order, for instance. Anything that would need editing when a result changes belongs in the README.

---

## 0. Yes, this is wave propagation

Specifically **time-domain elastic wave propagation**: launch a short ultrasonic pulse and march
it forward in time through water and steel, watching it reflect, refract, convert between wave
types, and scatter off a crack.

Both codes do the same physics. They differ **only in how they approximate space.**

---

## 1. The physical problem

A robot inside an oil pipe. A 256-element ultrasound array sits 20 mm from the wall, in water.
The steel wall is 9.525 mm thick and curved (pipe ~400 mm across). A crack sits on the **far**
side - the outer wall - 4.0 mm deep and 1.0 mm wide, i.e. 42% through-wall. It must be found and
its depth measured from the echoes alone.

**The answer we need is a TIME, not a brightness.** Depth is inferred from when echoes arrive
along a known path. This is why arrival-time accuracy matters far more than amplitude accuracy
here, and it drives every numerical choice in the project.

---

## 2. The one equation both codes solve

**Newton's second law for a continuous solid** (momentum balance):

$$\rho \frac{\partial^{2}\mathbf{u}}{\partial t^{2}} = \nabla \cdot \boldsymbol{\sigma}$$

In words: *mass x acceleration = net internal force.* $\mathbf{u}$ is displacement (how far a speck of
material has moved), $\rho$ is density, $\boldsymbol{\sigma}$ is stress (internal force per unit area).

To close it you need a material law - **Hooke's law**, the 3-D version of "spring force is
proportional to stretch":

$$\boldsymbol{\sigma} = \lambda (\nabla \cdot \mathbf{u}) \mathbf{I} + 2\mu \boldsymbol{\varepsilon}
\qquad \text{where} \qquad
\boldsymbol{\varepsilon} = \tfrac{1}{2}\left(\nabla \mathbf{u} + \nabla \mathbf{u}^{\mathsf{T}}\right)$$

$\lambda$ and $\mu$ are the two stiffness constants (Lame parameters). $\mu$ is the **shear**
stiffness - resistance to sliding. $\varepsilon$ is strain.

**That is the entire model in both codes.** Substitute one into the other and you have a wave
equation.

### Why two kinds of wave fall out

The equation supports two independent modes:

| | motion relative to travel | speed | speed in steel | visible in |
|---|---|---|---|---|
| **P** (pressure, longitudinal) | along | $c_P = \sqrt{(\lambda + 2\mu)/\rho}$ | **5700 m/s** | $\nabla \cdot \mathbf{u}$ |
| **S** (shear, transverse) | across | $c_S = \sqrt{\mu/\rho}$ | **3100 m/s** | $\nabla \times \mathbf{u}$ |

Inverting those gives the constants from measured speeds, which is how material data is actually
supplied:

$$\mu = \rho c_S^{2} \qquad\qquad \lambda = \rho \left(c_P^{2} - 2 c_S^{2}\right)$$

They separate cleanly because of the **Helmholtz decomposition**: P is curl-free so it shows up
in $\nabla \cdot \mathbf{u}$, S is divergence-free so it shows up in $\nabla \times \mathbf{u}$. That is exactly how the wavefield
animation renders them - $\nabla \cdot \mathbf{u}$ in the water, $\nabla \times \mathbf{u}$ in the steel.

### Water is the same equation with $\mu = 0$

Set the shear stiffness to zero and Hooke's law collapses to $\boldsymbol{\sigma} = \lambda (\nabla \cdot \mathbf{u}) \mathbf{I}$ -
pure pressure, no shear. **That IS the acoustic wave equation.** So one equation and one solver
cover both water and steel; you only change $\mu$ per region. Water: 1500 m/s, $\mu = 0$.

The pressure the sensors record is therefore:

$$p = -\lambda_{\text{water}} \nabla \cdot \mathbf{u}$$

This matters for correctness: the k-Wave sensors record **pressure**, so ours must output
pressure too, not displacement.

### Impedance - why a reflection happens at all

A wave reflects at an interface because the two materials resist motion differently. The measure
is **acoustic impedance**:

$$Z = \rho c \qquad\qquad R = \frac{Z_2 - Z_1}{Z_2 + Z_1}$$

$R$ is the fraction of amplitude reflected at normal incidence. Two consequences worth carrying
around:

- **Water against steel is a big mismatch**, so most of the pulse bounces off the inner wall and
  only a modest fraction gets in. That is why the front-wall echo is the loudest thing in the
  record and why the crack echo has to be dug out of its shadow.
- **A crack is a nearly perfect reflector**, because on the far side there is air: $Z \approx 0$,
  so $R \approx -1$. This is also why filling a crack with a very soft fictitious solid - as a
  grid method must - is a smaller sin than it appears. The impedance contrast is still enormous, so
  $|R|$ is still close to 1 and almost no energy enters the filler to be got wrong. If you want to
  predict whether an approximation matters, compute the impedance it changes.

---

## 3. Why the beam is steered 20 degrees

This is the cleverest part of the inspection method and worth being able to explain.

Waves bend at an interface by **Snell's law**, which conserves the along-interface component of
slowness:

$$\frac{\sin \theta_{\text{water}}}{c_{\text{water}}} = \frac{\sin \theta_{P}}{c_{P}} = \frac{\sin \theta_{S}}{c_{S}}$$

Put in 20 degrees from water into steel:

- **P wave:** $\sin \theta_P = 5700 \sin 20^\circ / 1500 = 1.30$. Greater than 1 - **no solution.**
  The compression wave cannot propagate into the steel; it becomes evanescent and dies at the
  surface. It stops being possible past the **critical angle** $\arcsin(1500/5700) = 15.3^\circ$.
- **S wave:** $\sin \theta_S = 3100 \sin 20^\circ / 1500 = 0.707$, so $\theta_S = 45^\circ$.

**So a 20 degree tilt in water produces an almost pure SHEAR wave at 45 degrees in the steel.**
That is called **mode conversion**, and it is deliberate. The 45 degree shear beam skips off the
outer wall and strikes the crack side-on, where a narrow notch scatters strongly. Hit the same
crack head-on at 0 degrees and it is nearly invisible - and there is essentially no mode
conversion at normal incidence, so **0 degrees cannot test this method at all.**

### The array has its own physics: grating lobes

A phased array steers by delaying its elements, and that only works if the elements are close
enough together. If they are not, the delays that aim the beam at $\theta_s$ *also* bring every
element back into phase at a second angle, producing a **full-amplitude copy of the main beam**
pointing somewhere useless. The no-grating-lobe condition, for element pitch $d$, is:

$$d \;\le\; \frac{\lambda}{1 + |\sin \theta_s|}
\qquad\text{and if violated}\qquad
\sin \theta_g = \sin \theta_s - \frac{\lambda}{d}$$

Three things follow, and they matter when you are staring at an image wondering what is real:

- The condition tightens as you **steer further** and as **frequency rises** ($\lambda$ falls).
  A broadband pulse can therefore be grating-lobe-free at its centre frequency and not at its
  top end.
- A grating lobe is **not a sidelobe**, and **apodisation cannot remove it.** Tapering element
  amplitudes suppresses ordinary sidelobes, but at the alias angle every element is back in phase
  by definition, so a tapered array sums to the same thing there. The only real levers are lower
  steering angle, lower frequency, or finer pitch - all hardware.
- It is a property of the **instrument**, not of either simulation. If both models show a bright
  feature at the alias angle, that is the array being modelled correctly by both.

### What "TT-T" means

It is a **ray path**, written as a recipe. The letters before the dash are the **transmit** path,
the letter after it is the **receive** path. Each letter is one leg of the journey, and the letter
says which wave type that leg travels as:

```
L = Longitudinal  (compression, P)  5700 m/s in steel
T = Transverse    (shear, S)        3100 m/s in steel
```

So **TT-T** = transmit as **two shear legs**, receive as **one shear leg**:

```
   array  ==========================================   z = 0
             \                              ^
      water   \  P, 1500 m/s                |  P back up to the array
               v                            |
   ID  --------[1]-------------------------[4]------   <-- mode conversion: P -> 45 deg S
                 \                        /
      steel       \  T                   /  T   (receive, "direct")
                   \                    /
   OD  -------------[2]====[CRACK]====[3]-----------
                       half-skip along
                       the outer wall
```

- **Transmit, leg 1 (T):** the converted shear wave crosses the wall, inner to outer.
- **Transmit, leg 2 (T):** it takes a **half-skip** - one bounce off the outer wall - and arrives
  at the crack from the side. "Half" because it bounces once; a full skip would return to the
  inner wall first.
- **Receive (T):** what the crack scatters travels **direct** back to the inner wall, converts
  back to P in the water, and reaches the array.

**Why the recipe matters.** The beamformer does not search for echoes; it *assumes* this path and
computes how long it should take (§7). Feed it the wrong path and it looks up the wrong arrival
time, so the crack images at the wrong depth or not at all. The path is the measurement.

Consequences worth knowing:

- **`TT-T` and the other multi-bounce modes apply an angle filter** (pass 60&deg;, cut 80&deg;) to
  reject rays arriving at implausible angles. Simple modes like `TT` and `LL` do **not**. Applying
  the filter to the wrong mode silently invalidates a comparison.
- **One transmit event per steering angle.** This is not full-matrix capture - the whole array
  fires once as a steered plane wave, and all 128 receive channels are summed. That is why the
  steering angle is a first-class parameter of the scenario.
- **A backwall crack needs the skip.** The notch is cut into the outer wall, so its faces are
  roughly perpendicular to it. A 45&deg; shear wave and a perpendicular face form a corner
  reflector, which is why 45&deg; shear is the classic choice for far-wall cracks - and why
  getting $c_S$ and the 45&deg; right (§2, §3) matters more than any amplitude detail.

---

## 4. Two ways to approximate space

Both codes need spatial derivatives ($\nabla \cdot \boldsymbol{\sigma}$). That is the only place they diverge.

### k-Wave: pseudospectral (PSTD)

A **uniform square grid**, 50 um spacing. Derivatives are computed with the **Fourier
transform** - in Fourier space, differentiating is multiplying by $ik$. Each timestep is
FFT, multiply, inverse FFT.

- **Strength - genuinely excellent.** It uses *every* grid point to compute a derivative at one
  point, so it needs very few points per wavelength (approaching 2 in the ideal limit, where
  plain finite differences need 10-20). **Per unit of computation, in smooth uniform material,
  it beats FEM.** Say this openly; it is true and conceding it costs nothing.
- **Weakness - geometry must be painted onto a fixed grid.** A curved wall does not follow square
  pixels, and neither does a crack. Two consequences:
  - Curved walls are approximated. k-Wave mitigates this well: it computes the geometry
    analytically and blends it over about +-62.5 um, so it is **sub-pixel accurate, not a crude
    staircase**. Do not accuse it of staircasing.
  - **A grid cannot contain a hole.** The crack is physically air, but it has to be filled with
    fictitious material - 500 m/s compression, zero shear.

### Ours: finite element / spectral element (FEniCS/DOLFINx)

The domain is chopped into ~46,000 cells whose **edges lie exactly on the true geometry** - on
the curved arcs, on the crack faces. Inside each cell the solution is a polynomial of degree $p$
(we use $p = 4$).

- **Strength - geometry is exact**, and the crack is a genuine void whose faces get the
  physically correct condition: **traction-free**, meaning "nothing pushes back", which is what
  steel against air actually is. No fictitious filler.
- **Weakness - more unknowns for the same accuracy in smooth regions** (1.47 million here), and
  dispersion must be controlled by raising polynomial order.

### The weak form - why FEM is possible at all

You cannot apply a second derivative to a piecewise polynomial; it does not have one. So multiply
the equation by a **test function** $\mathbf{v}$, integrate over the domain, and integrate by parts to
move one derivative onto $\mathbf{v}$:

$$\int_{\Omega} \rho \ddot{\mathbf{u}} \cdot \mathbf{v} \, \mathrm{d}\Omega + \int_{\Omega} \boldsymbol{\sigma}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{v}) \, \mathrm{d}\Omega = \text{boundary terms}$$

Now only **first** derivatives appear anywhere. That trade is the whole reason the method works.
It also has a quiet bonus: **traction continuity across the water/steel interface, and the
traction-free crack faces, are the *natural* boundary conditions of this form** - they come for
free by dropping a boundary term, rather than being imposed. Displacement continuity is automatic
because the cells share nodes.

Discretising in space gives a matrix system:

$$\mathbf{M} \ddot{\mathbf{u}} + \mathbf{K} \mathbf{u} = \mathbf{0}$$

$\mathbf{M}$ is the mass matrix, $\mathbf{K}$ the stiffness matrix.

### Explicit time stepping and mass lumping

Central differences in time:

$$\mathbf{u}^{n+1} = 2\mathbf{u}^{n} - \mathbf{u}^{n-1} - \Delta t^{2} \mathbf{M}^{-1} \mathbf{K} \mathbf{u}^{n}$$

**Mass lumping** makes $\mathbf{M}$ diagonal, so $\mathbf{M}^{-1}$ is elementwise division - no linear solve per
step, just one sparse matrix-vector multiply. Placing the polynomial nodes at
**Gauss-Lobatto-Legendre (GLL)** points makes the lumped mass exactly diagonal *and* positive,
which is what keeps a high-order method cheap.

**There is a real constraint hidden in that sentence.** The trick works because GLL nodes double
as a quadrature rule: integrate the mass term at the same points where the polynomial is defined
and the off-diagonal entries vanish *exactly*. That coincidence holds on **quadrilaterals and
hexahedra**, where the basis is a tensor product of 1-D GLL bases - so a quadrilateral mesh is not
a stylistic preference, it is what makes the scheme legitimate. On **triangles and tetrahedra**
above low order there is no such coincidence: naive lumping loses accuracy, and doing it properly
needs a purpose-built element family with extra interior nodes. If you ever see a high-order
simplex mesh with a lumped mass, ask which family it uses.

Two rules follow from explicit stepping:

- **CFL condition.** $\Delta t$ must be small enough that a wave cannot cross a cell in one step,
  roughly $\Delta t \lesssim h / (c\, p^{2})$. Violate it and the solution explodes. One global $\Delta t$ is set by
  the **smallest cell over the fastest wave speed** - which is why refining one small region is
  expensive everywhere, and why the crack tip is deliberately *not* over-refined.
- **Dispersion.** See below.

---

## 5. Dispersion - the error that actually matters here

On any discretisation, the computed wave speed is slightly wrong, and the error depends on
frequency and direction. Different frequencies therefore drift apart as they travel, and
**arrival times shift.**

Since this inspection measures crack depth *by arrival time*, this is the dominant error. A speed
error becomes a depth error directly.

**Raising polynomial order crushes dispersion very fast** - much faster than refining the mesh.
Measured on one mesh: timing error **2.19% at degree 1, 0.011% at degree 2, 0.001% at degree 4.**
The cost is that the stable $\Delta t$ shrinks like $h/(c p^{2})$, so higher order means more steps, each
individually cheap.

**Order beats refinement for timing.** That is why this project is built on spectral elements.

### Why this solver is a good fit for a GPU, and how to tell in advance

Worth knowing as a general rule. One explicit step is a single sparse matrix-vector product
plus a few elementwise vector operations. Count the arithmetic against the bytes moved and the
answer is lopsided: roughly two floating-point operations per matrix entry, but twelve bytes
must be fetched to get that entry. The kernel is therefore **memory-bandwidth bound**, not
compute bound - it spends its life waiting for memory.

That single fact predicts a lot:

- **Adding cores barely helps.** Cores on one socket share a memory bus, so parallel speed-up
  saturates far below the core count.
- **Moving to a GPU helps enormously**, because a GPU's advantage over a CPU is far larger in
  memory bandwidth than in cores, and bandwidth is the binding constraint.
- **Double precision is nearly free here**, which is counter-intuitive on consumer hardware
  where fp64 *arithmetic* runs at a small fraction of fp32. If you are waiting for memory, the
  arithmetic rate is not what limits you - only the doubled byte count is.
- **Matrix-free methods look wrong on a CPU and right on a GPU.** They recompute element
  contributions instead of storing them, trading bandwidth for arithmetic. That is a bad trade
  when arithmetic is scarce and a good one when it is abundant.

The general lesson: before optimising, work out whether the kernel is starved of arithmetic or
of bandwidth. It tells you which hardware and which algorithm will help, and it explains why
two plausible-sounding speed-ups can have opposite outcomes.

### The bandwidth trap - the project's biggest single lesson

The transmitted pulse is only **one cycle** long. A short pulse is wide in frequency: a 1-cycle
4 MHz burst carries real energy out to 6-8 MHz - roughly 100% bandwidth.

The standard rule of thumb "4 nodes per wavelength" must therefore be satisfied at the pulse's
**upper usable frequency, not its centre frequency.** A mesh sized for 4 MHz gives only ~2.7
nodes per wavelength at 7 MHz, over a water path 53 wavelengths long. The mesh then acts as a
low-pass filter on your own pulse while it is in transit.

### How that fixes the cell size

The rule of thumb is the mesh design rule. Turn it around: with a target of $N$ nodes per
wavelength and a degree-$p$ element carrying $p$ node spacings across a cell,

$$h \;\approx\; \frac{p\,\lambda_{\min}}{N}, \qquad \lambda = \frac{c}{f}$$

Three consequences, and the first one surprises people:

- **The slowest material needs the finest cells.** $\lambda = c/f$, so water at 1500 m/s has a
  shorter wavelength than steel at 3100 m/s. The water gap is meshed finer than the steel wall,
  even though the crack we are hunting is in the steel.
- **The mesh is therefore non-uniform, and graded.** Each region gets a size from its own
  wavelength, and the transitions are ramped rather than stepped - an abrupt change in element
  size scatters a wave much like a change in material does, and it also produces badly shaped
  cells.
- **Order is the cheaper knob than refinement.** Resolution goes as $p\,\lambda/h$, and
  dispersion falls much faster with $p$ than with $h$ - while the stable step shrinks like
  $h/(c p^{2})$, so halving $h$ doubles the number of steps.

Two places where wavelength is deliberately *not* the criterion. At the **transducer face** the
mesh must resolve individual array elements, which is a geometry requirement finer than any
wavelength demands. At the **crack** it would be a mistake to refine: one global explicit time
step is set by the smallest cell, so refining the scatterer slows the whole solve - and it buys
nothing, because in FEM the crack faces are exact geometry at *any* cell size. Conformity comes
from the boundary, not from refinement. That is the whole geometric advantage over a grid.

And one exception worth knowing: **past a critical angle, size against the evanescent decay
length rather than the wavelength.** An evanescent boundary layer can be thinner than a single
cell, and no wavelength-based rule sees it.

**Axial resolution is set by BANDWIDTH, not centre frequency.** Roughly, the shortest feature
you can separate along the beam is

$$\Delta z \;\approx\; \frac{c}{2 B}$$

with $B$ the usable bandwidth - the centre frequency does not appear. So a mesh that quietly
low-passes your own pulse costs you depth resolution directly: a notch images too deep, the crack
response weakens, and the back-wall arc smears. Every symptom has one cause, which is why this is
the single most useful diagnostic pattern in the project.

A related trap when diagnosing it: if the high band is missing, decide whether it was **lost or
merely delayed** before blaming anything. Energy genuinely destroyed by numerical dispersion
scales with element order and disappears from a whole-record measure; energy merely delayed is
still there if you look in a wider window. Those two call for completely different fixes.

---

## 6. Absorbing boundaries, and the three ways to do them

The simulated domain is finite but the real pipe is not. Without treatment, waves reflect off the
artificial edges and come back in as fake echoes. There are three standard approaches, and they
sit on a clear cost/quality ladder - worth knowing all three, because choosing between them is a
recurring decision.

**1. Dashpot (a first-order absorbing boundary condition).** Apply a traction proportional to
velocity on the boundary, so the boundary behaves like a shock absorber:

$$\text{traction} = -\rho \left[\, c_P (\dot{\mathbf{u}} \cdot \mathbf{n})\,\mathbf{n}
\;+\; c_S \left(\dot{\mathbf{u}} - (\dot{\mathbf{u}} \cdot \mathbf{n})\,\mathbf{n}\right) \right]$$

Nearly free - it adds one boundary term, and if the boundary is axis-aligned the coefficients stay
a diagonal vector, so the explicit scheme is unchanged. It is **exact for a plane wave arriving
head-on** and leaks progressively as incidence becomes oblique. Note it needs the *right speed per
component*: the normal part of the motion carries the compression wave, the tangential part the
shear wave, and using one speed for both over-damps whichever it got wrong.

**2. Sponge (graded damping).** Add a damping term that ramps smoothly from zero to strong across
a band of cells. The point of the ramp is that there is **no impedance step anywhere** for a wave
to reflect off - a sudden onset of damping is itself a reflector, which defeats the purpose. Also
nearly free, and it stacks with a dashpot. The counter-intuitive part: **more damping is not
better.** Past an optimum, the ramp becomes stiff enough to reflect on its own, so a "stronger"
sponge performs worse. It has an optimum, not a maximum.

**3. PML (perfectly matched layer).** The textbook answer, and what k-Wave uses: a surrounding
region whose equations are analytically continued into complex coordinates so waves enter and
decay with essentially no reflection at any angle. Far better than the other two, and far more
work - it needs extra field variables, its own time integration, and careful tuning.

**The lesson about choosing between them is the useful part.** Absorbing boundaries are a classic
place to spend effort on the wrong thing, because a bad boundary produces *plausible* artifacts -
late-arriving energy that looks like structure. So: establish that the boundary really is your
problem before paying for a PML. A better absorber cannot fix something the absorber was never
causing, and the cheap options above are enough to answer that question.

It does matter when it is the problem: a reflecting transducer plane once bounced the front-wall
echo straight back and produced a **15% back-wall timing error**, which a dashpot took to 0.19%.

Note the crack faces and outer wall are **not** absorbing - they are traction-free, which is the
correct physical condition for steel against air. Getting that backwards would be a physics error,
not a boundary-treatment choice.

---

## 7. How the comparison is made fair

This is the strongest design decision in the project.

Each solver outputs **the same object**: `channel_data`, a table of 22,801 time samples x 256
elements - literally the pressure-vs-time each array element would record. Same sample rate
(380 MHz), same duration (60 us), same time origin.

Getting that contract right is precision-critical. In particular **t = 0 is the instant the
first-firing element fires**: the steering delay law is *min-shifted* to zero, not centred. Their
delay span is 19.146 us at 20 degrees, so getting time zero wrong is a fixed bias of up to 19 us,
which would silently destroy the comparison.

Then an image is formed by **Kirchhoff migration**, also called delay-and-sum:

> For each pixel in the steel, work out how long a wave should take: transmitter, bounce off the
> outer wall, that pixel, back to each receiver. Look up that arrival time in each of the 128
> receiver traces and add the values. If something really scatters at that pixel, all 128 traces
> agree and the sum is large. If nothing is there, they disagree and cancel to near zero.

Written out, the image at pixel $\mathbf{x}$ is a coherent sum over receivers:

$$I(\mathbf{x}) \;=\; \left| \sum_{r} A_r\!\left( t_{\text{tx}}(\mathbf{x}) + t_{\text{rx}}(\mathbf{x}, r) \right) \right|$$

The sum is done on the **analytic signal** $A_r$ (via a Hilbert transform), so it is complex and
phases add coherently - that is what makes the cancellation work. Transmit travel times come from
Snell ray shooting; receive travel times from a ray search.

### The imaging chain has its own error sources

Two are worth understanding, because they are easy to mistake for solver problems.

**Sampling and aliasing.** Any resampling step folds content above the new Nyquist frequency back
into the band, at $f_{\text{alias}} = |f - k f_s|$. Once folded it is indistinguishable from real
signal and no later filter can remove it - so an anti-alias filter has to come *before* the
decimation, not after. A band-limited solver and a broadband one are affected differently by the
same naive resampling, which makes this a genuine source of asymmetry between two models.

**Operator aliasing.** Delay-and-sum sums along a curved surface in the data. Where that surface
is steep relative to the sample spacing, adjacent contributions stop being coherent and the sum
picks up streaks instead of cancelling. The standard fix is a filter applied to the migration
operator itself, and it costs a little resolution - which is why it has an aggressiveness setting
rather than being simply on.

### The rule that protects the comparison

**Any change to the shared processing must be applied to both datasets.** This sounds obvious and
is the easiest thing in the project to get wrong, because processing changes are cheap to try and
each one is individually defensible. The moment a processing choice differs between the two sides,
the experiment silently stops being a comparison of solvers and becomes a comparison of
processing - and it will still produce a confident number.

**The crucial move: we run THEIR beamformer, not a reimplementation.** Their imaging code is a
Python package in the research team's repository, called directly and unmodified on both datasets,
with the same options their own simulation script passes.

```
their k-Wave solver --+
                      +--> THEIR beamformer --> 370x358 image --> metrics
our FEM solver     ---+        (identical)
```

Because the imaging half is identical for both, **any difference in the final image is
attributable to the forward solver alone.** That converts "here are two pictures" into a
controlled experiment, and it pre-empts the obvious objection that we simply wrote a better image
processor.

---

## 8. What "more accurate" means

We chose the crack, so ground truth is known exactly: 4.0 mm deep, 1.0 mm wide, at x = 38.25 mm.
Accuracy is measurable, not a matter of taste. Three families of metric:

1. **Sizing** - imaged notch extent against the true 4.0 mm.
2. **Position** - imaged crack x against the true 38.25 mm.
3. **Contrast** - crack peak divided by the wall clutter, in dB.

On contrast: **a defect-free steel wall ought to image as pure black.** Any brightness in a
healthy wall is numerical junk. So the ratio asks "how far does the real crack stand above the
fake stuff?", and higher is better.

**Absolute amplitudes are NOT comparable between the two solvers** - k-Wave drives a 2e-6
velocity source, we apply unit traction. Only **within-image ratios** mean anything. Processing
changes can rescale amplitudes too, so raw pixel values are not comparable across imaging
configurations either.

### Normalisation is a measurement choice, and it can hide the thing you are measuring

The usual fix is to normalise each panel to **its own maximum**. That is right for asking "is the
crack visible in this image?", and it is exactly **wrong** for asking "which image has less
clutter?" - because it silently rescales the difference away. Two panels with very different
clutter render identically.

To compare clutter across images you need a **shared scale referenced to something physical in
each image.** Referencing each panel to *its own crack peak* works: it is the same quantity every
contrast metric already uses, so the picture and the numbers finally measure the same thing.

The related trap: **do not ask anyone to judge a small change by eye.** A change of well under a
decibel is real and measurable and simply invisible in a heat map. If a claim is at that scale, it
belongs in a profile plot or a table, and a figure that appears to show it is overselling.

### Two analysis traps that produced confident wrong numbers here

- **A window-argmax is not an arrival pick.** On a decaying ringdown tail, taking the maximum
  inside a window always returns *something*, so it invents echoes that do not exist. Require
  peak **prominence**, and report unresolved echoes as unresolved.
- **A claim that flips with the threshold is not a result.** Any metric must be swept over its
  arbitrary choices - threshold, region-of-interest width, guard distance - and the ordering must
  be unanimous or near it before it is quoted.

---

## 9. How the experiments are designed, and why

The physics above is only half the method. These are the experimental patterns this project runs
on - worth understanding separately, because they are what turn "two pictures" into evidence, and
they transfer to any comparison of two models.

**Change one thing, inside your own model.** The competitor's code is not yours to modify, but
yours is. So if you claim your geometry is why you win, test it by making *your own* geometry
worse - staircase your own wall onto their grid, fill your own crack with their filler material -
and measure what it costs. That converts an argument you cannot settle into a measurement you can.
It is also the only honest way to attack your own favourite explanation.

**Write the prediction and the falsifier down first.** Before running the test, record what the
result must be for your explanation to survive, and what result would kill it. Without that, any
outcome can be narrated as support after the fact - and it will be, because the narration is
easier than the alternative. A prediction that landed inside a range recorded in advance is worth
far more than the same number produced and then explained.

**Replicate on a mirrored configuration.** Here that means flipping the steering angle. The
specimen is symmetric and the notch is on-axis, so anything that changes materially between $+20$
and $-20$ degrees is telling you about the *model*, not the specimen. It is the cheapest
independent test available, and a result that fails to replicate across it was never a result.

**Run a null test whenever a change carries a passenger.** If enabling the thing you care about
also switches something else - a different compute engine, a different code path, a different
library - first run the new path with your feature *off*. It must reproduce the old answer. If it
does not, the two effects are confounded and every number that follows is uninterpretable.

**Gate on reproduction before trusting a new arm.** Re-run the *unchanged* configuration and
require it to reproduce the stored result exactly. If the baseline has moved, the new arm's number
means nothing, and you would rather learn that from a check than from a conclusion.

**Sweep the arbitrary choices.** Every image metric has knobs no physics fixes: a brightness
threshold, a region-of-interest width, a guard distance. Any of them can be chosen to favour any
conclusion. So sweep them all and require the ordering to be unanimous, or nearly so, before
quoting it. **A claim that flips with the threshold is not a claim.** And pin the definition in
code - a metric that lives only in a report will drift.

**Simulate the defect-free case.** Run the identical specimen with no defect at all. Then every
feature that appears is numerical *by construction*, which is the only clean way to measure an
artifact floor. It also converts "is that bright patch the crack?" from an opinion into a
subtraction.

**Keep any processing change symmetric.** See section 7 - it is important enough to appear twice.

**Treat a negative result as a result.** Eliminating a candidate cause is progress: it is what
stops you spending weeks building an expensive fix for something that was never the problem. Most
of the candidate explanations tested in this project were eliminated, and the eliminations are
more valuable than the survivors, because each one narrows where the real answer can be.

---

## 10. Glossary

| term | meaning |
|---|---|
| **P wave** | compression wave, motion along travel direction, 5700 m/s in steel |
| **S wave** | shear wave, motion across travel direction, 3100 m/s in steel; does not exist in water |
| **Mode conversion** | a P wave in water becoming an S wave in steel at an angled interface |
| **Critical angle** | incidence past which a given transmitted mode cannot propagate (15.3 deg for P into steel) |
| **TT-T** | the imaging path: transmit half-skip off the outer wall, receive direct, both legs shear |
| **Channel data** | pressure-vs-time recorded per array element; the interface between solver and imaging |
| **Beamforming** | turning channel data into an image; here Kirchhoff migration / delay-and-sum |
| **Clutter** | image brightness that is not a real feature; numerical noise |
| **CNR** | contrast-to-noise ratio, crack peak over clutter, in dB |
| **DOF** | degrees of freedom - the number of unknowns the solver carries (1.47 M here) |
| **CFL** | stability limit tying timestep to cell size over wave speed |
| **Dispersion** | discretisation error making wave speed frequency-dependent; shifts arrival times |
| **ppw / nodes per wavelength** | resolution measure; must be met at the pulse's top frequency, not its centre |
| **PML** | perfectly matched layer, a high-quality absorbing boundary (k-Wave uses one) |
| **Dashpot ABC** | first-order absorbing boundary, $\text{traction} \approx -\rho c \dot{\mathbf{u}}$ (we use this) |
| **Traction-free** | boundary with nothing pushing back; the correct condition for steel against air |
| **Weak form** | the integrated-by-parts equation FEM actually solves, needing only first derivatives |
| **Mass lumping** | making the mass matrix diagonal so explicit stepping needs no linear solve |
| **GLL** | Gauss-Lobatto-Legendre node placement; makes lumped mass exactly diagonal and positive |
| **Spectral element** | high-order FEM on GLL nodes; low dispersion at moderate cost |
| **Pseudospectral (PSTD)** | k-Wave's method: uniform grid, derivatives via FFT |
| **Impedance** | $Z = \rho c$; the mismatch between two materials sets how much reflects |
| **Axial resolution** | smallest separable feature along the beam, $\approx c/2B$; set by bandwidth, not centre frequency |
| **Grating lobe** | full-amplitude copy of the steered beam at a second angle when pitch is too coarse; not removable by apodisation |
| **Apodisation** | tapering element amplitudes across the aperture; suppresses ordinary sidelobes only |
| **Analytic signal** | complex signal from a Hilbert transform; lets delay-and-sum add phases coherently |
| **Aliasing** | content above the sampling Nyquist folded irreversibly into the band; filter *before* resampling, never after |
| **Operator aliasing** | streak clutter from summing along a steep migration surface; fixed by filtering the operator itself |
| **Sponge layer** | graded damping band used as a cheap absorbing boundary; has an optimum strength, not a maximum |
| **Pre-registration** | recording the prediction and the falsifier before running the test |
| **Null test** | running a new code path with the new feature disabled, to prove it changes nothing by itself |
