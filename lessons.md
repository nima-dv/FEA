# Lessons: the theory behind this project

Plain-language explanation of what is being simulated, the equations behind it, how the two
approaches differ, and how the comparison is made fair. Written to be understood and presented,
not to be rigorous.

Scope of this file: **concepts and equations that stay true regardless of results.** Measured
numbers, current status and open questions live in the branch README
(`F:\code\readme\rnd-nima-FEA-README.md`); slide-by-slide talking points live in `PITCH.md`.
Keep it that way - no results, no history, no changelog here.

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
which is what keeps a high-order method cheap. We take 163,680 such steps, about 2.4 hours.

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

### The bandwidth trap - the project's biggest single lesson

The transmitted pulse is only **one cycle** long. A short pulse is wide in frequency: a 1-cycle
4 MHz burst carries real energy out to 6-8 MHz - roughly 100% bandwidth.

The standard rule of thumb "4 nodes per wavelength" must therefore be satisfied at the pulse's
**upper usable frequency, not its centre frequency.** A mesh sized for 4 MHz gives only ~2.7
nodes per wavelength at 7 MHz, over a water path 53 wavelengths long. The mesh then acts as a
low-pass filter on your own pulse while it is in transit.

**Axial resolution is set by BANDWIDTH, not centre frequency.** So losing the top of the band
smears everything: a true 4.0 mm notch imaged as 8.07 mm, a weak crack response, and no distinct
back-wall arc. Fixing it - degree 4 on a finer mesh - is what turned a loss into a win.

Two wrong explanations were eliminated by measurement rather than argument: the **source** was
never the limit (our emitted pulse is broader than k-Wave's *returned* echo), and the energy is
**genuinely lost, not merely delayed** (a whole-record measure shows the high band missing, and
the loss scales with element order - the signature of high-wavenumber numerical dispersion).

---

## 6. Absorbing boundaries

The simulated domain is finite but the real pipe is not. Without treatment, waves reflect off the
artificial edges and come back in as fake echoes.

- **k-Wave uses a PML** (perfectly matched layer): a surrounding region whose equations are
  modified so waves enter and decay without reflecting. Very effective.
- **We use a dashpot absorbing boundary condition**: a traction proportional to velocity,
  $\text{traction} \approx -\rho c \dot{\mathbf{u}}$, applied on the array plane and side walls. It is **first-order** -
  it absorbs normal incidence well and leaks somewhat at oblique incidence.

This matters more than it sounds. A reflecting transducer plane once bounced the front-wall echo
back and produced a **15% back-wall timing error**; the dashpot took that to 0.19%. Upgrading to
a PML is the textbook answer when a boundary really is the problem. Establish that it is before
paying for one: a better absorber cannot fix something the absorber was never causing.

Note the crack faces and outer wall are **not** absorbing - they are traction-free, which is the
correct physical condition for steel against air.

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

The sum is done on the **analytic signal** (via a Hilbert transform), so it is complex and phases
add coherently - that is what makes the cancellation work. Transmit travel times come from Snell
ray shooting; receive travel times from a ray search.

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
velocity source, we apply unit traction. Only **within-image ratios** mean anything. This is also
why comparison panels are normalised to their own maximum, never to a shared one.

### Two analysis traps that produced confident wrong numbers here

- **A window-argmax is not an arrival pick.** On a decaying ringdown tail, taking the maximum
  inside a window always returns *something*, so it invents echoes that do not exist. Require
  peak **prominence**, and report unresolved echoes as unresolved.
- **A claim that flips with the threshold is not a result.** Any metric must be swept over its
  arbitrary choices - threshold, region-of-interest width, guard distance - and the ordering must
  be unanimous or near it before it is quoted.

---

## 9. Glossary

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
