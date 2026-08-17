r"""
B1 - ANGLE-RESOLVED FLUID/SOLID REFLECTION, TRANSMISSION AND MODE CONVERSION
============================================================================
Validates the monolithic single-displacement-field FEM (the one used everywhere else in
this project) against the EXACT plane-wave solution for a water|steel half-space at
oblique incidence - including the P->S MODE CONVERSION, which is the whole point.

Water: c_f = 1500 m/s, rho_f = 1000, mu = 0.       Steel: c_P = 5700, c_S = 3100, rho = 7850.
Critical angles:  theta_P* = asin(1500/5700) = 15.26 deg,  theta_S* = asin(1500/3100) = 28.94 deg.
Beyond theta_S* BOTH transmitted waves are evanescent -> |R| = 1 exactly (total reflection).


-------------------------------------------------------------------------------------
1. THE EXACT FLUID-SOLID COEFFICIENTS (derived here, not copied)
-------------------------------------------------------------------------------------
Interface at Y = 0.  Fluid occupies Y < 0, solid Y > 0.  x is along the interface.
Everything varies as exp(i(k_x x - omega t)); k_x is conserved across the interface
(that IS Snell's law).  Amplitude A = 1 for the incident potential.

    k_x  = omega sin(theta) / c_f          (horizontal wavenumber, common to all 4 waves)
    k_fy = omega cos(theta) / c_f          (fluid vertical wavenumber)
    p    = sqrt(omega^2/c_P^2 - k_x^2)     (solid P vertical wavenumber)
    s    = sqrt(omega^2/c_S^2 - k_x^2)     (solid S vertical wavenumber)

Branch choice: Im(p), Im(s) >= 0 so that exp(i p Y) DECAYS as Y -> +infinity.
numpy's principal sqrt of a negative real gives +i|.|, which is exactly that branch.
=> past a critical angle the corresponding wave is evanescent, not propagating.

FLUID (Y<0): displacement potential phi, u = grad(phi).
    phi = exp(i(k_x x + k_fy Y)) + R exp(i(k_x x - k_fy Y))
    u_y = d phi/dY                        -> at Y=0:  i k_fy (1 - R)
    sigma_yy = lam_f div(u) = lam_f lap(phi) = -lam_f (omega^2/c_f^2) phi = -rho_f omega^2 phi
                                          -> at Y=0:  -rho_f omega^2 (1 + R)
    sigma_xy = 0                          (mu = 0: a fluid carries no shear)

SOLID (Y>0): Helmholtz decomposition u = grad(phi_s) + curl(psi e_z), i.e.
    u_x = d phi_s/dx + d psi/dY ,   u_y = d phi_s/dY - d psi/dx
    phi_s = T_P exp(i(k_x x + p Y)) ,   psi = T_S exp(i(k_x x + s Y))
  At Y = 0 (writing only the Y=0 values, common factor exp(i k_x x) dropped):
    u_x = i k_x T_P + i s   T_S
    u_y = i p   T_P - i k_x T_S
  Stresses.  Using lap(phi_s) = -(omega^2/c_P^2) phi_s and rho omega^2 = mu (k_x^2+s^2):
    sigma_yy = lam lap(phi_s) + 2 mu du_y/dY
             = -[lam(k_x^2+p^2) + 2 mu p^2] phi_s + 2 mu k_x s psi
             = -mu (s^2 - k_x^2) T_P + 2 mu k_x s T_S           <- the neat symmetric form
    sigma_xy = mu (du_x/dY + du_y/dx)
             = mu [ -2 k_x p T_P + (k_x^2 - s^2) T_S ]

THREE INTERFACE CONDITIONS -> 3x3 system for (R, T_P, T_S):
  (1) normal displacement continuous   u_y|fluid = u_y|solid
        k_fy R + p T_P - k_x T_S = k_fy
  (2) normal traction continuous       sigma_yy|fluid = sigma_yy|solid
        -rho_f omega^2 R + mu(s^2-k_x^2) T_P - 2 mu k_x s T_S = rho_f omega^2
  (3) shear traction vanishes          sigma_xy|solid = 0
        -2 k_x p T_P + (k_x^2 - s^2) T_S = 0

SANITY (done analytically): at theta = 0, k_x = 0 so (3) gives T_S = 0 (no mode conversion
at normal incidence), and (1)+(2) collapse to R = (Z2 - Z1)/(Z1 + Z2) with Z = rho c_P,
i.e. |R| = 0.9351 - the number already independently validated in toys/fluid_solid.py.

DISPLACEMENT-amplitude coefficients (what a transducer measures) for PROPAGATING modes:
    |u_inc| = (omega/c_f)|A| ,  |u_P| = (omega/c_P)|T_P| ,  |u_S| = (omega/c_S)|T_S|
    => D_P = T_P c_f/c_P ,   D_S = T_S c_f/c_S

WHAT WE ACTUALLY MEASURE, and why div/curl is the right observable
    div(u) sees ONLY P (S is divergence-free), curl(u) sees ONLY S (P is curl-free), and
    because each potential satisfies its own Helmholtz equation these relations are EXACT
    (also for evanescent waves - no plane-wave-amplitude assumption needed):
        div(u_inc)  = -(omega^2/c_f^2) phi        (fluid)
        div(u_P)    = -(omega^2/c_P^2) phi_s      (solid)
        curl(u_S)   = -lap(psi) = +(omega^2/c_S^2) psi   <- NOTE THE OPPOSITE SIGN
    So a complex spectral ratio of div/curl at a solid receiver against div at a fluid
    receiver, with the known geometric phase removed, returns T_P and T_S directly.
    This separates P from S EXACTLY even when their arrivals overlap in time, which is
    what makes a small (fast) model possible.

ENERGY: time-averaged normal Poynting flux -1/2 Re(sigma_yj conj(u_dot_j)).
On the fluid side this collapses to (1/2) rho_f omega^3 k_fy (1 - |R|^2), so the
transmitted fraction must be 1 - |R|^2.  The script computes the solid-side flux from the
FULL field (P+S together, cross terms included) and checks it equals 1 - |R|^2; it also
prints the sum of the individual P and S fluxes so the reader can see the cross terms are
zero.  That is an end-to-end check on the algebra above, independent of the FEM.


-------------------------------------------------------------------------------------
2. THE NUMERICAL EXPERIMENT
-------------------------------------------------------------------------------------
Same machinery as toys/fluid_solid.py: ONE displacement field u over the whole box,
region-wise DG0 material with mu = 0 in the water, GLL spectral elements, row-sum lumped
mass, explicit leapfrog.  Displacement continuity is automatic (shared nodes) and traction
continuity is the natural BC, so this single field IS an acoustic-elastic coupling.

  * horizontal interface, water below, steel above
  * source: an obliquely-propagating, Gaussian-windowed tone burst launched as an initial
    condition u = grad(Phi), v = -c_f dPhi/dxi.  Being an exact gradient it is EXACTLY
    curl-free, so ANY shear seen in the steel is provably mode-converted, not injected.
    (A beam simply tapered as n*g(xi)*T(eta) is NOT curl-free - curl = -g T' - which would
    inject shear at the beam edges.  Hence the potential formulation.)
  * Lysmer-Kuhlemeyer dashpot on all four outer edges, traction = -rho(c_P u_n + c_S u_t),
    lumped the same way as the mass so the scheme stays explicit and diagonal.
  * receivers: div(u) in the fluid and div(u)/curl(u) in the steel, obtained by interpolating
    the UFL expression into DG0 restricted to the receiver cells (DG0's single interpolation
    point is the cell centroid -> exact point evaluation, no smoothing).
    EVERY receiver group sits on its OWN beam axis, which matters: for a finite-width beam
    the reflected axis leaves the interface at -theta, so at depth d it is at
    x_H + d tan(theta) while the incident axis is at x_H - d tan(theta).  Measuring the
    reflected pulse on the incident axis puts it 2 d sin(theta) off-beam-centre and reads
    |R| far too low (16% low at 20 deg with b = 4 mm - measured, then fixed).
    Each group is a 3- or 5-point lateral fan so the beam peak is captured, and each steel
    group spans three depths so the arrival-time gradient can be fitted.
  * transmitted ANGLE and SPEED come from a least-squares plane fit of envelope-peak
    arrival time over the receiver grid: t = t0 + p_x dx + p_y dy, then
    tan(theta_t) = p_x/p_y and c_t = 1/|p|; that estimate is then refined with the spectral
    PHASE gradient, using the envelope fit only to resolve the cycle (see phase_refine).
    Note this does NOT assume Snell (which fixes p_x trivially) - p_y is measured
    independently, so the direction is a real result.
  * the domain is re-sized per angle (a tilted beam needs a thicker fluid layer) so small
    angles stay cheap.

-------------------------------------------------------------------------------------
3. THREE THINGS THAT HAD TO BE GOT RIGHT (each was measured wrong first, then fixed)
-------------------------------------------------------------------------------------
(a) THE REFLECTED BEAM IS NOT ON THE INCIDENT AXIS.  It leaves the interface at -theta, so
    at depth d it sits at x_H + d tan(theta) while the incident axis is at x_H - d tan(theta).
    Reading the reflected pulse on the incident axis puts it 2 d sin(theta) off beam centre
    and gave |R| = 0.778 instead of 0.911 at 20 deg.  Each fan is now on its own axis.

(b) FREE-FIELD REFERENCE RUN kills diffraction bias.  The incident and reflected pulses are
    measured at points whose distance from the source differs by 2 d/cos(theta), and a beam
    of finite width spreads over that distance - biasing |R| low by ~10% at 60 deg.
    Fix: repeat the identical mesh and source with WATER EVERYWHERE and read the free field
    at the MIRROR point M = (x2, y_i + |Y2|) of the reflected receiver F2 = (x2, y_i - |Y2|).
    A flat mirror preserves a beam, so the reflected field at F2 and the free field at M have
    identical phase and identical diffraction history: their ratio is EXACTLY R, with no
    geometric phase term and with spreading, numerical dispersion and pulse distortion all
    cancelling.  The reference run must use the SAME dt as the main one, or the difference in
    temporal dispersion, ~(w dt)^2/24, leaves tens of degrees of spurious phase in arg(R).

(c) A BEAM IS NOT A PLANE WAVE.  A beam of 1/e half-width b carries an angular spectrum of
    1/e half-width lambda/(pi b) (3.4 deg here), so it necessarily measures the coefficient
    AVERAGED over that spread.  Harmless where the coefficients are smooth; decisive next to
    theta_P* = 15.26 deg and next to the leaky-Rayleigh angle asin(c_f/c_R) = 31.5 deg, where
    |R| = 1 but arg(R) swings through a pole and a finite beam reflects non-specularly (Schoch
    displacement).  The script therefore reports the angular-spectrum-weighted exact value
    alongside the plane-wave one, so a residual can be attributed to the beam or to the FEM
    rather than argued about.  It does NOT tune anything.

-------------------------------------------------------------------------------------
4. KNOWN LIMITS (measured, not guessed - see the tables the script prints)
-------------------------------------------------------------------------------------
* AMPLITUDES are the strong result: |R|, |D_P| and |D_S| land within ~1% of exact wherever
  the coefficients are smooth, and |R| holds to a few tenths of a percent even at 60 deg.
* THE PHASE OF R PAST THE S CRITICAL ANGLE IS THE WEAK NUMBER.  Beyond theta_S* the whole of
  arg(R) is generated by an evanescent boundary layer in the steel of thickness
  1/sqrt(kx^2-(w/c_S)^2) - 0.95 mm at 30 deg but only 0.17 mm at 60 deg, i.e. LESS THAN ONE
  ELEMENT at h = 0.2 mm.  Measured h-refinement of the arg(R) error (--angles 50,60 --h ...):

        h [mm]     50 deg     60 deg      |R| err at 60 deg
        0.28       18.1 deg   30.7 deg    1.84 %
        0.20        9.1 deg   21.6 deg    0.29 %
        0.14        5.4 deg   15.3 deg    0.29 %

  i.e. clean FIRST order in h at 60 deg (error ratios 1.42, 1.41 for h ratios 1.40, 1.43) and
  nearer second order at 50 deg, while |R| is already converged.  First order means a few
  degrees of phase is not affordable by refinement alone.  PRACTICAL RULE for the rest of the
  project: if a beyond-critical-angle interaction has to be phase-accurate, size h against the
  evanescent decay length, not against the wavelength.  Amplitudes do not care.
* NEAR A CRITICAL ANGLE OR THE RAYLEIGH ANGLE the finite beam, not the FEM, sets the accuracy;
  the printed beam-averaged exact value is the right thing to compare against there.  It is an
  order-of-effect estimate, not a target: at 25 deg it overshoots (0.155 vs a measured 0.135
  and a plane-wave 0.134) because it ignores the relative phases of the angular components.
* The transmitted P degrades as theta_P* is approached and cannot be measured at it.  Snell
  MAGNIFIES the beam's angular spread on refraction by (c_P/c_f) cos(theta)/cos(theta_P), which
  is 7.1x at 13 deg: a +-3.4 deg incident beam becomes a +-24 deg fan in the steel, so there is
  no longer a beam whose peak amplitude means anything (hence |D_P| 11% low at 13 deg while the
  beam-averaged coefficient itself has barely moved).  At 15 deg the P refracts to 79.6 deg and
  runs along the interface as a head wave; the script flags that 'graz' rather than quoting it.

RESULTS (2026-08-11, h = 0.20 mm, Q3, f0 = 2 MHz, 2.5 cycles, 12 angles, 5.8 min wall)
  |R|      0.01-0.29% everywhere except 15 deg (2.97%), 30 deg (23.7%) and 35 deg (11.6%),
           all three within 4 deg of theta_P* or the Rayleigh angle and all three matched by
           the beam-averaged exact value (e.g. 30 deg: FEM 0.763, beam 0.688, plane 1.000).
           Normal incidence: 0.9351 vs 0.9351 exact.
  |D_P|    0.65% / 0.72% / 1.30% at 0 / 5 / 10 deg; 10.9% at 13 deg (angular magnification).
  |D_S|    0.22% / 1.72% / 2.95% / 0.81% / 0.43% at 5 / 10 / 13 / 20 / 25 deg.
           66% at 15 deg - and the beam-averaged exact value is 0.0748 against a measured
           0.0758, i.e. the FEM is right and the plane-wave number (0.0456) is the wrong
           yardstick 0.4 deg from a critical angle.
  arg(R)   <= 2.0 deg out to 25 deg; degrades past theta_S* exactly as the evanescent layer
           thins (n/evS 14.2 at 30 deg -> 2.5 at 60 deg; error 2.7 deg -> 21.6 deg).
  angles   transmitted S direction within 0.55 deg of Snell at 20 deg (45.53 vs 44.98) and
           c_S 3117 vs 3100 m/s, from a slowness fit that never assumes Snell.
  energy   <= 1.5% residual away from the critical/Rayleigh region, worst 0.59% excluding it.
  the free-field reference is worth a lot: at 60 deg |R| is 1.0029 with it, 0.9095 without.

RUN
  ./run.ps1 python3 validation/zoeppritz.py                 # full sweep
  ./run.ps1 python3 validation/zoeppritz.py --check         # analytic self-tests only
  ./run.ps1 python3 validation/zoeppritz.py --angles 0,20 --h 0.3
Outputs: results/zoeppritz/{amplitude,phase,error,angles}.png + table.txt
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.signal import hilbert

import basix
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem import (Function, assemble_matrix, assemble_vector, form,
                         functionspace)
from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points

# ============================================================================================
# materials
# ============================================================================================
C_F, RHO_F = 1500.0, 1000.0                 # water
C_P, C_S, RHO_S = 5700.0, 3100.0, 7850.0    # steel
MU_S = RHO_S * C_S**2
LAM_S = RHO_S * (C_P**2 - 2.0 * C_S**2)
LAM_F = RHO_F * C_F**2                      # water bulk modulus (mu = 0)
TH_P_CRIT = np.degrees(np.arcsin(C_F / C_P))   # 15.258 deg
TH_S_CRIT = np.degrees(np.arcsin(C_F / C_S))   # 28.938 deg
R_NORMAL = abs(RHO_F * C_F - RHO_S * C_P) / (RHO_F * C_F + RHO_S * C_P)   # 0.9351

# Free-surface Rayleigh speed of the steel, from the Rayleigh function
#   (2 - v^2/cS^2)^2 = 4 sqrt(1-v^2/cP^2) sqrt(1-v^2/cS^2).
# Past theta_S* the plane-wave |R| is exactly 1, but the PHASE of R swings rapidly through the
# angle asin(c_f/c_R): that is the leaky-Rayleigh pole, and a finite beam there does NOT reflect
# specularly (Schoch displacement) - the reason the 30 deg point in the sweep is an outlier.
C_RAYLEIGH = brentq(lambda v: (2 - v**2 / C_S**2) ** 2
                    - 4 * np.sqrt(1 - v**2 / C_P**2) * np.sqrt(1 - v**2 / C_S**2),
                    1.0, C_S * (1 - 1e-12))                  # 2870.1 m/s
TH_RAYLEIGH = np.degrees(np.arcsin(C_F / C_RAYLEIGH))        # 31.509 deg

OUT = Path(__file__).resolve().parents[1] / "results" / "zoeppritz"


# ============================================================================================
# 1. EXACT SOLUTION
# ============================================================================================
def analytic(theta_deg, omega=1.0):
    """Exact fluid->solid coefficients. omega only scales the wavenumbers (the coefficients
    themselves are frequency independent - a flat interface has no length scale)."""
    th = np.radians(theta_deg)
    kx = omega * np.sin(th) / C_F
    kfy = omega * np.cos(th) / C_F
    # principal sqrt -> Im >= 0 -> exp(i p Y) decays into the solid (evanescent past critical)
    p = np.sqrt(complex((omega / C_P) ** 2 - kx**2))
    s = np.sqrt(complex((omega / C_S) ** 2 - kx**2))

    M = np.array([
        [kfy,               p,                     -kx                 ],
        [-RHO_F * omega**2, MU_S * (s**2 - kx**2),  -2 * MU_S * kx * s  ],
        [0.0,               -2 * kx * p,            kx**2 - s**2        ],
    ], dtype=complex)
    rhs = np.array([kfy, RHO_F * omega**2, 0.0], dtype=complex)
    R, TP, TS = np.linalg.solve(M, rhs)

    # displacement coefficients (meaningful for propagating modes)
    DP = TP * C_F / C_P
    DS = TS * C_F / C_S
    # transmitted angles from Snell (nan when evanescent)
    sP, sS = C_P * np.sin(th) / C_F, C_S * np.sin(th) / C_F
    thP = np.degrees(np.arcsin(sP)) if sP <= 1.0 else np.nan
    thS = np.degrees(np.arcsin(sS)) if sS <= 1.0 else np.nan
    return dict(theta=theta_deg, R=R, TP=TP, TS=TS, DP=DP, DS=DS,
                thP=thP, thS=thS, kx=kx, kfy=kfy, p=p, s=s,
                P_prop=abs(p.imag) < 1e-12, S_prop=abs(s.imag) < 1e-12)


def _solid_flux(TP, TS, kx, p, s, omega):
    """Time-averaged normal energy flux carried into the solid by the given (P,S) pair.
    flux = -1/2 Re(sigma_yy conj(u_dot_y) + sigma_xy conj(u_dot_x)),  u_dot = -i omega u
         = (omega/2) Im(sigma_yy conj(u_y) + sigma_xy conj(u_x))."""
    ux = 1j * kx * TP + 1j * s * TS
    uy = 1j * p * TP - 1j * kx * TS
    syy = -MU_S * (s**2 - kx**2) * TP + 2 * MU_S * kx * s * TS
    sxy = MU_S * (-2 * kx * p * TP + (kx**2 - s**2) * TS)
    return 0.5 * omega * np.imag(syy * np.conj(uy) + sxy * np.conj(ux))


def analytic_energy(theta_deg, omega=1.0):
    """Returns (transmitted fraction from fluid side = 1-|R|^2, solid-side total flux
    fraction incl. cross terms, sum of individual P and S flux fractions)."""
    a = analytic(theta_deg, omega)
    inc = 0.5 * RHO_F * omega**3 * a["kfy"]        # incident flux for A = 1
    tot = _solid_flux(a["TP"], a["TS"], a["kx"], a["p"], a["s"], omega) / inc
    onlyP = _solid_flux(a["TP"], 0.0, a["kx"], a["p"], a["s"], omega) / inc
    onlyS = _solid_flux(0.0, a["TS"], a["kx"], a["p"], a["s"], omega) / inc
    return 1.0 - abs(a["R"]) ** 2, tot, onlyP, onlyS


def analytic_selfcheck():
    """Independent checks on the algebra above. Any failure => the formula is wrong."""
    print("ANALYTIC SELF-CHECKS")
    a0 = analytic(0.0)
    print(f"  normal incidence |R|        = {abs(a0['R']):.6f}   (impedance formula "
          f"{R_NORMAL:.6f})   {'OK' if abs(abs(a0['R'])-R_NORMAL) < 1e-10 else 'FAIL'}")
    print(f"  normal incidence |T_S|      = {abs(a0['TS']):.3e}  (must be 0)          "
          f"{'OK' if abs(a0['TS']) < 1e-14 else 'FAIL'}")
    print(f"  critical angles             P {TH_P_CRIT:.3f} deg, S {TH_S_CRIT:.3f} deg")
    # locate the fastest phase swing of R past theta_S* and compare with the Rayleigh angle
    tg = np.linspace(TH_S_CRIT + 0.05, 60.0, 4000)
    ph = np.unwrap(np.angle([analytic(t)["R"] for t in tg]))
    th_swing = tg[int(np.argmax(np.abs(np.gradient(ph, tg))))]
    print(f"  Rayleigh speed {C_RAYLEIGH:.1f} m/s -> leaky-Rayleigh angle {TH_RAYLEIGH:.3f} deg;"
          f" fastest arg(R) swing at {th_swing:.3f} deg   "
          f"{'OK (pole located)' if abs(th_swing-TH_RAYLEIGH) < 1.5 else 'CHECK'}")
    worst = 0.0
    print("   theta   1-|R|^2    solid flux   (P only)  (S only)  cross-term  |R|")
    for th in [0, 5, 10, 15, 15.5, 20, 25, 28, 30, 40, 60]:
        lhs, tot, oP, oS = analytic_energy(th)
        worst = max(worst, abs(lhs - tot))
        print(f"  {th:6.2f}  {lhs:9.6f}  {tot:11.6f}  {oP:9.6f} {oS:9.6f}  "
              f"{tot-oP-oS:+10.2e}  {abs(analytic(th)['R']):.6f}")
    print(f"  worst |(1-|R|^2) - solid flux| = {worst:.2e}   "
          f"{'OK (energy conserved, cross terms vanish)' if worst < 1e-12 else 'FAIL'}")
    # frequency independence
    r1, r2 = analytic(23.0, 1.0)["R"], analytic(23.0, 7.7e6)["R"]
    print(f"  frequency independence at 23 deg: |R(w=1)-R(w=7.7e6)| = {abs(r1-r2):.2e}")
    return worst < 1e-12 and abs(abs(a0["R"]) - R_NORMAL) < 1e-10 and abs(a0["TS"]) < 1e-14


# ============================================================================================
# 2. FEM PIECES
# ============================================================================================
def build_operators(domain, y_i, degree):
    """GLL-SEM vector space, region-wise material, stiffness K (scipy), lumped mass m,
    lumped dashpot c.  Water is y < y_i (mu = 0), steel is y > y_i."""
    el = basix.ufl.element("Lagrange", domain.basix_cell(), degree,
                           lagrange_variant=basix.LagrangeVariant.gll_warped, shape=(2,))
    V = functionspace(domain, el)
    Q = functionspace(domain, ("DG", 0))
    lam, mu, rho, cp, cs = (Function(Q) for _ in range(5))
    lam.interpolate(lambda X: np.where(X[1] < y_i, LAM_F, LAM_S))
    mu.interpolate(lambda X: np.where(X[1] < y_i, 0.0, MU_S))
    rho.interpolate(lambda X: np.where(X[1] < y_i, RHO_F, RHO_S))
    cp.interpolate(lambda X: np.where(X[1] < y_i, C_F, C_P))
    cs.interpolate(lambda X: np.where(X[1] < y_i, 0.0, C_S))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))
    sig = lambda w: lam * ufl.tr(eps(w)) * ufl.Identity(2) + 2.0 * mu * eps(w)
    K = assemble_matrix(form(ufl.inner(sig(u), eps(v)) * ufl.dx))
    K.scatter_reverse()
    K = K.to_scipy()
    m = assemble_vector(form(rho * ufl.inner(v, ufl.as_vector((1.0, 1.0))) * ufl.dx)).array.copy()
    assert np.all(m > 0)

    # Lysmer-Kuhlemeyer dashpot: traction = -rho(c_P (u'.n)n + c_S (u'.t)t).
    # Its diagonal in the (x,y) basis is rho(c_P n_i^2 + c_S t_i^2) with t = (-n_y, n_x);
    # on axis-aligned edges the off-diagonal is identically zero, so lumping is exact here.
    nrm = ufl.FacetNormal(domain)
    d = ufl.as_vector((rho * (cp * nrm[0] ** 2 + cs * nrm[1] ** 2),
                       rho * (cp * nrm[1] ** 2 + cs * nrm[0] ** 2)))
    c = assemble_vector(form(ufl.inner(v, d) * ufl.ds)).array.copy()
    return V, K, m, c


def stable_dt(K, m, safety=0.85):
    """dt_max = 2/sqrt(lambda_max(M^-1 K)) for central differences; lambda_max by power
    iteration (cheap: a few dozen matvecs). Beats guessing a CFL constant."""
    z = np.random.default_rng(0).standard_normal(m.size)
    lam = 0.0
    for _ in range(60):
        z = (K @ z) / m
        lam = np.linalg.norm(z)
        z /= lam
    return safety * 2.0 / np.sqrt(lam)


class Probe:
    """div(u) and curl(u) at a set of points, via DG0 interpolation restricted to the
    containing cells. DG0's single interpolation point is the cell centroid, so this is an
    EXACT point evaluation (verified against u=(x^2,y^3) -> div = 2x+3y^2)."""

    def __init__(self, domain, V, pts_xy):
        self.uf = Function(V)
        Q0 = functionspace(domain, ("DG", 0))
        ip = Q0.element.interpolation_points
        self.dE = fem.Expression(ufl.div(self.uf), ip)
        self.cE = fem.Expression(ufl.grad(self.uf)[1, 0] - ufl.grad(self.uf)[0, 1], ip)
        self.dF, self.cF = Function(Q0), Function(Q0)

        pts = np.zeros((len(pts_xy), 3))
        pts[:, :2] = pts_xy
        tree = bb_tree(domain, domain.topology.dim)
        cand = compute_colliding_cells(domain, compute_collisions_points(tree, pts), pts)
        cells = []
        for i in range(len(pts)):
            lk = cand.links(i)
            if len(lk) == 0:
                raise RuntimeError(f"receiver {pts_xy[i]} outside the mesh")
            cells.append(lk[0])
        self.cells = np.array(cells, dtype=np.int32)
        self.dofs = np.array([Q0.dofmap.cell_dofs(c)[0] for c in self.cells])
        # true receiver coordinates = cell centroids (what DG0 actually evaluates at);
        # used for the geometric phase corrections, so they must be the real ones.
        self.xy = mesh.compute_midpoints(domain, domain.topology.dim, self.cells)[:, :2]
        # callers must pre-snap onto centroids; if not, the request is silently displaced
        off = np.max(np.abs(self.xy - np.asarray(pts_xy)))
        if off > 1e-10:
            raise RuntimeError(f"receiver not on a cell centroid (off by {off:.3e} m) - snap it")

    def __call__(self, U):
        self.uf.x.array[:] = U
        self.dF.interpolate(self.dE, cells0=self.cells)
        self.cF.interpolate(self.cE, cells0=self.cells)
        return self.dF.x.array[self.dofs].copy(), self.cF.x.array[self.dofs].copy()


def march(K, m, c, u0, v0, dt, nsteps, probe):
    """Explicit central difference with lumped viscous damping:
       (m/dt^2 + c/2dt) u^{n+1} = 2m/dt^2 u^n - (m/dt^2 - c/2dt) u^{n-1} - K u^n"""
    A = m / dt**2 + c / (2 * dt)
    B = 2.0 * m / dt**2
    Cm = m / dt**2 - c / (2 * dt)
    u_old = u0.copy()
    a0 = (-(K @ u_old) - c * v0) / m
    u = u_old + dt * v0 + 0.5 * dt**2 * a0
    dv0, cv0 = probe(u_old)
    dv1, cv1 = probe(u)
    D, Cl = [dv0, dv1], [cv0, cv1]
    for _ in range(1, nsteps):
        u_new = (B * u - Cm * u_old - (K @ u)) / A
        u_old, u = u, u_new
        d_, c_ = probe(u)
        D.append(d_)
        Cl.append(c_)
    return np.array(D), np.array(Cl)


# ============================================================================================
# 3. MEASUREMENT HELPERS
# ============================================================================================
def env_peak(t, x):
    """(peak time, peak envelope amplitude) with parabolic sub-sample refinement."""
    e = np.abs(hilbert(x))
    i = int(np.argmax(e))
    if 0 < i < len(e) - 1:
        a, b, cc = e[i - 1], e[i], e[i + 1]
        den = a - 2 * b + cc
        frac = 0.5 * (a - cc) / den if den != 0 else 0.0
        return t[i] + frac * (t[1] - t[0]), b - 0.25 * (a - cc) * frac
    return t[i], e[i]


def dft(x, dt, f):
    """Single-frequency DFT at an ARBITRARY f (not an FFT bin). Bin quantisation would be
    fatal here: two runs with different dt land on different bins, and a 60 kHz offset over
    an 8 us arrival is ~3 rad of spurious phase.
    Convention: for x = cos(wt+phi) this returns ~(N/2)exp(i phi), i.e. the exp(+i w t)
    convention. A field written as Re[C exp(-i w t)] therefore maps to conj(C), so callers
    conjugate before interpreting the result as a physical coefficient."""
    x = np.asarray(x)
    return x @ np.exp(-2j * np.pi * f * np.arange(len(x)) * dt)


def slowness_fit(xy, t):
    """Least-squares plane fit t = t0 + p_x dx + p_y dy over the receiver grid.
    Returns (slowness vector, theta_deg from +y axis, speed).
    p_y is measured, so the direction is NOT Snell by construction."""
    A = np.column_stack([np.ones(len(t)), xy[:, 0] - xy[:, 0].mean(), xy[:, 1] - xy[:, 1].mean()])
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    p = coef[1:3]
    return p, np.degrees(np.arctan2(p[0], p[1])), 1.0 / np.hypot(*p)


def phase_refine(xy, Z, p0, omega):
    """Refine the slowness with the spectral PHASE gradient (Snell is a statement about phase
    slowness, and phase is far more precise than an envelope-peak pick).
    Phases wrap every wavelength, so we unwrap against the envelope-fit prediction p0: the
    residual only has to be within +-pi, which it is by a wide margin (the envelope fit is
    good to ~1.5 deg = ~0.2 rad over the array), so no answer is assumed - only the cycle."""
    k0 = omega * np.asarray(p0)
    x, y = xy[:, 0] - xy[:, 0].mean(), xy[:, 1] - xy[:, 1].mean()
    # rfft-bin phase corresponds to conj(field) -> phi = -(k.x) + const
    res = np.angle(np.exp(1j * (np.angle(Z) + (k0[0] * x + k0[1] * y))))
    A = np.column_stack([np.ones(len(x)), x, y])
    coef, *_ = np.linalg.lstsq(A, res, rcond=None)
    k = k0 - coef[1:3]
    return np.degrees(np.arctan2(k[0], k[1])), omega / np.hypot(*k)


def beam_average(theta_deg, b, f0, key):
    """Angular-spectrum-weighted average of an exact coefficient over the finite beam's
    angular spread - the yardstick for how much of a residual is 'finite beam' rather than
    'wrong FEM'.  A Gaussian beam of 1/e half-width b has an angular amplitude spectrum
    exp(-(dth/dth_e)^2) with dth_e = lambda/(pi b).  Mode conversion is ODD in incidence
    angle, so components arriving from the other side of the normal enter with T_S negated -
    which is exactly why |D_S| -> 0 at normal incidence.
    This is an ESTIMATE: it ignores the (small, quadratic) relative phases of the angular
    components at the receiver, which is why we quote it as an order-of-effect, not a target."""
    dthe = (C_F / f0) / (np.pi * b)
    d = np.linspace(-3 * dthe, 3 * dthe, 151)
    w = np.exp(-(d / dthe) ** 2)
    tot = 0.0 + 0.0j
    for dd, ww in zip(d, w):
        a, sgn = np.radians(theta_deg) + dd, 1.0
        if a < 0.0:
            a, sgn = -a, -1.0                      # mirror: mode conversion flips sign
        c = analytic(np.degrees(a))[key]
        tot += ww * (sgn * c if key in ("DS", "TS") else c)
    return tot / w.sum()


# ============================================================================================
# 4. ONE ANGLE
# ============================================================================================
def run_angle(theta_deg, h, degree, f0, ncyc, steel_mm, verbose=True):
    th = np.radians(theta_deg)
    ct, st = np.cos(th), np.sin(th)
    ex = analytic(theta_deg)

    lam_f = C_F / f0
    a_xi = 0.5 * ncyc * lam_f                       # 1/e half-length of the burst along the ray
    b = min(4.0e-3, 2.2e-3 / max(st, 0.55))         # beam 1/e half-width (narrower at big angles)

    # Receivers are read from DG0, i.e. at cell CENTROIDS, so snap every requested coordinate
    # onto a centroid: (floor(v/h)+0.5)h. Without this a nominal depth that happens to be a
    # multiple of h lands on a cell boundary, the containing-cell search may pick either side,
    # and the mirror pair used for the free-field reference stops being an exact mirror.
    snap = lambda v: (np.floor(v / h) + 0.5) * h

    # --- geometry (per angle: a tilted beam needs a thicker fluid layer) --------------------
    d_f = snap(1.0e-3 + 1.4e-3 / ct)                # fluid receiver depth below interface
    clear_y = 2.5 * a_xi * ct + 2.3 * b * st        # source support half-extent in y
    h_src = max(clear_y + 0.3e-3, d_f + 2.2 * a_xi + 0.3e-3)   # source depth below interface
    fluid_mm = (h_src + clear_y + 0.5e-3) * 1e3
    ext_x = 2.2 * a_xi * st + 2.0 * b * ct          # source support half-extent in x
    off_rec = np.array([-2.0e-3, -1.0e-3, 0.0, 1.0e-3, 2.0e-3])   # lateral fan (find beam peak)
    off_f = np.array([-1.0e-3, 0.0, 1.0e-3])

    def mode_geom(theta_t, prop, c_mode):
        """Receiver depths and axis slope for one transmitted mode, plus the steel thickness
        needed so the first backwall echo lands AFTER the direct arrival.  The depth span
        shrinks with cos(theta_t) so a near-grazing ray does not need a huge lateral run;
        a grazing ray is flagged unreachable (its coefficient is then not measurable, which
        is the honest answer - a plane-wave amplitude has no meaning for a head wave)."""
        if not prop:
            return snap(np.array([1.2e-3, 2.6e-3, 4.0e-3])), 0.0, True, 11.0e-3
        tq = np.tan(np.radians(theta_t))
        cq = np.clip(np.cos(np.radians(theta_t)), 0.45, 1.0)
        d = snap(1.2e-3 + np.array([0.0, 1.4e-3, 2.8e-3]) * cq)
        r = d.max() * np.hypot(1.0, tq)             # ray length to the deepest receiver
        need = 0.5 * (C_P * (r / c_mode + 0.9 * ncyc / f0) + d.max())
        return d, tq, d.max() * tq <= 8.0e-3, max(11.0e-3, need)

    dP, tanP, P_reach, sP_ = mode_geom(ex["thP"], ex["P_prop"], C_P)
    dS, tanS, S_reach, sS_ = mode_geom(ex["thS"], ex["S_prop"], C_S)
    steel = max(steel_mm * 1e-3, sP_ if P_reach else 0.0, sS_ if S_reach else 0.0)

    right_need = max(2.0 * b / ct,
                     dP.max() * tanP * P_reach + off_rec.max() + 1.0e-3,
                     dS.max() * tanS * S_reach + off_rec.max() + 1.0e-3,
                     d_f * st / ct + off_f.max() + 1.0e-3)
    x_H = h_src * st / ct + ext_x + 0.5e-3
    Lx_mm = (x_H + right_need + 0.5e-3) * 1e3

    Nx = int(round(Lx_mm * 1e-3 / h))
    Nyf = int(round(fluid_mm * 1e-3 / h))
    Nys = int(round(steel / h))
    Lx, y_i, Ly = Nx * h, Nyf * h, (Nyf + Nys) * h
    steel = Nys * h

    # --- source: u = grad(Phi), Phi = f(xi) w(eta): exactly curl-free -----------------------
    n_hat = np.array([st, ct])
    t_hat = np.array([-ct, st])
    Sx, Sy = x_H - h_src * st / ct, y_i - h_src
    k = 2 * np.pi * f0 / C_F

    def fields(X):
        dx, dy = X[0] - Sx, X[1] - Sy
        xi = n_hat[0] * dx + n_hat[1] * dy
        eta = t_hat[0] * dx + t_hat[1] * dy
        E = np.exp(-(xi / a_xi) ** 2)
        S_, Cc = np.sin(k * xi), np.cos(k * xi)
        f = E * S_
        fp = E * (-2 * xi / a_xi**2) * S_ + E * k * Cc
        fpp = (E * (4 * xi**2 / a_xi**4 - 2 / a_xi**2) * S_
               - 4 * E * xi / a_xi**2 * k * Cc - E * k**2 * S_)
        w = np.exp(-(eta / b) ** 2)
        wp = w * (-2 * eta / b**2)
        u = np.vstack([n_hat[0] * fp * w + t_hat[0] * f * wp,
                       n_hat[1] * fp * w + t_hat[1] * f * wp])
        v = -C_F * np.vstack([n_hat[0] * fpp * w + t_hat[0] * fp * wp,
                              n_hat[1] * fpp * w + t_hat[1] * fp * wp])
        return u, v

    # --- receivers: one fan per beam axis ---------------------------------------------------
    pts = []
    xr_f = [snap(x_H + d_f * st / ct + oo) for oo in off_f]     # reflected-axis x, snapped once
    for oo in off_f:                                  # 0-2  incident axis  (x = x_H - d tan th)
        pts.append([snap(x_H - d_f * st / ct + oo), y_i - d_f])
    for xx in xr_f:                                   # 3-5  reflected axis (x = x_H + d tan th)
        pts.append([xx, y_i - d_f])
    nmode = len(off_rec) * 3
    for dd_, tanq, reach in ((dP, tanP, P_reach), (dS, tanS, S_reach)):   # P fan then S fan
        for dd in dd_:
            for oo in off_rec:
                pts.append([snap(x_H + dd * tanq * reach + oo), y_i + dd])
    for xx in xr_f:                                   # MIRROR of the reflected fan (see below)
        pts.append([xx, y_i + d_f])
    pts = np.array(pts)
    i1, i2 = np.arange(0, 3), np.arange(3, 6)
    iP, iS = np.arange(6, 6 + nmode), np.arange(6 + nmode, 6 + 2 * nmode)
    iM = np.arange(6 + 2 * nmode, 9 + 2 * nmode)

    # --- timing -----------------------------------------------------------------------------
    t_hit = h_src / (C_F * ct)                        # burst centre crosses the interface
    t_inc = (h_src - d_f) / (C_F * ct)
    t_ref = (h_src + d_f) / (C_F * ct)
    T = t_ref + 2.2 * ncyc / f0
    # earliest backwall echo is at the DEEPEST receiver -> global safe gate for the steel
    t_cut = t_hit + (2 * steel - max(dP.max(), dS.max())) / C_P

    def simulate(y_mat, Tend, dt_fixed=None):
        """One march on the standard mesh. y_mat is the material interface height: pass y_i for
        the real water|steel problem, or something above the top to make the box ALL WATER.
        dt_fixed forces the reference run onto the MAIN run's step: central differences have
        a temporal dispersion of order (w dt)^2/24, so a different dt gives a different
        numerical phase velocity in the water and the reference phase no longer cancels
        (measured: 44 deg of spurious phase in arg(R) at 60 deg before this was pinned)."""
        dom = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]], [Nx, Nyf + Nys],
                                    mesh.CellType.quadrilateral)
        V, K, m, cd = build_operators(dom, y_mat, degree)
        dt = stable_dt(K, m) if dt_fixed is None else dt_fixed
        uF, vF = Function(V), Function(V)
        uF.interpolate(lambda X: fields(X)[0])
        vF.interpolate(lambda X: fields(X)[1])
        pr = Probe(dom, V, pts)
        ns = int(np.ceil(Tend / dt))
        t0 = time.time()
        Dv, Cv = march(K, m, cd, uF.x.array.copy(), vF.x.array.copy(), dt, ns, pr)
        return pr.xy, dt, np.arange(len(Dv)) * dt, Dv, Cv, ns, m.size, time.time() - t0

    if verbose:
        print(f"\n=== theta = {theta_deg:g} deg ===")
        print(f"  domain {Lx*1e3:.2f} x {Ly*1e3:.2f} mm  (water {y_i*1e3:.2f}, steel "
              f"{steel*1e3:.2f})  {Nx}x{Nyf+Nys} Q{degree} cells")
        print(f"  beam b={b*1e3:.2f} mm (angular spread ~{np.degrees(lam_f/(np.pi*b)):.2f} deg), "
              f"burst {ncyc:g} cyc @ {f0/1e6:g} MHz, {degree*lam_f/h:.1f} nodes/lam_water")
        for kk, rr, aa in (("P", P_reach, ex["thP"]), ("S", S_reach, ex["thS"])):
            if not rr and ex[kk + "_prop"]:
                print(f"  NOTE transmitted {kk} is grazing (theta_{kk}={aa:.1f} deg): its beam "
                      f"runs along the interface, coefficient not measurable")

    xy, dt, tt, D, Cu, nsteps, ndof, wall = simulate(y_i, T)
    # FREE-FIELD REFERENCE: identical mesh and source but water everywhere.
    #   The reflected field at F2 = (x2, y_i-|Y2|) and the free field at the MIRROR point
    #   M = (x2, y_i+|Y2|) have identical phase and identical diffraction history (a flat
    #   mirror preserves a beam), so their ratio is EXACTLY R - no geometric phase term, and
    #   Gaussian-beam spreading, numerical dispersion and pulse distortion all cancel.
    #   Without this, spreading over the extra path biases |R| low by ~10% at 60 deg (measured:
    #   the table's 'no-ref' column).  dt is pinned to the main run's - see simulate().
    xyr, dtr, ttr, Dr, _, nsr, _, wallr = simulate(Ly + h, t_ref + 1.6 * ncyc / f0, dt_fixed=dt)
    assert np.allclose(xy, xyr)
    if verbose:
        print(f"  main dt {dt*1e9:.3f} ns, {nsteps} steps, T {T*1e6:.2f} us, "
              f"gate t<{t_cut*1e6:.2f} us, {ndof} dofs, {wall:.1f}s")
        print(f"  free-field reference: dt {dtr*1e9:.3f} ns, {nsr} steps, {wallr:.1f}s")

    # --- fluid: |R| (amplitude and phase) ---------------------------------------------------
    split = 0.5 * (t_inc + t_ref)
    gi = np.where((tt < split)[:, None], D[:, i1], 0.0)     # incident-axis fan, early window
    gr = np.where((tt > split)[:, None], D[:, i2], 0.0)     # reflected-axis fan, late window
    pk_i = [env_peak(tt, gi[:, j]) for j in range(3)]
    pk_r = [env_peak(tt, gr[:, j]) for j in range(3)]
    j1 = int(np.argmax([p[1] for p in pk_i]))               # beam-peak receiver of each fan
    j2 = int(np.argmax([p[1] for p in pk_r]))
    t_pk_inc, A_inc = pk_i[j1]
    A_ref = pk_r[j2][1]
    # same lateral index in the mirror fan -> the two points are exact mirrors (uniform mesh,
    # interface on a cell boundary => centroid offsets above and below y_i are equal)
    assert abs((y_i - xy[i2[j2], 1]) - (xy[iM[j2], 1] - y_i)) < 1e-12
    assert abs(xy[i2[j2], 0] - xy[iM[j2], 0]) < 1e-12
    A_free = env_peak(ttr, Dr[:, iM[j2]])[1]
    R_abs = A_ref / A_free
    R_raw = A_ref / A_inc            # what you get WITHOUT the reference run (diffraction bias)
    # curl leakage in the fluid: must be ~0 (source is an exact gradient, water has mu=0)
    curl_leak = np.abs(hilbert(Cu[:, i1[j1]])).max() / (np.abs(hilbert(D[:, i1[j1]])).max() + 1e-300)

    Rc = np.array([np.conj(dft(gr[:, j2], dt, f) / dft(Dr[:, iM[j2]], dtr, f))
                   for f in f0 * np.array([0.8, 1.0, 1.25])])
    R_cplx = Rc[1]
    vel_err = (t_pk_inc - t_inc) / t_inc     # numerical group-velocity error in the water
    x1, Y1 = xy[i1[j1], 0], xy[i1[j1], 1] - y_i             # Y < 0 in the fluid

    # --- steel: transmitted P (from div) and S (from curl) ----------------------------------
    def steel_mode(idx, field, is_S, prop, c_mode):
        traces = np.where((tt < t_cut)[:, None], field[:, idx], 0.0)
        pk = [env_peak(tt, traces[:, j]) for j in range(len(idx))]
        times = np.array([p[0] for p in pk])
        amps = np.array([p[1] for p in pk])
        jb = int(np.argmax(amps))
        obs = amps[jb] / A_inc                          # observable amplitude ratio
        Yr, xr = xy[idx[jb], 1] - y_i, xy[idx[jb], 0]
        sgn = -1.0 if is_S else 1.0                     # curl(u_S) = +w^2/c_S^2 psi vs div = -...

        def geo(w):
            kx, kfy = w * st / C_F, w * ct / C_F
            pq = np.sqrt(complex((w / c_mode) ** 2 - kx**2))
            return np.exp(1j * (kx * (xr - x1) + pq * Yr - kfy * Y1))

        # exact prediction of the SAME observable (includes evanescent decay at this depth)
        Tq = ex["TS"] if is_S else ex["TP"]
        pred = sgn * Tq * (C_F / c_mode) ** 2 * geo(2 * np.pi * f0)
        # back out the interface coefficient (only meaningful for a propagating mode)
        w0 = 2 * np.pi * f0
        rat = dft(traces[:, jb], dt, f0) / dft(gi[:, j1], dt, f0)
        Tm = sgn * np.conj(rat) * (c_mode / C_F) ** 2 / geo(w0)
        ang = spd = angp = spdp = np.nan
        if prop:
            p, ang, spd = slowness_fit(xy[idx], times)
            Z = np.array([dft(traces[:, j], dt, f0) for j in range(len(idx))])
            angp, spdp = phase_refine(xy[idx], Z, p, w0)
        nl = len(off_rec)
        by_depth = [amps[nl * q:nl * (q + 1)].max() / A_inc for q in range(len(idx) // nl)]
        return dict(obs=obs, obs_exact=abs(pred), D=Tm * C_F / c_mode, ang=angp, spd=spdp,
                    ang_env=ang, spd_env=spd, t_peak=times[jb], by_depth=by_depth,
                    obs_axis=amps[nl // 2] / A_inc)   # strictly on-axis, shallowest receiver

    resP = steel_mode(iP, D, False, ex["P_prop"] and P_reach, C_P)
    resS = steel_mode(iS, Cu, True, ex["S_prop"] and S_reach, C_S)
    if verbose:
        f3 = lambda v: "[" + " ".join(f"{q:.3e}" for q in v) + "]"
        print(f"  |R|={R_abs:.4f} (exact {abs(ex['R']):.4f}, no-reference {R_raw:.4f})   "
              f"obs_P={resP['obs']:.3e} (exact {resP['obs_exact']:.3e}) "
              f"by depth {f3(resP['by_depth'])}")
        print(f"  obs_S={resS['obs']:.3e} (exact {resS['obs_exact']:.3e}) "
              f"by depth {f3(resS['by_depth'])}")

    # --- measured energy balance (propagating modes only) -----------------------------------
    E = R_abs**2
    if ex["P_prop"] and P_reach:
        E += (RHO_S * C_P * np.cos(np.radians(ex["thP"]))) / (RHO_F * C_F * ct) * abs(resP["D"]) ** 2
    if ex["S_prop"] and S_reach:
        E += (RHO_S * C_S * np.cos(np.radians(ex["thS"]))) / (RHO_F * C_F * ct) * abs(resS["D"]) ** 2

    # Resolution of the EVANESCENT boundary layer in the steel. Past a critical angle the
    # corresponding wave decays as exp(-Im(q) Y) with Im(q) = sqrt(kx^2 - (w/c)^2). Near grazing
    # incidence that decay length falls BELOW one element, and since arg(R) is produced entirely
    # by this layer the phase (not the amplitude, which is pinned at |R|=1) degrades there.
    w0 = 2 * np.pi * f0
    kx0 = w0 * st / C_F
    def _npd(c_mode):
        q = np.sqrt(complex((w0 / c_mode) ** 2 - kx0**2))
        return np.inf if abs(q.imag) < 1e-9 else degree / (q.imag * h)

    return dict(theta=theta_deg, ex=ex, R_abs=R_abs, R_raw=R_raw, R_cplx=R_cplx, R_band=Rc,
                npd_P=_npd(C_P), npd_S=_npd(C_S),
                P=resP, S=resS, energy=E, curl_leak=curl_leak, vel_err=vel_err,
                P_reach=P_reach, S_reach=S_reach, b=b,
                bavg={k: beam_average(theta_deg, b, f0, k) for k in ("R", "DP", "DS")})


# ============================================================================================
# 5. REPORTING
# ============================================================================================
def meas_ok(r, k):
    """True when mode k has a propagating transmitted wave whose beam stays in the model."""
    return r["ex"][k + "_prop"] and r[k + "_reach"]


def table(res):
    L = []
    ap = L.append
    ap("=" * 118)
    ap("MEASURED vs EXACT   (D = displacement amplitude coefficient; 'evan' = evanescent, no "
       "propagating wave)")
    ap("=" * 118)
    ap(f"{'th':>4} | {'|R| meas':>9} {'|R| exact':>9} {'err%':>7} | "
       f"{'argR m':>7} {'argR x':>7} | {'|D_P| m':>8} {'|D_P| x':>8} {'err%':>7} | "
       f"{'|D_S| m':>8} {'|D_S| x':>8} {'err%':>7} | {'E':>7}")
    ap("-" * 118)
    for r in res:
        ex = r["ex"]
        e_R = 100 * abs(r["R_abs"] - abs(ex["R"])) / abs(ex["R"])
        row = (f"{r['theta']:4.0f} | {r['R_abs']:9.4f} {abs(ex['R']):9.4f} {e_R:7.2f} | "
               f"{np.degrees(np.angle(r['R_cplx'])):7.1f} {np.degrees(np.angle(ex['R'])):7.1f} | ")
        for key in ("P", "S"):
            m_, x_ = abs(r[key]["D"]), abs(ex["D" + key])
            if not ex[key + "_prop"]:      # evanescent: compare the observable at the receiver
                row += f"{r[key]['obs']:8.1e} {r[key]['obs_exact']:8.1e} {'evan':>7} | "
            elif not r[key + "_reach"]:    # propagating but grazing -> not measurable
                row += f"{'-':>8} {x_:8.4f} {'graz':>7} | "
            elif x_ > 1e-3:
                row += f"{m_:8.4f} {x_:8.4f} {100*abs(m_-x_)/x_:7.2f} | "
            else:
                row += f"{m_:8.4f} {x_:8.4f} {'~0':>7} | "
        row += f"{r['energy']:7.4f}"
        ap(row)
    ap("-" * 118)
    ap("")
    ap("PHASE [deg].  Convention: fields ~ exp(i(k.x - wt)), u_S = curl(psi e_z).  arg(R) comes")
    ap("from the free-field reference run, so no geometric phase is subtracted from it at all.")
    ap("'band' is the spread of arg(R) measured at 0.8/1.0/1.25 f0 - the coefficients are exactly")
    ap("frequency independent, so any spread is purely numerical dispersion (an error bar).")
    ap(f"{'th':>4} | {'argR FEM':>9} {'argR exact':>10} {'band':>6} | "
       f"{'argDP FEM':>9} {'exact':>8} | {'argDS FEM':>9} {'exact':>8}")
    ap("-" * 84)
    for r in res:
        ex = r["ex"]
        band = np.degrees(np.ptp(np.angle(r["R_band"] * np.conj(r["R_cplx"]))))
        row = (f"{r['theta']:4.0f} | {np.degrees(np.angle(r['R_cplx'])):9.1f} "
               f"{np.degrees(np.angle(ex['R'])):10.1f} {band:6.1f} | ")
        for k in ("P", "S"):
            m = f"{np.degrees(np.angle(r[k]['D'])):9.1f}" if meas_ok(r, k) else f"{'-':>9}"
            row += f"{m} {np.degrees(np.angle(ex['D'+k])):8.1f} | "
        ap(row)
    ap("-" * 84)
    ap("")
    ap("TRANSMITTED DIRECTION AND SPEED, from a least-squares fit of the slowness vector over")
    ap("the receiver grid (envelope-peak times, then refined with the spectral phase gradient).")
    ap("p_y is measured independently, so the direction is NOT Snell by construction.")
    ap("The last two columns are the FEM nodes spanning the evanescent decay length in the steel")
    ap("past each critical angle ('-' = the wave still propagates). arg(R) is generated entirely")
    ap("by that boundary layer, so where it drops below ~3 nodes the PHASE of R degrades.")
    ap(f"{'th':>4} | {'thP env':>8} {'thP ph':>8} {'thP Snell':>9} {'cP m':>6} | "
       f"{'thS env':>8} {'thS ph':>8} {'thS Snell':>9} {'cS m':>6} | {'curl lk':>8} {'v err%':>7}"
       f" | {'n/evP':>6} {'n/evS':>6}")
    ap("-" * 122)
    for r in res:
        ex = r["ex"]
        f = lambda v, w=8: " " * (w - 4) + "  - " if not np.isfinite(v) else f"{v:{w}.2f}"
        g = lambda v: "     -" if not np.isfinite(v) else f"{v:6.0f}"
        row = f"{r['theta']:4.0f} | "
        for k in ("P", "S"):
            show = meas_ok(r, k) and abs(ex["D" + k]) > 1e-3     # a real, measurable beam
            m = r[k]
            row += (f"{f(m['ang_env'] if show else np.nan)} {f(m['ang'] if show else np.nan)} "
                    f"{f(ex['th' + k], 9)} {g(m['spd'] if show else np.nan)} | ")
        nn = lambda v: "     -" if not np.isfinite(v) else f"{v:6.2f}"
        ap(row + f"{r['curl_leak']:8.2e} {100*r['vel_err']:7.3f} | "
           f"{nn(r['npd_P'])} {nn(r['npd_S'])}")
    ap("-" * 122)
    ap("")
    ap("FINITE-BEAM CHECK.  The model launches a beam of finite width, not a true plane wave,")
    ap("so it necessarily measures the coefficient averaged over the beam's angular spectrum")
    ap("(+-lam/(pi b) at 1/e).  Where 'beam' differs from 'plane', the residual is the beam,")
    ap("not the FEM - this matters only next to a critical angle and the Rayleigh angle.")
    ap("'no-ref' is |R| computed WITHOUT the free-field reference run: the gap to '|R| FEM' is")
    ap("the Gaussian-beam diffraction bias that the reference run removes.")
    ap(f"{'th':>4} {'b/mm':>5} {'spread':>7} | {'|R| FEM':>8} {'plane':>8} {'beam':>8} "
       f"{'no-ref':>8} | {'|D_P| FEM':>9} {'plane':>8} {'beam':>8} | "
       f"{'|D_S| FEM':>9} {'plane':>8} {'beam':>8}")
    ap("-" * 122)
    for r in res:
        sp = np.degrees((C_F / 2.0e6) / (np.pi * r["b"]))
        row = f"{r['theta']:4.0f} {r['b']*1e3:5.2f} {sp:6.2f}d | "
        row += (f"{r['R_abs']:8.4f} {abs(r['ex']['R']):8.4f} {abs(r['bavg']['R']):8.4f} "
                f"{r['R_raw']:8.4f} | ")
        for k in ("P", "S"):
            m = f"{abs(r[k]['D']):9.4f}" if meas_ok(r, k) else f"{'-':>9}"
            pr = r["ex"][k + "_prop"]
            ee = f"{abs(r['ex']['D'+k]):8.4f}" if pr else f"{'evan':>8}"
            bb = f"{abs(r['bavg']['D'+k]):8.4f}" if pr else f"{'evan':>8}"
            row += f"{m} {ee} {bb} | "
        ap(row)
    ap("-" * 122)
    return "\n".join(L)


def plots(res):
    OUT.mkdir(parents=True, exist_ok=True)
    thd = np.linspace(0.01, 60, 601)
    ana = [analytic(t) for t in thd]
    aR = np.array([abs(a["R"]) for a in ana])
    # Past a critical angle the transmitted coefficient is the amplitude of an EVANESCENT
    # boundary layer, not of a transmitted beam - so mask it rather than plot a misleading
    # (and, near the Rayleigh pole, divergent) curve on a transmitted-amplitude axis.
    aDP = np.array([abs(a["DP"]) if a["P_prop"] else np.nan for a in ana])
    aDS = np.array([abs(a["DS"]) if a["S_prop"] else np.nan for a in ana])
    th_m = np.array([r["theta"] for r in res])

    def crit(ax):
        ax.axvline(TH_P_CRIT, color="0.4", ls=":", lw=1)
        ax.axvline(TH_S_CRIT, color="0.4", ls="-.", lw=1)
        ax.axvline(TH_RAYLEIGH, color="C3", ls="--", lw=1, alpha=0.6)
        ax.text(TH_RAYLEIGH, ax.get_ylim()[1] * 0.98, f" leaky Rayleigh {TH_RAYLEIGH:.1f}",
                fontsize=7, va="top", rotation=90, color="C3")
        ax.text(TH_P_CRIT, ax.get_ylim()[1] * 0.98, f" P crit {TH_P_CRIT:.2f}",
                fontsize=7, va="top", rotation=90)
        ax.text(TH_S_CRIT, ax.get_ylim()[1] * 0.98, f" S crit {TH_S_CRIT:.2f}",
                fontsize=7, va="top", rotation=90)

    # ---- amplitudes
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(thd, aR, "C0-", label="|R| exact")
    ax.plot(thd, aDP, "C1-", label="|D_P| exact (transmitted P)")
    ax.plot(thd, aDS, "C2-", label="|D_S| exact (MODE-CONVERTED shear)")
    mk = lambda k: np.array([abs(r[k]["D"]) if meas_ok(r, k) else np.nan for r in res])
    ax.plot(th_m, [r["R_abs"] for r in res], "C0o", ms=6, label="|R| FEM")
    ax.plot(th_m, mk("P"), "C1s", ms=6, label="|D_P| FEM")
    ax.plot(th_m, mk("S"), "C2^", ms=6, label="|D_S| FEM")
    for k, c, lab, mask in (("R", "C0", "exact averaged over the beam's angular spectrum", None),
                            ("DP", "C1", None, "P"), ("DS", "C2", None, "S")):
        v = [abs(r["bavg"][k]) if (mask is None or r["ex"][mask + "_prop"]) else np.nan
             for r in res]
        ax.plot(th_m, v, c, marker="x", ls="none", ms=7, mew=1.4, alpha=0.85, label=lab)
    ax.axhline(R_NORMAL, color="0.7", lw=0.8)
    ax.set_ylim(0, 1.20)
    crit(ax)
    ax.set_xlabel("incidence angle in water [deg]")
    ax.set_ylabel("displacement amplitude coefficient")
    ax.set_title("Water|steel: reflection, transmitted P, and mode-converted S\n"
                 f"FEM markers vs exact curves (normal incidence |R| = {R_NORMAL:.4f})\n"
                 "transmitted curves stop where that mode goes evanescent", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "amplitude.png", dpi=130)

    # ---- phases
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(thd, np.degrees(np.angle([a["R"] for a in ana])), "C0-", label="arg R exact")
    msk = lambda k: np.array([np.degrees(np.angle(a["D" + k])) if a[k + "_prop"] else np.nan
                              for a in ana])
    ax.plot(thd, msk("P"), "C1-", label="arg D_P exact")
    ax.plot(thd, msk("S"), "C2-", label="arg D_S exact")
    ax.plot(th_m, [np.degrees(np.angle(r["R_cplx"])) for r in res], "C0o", label="arg R FEM")
    pa = [np.degrees(np.angle(r["P"]["D"])) if meas_ok(r, "P") else np.nan for r in res]
    sa = [np.degrees(np.angle(r["S"]["D"])) if meas_ok(r, "S") else np.nan for r in res]
    ax.plot(th_m, pa, "C1s", label="arg D_P FEM")
    ax.plot(th_m, sa, "C2^", label="arg D_S FEM")
    # spread of arg R over the -6 dB band: a direct estimate of the numerical phase error
    lo = [np.degrees(np.angle(r["R_band"][0])) for r in res]
    hi = [np.degrees(np.angle(r["R_band"][2])) for r in res]
    ax.fill_between(th_m, lo, hi, color="C0", alpha=0.18,
                    label="arg R FEM spread over 0.8-1.25 f0")
    crit(ax)
    ax.set_xlabel("incidence angle [deg]")
    ax.set_ylabel("phase [deg]")
    ax.set_title("Phase of the coefficients (sign convention: exp(i(k.x - wt)), "
                 "u_S = curl(psi e_z))")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "phase.png", dpi=130)

    # ---- errors
    fig, ax = plt.subplots(figsize=(9, 4.6))
    eR = [100 * abs(r["R_abs"] - abs(r["ex"]["R"])) / abs(r["ex"]["R"]) for r in res]
    ok_ = lambda r, k: meas_ok(r, k) and abs(r["ex"]["D" + k]) > 1e-3
    eP = [100 * abs(abs(r["P"]["D"]) - abs(r["ex"]["DP"])) / abs(r["ex"]["DP"])
          if ok_(r, "P") else np.nan for r in res]
    eS = [100 * abs(abs(r["S"]["D"]) - abs(r["ex"]["DS"])) / abs(r["ex"]["DS"])
          if ok_(r, "S") else np.nan for r in res]
    ax.semilogy(th_m, np.maximum(eR, 1e-3), "C0o-", label="|R|")
    ax.semilogy(th_m, eP, "C1s-", label="|D_P|")
    ax.semilogy(th_m, eS, "C2^-", label="|D_S|")
    ax.semilogy(th_m, [100 * abs(1 - r["energy"]) for r in res], "k.--",
                label="energy residual |1-E|")
    crit(ax)
    ax.set_xlabel("incidence angle [deg]")
    ax.set_ylabel("relative error [%]")
    ax.set_title("FEM vs exact: relative error per coefficient")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "error.png", dpi=130)

    # ---- angles
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(thd, [a["thP"] for a in ana], "C1-", label="theta_P Snell")
    ax.plot(thd, [a["thS"] for a in ana], "C2-", label="theta_S Snell")
    ax.plot(th_m, [r["P"]["ang"] for r in res], "C1s", ms=7, label="theta_P FEM (slowness fit)")
    ax.plot(th_m, [r["S"]["ang"] for r in res], "C2^", ms=7, label="theta_S FEM (slowness fit)")
    crit(ax)
    ax.set_xlabel("incidence angle [deg]")
    ax.set_ylabel("transmitted angle in steel [deg]")
    ax.set_title("Transmitted P and mode-converted S directions, measured from the\n"
                 "arrival-time gradient (not imposed) vs Snell's law")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "angles.png", dpi=130)
    plt.close("all")


# ============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", default="0,5,10,13,15,20,25,30,40,50,60")
    ap.add_argument("--h", type=float, default=0.20, help="element size [mm]")
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--f0", type=float, default=2.0, help="centre frequency [MHz]")
    ap.add_argument("--ncyc", type=float, default=2.5)
    ap.add_argument("--steel", type=float, default=11.0, help="steel thickness [mm]")
    ap.add_argument("--check", action="store_true", help="analytic self-tests only")
    args = ap.parse_args()

    print(f"water  c={C_F} rho={RHO_F} mu=0        Z1={RHO_F*C_F:.4e}")
    print(f"steel  cP={C_P} cS={C_S} rho={RHO_S}   Z2={RHO_S*C_P:.4e}")
    print(f"       lam={LAM_S:.4e}  mu={MU_S:.4e}\n")
    ok = analytic_selfcheck()
    if args.check:
        return

    angles = [float(a) for a in args.angles.split(",")]
    res = []
    t00 = time.time()
    for a in angles:
        res.append(run_angle(a, args.h * 1e-3, args.degree, args.f0 * 1e6, args.ncyc, args.steel))
    print(f"\nsweep wall time {(time.time()-t00)/60:.1f} min")

    txt = table(res)
    print("\n" + txt)

    # ---- the four mandatory self-checks ----------------------------------------------------
    r0 = res[0] if res[0]["theta"] == 0 else None
    lines = ["", "MANDATORY SELF-CHECKS", "-" * 70]
    if r0 is not None:
        e = 100 * abs(r0["R_abs"] - R_NORMAL) / R_NORMAL
        lines.append(f"1. normal-incidence |R|: measured {r0['R_abs']:.4f} vs 0.9351 "
                     f"({e:.2f}%)  {'PASS' if e < 2 else 'FAIL'}")
        sh, sha = r0["S"]["obs"], r0["S"]["obs_axis"]
        lines.append(f"2. shear at normal incidence: |curl|/|div_inc| = {sha:.2e} on the beam "
                     f"axis, {sh:.2e} worst over the +-2 mm fan  {'PASS' if sh < 1e-2 else 'FAIL'}")
        lines.append(f"   (the residue is the finite beam's +-{np.degrees((C_F/2e6)/(np.pi*4e-3)):.1f} "
                     f"deg angular spread, not injected shear: the source is an exact gradient)")
    near = lambda t: min(abs(t - TH_P_CRIT), abs(t - TH_S_CRIT), abs(t - TH_RAYLEIGH)) < 4.0
    clean = [r for r in res if not near(r["theta"])]
    worst_c = max(abs(1 - r["energy"]) for r in clean)
    worst_E = max(abs(1 - r["energy"]) for r in res)
    lines.append(f"3. energy balance E = |R|^2 + sum(flux ratios), residual |1-E|:")
    lines.append("     " + "  ".join(f"{r['theta']:.0f}d:{100*abs(1-r['energy']):.1f}%"
                                     for r in res))
    lines.append(f"   worst {worst_E*100:.2f}% overall; {worst_c*100:.2f}% excluding angles "
                 f"within 4 deg of a critical or Rayleigh angle ({TH_P_CRIT:.1f}/"
                 f"{TH_S_CRIT:.1f}/{TH_RAYLEIGH:.1f})  {'PASS' if worst_c < 0.05 else 'CHECK'}")
    ev = [r for r in res if r["theta"] > TH_P_CRIT]
    if ev:
        lines.append("4. beyond the P critical angle (P evanescent, shear dominant):")
        for r in ev:
            lines.append(f"     theta={r['theta']:4.0f}  |div_P|/|div_inc| = {r['P']['obs']:.2e} "
                         f"(exact {r['P']['obs_exact']:.2e})   "
                         f"|curl_S|/|div_inc| = {r['S']['obs']:.2e}   "
                         f"S/P = {r['S']['obs']/max(r['P']['obs'],1e-300):8.1f}x   "
                         f"P by depth {' '.join(f'{q:.1e}' for q in r['P']['by_depth'])}")
    r20 = [r for r in res if abs(r["theta"] - 20) < 1e-9]
    if r20:
        r = r20[0]
        lines.append(f"   shear angle at 20 deg: measured {r['S']['ang']:.2f} deg, "
                     f"Snell {r['ex']['thS']:.2f} deg "
                     f"(err {abs(r['S']['ang']-r['ex']['thS']):.2f} deg); "
                     f"measured c_S {r['S']['spd']:.0f} m/s vs 3100")
    lines.append(f"   analytic self-checks: {'PASS' if ok else 'FAIL'}")
    lines.append("-" * 70)
    extra = "\n".join(lines)
    print(extra)

    plots(res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table.txt").write_text(txt + "\n" + extra + "\n")
    print(f"\nwrote {OUT}/{{amplitude,phase,error,angles}}.png and table.txt")


if __name__ == "__main__":
    main()
