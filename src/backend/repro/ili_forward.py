r"""
ILI FORWARD SOLVE - the flagship simulation.

Elastodynamic FEM of the research team's frozen ILI scenario, producing channel data
in exactly the format their beamformer consumes, so their imaging chain can be run on
our physics and the resulting image compared with k-Wave's.

SCENARIO (see C:/code/readme/rnd-nima-fea-README.md section 3)
  water standoff 20 mm -> curved steel wall 9.525 mm (R_ID 193.675, R_OD 203.200 mm)
  4.0 x 1.0 mm traction-free notch cut into the OD, on the beam axis
  256 elements, 0.30 mm pitch, 4 MHz 1-cycle toneburst, steering 0 / -20 / +20 deg
  record PRESSURE at the element faces to 60 us

NUMERICS
  GLL spectral elements (low dispersion -> accurate ARRIVAL TIMES, which is the stated
  priority: timing >> amplitude); --degree defaults to 3, every published run uses 4.
  Row-sum lumped mass: the mesh is QUADS, so on the tensor-product GLL basis the row sum
  IS the exact classical SEM diagonal mass (verified to 7e-17 against the GLL weights).
  Explicit leapfrog.
  Monolithic single-displacement field with mu = 0 in the water: the fluid stiffness
  reduces to the volumetric lambda*div(u)div(v) term, which IS the acoustic wave, and
  displacement/traction continuity at the ID is then automatic. Validated separately
  (normal-incidence |R| 0.00% error; oblique mode conversion 0.81% at 20 deg).

THE THREE THINGS THAT MUST BE EXACTLY RIGHT
  1. QUANTITY. Their sensor records PRESSURE, not displacement. In the fluid
     p = -lambda_f * div(u). We form it as a LINEAR FUNCTIONAL per element:
        p_e(t) = (1/|F_e|) * integral_{F_e} -lambda_f div(u) ds
     Because div is linear, assembling that form against the test function gives a
     fixed row vector w_e with w_e . u = the integral. So p = W @ u is EXACT and costs
     one sparse mat-vec per step - no projection, no interpolation error.
  2. TIME ZERO. t = 0 is the instant the FIRST-FIRING element fires, with the polar
     delay law MIN-SHIFTED to zero. We do not re-derive that law - we call THEIR
     `bf.ray.polar_wave_angles_traveltimes`, so the convention matches by construction.
     Getting this wrong is a fixed bias of up to 19 us.
  3. TIME BASE. Our CFL-limited dt (~0.9 ns) is not their 2.6316 ns. We record at our
     dt and resample onto their 380 MHz grid (22801 samples) at the end.

SOURCE MODEL
  A "soft" traction source on the array facets, co-located with a dashpot absorbing
  condition on the same facets. A prescribed-displacement (Dirichlet) source would make
  the transducer plane a rigid reflector; we learned that the hard way - it bounced the
  front-wall echo back and caused a 15% back-wall timing error until a dashpot fixed it.

RUN
  ./run.ps1 python3 repro/ili_forward.py --probe              # timing probe, then stop
  ./run.ps1 python3 repro/ili_forward.py --angle 0            # full solve, 0 deg
  ./run.ps1 python3 repro/ili_forward.py --angle 20 --no-crack   # baseline (healthy wall)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import ufl
import basix
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import functionspace, form, Function, assemble_vector
from dolfinx.fem.petsc import assemble_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.bf_loader import load_beamformer  # noqa: E402
from lib.paths import RESULTS

# ---------------------------------------------------------------------------------------
# Frozen scenario constants (SI). Must match mesh/ili_mesh.py.
# ---------------------------------------------------------------------------------------
C_P, C_S, RHO_S = 5700.0, 3100.0, 7850.0          # steel
C_F, RHO_F = 1500.0, 1000.0                        # water
MU_S = RHO_S * C_S**2
LAM_S = RHO_S * (C_P**2 - 2 * C_S**2)
LAM_F = RHO_F * C_F**2

R_ID, R_OD = 0.193675, 0.203200
X_C, Z_C = 0.03825, -0.173675                      # pipe centre; R_ID + Z_C = 20 mm standoff
STANDOFF = 0.020
N_ELEM, PITCH, KERF = 256, 0.30e-3, 0.05e-3
ELEM_W = PITCH - KERF                              # 0.25 mm active width
ARRAY_X0 = 0.0
F0, N_CYCLE = 4.0e6, 1
T_END = 60.0e-6
DT_KWAVE = 2.6315789473684212e-9                   # their sample period (380 MHz)
N_SAMP_KWAVE = 22801

TAG_FLUID, TAG_STEEL = 1, 2
TAG_FILL = 3                                       # only present on a --notch-fill mesh
TAG_ARRAY, TAG_ID, TAG_OD, TAG_NOTCH, TAG_ABC = 10, 11, 12, 13, 14

# C5: k-Wave's "outside" material, taken verbatim from their driver script -
#     sound_speed_compression = 500, sound_speed_shear = 0, density = 500.
# Shear speed zero means mu = 0, so this is an ACOUSTIC material and our existing fluid
# machinery handles it unchanged. Its impedance is 2.5e5 against steel's 4.47e7, giving a
# reflection coefficient of -0.989 where a true free surface gives -1.000.
NOTCH_X = X_C          # the notch is on the beam axis, which is also the pipe centre's x
C_FILL, RHO_FILL = 500.0, 500.0
LAM_FILL = RHO_FILL * C_FILL**2

DEG = 3
CFL = 0.30

MESH_DIR = RESULTS / "ili_mesh"
OUT = RESULTS / "ili_forward"


# ---------------------------------------------------------------------------------------
def load_mesh(path: Path):
    """Read the gmsh mesh + tags. DOLFINx 0.11 moved gmshio to dolfinx.io.gmsh and
    returns a MeshData object; older versions returned a 3-tuple. Handle both."""
    from dolfinx.io import gmsh as dgmsh
    res = dgmsh.read_from_msh(str(path), MPI.COMM_WORLD, gdim=2)
    if isinstance(res, tuple):
        return res[0], res[1], res[2]
    return res.mesh, res.cell_tags, res.facet_tags


def toneburst(t: np.ndarray, f0: float, n_cycle: int) -> np.ndarray:
    """k-Wave's toneBurst: n_cycle sine cycles under a Gaussian envelope.

    k-Wave windows the burst with a Gaussian of the same total length; we reproduce
    that (a bare truncated sine would inject a broadband step and pollute the spectrum).
    """
    dur = n_cycle / f0
    s = np.sin(2 * np.pi * f0 * t)
    # Gaussian window centred on the burst, ~4 sigma across its length (k-Wave default)
    sigma = dur / 4.0
    w = np.exp(-0.5 * ((t - dur / 2) / sigma) ** 2)
    out = s * w
    out[(t < 0) | (t > dur)] = 0.0
    return out


def element_delays(bf, angle_deg: float) -> np.ndarray:
    """Per-element transmit delays [s] for the polar plane wave.

    We call THEIR ray code so the convention matches their beamformer by construction,
    then min-shift to zero: t = 0 is when the FIRST element fires.
    """
    x_el_mm = (ARRAY_X0 + np.arange(N_ELEM) * PITCH) * 1e3
    pos_mm = np.vstack((x_el_mm, np.zeros_like(x_el_mm)))          # (2, 256), z = 0
    trav, _ = bf.ray.polar_wave_angles_traveltimes(
        pos_mm, np.deg2rad(angle_deg), X_C * 1e3, Z_C * 1e3, R_ID * 1e3, C_F)
    d = np.asarray(trav, dtype=float).ravel()[:N_ELEM]
    if d.size != N_ELEM:
        raise RuntimeError(f"delay law returned {d.size} values, expected {N_ELEM}")
    # UNITS: their function is called with positions in MILLIMETRES and speed in m/s, so
    # it returns mm/(m/s) = MILLISECONDS. Convert to seconds. This is not cosmetic - left
    # unconverted the delay span came out 2774.752 us instead of 2.775 us, a 1000x error
    # that would have silently destroyed every arrival time.
    # Cross-check against their own recorded tx_delays: span 2.775 us at 0 deg,
    # 19.146 us at 20 deg. Asserted below.
    d = d * 1e-3
    d = d - d.min()                                                 # MIN-shifted
    span_us = np.ptp(d) * 1e6
    expected = {0.0: 2.775, 20.0: 19.146, -20.0: 19.146}.get(round(angle_deg, 3))
    if expected is not None and abs(span_us - expected) / expected > 0.02:
        raise RuntimeError(
            f"delay span {span_us:.3f} us disagrees with k-Wave's recorded "
            f"{expected} us at {angle_deg} deg by more than 2% - check units/convention")
    return d


# ---------------------------------------------------------------------------------------
def main() -> None:
    global DEG              # --degree may override it; must precede any use of DEG
    ap = argparse.ArgumentParser()
    ap.add_argument("--angle", type=float, default=0.0, help="steering angle [deg]")
    ap.add_argument("--mesh", default=str(MESH_DIR / "ili_mesh.msh"))
    ap.add_argument("--probe", action="store_true",
                    help="time a few hundred steps, extrapolate the full cost, then stop")
    ap.add_argument("--probe-steps", type=int, default=300)
    ap.add_argument("--t-end", type=float, default=T_END)
    ap.add_argument("--tag", default=None, help="output name suffix (default: angle)")
    ap.add_argument("--degree", type=int, default=DEG,
                    help="GLL element degree (default %d). RAISE THIS, do not refine the "
                         "mesh, if the returned pulse is too narrow-band: a 1-cycle burst "
                         "carries energy to ~2x the centre frequency, and sizing for the "
                         "CENTRE frequency low-passes the pulse during propagation, which "
                         "destroys axial resolution." % DEG)
    ap.add_argument("--snapshots", type=int, default=0,
                    help="save this many wavefield snapshots for the animations (0 = off). "
                         "Cheap: two DG0 interpolations per snapshot.")
    ap.add_argument("--snap-window", default=None, metavar="T0,T1",
                    help="restrict snapshots to this time window in us (default: whole run). "
                         "At 20 deg the interesting span is ~20,45 - the delay law alone "
                         "occupies the first 19.1 us.")
    ap.add_argument("--cfl", type=float, default=CFL,
                    help="CFL constant in dt = CFL*h_min/(c*p^2), default %.2f. This heuristic "
                         "is very conservative: tools/cfl_limit.py measures the EXACT limit as "
                         "2/sqrt(lambda_max(M^-1 K)) and finds our default sits at only ~16%% of "
                         "it, so ~5x larger steps are stable. Raising it must be validated "
                         "against arrival times, not merely checked for non-divergence." % CFL)
    ap.add_argument("--smear-mm", type=float, default=0.0,
                    help="E1: SMEAR the notch faces over this width in mm instead of keeping "
                         "them sharp (0 = off, sharp). k-Wave applies an analytically computed "
                         "soft interface of 2.5 pixels = 0.125 mm to its ID/OD and cracks, so "
                         "its geometry is smeared rather than staircased. This emulates that in "
                         "our solver to test whether interface SHARPNESS is what separates us. "
                         "Requires a --notch-fill mesh, so the notch interior has cells to "
                         "blend into.")
    ap.add_argument("--snap-degree", type=int, default=2,
                    help="DG degree for the snapshot sampling (default 2). NOT a free knob: "
                         "one sample per cell (degree 0) is only ~1.5 samples per water "
                         "wavelength on a 0.25 mm mesh - BELOW Nyquist - and the rendered "
                         "wavefronts come out beaded with aliasing speckle. Degree 2 gives 9 "
                         "samples per quad, ~4.5 per wavelength.")
    ap.add_argument("--gpu", action="store_true",
                    help="run the TIME LOOP on the GPU via cupy/cuSPARSE. Needs the "
                         "dvfenics:gpu image and --gpus all (use run.ps1 -Gpu). Meshing and "
                         "assembly stay on the CPU either way: DOLFINx has no native GPU "
                         "assembly and the stock image ships PETSc without CUDA, but the loop "
                         "touches neither - it is one SpMV plus elementwise vector work, which "
                         "is memory-bandwidth bound and measured 23x faster on one consumer "
                         "card (tools/gpu_probe.py). Results are NOT bit-identical to the CPU "
                         "path: cuSPARSE reduces in a different order, so expect fp64 round-off "
                         "differences. Validated against the stored CPU channel data - see "
                         "tools/gpu_gate.py.")
    ap.add_argument("--abc-legacy", action="store_true",
                    help="use the OLD absorbing boundary: one speed (c_P) on both displacement "
                         "components everywhere in the steel. Wrong wave for a shear method - it "
                         "reflects 30%% of a shear wave's amplitude at normal incidence. Kept "
                         "only so the improvement can be measured as a single variable.")
    ap.add_argument("--sponge-mm", type=float, default=0.0, metavar="L",
                    help="graded absorbing sponge of this width in EACH lateral margin, "
                         "outboard of the aperture (0 = off). Same material, so no impedance "
                         "jump; damping ramps quadratically from zero so there is no sharp "
                         "feature to reflect off. 8.0 is ~10 shear wavelengths at 4 MHz and "
                         "fits the existing dead margin, so it costs no extra cells.")
    ap.add_argument("--sponge-db", type=float, default=40.0, metavar="DB",
                    help="target ROUND-TRIP attenuation through the sponge, in dB (default 40). "
                         "MORE IS NOT BETTER: tests/test_abc.py measures the actual return and "
                         "it is best at 40 (-55 dB achieved), then DEGRADES - 60 gives -47, 200 "
                         "gives -43 - because damping strong enough to stop the wave makes the "
                         "layer effectively rigid, and a rigid layer reflects. Do not raise this "
                         "'to be safe'.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    DEG = args.degree
    bf = load_beamformer()

    # --- mesh ---------------------------------------------------------------------------
    t0 = time.time()
    domain, ct, ft = load_mesh(Path(args.mesh))
    x = domain.geometry.x
    print(f"mesh: {domain.topology.index_map(domain.topology.dim).size_local} cells, "
          f"{domain.geometry.x.shape[0]} vertices, loaded in {time.time()-t0:.1f}s")
    # Guard against a mm-vs-metre mixup, which would silently scale every wavespeed.
    assert 0.02 < x[:, 1].max() < 0.05, \
        f"mesh z-extent {x[:,1].max():.4g} m looks wrong - is the mesh in mm?"
    print(f"      extent x [{x[:,0].min()*1e3:.2f}, {x[:,0].max()*1e3:.2f}] mm, "
          f"z [{x[:,1].min()*1e3:.2f}, {x[:,1].max()*1e3:.2f}] mm")

    # --- spaces + region-wise material --------------------------------------------------
    el = basix.ufl.element("Lagrange", domain.basix_cell(), DEG,
                           lagrange_variant=basix.LagrangeVariant.gll_warped, shape=(2,))
    V = functionspace(domain, el)
    Q = functionspace(domain, ("DG", 0))
    ndof = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    print(f"      degree-{DEG} vector DOF = {ndof}")

    lam, mu, rho = Function(Q), Function(Q), Function(Q)
    lam.x.array[:] = LAM_F
    mu.x.array[:] = 0.0
    rho.x.array[:] = RHO_F
    steel_cells = ct.find(TAG_STEEL)
    lam.x.array[steel_cells] = LAM_S
    mu.x.array[steel_cells] = MU_S
    rho.x.array[steel_cells] = RHO_S
    fill_cells = ct.find(TAG_FILL)
    if fill_cells.size:
        # A --notch-fill mesh: the notch interior exists as cells and gets k-Wave's "outside"
        # material instead of being absent (absent = traction-free = a true free surface).
        lam.x.array[fill_cells] = LAM_FILL
        mu.x.array[fill_cells] = 0.0
        rho.x.array[fill_cells] = RHO_FILL
    print(f"      steel cells {steel_cells.size}, fluid cells "
          f"{ct.find(TAG_FLUID).size}, notch-fill cells {fill_cells.size}"
          + (f" (c {C_FILL:.0f} m/s, rho {RHO_FILL:.0f} - k-Wave's 'outside')"
             if fill_cells.size else " (none: notch is a traction-free void)"))

    if args.smear_mm > 0:
        # Blend steel into the notch fill across a band of width --smear-mm, evaluated at
        # QUADRATURE POINTS rather than per cell. That matters: our cells are ~0.24 mm, wider
        # than the 0.125 mm band being emulated, so a per-cell material could not represent the
        # transition at all. Degree-4 elements carry 5x5 quadrature points per cell (~48 um
        # spacing), which resolves a 0.125 mm band with ~3 points.
        if not fill_cells.size:
            raise SystemExit("--smear-mm needs a --notch-fill mesh: with the notch absent from "
                             "the mesh there is nothing to blend into.")
        xy = ufl.SpatialCoordinate(domain)
        r = ufl.sqrt((xy[0] - X_C) ** 2 + (xy[1] - Z_C) ** 2)
        # Signed distance to the notch slot: negative inside, positive in the steel. Exact for
        # each half-plane and adequate near the tip corners, which is where it matters.
        hw, z_tip = 0.5e-3, Z_C + R_OD - 4.0e-3
        d = ufl.max_value(abs(xy[0] - NOTCH_X) - hw, z_tip - xy[1])
        w = args.smear_mm * 1e-3
        phi = ufl.min_value(ufl.max_value(0.5 + d / w, 0.0), 1.0)      # 0 in notch, 1 in steel
        # The ID stays SHARP. Only the notch is smeared, so sharpness at the notch is the one
        # variable against the C5 run (filled notch, sharp faces).
        in_water = ufl.lt(r, R_ID)
        lam_e = ufl.conditional(in_water, LAM_F, LAM_FILL + phi * (LAM_S - LAM_FILL))
        mu_e = ufl.conditional(in_water, 0.0, phi * MU_S)
        rho_e = ufl.conditional(in_water, RHO_F, RHO_FILL + phi * (RHO_S - RHO_FILL))
        lam, mu, rho = lam_e, mu_e, rho_e
        print(f"      SMEARED notch faces over {args.smear_mm:.4f} mm "
              f"({args.smear_mm/0.05:.1f} k-Wave pixels; theirs is 2.5). ID left sharp.")

    u_tr, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))
    sig = lambda w: lam * ufl.tr(eps(w)) * ufl.Identity(2) + 2 * mu * eps(w)

    # --- operators ----------------------------------------------------------------------
    t0 = time.time()
    K = assemble_matrix(form(ufl.inner(sig(u_tr), eps(v)) * ufl.dx))
    K.assemble()
    from dolfinx.la import matrix_csr  # noqa: F401  (kept for clarity of intent)
    Ks = K.to_scipy() if hasattr(K, "to_scipy") else None
    if Ks is None:                      # PETSc matrix -> scipy
        import scipy.sparse as sp
        ai, aj, av = K.getValuesCSR()
        Ks = sp.csr_matrix((av, aj, ai), shape=K.getSize())
    m = assemble_vector(form(rho * ufl.inner(v, ufl.as_vector((1.0, 1.0))) * ufl.dx)).array.copy()
    print(f"      K nnz {Ks.nnz}, lumped mass min {m.min():.3e} "
          f"(assembled in {time.time()-t0:.1f}s)")
    assert m.min() > 0, "lumped mass has a non-positive entry - GLL variant wrong?"

    # --- absorbing boundary: dashpot on the array plane and the side walls ---------------
    ds_all = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    nrm = ufl.FacetNormal(domain)
    # Lysmer-Kuhlemeyer dashpot:
    #     traction = -rho [ c_P (u_dot.n) n  +  c_S (u_dot - (u_dot.n) n) ]
    # A P wave leaves NORMAL to the facet and must see rho*c_P; an S wave shears ALONG it and
    # must see rho*c_S. Every absorbing facet here is AXIS-ALIGNED (side walls x = const, the
    # z = 0 plane), so the tensor rho[c_P nn' + c_S(I - nn')] is diagonal and the
    # per-component coefficients below are EXACT - nothing is dropped, and the damping stays a
    # lumped diagonal vector so the leapfrog is unchanged.
    #
    # WHY THIS CHANGED. The previous version used ONE speed (C_P) for BOTH components
    # everywhere in the steel. For TT-T that is the wrong wave: the beam in the steel is
    # SHEAR, so it met rho*C_P against its true rho*C_S and reflected
    #     |Z_S - Z_P| / (Z_S + Z_P) = 30% of its amplitude, i.e. -10.6 dB,
    # at NORMAL incidence, off whichever side wall it was travelling toward. Measured
    # consequence: FEM wall clutter was 1.8-4.2 dB WORSE than k-Wave's in the last ~10 mm
    # before that wall, and better than k-Wave's everywhere else - and the excess swapped
    # ends when the steering angle flipped. --abc-legacy restores it for comparison.
    cP_eff = ufl.conditional(ufl.gt(mu, 0.0), C_P, C_F)
    cS_eff = ufl.conditional(ufl.gt(mu, 0.0), C_S, 0.0)   # water carries no shear wave
    if args.abc_legacy:
        c_x = c_z = rho * cP_eff
    else:
        c_x = rho * (cP_eff * nrm[0]**2 + cS_eff * (1.0 - nrm[0]**2))
        c_z = rho * (cP_eff * nrm[1]**2 + cS_eff * (1.0 - nrm[1]**2))
    damp_form = (c_x * v[0] + c_z * v[1]) * (ds_all(TAG_ARRAY) + ds_all(TAG_ABC))
    c_damp = assemble_vector(form(damp_form)).array.copy()
    print(f"      dashpot dofs {(c_damp != 0).sum()} "
          f"({'LEGACY scalar c_P' if args.abc_legacy else 'directional c_P/c_S'}"
          f"; array + side facets; OD/notch stay traction-free)")

    # --- optional sponge layer in the dead margins ---------------------------------------
    # A single-facet dashpot is first order: it is exact only at normal incidence and leaks
    # badly at grazing angles. A sponge fixes what the dashpot cannot.
    #
    # Crucially it uses the SAME material - same rho, same speeds - so there is no impedance
    # jump for the wave to reflect off. What ramps up is a viscous damping d(x), quadratically
    # from ZERO at the inner edge. Gradual is the whole trick: any abrupt change, in material
    # OR in damping, reflects. The wave is then attenuated on the way in and again on the way
    # back out, so what returns is d_max-suppressed twice.
    #
    # It lives in the margins OUTBOARD of the aperture, which contribute nothing to the
    # image, so it costs no extra cells - the domain is already 8 mm wider than the array at
    # each end, and 8 mm is ~10 shear wavelengths at 4 MHz.
    if args.sponge_mm > 0:
        L = args.sponge_mm * 1e-3
        # Derived from the array spec, not hard-coded, so this stays correct if the aperture
        # or element count changes. Nothing here depends on the steering angle.
        ap_x0, ap_x1 = ARRAY_X0, ARRAY_X0 + (N_ELEM - 1) * PITCH
        xgeo = domain.geometry.x
        x_lo, x_hi = float(xgeo[:, 0].min()), float(xgeo[:, 0].max())
        assert x_lo + L <= ap_x0 + 1e-9, (
            f"sponge would reach into the aperture on the left: margin "
            f"{(ap_x0 - x_lo)*1e3:.2f} mm < requested {args.sponge_mm} mm")
        assert x_hi - L >= ap_x1 - 1e-9, (
            f"sponge would reach into the aperture on the right: margin "
            f"{(x_hi - ap_x1)*1e3:.2f} mm < requested {args.sponge_mm} mm")
        xs = ufl.SpatialCoordinate(domain)
        s_l = ufl.max_value(0.0, (x_lo + L - xs[0]) / L)     # 0 at inner edge -> 1 at wall
        s_r = ufl.max_value(0.0, (xs[0] - (x_hi - L)) / L)
        s = ufl.max_value(s_l, s_r)
        # Amplitude decays as exp(-d*x/(2*rho*c)). For d = d_max*s^2 the integral over the
        # layer is d_max*L/3, so the ROUND TRIP exponent is d_max*L/(3*rho*c). Setting that
        # to (dB/20)*ln(10) gives the constant below (= 3*ln(10)/20). Size it on the FASTEST
        # local wave, which attenuates least; the slow wave then gets more than asked for.
        d_max = 0.34539 * rho * cP_eff * args.sponge_db / L
        c_sponge = assemble_vector(
            form(d_max * s**2 * ufl.inner(v, ufl.as_vector((1.0, 1.0))) * ufl.dx)).array
        c_damp = c_damp + c_sponge
        print(f"      sponge {args.sponge_mm:.1f} mm each margin, target round trip "
              f"-{args.sponge_db:.0f} dB, {(c_sponge != 0).sum()} dofs damped")

    # --- per-element source and pressure functionals -------------------------------------
    # Mark each element's active face with its own facet tag so we can assemble one
    # functional per element. Facets on TAG_ARRAY whose midpoint x falls inside element i's
    # active width belong to element i.
    afac = ft.find(TAG_ARRAY)
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    fmid = dmesh.compute_midpoints(domain, fdim, afac)
    xc_el = ARRAY_X0 + np.arange(N_ELEM) * PITCH
    # nearest element centre, then reject facets outside the active (non-kerf) width
    idx = np.abs(fmid[:, 0][:, None] - xc_el[None, :]).argmin(axis=1)
    inside = np.abs(fmid[:, 0] - xc_el[idx]) <= ELEM_W / 2
    print(f"      array facets {afac.size}, assigned to elements {inside.sum()} "
          f"(rest fall in the kerf gaps)")

    import scipy.sparse as sp
    n_local = ndof
    S_cols, P_rows = [], []
    normal = ufl.FacetNormal(domain)
    covered = 0
    for e in range(N_ELEM):
        sel = np.where(inside & (idx == e))[0]
        if sel.size == 0:
            S_cols.append(sp.csr_matrix((n_local, 1)))
            P_rows.append(sp.csr_matrix((1, n_local)))
            continue
        covered += 1
        etag = dmesh.meshtags(domain, fdim, np.sort(afac[sel]),
                              np.full(sel.size, 1, dtype=np.int32))
        dse = ufl.Measure("ds", domain=domain, subdomain_data=etag)(1)
        # source: unit traction along the inward face normal -> injects a normal velocity
        s = assemble_vector(form(ufl.inner(v, -normal) * dse)).array.copy()
        # pressure: p = -lambda_f div(u), face-averaged.  area from a unit form.
        area = assemble_vector(form(ufl.inner(v, ufl.as_vector((1.0, 1.0))) * dse)
                               ).array.sum() / 2.0
        w = assemble_vector(form(-LAM_F * ufl.div(v) * dse)).array.copy() / max(area, 1e-30)
        S_cols.append(sp.csr_matrix(s.reshape(-1, 1)))
        P_rows.append(sp.csr_matrix(w.reshape(1, -1)))
    S = sp.hstack(S_cols).tocsr()          # (ndof, 256)
    W = sp.vstack(P_rows).tocsr()          # (256, ndof)
    print(f"      built source/pressure functionals for {covered}/{N_ELEM} elements "
          f"(S nnz {S.nnz}, W nnz {W.nnz})")
    assert covered == N_ELEM, f"only {covered} of {N_ELEM} elements found facets"

    # --- time step from the ACTUAL mesh, per cell ----------------------------------------
    # dt <= CFL * h / (c * p^2), minimised over cells: the notch region in steel is the
    # binding constraint (see the H_NOTCH note in mesh/ili_mesh.py).
    tdim = domain.topology.dim
    ncell = domain.topology.index_map(tdim).size_local
    # Use the per-cell MINIMUM EDGE length, not domain.h(). domain.h() returns the cell
    # DIAMETER (longest vertex-to-vertex distance), which for a quad is the diagonal and so
    # overestimates the stable dt by ~sqrt(2) or more on stretched cells. Measured here:
    # diameter-based dt was 1.60 ns against 0.86 ns from min edge - an optimistic dt that
    # would look fine in a 300-step probe and diverge thousands of steps later.
    cmap = domain.geometry.dofmap.reshape(ncell, -1)
    xg = domain.geometry.x
    h_min = np.full(ncell, np.inf)
    for a in range(cmap.shape[1]):
        for b in range(a + 1, cmap.shape[1]):
            d = np.linalg.norm(xg[cmap[:, a]] - xg[cmap[:, b]], axis=1)
            h_min = np.minimum(h_min, d)
    c_cell = np.full(ncell, C_F)
    c_cell[steel_cells] = C_P
    if fill_cells.size:
        # Use the fill's OWN wavespeed for its CFL limit. At 500 m/s these cells are far from
        # binding even though they are small, so the notch fill costs no extra time steps.
        # EXCEPT when smearing: those cells then contain part steel, so they can carry the fast
        # wave and must be treated as steel for stability. Conservative on purpose.
        c_cell[fill_cells] = C_P if args.smear_mm > 0 else C_FILL
    dt_cell = args.cfl * h_min / (c_cell * DEG**2)
    if abs(args.cfl - CFL) > 1e-12:
        print(f"      CFL overridden: {args.cfl:.3f} (default {CFL:.3f}) -> "
              f"{args.cfl/CFL:.2f}x the usual step")
    dt = float(dt_cell.min())
    icrit = int(dt_cell.argmin())
    h = h_min
    nsteps = int(np.ceil(args.t_end / dt))
    print(f"\ntime: dt {dt*1e9:.4f} ns (binding cell {icrit}, h {h[icrit]*1e3:.4f} mm, "
          f"c {c_cell[icrit]:.0f} m/s), {nsteps} steps to {args.t_end*1e6:.1f} us")

    # --- source waveform per element -----------------------------------------------------
    delays = element_delays(bf, args.angle)
    print(f"delays: span {np.ptp(delays)*1e6:.3f} us, min {delays.min():.3e} "
          f"(min-shifted), max {delays.max()*1e6:.3f} us")

    # --- leapfrog ------------------------------------------------------------------------
    inv_a = 1.0 / (m / dt**2 + c_damp / (2 * dt))
    b_co = m / dt**2 - c_damp / (2 * dt)
    tm = 2.0 * m / dt**2

    # --- optional GPU offload of the time loop -------------------------------------------
    # Everything above is CPU: meshing, assembly, the boundary vector, the source and
    # receiver operators. Only the loop moves. `xp` is numpy or cupy, so the loop body below
    # is identical in both cases and cannot drift between the two paths.
    xp = np
    if args.gpu:
        import cupy as xp                                    # noqa: F811
        import cupyx.scipy.sparse as xsp
        Ks = xsp.csr_matrix(Ks)
        S = xsp.csr_matrix(S)
        W = xsp.csr_matrix(W)
        inv_a, b_co, tm = (xp.asarray(a) for a in (inv_a, b_co, tm))
        dev = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        free, total = xp.cuda.runtime.memGetInfo()
        print(f"      GPU time loop on {dev}: K on device, "
              f"{(total-free)/2**30:.2f} of {total/2**30:.1f} GiB used")

    u_old = xp.zeros(n_local)
    u_cur = xp.zeros(n_local)
    nrec = args.probe_steps if args.probe else nsteps
    rec = xp.zeros((nrec, N_ELEM))

    # --- optional wavefield snapshots (for the animations) -------------------------------
    # Stored as discontinuous-Lagrange sample values with their coordinates, NOT as XDMF, so
    # the renderer needs only numpy + matplotlib and never has to enter the container.
    # We save div(u) and curl(u) rather than u itself: by Helmholtz decomposition P is
    # curl-free and S is divergence-free, so these two fields separate the wave types and
    # make mode conversion at the ID directly visible. That is the whole point of D1.
    snaps_div, snaps_curl, snap_t, snap_xy = [], [], [], None
    snap_steel = None
    snap_set: set[int] = set()
    if args.snapshots > 0 and not args.probe:
        Qs = functionspace(domain, ("DG", args.snap_degree))
        u_fn = Function(V)
        ip = Qs.element.interpolation_points       # DOLFINx 0.11: attribute, not a method
        e_div = fem.Expression(ufl.div(u_fn), ip)
        e_curl = fem.Expression(u_fn[1].dx(0) - u_fn[0].dx(1), ip)
        f_div, f_curl = Function(Qs), Function(Qs)
        # Per-SAMPLE steel mask. mu lives on DG0, so it cannot be sliced to match a
        # higher-degree sample layout; interpolate it up instead. The renderer needs this to
        # know which samples are steel (plot curl) and which are water (plot div).
        mu_s = Function(Qs)
        mu_s.interpolate(mu)
        snap_steel = mu_s.x.array > 0.0
        if args.snap_window:
            ta, tb = (float(v) * 1e-6 for v in args.snap_window.split(","))
        else:
            ta, tb = 0.0, args.t_end
        n0, n1 = max(2, int(ta / dt)), min(nrec - 1, int(tb / dt))
        snap_set = set(np.unique(np.linspace(n0, n1, args.snapshots).astype(int)).tolist())
        snap_xy = Qs.tabulate_dof_coordinates()[:, :2].astype(np.float32)
        print(f"snapshots: {len(snap_set)} frames over {n0*dt*1e6:.2f}-{n1*dt*1e6:.2f} us, "
              f"DG{args.snap_degree} -> {snap_xy.shape[0]} samples/frame "
              f"({snap_xy.shape[0]/ncell:.1f} per cell)")

    def src(n: int):
        # The waveform is 256 numbers, so it is built on the host either way and pushed to
        # the device if needed - the transfer is negligible against a 1.3 GB/step SpMV.
        g = toneburst(n * dt - delays, F0, N_CYCLE)
        if not g.any():
            return None
        return S @ (xp.asarray(g) if args.gpu else g)

    t0 = time.time()
    for n in range(2, nrec):
        f = src(n)
        rhs = tm * u_cur - b_co * u_old - (Ks @ u_cur)
        if f is not None:
            rhs = rhs + f
        u_new = inv_a * rhs
        u_old, u_cur = u_cur, u_new
        rec[n] = W @ u_cur
        if n in snap_set:
            # DOLFINx interpolation is a host operation, so pull this frame back. Only a
            # handful of steps do this, so the copy does not affect the step rate.
            u_fn.x.array[:u_cur.size] = xp.asnumpy(u_cur) if args.gpu else u_cur
            f_div.interpolate(e_div)
            f_curl.interpolate(e_curl)
            snaps_div.append(f_div.x.array.astype(np.float32).copy())
            snaps_curl.append(f_curl.x.array.astype(np.float32).copy())
            snap_t.append(n * dt)
        # Divergence guard: an explicit scheme past its CFL limit grows geometrically.
        # Check periodically so a marginal dt fails in seconds rather than after an hour.
        if n % 500 == 0:
            # float() forces a device sync on the GPU path. Every 500 steps that is
            # unmeasurable; doing it per step would serialise the loop and throw the
            # speedup away.
            peak = float(xp.abs(u_cur).max())
            if not np.isfinite(peak) or peak > 1e6 * max(float(xp.abs(rec).max()), 1e-30):
                raise RuntimeError(
                    f"solution diverging at step {n}: max|u| = {peak:.3e}. "
                    f"dt = {dt*1e9:.4f} ns is above the stability limit - lower CFL.")
        if args.probe and n % 100 == 0:
            print(f"  step {n}/{nrec}  {(time.time()-t0)/max(n-1,1)*1e3:.2f} ms/step")
        elif not args.probe and n % 10000 == 0:
            # A production solve is hours long. Print an ETA occasionally so a background run
            # can be judged alive and on schedule without waiting for it to finish.
            el = time.time() - t0
            print(f"  step {n}/{nrec} ({100*n/nrec:.0f}%)  {el/(n-1)*1e3:.1f} ms/step  "
                  f"elapsed {el/60:.1f} min  ETA {(nrec-n)*el/(n-1)/60:.1f} min")

    per = (time.time() - t0) / max(nrec - 2, 1)
    print(f"\n{per*1e3:.3f} ms/step over {nrec-2} steps")
    if args.probe:
        tot = per * nsteps
        print(f"PROBE: full solve = {nsteps} steps x {per*1e3:.3f} ms = "
              f"{tot/60:.1f} min ({tot/3600:.2f} h) per angle")
        print(f"       3 angles + 1 baseline = {4*tot/3600:.2f} h")
        print(f"       peak |p| so far {float(xp.abs(rec).max()):.4g}")
        return

    # --- resample onto their 380 MHz time base -------------------------------------------
    if args.gpu:
        rec = xp.asnumpy(rec)                # one transfer, ~47 MB, at the end of the solve
    t_ours = np.arange(nrec) * dt
    t_theirs = np.arange(N_SAMP_KWAVE) * DT_KWAVE
    ch = np.empty((N_SAMP_KWAVE, N_ELEM))
    for e in range(N_ELEM):
        ch[:, e] = np.interp(t_theirs, t_ours, rec[:, e])

    name = args.tag or f"{args.angle:+.0f}deg".replace("+", "p").replace("-", "m")
    np.savez_compressed(OUT / f"channel_data_{name}.npz",
                        channel_data=ch, dt=DT_KWAVE, angle=args.angle,
                        tx_delays=delays, dt_solver=dt, t_end=args.t_end)
    print(f"wrote {OUT/f'channel_data_{name}.npz'}  shape {ch.shape}  "
          f"peak |p| {np.abs(ch).max():.4g}")

    if snaps_div:
        # steel mask lets the renderer draw the wall without re-deriving the geometry;
        # mu is DG0 on the same mesh, so its ordering matches snap_xy.
        p = OUT / f"wavefield_{name}.npz"
        np.savez_compressed(
            p, x=snap_xy[:, 0], z=snap_xy[:, 1],
            div=np.asarray(snaps_div), curl=np.asarray(snaps_curl),
            t=np.asarray(snap_t), steel=snap_steel,
            angle=args.angle, degree=DEG, snap_degree=args.snap_degree)
        print(f"wrote {p}  {len(snaps_div)} frames x {snap_xy.shape[0]} samples "
              f"({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
