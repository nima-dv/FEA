r"""
Step 4 — ILI reproduction GATE (Stage 2a): reproduce the research team's k-Wave crack sim in FEniCS
and validate ECHO ARRIVAL TIMES (timing-first) against analytic geometric time-of-flight.

WHAT IT MODELS (assembled from the validated Step 0-3 machinery)
---------------------------------------------------------------
Pulse-echo inspection of a steel pipe wall through a fluid standoff, air-backed, with a backwall
crack notch. Layers along x (depth):   fluid (water) | steel wall (9.525 mm = 0.375") | air-like
plus a CRACK NOTCH cut into the steel from the back wall (3 mm deep, 0.5 mm wide).
Three echoes return to the transducer plane:
    FRONT WALL (fluid/steel)   BACK WALL (steel/air)   CRACK TIP (3 mm short of the back wall)

PHYSICS = the validated stack: monolithic displacement field, region-wise material (mu=0 in fluid &
air), GLL spectral elements + explicit leapfrog, + a DASHPOT absorbing boundary at the transducer
plane (so the strong front-wall echo does not reverberate and swamp the wall echoes). k-Wave is the
oracle; we validate against analytic geometric time-of-flight (no MATLAB needed).

ANALYTIC ARRIVALS at receiver x=xr, source at x=x0 (fluid c_f, steel compression c_L):
    t_FW    = (S-x0)/c_f + (S-xr)/c_f
    t_crack = t_FW + 2 (T - h_crack)/c_L
    t_BW    = t_FW + 2  T          /c_L      (crack is 2*h_crack/c_L ~ 1.05 us EARLIER than BW)

THE CRACK IS SUB-WAVELENGTH (0.5 mm), so at normal incidence its tip echo is WEAK vs the specular
back-wall echo (this is real: it's why ILI uses angled/mode-converted beams — Stage 2b/2c). To
extract & validate the crack echo we use BASELINE SUBTRACTION: run WITH and WITHOUT the crack and
difference the B-scans -> the FW/BW common echoes cancel, leaving the crack-tip echo.

GATE SIMPLIFICATIONS (documented; do not affect the timing physics being tested): flat wall (pipe
R=203mm -> 0.16mm sagitta, negligible); standoff 20->8 mm (fixed fluid delay only); plane-wave source
at normal incidence (script default 0 deg), not the 256-el steered array; broadband short pulse
(arrival TIMES are frequency-independent); structured mesh (front wall on a cell edge; notch at grid
scale — gmsh conforming mesh is the 2b refinement); 2-D, u_y=0 symmetry walls.

RUN (~8 min, two solves):  ./run.ps1 python3 repro/ili_gate.py   -> results/ili_gate/*.png
"""

import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import ufl
import basix
from dolfinx import mesh, fem
from dolfinx.fem import functionspace, form, Function, assemble_matrix, assemble_vector

# ---- materials (elastic; mu=0 for fluid & air-like) -----------------------------------------
cP_s, cS_s, rho_s = 5700.0, 3100.0, 7850.0
c_f, rho_f = 1500.0, 1000.0
c_o, rho_o = 500.0, 500.0
mu_s = rho_s * cS_s**2
lam_s = rho_s * (cP_s**2 - 2*cS_s**2)
lam_f = rho_f * c_f**2
lam_o = rho_o * c_o**2

# ---- geometry [m] ---------------------------------------------------------------------------
S = 8.0e-3           # fluid standoff (simplified from 20 mm)
T = 9.525e-3         # steel wall thickness (0.375", faithful)
h_crack = 3.0e-3     # crack depth from the back wall (faithful)
w_crack = 0.5e-3     # crack width (faithful)
x_fw, x_bw = S, S + T
x_tip = x_bw - h_crack
Lx, Ly = 0.0205, 0.016                              # Lx multiple of 0.25mm -> x_fw on a cell edge
y_crack = Ly / 2
x0, xr, sigma = 2.0e-3, 1.0e-3, 0.6e-3
deg, Nx, Ny = 3, 82, 64
cell = Lx / Nx                                      # 0.25 mm

t_fw = (S - x0)/c_f + (S - xr)/c_f
t_crack = t_fw + 2*(T - h_crack)/cP_s
t_bw = t_fw + 2*T/cP_s
print(f"analytic arrivals:  FW={t_fw*1e6:.3f}  crack={t_crack*1e6:.3f}  BW={t_bw*1e6:.3f} us"
      f"   (crack {(t_bw-t_crack)*1e6:.3f} us before BW = 2*{h_crack*1e3:.0f}mm/c_L)")

OUTDIR = Path(__file__).resolve().parents[1] / "results" / "ili_gate"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- mesh, space, and the pieces shared by both runs ----------------------------------------
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [Lx, Ly]], [Nx, Ny],
                               mesh.CellType.quadrilateral)
el = basix.ufl.element("Lagrange", domain.basix_cell(), deg,
                       lagrange_variant=basix.LagrangeVariant.gll_warped, shape=(2,))
V = functionspace(domain, el)
Q = functionspace(domain, ("DG", 0))
u_tr, v = ufl.TrialFunction(V), ufl.TestFunction(V)
eps = lambda w: ufl.sym(ufl.grad(w))

# dashpot absorbing boundary at x=0 (lumped damping on u_x)
abs_facets = mesh.locate_entities_boundary(domain, 1, lambda X: np.isclose(X[0], 0.0))
ft = mesh.meshtags(domain, 1, abs_facets, np.ones(abs_facets.size, dtype=np.int32))
ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
c_damp = assemble_vector(form(rho_f * c_f * v[0] * ds(1))).array.copy()

nx = V.tabulate_dof_coordinates()
# right-going plane P source in the fluid
f  = lambda X: np.exp(-((X[0]-x0)/sigma)**2)
fp = lambda X: np.exp(-((X[0]-x0)/sigma)**2) * (-2*(X[0]-x0)/sigma**2)
uf = Function(V); uf.interpolate(lambda X: np.vstack([f(X), np.zeros_like(X[0])]))
vf = Function(V); vf.interpolate(lambda X: np.vstack([-c_f*fp(X), np.zeros_like(X[0])]))
u0, v0 = uf.x.array.copy(), vf.x.array.copy()
tb = np.where(np.isclose(nx[:, 1], 0.0) | np.isclose(nx[:, 1], Ly))[0]
cidx = 2*tb + 1                                     # u_y = 0 on top/bottom
rmask = np.abs(nx[:, 0] - xr) < cell/4
ridx = np.where(rmask)[0]
ry = nx[ridx, 1]; order = np.argsort(ry); ridx, ry = ridx[order], ry[order]

dt = 0.4 * (cell/deg**2) / cP_s
T_end = t_bw + 3.0e-6
nsteps = int(T_end/dt)
tgrid = np.arange(nsteps + 1) * dt


def material_fields(with_crack):
    xc = Q.tabulate_dof_coordinates(); x, y = xc[:, 0], xc[:, 1]
    lam = np.full(x.shape, lam_s); mu = np.full(x.shape, mu_s); rho = np.full(x.shape, rho_s)
    fluid = x < x_fw
    airlike = x >= x_bw
    if with_crack:
        airlike = airlike | ((x >= x_tip) & (x < x_bw) & (np.abs(y - y_crack) <= w_crack/2))
    lam[fluid], mu[fluid], rho[fluid] = lam_f, 0.0, rho_f
    lam[airlike], mu[airlike], rho[airlike] = lam_o, 0.0, rho_o
    out = []
    for arr in (lam, mu, rho):
        g = Function(Q); g.x.array[:] = arr; out.append(g)
    return out


def run(with_crack):
    lam_fn, mu_fn, rho_fn = material_fields(with_crack)
    sig = lambda w: lam_fn * ufl.tr(eps(w)) * ufl.Identity(2) + 2.0 * mu_fn * eps(w)
    K = assemble_matrix(form(ufl.inner(sig(u_tr), eps(v)) * ufl.dx)); K.scatter_reverse(); K = K.to_scipy()
    m = assemble_vector(form(rho_fn * ufl.inner(v, ufl.as_vector((1.0, 1.0))) * ufl.dx)).array.copy()
    inv_a = 1.0 / (m/dt**2 + c_damp/(2*dt)); bcoef = m/dt**2 - c_damp/(2*dt); two_m = 2.0*m/dt**2
    u_old = u0.copy(); u_old[cidx] = 0.0
    u = u_old + dt*v0 - 0.5*dt**2 * (c_damp*v0 + K @ u_old)/m; u[cidx] = 0.0
    bscan = [u_old[2*ridx], u[2*ridx]]
    for _ in range(1, nsteps):
        u_new = inv_a * (two_m*u - bcoef*u_old - (K @ u)); u_new[cidx] = 0.0
        u_old, u = u, u_new
        bscan.append(u[2*ridx])
    return np.array(bscan)


print(f"mesh {Nx}x{Ny} deg {deg}  DOFs {V.dofmap.index_map.size_global}  dt {dt*1e9:.2f} ns  steps {nsteps}")
print("solve 1/2: with crack ...");    Bc = run(True)
print("solve 2/2: baseline (no crack) ..."); Br = run(False)
diff = Bc - Br                                       # baseline subtraction isolates the crack echo


def peak_time(trace, tmin, tmax):
    w = (tgrid >= tmin) & (tgrid <= tmax)
    return tgrid[np.where(w)[0][np.argmax(np.abs(trace[w]))]]


j_away = int(np.argmin(np.abs(ry - 2.0e-3)))
j_crack = int(np.argmin(np.abs(ry - y_crack)))
fw = peak_time(Bc[:, j_away], 0, t_fw + 1e-6)
bw = peak_time(Bc[:, j_away], t_fw + 1e-6, T_end)                   # away -> back wall
# crack tip = the difference peak BETWEEN the front wall and the back wall (isolate it from the
# stronger "removed back-wall" feature by ending the window halfway to t_bw).
ck = peak_time(diff[:, j_crack], t_fw + 1e-6, 0.5*(t_crack + t_bw))
print(f"\nFRONT WALL : analytic {t_fw*1e6:7.3f}  measured {fw*1e6:7.3f} us   err {abs(fw-t_fw)/t_fw*100:.2f}%")
print(f"BACK  WALL : analytic {t_bw*1e6:7.3f}  measured {bw*1e6:7.3f} us   err {abs(bw-t_bw)/t_bw*100:.2f}%  (away from crack)")
print(f"CRACK TIP  : analytic {t_crack*1e6:7.3f}  measured {ck*1e6:7.3f} us   err {abs(ck-t_crack)/t_crack*100:.2f}%  (baseline-subtracted)")
print(f"crack earlier than BW: analytic {(t_bw-t_crack)*1e6:.3f}  measured {(bw-ck)*1e6:.3f} us")

# ---- figures --------------------------------------------------------------------------------
gx = np.linspace(0, Lx, 400); gy = np.linspace(0, Ly, 300); GX, GY = np.meshgrid(gx, gy)
reg = np.ones_like(GX); reg[(GX >= x_fw) & (GX < x_bw)] = 2; reg[GX >= x_bw] = 0
reg[(GX >= x_tip) & (GX < x_bw) & (np.abs(GY - y_crack) <= w_crack/2)] = 0
fig, ax = plt.subplots(figsize=(6, 4))
ax.contourf(gx*1e3, gy*1e3, reg, levels=[-.5, .5, 1.5, 2.5], colors=["#eee", "#9cf", "#bbb"])
ax.axvline(xr*1e3, color="r", ls=":", label="receiver plane / absorbing wall")
ax.text(S*1e3/2, Ly*1e3*.9, "FLUID", ha="center"); ax.text((S+T/2)*1e3, Ly*1e3*.9, "STEEL", ha="center")
ax.annotate("crack (3x0.5mm)", (x_tip*1e3, y_crack*1e3), (x_tip*1e3-6, y_crack*1e3+3),
            arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("x depth [mm]"); ax.set_ylabel("y [mm]"); ax.set_aspect("equal"); ax.legend(loc="lower right")
ax.set_title("ILI gate geometry"); fig.tight_layout(); fig.savefig(OUTDIR/"geometry.png", dpi=120)

def bscan_plot(B, fname, title, sat, mark_crack=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    lim = sat * np.max(np.abs(B))
    ax.imshow(B, aspect="auto", origin="lower", cmap="gray_r", vmin=-lim, vmax=lim,
              extent=[ry.min()*1e3, ry.max()*1e3, tgrid.min()*1e6, tgrid.max()*1e6])
    ax.axhline(t_fw*1e6, color="g", ls="--", lw=1, label="front wall (analytic)")
    ax.axhline(t_bw*1e6, color="b", ls="--", lw=1, label="back wall (analytic)")
    if mark_crack:
        ax.plot([(y_crack-w_crack)*1e3, (y_crack+w_crack)*1e3], [t_crack*1e6]*2, "r-", lw=3,
                label="crack tip (analytic)")
    ax.set_xlabel("receiver y [mm]"); ax.set_ylabel("time [us]"); ax.set_title(title)
    ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig(OUTDIR/fname, dpi=120)

bscan_plot(Bc, "bscan.png", "B-scan (with crack): front & back wall echoes", 0.12)
bscan_plot(diff, "bscan_diff.png",
           "Baseline-subtracted B-scan: the CRACK-TIP echo, isolated at y=8mm", 0.5, mark_crack=True)

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(tgrid*1e6, Bc[:, j_crack]/np.max(np.abs(Bc[:, j_crack])), lw=1, label="behind crack (raw)")
ax.plot(tgrid*1e6, diff[:, j_crack]/np.max(np.abs(diff[:, j_crack])), lw=1.3, label="crack - baseline")
for tt, c, lab in [(t_fw, "g", "FW"), (t_crack, "r", "crack"), (t_bw, "b", "BW")]:
    ax.axvline(tt*1e6, color=c, ls="--", lw=1); ax.text(tt*1e6, 1.05, lab, color=c, ha="center")
ax.set_xlabel("time [us]"); ax.set_ylabel("u_x at y=8mm (norm.)"); ax.legend(loc="lower right")
ax.set_title("Crack echo isolated by baseline subtraction -> peaks at the analytic crack-tip time")
fig.tight_layout(); fig.savefig(OUTDIR/"ascans.png", dpi=120)
np.savez(OUTDIR/"bscans.npz", Bc=Bc, Br=Br, tgrid=tgrid, ry=ry,
         t_fw=t_fw, t_crack=t_crack, t_bw=t_bw)     # so re-analysis needs no re-solve
print("\nwrote results/ili_gate/{geometry, bscan, bscan_diff, ascans}.png + bscans.npz")
