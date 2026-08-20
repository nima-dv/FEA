r"""
Step 4 / Stage 2b — ANGLE-BEAM SHEAR inspection: the mode-converted (L1->S2) crack detector.

WHY (this is how ILI actually finds cracks)
--------------------------------------------
Normal incidence gave only a weak crack echo (sub-wavelength notch, §11.7). Real inspection tilts the
beam. At ~20 deg incidence in the fluid the transmitted P wave is BEYOND its critical angle
(sin_crit = c_f/c_P = 1500/5700 -> 15.3 deg) so it is EVANESCENT -> the only propagating wave in the
steel is the MODE-CONVERTED SHEAR wave, refracted to ~45 deg (Snell with c_S=3100):
    sin(phi_S) = (c_S/c_f) sin(theta) = (3100/1500) sin(20) = 0.707 -> phi_S = 45 deg.
That 45-deg shear wave sweeps to the back-wall crack and reflects strongly off the crack face — a far
better crack signal than normal incidence. This is the L1-S2 path the research team's beamformer uses.

We fire a directed 20-deg beam, aimed so the 45-deg shear reaches the backwall crack, and visualize
curl(u) (= the shear field). Baseline subtraction (with/without crack) isolates the crack's shear echo.

DEMONSTRATION (qualitative): shows the mode-conversion + shear-crack interaction mechanism. Full
timing validation of the L-S-S-S path + beamforming is Stage 2c. Same validated physics + dashpot ABC.

RUN (~6 min, 2 solves):  ./run.ps1 python3 repro/ili_angled.py  -> results/ili_angled/*.gif
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from mpi4py import MPI
import ufl
import basix
from dolfinx import mesh, fem
from dolfinx.fem import functionspace, form, Function, assemble_matrix, assemble_vector
from lib.paths import RESULTS

cP_s, cS_s, rho_s = 5700.0, 3100.0, 7850.0
c_f, rho_f = 1500.0, 1000.0
c_o, rho_o = 500.0, 500.0
mu_s = rho_s*cS_s**2; lam_s = rho_s*(cP_s**2-2*cS_s**2); lam_f = rho_f*c_f**2; lam_o = rho_o*c_o**2

S, T, h_crack, w_crack = 8e-3, 9.525e-3, 3e-3, 0.5e-3
x_fw, x_bw = S, S+T; x_tip = x_bw-h_crack
Lx, Ly = 0.0205, 0.030
y_crack = 0.021                                   # backwall crack in the upper region
theta = np.deg2rad(20.0)
nP = np.array([np.cos(theta), np.sin(theta)])     # P polarization / propagation direction
x0b, y0b, sig_par, w_perp = 2e-3, 9.3e-3, 0.6e-3, 4e-3   # beam aimed so 45-deg shear hits the crack
deg, Nx, Ny = 2, 68, 100
cell = Lx/Nx
OUT = RESULTS/"ili_angled"; OUT.mkdir(parents=True, exist_ok=True)

domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [Lx, Ly]], [Nx, Ny], mesh.CellType.quadrilateral)
el = basix.ufl.element("Lagrange", domain.basix_cell(), deg,
                       lagrange_variant=basix.LagrangeVariant.gll_warped, shape=(2,))
V = functionspace(domain, el); Q = functionspace(domain, ("DG", 0))
u_tr, v = ufl.TrialFunction(V), ufl.TestFunction(V)
eps = lambda w: ufl.sym(ufl.grad(w))
abs_facets = mesh.locate_entities_boundary(domain, 1, lambda X: np.isclose(X[0], 0.0))
ft = mesh.meshtags(domain, 1, abs_facets, np.ones(abs_facets.size, dtype=np.int32))
dsm = ufl.Measure("ds", domain=domain, subdomain_data=ft)
c_damp = assemble_vector(form(rho_f*c_f*v[0]*dsm(1))).array.copy()
nx = V.tabulate_dof_coordinates()


def envelope(X):
    dx, dy = X[0]-x0b, X[1]-y0b
    par = dx*nP[0]+dy*nP[1]; perp = -dx*nP[1]+dy*nP[0]
    return np.exp(-(par/sig_par)**2)*np.exp(-(perp/w_perp)**2), par


e0, par0 = envelope(nx.T)                          # (used only to shape; interpolate below)
uf = Function(V)
uf.interpolate(lambda X: (lambda env: np.vstack([nP[0]*env[0], nP[1]*env[0]]))(envelope(X)))
vf = Function(V)
vf.interpolate(lambda X: (lambda env: np.vstack([c_f*nP[0]*(2*env[1]/sig_par**2)*env[0],
                                                 c_f*nP[1]*(2*env[1]/sig_par**2)*env[0]]))(envelope(X)))
u0, v0 = uf.x.array.copy(), vf.x.array.copy()

dt = 0.4*(cell/deg**2)/cP_s
T_end = 14e-6
nsteps = int(T_end/dt); cap = max(1, nsteps//70)


def material(with_crack):
    xc = Q.tabulate_dof_coordinates(); x, y = xc[:, 0], xc[:, 1]
    lam = np.full(x.shape, lam_s); mu = np.full(x.shape, mu_s); rho = np.full(x.shape, rho_s)
    fluid = x < x_fw; air = x >= x_bw
    if with_crack:
        air = air | ((x >= x_tip) & (x < x_bw) & (np.abs(y-y_crack) <= w_crack/2))
    lam[fluid], mu[fluid], rho[fluid] = lam_f, 0, rho_f
    lam[air], mu[air], rho[air] = lam_o, 0, rho_o
    g = []
    for a in (lam, mu, rho):
        fn = Function(Q); fn.x.array[:] = a; g.append(fn)
    return g


def run(with_crack):
    lam_fn, mu_fn, rho_fn = material(with_crack)
    sig = lambda w: lam_fn*ufl.tr(eps(w))*ufl.Identity(2)+2*mu_fn*eps(w)
    K = assemble_matrix(form(ufl.inner(sig(u_tr), eps(v))*ufl.dx)); K.scatter_reverse(); K = K.to_scipy()
    m = assemble_vector(form(rho_fn*ufl.inner(v, ufl.as_vector((1.0, 1.0)))*ufl.dx)).array.copy()
    inv_a = 1/(m/dt**2+c_damp/(2*dt)); b = m/dt**2-c_damp/(2*dt); tm = 2*m/dt**2
    u_old = u0.copy(); u = u_old+dt*v0-0.5*dt**2*(c_damp*v0+K@u_old)/m
    snaps, times = [u_old.copy()], [0.0]
    for n in range(1, nsteps):
        un = inv_a*(tm*u-b*u_old-(K@u)); u_old, u = u, un
        if n % cap == 0:
            snaps.append(u.copy()); times.append(n*dt)
    return snaps, times


print(f"20deg beam -> 45deg shear in steel. deg {deg} mesh {Nx}x{Ny} steps {nsteps}")
print("solve 1/2 (crack)..."); snaps_c, times = run(True)
print("solve 2/2 (baseline)..."); snaps_r, _ = run(False)

# curl(u) = the shear field
W = functionspace(domain, ("Lagrange", 1)); ip = W.element.interpolation_points
Wx = W.tabulate_dof_coordinates(); triW = mtri.Triangulation(Wx[:, 0]*1e3, Wx[:, 1]*1e3)
utmp = Function(V)


def curl_of(snap):
    utmp.x.array[:] = snap
    cf = Function(W); cf.interpolate(fem.Expression(ufl.grad(utmp)[1, 0]-ufl.grad(utmp)[0, 1], ip))
    return cf.x.array.copy()


curl_c = [curl_of(s) for s in snaps_c]
curl_d = [curl_of(c)-curl_of(r) for c, r in zip(snaps_c, snaps_r)]


def make_gif(frames, fname, title, sat):
    vmax = sat*max(np.abs(f).max() for f in frames); lv = np.linspace(-vmax, vmax, 25)
    fig, ax = plt.subplots(figsize=(5.5, 6.5))

    def upd(k):
        ax.clear(); ax.tricontourf(triW, frames[k], levels=lv, cmap="RdBu_r", extend="both")
        ax.axvline(x_fw*1e3, color="k", lw=0.8); ax.axvline(x_bw*1e3, color="k", lw=0.8)
        ax.add_patch(Rectangle((x_tip*1e3, (y_crack-w_crack/2)*1e3), h_crack*1e3, w_crack*1e3,
                               fill=False, ec="lime", lw=1.5))
        ax.text(S*1e3/2, Ly*1e3*.97, "FLUID", ha="center", fontsize=8)
        ax.text((S+T/2)*1e3, Ly*1e3*.97, "STEEL", ha="center", fontsize=8)
        ax.set_aspect("equal"); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title(f"{title}\nt = {times[k]*1e6:5.2f} us")
    FuncAnimation(fig, upd, frames=len(frames)).save(OUT/fname, writer=PillowWriter(fps=12))
    kk = int(0.8*len(frames)); fig2, ax2 = plt.subplots(figsize=(5.5, 6.5))
    ax = ax2; upd(kk); fig2.savefig(OUT/fname.replace(".gif", "_frame.png"), dpi=110); plt.close("all")


print("rendering shear_field.gif ..."); make_gif(curl_c, "shear_field.gif", "Mode-converted SHEAR field curl(u)", 0.10)
print("rendering shear_crack_scatter.gif ..."); make_gif(curl_d, "shear_crack_scatter.gif",
                                                          "Crack shear-scatter (with - without crack)", 0.5)
print("wrote results/ili_angled/{shear_field, shear_crack_scatter}.gif (+ *_frame.png)")
