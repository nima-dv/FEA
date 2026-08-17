r"""
Animate the ILI gate wavefield -> GIFs (better visualization of the wave and the crack).

Produces (results/ili_gate/):
  wavefield.gif      -- u_x field (with crack): the pulse crossing the fluid, reflecting off the
                        front wall, transmitting into the steel, bouncing off the back wall.
  crack_scatter.gif  -- the DIFFERENCE field (with-crack minus baseline): isolates the crack, so you
                        literally see the wave scatter off the crack tip.

Uses the validated Step-3b/4 physics (coupling + GLL-SEM + leapfrog + dashpot ABC). Coarser mesh than
the timing run (this is qualitative visualization, not a timing measurement).

RUN (~4 min, 2 solves):  ./run.ps1 python3 repro/animate_gate.py
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

# materials
cP_s, cS_s, rho_s = 5700.0, 3100.0, 7850.0
c_f, rho_f = 1500.0, 1000.0
c_o, rho_o = 500.0, 500.0
mu_s = rho_s*cS_s**2; lam_s = rho_s*(cP_s**2-2*cS_s**2); lam_f = rho_f*c_f**2; lam_o = rho_o*c_o**2
# geometry
S, T, h_crack, w_crack = 8e-3, 9.525e-3, 3e-3, 0.5e-3
x_fw, x_bw = S, S+T; x_tip = x_bw-h_crack
Lx, Ly = 0.0205, 0.016; y_crack = Ly/2
x0, sigma = 2e-3, 0.6e-3
deg, Nx, Ny = 2, 68, 53                       # coarser (viz only)
cell = Lx/Nx
t_bw = 2*(S-x0)/c_f + 2*T/cP_s                # ~ end of interest
OUT = Path(__file__).resolve().parents[1]/"results"/"ili_gate"; OUT.mkdir(parents=True, exist_ok=True)

domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [Lx, Ly]], [Nx, Ny], mesh.CellType.quadrilateral)
el = basix.ufl.element("Lagrange", domain.basix_cell(), deg,
                       lagrange_variant=basix.LagrangeVariant.gll_warped, shape=(2,))
V = functionspace(domain, el); Q = functionspace(domain, ("DG", 0))
u_tr, v = ufl.TrialFunction(V), ufl.TestFunction(V)
eps = lambda w: ufl.sym(ufl.grad(w))
abs_facets = mesh.locate_entities_boundary(domain, 1, lambda X: np.isclose(X[0], 0.0))
ft = mesh.meshtags(domain, 1, abs_facets, np.ones(abs_facets.size, dtype=np.int32))
ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
c_damp = assemble_vector(form(rho_f*c_f*v[0]*ds(1))).array.copy()
nx = V.tabulate_dof_coordinates()
f  = lambda X: np.exp(-((X[0]-x0)/sigma)**2)
fp = lambda X: np.exp(-((X[0]-x0)/sigma)**2)*(-2*(X[0]-x0)/sigma**2)
uf = Function(V); uf.interpolate(lambda X: np.vstack([f(X), np.zeros_like(X[0])]))
vf = Function(V); vf.interpolate(lambda X: np.vstack([-c_f*fp(X), np.zeros_like(X[0])]))
u0, v0 = uf.x.array.copy(), vf.x.array.copy()
tb = np.where(np.isclose(nx[:, 1], 0.0) | np.isclose(nx[:, 1], Ly))[0]; cidx = 2*tb+1
dt = 0.4*(cell/deg**2)/cP_s
nsteps = int((t_bw+2.5e-6)/dt)
cap_every = max(1, nsteps//70)


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
    sig = lambda w: lam_fn*ufl.tr(eps(w))*ufl.Identity(2) + 2*mu_fn*eps(w)
    K = assemble_matrix(form(ufl.inner(sig(u_tr), eps(v))*ufl.dx)); K.scatter_reverse(); K = K.to_scipy()
    m = assemble_vector(form(rho_fn*ufl.inner(v, ufl.as_vector((1.0, 1.0)))*ufl.dx)).array.copy()
    inv_a = 1/(m/dt**2+c_damp/(2*dt)); b = m/dt**2-c_damp/(2*dt); tm = 2*m/dt**2
    u_old = u0.copy(); u_old[cidx] = 0
    u = u_old+dt*v0-0.5*dt**2*(c_damp*v0+K@u_old)/m; u[cidx] = 0
    snaps, times = [u_old[0::2].copy()], [0.0]
    for n in range(1, nsteps):
        un = inv_a*(tm*u-b*u_old-(K@u)); un[cidx] = 0
        u_old, u = u, un
        if n % cap_every == 0:
            snaps.append(u[0::2].copy()); times.append(n*dt)
    return snaps, times


print(f"deg {deg} mesh {Nx}x{Ny} steps {nsteps}; solve 1/2 (crack)...")
snaps_c, times = run(True)
print("solve 2/2 (baseline)...")
snaps_r, _ = run(False)
diff = [c - r for c, r in zip(snaps_c, snaps_r)]

triang = mtri.Triangulation(nx[:, 0]*1e3, nx[:, 1]*1e3)   # one coord per node; matches u[0::2] (u_x)


def make_gif(frames, fname, title, sat):
    vmax = sat*max(np.abs(fr).max() for fr in frames)
    lv = np.linspace(-vmax, vmax, 25)
    fig, ax = plt.subplots(figsize=(7, 5.2))

    def upd(k):
        ax.clear()
        ax.tricontourf(triang, frames[k], levels=lv, cmap="RdBu_r", extend="both")
        ax.axvline(x_fw*1e3, color="k", lw=0.8); ax.axvline(x_bw*1e3, color="k", lw=0.8)
        ax.add_patch(Rectangle((x_tip*1e3, (y_crack-w_crack/2)*1e3), h_crack*1e3, w_crack*1e3,
                               fill=False, ec="lime", lw=1.5))
        ax.text(S*1e3/2, Ly*1e3*.93, "FLUID", ha="center", fontsize=8)
        ax.text((S+T/2)*1e3, Ly*1e3*.93, "STEEL", ha="center", fontsize=8)
        ax.set_aspect("equal"); ax.set_xlabel("x depth [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title(f"{title}\nt = {times[k]*1e6:5.2f} us")

    anim = FuncAnimation(fig, upd, frames=len(frames))
    anim.save(OUT/fname, writer=PillowWriter(fps=12))
    plt.close(fig)
    # a representative still (wave near the wall) for quick inspection
    kmid = int(0.72*len(frames))
    fig, ax = plt.subplots(figsize=(7, 5.2)); upd(kmid); fig.savefig(OUT/fname.replace(".gif", "_frame.png"), dpi=110); plt.close(fig)


print("rendering wavefield.gif ...")
make_gif(snaps_c, "wavefield.gif", "ILI wavefield u_x (with crack)", 0.12)
print("rendering crack_scatter.gif ...")
make_gif(diff, "crack_scatter.gif", "Crack-scattered field (with-crack minus baseline)", 0.5)
print("wrote results/ili_gate/{wavefield,crack_scatter}.gif (+ *_frame.png)")
