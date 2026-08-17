r"""
Faithful ILI reproduction: CORROSION (ID) + CRACK (OD), realistic rendering + animation.

Matches the k-Wave sim (kwave_ili_func.m) which runs deformation="corrosion" AND include_crack:
  * transducer INSIDE the pipe -> ID = front wall (near) -> steel (9.525mm) -> OD = back wall (far) -> air
  * CORROSION on the ID (front/inner wall): irregular metal loss, profile = random + sin(7) + sin(2),
    smoothed (k-Wave 'corrosion' case, applied to the 'id' side) -> a rough eaten inner surface.
  * CRACK on the OD (backwall): exact k-Wave notch, 0.5mm wide x 3.0mm deep, cut radially into the OD,
    air-filled (crack_coords_base, angle=0). Both are real AIR voids in the material (not annotations).
  * REAL 20 mm standoff, normal incidence (script default).

Physics = validated stack (mu=0 fluid/air coupling, GLL-SEM, explicit leapfrog, dashpot ABC). Two
solves (crack vs no-crack, corrosion in BOTH) so the difference isolates the CRACK. Snapshots saved
to snaps.npz -> re-render with render_realistic.py, no re-solve.

RUN (~8 min):  ./run.ps1 python3 repro/ili_realistic.py  -> results/ili_realistic/*
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

cP_s, cS_s, rho_s = 5700.0, 3100.0, 7850.0
c_f, rho_f = 1500.0, 1000.0
c_o, rho_o = 500.0, 500.0
mu_s = rho_s*cS_s**2; lam_s = rho_s*(cP_s**2-2*cS_s**2); lam_f = rho_f*c_f**2; lam_o = rho_o*c_o**2

S, T, h_crack, w_crack, root_ext = 20.0e-3, 9.525e-3, 3.0e-3, 0.5e-3, 0.3e-3
x_fw, x_bw = S, S+T; x_tip = x_bw - h_crack
Lx, Ly, y_crack = 0.0324, 0.020, 0.010
x0, sigma = 2.0e-3, 0.6e-3
deg, Nx, Ny = 2, 108, 67
cell = Lx/Nx
OUT = Path(__file__).resolve().parents[1]/"results"/"ili_realistic"; OUT.mkdir(parents=True, exist_ok=True)

# --- CORROSION profile on the ID: random + sin(7) + sin(2), smoothed (k-Wave 'corrosion') ----
_rng = np.random.default_rng(1)
_yg = np.linspace(0, Ly, 500)
_noise = _rng.random(500) + np.sin(2*np.pi*7*_yg/Ly) + np.sin(2*np.pi*2*_yg/Ly)
_sm = np.convolve(_noise, np.ones(11)/11, mode="same")
_sm = (_sm - _sm.min())/(np.ptp(_sm)+1e-30)
corr_max = 1.0e-3                                    # up to 1 mm metal loss on the ID
_corr = corr_max*_sm
corr_of = lambda y: np.interp(y, _yg, _corr)         # corrosion depth (metal loss) vs y


def in_crack(x, y):
    return (x >= x_tip) & (x < x_bw + root_ext) & (np.abs(y - y_crack) <= w_crack/2)


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
f  = lambda X: np.exp(-((X[0]-x0)/sigma)**2)
fp = lambda X: np.exp(-((X[0]-x0)/sigma)**2)*(-2*(X[0]-x0)/sigma**2)
uf = Function(V); uf.interpolate(lambda X: np.vstack([f(X), np.zeros_like(X[0])]))
vf = Function(V); vf.interpolate(lambda X: np.vstack([-c_f*fp(X), np.zeros_like(X[0])]))
u0, v0 = uf.x.array.copy(), vf.x.array.copy()
tb = np.where(np.isclose(nx[:, 1], 0.0) | np.isclose(nx[:, 1], Ly))[0]; cidx = 2*tb+1
dt = 0.4*(cell/deg**2)/cP_s
t_bw = 2*(S-x0)/c_f + 2*T/cP_s
nsteps = int((t_bw+3e-6)/dt); cap = max(1, nsteps//90)


def material(with_crack):
    xc = Q.tabulate_dof_coordinates(); x, y = xc[:, 0], xc[:, 1]
    lam = np.full(x.shape, lam_s); mu = np.full(x.shape, mu_s); rho = np.full(x.shape, rho_s)
    fluid = x < (x_fw + corr_of(y))                  # corroded (rough, eaten) ID front surface
    air = x >= x_bw
    if with_crack:
        air = air | in_crack(x, y)
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
    u_old = u0.copy(); u_old[cidx] = 0
    u = u_old+dt*v0-0.5*dt**2*(c_damp*v0+K@u_old)/m; u[cidx] = 0
    snaps, times = [u_old[0::2].copy()], [0.0]
    for n in range(1, nsteps):
        un = inv_a*(tm*u-b*u_old-(K@u)); un[cidx] = 0
        u_old, u = u, un
        if n % cap == 0:
            snaps.append(u[0::2].copy()); times.append(n*dt)
    return snaps, times


print(f"standoff {S*1e3:.0f}mm, wall {T*1e3:.3f}mm | CORROSION on ID (<= {corr_max*1e3:.1f}mm loss) + "
      f"CRACK on OD ({h_crack*1e3:.0f}x{w_crack*1e3:.1f}mm notch)")
print(f"deg {deg} mesh {Nx}x{Ny} steps {nsteps}; solve 1/2 (corrosion+crack)..."); snaps_c, times = run(True)
print("solve 2/2 (corrosion only, baseline)..."); snaps_r, _ = run(False)
diff = [c-r for c, r in zip(snaps_c, snaps_r)]
np.savez_compressed(OUT/"snaps.npz", snaps_c=np.array(snaps_c), snaps_r=np.array(snaps_r),
                    times=np.array(times), coords=nx[:, :2], corr_y=_yg, corr=_corr)
triang = mtri.Triangulation(nx[:, 0]*1e3, nx[:, 1]*1e3)

_yl = np.linspace(0, Ly, 400)
_idprofile = (x_fw + corr_of(_yl))*1e3               # corroded ID surface (mm) for overlays


def draw_scene(ax):
    ax.plot(_idprofile, _yl*1e3, color="k", lw=1.0)                 # rough corroded ID (front wall)
    ax.axvline(x_bw*1e3, color="k", lw=1.0)                        # OD (back wall)
    ax.add_patch(Rectangle((x_tip*1e3, (y_crack-w_crack/2)*1e3), (x_bw-x_tip+root_ext)*1e3, w_crack*1e3,
                           facecolor="black", edgecolor="black", zorder=5))            # crack void
    ax.annotate("crack (backwall/OD)", (x_bw*1e3, (y_crack+w_crack)*1e3+0.6), (x_bw*1e3-9, Ly*1e3*0.82),
                fontsize=8, arrowprops=dict(arrowstyle="->"))
    ax.annotate("corrosion (ID)", (x_fw*1e3, Ly*1e3*0.3), (x_fw*1e3-11, Ly*1e3*0.2),
                fontsize=8, arrowprops=dict(arrowstyle="->"))
    ax.text(S*1e3*0.5, Ly*1e3*0.05, "FLUID", ha="center", fontsize=8)
    ax.text((S+T/2)*1e3, Ly*1e3*0.05, "STEEL", ha="center", fontsize=8)
    ax.text(0.6, Ly*1e3*0.5, "transducer\n(inside pipe)", rotation=90, va="center", fontsize=7)
    ax.set_aspect("equal"); ax.set_xlabel("x depth [mm]"); ax.set_ylabel("y [mm]")
    ax.set_xlim(0, Lx*1e3); ax.set_ylim(0, Ly*1e3)


# geometry figure
fig, ax = plt.subplots(figsize=(8, 5))
gx = np.linspace(0, Lx, 500); gy = np.linspace(0, Ly, 320); GX, GY = np.meshgrid(gx, gy)
reg = np.where(GX < (x_fw + corr_of(GY)), 0, np.where(GX < x_bw, 1, 2)).astype(float)
reg[in_crack(GX, GY)] = 2
ax.contourf(gx*1e3, gy*1e3, reg, levels=[-.5, .5, 1.5, 2.5], colors=["#cfe8ff", "#c9c9c9", "#ffffff"])
draw_scene(ax); ax.set_title("Faithful ILI geometry — corrosion on the ID + crack on the OD")
fig.tight_layout(); fig.savefig(OUT/"geometry.png", dpi=120)


def make_gif(frames, tt, fname, title):
    gmax = max(np.abs(fr).max() for fr in frames)+1e-30; scale = 0.06*gmax
    vmax = np.arcsinh(gmax/scale); lv = np.linspace(-vmax, vmax, 25)
    fig, ax = plt.subplots(figsize=(8, 5))

    def upd(k):
        ax.clear(); ax.tricontourf(triang, np.arcsinh(frames[k]/scale), levels=lv, cmap="seismic", extend="both")
        draw_scene(ax); ax.set_title(f"{title}\nt = {tt[k]*1e6:5.2f} us")
    FuncAnimation(fig, upd, frames=len(frames)).save(OUT/fname, writer=PillowWriter(fps=12))
    kk = int(0.85*len(frames)); fig2, ax = plt.subplots(figsize=(8, 5)); upd(kk)
    fig2.savefig(OUT/fname.replace(".gif", "_frame.png"), dpi=120); plt.close("all")


times = np.array(times)
w = times <= 20e-6
print("rendering wavefield.gif ..."); make_gif([snaps_c[i] for i in np.where(w)[0]], times[w],
                                               "wavefield.gif", "ILI wavefield u_x (corrosion ID + crack OD)")
w2 = (times >= 12e-6) & (times <= 25e-6)
print("rendering crack_echo.gif ..."); make_gif([diff[i] for i in np.where(w2)[0]], times[w2],
                                                "crack_echo.gif", "Crack echo (corrosion+crack minus corrosion-only)")
print("wrote results/ili_realistic/{geometry.png, wavefield.gif, crack_echo.gif} (+ *_frame.png, snaps.npz)")
