r"""
Stage 2c — BEAMFORMED ILI IMAGE (the k-Wave sim's end-product), from the FEniCS RF data.

PIPELINE (matches the real inspection, 0-deg default = plane-wave transmit)
  1. transmit a plane wave; record the RF at a dense receiver array (each node at the transducer plane
     = one element) at every time step -> RF(element, t)   [the "channel data"].
  2. beamform by plane-wave delay-and-sum (PWI): for each image pixel P, coherently sum every element's
     RF at the round-trip delay  tau = t_transmit(P) + t_receive(P, element).
     * t_transmit: plane wave, straight in x, layered (fluid c_f then steel c_L).
     * t_receive : pixel -> element, refracting at the ID by FERMAT'S PRINCIPLE (Snell) through the
       fluid/steel layers -> reflectors focus at the geometrically correct depth.
     Analytic-signal (Hilbert) DAS -> smooth envelope image, shown in dB.

Two images: full data (corroded ID + back wall + crack tip) and baseline-subtracted (crack isolated).
Geometry: the faithful corrosion(ID)+crack(OD) model. Physics = the validated FEM stack + dashpot ABC.

RUN (~9 min):  ./run.ps1 python3 repro/ili_beamform.py  -> results/ili_beamform/*.png
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
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
x_fw, x_bw = S, S+T; x_tip = x_bw-h_crack
Lx, Ly, y_crack = 0.0324, 0.020, 0.010
x0, sigma, x_r = 2.0e-3, 0.6e-3, 1.0e-3
deg, Nx, Ny = 2, 108, 67
cell = Lx/Nx
OUT = Path(__file__).resolve().parents[1]/"results"/"ili_beamform"; OUT.mkdir(parents=True, exist_ok=True)

_rng = np.random.default_rng(1)
_yg = np.linspace(0, Ly, 500)
_sm = np.convolve(_rng.random(500)+np.sin(2*np.pi*7*_yg/Ly)+np.sin(2*np.pi*2*_yg/Ly), np.ones(11)/11, "same")
_sm = (_sm-_sm.min())/(np.ptp(_sm)+1e-30); _corr = 1.0e-3*_sm
corr_of = lambda y: np.interp(y, _yg, _corr)
in_crack = lambda x, y: (x >= x_tip) & (x < x_bw+root_ext) & (np.abs(y-y_crack) <= w_crack/2)

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

# receiver array: every node at x = x_r (each = an element)
rmask = np.abs(nx[:, 0]-x_r) < cell/2
ridx = np.where(rmask)[0]; order = np.argsort(nx[ridx, 1]); ridx = ridx[order]
elem_y = nx[ridx, 1]

dt = 0.4*(cell/deg**2)/cP_s
t_bw = 2*(S-x0)/c_f + 2*T/cP_s
nsteps = int((t_bw+3e-6)/dt)
tgrid = np.arange(nsteps)*dt


def material(with_crack):
    xc = Q.tabulate_dof_coordinates(); x, y = xc[:, 0], xc[:, 1]
    lam = np.full(x.shape, lam_s); mu = np.full(x.shape, mu_s); rho = np.full(x.shape, rho_s)
    fluid = x < (x_fw+corr_of(y)); air = x >= x_bw
    if with_crack:
        air = air | in_crack(x, y)
    lam[fluid], mu[fluid], rho[fluid] = lam_f, 0, rho_f
    lam[air], mu[air], rho[air] = lam_o, 0, rho_o
    out = []
    for a in (lam, mu, rho):
        fn = Function(Q); fn.x.array[:] = a; out.append(fn)
    return out


def run(with_crack):
    lam_fn, mu_fn, rho_fn = material(with_crack)
    sig = lambda w: lam_fn*ufl.tr(eps(w))*ufl.Identity(2)+2*mu_fn*eps(w)
    K = assemble_matrix(form(ufl.inner(sig(u_tr), eps(v))*ufl.dx)); K.scatter_reverse(); K = K.to_scipy()
    m = assemble_vector(form(rho_fn*ufl.inner(v, ufl.as_vector((1.0, 1.0)))*ufl.dx)).array.copy()
    inv_a = 1/(m/dt**2+c_damp/(2*dt)); b = m/dt**2-c_damp/(2*dt); tm = 2*m/dt**2
    u_old = u0.copy(); u_old[cidx] = 0
    u = u_old+dt*v0-0.5*dt**2*(c_damp*v0+K@u_old)/m; u[cidx] = 0
    RF = np.empty((nsteps, ridx.size)); RF[0] = u_old[2*ridx]; RF[1] = u[2*ridx]
    for n in range(2, nsteps):
        un = inv_a*(tm*u-b*u_old-(K@u)); un[cidx] = 0
        u_old, u = u, un
        RF[n] = u[2*ridx]
    return RF


print(f"deg {deg} mesh {Nx}x{Ny} steps {nsteps}, {ridx.size} elements")
print("solve 1/2 (corrosion+crack)..."); RF_c = run(True)
print("solve 2/2 (corrosion only)..."); RF_r = run(False)


def beamform(RF):
    xi = np.linspace(18e-3, 31e-3, 150); yi = np.linspace(0, Ly, 120)
    XI, YI = np.meshgrid(xi, yi); Xf, Yf = XI.ravel(), YI.ravel()
    ttx = (np.minimum(Xf, x_fw)-x0)/c_f + np.maximum(0.0, Xf-x_fw)/cP_s   # plane wave, fluid then steel
    steel = Xf > x_fw
    yc = np.linspace(0, Ly, 90)[None, :]
    img = np.zeros(Xf.shape, dtype=complex)
    for e, ye in enumerate(elem_y):
        trx = np.empty(Xf.shape)
        trx[~steel] = np.sqrt((Xf[~steel]-x_r)**2 + (Yf[~steel]-ye)**2)/c_f
        xs = Xf[steel][:, None]; ys = Yf[steel][:, None]
        trx[steel] = (np.sqrt((xs-x_fw)**2+(ys-yc)**2)/cP_s +
                      np.sqrt((x_fw-x_r)**2+(yc-ye)**2)/c_f).min(axis=1)
        tau = ttx + trx
        a = hilbert(RF[:, e])
        img += np.interp(tau, tgrid, a.real) + 1j*np.interp(tau, tgrid, a.imag)
    return xi, yi, np.abs(img).reshape(XI.shape)


print("beamforming full data..."); xi, yi, img_full = beamform(RF_c)
print("beamforming difference (crack)..."); _, _, img_dif = beamform(RF_c - RF_r)


def overlay(ax):
    yl = np.linspace(0, Ly, 300)
    ax.plot((x_fw+corr_of(yl))*1e3, yl*1e3, "c-", lw=1.0, label="ID (corroded)")
    ax.axvline(x_bw*1e3, color="c", lw=1.0, label="OD")
    ax.add_patch(Rectangle((x_tip*1e3, (y_crack-w_crack/2)*1e3), h_crack*1e3, w_crack*1e3,
                           fill=False, edgecolor="lime", lw=1.5, label="crack"))
    ax.set_xlabel("x depth [mm]"); ax.set_ylabel("y [mm]"); ax.set_aspect("equal")


def show(xi, yi, img, fname, title):
    db = 20*np.log10(img/img.max()+1e-9)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(db, extent=[xi[0]*1e3, xi[-1]*1e3, yi[0]*1e3, yi[-1]*1e3], origin="lower",
                   aspect="equal", cmap="inferno", vmin=-30, vmax=0)
    overlay(ax); ax.legend(loc="upper left", fontsize=7)
    fig.colorbar(im, ax=ax, label="dB"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(OUT/fname, dpi=130); plt.close(fig)


show(xi, yi, img_full, "beamformed_full.png", "Beamformed ILI image (corrosion + crack) [dB]")
show(xi, yi, img_dif, "beamformed_crack.png", "Beamformed crack image (baseline-subtracted) [dB]")
np.savez_compressed(OUT/"rf.npz", RF_c=RF_c, RF_r=RF_r, tgrid=tgrid, elem_y=elem_y)
print("wrote results/ili_beamform/{beamformed_full, beamformed_crack}.png + rf.npz")
