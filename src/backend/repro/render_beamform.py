r"""
Re-beamform the ILI image from saved RF (results/ili_beamform/rf.npz) — NO re-solve.
Adds a TIME GATE that removes the dominant front-wall echo so the back-wall + crack region images
cleanly (standard NDT practice), and focuses the image on the wall region of interest.

RUN:  ./run.ps1 python3 repro/render_beamform.py
"""
from pathlib import Path
import numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from lib.paths import RESULTS

OUT = RESULTS/"ili_beamform"
d = np.load(OUT/"rf.npz")
RF_c, RF_r, tgrid, elem_y = d["RF_c"], d["RF_r"], d["tgrid"], d["elem_y"]

c_f, cP_s = 1500.0, 5700.0
S, T = 20.0e-3, 9.525e-3
x_fw, x_bw = S, S+T
x0, x_r = 2.0e-3, 1.0e-3
h_crack, w_crack, root_ext, y_crack, Ly = 3.0e-3, 0.5e-3, 0.3e-3, 0.010, 0.020
x_tip = x_bw-h_crack
_rng = np.random.default_rng(1); _yg = np.linspace(0, Ly, 500)
_sm = np.convolve(_rng.random(500)+np.sin(2*np.pi*7*_yg/Ly)+np.sin(2*np.pi*2*_yg/Ly), np.ones(11)/11, "same")
_sm = (_sm-_sm.min())/(np.ptp(_sm)+1e-30); corr_of = lambda y: np.interp(y, _yg, 1.0e-3*_sm)

t_fw = (x_fw-x0)/c_f + (x_fw-x_r)/c_f              # front-wall echo time
gate = 0.5*(1+np.tanh((tgrid-(t_fw+0.9e-6))/0.3e-6))   # smooth gate: keep t > FW echo


def beamform(RF, xlo=24e-3, xhi=31e-3):
    xi = np.linspace(xlo, xhi, 130); yi = np.linspace(0, Ly, 120)
    XI, YI = np.meshgrid(xi, yi); Xf, Yf = XI.ravel(), YI.ravel()
    ttx = (np.minimum(Xf, x_fw)-x0)/c_f + np.maximum(0.0, Xf-x_fw)/cP_s
    steel = Xf > x_fw; yc = np.linspace(0, Ly, 90)[None, :]
    img = np.zeros(Xf.shape, complex)
    for e, ye in enumerate(elem_y):
        trx = np.empty(Xf.shape)
        trx[~steel] = np.sqrt((Xf[~steel]-x_r)**2+(Yf[~steel]-ye)**2)/c_f
        xs = Xf[steel][:, None]; ys = Yf[steel][:, None]
        trx[steel] = (np.sqrt((xs-x_fw)**2+(ys-yc)**2)/cP_s+np.sqrt((x_fw-x_r)**2+(yc-ye)**2)/c_f).min(1)
        tau = ttx+trx; a = hilbert(RF[:, e])
        img += np.interp(tau, tgrid, a.real)+1j*np.interp(tau, tgrid, a.imag)
    return xi, yi, np.abs(img).reshape(XI.shape)


def overlay(ax):
    yl = np.linspace(0, Ly, 300)
    ax.plot((x_fw+corr_of(yl))*1e3, yl*1e3, "c-", lw=1.0, label="ID (corroded)")
    ax.axvline(x_bw*1e3, color="c", lw=1.0, label="OD")
    ax.add_patch(Rectangle((x_tip*1e3, (y_crack-w_crack/2)*1e3), h_crack*1e3, w_crack*1e3,
                           fill=False, edgecolor="lime", lw=1.6, label="true crack"))
    ax.set_xlabel("x depth [mm]"); ax.set_ylabel("y [mm]"); ax.set_aspect("equal")


def show(xi, yi, img, fname, title):
    db = 20*np.log10(img/img.max()+1e-9)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(db, extent=[xi[0]*1e3, xi[-1]*1e3, yi[0]*1e3, yi[-1]*1e3], origin="lower",
                   aspect="equal", cmap="inferno", vmin=-25, vmax=0)
    overlay(ax); ax.legend(loc="upper left", fontsize=7); fig.colorbar(im, ax=ax, label="dB")
    ax.set_title(title); fig.tight_layout(); fig.savefig(OUT/fname, dpi=140); plt.close(fig)


xi, yi, img_gated = beamform(RF_c*gate[:, None])
show(xi, yi, img_gated, "beamformed_gated.png", "Beamformed image, front-wall gated out (back wall + crack) [dB]")
xi, yi, img_crack = beamform((RF_c-RF_r))
show(xi, yi, img_crack, "beamformed_crack.png", "Beamformed crack image (baseline-subtracted) [dB]")
print("wrote results/ili_beamform/{beamformed_gated, beamformed_crack}.png")
