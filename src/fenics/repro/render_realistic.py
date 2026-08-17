r"""
Re-render the faithful ILI animation from saved snapshots (results/ili_realistic/snaps.npz) — NO
re-solve. Focuses each GIF on its clean, informative time window and uses arcsinh compression.

  wavefield.gif  : t <= 20 us  -- pulse crosses the fluid, reflects off the ID, transmits through the
                   steel, and reaches the OD/crack (before the back-wall multiples pile up).
  crack_echo.gif : 12-25 us     -- with-crack minus baseline; the crack lights up as the wave hits it.

RUN:  ./run.ps1 python3 repro/render_realistic.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[1]/"results"/"ili_realistic"
d = np.load(OUT/"snaps.npz")
Sc, Sr, times, coords = d["snaps_c"], d["snaps_r"], d["times"], d["coords"]
diff = Sc - Sr
triang = mtri.Triangulation(coords[:, 0]*1e3, coords[:, 1]*1e3)

# geometry (mm) for overlays
S, T, h_crack, w_crack, root_ext = 20.0, 9.525, 3.0, 0.5, 0.3
x_fw, x_bw = S, S+T; x_tip = x_bw-h_crack
Lx, Ly, y_crack = 32.4, 20.0, 10.0
corr_y = d["corr_y"]*1e3; corr = d["corr"]*1e3               # corroded-ID profile (mm)
_yl = np.linspace(0, Ly, 400)
_idprofile = x_fw + np.interp(_yl, corr_y, corr)


def draw_scene(ax):
    ax.plot(_idprofile, _yl, color="k", lw=1.0); ax.axvline(x_bw, color="k", lw=1.0)
    ax.add_patch(Rectangle((x_tip, y_crack-w_crack/2), (x_bw-x_tip+root_ext), w_crack,
                           facecolor="black", edgecolor="black", zorder=5))
    ax.annotate("crack (backwall/OD)", (x_bw, y_crack+w_crack+0.6), (x_bw-9, Ly*0.82),
                fontsize=8, arrowprops=dict(arrowstyle="->"))
    ax.text(S*0.5, Ly*0.05, "FLUID", ha="center", fontsize=8)
    ax.text(S+T/2, Ly*0.05, "STEEL", ha="center", fontsize=8)
    ax.text(0.6, Ly*0.5, "transducer\n(inside pipe)", rotation=90, va="center", fontsize=7)
    ax.text(x_fw, Ly*1.01, "ID (front wall)", ha="center", fontsize=7)
    ax.text(x_bw, Ly*1.01, "OD (back wall)", ha="center", fontsize=7)
    ax.set_aspect("equal"); ax.set_xlabel("x depth [mm]"); ax.set_ylabel("y [mm]")
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)


def render(frames, tt, fname, title, scale_frac):
    gmax = np.abs(frames).max() + 1e-30
    scale = scale_frac*gmax; vmax = np.arcsinh(gmax/scale); lv = np.linspace(-vmax, vmax, 25)
    fig, ax = plt.subplots(figsize=(8, 5))

    def upd(k):
        ax.clear(); ax.tricontourf(triang, np.arcsinh(frames[k]/scale), levels=lv, cmap="seismic", extend="both")
        draw_scene(ax); ax.set_title(f"{title}\nt = {tt[k]*1e6:5.2f} us")
    FuncAnimation(fig, upd, frames=len(frames)).save(OUT/fname, writer=PillowWriter(fps=12))
    plt.close(fig)


def still(frame, t, fname, title, scale_frac):
    gmax = np.abs(frame).max()+1e-30; scale = scale_frac*gmax; vmax = np.arcsinh(gmax/scale)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.tricontourf(triang, np.arcsinh(frame/scale), levels=np.linspace(-vmax, vmax, 25),
                   cmap="seismic", extend="both")
    draw_scene(ax); ax.set_title(f"{title}\nt = {t*1e6:.2f} us")
    fig.savefig(OUT/fname, dpi=120); plt.close(fig)


# stills to judge the clean phase
for tt in (6e-6, 10e-6, 13e-6, 16e-6, 19e-6):
    k = int(np.argmin(np.abs(times-tt)))
    still(Sc[k], times[k], f"still_{int(tt*1e6)}us.png", "ILI wavefield u_x (with backwall crack)", 0.08)

wsel = times <= 20e-6
render(Sc[wsel], times[wsel], "wavefield.gif", "ILI wavefield u_x (with backwall crack)", 0.08)
csel = (times >= 12e-6) & (times <= 25e-6)
render(diff[csel], times[csel], "crack_echo.gif", "Crack echo (with-crack minus baseline)", 0.06)
print("re-rendered wavefield.gif, crack_echo.gif + still_*.png")
