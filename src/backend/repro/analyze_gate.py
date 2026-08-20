r"""
Crack-tip timing from the saved ILI-gate B-scans (no re-solve — loads results/ili_gate/bscans.npz).

The crack is a sub-wavelength (0.5 mm) notch, so at normal incidence its tip echo is WEAK but its
TIMING is exact. In the baseline-subtracted trace at y=8 mm it is the echo BETWEEN the front wall and
the (stronger) 'removed back-wall' feature. We pick it as the strongest |diff| in the window
[front wall+1us, back wall-0.9us] (i.e. before the back-wall feature) and compare to analytic ToF.

RUN:  ./run.ps1 python3 repro/analyze_gate.py
"""
from pathlib import Path
import numpy as np
from lib.paths import RESULTS

d = np.load(RESULTS / "ili_gate" / "bscans.npz")
Bc, Br, tgrid, ry = d["Bc"], d["Br"], d["tgrid"], d["ry"]
t_fw, t_crack, t_bw = float(d["t_fw"]), float(d["t_crack"]), float(d["t_bw"])
diff = Bc - Br
jc = int(np.argmin(np.abs(ry - ry.mean())))                     # y = 8 mm (crack center)

# crack-tip echo: between the front wall and the back-wall feature
wc = (tgrid > t_fw + 1e-6) & (tgrid < t_bw - 0.9e-6)
i_ck = np.where(wc)[0][np.argmax(np.abs(diff[wc, jc]))]
t_ck, a_ck = tgrid[i_ck], np.abs(diff[i_ck, jc])
# back-wall-removal feature (for an honest amplitude comparison)
wb = (tgrid > t_bw - 0.6e-6) & (tgrid < t_bw + 0.6e-6)
a_bw = np.max(np.abs(diff[wb, jc]))

print(f"analytic:  FW={t_fw*1e6:.3f}  crack={t_crack*1e6:.3f}  BW={t_bw*1e6:.3f} us")
print(f"CRACK-TIP echo (y=8mm, baseline-subtracted):")
print(f"   measured {t_ck*1e6:.3f} us   analytic {t_crack*1e6:.3f} us   err {abs(t_ck-t_crack)/t_crack*100:.2f}%")
print(f"   crack-earlier-than-BW: measured {(t_bw-t_ck)*1e6:.3f} us  (analytic {(t_bw-t_crack)*1e6:.3f})")
print(f"   amplitude: crack echo is {a_ck/a_bw*100:.0f}% of the back-wall feature "
      f"(weak -> sub-wavelength notch at normal incidence; motivates angled/mode-converted beams)")
