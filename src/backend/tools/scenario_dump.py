r"""Print the frozen scenario constants as JSON. READ-ONLY: imports, never computes.

The GUI needs the geometry, array, materials and mesh-size numbers to draw its cross-section
and its consequences panel. Those numbers already exist in three places - mesh/ili_mesh.py,
repro/ili_forward.py and lib/tt_t_image.py's FROZEN - so this dumps them instead of letting
the GUI become a fourth copy that can drift out of step. Lengths in mm, everything else SI.

    docker/run.ps1 python3 tools/scenario_dump.py
"""
import glob
import json
import sys

# docker/run.ps1 exports PYTHONPATH=/work, which REPLACES the image's own PYTHONPATH - and
# that is where dolfinx and gmsh live (they are not in site-packages). Without this, importing
# the very modules this script exists to read fails with ModuleNotFoundError. Appending, not
# prepending, so /work still wins for our own packages.
sys.path += glob.glob("/usr/local/dolfinx-real/lib/python3.*/dist-packages") + ["/usr/local/lib"]

from mesh import ili_mesh as g          # geometry + mesh size targets (mm)
from repro import ili_forward as s      # materials, source, time base (SI)
from lib.tt_t_image import FROZEN       # what the imaging chain assumes (SI)

print(json.dumps({
    "geometry": {"r_id": g.R_ID, "r_od": g.R_OD, "wall": g.R_OD - g.R_ID,
                 "x_c": g.X_C, "z_c": g.Z_C, "standoff": g.STANDOFF},
    "array": {"n_elem": g.N_ELEM, "pitch": g.PITCH, "kerf": s.KERF * 1e3,
              "aperture": g.ARRAY_X1 - g.ARRAY_X0, "x0": g.ARRAY_X0, "f0": s.F0,
              "n_cycle": s.N_CYCLE},
    "materials": {"c_P": s.C_P, "c_S": s.C_S, "rho_s": s.RHO_S,
                  "c_f": s.C_F, "rho_f": s.RHO_F},
    "notch": {"x": g.NOTCH_X, "depth": g.NOTCH_DEPTH, "width": g.NOTCH_WIDTH},
    "mesh_h": {"water": g.H_WATER, "steel": g.H_STEEL, "notch": g.H_NOTCH,
               "array": g.H_ARRAY, "stair": g.STAIR_H},
    "domain": {"x_min": g.X_MIN, "x_max": g.X_MAX,
               "x_limit_lo": g.X_LIMIT_LO, "x_limit_hi": g.X_LIMIT_HI},
    "time": {"t_end": s.T_END, "cfl": s.CFL, "dt_kwave": s.DT_KWAVE,
             "n_samp_kwave": s.N_SAMP_KWAVE},
    # Cross-check, not decoration: the imaging chain and the mesh must agree on the pipe, or
    # every measured depth is against the wrong reference.
    "frozen_agrees": (abs(FROZEN["r_id"] * 1e3 - g.R_ID) < 1e-9
                      and abs(FROZEN["notch_depth"] * 1e3 - g.NOTCH_DEPTH) < 1e-9
                      and FROZEN["n_elem"] == g.N_ELEM and FROZEN["f0"] == s.F0),
}, indent=2))
