r"""
Conforming 2-D finite-element mesh for the ILI ultrasound simulation (gmsh -> DOLFINx).

WHY THIS FILE EXISTS
====================
Every previous script in this repo (repro/ili_*.py) discretised the pipe on a STRUCTURED
RECTANGULAR grid, exactly like the k-Wave reference: the curved ID/OD surfaces became a
staircase of grid cells, and the notch became a block of "air" voxels. A staircased curved
interface is not a small cosmetic error for a specular pulse-echo problem:

  * the local surface NORMAL of a staircase is always axis-aligned (0 or 90 deg), never the
    true radial direction, so the reflected/refracted ray directions are wrong by up to
    45 deg locally, and mode conversion at the interface is driven by the wrong angle;
  * the step edges are point scatterers that radiate spurious diffracted energy at every
    riser (grid-scale "surface roughness" of amplitude ~dx/2);
  * the arrival time of the backwall echo has a systematic bias of order dx/2 / c.

This mesh removes that error class: the ID and OD surfaces are EXACT circular arcs of the
CAD geometry, the mesh conforms to them (element edges lie ON the arc, nodes lie exactly on
it), and the notch is cut as real geometry so its faces and its two sharp tip corners are
genuine mesh boundaries. Verification #5 below is the quantitative proof (max radial
deviation of the meshed arcs, in microns, versus ~dx/2 = hundreds of microns for a voxel grid).

GEOMETRY (all dimensions mm in the parameter block; the mesh is written in METRES, SI,
matching the solver scripts in repro/)
====================================================================================
Transducer array face is the straight line z = 0, x in [0, 76.5] (256 elements at 0.30 mm
pitch, centres at x = i*0.30). The array looks radially OUTWARD (+z) at the pipe wall.

The pipe centre is placed at (x_c, z_c) = (38.25, -173.675) mm, i.e. directly "below" the
centre of the array. Consequence, and the identity this whole coordinate choice exists to
satisfy (asserted at run time):

        R_ID + z_c = 193.675 - 173.675 = 20.000 mm = the water standoff

so on the beam axis (x = 38.25) the ID surface is exactly 20 mm from the array face, the OD
is at z = R_OD + z_c = 29.525 mm, and the wall thickness on axis is 9.525 mm.

  fluid (water)  : z = 0 (array plane)  ->  ID arc, r = R_ID = 193.675 mm
  steel          : ID arc               ->  OD arc, r = R_OD = 203.200 mm, MINUS the notch
  notch (VOID)   : radial slot cut into the OD on the beam axis, 4.0 mm deep, 1.0 mm wide

The notch is a parallel-sided slot (an EDM/ machined notch, matching the k-Wave reference's
rotated-rectangle crack), whose axis is the radial direction at angle 0. Because angle 0 is
the beam axis and the pipe centre lies on it, "radial" there is exactly +z, so the two side
walls are the vertical lines x = 38.25 -+ 0.5 mm and the tip face is the horizontal line
z = z_c + R_OD - 4.0 = 25.525 mm. Depth measured radially on the axis is therefore exactly
4.000 mm; at the two tip CORNERS (off-axis by 0.5 mm) the radial depth is 3.9994 mm, a
0.6 micron consequence of the tip being flat rather than an arc. Verification #6 measures
this from the mesh, not from the CAD input.

WHAT IS *NOT* MESHED (deliberate)
=================================
The notch interior, and everything radially beyond the OD, are simply absent from the mesh.
Their faces then satisfy the natural boundary condition of the elastodynamic weak form,
which is exactly traction-free = a free surface against air/vacuum. That is physically
correct for steel-air (impedance ratio ~1e-5, reflection coefficient -0.99997) and strictly
better than the k-Wave reference, which had to fill air with a fictitious 500 m/s, 500 kg/m3
material because a finite-difference grid cannot leave a hole. So the meshed z-extent stops
at the OD (max z = 29.525 mm on axis) even though the nominal drawing extent goes to ~36 mm;
adding an unmeshed 5 mm "outside" band would be geometry with no elements in it.

PHYSICAL TAGS - THE SOLVER DEPENDS ON THESE NUMBERS
===================================================
Surfaces (dim 2, cell tags):
    1   FLUID   water, between the array plane z = 0 and the ID arc
    2   STEEL   pipe wall, between the ID and OD arcs, minus the notch
Curves (dim 1, facet tags):
    10  ARRAY   transducer face, z = 0, x in [0, 76.5]  (source / receive aperture ONLY;
                the z = 0 line outboard of the aperture is tag 14, not 10, so the solver
                gets the exact 76.5 mm aperture without any coordinate filtering)
    11  ID      inner pipe surface, r = R_ID. INTERNAL interface: it bounds both surfaces
                (fluid on one side, steel on the other) -> fluid-solid coupling facets
    12  OD      outer pipe surface, r = R_OD (two curves, split by the notch mouth)
    13  NOTCH   the three notch walls (right wall, flat tip face, left wall) - free surface
    14  ABC     everything else: the two side walls x = -8 and x = +85 (fluid and steel
                parts) and the z = 0 line outboard of the array aperture. Absorbing BC.

RESOLUTION
==========
At 4 MHz: water 0.375 mm, steel shear 0.775 mm, steel P 1.425 mm. The solver uses degree-3
GLL spectral elements -> ~4 nodes per shortest wavelength means cell size ~ 3*(0.375/4) =
0.28 mm. WATER has the SHORTEST wavelength, so the fluid must be the finest region; steel
may be ~2x coarser (its shear wavelength is ~2x the water one). The notch and its tip
corners get ~0.09 mm because that is the scatterer whose diffraction we are trying to
resolve. All three are parameters below; --scale multiplies them (>1 coarser and faster,
<1 finer for convergence studies).

RUN
===
  MSYS_NO_PATHCONV=1 docker run --rm -v "C:/code/DVCode/Fea/research/fenics:/work" \
      -w /work dvfenics:bf python3 mesh/ili_mesh.py [--scale 1.0] [--quad]

Outputs -> results/ili_mesh/ : ili_mesh.msh, ili_mesh.xdmf/.h5, ili_mesh.png

The solver can load EITHER artifact:
    from dolfinx.io import gmsh as dgmsh                      # note: 0.11 moved gmshio here
    md = dgmsh.read_from_msh("results/ili_mesh/ili_mesh.msh", MPI.COMM_WORLD, gdim=2)
    mesh, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
or the XDMF, whose grid names are "mesh", "cell_tags", "facet_tags".
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

import gmsh
from lib.paths import RESULTS

# ----------------------------------------------------------------------------------------
# GEOMETRY PARAMETERS (mm - exact drawing numbers, never rounded)
# ----------------------------------------------------------------------------------------
MM = 1.0e-3                     # mm -> m; the mesh is written in metres (SI, like the solvers)

R_ID = 193.675                  # pipe inner radius
R_OD = 203.200                  # pipe outer radius  (wall = 9.525 mm)
X_C, Z_C = 38.25, -173.675      # pipe centre: chosen so R_ID + Z_C == 20.0 mm standoff
STANDOFF = 20.0                 # water standoff on the beam axis

ARRAY_X0, ARRAY_X1 = 0.0, 76.5  # transducer aperture at z = 0
N_ELEM, PITCH = 256, 0.30       # 256 elements, centres at x = i*PITCH (255*0.30 = 76.5)

X_MIN, X_MAX = -8.0, 85.0       # lateral domain extent (--x-min / --x-max override)
# HARD GEOMETRIC CEILING on widening. Water occupies z = 0 (flat array plane) up to the ID
# arc, so the domain is only valid while the arc stays BELOW z = 0. The arc reaches z = 0
# where |x - X_C| = sqrt(R_ID^2 - Z_C^2), which for this pipe is x = -47.46 and +123.96 mm.
# Max usable width is therefore 171.4 mm against the default 93.0 - a factor of 1.84, no
# more. Widening past that is not a flag change: it needs the water region rebuilt to follow
# the arc at constant standoff.
X_LIMIT_LO = X_C - math.sqrt(R_ID ** 2 - Z_C ** 2)
X_LIMIT_HI = X_C + math.sqrt(R_ID ** 2 - Z_C ** 2)

NOTCH_DEPTH = 4.0               # radially inward from the OD
NOTCH_WIDTH = 1.0               # slot width (parallel sided)
NOTCH_X = 38.25                 # centred on the beam axis
INCLUDE_NOTCH = True            # --no-notch builds the HEALTHY wall (clutter-floor baseline)

# ----------------------------------------------------------------------------------------
# TARGET CELL SIZES (mm, before --scale).  Water is finest: shortest wavelength.
# ----------------------------------------------------------------------------------------
H_WATER = 0.28                  # 3 * (0.375 / 4): ~4 nodes per water wavelength at p=3
H_STEEL = 0.56                  # 2x: steel shear wavelength is ~2x the water one
#
# H_NOTCH: BEWARE - this is a CFL trap, not a free accuracy knob. An explicit solver has
# ONE global time step set by the SMALLEST cell (dt ~ C h / (c p^2)), and the notch sits in
# steel where c is highest. Refining the tip to 0.09 mm drives dt to 0.34 ns => ~178k steps
# to 60 us, versus ~3.4 ns / 18k steps for 0.56 mm steel cells: a 10x runtime penalty.
# It buys nothing, because (a) degree-3 elements at 0.4 mm already give 0.13 mm node
# spacing against a 0.775 mm shear wavelength, and (b) the notch FACES are exact geometry
# whatever the cell size - conformity comes from the boundary, not from refinement.
# Default is therefore coarse enough to keep dt sane; override with --h-notch if a
# convergence study needs it.
H_NOTCH = 0.30                  # at the notch walls / tip corners (the scatterer)
# Transducer plane: must resolve INDIVIDUAL elements (0.30 mm pitch, 0.25 mm active
# width), not just the wavelength. ~3 facets per element. Free in CFL terms - see the
# note in size_callback.
H_ARRAY = 0.09                  # cell size at the array face (z = 0)
ARRAY_BAND = 0.60               # ramp back up to H_WATER over this depth (mm)
NOTCH_REFINE_R0 = 0.30          # keep H_NOTCH within this distance of the notch (mm)
NOTCH_REFINE_R1 = 3.00          # ... blend back to H_STEEL by here (mm)
ID_GRADE_LEN = 1.50             # grade steel from H_WATER at the ID to H_STEEL over this (mm)

# --- the C4 controlled experiment: --staircase -------------------------------------------
# Replaces the exact ID arc with the PIXEL STAIRCASE a Cartesian solver is forced to use, at
# k-Wave's own 50 um grid spacing. Everything else - sizing fields, grading, algorithm,
# element order, source, solver - is untouched, so a difference in the resulting image is
# attributable to the geometry representation and nothing else. This is what turns "we win on
# geometry" into "we win BECAUSE the mesh conforms".
STAIRCASE = False
STAIR_PIXEL = 0.05              # k-Wave's dx = dy = 50 um
STAIR_H = 0.05                  # cell size in the staircase band. Costs cells but NOT dt:
                                # dt is already set by the 50 um vertical risers, which are
                                # forced geometry whatever the size field says.
STAIR_BAND = 0.25               # ... graded back to H_WATER over this distance (mm)

# Physical tags (documented in the module docstring - the solver hard-codes these).
TAG_FLUID, TAG_STEEL = 1, 2
TAG_FILL = 3                    # --notch-fill only: the notch meshed as k-Wave's "outside"
TAG_ARRAY, TAG_ID, TAG_OD, TAG_NOTCH, TAG_ABC = 10, 11, 12, 13, 14

# --- C5: --notch-fill ---------------------------------------------------------------------
# k-Wave cannot leave a hole in a finite-difference grid, so it fills the crack void with its
# "outside" material and calls that air. From their driver script, verbatim:
#     sound_speed_compression = 500,  sound_speed_shear = 0,  density = 500
# i.e. an ACOUSTIC (mu = 0) material, which our fluid machinery already handles. This variant
# meshes the notch as a third region so we can run the identical geometry with the void either
# filled that way or left traction-free, and see whether the substitution is what costs them.
NOTCH_FILL = False

OUT = RESULTS / "ili_mesh"


def z_arc(x_mm: float, radius: float) -> float:
    """z of the upper (array-facing) branch of the circle of given radius, at abscissa x."""
    dx = x_mm - X_C
    return Z_C + math.sqrt(radius * radius - dx * dx)


def staircase_vertices(radius: float, x_from: float, x_to: float, q: float):
    """Vertices of the PIXEL-STAIRCASE approximation of an arc, walking x_from -> x_to (mm).

    This is what a Cartesian grid does to a curved interface: every pixel column of width q
    is wholly one material, so the interface can only ever run along pixel EDGES. Quantising
    z to the nearest pixel edge and emitting a horizontal run + a vertical riser at each
    change reproduces exactly that shape - here at k-Wave's own q = 50 um.

    Returns [(x, z), ...] in mm, starting on the x_from column and ending on the x_to column.
    The arc is monotone either side of the apex, but nothing here assumes that.
    """
    step = -q if x_to < x_from else q
    xs = np.arange(x_from, x_to + 0.5 * step, step)
    lvl = np.round(np.array([z_arc(float(x), radius) for x in xs]) / q) * q
    verts = [(float(xs[0]), float(lvl[0]))]
    for x, z in zip(xs[1:], lvl[1:]):
        if z != verts[-1][1]:
            verts.append((float(x), verts[-1][1]))      # end of the horizontal run
            verts.append((float(x), float(z)))          # the vertical riser
    if verts[-1][0] != float(xs[-1]):
        verts.append((float(xs[-1]), verts[-1][1]))     # close out the final run
    # Drop the two end columns. A riser sitting exactly on x = X_MIN/X_MAX is COLLINEAR with
    # the side wall, so it separates nothing and bounds only one surface - which trips the
    # conformity assertion rather than producing a usable mesh. The caller instead joins the
    # exact arc endpoint to the first interior vertex, a single sub-pixel (<= q) diagonal at
    # the absorbing boundary, far from the aperture.
    # The margin is a tolerance, not decoration: np.arange's last sample lands within a
    # float epsilon of the boundary, so an exact `lo < x < hi` test keeps a vertex ~1e-11 mm
    # inside and the closing segment comes out shorter than OCC's tolerance. OCC then quietly
    # drops it, and the curve bounds one surface instead of two.
    lo, hi = min(x_from, x_to), max(x_from, x_to)
    tol = 0.25 * q
    return [(x, z) for x, z in verts if lo + tol < x < hi - tol]


# ========================================================================================
# 1. GEOMETRY
# ========================================================================================
def build_geometry() -> dict:
    """
    Build the two regions from EXPLICIT curve loops rather than boolean operations.

    This matters. The obvious route (disk_OD - disk_ID, intersect with a box, cut the notch,
    then `fragment` the two regions together) works, but it leaves the fluid/steel interface
    at the mercy of OCC's boolean tolerance: fragment may or may not return a single shared
    edge, and if it returns two coincident-but-distinct edges the mesh is silently
    NON-CONFORMING at the ID - the exact defect this mesh exists to avoid, and one that is
    invisible in a picture.

    Building the loops by hand means the ID arc is ONE curve entity referenced by BOTH
    surface loops. Conformity at the fluid-solid interface is then structural: gmsh cannot
    produce mismatched nodes on an edge that only exists once. (Verification #5 asserts the
    curve really does bound 2 surfaces, so this claim is checked, not assumed.)

    The notch is likewise built INTO the steel loop (OD arc -> down the right wall -> across
    the tip -> up the left wall -> OD arc) instead of being cut out afterwards, so its two
    sharp tip corners are exact geometric vertices and therefore exact mesh nodes.
    """
    occ = gmsh.model.occ

    def pt(x_mm: float, z_mm: float) -> int:
        return occ.addPoint(x_mm * MM, z_mm * MM, 0.0)

    # --- points --------------------------------------------------------------------------
    p_centre = pt(X_C, Z_C)                                  # pipe axis (arc centre)

    p_bot_l = pt(X_MIN, 0.0)                                 # z = 0 line, left of aperture
    p_arr_0 = pt(ARRAY_X0, 0.0)                              # aperture start
    p_arr_1 = pt(ARRAY_X1, 0.0)                              # aperture end
    p_bot_r = pt(X_MAX, 0.0)

    p_id_l = pt(X_MIN, z_arc(X_MIN, R_ID))                   # ID arc ends
    p_id_r = pt(X_MAX, z_arc(X_MAX, R_ID))
    p_od_l = pt(X_MIN, z_arc(X_MIN, R_OD))                   # OD arc ends
    p_od_r = pt(X_MAX, z_arc(X_MAX, R_OD))

    # ponytail: the notch is hard-wired at angle 0, where "radial" == +z, so its walls are
    # plain vertical lines. For a notch at a non-zero circumferential angle, rotate these
    # four points about (X_C, Z_C) by that angle - the loop structure is unchanged.
    xn_l, xn_r = NOTCH_X - NOTCH_WIDTH / 2, NOTCH_X + NOTCH_WIDTH / 2
    z_tip = Z_C + R_OD - NOTCH_DEPTH                         # flat tip face, 25.525 mm
    p_tip_l, p_tip_r = pt(xn_l, z_tip), pt(xn_r, z_tip)      # the two sharp tip corners
    p_mouth_l = pt(xn_l, z_arc(xn_l, R_OD))                  # notch mouth on the OD
    p_mouth_r = pt(xn_r, z_arc(xn_r, R_OD))

    # --- curves --------------------------------------------------------------------------
    # addCircleArc(start, centre, end) -> a TRUE arc of the exact circle. gmsh meshes it by
    # placing nodes on the circle itself (not on a chord), which is why node radii come out
    # exact to machine precision in verification #5.
    if STAIRCASE:
        # C4: the ID as a 50 um pixel staircase instead of an exact arc. Still ONE set of
        # curve entities shared by both surface loops, so the MESH stays conforming (no
        # hanging nodes) - the only thing that changes is the SHAPE of the interface. That is
        # the single variable under test.
        sv = staircase_vertices(R_ID, X_MAX, X_MIN, STAIR_PIXEL)
        chain = [p_id_r] + [pt(vx, vz) for vx, vz in sv] + [p_id_l]
        id_curves = [occ.addLine(a, b) for a, b in zip(chain[:-1], chain[1:])]
        print(f"    STAIRCASE ID: {len(sv)} staircase vertices, {len(id_curves)} segments "
              f"at q = {STAIR_PIXEL * 1e3:.0f} um (k-Wave's grid spacing)")
    else:
        id_curves = [occ.addCircleArc(p_id_r, p_centre, p_id_l)]
    if INCLUDE_NOTCH:
        c_od_r = occ.addCircleArc(p_od_r, p_centre, p_mouth_r)   # OD, right of notch mouth
        c_od_l = occ.addCircleArc(p_mouth_l, p_centre, p_od_l)   # OD, left of notch mouth
    else:
        # HEALTHY wall: one unbroken OD arc. This variant exists for the clutter-floor
        # comparison - a defect-free wall must image as black, so whatever it does image
        # is numerical, and that is the headline "better than k-Wave" measurement. Keeping
        # it in the same script guarantees the healthy and cracked meshes differ ONLY by
        # the notch: same sizing fields, same grading, same algorithm.
        c_od_r = occ.addCircleArc(p_od_r, p_centre, p_od_l)
        c_od_l = None

    l_bot_l = occ.addLine(p_bot_l, p_arr_0)                  # z = 0, outboard (ABC)
    l_array = occ.addLine(p_arr_0, p_arr_1)                  # z = 0, the 76.5 mm aperture
    l_bot_r = occ.addLine(p_arr_1, p_bot_r)                  # z = 0, outboard (ABC)
    l_fluid_r = occ.addLine(p_bot_r, p_id_r)                 # fluid side walls
    l_fluid_l = occ.addLine(p_id_l, p_bot_l)
    l_steel_r = occ.addLine(p_id_r, p_od_r)                  # steel side walls
    l_steel_l = occ.addLine(p_od_l, p_id_l)
    if INCLUDE_NOTCH:
        l_notch_r = occ.addLine(p_mouth_r, p_tip_r)          # notch: right wall
        l_notch_t = occ.addLine(p_tip_r, p_tip_l)            # notch: flat tip face
        l_notch_l = occ.addLine(p_tip_l, p_mouth_l)          # notch: left wall
        notch_curves = [l_notch_r, l_notch_t, l_notch_l]
        od_curves = [c_od_r, c_od_l]
        if NOTCH_FILL:
            # C5: mesh the notch interior so it can be given k-Wave's "outside" material.
            # The arc across the notch mouth is the fill region's outer boundary and stays
            # traction-free, since air still lies beyond the OD in both arms - the ONLY
            # variable under test is what sits inside the notch.
            c_od_notch = occ.addCircleArc(p_mouth_r, p_centre, p_mouth_l)
            od_curves.append(c_od_notch)
        steel_loop = [*id_curves, l_steel_r, c_od_r, l_notch_r, l_notch_t,
                      l_notch_l, c_od_l, l_steel_l]
    else:
        notch_curves = []
        od_curves = [c_od_r]
        steel_loop = [*id_curves, l_steel_r, c_od_r, l_steel_l]

    # --- surfaces ------------------------------------------------------------------------
    # id_curves appear in BOTH loops == the shared, conforming fluid/solid interface.
    loop_fluid = occ.addCurveLoop([l_bot_l, l_array, l_bot_r, l_fluid_r, *id_curves,
                                   l_fluid_l])
    loop_steel = occ.addCurveLoop(steel_loop)
    s_fluid = occ.addPlaneSurface([loop_fluid])
    s_steel = occ.addPlaneSurface([loop_steel])
    s_fill = None
    if INCLUDE_NOTCH and NOTCH_FILL:
        # Same three notch curves as the steel loop, so the steel/fill interface is a single
        # shared set of entities and the mesh is conforming across it by construction.
        loop_fill = occ.addCurveLoop([l_notch_r, l_notch_t, l_notch_l, c_od_notch])
        s_fill = occ.addPlaneSurface([loop_fill])
    occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [s_fluid], TAG_FLUID); gmsh.model.setPhysicalName(2, TAG_FLUID, "fluid")
    gmsh.model.addPhysicalGroup(2, [s_steel], TAG_STEEL); gmsh.model.setPhysicalName(2, TAG_STEEL, "steel")
    if s_fill is not None:
        gmsh.model.addPhysicalGroup(2, [s_fill], TAG_FILL)
        gmsh.model.setPhysicalName(2, TAG_FILL, "notch_fill")
    gmsh.model.addPhysicalGroup(1, [l_array], TAG_ARRAY); gmsh.model.setPhysicalName(1, TAG_ARRAY, "array")
    gmsh.model.addPhysicalGroup(1, id_curves, TAG_ID); gmsh.model.setPhysicalName(1, TAG_ID, "id_arc")
    gmsh.model.addPhysicalGroup(1, od_curves, TAG_OD); gmsh.model.setPhysicalName(1, TAG_OD, "od_arc")
    if notch_curves:
        gmsh.model.addPhysicalGroup(1, notch_curves, TAG_NOTCH)
        gmsh.model.setPhysicalName(1, TAG_NOTCH, "notch")
    gmsh.model.addPhysicalGroup(1, [l_bot_l, l_bot_r, l_fluid_r, l_fluid_l, l_steel_r, l_steel_l], TAG_ABC)
    gmsh.model.setPhysicalName(1, TAG_ABC, "absorbing")

    return {
        "s_fluid": s_fluid, "s_steel": s_steel, "s_fill": s_fill, "id_curves": id_curves,
        "notch_curves": notch_curves,
        "tip_corners": ([(xn_l * MM, z_tip * MM), (xn_r * MM, z_tip * MM)]
                        if INCLUDE_NOTCH else []),
    }


# ========================================================================================
# 2. MESH SIZING
# ========================================================================================
def set_sizing(geo: dict, scale: float, quad: bool) -> None:
    """
    Size control = a Distance/Threshold field for the notch refinement, MIN-ed with a
    per-region size supplied by a Python callback. The callback is the honest way to get a
    hard fluid/steel size split: gmsh has no "size inside this surface" primitive that is
    reliable for 2-D, and a MathEval expression on r would need a ternary the field parser
    does not portably support.

    Two details that are not cosmetic:

      * the fluid size is applied for r <= R_ID and never relaxed, so the water - which has
        the SHORTEST wavelength - can never be accidentally coarsened;
      * the steel size does not JUMP from h_water to h_steel at the ID. A discontinuous size
        jump across an interface makes the first layer of steel elements slivers (measured:
        it dropped min gamma to 0.46, and every one of the worst cells sat on the ID). The
        steel size is therefore GRADED from h_water at the interface up to h_steel over
        ID_GRADE_LEN. Free side benefit: the shear wave born by mode conversion at the ID
        (lambda_S = 0.775 mm) is best resolved exactly there.
    """
    h_water, h_steel, h_notch = H_WATER * scale * MM, H_STEEL * scale * MM, H_NOTCH * scale * MM

    # Take full control of sizing: no size from CAD points, no curvature adaptation
    # (the arcs are R ~ 0.2 m, curvature would ask for absurdly large cells), and do not
    # smear boundary sizes into the interior.
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    field = gmsh.model.mesh.field
    if geo["notch_curves"]:
        f_dist = field.add("Distance")
        field.setNumbers(f_dist, "CurvesList", geo["notch_curves"])
        field.setNumber(f_dist, "Sampling", 400)
        f_thr = field.add("Threshold")
        field.setNumber(f_thr, "InField", f_dist)
        field.setNumber(f_thr, "SizeMin", h_notch)
        field.setNumber(f_thr, "SizeMax", h_steel)
        field.setNumber(f_thr, "DistMin", NOTCH_REFINE_R0 * scale * MM)
        field.setNumber(f_thr, "DistMax", NOTCH_REFINE_R1 * MM)
        field.setAsBackgroundMesh(f_thr)

    r_id_m, x_c_m, z_c_m = R_ID * MM, X_C * MM, Z_C * MM
    grade = ID_GRADE_LEN * MM

    h_array = H_ARRAY * scale * MM
    array_band = ARRAY_BAND * MM

    stair_h, stair_band = STAIR_H * MM, STAIR_BAND * MM

    def size_callback(dim, tag, x, y, z, lc):
        r = math.hypot(x - x_c_m, y - z_c_m)
        if STAIRCASE and abs(r - r_id_m) <= stair_band:
            # Resolve the 50 um risers properly on BOTH sides of the staircased ID. This does
            # not cost dt: the risers are forced geometry, so the smallest edge - and hence
            # the explicit time step - is already 50 um whatever the size field asks for. It
            # only buys element quality, which is what stops the comparison being confounded
            # by slivers rather than by the staircase itself.
            t = abs(r - r_id_m) / stair_band
            lc = min(lc, stair_h + t * (h_water - stair_h))
        if r <= r_id_m:                                     # water
            h = h_water
            # Refine a thin band at the transducer plane so INDIVIDUAL ELEMENTS are
            # resolvable. The array is 256 elements on a 0.30 mm pitch with a 0.25 mm
            # active width; at h_water = 0.28 mm a facet is as wide as a whole element,
            # so ~28 elements ended up with no facet of their own and the source could
            # not be defined. This costs nothing in CFL: water is 3.8x slower than
            # steel, so even 0.08 mm cells here give dt ~ 1.1 ns, above the ~0.86 ns
            # the steel already imposes. Ramped over ARRAY_BAND to avoid a size jump.
            if y <= array_band:
                t = y / array_band
                h = min(h, h_array + t * (h_water - h_array))
            return min(lc, h)
        t = min((r - r_id_m) / grade, 1.0)                   # graded steel
        return min(lc, h_water + t * (h_steel - h_water))

    gmsh.model.mesh.setSizeCallback(size_callback)

    if quad:
        # Frontal-Delaunay-for-quads + full-quad Blossom recombination. The solver's
        # tensor-product GLL basis prefers quads; whether this geometry survives it is
        # reported honestly by the caller (count of leftover triangles).
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        # 3 = blossom FULL-quad, which guarantees all-quads by subdividing - and therefore
        # requires an even number of edges on every boundary loop. The staircased ID adds ~457
        # one-element segments, which breaks that parity ("1D mesh cannot be divided by 2"), so
        # that variant uses 1 = plain blossom. Any leftover triangles are counted and reported
        # by verification 2, so this is visible rather than silent.
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1 if STAIRCASE else 3)
    else:
        gmsh.option.setNumber("Mesh.Algorithm", 6)          # Frontal-Delaunay triangles


# ========================================================================================
# 3. HELPERS FOR VERIFICATION
# ========================================================================================
TRI, QUAD = 2, 3                                            # gmsh element type ids
NODES_PER = {TRI: 3, QUAD: 4}
EDGES_OF = {TRI: [(0, 1), (1, 2), (2, 0)], QUAD: [(0, 1), (1, 2), (2, 3), (3, 0)]}


def node_coords() -> dict:
    tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)[:, :2]
    lut = np.zeros((int(tags.max()) + 1, 2))
    lut[tags.astype(int)] = coords
    return {"lut": lut, "tags": tags.astype(int)}


def region_cells(surface_tag: int, lut: np.ndarray):
    """Return [(etype, connectivity (n_cells, nodes_per_cell) as coordinates index)]."""
    etypes, etags, enodes = gmsh.model.mesh.getElements(2, surface_tag)
    out = []
    for et, tg, nd in zip(etypes, etags, enodes):
        npc = NODES_PER[int(et)]
        out.append((int(et), tg, np.asarray(nd, dtype=int).reshape(-1, npc)))
    return out


def edge_lengths(cells, lut) -> np.ndarray:
    """All (unique-ish) element edge lengths in a region, in metres."""
    lens = []
    for et, _, conn in cells:
        for a, b in EDGES_OF[et]:
            d = lut[conn[:, a]] - lut[conn[:, b]]
            lens.append(np.hypot(d[:, 0], d[:, 1]))
    return np.concatenate(lens)


def radii(xy: np.ndarray) -> np.ndarray:
    return np.hypot(xy[:, 0] - X_C * MM, xy[:, 1] - Z_C * MM)


def group_nodes(dim: int, tag: int) -> np.ndarray:
    _, coords = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)
    return coords.reshape(-1, 3)[:, :2]


def group_line_elements(tag: int, lut: np.ndarray):
    """Midpoints and count of the 1-D elements of a physical curve group."""
    mids, count = [], 0
    for ent in gmsh.model.getEntitiesForPhysicalGroup(1, tag):
        etypes, etags, enodes = gmsh.model.mesh.getElements(1, int(ent))
        for et, tg, nd in zip(etypes, etags, enodes):
            conn = np.asarray(nd, dtype=int).reshape(len(tg), -1)
            count += len(tg)
            mids.append(0.5 * (lut[conn[:, 0]] + lut[conn[:, 1]]))
    return (np.concatenate(mids) if mids else np.zeros((0, 2))), count


# ========================================================================================
# MAIN
# ========================================================================================
def main() -> None:
    global H_NOTCH, INCLUDE_NOTCH, STAIRCASE, NOTCH_FILL, X_MIN, X_MAX  # CLI may override
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply all target cell sizes (>1 coarser/faster, <1 finer)")
    ap.add_argument("--quad", action="store_true",
                    help="recombine into quadrilaterals (tensor-product GLL friendly)")
    ap.add_argument("--x-min", type=float, default=None, metavar="MM",
                    help="left edge of the domain (default %.1f). Widening the lateral margins "
                         "delays side-wall reflections: at the default 8 mm margin the first "
                         "return reaches the array at about 29.5 us, BEFORE the crack echo at "
                         "33-40 us; at 45 mm it lands near 42.5 us, after it. Bounded by the "
                         "geometry - see X_LIMIT_LO." % X_MIN)
    ap.add_argument("--x-max", type=float, default=None, metavar="MM",
                    help="right edge of the domain (default %.1f). See --x-min." % X_MAX)
    ap.add_argument("--no-plot", action="store_true", help="skip the PNG")
    ap.add_argument("--no-notch", action="store_true",
                    help="build the HEALTHY (defect-free) wall instead. Same sizing fields and "
                         "algorithm, so it differs from the cracked mesh ONLY by the notch. "
                         "Writes ili_mesh_healthy.* so it cannot overwrite the cracked mesh.")
    ap.add_argument("--notch-fill", action="store_true",
                    help="C5 CONTROLLED EXPERIMENT: mesh the notch interior and fill it with "
                         "k-Wave's 'outside' material (c 500 m/s, shear 0, rho 500) instead of "
                         "leaving it a traction-free void. Identical notch geometry either way, "
                         "so the single variable is the void treatment - the last untested "
                         "candidate for why we beat them. Writes ili_mesh*_fill.*")
    ap.add_argument("--staircase", action="store_true",
                    help="C4 CONTROLLED EXPERIMENT: represent the ID as a %.0f um PIXEL "
                         "STAIRCASE (k-Wave's grid spacing) instead of an exact arc. The OD "
                         "stays exact and every other setting is untouched, so the only "
                         "variable against the normal mesh is the ID representation. Writes "
                         "ili_mesh*_stair.* so it cannot overwrite a conforming mesh."
                         % (STAIR_PIXEL * 1e3,))
    ap.add_argument("--h-notch", type=float, default=None,
                    help="override the notch-region target cell size in mm (default %.2f). "
                         "Smaller is NOT better: it shrinks the global explicit time step. "
                         "See the H_NOTCH comment." % H_NOTCH)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    # Apply the notch-size override globally so every sizing field and printout agrees.
    if args.x_min is not None or args.x_max is not None:
        X_MIN = args.x_min if args.x_min is not None else X_MIN
        X_MAX = args.x_max if args.x_max is not None else X_MAX
        # Refuse rather than emit a mesh whose wall pokes through the transducer plane. A
        # gmsh failure here would be obscure; a geometry failure would be worse, because it
        # would look like a mesh that ran.
        if X_MIN < X_LIMIT_LO or X_MAX > X_LIMIT_HI:
            raise SystemExit(
                "lateral extent [%.2f, %.2f] mm leaves the valid range [%.2f, %.2f]: the ID "
                "arc reaches z = 0 there, so the pipe wall would sit above the flat array "
                "plane. Max usable width is %.1f mm (%.2fx the default 93.0). Going wider "
                "needs the water region rebuilt to follow the arc at constant standoff."
                % (X_MIN, X_MAX, X_LIMIT_LO, X_LIMIT_HI,
                   X_LIMIT_HI - X_LIMIT_LO, (X_LIMIT_HI - X_LIMIT_LO) / 93.0))
        print("    lateral extent overridden: [%.2f, %.2f] mm, width %.1f mm (%.2fx default); "
              "margins %.1f / %.1f mm outboard of the aperture"
              % (X_MIN, X_MAX, X_MAX - X_MIN, (X_MAX - X_MIN) / 93.0,
                 ARRAY_X0 - X_MIN, X_MAX - ARRAY_X1))

    if args.h_notch is not None:
        H_NOTCH = args.h_notch
    if args.no_notch:
        INCLUDE_NOTCH = False
        print("*** HEALTHY (no-notch) variant: defect-free wall ***")
    if args.notch_fill:
        if args.no_notch:
            raise SystemExit("--notch-fill and --no-notch are contradictory: there is no notch "
                             "to fill on the healthy variant.")
        NOTCH_FILL = True
        print("*** C5 NOTCH-FILL variant: notch meshed and filled with k-Wave's 'outside' "
              "material (c 500, shear 0, rho 500) instead of being a traction-free void ***")
    if args.staircase:
        STAIRCASE = True
        print(f"*** C4 STAIRCASE variant: ID represented as a {STAIR_PIXEL*1e3:.0f} um pixel "
              f"staircase, OD exact ***")
    # Variant-aware output stem so the healthy mesh can never overwrite the cracked one.
    # Variant- AND scale-aware stem: a convergence study generates several meshes and must
    # not have --scale 0.8 silently overwrite the scale-1.0 mesh the solver is already using.
    stem = "ili_mesh_healthy" if not INCLUDE_NOTCH else "ili_mesh"
    if abs(args.scale - 1.0) > 1e-9:
        stem += f"_s{args.scale:g}".replace(".", "p")
    if abs((X_MAX - X_MIN) - 93.0) > 1e-9:
        # Lateral extent must be in the name for the same reason the scale is: a widened
        # domain must never silently overwrite the mesh every published solve used.
        stem += f"_w{X_MAX - X_MIN:.0f}"
    if NOTCH_FILL:
        stem += "_fill"
    if STAIRCASE:
        stem += "_stair"
    if not args.quad:
        # Cell type must be in the name too. Without this a triangle build silently overwrites
        # the quad mesh of the same variant - and the quad meshes are what every existing solve
        # used, so that would break reproducibility of results already on disk.
        stem += "_tri"
    if args.h_notch is not None:
        # ...and so must the notch cell size, for exactly the same reason. It was the one
        # variant-producing flag missing from this stem: two meshes differing only in
        # --h-notch are DIFFERENT meshes, and until now the second silently overwrote the
        # first. Found while wiring the GUI, which exposes this as a user-facing parameter.
        stem += f"_hn{args.h_notch:g}".replace(".", "p")

    # --- VERIFICATION 1: the standoff identity that the coordinate choice exists for -----
    assert abs((R_ID + Z_C) - STANDOFF) < 1e-12, \
        f"standoff identity violated: R_ID + z_c = {R_ID + Z_C} != {STANDOFF}"
    assert abs((R_OD - R_ID) - 9.525) < 1e-12, "wall thickness is not 9.525 mm"
    assert abs((N_ELEM - 1) * PITCH - (ARRAY_X1 - ARRAY_X0)) < 1e-12, "aperture != 255*pitch"
    print("=" * 88)
    print("ILI 2-D CONFORMING MESH")
    print("=" * 88)
    print(f"[1] standoff identity      R_ID + z_c = {R_ID:.3f} + ({Z_C:.3f}) = "
          f"{R_ID + Z_C:.12f} mm == {STANDOFF} mm   OK (asserted)")
    print(f"    wall {R_OD - R_ID:.3f} mm | OD on axis z = {R_OD + Z_C:.3f} mm | "
          f"aperture {ARRAY_X1 - ARRAY_X0:.1f} mm = {N_ELEM - 1}*{PITCH} mm")
    print(f"    target cells (scale {args.scale}): water {H_WATER * args.scale:.4f} mm, "
          f"steel {H_STEEL * args.scale:.4f} mm, notch {H_NOTCH * args.scale:.4f} mm | "
          f"cells: {'QUAD (recombined)' if args.quad else 'TRIANGLE'}")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("ili_2d")
    geo = build_geometry()
    set_sizing(geo, args.scale, args.quad)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.removeDuplicateNodes()

    msh_path = OUT / f"{stem}.msh"
    gmsh.write(str(msh_path))

    nc = node_coords()
    lut = nc["lut"]
    cells_f = region_cells(geo["s_fluid"], lut)
    cells_s = region_cells(geo["s_steel"], lut)
    # The fill region must be counted HERE, while gmsh is still initialised. Counting it later
    # in verification 7 threw "Gmsh has not been initialized", because gmsh.finalize() has run
    # by then and the round trip is reading the written file instead.
    cells_x = region_cells(geo["s_fill"], lut) if geo["s_fill"] is not None else []
    n_fill = sum(len(c[1]) for c in cells_x)
    # Count nodes actually USED by cells. gmsh also carries a node for the arc-centre
    # geometry point, which belongs to no element and is dropped when the .msh is written.
    n_nodes = len(np.unique(np.concatenate([c[2].ravel() for c in cells_f + cells_s])))
    n_f = sum(len(c[1]) for c in cells_f)
    n_s = sum(len(c[1]) for c in cells_s)
    n_tri = sum(len(c[1]) for c in cells_f + cells_s if c[0] == TRI)
    n_quad = sum(len(c[1]) for c in cells_f + cells_s if c[0] == QUAD)

    # --- VERIFICATION 2: counts ----------------------------------------------------------
    print("-" * 88)
    if n_fill:
        h_fill = H_NOTCH * args.scale
        lam_fill = 500.0 / 4.0e6 * 1e3          # 500 m/s at 4 MHz = 0.125 mm
        print("-" * 88)
        print(f"    NOTCH FILL: {n_fill} cells at ~{h_fill:.4f} mm. k-Wave's 'outside' is "
              f"c 500 m/s -> lambda {lam_fill:.4f} mm, so degree-4 nodes (h/4) give "
              f"{4 * lam_fill / h_fill:.2f} per wavelength")
        print(f"    against k-Wave's {lam_fill / 0.05:.2f} points per wavelength on its 50 um "
              f"grid - comparable, so this is a FAITHFUL emulation of their treatment,")
        print(f"    which means it deliberately conflates the material substitution with the "
              f"under-resolution of it, exactly as k-Wave does.")
    print(f"[2] cells {n_f + n_s + n_fill} (fluid {n_f}, steel {n_s}, fill {n_fill}) | "
          f"p1 nodes {n_nodes}")

    # --- VERIFICATION 3: quality ---------------------------------------------------------
    # gamma (inradius/circumradius, 1 = equilateral) for triangles; minSICN (scaled
    # inverse condition number, 1 = perfect, <=0 = inverted) for quads.
    qname = "minSICN" if n_quad else "gamma"
    q_all, q_cent, q_reg = [], [], []
    for label, cells in (("fluid", cells_f), ("steel", cells_s)):
        for et, tg, conn in cells:
            q = np.asarray(gmsh.model.mesh.getElementQualities(tg, qname))
            q_all.append(q)
            q_cent.append(lut[conn].mean(axis=1))
            q_reg += [label] * len(tg)
    q_all = np.concatenate(q_all)
    q_cent = np.concatenate(q_cent)
    q_reg = np.array(q_reg)
    order = np.argsort(q_all)
    tips = np.array(geo["tip_corners"]).reshape(-1, 2)
    print("-" * 88)
    print(f"[3] quality ({qname}): min {q_all.min():.4f}  mean {q_all.mean():.4f}  "
          f"median {np.median(q_all):.4f}  p1 {np.percentile(q_all, 1):.4f}  "
          f"|  cells < 0.30: {(q_all < 0.30).sum()}  < 0.50: {(q_all < 0.50).sum()}")
    print("    worst 5 cells (position in mm, distance to nearest notch tip corner):"
          if tips.size else "    worst 5 cells (position in mm; no notch in this variant):")
    for i in order[:5]:
        # tips is empty on the healthy variant, so there is no corner to measure against.
        d = np.hypot(*(tips - q_cent[i]).T).min() * 1e3 if tips.size else float("nan")
        print(f"      {qname} {q_all[i]:.4f}  at (x={q_cent[i, 0] * 1e3:8.3f}, "
              f"z={q_cent[i, 1] * 1e3:8.3f}) mm  in {q_reg[i]:5s}  d_tip = {d:8.3f} mm")

    # --- VERIFICATION 4: achieved cell sizes per region ----------------------------------
    e_f, e_s = edge_lengths(cells_f, lut) * 1e3, edge_lengths(cells_s, lut) * 1e3
    lam_w = 0.375
    print("-" * 88)
    print(f"[4] achieved edge length (mm)   min      mean      max     target")
    print(f"    fluid (water)              {e_f.min():.4f}   {e_f.mean():.4f}   {e_f.max():.4f}   "
          f"{H_WATER * args.scale:.4f}")
    print(f"    steel                      {e_s.min():.4f}   {e_s.mean():.4f}   {e_s.max():.4f}   "
          f"{H_STEEL * args.scale:.4f}")
    # p=3 puts 3 node spacings across a cell, so nodes per wavelength ~ 3*lambda/h.
    ppw_mean = 3 * lam_w / e_f.mean()
    ppw_worst = 3 * lam_w / e_f.max()
    print(f"    water lambda = {lam_w} mm -> p3 nodes per wavelength: mean {ppw_mean:.2f}, "
          f"worst (longest) cell {ppw_worst:.2f}   (design target 4)")
    h_t = H_WATER * args.scale
    print(f"    fluid target cell size {h_t:.4f} mm: MEAN "
          f"{'MET' if e_f.mean() <= 1.05 * h_t else 'NOT MET'} ({e_f.mean():.4f} mm, "
          f"{e_f.mean() / h_t:.2f}x); longest fluid edge {e_f.max() / h_t:.2f}x the target "
          f"(unstructured spread, not coarsening)")
    print(f"    the fluid is nowhere coarsened: max fluid edge {e_f.max():.4f} mm < steel "
          f"target {H_STEEL * args.scale:.4f} mm. To get >= 4 p3 nodes/lambda in EVERY fluid "
          f"cell (not just on average) run --scale {0.28 / e_f.max() * args.scale:.2f}.")

    # --- VERIFICATION 5: geometric conformity (THE headline number) ----------------------
    print("-" * 88)
    print("[5] geometric conformity of the meshed arcs (deviation from the exact circle):")
    dev_report = {}
    for tag, name, R in ((TAG_ID, "ID", R_ID), (TAG_OD, "OD", R_OD)):
        gn = group_nodes(1, tag)
        dev_node = np.abs(radii(gn) - R * MM).max() * 1e6                # microns
        mids, n_line = group_line_elements(tag, lut)
        dev_chord = np.abs(radii(mids) - R * MM).max() * 1e6             # microns (sagitta)
        dev_report[name] = (dev_node, dev_chord, len(gn), n_line)
        note = ("STAIRCASED BY DESIGN" if (STAIRCASE and name == "ID")
                else "machine precision")
        print(f"    {name} arc: {len(gn):5d} boundary nodes, {n_line:5d} facets | "
              f"max |r - R| at NODES = {dev_node:8.2e} um ({note}) | "
              f"max chord-midpoint sagitta = {dev_chord:7.4f} um")
    stair = 0.5 * H_WATER * args.scale * 1e3
    print(f"    a voxel/staircase grid at the same cell size would deviate by ~dx/2 = "
          f"{stair:.1f} um  ->  {stair / max(dev_report['OD'][1], 1e-9):.0f}x worse")
    if STAIRCASE:
        # The C4 variant's whole point: quote the ID deviation it was BUILT to have, so the
        # controlled experiment reports its own independent variable rather than a defect.
        q_half = 0.5 * STAIR_PIXEL * 1e3
        print(f"    *** C4 VARIANT: the ID is a {STAIR_PIXEL*1e3:.0f} um pixel staircase on "
              f"purpose. Measured node deviation {dev_report['ID'][0]:.1f} um vs the expected "
              f"~q/2 = {q_half:.1f} um.")
        print(f"        The OD stays exact, so the ONE variable against the conforming run is "
              f"the ID representation.")
    # conformity of the fluid/solid interface: EVERY ID curve must bound BOTH surfaces
    for c in geo["id_curves"]:
        up, _ = gmsh.model.getAdjacencies(1, c)
        assert len(up) == 2, \
            f"ID curve {c} bounds {len(up)} surfaces, expected 2 (mesh NON-conforming)"
    up, _ = gmsh.model.getAdjacencies(1, geo["id_curves"][0])
    print(f"    all {len(geo['id_curves'])} ID curve(s) bound surfaces "
          f"{sorted(int(u) for u in up)} -> shared entities, fluid/solid nodes are coincident "
          f"BY CONSTRUCTION (asserted)")

    # --- VERIFICATION 6: notch dimensions measured FROM THE MESH -------------------------
    print("-" * 88)
    if not INCLUDE_NOTCH:
        print("[6] notch measurement SKIPPED - healthy (no-notch) variant: tag 13 is",
              "absent by construction, so there is nothing to measure.")
    else:
        nn = group_nodes(1, TAG_NOTCH) if INCLUDE_NOTCH else np.empty((0, 3))
        w_meas = (nn[:, 0].max() - nn[:, 0].min()) * 1e3
        z_tip_meas = nn[:, 1].min() * 1e3
        depth_axial = (R_OD + Z_C) - z_tip_meas
        # Radial depth: deepest point of the slot is the CENTRE of the flat tip face (closest to
        # the pipe axis); the two corners are 0.5 mm off axis and so are marginally shallower.
        depth_radial_max = (R_OD * MM - radii(nn).min()) * 1e3
        corner_nodes = nn[(np.abs(nn[:, 1] - nn[:, 1].min()) < 1e-9)
                          & (np.abs(np.abs(nn[:, 0] - NOTCH_X * MM) - NOTCH_WIDTH / 2 * MM) < 1e-9)]
        depth_radial_corner = (R_OD * MM - radii(corner_nodes).max()) * 1e3
        tip_face = nn[np.abs(nn[:, 1] - nn[:, 1].min()) < 1e-9]
        print(f"[6] notch measured from mesh boundary nodes ({len(nn)} nodes on tag {TAG_NOTCH}):")
        print(f"    width  (x span of the walls)          {w_meas:.6f} mm   (CAD {NOTCH_WIDTH})")
        print(f"    tip face z                            {z_tip_meas:.6f} mm   "
              f"(CAD {Z_C + R_OD - NOTCH_DEPTH:.6f}), {len(tip_face)} nodes on the flat tip")
        print(f"    depth on axis (OD apex - tip)         {depth_axial:.6f} mm   (CAD {NOTCH_DEPTH})")
        print(f"    max radial depth (tip face centre)    {depth_radial_max:.6f} mm")
        print(f"    radial depth at the 2 tip CORNERS     {depth_radial_corner:.6f} mm   "
              f"({(NOTCH_DEPTH - depth_radial_corner) * 1e3:.2f} um shallower: the tip face is "
              f"FLAT, corners sit 0.5 mm off axis - expected, not an error)")
        assert abs(w_meas - NOTCH_WIDTH) < 1e-6 and abs(depth_axial - NOTCH_DEPTH) < 1e-6, \
            "notch dimensions in the mesh do not match the CAD input"

    # ---- capture what the plot needs, then hand gmsh over to dolfinx -------------------
    plot_data = {
        "lut": lut,
        "fluid": [(et, conn) for et, _, conn in cells_f],
        "steel": [(et, conn) for et, _, conn in cells_s],
    }
    gmsh.finalize()

    # --- VERIFICATION 7: DOLFINx round trip ---------------------------------------------
    from mpi4py import MPI
    from dolfinx import fem, io
    from dolfinx.io import gmsh as dgmsh

    md = dgmsh.read_from_msh(str(msh_path), MPI.COMM_WORLD, gdim=2)
    dmesh, ct, ft = md.mesh, md.cell_tags, md.facet_tags
    print("-" * 88)
    print(f"[7] DOLFINx {__import__('dolfinx').__version__} round trip: "
          f"cell type {dmesh.topology.cell_name()}, "
          f"{dmesh.topology.index_map(2).size_global} cells, "
          f"{dmesh.geometry.x.shape[0]} vertices")
    exp_cells = {TAG_FLUID: n_f, TAG_STEEL: n_s}
    if n_fill:
        exp_cells[TAG_FILL] = n_fill
    names = {TAG_FLUID: "fluid", TAG_STEEL: "steel", TAG_FILL: "fill "}
    for tag, want in exp_cells.items():
        got = int((ct.values == tag).sum())
        print(f"    cell tag {tag:2d} ({names[tag]}): "
              f"{got} (gmsh {want}) {'OK' if got == want else 'MISMATCH'}")
        assert got == want
    for tag, name in ((TAG_ARRAY, "array"), (TAG_ID, "id"), (TAG_OD, "od"),
                      (TAG_NOTCH, "notch"), (TAG_ABC, "abc")):
        got = int((ft.values == tag).sum())
        print(f"    facet tag {tag:2d} ({name:5s}): {got}")
        if tag == TAG_NOTCH and not INCLUDE_NOTCH:
            continue                     # healthy variant has no notch facets, by design
        assert got > 0, f"facet tag {tag} vanished in the round trip"

    # degree-3 vector space: the real DOF count the solver will pay for
    V = fem.functionspace(dmesh, ("Lagrange", 3, (2,)))
    ndof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    print(f"    degree-3 VECTOR dofs = {ndof} ({ndof // 2} p3 nodes x 2 components)")

    dmesh.topology.create_connectivity(1, 2)
    xdmf_path = OUT / f"{stem}.xdmf"
    with io.XDMFFile(dmesh.comm, str(xdmf_path), "w") as f:
        f.write_mesh(dmesh)
        f.write_meshtags(ct, dmesh.geometry)
        f.write_meshtags(ft, dmesh.geometry)
    # ... and read the XDMF straight back, so the artifact the solver may load is proven
    # loadable too (grid names: "mesh", "cell_tags", "facet_tags").
    with io.XDMFFile(MPI.COMM_WORLD, str(xdmf_path), "r") as f:
        m2 = f.read_mesh(name="mesh")
        ct2 = f.read_meshtags(m2, name="cell_tags")
        m2.topology.create_connectivity(1, 2)
        ft2 = f.read_meshtags(m2, name="facet_tags")
    assert m2.topology.index_map(2).size_global == n_f + n_s + n_fill
    assert int((ct2.values == TAG_FLUID).sum()) == n_f and int((ct2.values == TAG_STEEL).sum()) == n_s
    _expect = ([TAG_ARRAY, TAG_ID, TAG_OD, TAG_NOTCH, TAG_ABC] if INCLUDE_NOTCH
               else [TAG_ARRAY, TAG_ID, TAG_OD, TAG_ABC])
    assert sorted(np.unique(ft2.values)) == _expect
    counts = lambda a: {int(k): int(v) for k, v in zip(*np.unique(a, return_counts=True))}
    print(f"    XDMF re-read OK: {m2.topology.index_map(2).size_global} cells, cell tags "
          f"{counts(ct2.values)}, facet tags {counts(ft2.values)}")
    print(f"    wrote {msh_path.name}, {stem}.xdmf/.h5 -> {OUT}")

    if not args.no_plot:
        plot(plot_data, args, ndof, n_f + n_s, n_nodes, dev_report, stem)
        print(f"    wrote {stem}.png -> {OUT}")
    print("=" * 88)


# ========================================================================================
# FIGURE
# ========================================================================================
def polys(cells, lut):
    """Cell polygons in mm (the mesh itself is in metres; the figure is labelled in mm)."""
    return [lut[c] * 1e3 for et, conn in cells for c in conn]


def plot(d, args, ndof, ncell, nnode, dev, stem="ili_mesh") -> None:
    C_W, C_S = "#bcdcf5", "#d9d9d9"
    fig, ax = plt.subplots(figsize=(13.0, 6.6))

    for cells, fc, lw in ((d["fluid"], C_W, 0.12), (d["steel"], C_S, 0.12)):
        ax.add_collection(PolyCollection(polys(cells, d["lut"]), facecolors=fc,
                                         edgecolors="#33333366", linewidths=lw))

    th = np.linspace(0, np.pi, 4000)
    for R, col, name in ((R_ID, "#0b6fb5", "ID"), (R_OD, "#b52d0b", "OD")):
        x = (X_C + R * np.cos(th)); z = (Z_C + R * np.sin(th))
        m = (x >= X_MIN - 1) & (x <= X_MAX + 1) & (z >= -1)
        ax.plot(x[m], z[m], col, lw=1.6, zorder=5,
                label=f"{name} arc r={R:.3f} mm (exact)")
    ax.plot([ARRAY_X0, ARRAY_X1], [0, 0], color="#111111", lw=4.0, solid_capstyle="butt",
            zorder=6, label=f"transducer: {N_ELEM} el @ {PITCH} mm = {ARRAY_X1:.1f} mm")

    ax.set_xlim(X_MIN - 2, X_MAX + 2); ax.set_ylim(-5.0, 33)
    ax.set_aspect("equal"); ax.set_xlabel("x  [mm]"); ax.set_ylabel("z (beam axis, depth from array)  [mm]")
    ax.annotate("WATER (tag 1)", (10, 8), fontsize=11, color="#0b4f7a", weight="bold")
    ax.annotate("STEEL (tag 2)", (10, 21.5), fontsize=11, color="#5a5a5a", weight="bold")
    ax.annotate("air / not meshed\n(traction-free)", (60, 30.4), fontsize=9, color="#7a2d0b", ha="center")
    ax.annotate(f"NOTCH {NOTCH_DEPTH} x {NOTCH_WIDTH} mm\n(void, tag 13)", (NOTCH_X, 31.2),
                fontsize=9.5, color="#7a2d0b", ha="center", weight="bold")
    ax.annotate("", xy=(NOTCH_X, 29.9), xytext=(NOTCH_X, 30.9),
                arrowprops=dict(arrowstyle="->", color="#7a2d0b", lw=1.2))
    ax.annotate(f"standoff {STANDOFF:.1f} mm", (NOTCH_X + 1.2, 9.0), fontsize=9, color="#0b4f7a",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
    ax.annotate("", xy=(NOTCH_X, 0), xytext=(NOTCH_X, STANDOFF),
                arrowprops=dict(arrowstyle="<->", color="#0b4f7a", lw=1.0))
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95, ncol=1)
    ax.set_title(f"ILI 2-D conforming mesh (gmsh -> DOLFINx)  |  "
                 f"{'quads' if args.quad else 'triangles'}: {ncell} cells, {nnode} p1 nodes, "
                 f"{ndof} p3 vector DOF  |  scale={args.scale}\n"
                 f"meshed arcs vs the exact circles: boundary nodes "
                 f"{max(dev['ID'][0], dev['OD'][0]):.1e} um (machine precision), "
                 f"chord midpoints ID {dev['ID'][1]:.3f} / OD {dev['OD'][1]:.3f} um\n"
                 f"a staircased voxel grid at the same cell size would deviate by ~"
                 f"{0.5 * H_WATER * args.scale * 1e3:.0f} um", fontsize=10.5)

    # ---- zoomed inset on the notch tip -------------------------------------------------
    w = 1.9
    x0, x1 = NOTCH_X - w, NOTCH_X + w
    z_tip = Z_C + R_OD - NOTCH_DEPTH
    z0, z1 = z_tip - 1.0, R_OD + Z_C + 0.7
    axi = ax.inset_axes([0.745, 0.22, 0.235, 0.58], xlim=(x0, x1), ylim=(z0, z1))
    axi.set_facecolor("white")
    for cells, fc in ((d["fluid"], C_W), (d["steel"], C_S)):
        axi.add_collection(PolyCollection(polys(cells, d["lut"]), facecolors=fc,
                                          edgecolors="#333333aa", linewidths=0.28))
    x = (X_C + R_OD * np.cos(th)); z = (Z_C + R_OD * np.sin(th))
    axi.plot(x, z, "#b52d0b", lw=1.3)
    for xw in (NOTCH_X - NOTCH_WIDTH / 2, NOTCH_X + NOTCH_WIDTH / 2):
        axi.plot([xw, xw], [z_tip, z_arc(xw, R_OD)], "#7a2d0b", lw=1.3)
    axi.plot([NOTCH_X - NOTCH_WIDTH / 2, NOTCH_X + NOTCH_WIDTH / 2], [z_tip, z_tip],
             "#7a2d0b", lw=1.3)
    axi.scatter([NOTCH_X - NOTCH_WIDTH / 2, NOTCH_X + NOTCH_WIDTH / 2], [z_tip, z_tip],
                s=22, facecolor="none", edgecolor="#d62728", lw=1.3, zorder=8)
    # adjustable="datalim" keeps the inset BOX where it was placed and widens the view
    # instead, so the inset can never spill out over the parent axes.
    axi.set_aspect("equal", adjustable="datalim")
    axi.set_title(f"ZOOM: notch tip, h={H_NOTCH * args.scale:.3f} mm", fontsize=8.5)
    axi.tick_params(labelsize=7)
    axi.annotate("sharp tip corners\n(exact mesh vertices)", xy=(NOTCH_X - NOTCH_WIDTH / 2, z_tip),
                 xytext=(x0 + 0.12, z_tip - 0.80), fontsize=7, color="#d62728", ha="left",
                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))
    axi.annotate("notch walls\n(tag 13, free)", (NOTCH_X + NOTCH_WIDTH / 2 + 0.15, z_tip + 1.6),
                 fontsize=7, color="#7a2d0b",
                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    axi.annotate("OD arc (exact)", (x1 - 1.75, z1 - 0.33), fontsize=7, color="#b52d0b")
    ax.indicate_inset_zoom(axi, edgecolor="#d62728", lw=1.0)

    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
