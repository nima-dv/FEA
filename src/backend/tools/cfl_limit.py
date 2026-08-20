r"""
What is the ACTUAL stability limit of our time step? Measure it instead of using a rule of thumb.

WHY THIS MATTERS MORE THAN PARALLELISM
Our solver picks dt from a heuristic, dt = CFL * h_min / (c * p^2) with CFL = 0.30, which gives
0.3666 ns on the production mesh and 163680 steps. SPECFEM2D - the same method family, GLL nodes,
lumped mass, explicit second-order stepping - documents a very different rule: dt <= 0.697 *
(minimum GLL point spacing) / c at degree 4. On our mesh that would be about 2.4 ns, roughly
SIX TIMES larger. If our heuristic is that conservative, we are paying a 6x runtime penalty for
nothing, and no amount of MPI or matrix-free work would recover as much.

The rule of thumb is not the real limit though, and neither is theirs. For an explicit
central-difference scheme with a lumped (diagonal) mass matrix, the exact stability condition is

        dt <= 2 / omega_max,    omega_max = sqrt( lambda_max( M^-1 K ) )

so the honest answer is to compute lambda_max directly by power iteration and read dt_max off
it. That costs a few dozen mat-vecs and settles the question with no heuristic at all.

Reported alongside: the dt our solver would currently choose, SPECFEM2D's rule, and the measured
limit, so the size of the gap is explicit. A stability limit is a CEILING - production should
sit below it with margin, since the source term and the absorbing boundary are not included in
this eigenvalue.

RUN
  ./run.ps1 python3 tools/cfl_limit.py --mesh results/ili_mesh/ili_mesh_s0p8.msh --degree 4
"""
from __future__ import annotations

import argparse

import basix
import numpy as np
import scipy.sparse as sp
import ufl
from dolfinx.fem import Function, functionspace, form, assemble_vector
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import gmsh as dgmsh
from mpi4py import MPI

C_P, C_S, RHO_S = 5700.0, 3100.0, 7850.0
C_F, RHO_F = 1500.0, 1000.0
MU_S, LAM_S = RHO_S * C_S**2, RHO_S * (C_P**2 - 2 * C_S**2)
LAM_F = RHO_F * C_F**2
TAG_FLUID, TAG_STEEL = 1, 2
CFL_OURS = 0.30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="results/ili_mesh/ili_mesh_s0p8.msh")
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--t-end", type=float, default=60e-6)
    args = ap.parse_args()

    res = dgmsh.read_from_msh(args.mesh, MPI.COMM_WORLD, gdim=2)
    domain, ct = (res.mesh, res.cell_tags) if not isinstance(res, tuple) else (res[0], res[1])
    el = basix.ufl.element("Lagrange", domain.basix_cell(), args.degree,
                           lagrange_variant=basix.LagrangeVariant.gll_warped, shape=(2,))
    V = functionspace(domain, el)
    Q = functionspace(domain, ("DG", 0))
    ndof = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    tdim = domain.topology.dim
    ncell = domain.topology.index_map(tdim).size_local

    lam, mu, rho = Function(Q), Function(Q), Function(Q)
    lam.x.array[:] = LAM_F
    mu.x.array[:] = 0.0
    rho.x.array[:] = RHO_F
    steel = ct.find(TAG_STEEL)
    lam.x.array[steel] = LAM_S
    mu.x.array[steel] = MU_S
    rho.x.array[steel] = RHO_S

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))
    sig = lambda w: lam * ufl.tr(eps(w)) * ufl.Identity(2) + 2 * mu * eps(w)
    A = assemble_matrix(form(ufl.inner(sig(u), eps(v)) * ufl.dx))
    A.assemble()
    ai, aj, av = A.getValuesCSR()
    K = sp.csr_matrix((av, aj, ai), shape=A.getSize())
    m = assemble_vector(form(rho * ufl.inner(v, ufl.as_vector((1.0, 1.0))) * ufl.dx)).array.copy()
    assert m.min() > 0
    print(f"{args.mesh}\n  {ncell} cells, degree {args.degree}, {ndof} dofs, {K.nnz} nnz")

    # --- the heuristic our solver uses, reproduced exactly -------------------------------
    cmap = domain.geometry.dofmap.reshape(ncell, -1)
    xg = domain.geometry.x
    h_min = np.full(ncell, np.inf)
    for a in range(cmap.shape[1]):
        for b in range(a + 1, cmap.shape[1]):
            h_min = np.minimum(h_min, np.linalg.norm(xg[cmap[:, a]] - xg[cmap[:, b]], axis=1))
    c_cell = np.full(ncell, C_F)
    c_cell[steel] = C_P
    dt_ours = float((CFL_OURS * h_min / (c_cell * args.degree ** 2)).min())

    # --- SPECFEM2D's documented rule: 0.697 * min GLL spacing / c at degree 4 -------------
    # Minimum GLL spacing inside a degree-p element is a fixed fraction of the element size.
    gll = basix.create_element(basix.ElementFamily.P, basix.CellType.interval, args.degree,
                               basix.LagrangeVariant.gll_warped).points[:, 0]
    frac = float(np.diff(np.sort(gll)).min())          # 0.173 of the element size at p=4
    dt_spec = float((0.697 * frac * h_min / c_cell).min())

    # --- the EXACT limit: power iteration on M^-1 K --------------------------------------
    rng = np.random.default_rng(0)
    x = rng.standard_normal(ndof)
    x /= np.linalg.norm(x)
    lam_max = 0.0
    for i in range(args.iters):
        y = (K @ x) / m
        nrm = np.linalg.norm(y)
        if nrm == 0:
            break
        x_new = y / nrm
        lam_new = float(x_new @ ((K @ x_new) / m))     # Rayleigh quotient
        conv = abs(lam_new - lam_max) / max(lam_new, 1e-30)
        lam_max, x = lam_new, x_new
        if i > 20 and conv < 1e-6:
            break
    print(f"  power iteration: {i+1} its, lambda_max {lam_max:.6e}, "
          f"relative change {conv:.2e}")
    omega = np.sqrt(lam_max)
    dt_exact = 2.0 / omega

    def steps(dt):
        return int(np.ceil(args.t_end / dt))

    print("\n" + "=" * 74)
    print(f"{'time step':<34}{'dt [ns]':>12}{'steps':>12}{'vs ours':>14}")
    print("-" * 74)
    print(f"{'ours (heuristic CFL 0.30)':<34}{dt_ours*1e9:12.4f}{steps(dt_ours):12d}"
          f"{1.0:13.2f}x")
    print(f"{'SPECFEM2D rule (0.697*dGLL/c)':<34}{dt_spec*1e9:12.4f}{steps(dt_spec):12d}"
          f"{dt_spec/dt_ours:13.2f}x")
    print(f"{'EXACT limit 2/sqrt(lambda_max)':<34}{dt_exact*1e9:12.4f}{steps(dt_exact):12d}"
          f"{dt_exact/dt_ours:13.2f}x")
    print("-" * 74)
    print(f"our dt sits at {dt_ours/dt_exact*100:.1f}% of the true stability limit")
    print(f"a run at 80% of the limit would take {steps(0.8*dt_exact)} steps, "
          f"{dt_ours/(0.8*dt_exact):.2f}x fewer than now")
    print("=" * 74)
    print("The exact limit is a CEILING for the stiffness operator alone. The source term and\n"
          "the dashpot absorbing boundary are not in this eigenvalue, so production must keep\n"
          "margin - and any change must be validated against the existing arrival times, not\n"
          "just checked for non-divergence.")


if __name__ == "__main__":
    main()
