r"""
Can DOLFINx parallelise the hot loop of our solver, and by how much? Measure, do not assume.

The production solve spends essentially all of its time in one operation per time step: a
sparse matrix-vector product with the stiffness matrix. Our solver currently converts that
matrix to scipy and multiplies with numpy, which is SINGLE-THREADED - so the honest question
is not "does DOLFINx support MPI" (it does, by design) but "what speed-up would we actually
get on OUR matrix, at OUR problem size, in OUR container".

This script measures exactly that: assemble the real degree-4 stiffness matrix on the real
production mesh, then time PETSc MatMult, which is the parallel equivalent of the scipy
product in the time loop.

RUN
  # serial baseline
  docker run ... python3 tools/mpi_scaling.py
  # 2, 4, 6 ranks
  docker run ... mpirun -n 4 python3 tools/mpi_scaling.py
"""
from __future__ import annotations

import argparse
import time

import basix
import numpy as np
import ufl
from dolfinx.fem import functionspace, form
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import gmsh as dgmsh
from mpi4py import MPI

# Steel, as constants. Material values change neither the sparsity pattern nor the cost of a
# mat-vec, and using constants keeps this script independent of the mesh's cell tags.
C_P, C_S, RHO = 5700.0, 3100.0, 7850.0
MU = RHO * C_S ** 2
LAM = RHO * (C_P ** 2 - 2 * C_S ** 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="results/ili_mesh/ili_mesh_s0p8.msh")
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--reps", type=int, default=60)
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size

    t0 = time.time()
    res = dgmsh.read_from_msh(args.mesh, comm, gdim=2)
    domain = res.mesh if not isinstance(res, tuple) else res[0]
    t_mesh = time.time() - t0

    el = basix.ufl.element("Lagrange", domain.basix_cell(), args.degree,
                           lagrange_variant=basix.LagrangeVariant.gll_warped, shape=(2,))
    V = functionspace(domain, el)
    im = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    n_loc, n_glob = im.size_local * bs, im.size_global * bs

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))
    sig = lambda w: LAM * ufl.tr(eps(w)) * ufl.Identity(2) + 2 * MU * eps(w)

    t0 = time.time()
    A = assemble_matrix(form(ufl.inner(sig(u), eps(v)) * ufl.dx))
    A.assemble()
    t_asm = time.time() - t0

    x, y = A.createVecRight(), A.createVecLeft()
    x.set(1.0)
    for _ in range(5):                       # warm-up: first call pays setup costs
        A.mult(x, y)
    comm.Barrier()
    t0 = time.time()
    for _ in range(args.reps):
        A.mult(x, y)
    comm.Barrier()
    per = (time.time() - t0) / args.reps

    # THE comparison that decides whether MPI is worth the refactor. Our time loop does not
    # use PETSc at all - it converts the matrix to scipy and multiplies with numpy, which is
    # single-threaded. So the honest baseline for "what we have today" is scipy on one core,
    # not PETSc on one core, and PETSc's serial MatMult turns out to be the slower of the two.
    t_scipy = float("nan")
    if size == 1:
        import scipy.sparse as sp
        ai, aj, av = A.getValuesCSR()
        Ks = sp.csr_matrix((av, aj, ai), shape=A.getSize())
        xv = np.ones(Ks.shape[1])
        for _ in range(3):
            Ks @ xv
        t0 = time.time()
        for _ in range(args.reps):
            Ks @ xv
        t_scipy = (time.time() - t0) / args.reps

    cells_loc = domain.topology.index_map(domain.topology.dim).size_local
    cells_glob = domain.topology.index_map(domain.topology.dim).size_global
    info = comm.gather((rank, cells_loc, n_loc, A.getInfo()["nz_used"]), root=0)

    if rank == 0:
        print(f"ranks {size}   mesh {args.mesh}   degree {args.degree}")
        print(f"  global: {cells_glob} cells, {n_glob} dofs, "
              f"{sum(i[3] for i in info):.0f} nonzeros")
        print(f"  mesh read {t_mesh:.1f}s, matrix assembly {t_asm:.1f}s")
        # Partition balance matters: one overloaded rank sets the pace for every step.
        cl = np.array([i[1] for i in info], dtype=float)
        print(f"  partition: cells/rank min {cl.min():.0f} max {cl.max():.0f} "
              f"(imbalance {cl.max()/max(cl.mean(),1):.2f}x)")
        print(f"  MatMult {per*1e3:.3f} ms   ->  a 163680-step solve would spend "
              f"{per*163680/3600:.2f} h in mat-vec alone")
        if size == 1 and np.isfinite(t_scipy):
            print(f"  scipy CSR mat-vec (what the solver ACTUALLY uses today) "
                  f"{t_scipy*1e3:.3f} ms")
            print(f"  -> PETSc serial is {per/t_scipy:.2f}x the scipy time, so the honest "
                  f"parallel baseline is scipy, not PETSc serial")
        print(f"MATMULT_MS {size} {per*1e3:.4f}")


if __name__ == "__main__":
    main()
