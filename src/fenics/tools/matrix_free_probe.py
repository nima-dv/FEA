r"""
Is a MATRIX-FREE stiffness operator actually faster for us? Prototype, verify, then measure.

WHY THIS EXISTS
The time loop's cost is one sparse mat-vec per step. Measured on the production mesh
(1.47M degree-4 DOF, 105.4M nonzeros): 51.6 ms with scipy, sustaining ~24 GB/s, i.e. this
machine's memory bandwidth. The matrix is 1.26 GB and gets streamed from DRAM every step, so
adding cores cannot help much - measured MPI speed-up was only 1.71x.

A matrix-free spectral-element operator never builds that matrix. It evaluates the stiffness
action element by element with the small 1-D GLL derivative matrix, which stays in cache, and
touches only the degrees of freedom and the geometric factors. Traffic drops from 1.26 GB to
tens of MB per step. This is standard practice for high-order elements and is exactly what
SPECFEM2D does.

The claim "3-5x" was previously in our management brief as an ESTIMATE. Estimating is what
produced the wrong MPI number, so this script measures it instead.

CORRECTNESS FIRST
A fast wrong operator is worthless, so this verifies against an assembled matrix before timing
anything. One subtlety makes that comparison meaningful: the matrix-free operator uses
COLLOCATED GLL quadrature (quadrature points = nodes), which is the standard spectral-element
choice and the same assumption that makes our lumped mass matrix consistent. Our production K
is assembled with FFCx's default, more accurate rule, so the two are genuinely different
operators. We therefore assemble a GLL-quadrature reference here and check against that - and
the difference between the two rules is itself reported, because adopting matrix-free would
mean adopting GLL quadrature.

RUN
  ./run.ps1 python3 tools/matrix_free_probe.py
  ./run.ps1 python3 tools/matrix_free_probe.py --degree 4 --mesh results/ili_mesh/ili_mesh_s0p8.msh
"""
from __future__ import annotations

import argparse
import time

import basix
import numpy as np
import scipy.sparse as sp
import ufl
from dolfinx.fem import functionspace, form
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import gmsh as dgmsh
from mpi4py import MPI

C_P, C_S, RHO_S = 5700.0, 3100.0, 7850.0
C_F, RHO_F = 1500.0, 1000.0
MU_S, LAM_S = RHO_S * C_S**2, RHO_S * (C_P**2 - 2 * C_S**2)
LAM_F = RHO_F * C_F**2
TAG_STEEL = 2


def tensor_perm(degree: int) -> np.ndarray:
    """Map DOLFINx's scalar dof ordering on a quad to lexicographic (i_xi, j_eta) order.

    DOLFINx orders cell dofs as vertices, then edges, then interior - NOT lexicographically.
    Sum factorisation needs the tensor-product layout, so we recover the permutation from the
    element's own node coordinates rather than hard-coding a pattern that would silently break
    at a different degree.
    """
    el = basix.create_element(basix.ElementFamily.P, basix.CellType.quadrilateral, degree,
                              basix.LagrangeVariant.gll_warped)
    pts = el.points
    n = degree + 1
    # round before sorting: node coordinates carry float noise that would scramble the keys
    key = np.round(pts, 10)
    order = np.lexsort((key[:, 0], key[:, 1]))          # xi fastest, eta slowest
    assert order.size == n * n
    return order


def gll_1d(degree: int):
    """1-D GLL points, weights, and the derivative matrix D[i, k] = dphi_k/dxi at point i."""
    n = degree + 1
    el = basix.create_element(basix.ElementFamily.P, basix.CellType.interval, degree,
                              basix.LagrangeVariant.gll_warped)
    pts = el.points[:, 0]
    order = np.argsort(pts)
    pts = pts[order]
    tab = el.tabulate(1, pts.reshape(-1, 1))            # (deriv, npts, ndofs, 1)
    D = tab[1, :, :, 0][np.ix_(np.arange(n), order)]
    # GLL weights on [0,1] from the quadrature rule of matching exactness
    qp, qw = basix.make_quadrature(basix.CellType.interval, 2 * degree - 1,
                                   basix.QuadratureType.gll)
    w = qw[np.argsort(qp[:, 0])]
    return pts, w, D


class MatrixFree:
    """y = K u for 2-D isotropic elasticity, evaluated by sum factorisation."""

    def __init__(self, domain, V, degree, lam_c, mu_c):
        self.n = degree + 1
        _, self.w, self.D = gll_1d(degree)
        perm = tensor_perm(degree)

        # --- element dof indices in tensor-product order, for the BLOCKED vector space ------
        dm = V.dofmap.list                              # (ne, ndofs_scalar)
        self.ne = dm.shape[0]
        sc = dm[:, perm]                                # (ne, n*n) scalar dofs, lexicographic
        bs = V.dofmap.index_map_bs
        self.idx = (bs * sc[:, :, None] + np.arange(bs)[None, None, :]) \
            .reshape(self.ne, self.n, self.n, bs)

        # --- geometry: J^-1 and detJ at every GLL point of every element -------------------
        # Tabulate the (bilinear) geometry element's derivatives at the GLL points, so this
        # works for any straight-sided quad without assuming a vertex ordering.
        gdm = domain.geometry.dofmap.reshape(self.ne, -1)
        gx = domain.geometry.x[:, :2]
        gel = basix.create_element(basix.ElementFamily.P, basix.CellType.quadrilateral, 1,
                                   basix.LagrangeVariant.gll_warped)
        gp = tensor_perm(1)
        pts1d, _, _ = gll_1d(degree)
        qp = np.stack(np.meshgrid(pts1d, pts1d, indexing="ij"), axis=-1).reshape(-1, 2)
        gt = gel.tabulate(1, qp)                        # (3, nq, 4, 1)
        # Index in two unambiguous steps. Mixing a fancy index with slices in one expression
        # silently transposes the result, which is how this first produced a negative Jacobian.
        dNdxi = gt[1, :, :, 0][:, gp]                   # (nq, 4), columns in tensor order
        dNdeta = gt[2, :, :, 0][:, gp]
        xe = gx[gdm]                                    # (ne, 4, 2)
        J = np.empty((self.ne, qp.shape[0], 2, 2))
        J[:, :, :, 0] = np.einsum("qa,ead->eqd", dNdxi, xe)
        J[:, :, :, 1] = np.einsum("qa,ead->eqd", dNdeta, xe)
        det = J[:, :, 0, 0] * J[:, :, 1, 1] - J[:, :, 0, 1] * J[:, :, 1, 0]
        n_neg = int((det <= 0).sum())
        if n_neg:
            print(f"  note: {n_neg} of {det.size} quadrature points have detJ <= 0 "
                  f"(cell orientation); using |detJ|, which is what the integral needs")
        det = np.abs(det)
        assert det.min() > 0, "a zero Jacobian: degenerate cell in the mesh"
        Jinv = np.empty_like(J)
        Jinv[:, :, 0, 0] = J[:, :, 1, 1] / det
        Jinv[:, :, 1, 1] = J[:, :, 0, 0] / det
        Jinv[:, :, 0, 1] = -J[:, :, 0, 1] / det
        Jinv[:, :, 1, 0] = -J[:, :, 1, 0] / det
        s = (self.ne, self.n, self.n)
        self.Jinv = Jinv.reshape(*s, 2, 2)
        # fold quadrature weights and |detJ| into one factor, once
        self.wdet = det.reshape(*s) * self.w[None, :, None] * self.w[None, None, :]
        self.lam = lam_c[:, None, None]
        self.mu = mu_c[:, None, None]
        self.ndof = V.dofmap.index_map.size_local * bs

    def __call__(self, u: np.ndarray) -> np.ndarray:
        D, Ji, wd = self.D, self.Jinv, self.wdet
        U = u[self.idx]                                             # (ne, n, n, 2)
        # reference gradients by sum factorisation
        gxi = np.einsum("ik,ekjc->eijc", D, U)
        get = np.einsum("jk,eikc->eijc", D, U)
        GR = np.stack((gxi, get), axis=-1)                          # (ne,n,n,2comp,2ref)
        grad = np.einsum("eijcr,eijrd->eijcd", GR, Ji)              # physical gradient
        epsm = 0.5 * (grad + np.swapaxes(grad, -1, -2))
        tr = epsm[..., 0, 0] + epsm[..., 1, 1]
        sig = 2.0 * self.mu[..., None, None] * epsm
        sig[..., 0, 0] += self.lam * tr
        sig[..., 1, 1] += self.lam * tr
        S = sig * wd[..., None, None]
        T = np.einsum("eijcd,eijrd->eijcr", S, Ji)                  # back to reference
        R = (np.einsum("ki,ekjc->eijc", D, T[..., 0])
             + np.einsum("kj,eikc->eijc", D, T[..., 1]))
        y = np.zeros(self.ndof)
        np.add.at(y, self.idx, R)                                   # scatter-add
        return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="results/ili_mesh/ili_mesh_s0p8.msh")
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    res = dgmsh.read_from_msh(args.mesh, MPI.COMM_WORLD, gdim=2)
    domain, ct = (res.mesh, res.cell_tags) if not isinstance(res, tuple) else (res[0], res[1])
    el = basix.ufl.element("Lagrange", domain.basix_cell(), args.degree,
                           lagrange_variant=basix.LagrangeVariant.gll_warped, shape=(2,))
    V = functionspace(domain, el)
    Q = functionspace(domain, ("DG", 0))
    ne = domain.topology.index_map(domain.topology.dim).size_local
    ndof = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    print(f"mesh {args.mesh}\n  {ne} cells, degree {args.degree}, {ndof} dofs")

    from dolfinx.fem import Function
    lam, mu = Function(Q), Function(Q)
    lam.x.array[:] = LAM_F
    mu.x.array[:] = 0.0
    steel = ct.find(TAG_STEEL)
    lam.x.array[steel] = LAM_S
    mu.x.array[steel] = MU_S

    u_t, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))
    sig = lambda w: lam * ufl.tr(eps(w)) * ufl.Identity(2) + 2 * mu * eps(w)

    # Reference operator with COLLOCATED GLL quadrature - the same rule the matrix-free
    # operator uses, so agreement should be to machine precision.
    dx_gll = ufl.dx(metadata={"quadrature_rule": "GLL",
                              "quadrature_degree": 2 * args.degree - 1})
    t0 = time.time()
    A = assemble_matrix(form(ufl.inner(sig(u_t), eps(v)) * dx_gll))
    A.assemble()
    ai, aj, av = A.getValuesCSR()
    K_gll = sp.csr_matrix((av, aj, ai), shape=A.getSize())
    t_asm = time.time() - t0
    print(f"  GLL-quadrature matrix: {K_gll.nnz} nnz, assembled in {t_asm:.1f}s "
          f"({K_gll.data.nbytes/1e9:.2f} GB of values alone)")

    mf = MatrixFree(domain, V, args.degree, lam.x.array[:ne].copy(), mu.x.array[:ne].copy())

    rng = np.random.default_rng(0)
    u = rng.standard_normal(ndof)
    y_ref, y_mf = K_gll @ u, mf(u)
    rel = np.linalg.norm(y_mf - y_ref) / np.linalg.norm(y_ref)
    print(f"\nCORRECTNESS  ||free - assembled|| / ||assembled|| = {rel:.3e}")
    if rel > 1e-10:
        print("  FAILED - the operator is wrong; the timings below are meaningless.")
    else:
        print("  PASS (machine precision) - same operator, different evaluation.")

    # How different is GLL quadrature from the default rule our production K uses?
    A2 = assemble_matrix(form(ufl.inner(sig(u_t), eps(v)) * ufl.dx))
    A2.assemble()
    ai, aj, av = A2.getValuesCSR()
    K_def = sp.csr_matrix((av, aj, ai), shape=A2.getSize())
    d = np.linalg.norm(K_def @ u - y_ref) / np.linalg.norm(y_ref)
    print(f"  (default-quadrature operator differs from GLL by {d:.3e} relative - adopting "
          f"matrix-free means adopting GLL, which is a real change to verify)")

    def timeit(f, n):
        f(u)
        t0 = time.time()
        for _ in range(n):
            f(u)
        return (time.time() - t0) / n

    t_csr = timeit(lambda z: K_def @ z, args.reps)
    t_free = timeit(mf, args.reps)
    print(f"\nSPEED (per mat-vec, mean of {args.reps})")
    print(f"  scipy CSR, default quadrature   {t_csr*1e3:8.2f} ms   <- what we run today")
    print(f"  matrix-free, sum factorisation  {t_free*1e3:8.2f} ms")
    print(f"  speed-up {t_csr/t_free:.2f}x")
    traffic_csr = (K_def.data.nbytes + K_def.indices.nbytes) / 1e9
    traffic_free = (u.nbytes + mf.Jinv.nbytes + mf.wdet.nbytes) / 1e9
    print(f"  memory traffic per apply: CSR ~{traffic_csr:.2f} GB vs free ~{traffic_free:.3f} GB"
          f"  ({traffic_csr/traffic_free:.0f}x less)")
    print(f"\n163680-step solve, mat-vec only: {t_csr*163680/3600:.2f} h -> "
          f"{t_free*163680/3600:.2f} h")


if __name__ == "__main__":
    main()
