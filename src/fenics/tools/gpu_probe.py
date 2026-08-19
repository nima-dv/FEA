r"""
Feasibility probe: is the leapfrog time loop worth moving to the GPU, and by how much?

WHY THIS PROBE EXISTS RATHER THAN A DIRECT PORT
-----------------------------------------------
Two performance estimates in this project were wrong and were corrected by measurement:
MPI was estimated at 4-8x and measured at 1.71x, and a matrix-free prototype was expected
to be faster and came out 10x SLOWER. So the GPU projection gets measured before anything
is ported.

WHAT IS ACTUALLY BEING TESTED
-----------------------------
repro/ili_forward.py's hot loop touches neither PETSc nor DOLFINx:

    rhs   = tm * u_cur - b_co * u_old - (Ks @ u_cur)
    u_new = inv_a * rhs

DOLFINx assembles Ks, m, S and W once; the loop is then scipy.sparse and numpy only. So
the GPU question is not "does FEniCS support CUDA" (it does not, natively, and the
cuda-dolfinx route is blocked because the official Docker/Conda dolfinx ships PETSc without
CUDA - verified: PETSc.Vec().setType('cuda') fails with error 86). The question is whether
a cupy/cuSPARSE swap of those two lines pays.

The measured CPU rate is almost exactly bandwidth-bound - roughly 1.26 GB moved per step at
about 24 GB/s gives the observed ~52 ms/step - so the projection is a bandwidth ratio, and
that is what this script measures rather than assumes.

HONEST SCOPE
------------
  * K is assembled on the real production mesh, so the SIZE and SPARSITY PATTERN are exact.
    Material constants are set uniform, because bandwidth depends on nnz and dtype, not on
    the values. This is a throughput measurement, not a physics run.
  * The per-step receiver extraction (W) and source injection (S) are excluded: together
    they are far under 1% of the nonzeros, so they cannot move the result.
  * Accuracy is checked here only as agreement between CPU and GPU on one step. That is NOT
    sufficient to adopt the GPU path - cuSPARSE reduces in a different order, so agreement
    will be near fp64 round-off rather than exact, and adoption needs validation against
    KNOWN ARRIVAL TIMES, not vector norms.

RUN
  docker run --rm --gpus all -v ...:/work -w /work dvfenics:gpu python3 tools/gpu_probe.py
  (add --cpu-only to run the CPU half on the non-GPU image)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as ss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repro"))
MESH = ROOT / "results" / "ili_mesh" / "ili_mesh_s0p8.msh"
DEG = 4


def assemble_K():
    """Assemble the stiffness matrix on the production mesh. Pattern-exact, values uniform."""
    import basix.ufl
    import ufl
    from dolfinx.fem import functionspace, form
    from dolfinx.fem.petsc import assemble_matrix
    # Reuse the solver's own loader rather than a copy of it, so this probe cannot drift
    # from the mesh the production run actually reads.
    from ili_forward import load_mesh

    t0 = time.time()
    domain, _, _ = load_mesh(MESH)
    el = basix.ufl.element("Lagrange", domain.basix_cell(), DEG,
                           lagrange_variant=basix.LagrangeVariant.gll_warped, shape=(2,))
    V = functionspace(domain, el)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))
    # Uniform steel. Values are irrelevant to throughput; the pattern is what matters.
    lam, mu = 1.0e11, 7.5e10
    sig = lambda w: lam * ufl.tr(eps(w)) * ufl.Identity(2) + 2 * mu * eps(w)
    K = assemble_matrix(form(ufl.inner(sig(u), eps(v)) * ufl.dx))
    K.assemble()
    if hasattr(K, "to_scipy"):
        Ks = K.to_scipy()
    else:
        ai, aj, av = K.getValuesCSR()
        Ks = ss.csr_matrix((av, aj, ai), shape=K.getSize())
    Ks = ss.csr_matrix(Ks, dtype=np.float64)
    # Normalise the values. Throughput depends on nnz and dtype, not on magnitudes, and the
    # synthetic recurrence below is NOT the real scheme - with raw stiffness values of order
    # 1e11 it diverges to inf within a few dozen steps and every comparison becomes nan.
    Ks.data /= np.abs(Ks.data).max()
    print(f"assembled in {time.time()-t0:.1f} s: "
          f"{Ks.shape[0]} rows, {Ks.nnz} nonzeros ({Ks.nnz/1e6:.1f} M), "
          f"{Ks.nnz/Ks.shape[0]:.1f} per row")
    return Ks


def step_bytes(Ks) -> float:
    """Bytes the leapfrog step must move, for the bandwidth figure.

    CSR: 8 B per value + 4 B per column index + 4 B per row pointer. Vectors: u_cur is read
    by the SpMV (and reused), u_old read, u_new written, and tm/b_co/inv_a read - counted as
    6 vector touches of 8 B, which is the optimistic end since caching helps some of them.
    """
    n = Ks.shape[0]
    return Ks.nnz * 12 + (n + 1) * 4 + 6 * n * 8


def bench_cpu(Ks, reps: int):
    n = Ks.shape[0]
    rng = np.random.default_rng(0)
    u_old = rng.standard_normal(n)
    u_cur = rng.standard_normal(n)
    tm, b_co, inv_a = (np.full(n, v) for v in (2.0, 1.0, 1.0))
    Ks @ u_cur                                     # warm up
    t0 = time.perf_counter()
    for _ in range(reps):
        rhs = tm * u_cur - b_co * u_old - (Ks @ u_cur)
        u_new = inv_a * rhs
        u_old, u_cur = u_cur, u_new
    dt = (time.perf_counter() - t0) / reps
    return dt, u_cur


def bench_gpu(Ks, reps: int):
    import cupy as cp
    import cupyx.scipy.sparse as csp

    n = Ks.shape[0]
    rng = np.random.default_rng(0)
    u_old_h = rng.standard_normal(n)
    u_cur_h = rng.standard_normal(n)
    tm_h, b_co_h, inv_a_h = (np.full(n, v) for v in (2.0, 1.0, 1.0))

    t0 = time.perf_counter()
    Kg = csp.csr_matrix(Ks)
    cp.cuda.Device().synchronize()
    upload = time.perf_counter() - t0
    used = cp.get_default_memory_pool().used_bytes()
    print(f"  matrix uploaded in {upload:.2f} s, device pool holds "
          f"{used/2**30:.2f} GiB of {cp.cuda.runtime.memGetInfo()[1]/2**30:.1f} GiB")

    u_old, u_cur = cp.asarray(u_old_h), cp.asarray(u_cur_h)
    tm, b_co, inv_a = cp.asarray(tm_h), cp.asarray(b_co_h), cp.asarray(inv_a_h)
    for _ in range(5):                             # warm up: JIT + cuSPARSE plan
        rhs = tm * u_cur - b_co * u_old - (Kg @ u_cur)
        u_new = inv_a * rhs
    cp.cuda.Device().synchronize()

    t0 = time.perf_counter()
    for _ in range(reps):
        rhs = tm * u_cur - b_co * u_old - (Kg @ u_cur)
        u_new = inv_a * rhs
        u_old, u_cur = u_cur, u_new
    cp.cuda.Device().synchronize()
    dt = (time.perf_counter() - t0) / reps
    return dt, cp.asnumpy(u_cur)


def check_agreement(Ks) -> float:
    """One step, identical inputs, CPU vs GPU.

    Measured on a SINGLE step deliberately. Comparing after a long synthetic recurrence
    measures how fast that recurrence amplifies round-off, not whether the two operators
    agree - which is the only thing this check is for.
    """
    import cupy as cp
    import cupyx.scipy.sparse as csp

    n = Ks.shape[0]
    rng = np.random.default_rng(1)
    u_old = rng.standard_normal(n)
    u_cur = rng.standard_normal(n)
    ref = 2.0 * u_cur - u_old - (Ks @ u_cur)
    Kg = csp.csr_matrix(Ks)
    got = cp.asnumpy(2.0 * cp.asarray(u_cur) - cp.asarray(u_old) - (Kg @ cp.asarray(u_cur)))
    return float(np.abs(ref - got).max() / max(np.abs(ref).max(), 1e-300))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40, help="timed steps per backend")
    ap.add_argument("--cpu-only", action="store_true", help="skip the GPU half")
    args = ap.parse_args()

    if not MESH.exists():
        raise SystemExit(f"missing {MESH} - build it with mesh/ili_mesh.py --scale 0.8 --quad")

    Ks = assemble_K()
    gb = step_bytes(Ks) / 1e9
    print(f"one leapfrog step moves about {gb:.3f} GB\n")

    print("CPU (scipy.sparse):")
    t_cpu, u_cpu = bench_cpu(Ks, args.reps)
    print(f"  {t_cpu*1e3:.2f} ms/step -> {gb/t_cpu:.1f} GB/s achieved")
    print(f"  163680 steps = {t_cpu*163680/3600:.2f} h per angle\n")

    if args.cpu_only:
        return

    print("GPU (cupy / cuSPARSE):")
    t_gpu, u_gpu = bench_gpu(Ks, args.reps)
    print(f"  {t_gpu*1e3:.3f} ms/step -> {gb/t_gpu:.1f} GB/s achieved")
    print(f"  163680 steps = {t_gpu*163680/60:.1f} min per angle\n")

    rel = check_agreement(Ks)
    print("=" * 68)
    print(f"SPEEDUP  {t_cpu/t_gpu:.1f}x   ({t_cpu*1e3:.2f} -> {t_gpu*1e3:.3f} ms/step)")
    print(f"agreement on ONE step from identical inputs: max rel deviation {rel:.3e}")
    print("  Round-off scale is expected and fine; cuSPARSE reduces in a different order.")
    print("  This is NOT an accuracy gate - adoption needs validation against known")
    print("  arrival times, not against a vector norm.")
    print("=" * 68)


if __name__ == "__main__":
    main()
