r"""
The GPU acceptance gate: does the GPU time loop produce trustworthy channel data?

WHY A GATE AND NOT A SPOT CHECK
-------------------------------
cuSPARSE reduces in a different order than scipy, so the GPU path is NOT bit-identical and
never will be. Round-off agreement on one step (measured at 2.6e-16 by tools/gpu_probe.py)
says the operators match; it says nothing about what 163,680 steps of that difference does
to an ARRIVAL TIME. Arrival time is the entire measurement in this project - depth is
inferred from it - so the gate is a timing gate, not a norm.

THRESHOLDS, PRE-REGISTERED BEFORE THE FIRST RUN
-----------------------------------------------
  PASS requires ALL of:
    1. per-element arrival-time shift  < 0.01 sample (0.026 ns at the 380 MHz base)
    2. max relative deviation          < 1e-6
    3. relative L2 error               < 1e-8
  FAIL on any of:
    - arrival-time shift  > 0.1 sample, or
    - max relative deviation > 1e-3
  Anything between PASS and FAIL is MARGINAL: report, do not adopt, investigate.

Criterion 1 is the one that matters. A solver can agree to 1e-6 in amplitude and still be
late, and late is a depth error.

USAGE
  python3 tools/gpu_gate.py --a results/ili_forward/channel_data_gate_cpu_3us.npz \
                           --b results/ili_forward/channel_data_gate_gpu_3us.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TOL_SHIFT_PASS = 0.01      # samples
TOL_SHIFT_FAIL = 0.10
TOL_REL_PASS = 1e-6
TOL_REL_FAIL = 1e-3
TOL_L2_PASS = 1e-8


def subsample_lag(a: np.ndarray, b: np.ndarray) -> float:
    """Lag of b relative to a, in samples, refined to sub-sample by parabolic fit.

    Full cross-correlation rather than a first-break pick: a threshold crossing moves by a
    whole sample as soon as noise nudges it past the threshold, which would hide exactly the
    sub-sample drift this gate is looking for.
    """
    a = a - a.mean()
    b = b - b.mean()
    n = 1 << int(np.ceil(np.log2(len(a) * 2)))
    A = np.fft.rfft(a, n)
    B = np.fft.rfft(b, n)
    xc = np.fft.irfft(A * np.conj(B), n)
    xc = np.concatenate((xc[-(len(a) - 1):], xc[:len(a)]))     # lags -(N-1) .. +(N-1)
    k = int(np.argmax(np.abs(xc)))
    if k == 0 or k == len(xc) - 1:
        return float(k - (len(a) - 1))
    y0, y1, y2 = np.abs(xc[k - 1]), np.abs(xc[k]), np.abs(xc[k + 1])
    denom = y0 - 2 * y1 + y2
    frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    return float((k - (len(a) - 1)) + frac)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference channel_data npz (CPU)")
    ap.add_argument("--b", required=True, help="channel_data npz under test (GPU)")
    ap.add_argument("--min-energy", type=float, default=1e-3,
                    help="skip elements whose energy is below this fraction of the loudest; "
                         "a silent trace has no arrival time to compare (default 1e-3)")
    args = ap.parse_args()

    da, db = np.load(args.a), np.load(args.b)
    A, B = da["channel_data"], db["channel_data"]
    if A.shape != B.shape:
        raise SystemExit(f"shape mismatch {A.shape} vs {B.shape} - not the same configuration")
    dt = float(da["dt"])
    print(f"comparing {Path(args.a).name}\n      vs  {Path(args.b).name}")
    print(f"shape {A.shape}, dt {dt*1e9:.4f} ns, solver dt "
          f"{float(da['dt_solver'])*1e9:.4f} / {float(db['dt_solver'])*1e9:.4f} ns")
    if abs(float(da["dt_solver"]) - float(db["dt_solver"])) > 1e-18:
        print("  WARNING: solver time steps differ - this is not a backend-only comparison")

    # --- amplitude agreement ----------------------------------------------------------
    scale = np.abs(A).max()
    max_rel = float(np.abs(A - B).max() / scale)
    l2_rel = float(np.linalg.norm(A - B) / np.linalg.norm(A))
    print(f"\namplitude:  max rel deviation {max_rel:.3e}   relative L2 {l2_rel:.3e}")

    # --- timing agreement, the criterion that matters ---------------------------------
    energy = (A ** 2).sum(axis=0)
    live = np.where(energy > args.min_energy * energy.max())[0]
    lags = np.array([subsample_lag(A[:, e], B[:, e]) for e in live])
    worst = int(np.argmax(np.abs(lags)))
    print(f"timing:     {live.size} of {A.shape[1]} elements carry signal")
    print(f"            max |shift| {np.abs(lags).max():.3e} samples "
          f"= {np.abs(lags).max()*dt*1e12:.3f} ps  (element {live[worst]})")
    print(f"            mean shift  {lags.mean():+.3e} samples "
          f"(a systematic lead/lag would show up here)")

    # --- verdict ----------------------------------------------------------------------
    shift = float(np.abs(lags).max())
    fail = shift > TOL_SHIFT_FAIL or max_rel > TOL_REL_FAIL
    passed = (shift < TOL_SHIFT_PASS and max_rel < TOL_REL_PASS and l2_rel < TOL_L2_PASS)
    print("\n" + "=" * 70)
    print(f"{'criterion':<34}{'measured':>14}{'threshold':>12}  verdict")
    for name, val, tol, ok in (
        ("arrival-time shift [samples]", shift, TOL_SHIFT_PASS, shift < TOL_SHIFT_PASS),
        ("max relative deviation", max_rel, TOL_REL_PASS, max_rel < TOL_REL_PASS),
        ("relative L2 error", l2_rel, TOL_L2_PASS, l2_rel < TOL_L2_PASS),
    ):
        print(f"{name:<34}{val:>14.3e}{tol:>12.0e}  {'pass' if ok else 'MISS'}")
    print("=" * 70)
    print("GATE: " + ("FAIL - do not use the GPU path" if fail else
                      "PASS" if passed else
                      "MARGINAL - report, do not adopt, investigate"))
    print("=" * 70)


if __name__ == "__main__":
    main()
