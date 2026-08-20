r"""
Self-checks for the absorbing-boundary work. Pure numpy - no FEniCS, no container needed.

    python3 tests/test_abc.py

Two things are checked, both of which would otherwise fail silently inside a 2.4 h solve:

  1. The dashpot impedance algebra, including the size of the bug that was fixed.
  2. The sponge's d_max scaling, by ACTUALLY simulating a 1-D wave hitting the layer and
     measuring what comes back. The formula in ili_forward.py claims a round-trip attenuation
     of --sponge-db; this measures it rather than trusting the derivation.

Nothing here depends on the steering angle or on the ILI geometry.
"""
from __future__ import annotations

import numpy as np

C_P, C_S, RHO = 5700.0, 3100.0, 7850.0


def dashpot_reflection(c_true: float, c_dashpot: float) -> float:
    """|R| for a wave of speed c_true meeting a dashpot tuned to c_dashpot, normal incidence."""
    z_true, z_damp = RHO * c_true, RHO * c_dashpot
    return abs(z_true - z_damp) / (z_true + z_damp)


def test_dashpot_impedance() -> None:
    # Matched: a dashpot on the right wave is perfect at normal incidence.
    assert dashpot_reflection(C_S, C_S) < 1e-12
    assert dashpot_reflection(C_P, C_P) < 1e-12

    # The bug: shear wave meeting a c_P-tuned dashpot. This is why the fix exists.
    r = dashpot_reflection(C_S, C_P)
    assert abs(r - 0.2955) < 1e-3, r
    db = 20 * np.log10(r)
    assert -10.7 < db < -10.5, db
    print(f"  dashpot: shear vs c_P-tuned -> |R| {r:.4f} ({db:+.1f} dB)   [the bug]")
    print(f"  dashpot: shear vs c_S-tuned -> |R| {dashpot_reflection(C_S, C_S):.1e}   [fixed]")


def sponge_d_max(rho: float, c: float, db: float, L: float) -> float:
    """The formula used in ili_forward.py. Quadratic ramp, round-trip target."""
    return 0.34539 * rho * c * db / L


def measure_sponge(db_target: float, L: float, c: float, ppw: float = 40.0,
                   n_layer: int = 400) -> float:
    """1-D wave into a graded sponge backed by a rigid wall; return measured round-trip |R|.

    Explicit central differences on rho*u_tt = E*u_xx - d(x)*u_t, with d ramping as s^2 over
    the layer. A Gaussian pulse travels in, reflects off the hard end, and comes back; we
    compare the returned amplitude with the incident one at the same probe point.
    """
    E = rho_ = None
    rho_ = RHO
    E = rho_ * c * c                      # so that sqrt(E/rho) == c
    lam = L / 6.0                         # pulse wavelength: several fit inside the layer
    dx = lam / ppw
    n_free = int(6 * lam / dx)            # free region ahead of the layer
    n = n_free + n_layer
    dx = L / n_layer                      # make the layer exactly L wide
    lam = ppw * dx
    n_free = int(8 * lam / dx)
    n = n_free + n_layer

    d = np.zeros(n + 1)
    s = np.linspace(0.0, 1.0, n_layer + 1)
    d[n_free:] = sponge_d_max(rho_, c, db_target, L) * s**2

    dt = 0.4 * dx / c                     # comfortably inside the CFL limit
    u_old = np.zeros(n + 1)
    u = np.zeros(n + 1)

    # Gaussian pulse launched in the free region, travelling +x.
    x = np.arange(n + 1) * dx
    x0 = 3 * lam
    width = lam / 3.0
    u = np.exp(-((x - x0) / width) ** 2)
    u_old = np.exp(-((x - (x0 - c * dt)) / width) ** 2)

    probe = n_free // 2
    trace = []
    # long enough for the pulse to cross the free region, traverse the layer twice, and return
    nsteps = int(6 * (n_free + 2 * n_layer) * dx / (c * dt))
    for _ in range(nsteps):
        lap = np.zeros_like(u)
        lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        a = 1.0 / (rho_ / dt**2 + d / (2 * dt))
        b = rho_ / dt**2 - d / (2 * dt)
        u_new = a * (E * lap + 2 * rho_ * u / dt**2 - b * u_old)
        u_new[0] = 0.0                    # rigid: incident pulse is launched away from it
        u_new[-1] = 0.0                   # rigid wall behind the sponge - worst case
        u_old, u = u, u_new
        trace.append(u[probe])

    tr = np.abs(np.asarray(trace))
    # First arrival is the incident pulse; the last big feature is what the sponge let back.
    half = len(tr) // 3
    incident = tr[:half].max()
    returned = tr[half:].max()
    return returned / incident


def test_sponge_scaling() -> None:
    L = 8.0e-3
    for db in (20.0, 40.0):
        meas = measure_sponge(db, L, C_S)
        want = 10 ** (-db / 20)
        print(f"  sponge {L*1e3:.0f} mm, target -{db:.0f} dB: "
              f"measured |R| {meas:.2e} (want <= {want:.2e}, "
              f"{20*np.log10(max(meas,1e-12)):+.1f} dB)")
        # The derivation is a ray-theory estimate, so allow an order of magnitude, but it
        # must be in the right ballpark AND must improve when we ask for more attenuation.
        assert meas < want * 12.0, f"sponge far weaker than designed: {meas:.2e} vs {want:.2e}"
    strong = measure_sponge(40.0, L, C_S)
    weak = measure_sponge(20.0, L, C_S)
    assert strong < weak, f"asking for more dB did not help: {strong:.2e} !< {weak:.2e}"


def test_more_damping_is_not_better() -> None:
    """The default must sit at the optimum, not past it.

    Damping strong enough to stop the wave makes the layer effectively RIGID, and a rigid
    layer reflects. Measured: -40 dB target is best (about -55 dB achieved); -60 gives -47,
    -200 gives -43. This guards the default against being raised 'to be safe'.
    """
    L = 8.0e-3
    best = measure_sponge(40.0, L, C_S)
    for db in (60.0, 200.0):
        worse = measure_sponge(db, L, C_S)
        assert worse > best, (
            f"expected target -{db:.0f} dB to be WORSE than -40 dB (over-damping), "
            f"got {20*np.log10(worse):+.1f} vs {20*np.log10(best):+.1f} dB")
    print(f"  over-damping confirmed: -40 dB target is the optimum "
          f"({20*np.log10(best):+.1f} dB achieved); raising it degrades")


def test_sponge_is_gradual() -> None:
    """A sponge must start at zero damping. A step in d() reflects, which is the whole point."""
    L = 8.0e-3
    d_max = sponge_d_max(RHO, C_S, 60.0, L)
    s = np.linspace(0.0, 1.0, 101)
    d = d_max * s**2
    assert d[0] == 0.0
    assert d[1] / d_max < 1e-3, "ramp starts too steeply to be reflection-free"
    assert np.all(np.diff(d) >= 0.0), "ramp must be monotone"
    print(f"  ramp: d(0) = 0, d(L) = {d_max:.3e} kg/(m^3 s), monotone")


if __name__ == "__main__":
    print("dashpot impedance:")
    test_dashpot_impedance()
    print("sponge ramp shape:")
    test_sponge_is_gradual()
    print("sponge attenuation, measured on a 1-D wave:")
    test_sponge_scaling()
    print("over-damping guard:")
    test_more_damping_is_not_better()
    print("\nall absorbing-boundary self-checks passed")
