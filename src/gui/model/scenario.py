r"""Read-only scenario facts: geometry, array, materials, notch, mesh size targets.

WHERE THESE NUMBERS COME FROM, AND WHY NOT FROM HERE
The backend already defines them three times over - mesh/ili_mesh.py (geometry, cell sizes),
repro/ili_forward.py (materials, source, time base) and lib/tt_t_image.py's FROZEN (what the
imaging chain assumes). A fourth copy in the GUI would drift, and a GUI that draws a different
pipe than the solver meshes is worse than a GUI with no drawing at all. So we run
`tools/scenario_dump.py` INSIDE the container, which imports those modules and prints their
values as JSON, and cache the result next to this file so the app starts instantly and offline.

Lengths are millimetres (the drawing's own unit, and what the mesh script uses); speeds,
densities, frequencies and times are SI.

Nothing here is editable in the UI. Geometry and materials are backend constants, not flags -
see the GUI README's "deferred by explicit decision".
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, fields
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[3]                      # model -> gui -> src -> repo
CACHE = _HERE.parent / "scenario_cache.json"
DUMP_ARGV = ["python3", "tools/scenario_dump.py"]


@dataclass(frozen=True)
class Scenario:
    """One frozen snapshot of the benchmark scenario. mm unless the name says otherwise."""
    # --- pipe
    r_id: float
    r_od: float
    wall: float
    x_c: float
    z_c: float
    standoff: float
    # --- array
    n_elem: int
    pitch: float
    kerf: float
    aperture: float
    array_x0: float
    f0: float                # Hz
    n_cycle: int             # 1 cycle: that is where the 100% bandwidth comes from
    # --- materials (SI)
    c_p: float
    c_s: float
    rho_s: float
    c_f: float
    rho_f: float
    # --- notch
    notch_x: float
    notch_depth: float
    notch_width: float
    # --- mesh size targets, BEFORE --scale
    h_water: float
    h_steel: float
    h_notch: float
    h_array: float
    h_stair: float
    # --- domain
    x_min: float
    x_max: float
    x_limit_lo: float        # the mesh script REFUSES outside these: past here the pipe arc
    x_limit_hi: float        # rises above the flat array plane and the water region is wrong
    # --- time base
    t_end: float             # s
    cfl: float
    # --- provenance, shown in the UI so nobody trusts a fallback by accident
    source: str = "fallback"

    @property
    def array_x1(self) -> float:
        """Far end of the aperture, mm. widgets/crosssection.py asks for it by this name."""
        return self.array_x0 + self.aperture

    @property
    def f_upper(self) -> float:
        """Upper usable frequency of a 1-cycle burst: ~2 f0.

        The mesh must resolve THIS, not f0 - a 1-cycle 4 MHz pulse carries real energy to
        6-8 MHz, and a mesh sized for the centre frequency low-passes your own pulse in
        transit (docs/lessons.md, "the bandwidth trap").
        """
        return 2.0 * self.f0


# --------------------------------------------------------------------------------------
# LAST-RESORT FALLBACK. Used only when the container cannot be reached AND no cache exists.
# NEVER EDIT THESE TO CHANGE BEHAVIOUR. They exist so the window can open and say "backend
# unreachable", not so anyone can retune the app: the backend owns every one of these
# numbers, and editing a copy here only makes the GUI disagree with the solver silently.
# Verified equal to the container dump on 2026-08-20.
# --------------------------------------------------------------------------------------
FALLBACK = Scenario(
    r_id=193.675, r_od=203.200, wall=9.525, x_c=38.25, z_c=-173.675, standoff=20.0,
    n_elem=256, pitch=0.30, kerf=0.05, aperture=76.5, array_x0=0.0,
    f0=4.0e6, n_cycle=1,
    c_p=5700.0, c_s=3100.0, rho_s=7850.0, c_f=1500.0, rho_f=1000.0,
    notch_x=38.25, notch_depth=4.0, notch_width=1.0,
    h_water=0.28, h_steel=0.56, h_notch=0.30, h_array=0.09, h_stair=0.05,
    x_min=-8.0, x_max=85.0, x_limit_lo=-47.46464285639881, x_limit_hi=123.96464285639881,
    t_end=60.0e-6, cfl=0.30, source="fallback",
)


def from_dump(d: dict, source: str = "container") -> Scenario:
    """Flatten scenario_dump.py's grouped JSON into the dataclass the widgets want."""
    g, a, m, n, h, x, t = (d["geometry"], d["array"], d["materials"], d["notch"],
                           d["mesh_h"], d["domain"], d["time"])
    if not d.get("frozen_agrees", True):
        # The dump cross-checks the mesh against lib/tt_t_image.FROZEN. If they disagree,
        # measured depths are against the wrong reference and no GUI number means anything.
        raise ValueError("scenario_dump: mesh and imaging chain disagree on the scenario")
    return Scenario(
        r_id=g["r_id"], r_od=g["r_od"], wall=g["wall"], x_c=g["x_c"], z_c=g["z_c"],
        standoff=g["standoff"],
        n_elem=a["n_elem"], pitch=a["pitch"], kerf=a["kerf"], aperture=a["aperture"],
        array_x0=a["x0"], f0=a["f0"], n_cycle=a["n_cycle"],
        c_p=m["c_P"], c_s=m["c_S"], rho_s=m["rho_s"], c_f=m["c_f"], rho_f=m["rho_f"],
        notch_x=n["x"], notch_depth=n["depth"], notch_width=n["width"],
        h_water=h["water"], h_steel=h["steel"], h_notch=h["notch"], h_array=h["array"],
        h_stair=h["stair"],
        x_min=x["x_min"], x_max=x["x_max"],
        x_limit_lo=x["x_limit_lo"], x_limit_hi=x["x_limit_hi"],
        t_end=t["t_end"], cfl=t["cfl"], source=source,
    )


def refresh(timeout: float = 300.0) -> Scenario:
    """Run the dump in the container and rewrite the cache. Raises if the container fails.

    Slow (image start plus a dolfinx import), which is exactly why the result is cached and
    why the UI only calls this from an explicit "refresh scenario" action, never at startup.
    """
    ps = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
          "-File", str(REPO / "docker" / "run.ps1")] + DUMP_ARGV
    out = subprocess.run(ps, capture_output=True, text=True, timeout=timeout, cwd=str(REPO))
    if out.returncode != 0:
        raise RuntimeError(f"scenario_dump failed ({out.returncode}): "
                           f"{(out.stderr or out.stdout).strip()[-400:]}")
    # The container prints only JSON, but a docker warning on stdout would still be ahead of
    # it, so slice from the first brace rather than trusting the whole stream.
    text = out.stdout[out.stdout.index("{"):]
    sc = from_dump(json.loads(text))
    CACHE.write_text(json.dumps(json.loads(text), indent=2), encoding="ascii")
    return sc


def load(allow_container: bool = False) -> Scenario:
    """Cache -> fallback. Never raises, and by default never touches Docker.

    allow_container defaults to FALSE because the UI calls this while building the window: a
    machine with Docker stopped would otherwise sit through a socket timeout before painting.
    These constants change only when the backend changes, which is a commit away and not a
    runtime event, so the cache is the right default and refresh() is the explicit way in.
    """
    if CACHE.exists():
        try:
            return from_dump(json.loads(CACHE.read_text(encoding="ascii")), source="cache")
        except Exception:
            pass                              # corrupt cache: fall through, do not crash
    if allow_container:
        try:
            return refresh()
        except Exception:
            pass
    return FALLBACK


def describe(sc: Scenario | None = None) -> tuple[tuple[str, str], ...]:
    """Rows for the read-only Scenario panel: (label, text), in reading order.

    Formatted here rather than in the view because the unit is part of the fact - a bare
    "193.675" invites the reader to guess metres, and the drawing is in millimetres.
    """
    s = sc or load()
    return (
        ("pipe", f"ID {2*s.r_id:.2f} / OD {2*s.r_od:.2f} mm, wall {s.wall:.3f} mm"),
        ("standoff", f"{s.standoff:.1f} mm of water on the beam axis"),
        ("array", f"{s.n_elem} elements, {s.pitch:.2f} mm pitch, "
                  f"{s.aperture:.1f} mm aperture"),
        ("source", f"{s.f0/1e6:.0f} MHz, {s.n_cycle} cycle "
                   f"(usable to ~{s.f_upper/1e6:.0f} MHz)"),
        ("steel", f"c_P {s.c_p:.0f} / c_S {s.c_s:.0f} m/s, rho {s.rho_s:.0f} kg/m3"),
        ("water", f"c {s.c_f:.0f} m/s, rho {s.rho_f:.0f} kg/m3"),
        ("notch", f"{s.notch_depth:.1f} mm deep x {s.notch_width:.1f} mm wide at "
                  f"x = {s.notch_x:.2f} mm"),
        ("domain", f"{s.x_min:+.1f} to {s.x_max:+.1f} mm (limit "
                   f"{s.x_limit_lo:+.2f} to {s.x_limit_hi:+.2f})"),
        ("record", f"{s.t_end*1e6:.0f} us at CFL {s.cfl:.2f}"),
        ("these facts came from", s.source),
    )


def demo() -> None:
    sc = load()
    # The fallback must agree with whatever the cache holds, or the two disagree silently the
    # day Docker is down. Tolerance, not equality: the dump computes wall = r_od - r_id, which
    # is 9.524999999999977 in binary, and the hand-written copy says 9.525.
    for f in fields(Scenario):
        a, b = getattr(sc, f.name), getattr(FALLBACK, f.name)
        if f.name == "source":
            continue
        assert abs(a - b) <= 1e-9 * max(1.0, abs(b)), (f.name, a, b)
    # Identities the whole scenario rests on. If any of these break, the drawing is wrong.
    assert abs((sc.r_id + sc.z_c) - sc.standoff) < 1e-9, "standoff identity"
    assert abs((sc.r_od - sc.r_id) - sc.wall) < 1e-9, "wall thickness"
    assert abs((sc.n_elem - 1) * sc.pitch - sc.aperture) < 1e-9, "aperture != 255*pitch"
    assert sc.notch_depth < sc.wall, "a notch deeper than the wall is not a notch"
    assert sc.x_limit_lo < sc.x_min < sc.x_max < sc.x_limit_hi
    assert abs(sc.f_upper - 8.0e6) < 1.0, "1-cycle 4 MHz burst: size the mesh at 8 MHz"
    assert abs(sc.array_x1 - 76.5) < 1e-9, sc.array_x1
    rows = describe(sc)
    assert len(rows) >= 8 and all(len(r) == 2 and r[1] for r in rows)
    assert "9.525" in dict(rows)["pipe"], dict(rows)["pipe"]
    # Water is the slower material, so its wavelength is the shorter one - the reason the
    # water gap is meshed FINER than the steel wall even though the crack is in the steel.
    assert sc.c_f < sc.c_s < sc.c_p and sc.h_water < sc.h_steel
    print(f"scenario.demo: ok (source={sc.source})")


if __name__ == "__main__":
    demo()
