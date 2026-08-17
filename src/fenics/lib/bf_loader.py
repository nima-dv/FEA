r"""
Load the research team's `beamformer` package WITHOUT running its __init__.py.

WHY THIS EXISTS
---------------
We want to run *their* beamformer on *our* FEM channel data, so that a
FEM-vs-k-Wave image difference is attributable to the forward solver and not to
a re-implemented imaging chain.

Their `beamformer/__init__.py` does:

    from . import cli, internal, mig, plot, ray, readers, sizing, tof, utils

`cli` and `readers` pull in boto3 / s3path / ilipy / ilidataset - AWS plumbing and
DarkVision-internal packages that have nothing to do with imaging and that we
cannot install. So a plain `import beamformer` fails on machinery we never use.

We only need the imaging path: utils, internal, ray, tof, mig, plot.

THE TRICK
---------
Pre-register a bare module object named "beamformer" in sys.modules with its
__path__ pointing at their package directory. Python then treats it as an
already-imported package: `import beamformer.tof` finds the subpackage via
__path__ and, crucially, the relative imports *inside* their modules
(`from .. import internal`) resolve normally - because the parent package
exists. Their __init__.py is simply never executed, so cli/readers are never
touched.

This is read-only use of their source. We do not edit, patch, or install it
(no `pip install -e`, which would write egg-link metadata into their checkout).
The package is mounted read-only in the container; see run.ps1.

The one thing __init__.py does that we do want is registering their custom
matplotlib colormaps, so we replicate that (best-effort) at the end.

USAGE
-----
    from lib.bf_loader import load_beamformer
    bf = load_beamformer()
    img = bf.mig.kirchhoff_from_tof(..., engine="numpy")
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

# Where the package lives inside the container (read-only mount, see run.ps1).
# Falls back to the on-host submodule path so this also works outside Docker.
_CONTAINER_DIR = Path("/opt/bf/beamformer")
_HOST_DIR = (Path(__file__).resolve().parents[2]
             / "kwave" / "Libraries" / "PythonLibraries"
             / "beamformer" / "beamformer")

# Only the imaging path. Deliberately excludes cli/readers/sizing.
_SUBMODULES = ("internal", "utils", "ray", "tof", "mig", "plot")


def beamformer_dir() -> Path:
    for cand in (_CONTAINER_DIR, _HOST_DIR):
        if (cand / "__init__.py").is_file():
            return cand
    raise FileNotFoundError(
        f"beamformer package not found at {_CONTAINER_DIR} or {_HOST_DIR}.\n"
        "In Docker it must be mounted read-only at /opt/bf; on the host, run\n"
        "  git submodule update --init --depth 1 src/kwave\n"
        "and ensure sparse-checkout includes Libraries/PythonLibraries/beamformer/beamformer"
    )


def load_beamformer(submodules: tuple[str, ...] = _SUBMODULES) -> types.ModuleType:
    """Import the beamformer imaging path, bypassing its __init__.py. Idempotent."""
    pkg_dir = beamformer_dir()

    existing = sys.modules.get("beamformer")
    if existing is not None and getattr(existing, "_dv_shimmed", False):
        return existing

    pkg = types.ModuleType("beamformer")
    pkg.__path__ = [str(pkg_dir)]          # makes it a package for the import system
    pkg.__file__ = str(pkg_dir / "__init__.py")   # never executed; for tracebacks only
    pkg._dv_shimmed = True                 # marker so a second call is a no-op
    sys.modules["beamformer"] = pkg

    import importlib
    for name in submodules:
        setattr(pkg, name, importlib.import_module(f"beamformer.{name}"))

    # __init__.py registers their colormaps with matplotlib; imgz plots rely on
    # the default one, so mirror that. Best-effort: never fail the load for it.
    try:
        import matplotlib as _mpl
        for _name, _cm in pkg.plot.colormaps.items():
            if _name not in _mpl.colormaps:
                _mpl.colormaps.register(_cm)
    except Exception as exc:                # pragma: no cover - cosmetic only
        print(f"[bf_loader] colormap registration skipped: {exc}", file=sys.stderr)

    return pkg


if __name__ == "__main__":
    bf = load_beamformer()
    print(f"beamformer loaded from {beamformer_dir()}")
    for _n in _SUBMODULES:
        print(f"  bf.{_n:9s} {'OK' if hasattr(bf, _n) else 'MISSING'}")
    # the calls we actually depend on downstream
    for path in ("mig.kirchhoff_from_tof",
                 "tof.polar_wave_ray_pipe_one_wall",
                 "tof.polar_wave_ray_pipe_two_walls_reflect",
                 "tof.omnidirectional_fermat_pipe_one_wall",
                 "utils.imaging_grid", "utils.angle_filter_migration",
                 "utils.bandpass", "utils.optimal_fnumber", "utils.NDVolume",
                 "plot.plot_image"):
        obj = bf
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        print(f"  {'OK  ' if obj is not None else 'FAIL'} bf.{path}")
