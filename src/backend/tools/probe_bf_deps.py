r"""
Enumerate every third-party module the beamformer IMAGING path needs.

Repeatedly imports bf's imaging submodules; each time a module is missing it is
replaced with a MagicMock (which auto-creates any attribute, so `from x import y`
also succeeds) and the import is retried. The result is the complete missing-dep
list in one run instead of one-per-docker-invocation.

MagicMock stubs are for DEPENDENCY DISCOVERY ONLY - never for real beamforming.
A stub silently returns mocks instead of numbers, so any image produced with one
would be garbage. `verify_no_stubs()` in bf_loader-based scripts is the guard.

RUN:  ./run.ps1 python3 tools/probe_bf_deps.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.bf_loader import beamformer_dir  # noqa: E402

TARGETS = ("internal", "utils", "ray", "tof", "mig", "plot")
MAX_ROUNDS = 200

pkg_dir = beamformer_dir()
print(f"probing {pkg_dir}\n")

import types  # noqa: E402

pkg = types.ModuleType("beamformer")
pkg.__path__ = [str(pkg_dir)]
pkg._dv_shimmed = True
sys.modules["beamformer"] = pkg

import importlib  # noqa: E402

stubbed: list[str] = []
failures: list[str] = []

for target in TARGETS:
    for _ in range(MAX_ROUNDS):
        try:
            importlib.import_module(f"beamformer.{target}")
            break
        except ModuleNotFoundError as exc:
            name = exc.name
            if not name or name.startswith("beamformer"):
                failures.append(f"{target}: unresolvable -> {exc!r}")
                break
            if name in stubbed:
                failures.append(f"{target}: stubbing {name} did not help -> {exc!r}")
                break
            stubbed.append(name)
            sys.modules[name] = MagicMock()
        except ImportError as exc:
            # e.g. `from x import y` where x exists but lacks y
            name = getattr(exc, "name", None)
            if name and name not in stubbed:
                stubbed.append(name)
                sys.modules[name] = MagicMock()
                continue
            failures.append(f"{target}: {exc!r}")
            break
        except Exception as exc:                      # noqa: BLE001
            failures.append(f"{target}: {type(exc).__name__}: {exc}")
            break
    else:
        failures.append(f"{target}: exceeded {MAX_ROUNDS} rounds")

print("MISSING third-party modules (import name), in discovery order:")
for name in stubbed:
    print(f"  - {name}")
if not stubbed:
    print("  (none - everything already present)")

print("\nSubmodule import status:")
for target in TARGETS:
    ok = f"beamformer.{target}" in sys.modules
    print(f"  {'OK  ' if ok else 'FAIL'} beamformer.{target}")

if failures:
    print("\nProblems that stubbing could not resolve:")
    for f in failures:
        print(f"  ! {f}")
