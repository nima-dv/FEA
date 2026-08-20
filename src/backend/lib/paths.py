r"""Where things live. ONE definition, because this used to be spelled out in 25 files.

Every script used to build its own output path from its own location plus a "results"
literal, which silently encoded two assumptions: that results sit inside the source tree, and
that every script is exactly one directory below the backend root. The first stopped being
true when data was separated from source; the second was never checked.

RESOLUTION ORDER
  1. $DVFEA_RESULTS / $DVFEA_RAW if set. This is how the container gets them: docker/run.ps1
     mounts the host folders and exports these, so nothing inside the container needs to know
     the host layout.
  2. Otherwise, relative to this file: lib -> backend -> src -> repo root, then data/.
     This is the host fallback, used by the GUI and by anything run outside the container.

The env var is not an optional nicety inside the container - /work has no parent that
resembles the repo, so the fallback would be wrong rather than merely different. Hence the
check at the bottom: fail with a sentence that says what to do, rather than write 2 GB of
results into a path nobody will look in.
"""
from __future__ import annotations

import os
from pathlib import Path

#: the backend source root - the folder holding lib/, mesh/, repro/, tools/, viz/
BACKEND = Path(__file__).resolve().parents[1]


def _resolve(env: str, *fallback: str) -> Path:
    """Env var if set, else walk out to the repo root and down into data/.

    The walk is lazy and guarded: inside the container this file sits at /work/lib/paths.py,
    which has no third parent, so computing it eagerly raised IndexError before the env var
    it would never have needed was even read.
    """
    v = os.environ.get(env)
    if v:
        return Path(v)
    here = Path(__file__).resolve()
    if len(here.parents) < 4:                     # lib -> backend -> src -> repo
        raise RuntimeError(
            f"${env} is not set and this file is at {here}, which has no repo root above it. "
            f"Inside the container docker/run.ps1 sets it; set it by hand otherwise.")
    return here.parents[3].joinpath(*fallback)


#: our outputs: meshes, channel data, images, figures, built pages
RESULTS = _resolve("DVFEA_RESULTS", "data", "results")

#: the research team's raw k-Wave runs. READ-ONLY - mounted :ro in the container, and
#: nothing in this project may write here.
RAW = _resolve("DVFEA_RAW", "data", "raw")

#: THE PRESENTATION SUB-PROJECT. Its own data, scripts and documents - the evidence for the
#: R&D challenge, kept apart from routine run output on purpose so that the two can never be
#: confused and a stray run can never overwrite a published figure.
PRESENTATION = _resolve("DVFEA_PRESENTATION", "presentation")

#: where the published figures, channel data and reference cases live
PRES_DATA = PRESENTATION / "data"

if not RESULTS.parent.exists():
    raise RuntimeError(
        f"cannot resolve the results directory: {RESULTS} has no parent on disk.\n"
        f"Inside the container this must come from $DVFEA_RESULTS, which docker/run.ps1 "
        f"sets. If you are running a script by hand, set it:\n"
        f"    DVFEA_RESULTS=/results DVFEA_RAW=/raw python3 <script>")
