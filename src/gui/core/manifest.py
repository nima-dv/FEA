r"""One JSON per run: the experiment log.

Requirement 2 of the work order is that every job records the exact argv it ran, so a GUI run
reproduces from a terminal without the app. That is what this file is for. It also carries the
git commit, because argv alone does not identify a run: the same command against a different
commit is a different experiment, and this project's central claim is that every figure
reproduces from committed code.

The runtime numbers are here for a second reason. W2 wants to say "this configuration took
7.7 min last time" instead of estimating from a model of the solver, and the only trustworthy
source for that is what actually happened. history() serves exactly that: measured
(size, device, ms/step) triples, no model.

Manifests live in data/results/gui_runs/ - inside the results tree because they describe
results, in their own subdirectory because everything else under data/results is the
published record and nothing here may touch it.
"""
from __future__ import annotations

import dataclasses
import enum
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

RUNS_SUBDIR = "gui_runs"


def runs_dir(results_dir: Path) -> Path:
    return Path(results_dir) / RUNS_SUBDIR


@dataclass
class Manifest:
    """One job. Mutable on purpose: written at start, rewritten when the job ends.

    Writing at start means a crash or a hard kill still leaves evidence that the run existed;
    a manifest with exit_code None and no ended timestamp reads as "did not finish", which is
    information, whereas a missing file reads as "never happened", which would be a lie.
    """
    job_id: str
    stage: str
    argv: list[str]                              # the FULL docker argv, host side
    config: dict = field(default_factory=dict)   # RunConfig fields, enums flattened to values
    label: str = ""
    commit: str = ""
    started: float = 0.0                         # unix seconds
    ended: float | None = None
    exit_code: int | None = None
    state: str = "running"
    ms_per_step: float | None = None
    size: int | None = None                      # solver DOFs, or mesh cells for a mesh job
    outputs: list[dict] = field(default_factory=list)   # only files that ACTUALLY appeared

    @property
    def device(self) -> str:
        return str(self.config.get("device", "unknown"))


def _plain(v):
    """Enums to their .value, tuples to lists, Paths to str - so json.dump has no surprises."""
    if isinstance(v, enum.Enum):
        return v.value
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    return v


def config_dict(config) -> dict:
    """RunConfig -> JSON-able dict. Field names are kept verbatim so the file reads like the
    parameter form the user filled in."""
    return {k: _plain(v) for k, v in dataclasses.asdict(config).items()}


def git_commit(repo: Path) -> str:
    """Commit of the GUI/backend repo, with -dirty when the tree has uncommitted changes.

    The suffix matters: a run made from a dirty tree does NOT reproduce from the commit alone,
    and a manifest that hid that would be worse than one with no commit at all.
    """
    try:
        p = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        if p.returncode != 0:
            return ""
        sha = p.stdout.strip()
        d = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                           capture_output=True, text=True, timeout=30)
        return sha + ("-dirty" if d.stdout.strip() else "")
    except (OSError, subprocess.TimeoutExpired):
        return ""


def collect_outputs(results_dir: Path, expected: list[str]) -> list[dict]:
    """The expected outputs that exist, with size and mtime. Absent ones are simply omitted,
    which is how the queue view tells a job that "succeeded" but wrote nothing."""
    out = []
    for rel in expected:
        p = Path(results_dir) / rel
        if p.exists():
            st = p.stat()
            out.append({"path": rel, "bytes": st.st_size, "mtime": st.st_mtime})
    return out


def write(m: Manifest, results_dir: Path) -> Path:
    """Write (or rewrite) <results_dir>/gui_runs/<job_id>.json. Returns the path.

    Atomic replace: a manifest half-written when the app is killed would be unparseable, and
    history() would then choke on the whole directory rather than one run.
    """
    d = runs_dir(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{m.job_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dataclasses.asdict(m), indent=2, default=_plain) + "\n",
                   encoding="ascii")
    os.replace(tmp, path)
    return path


def read(path: Path) -> Manifest:
    data = json.loads(Path(path).read_text(encoding="ascii"))
    known = {f.name for f in dataclasses.fields(Manifest)}
    # Ignore unknown keys rather than crash: an older or newer app version must still be able
    # to read the log, because the log outlives the app.
    return Manifest(**{k: v for k, v in data.items() if k in known})


def load_all(results_dir: Path) -> list[Manifest]:
    out = []
    for p in sorted(runs_dir(results_dir).glob("*.json")):
        try:
            out.append(read(p))
        except (ValueError, OSError, TypeError):
            continue          # one corrupt file must not blind the whole history
    return sorted(out, key=lambda m: m.started)


def history(results_dir: Path) -> list[tuple[int, str, float]]:
    """Measured (size, device, ms_per_step) from past runs, oldest first.

    Only runs that reported BOTH a problem size and a ms/step are included - an estimate built
    on a half-known run would be worse than the model it replaces.
    """
    return [(m.size, m.device, m.ms_per_step) for m in load_all(results_dir)
            if m.size and m.ms_per_step]


def demo() -> None:
    """Self-check in a temp tree, so nothing is written under the real data/results."""
    import sys
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.spec import Device, RunConfig, Stage, plan            # noqa: E402

    c = RunConfig()
    cd = config_dict(c)
    assert cd["device"] == "gpu" and cd["notch"] == "present", cd
    assert cd["artifact_reduction"] == "none" and cd["snap_window"] == [18.0, 46.0], cd
    assert json.dumps(cd), "config must be JSON-able without a custom encoder"

    with tempfile.TemporaryDirectory() as td:
        results = Path(td)
        fwd = [j for j in plan(c) if j.stage is Stage.FORWARD][0]
        m = Manifest(job_id="gui_20260820_120000_forward", stage=fwd.stage.value,
                     argv=["docker", "run", "--rm", *fwd.argv], config=cd,
                     label=fwd.label, commit="deadbee", started=time.time())
        p = write(m, results)
        assert p == runs_dir(results) / f"{m.job_id}.json", p
        assert not list(runs_dir(results).glob("*.tmp")), "temp file must be replaced, not left"

        back = read(p)
        assert back.argv == m.argv and back.config == cd and back.exit_code is None
        assert back.device == "gpu" and back.state == "running"
        # The argv must still be the one a terminal can replay.
        assert "repro/ili_forward.py" in back.argv and "--abc-legacy" in back.argv

        # Finish it: same path, now with timings and whatever actually landed on disk.
        (results / "ili_forward").mkdir()
        (results / "ili_forward" / "channel_data_x.npz").write_bytes(b"x" * 11)
        m.ended, m.exit_code, m.state = time.time(), 0, "succeeded"
        m.ms_per_step, m.size = 4.1, 2094218
        m.outputs = collect_outputs(results, ["ili_forward/channel_data_x.npz",
                                              "ili_forward/never_written.npz"])
        write(m, results)
        assert len(list(runs_dir(results).glob("*.json"))) == 1, "finishing must not fork the log"
        fin = read(p)
        assert [o["path"] for o in fin.outputs] == ["ili_forward/channel_data_x.npz"]
        assert fin.outputs[0]["bytes"] == 11 and fin.exit_code == 0

        # A run with no measured ms/step must not pollute the runtime history.
        half = Manifest(job_id="gui_partial", stage="forward",
                        argv=["docker"], config={"device": Device.CPU.value},
                        started=time.time() - 100, size=999)
        write(half, results)
        assert history(results) == [(2094218, "gpu", 4.1)], history(results)

        # A corrupt manifest must not blind the reader.
        (runs_dir(results) / "gui_broken.json").write_text("{not json", encoding="ascii")
        assert len(load_all(results)) == 2 and len(history(results)) == 1

    sha = git_commit(Path(__file__).resolve().parents[3])
    assert sha and len(sha.split("-")[0]) >= 7, sha
    print(f"manifest.demo: ok (commit {sha})")


if __name__ == "__main__":
    demo()
