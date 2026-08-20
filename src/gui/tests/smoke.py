r"""End-to-end smoke test: does the app actually run a simulation?

Every module's own demo() checks its logic in isolation with the container faked out. This is
the one test that puts the whole chain together and lets Docker do real work: build a config,
plan it, submit it to the Runner, drive a real Qt event loop, and check that the files the
plan PROMISED are the files that appeared.

It is deliberately the cheapest run that still exercises everything: a coarse mesh, degree 3,
and t_end 3 us instead of 60, which is a few seconds rather than eight minutes. Physics is not
the point here; the plumbing is - argv construction, mounts, the results mount being writable,
progress parsing, manifest writing, and the promised-vs-actual output check.

  --keep   leave the outputs on disk for inspection
  RUN      .venv-gui/Scripts/python.exe src/gui/tests/smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QCoreApplication, QTimer                      # noqa: E402

from core import docker                                                  # noqa: E402
from core.runner import Runner                                           # noqa: E402
from model.spec import Device, RunConfig, Stage, plan                     # noqa: E402

# Unique h_notch so the mesh stem cannot collide with anything already on disk - the coarse
# notch is also what makes this run fast, since the smallest cell sets the global time step.
CFG = RunConfig(scale=1.0, degree=3, h_notch=0.60, t_end=3.0e-6, device=Device.GPU)
TIMEOUT_S = 600


def main() -> None:
    app = QCoreApplication(sys.argv)
    contract = docker.container_contract()
    print(f"image {contract.image} | gpu {contract.gpu_image} | results {contract.results_dir}")

    specs = plan(CFG, stages=(Stage.MESH, Stage.FORWARD))
    promised = [o for s in specs for o in s.outputs]
    print(f"plan: {[s.stage.value for s in specs]}")
    print(f"promises {len(promised)} file(s): {promised}")

    results = contract.results_dir
    for rel in promised:                       # start from a known-absent state
        (results / rel).unlink(missing_ok=True)

    runner = Runner(contract=contract)
    seen_progress: list[float] = []
    failures: list[str] = []

    def on_progress(job_id, prog):
        if prog.fraction is not None:
            seen_progress.append(prog.fraction)

    def on_finished(job_id, state):
        job = runner.job(job_id)
        code = getattr(job, "exit_code", None)
        print(f"  {job_id}: {state} (exit {code})")
        if state != "succeeded":
            failures.append(f"{job_id} -> {state} exit {code}")
        outstanding = [j for j in runner.jobs()
                       if getattr(j, "state", None) is not None
                       and getattr(j.state, "value", str(j.state))
                       in ("queued", "running")]
        if not outstanding:
            app.quit()

    runner.job_progress.connect(on_progress)
    runner.job_finished.connect(on_finished)
    runner.job_log.connect(lambda jid, line: print(f"    | {line}") if line.strip() else None)

    QTimer.singleShot(TIMEOUT_S * 1000, lambda: (failures.append("TIMEOUT"), app.quit()))
    runner.submit_plan(specs)
    app.exec()
    runner.shutdown()

    print()
    assert not failures, f"jobs did not succeed: {failures}"
    missing = [r for r in promised if not (results / r).is_file()]
    assert not missing, f"the plan promised files that never appeared: {missing}"
    assert seen_progress, "no progress was parsed from any job's output"
    assert max(seen_progress) > 0.5, f"progress never got past {max(seen_progress):.2f}"

    for rel in promised:
        f = results / rel
        print(f"  ok  {rel}  {f.stat().st_size / 1e6:.1f} MB")
    mans = sorted((results / "gui_runs").glob("*.json")) if (results / "gui_runs").is_dir() else []
    assert mans, "no manifest was written - a run that leaves no record is not reproducible"
    print(f"  ok  {len(mans)} manifest(s)")

    if "--keep" not in sys.argv:
        for rel in promised:
            (results / rel).unlink(missing_ok=True)
        # The manifests too. This is a plumbing test, and its throwaway 3 us runs do not
        # belong in the user's experiment log - the gallery reads that folder.
        for m in mans:
            m.unlink(missing_ok=True)
        print("  cleaned up (pass --keep to inspect)")
    print("\nsmoke: PASS")


if __name__ == "__main__":
    main()
