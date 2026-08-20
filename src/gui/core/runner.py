r"""The job queue: QProcess, FIFO, one solve at a time, and the tracked-file guard.

WHY QProcess AND NOT subprocess
A solve is 7.7 min on the GPU and 2.4 h on the CPU. Anything that blocks the Qt event loop
for that long is a frozen app, and a thread reading a pipe would still need to marshal every
line back to the GUI thread. QProcess already delivers stdout as signals on the GUI thread,
so there is no thread to get wrong.

WHY FORWARD JOBS SERIALISE
Two solves at once would share one GPU's memory and neither would fit, and on the CPU they
would halve each other's cores while doubling the wall clock. It also matches how the team
works: one solve, watched. Mesh and imaging are minutes at most, so they run alongside.

WHY THE TRACKED-FILE GUARD IS HERE AND NOT ONLY IN spec.py
GUI_TAG_PREFIX makes a collision with the published record unlikely; this makes it
impossible. data/results holds the k-Wave +20 deg baseline and everything scored against it,
those files ARE tracked in git, and a container running as root inside a rw mount will
happily overwrite one. So before a container starts, every expected output is checked with
`git ls-files --error-unmatch`, and a hit refuses the job unless the caller passed
allow_overwrite. Requirement 3 of the work order.
"""
from __future__ import annotations

import enum
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

if __package__:
    from . import docker, logparse, manifest
else:
    # Direct execution (`python src/gui/core/runner.py`) has no package context. src/gui on
    # the path is the convention the rest of the app uses, and main.py puts it there anyway.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core import docker, logparse, manifest


class JobState(enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED)


@dataclass
class Job:
    job_id: str
    spec: object                      # model.spec.JobSpec
    allow_overwrite: bool = False
    state: JobState = JobState.QUEUED
    argv: list[str] = field(default_factory=list)
    container: str = ""
    started: float = 0.0
    ended: float | None = None
    exit_code: int | None = None
    ms_per_step: float | None = None
    size: int | None = None
    error: str = ""
    proc: QProcess | None = None
    _tail: str = ""                   # incomplete last line between readyRead signals


def is_tracked(repo: Path, path: Path) -> bool:
    """Is `path` in git's index? The guard's single question.

    `git ls-files --error-unmatch` is the cheapest exact answer - .gitignore, sparse checkout
    and case folding are all git's business, and reimplementing that in Python is how you end
    up disagreeing with git about the one file that mattered.

    Exit codes are read carefully because two of them mean opposite things: 1 is "git looked
    and this file is not tracked", but 128 is "git could not look at all" (not a repository,
    path outside the worktree, no git on PATH). Only 1 may be treated as permission to write.
    The guard protects the published record, so unknown means refuse.
    """
    try:
        p = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", str(path)],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return True
    return p.returncode != 1


class Runner(QObject):
    """FIFO queue of container jobs. One Runner per app.

    Signals carry the job id, never the Job object, so a view cannot mutate queue state by
    holding a reference. Call job(job_id) for the details.
    """
    job_started = Signal(str)
    job_progress = Signal(str, object)        # job_id, logparse.Progress
    job_log = Signal(str, str)                # job_id, one line, VERBATIM
    job_finished = Signal(str, str)            # job_id, JobState.value
    queue_changed = Signal()

    def __init__(self, contract: docker.Contract | None = None, parent=None) -> None:
        super().__init__(parent)
        # Contract is injectable so demo() and the views can run with Docker stopped; when
        # None it is read (and cached) from docker/run.ps1 on first use.
        self._contract = contract
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._n = 0
        # Overridable seam for demo(): the only thing between a JobSpec and a real container.
        self.build_argv = docker.build_argv

    # --- contract ---------------------------------------------------------------------
    @property
    def contract(self) -> docker.Contract:
        if self._contract is None:
            self._contract = docker.container_contract()
        return self._contract

    # --- queue ------------------------------------------------------------------------
    def submit(self, spec, allow_overwrite: bool = False) -> str:
        """Queue one JobSpec. Returns its id. Nothing starts until the event loop turns."""
        self._n += 1
        stage = getattr(spec.stage, "value", str(spec.stage))
        job_id = f"gui_{time.strftime('%Y%m%d_%H%M%S')}_{self._n:03d}_{stage}"
        self._jobs[job_id] = Job(job_id=job_id, spec=spec, allow_overwrite=allow_overwrite)
        self._order.append(job_id)
        self.queue_changed.emit()
        self._pump()
        return job_id

    def submit_plan(self, specs, allow_overwrite: bool = False) -> list[str]:
        return [self.submit(s, allow_overwrite) for s in specs]

    def job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def jobs(self) -> list[Job]:
        return [self._jobs[i] for i in self._order]

    def running(self) -> list[Job]:
        return [j for j in self.jobs() if j.state is JobState.RUNNING]

    def cancel(self, job_id: str) -> None:
        """Kill the CONTAINER, not just the client. Queued jobs are simply dropped."""
        j = self._jobs.get(job_id)
        if j is None or j.state in TERMINAL:
            return
        if j.state is JobState.QUEUED:
            j.state = JobState.CANCELLED
            self.queue_changed.emit()
            self.job_finished.emit(job_id, j.state.value)
            self._pump()
            return
        j.state = JobState.CANCELLED          # read back in _on_finished as the verdict
        docker.kill(j.container)
        # docker kill makes `docker run` exit, which fires QProcess.finished and does the
        # bookkeeping. If the client is wedged (daemon gone), kill it too - the container is
        # already dead, so nothing is orphaned.
        if j.proc is not None and j.proc.state() != QProcess.NotRunning:
            j.proc.kill()

    def shutdown(self) -> None:
        """Kill every container this Runner owns. Call from the main window's closeEvent.

        Without this, closing the app leaves a solve running with no reader: it keeps the GPU
        and keeps writing into data/results, and the next run's guard sees a file that no
        manifest explains.
        """
        for j in self.jobs():
            if j.state is JobState.RUNNING:
                j.state = JobState.CANCELLED
                docker.kill(j.container)
                if j.proc is not None:
                    j.proc.kill()
                    j.proc.waitForFinished(3000)

    # --- scheduling -------------------------------------------------------------------
    def _pump(self) -> None:
        """Start whatever is eligible, in submission order."""
        for j in self.jobs():
            if j.state is not JobState.QUEUED:
                continue
            if self._is_solve(j) and any(self._is_solve(r) for r in self.running()):
                continue                       # one solve at a time; keep FIFO for the rest
            self._start(j)

    @staticmethod
    def _is_solve(job: Job) -> bool:
        return getattr(job.spec.stage, "value", job.spec.stage) == "forward"

    # --- the guard --------------------------------------------------------------------
    def _guard(self, job: Job) -> str:
        """Empty string if the job may run, else the reason it may not."""
        if job.allow_overwrite:
            return ""
        repo, results = self.contract.repo, self.contract.results_dir
        for rel in getattr(job.spec, "outputs", []):
            path = results / rel
            if is_tracked(repo, path):
                return (f"refusing to run: {rel} is git-tracked and would be overwritten. "
                        f"It is part of the published record. Change the tag, or pass "
                        f"allow_overwrite=True if you really mean to replace it.")
        return ""

    # --- process lifecycle ------------------------------------------------------------
    def _start(self, job: Job) -> None:
        reason = self._guard(job)
        if reason:
            job.state, job.error = JobState.FAILED, reason
            job.started = job.ended = time.time()
            self.job_log.emit(job.job_id, reason)
            self.queue_changed.emit()
            self.job_finished.emit(job.job_id, job.state.value)
            return

        job.container = job.job_id            # same string in docker ps and in the manifest
        try:
            job.argv = list(self.build_argv(job.spec, self.contract, job.container))
        except docker.DockerUnavailable as e:
            job.state, job.error = JobState.FAILED, str(e)
            job.started = job.ended = time.time()
            self.job_log.emit(job.job_id, str(e))
            self.queue_changed.emit()
            self.job_finished.emit(job.job_id, job.state.value)
            return

        p = QProcess(self)
        # Merged channels because the backend interleaves warnings on stderr with progress on
        # stdout, and splitting them would reorder the log the user has to read.
        p.setProcessChannelMode(QProcess.MergedChannels)
        p.setProgram(job.argv[0])
        p.setArguments(job.argv[1:])
        p.readyReadStandardOutput.connect(lambda jid=job.job_id: self._on_output(jid))
        p.finished.connect(lambda code, status, jid=job.job_id: self._on_finished(jid, code))
        p.errorOccurred.connect(lambda err, jid=job.job_id: self._on_error(jid, err))
        job.proc, job.state, job.started = p, JobState.RUNNING, time.time()
        self._write_manifest(job)             # before it can crash: see manifest.Manifest
        p.start()
        self.job_started.emit(job.job_id)
        self.queue_changed.emit()

    def _on_output(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None or job.proc is None:
            return
        chunk = bytes(job.proc.readAllStandardOutput()).decode("utf-8", "replace")
        # QProcess hands over arbitrary chunks, so a progress line can arrive in halves; the
        # remainder is carried until its newline or the parse silently misses every second one.
        job._tail += chunk.replace("\r\n", "\n")
        *lines, job._tail = job._tail.split("\n")
        for line in lines:
            self.job_log.emit(job_id, line)   # verbatim, matched or not
            prog = logparse.parse_line(job.spec.stage, line)
            if prog is not None:
                if prog.ms_per_step is not None:
                    job.ms_per_step = prog.ms_per_step
                self.job_progress.emit(job_id, prog)
            size = logparse.parse_size(line)
            if size is not None:
                job.size = size

    def _on_error(self, job_id: str, err) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        if err == QProcess.FailedToStart:
            job.error = f"could not start {job.argv[0] if job.argv else 'docker'}"
            self.job_log.emit(job_id, job.error)
            # FailedToStart does not emit finished() on every platform; close the job here.
            self._on_finished(job_id, -1)

    def _on_finished(self, job_id: str, code: int) -> None:
        job = self._jobs.get(job_id)
        if job is None or job.ended is not None:
            return                             # errorOccurred + finished can both arrive
        if job._tail:                          # a last line without its newline
            self.job_log.emit(job_id, job._tail)
            job._tail = ""
        job.ended, job.exit_code = time.time(), code
        # cancel() already set CANCELLED; a non-zero exit from a killed container is expected
        # and must not be reported as a failure.
        if job.state is not JobState.CANCELLED:
            job.state = JobState.SUCCEEDED if code == 0 else JobState.FAILED
        job.proc = None
        self._write_manifest(job)
        self.queue_changed.emit()
        self.job_finished.emit(job_id, job.state.value)
        self._pump()                           # the queued solve, if any, goes now

    # --- log --------------------------------------------------------------------------
    def _write_manifest(self, job: Job) -> manifest.Manifest | None:
        results = self.contract.results_dir
        m = manifest.Manifest(
            job_id=job.job_id,
            stage=getattr(job.spec.stage, "value", str(job.spec.stage)),
            argv=job.argv,
            config=manifest.config_dict(job.spec.config),
            label=getattr(job.spec, "label", ""),
            commit=manifest.git_commit(self.contract.repo),
            started=job.started,
            ended=job.ended,
            exit_code=job.exit_code,
            state=job.state.value,
            ms_per_step=job.ms_per_step,
            size=job.size,
            outputs=manifest.collect_outputs(results, list(getattr(job.spec, "outputs", []))),
        )
        try:
            manifest.write(m, results)
        except OSError as e:                   # a full or read-only volume must not kill a run
            self.job_log.emit(job.job_id, f"manifest not written: {e}")
            return None
        return m


def demo() -> None:
    """Self-check without Docker: a faked contract and `cmd /c echo` in place of a container."""
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer         # noqa: E402
    from model.spec import JobSpec, RunConfig, Stage, plan                  # noqa: E402

    repo_real = Path(__file__).resolve().parents[3]
    app = QCoreApplication.instance() or QCoreApplication([])

    # 1. The guard, against the real repo and the real published record.
    tracked = repo_real / "data" / "results" / "ili_forward" / "channel_data_deg4_s0p8_p20deg.npz"
    assert tracked.exists(), f"expected the published baseline at {tracked}"
    assert is_tracked(repo_real, tracked), "the +20 deg baseline IS tracked; guard must see it"
    assert not is_tracked(repo_real, repo_real / "data" / "results" / "no_such_file.npz")
    assert is_tracked(Path(tempfile.gettempdir()), tracked) is True, "no git -> refuse"

    with tempfile.TemporaryDirectory() as td:
        # A real (empty) repo, because the guard fails CLOSED: outside a worktree git cannot
        # answer, and "cannot answer" must never read as "safe to overwrite".
        subprocess.run(["git", "init", "-q", td], capture_output=True, timeout=30)
        fake = docker.Contract(image="dvfenics:bf", gpu_image="dvfenics:gpu",
                               gpu_args=("--gpus", "all"), mounts=(), env_args=(),
                               workdir="/work", repo=Path(td))
        jobs = {j.stage: j for j in plan(RunConfig())}

        # 2. A real QProcess round trip. cmd /c echo stands in for the container and prints a
        #    line the forward parser must recognise, so signal wiring and parsing are both
        #    exercised without Docker.
        line = "  step 140000/182457 (77%)  4.1 ms/step  elapsed 9.6 min  ETA 2.9 min"
        r = Runner(fake)
        r.build_argv = lambda spec, contract, name: [
            "cmd", "/c", f"echo {line}& echo      degree-4 vector DOF = 2094218"]
        log: list[str] = []
        prog: list[logparse.Progress] = []
        done: list[tuple[str, str]] = []
        r.job_log.connect(lambda jid, ln: log.append(ln))
        r.job_progress.connect(lambda jid, p: prog.append(p))
        r.job_finished.connect(lambda jid, st: done.append((jid, st)))

        jid = r.submit(jobs[Stage.FORWARD])
        assert r.job(jid).state is JobState.RUNNING, "submit must start it immediately"

        loop = QEventLoop()
        r.job_finished.connect(lambda *_: loop.quit())
        QTimer.singleShot(20000, loop.quit)
        loop.exec()

        assert done and done[0][1] == "succeeded", (done, log)
        assert any("step 140000/182457" in ln for ln in log), log
        assert prog and abs(prog[0].fraction - 140000 / 182457) < 1e-9
        j = r.job(jid)
        assert j.ms_per_step == 4.1 and j.size == 2094218, (j.ms_per_step, j.size)
        assert j.exit_code == 0 and j.ended and j.ended >= j.started

        # 3. The manifest landed under gui_runs and carries the argv and the timing.
        m = manifest.read(manifest.runs_dir(fake.results_dir) / f"{jid}.json")
        assert m.state == "succeeded" and m.ms_per_step == 4.1 and m.argv[0] == "cmd"
        assert m.config["angle"] == 20.0 and m.stage == "forward"
        assert manifest.history(fake.results_dir) == [(2094218, "gpu", 4.1)]

        # 4. Serialisation: two solves queued, only one runs.
        r2 = Runner(fake)
        r2.build_argv = lambda spec, contract, name: ["cmd", "/c", "echo hi"]
        a = r2.submit(jobs[Stage.FORWARD])
        b = r2.submit(jobs[Stage.FORWARD])
        c = r2.submit(jobs[Stage.MESH])
        assert r2.job(a).state is JobState.RUNNING and r2.job(b).state is JobState.QUEUED
        assert r2.job(c).state is JobState.RUNNING, "a mesh job may run beside a solve"
        assert len([x for x in r2.running() if Runner._is_solve(x)]) == 1

        # 5. Cancelling a queued job drops it; the next solve then starts on its own.
        r2.cancel(b)
        assert r2.job(b).state is JobState.CANCELLED
        loop2 = QEventLoop()
        QTimer.singleShot(5000, loop2.quit)
        loop2.exec()
        assert r2.job(a).state in TERMINAL and r2.job(c).state in TERMINAL, "jobs must finish"
        r2.shutdown()

        # 6. The guard refuses a job whose output is tracked - never starting a process.
        r3 = Runner(docker.Contract(image="i", gpu_image="i", gpu_args=(), mounts=(),
                                    env_args=(), workdir="/work", repo=repo_real))
        r3.build_argv = lambda *a: (_ for _ in ()).throw(AssertionError("must not start"))
        blocked: list[str] = []
        r3.job_finished.connect(lambda jid, st: blocked.append(st))
        published = JobSpec(Stage.FORWARD, RunConfig(), ["python3", "x.py"],
                            outputs=["ili_forward/channel_data_deg4_s0p8_p20deg.npz"])
        gid = r3.submit(published)
        assert blocked == ["failed"] and "git-tracked" in r3.job(gid).error, r3.job(gid).error
        # ... and runs it when the overwrite is explicit (build_argv would raise, which proves
        # the guard let it through).
        try:
            r3.submit(published, allow_overwrite=True)
        except AssertionError as e:
            assert "must not start" in str(e)
        else:
            raise AssertionError("allow_overwrite must bypass the guard")

    del app
    print("runner.demo: ok")


if __name__ == "__main__":
    demo()
