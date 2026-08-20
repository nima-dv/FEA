r"""The container contract, read once from docker/run.ps1, plus argv construction.

WHY THIS MODULE EXISTS AT ALL
docker/run.ps1 owns the mount and env decisions, and those decisions are load-bearing: the
results volume is mounted BACK at /work/results so that every command in the README, the
dossier and the lessons file still reads `--mesh results/ili_mesh/...`. Duplicating that
list here would let the two drift, and a drifted mount means a GUI run writes somewhere the
published commands never look. So: ask run.ps1 for the argv pieces via -PrintArgs, then
invoke docker ourselves.

WHY NOT JUST CALL run.ps1 TO RUN THE JOB
Because science arguments would then cross a second PowerShell parser, which has bitten this
project twice (`--xlim=-8,85` gets split on the comma and the minus is read as a parameter
name). QProcess -> docker.exe is one hop with no shell, so the argv the manifest records is
byte-for-byte the argv the container saw. run.ps1 keeps ownership of the contract; the GUI
keeps ownership of naming, quoting and cancellation.
"""
from __future__ import annotations

import functools
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Repo root: src/gui/core/docker.py -> core -> gui -> src -> repo
REPO = Path(__file__).resolve().parents[3]
RUN_PS1 = REPO / "docker" / "run.ps1"


class DockerUnavailable(RuntimeError):
    """Docker Desktop is not running, or run.ps1 could not be read. Message says which."""


@dataclass(frozen=True)
class Contract:
    """What run.ps1 -PrintArgs decided. Both image variants, one mount/env list.

    The CPU and GPU variants differ only in image tag and the --gpus switch, but neither is
    hardcoded here: the GPU tag lives in run.ps1 (and can be overridden by
    $env:DVFENICS_IMAGE), so we ask for it rather than assume `dvfenics:gpu`.
    """
    image: str
    gpu_image: str
    gpu_args: tuple[str, ...]
    mounts: tuple[str, ...]
    env_args: tuple[str, ...]
    workdir: str
    repo: Path

    def image_for(self, gpu: bool) -> str:
        return self.gpu_image if gpu else self.image

    def device_args(self, gpu: bool) -> tuple[str, ...]:
        return self.gpu_args if gpu else ()

    @property
    def results_dir(self) -> Path:
        """Host side of the /work/results mount. Manifests and the tracked-file guard need it."""
        return self.repo / "data" / "results"


def _print_args(gpu: bool) -> dict:
    if not RUN_PS1.is_file():
        raise DockerUnavailable(f"container contract missing: {RUN_PS1} not found")
    argv = ["powershell", "-NoProfile", "-File", str(RUN_PS1), "-PrintArgs"]
    if gpu:
        argv.append("-Gpu")
    try:
        # -PrintArgs exits before it touches the daemon, so this works with Docker stopped;
        # that is deliberate, because the parameter form must be usable while Docker boots.
        p = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as e:                       # no powershell on PATH
        raise DockerUnavailable(f"cannot run powershell: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise DockerUnavailable(f"{RUN_PS1.name} -PrintArgs timed out") from e
    if p.returncode != 0:
        raise DockerUnavailable(
            f"{RUN_PS1.name} -PrintArgs failed (exit {p.returncode}): {p.stderr.strip()}")
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError as e:
        raise DockerUnavailable(
            f"{RUN_PS1.name} -PrintArgs did not emit JSON: {p.stdout[:200]!r}") from e


def _strlist(v) -> tuple[str, ...]:
    """ConvertTo-Json collapses a one-element array to a bare scalar. Normalise both shapes."""
    if v is None:
        return ()
    return tuple(str(x) for x in (v if isinstance(v, list) else [v]))


@functools.lru_cache(maxsize=1)
def container_contract() -> Contract:
    """Read run.ps1's decisions. Cached: two PowerShell starts is ~1 s we pay once per session."""
    cpu, gpu = _print_args(False), _print_args(True)
    return Contract(
        image=str(cpu["image"]),
        gpu_image=str(gpu["image"]),
        gpu_args=_strlist(gpu.get("gpuArgs")),
        mounts=_strlist(cpu.get("mounts")),
        env_args=_strlist(cpu.get("envArgs")),
        workdir=str(cpu.get("workdir", "/work")),
        repo=Path(str(cpu.get("repo", REPO))),
    )


def build_argv(spec, contract: Contract, container_name: str) -> list[str]:
    """The full `docker run ...` argv for one JobSpec.

    --name is not cosmetic: cancel() has to kill the CONTAINER, and killing the docker client
    process leaves the container running (it is a separate daemon child). No -t either, since
    a pipe is not a TTY and docker refuses.
    """
    return [
        "docker", "run", "--rm", "--name", container_name,
        *contract.device_args(spec.gpu),
        *contract.mounts,
        *contract.env_args,
        "-w", contract.workdir,
        contract.image_for(spec.gpu),
        *spec.argv,
    ]


def kill(container_name: str) -> bool:
    """`docker kill`. True if it died here; False if it was already gone (not an error).

    A cancel arriving microseconds after the job finished by itself is normal, and so is a
    cancel for a container that never started because the image was missing.
    """
    if not shutil.which("docker"):
        return False
    p = subprocess.run(["docker", "kill", container_name],
                       capture_output=True, text=True, timeout=30)
    return p.returncode == 0


def _docker(*args: str, timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


@dataclass(frozen=True)
class Probe:
    """Everything the status bar shows. Cheap enough to re-run on a timer."""
    daemon: bool
    images: tuple[str, ...]
    gpu: bool
    free_gb: float
    note: str = ""

    def has_image(self, image: str) -> bool:
        return image in self.images


def probe(contract: Contract | None = None) -> Probe:
    """Daemon up? which images? GPU visible? free space where results land?

    The GPU answer is the HOST's (nvidia-smi), not a passthrough test: proving passthrough
    means starting a container, and the status bar refreshes too often to pay seconds for it.
    A guard that needs certainty should run tools/gpu_probe.py inside the container instead.
    """
    results = contract.results_dir if contract else REPO / "data" / "results"
    try:
        free_gb = shutil.disk_usage(results if results.exists() else REPO).free / 1e9
    except OSError as e:
        free_gb, note = 0.0, f"disk unreadable: {e}"
    else:
        note = ""

    if not shutil.which("docker"):
        return Probe(False, (), False, free_gb, note or "docker.exe not on PATH")
    try:
        if _docker("info", "--format", "{{.ServerVersion}}").returncode != 0:
            return Probe(False, (), False, free_gb, note or "Docker daemon not reachable")
        imgs = _docker("images", "--format", "{{.Repository}}:{{.Tag}}").stdout.split()
        gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired) as e:
        return Probe(False, (), False, free_gb, note or f"docker query failed: {e}")
    return Probe(True, tuple(imgs), gpu, free_gb, note)


def demo() -> None:
    """Self-check. Must pass with Docker stopped, so the contract is faked here."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.spec import RunConfig, Stage, plan            # noqa: E402

    fake = Contract(image="dvfenics:bf", gpu_image="dvfenics:gpu",
                    gpu_args=("--gpus", "all"),
                    mounts=("-v", "C:\\r\\src\\backend:/work",
                            "-v", "C:\\r\\data\\results:/work/results"),
                    env_args=("-e", "PYTHONPATH=/work"),
                    workdir="/work", repo=Path("C:/r"))
    assert fake.results_dir == Path("C:/r/data/results")

    jobs = {j.stage: j for j in plan(RunConfig())}
    fwd = build_argv(jobs[Stage.FORWARD], fake, "gui_fwd_1")
    assert fwd[:5] == ["docker", "run", "--rm", "--name", "gui_fwd_1"], fwd[:5]
    assert "--gpus" in fwd and fwd[fwd.index("--gpus") + 1] == "all"
    assert "dvfenics:gpu" in fwd and "dvfenics:bf" not in fwd
    # The container command must survive untouched and stay LAST - it is what the manifest
    # promises reproduces from a terminal.
    assert fwd[-len(jobs[Stage.FORWARD].argv):] == jobs[Stage.FORWARD].argv
    assert fwd.index("-w") < fwd.index("dvfenics:gpu"), "-w belongs to docker, not the script"

    mesh = build_argv(jobs[Stage.MESH], fake, "gui_mesh_1")
    assert "--gpus" not in mesh and "dvfenics:bf" in mesh, mesh
    assert mesh.count("--name") == 1

    assert _strlist("--gpus") == ("--gpus",), "one-element PS array arrives as a scalar"
    assert _strlist(None) == () and _strlist(["a", "b"]) == ("a", "b")

    # A missing contract script must say so rather than fail somewhere in docker.
    global RUN_PS1
    keep, RUN_PS1 = RUN_PS1, Path("C:/nope/run.ps1")
    try:
        _print_args(False)
    except DockerUnavailable as e:
        assert "missing" in str(e), e
    else:
        raise AssertionError("missing run.ps1 must raise DockerUnavailable")
    finally:
        RUN_PS1 = keep

    p = probe(fake)                       # must not raise with Docker down
    assert isinstance(p.daemon, bool) and p.free_gb >= 0.0
    print(f"docker.demo: ok (daemon={p.daemon} gpu={p.gpu} free={p.free_gb:.0f} GB "
          f"images={len(p.images)}{' - ' + p.note if p.note else ''})")


if __name__ == "__main__":
    demo()
