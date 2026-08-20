# Build-once + run-in-container helper for the DOLFINx research env.
#
# Usage (from anywhere; the repo root has a forwarder so ./run.ps1 also works):
#   ./run.ps1                                  # interactive bash shell
#   ./run.ps1 python3 repro/ili_forward.py ... # run a command
#   ./run.ps1 -Gpu python3 tools/gpu_probe.py  # same, on the CuPy image with the GPU
#   ./run.ps1 -Name myjob python3 ...          # named container, so it can be killed
#   ./run.ps1 -PrintArgs                       # emit the docker argv as JSON and exit
#
# Requires Docker Desktop running. First invocation builds the image.
#
# MOUNTS, and why each one is where it is
#   src/backend    -> /work     (rw)  the science code. Working directory, so every
#                                     documented command still reads `python3 repro/...`.
#   data/results   -> /work/results (rw)  our outputs. On the HOST these sit outside the source
#                                     tree, which is the point of the layout: 2.7 GB of
#                                     intermediates does not belong next to 900 kB of code.
#                                     Inside the container they are mounted BACK at
#                                     /work/results on purpose, so every command in the
#                                     README, dossier and lessons file still reads
#                                     `--mesh results/ili_mesh/...` and still works. The
#                                     host layout is the deliverable; the container view is
#                                     an implementation detail, and keeping it stable was
#                                     worth more than tidiness. Docker creates the empty
#                                     mountpoint src/backend/results on the host as a side
#                                     effect; it is gitignored and always empty.
#   data/raw       -> /raw      (ro)  the research team's k-Wave workspaces. READ-ONLY at the
#                                     mount, not merely by convention.
#   src/kwave/.../beamformer -> /opt/bf (ro)  their beamformer package, imported in place by
#                                     lib/bf_loader.py so nothing is ever written into their
#                                     checkout (a `pip install -e` would drop .egg-info in it).
#
# ENVIRONMENT
#   PYTHONPATH=/work        so `from lib.paths import RESULTS` works from any subdirectory
#                           without a sys.path dance in every script.
#   DVFEA_RESULTS=/work/results  lib/paths.py reads these. Without them it would fall back to a
#   DVFEA_RAW=/raw          host-shaped layout that does not exist inside the container.
param(
    [switch] $Gpu,
    [switch] $PrintArgs,
    # Position=0 is NOT decoration. Adding -Name made this an advanced param block, and
    # PowerShell then handed position 0 to the first non-switch parameter - so `./run.ps1
    # python3 foo.py` bound "python3" to -Name and docker tried to exec foo.py directly.
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)] $Cmd,
    [string] $Name
)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$repo = Split-Path $here -Parent

# dvfenics:bf = dvfenics:latest + the deps the research team's beamformer needs.
# Override with $env:DVFENICS_IMAGE if you need a different one.
$image = if ($env:DVFENICS_IMAGE) { $env:DVFENICS_IMAGE }
          elseif ($Gpu) { 'dvfenics:gpu' }
          else { 'dvfenics:bf' }
# -Gpu selects the CuPy layer (Dockerfile.gpu) and passes the device through. The GPU path
# only accelerates the TIME LOOP - assembly and meshing stay on the CPU either way, because
# the official FEniCSx GPU route needs a CUDA-enabled PETSc that the stock dolfinx image
# does not ship.
$gpuArgs = if ($Gpu) { @('--gpus', 'all') } else { @() }

$mounts = @(
    '-v', "$(Join-Path $repo 'src\backend'):/work",
    '-v', "$(Join-Path $repo 'data\results'):/work/results"
)
$rawDir = Join-Path $repo 'data\raw'
if (Test-Path $rawDir) { $mounts += @('-v', "${rawDir}:/raw:ro") }

# The beamformer package inside the sparse submodule checkout (may be absent if the
# submodule was not initialised - fine unless you are beamforming).
$bfDir = Join-Path $repo 'src\kwave\Libraries\PythonLibraries\beamformer'
if (Test-Path (Join-Path $bfDir 'beamformer\__init__.py')) {
    $mounts += @('-v', "$((Resolve-Path $bfDir).Path):/opt/bf:ro")
} elseif (-not $PrintArgs) {
    Write-Host "note: beamformer submodule not checked out; /opt/bf not mounted." -ForegroundColor DarkYellow
    Write-Host "      git submodule update --init --depth 1 src/kwave" -ForegroundColor DarkYellow
}

$envArgs = @(
    '-e', 'PYTHONPATH=/work',
    '-e', 'DVFEA_RESULTS=/work/results',
    '-e', 'DVFEA_RAW=/raw'
)
$nameArgs = if ($Name) { @('--name', $Name) } else { @() }

# -PrintArgs exists for the GUI: it reads the mount/env/image decisions from here ONCE, then
# invokes docker itself. That keeps this file the single source of truth for the container
# contract while letting the app own container naming, cancellation and argument quoting -
# passing science arguments through a second PowerShell layer has bitten us twice already
# (PowerShell splits `--xlim=-8,85` on the comma and reads the minus as a parameter).
if ($PrintArgs) {
    @{
        image   = $image
        gpuArgs = $gpuArgs
        mounts  = $mounts
        envArgs = $envArgs
        workdir = '/work'
        repo    = $repo
    } | ConvertTo-Json -Depth 4 -Compress
    exit 0
}

# Fail early with a clear message if the Docker daemon isn't up.
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker daemon not reachable. Start Docker Desktop and try again."
}

# Build the small extras image on first use (cheap layer over dolfinx/dolfinx:stable).
if (-not (docker images -q $image)) {
    Write-Host "Building $image (first run; pulls dolfinx/dolfinx:stable) ..." -ForegroundColor Cyan
    docker build -t $image $here
    if ($LASTEXITCODE -ne 0) { throw "docker build failed." }
}

if ($Cmd) {
    # Non-interactive command: no -t (avoids 'input device is not a TTY' under tooling).
    docker run --rm @nameArgs @gpuArgs @mounts @envArgs -w /work $image @Cmd
} else {
    docker run --rm -it @nameArgs @gpuArgs @mounts @envArgs -w /work $image bash
}
