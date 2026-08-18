# Build-once + run-in-container helper for the DOLFINx research env.
# Mounts THIS folder at /work inside the container, so host edits are live.
#
# Usage:
#   ./run.ps1                                 # interactive bash shell in the container
#   ./run.ps1 python3 toys/check_dolfinx.py   # run a command in the container
#
# Requires Docker Desktop running. First invocation builds the image (dolfinx stable + extras).
#
# MOUNTS
#   <this folder>            -> /work     (read-write: our code and results)
#   ../kwave/.../beamformer -> /opt/bf  (READ-ONLY: the research team's
#       beamformer package, via the git submodule. Mounted :ro on purpose - that repo
#       is read-only for us, and lib/bf_loader.py imports it in place rather than
#       pip-installing it, so nothing is ever written into their checkout.)
#   ../../data               -> /data    (READ-ONLY: the research team's raw k-Wave runs.
#       Lives outside /work because it is gitignored bulk data, ~423 MB. Feed it to
#       tools/extract_kwave_case.py as --run /data/derrell/<run folder>.)
param([Parameter(ValueFromRemainingArguments = $true)] $Cmd)
$ErrorActionPreference = 'Stop'
$here  = $PSScriptRoot
# dvfenics:bf = dvfenics:latest + the deps the research team's beamformer needs.
# Built as a separate tag so the older working image stays available as a rollback.
# Override with $env:DVFENICS_IMAGE if you need a different one.
$image = if ($env:DVFENICS_IMAGE) { $env:DVFENICS_IMAGE } else { 'dvfenics:bf' }

# The beamformer package inside the sparse submodule checkout (may be absent if the
# submodule was not initialised - that is fine unless you are beamforming).
$bfDir = Join-Path $here '..\kwave\Libraries\PythonLibraries\beamformer'
$mounts = @('-v', "${here}:/work")
if (Test-Path (Join-Path $bfDir 'beamformer\__init__.py')) {
    $bfFull = (Resolve-Path $bfDir).Path
    $mounts += @('-v', "${bfFull}:/opt/bf:ro")
} else {
    Write-Host "note: beamformer submodule not checked out; /opt/bf not mounted." -ForegroundColor DarkYellow
    Write-Host "      git submodule update --init --depth 1 ../kwave" -ForegroundColor DarkYellow
}

# The research team's raw runs. Read-only: we only ever extract from them.
$dataDir = Join-Path $here '..\..\data'
if (Test-Path $dataDir) {
    $mounts += @('-v', "$((Resolve-Path $dataDir).Path):/data:ro")
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
    docker run --rm @mounts -w /work $image @Cmd
} else {
    docker run --rm -it @mounts -w /work $image bash
}
