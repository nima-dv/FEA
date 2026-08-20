# Forwarder. The real launcher is docker/run.ps1; this exists so that every command in the
# README, the dossier and the lessons file still copy-pastes from the repo root. No param
# block on purpose: $args keeps -Gpu / -Name as tokens, and splatting an array re-parses
# them as parameters rather than passing them positionally.
& (Join-Path $PSScriptRoot 'docker\run.ps1') @args
