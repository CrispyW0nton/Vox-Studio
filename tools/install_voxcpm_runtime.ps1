[CmdletBinding()]
param(
    [string]$RuntimeRoot = "$env:LOCALAPPDATA\VoxStudio\engines\voxcpm2"
)

$ErrorActionPreference = "Stop"

$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uv) {
    $uvCandidates = @(
        "$env:USERPROFILE\.local\bin\uv.exe",
        "$env:LOCALAPPDATA\hermes\bin\uv.exe"
    )
    $uvPath = $uvCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($uvPath)) {
        throw "uv is required. Install it from https://docs.astral.sh/uv/."
    }
} else {
    $uvPath = $uv.Source
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$requirements = Join-Path $repositoryRoot "third_party\voxcpm_sidecar\runtime_requirements.txt"
$python = Join-Path $RuntimeRoot ".venv\Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
if (-not (Test-Path $python)) {
    & $uvPath venv --python 3.12 (Join-Path $RuntimeRoot ".venv")
}

& $uvPath pip install --python $python `
    --index-strategy unsafe-best-match `
    --extra-index-url "https://download.pytorch.org/whl/cu128" `
    "torch==2.11.0+cu128"
& $uvPath pip install --python $python -r $requirements

$env:HF_HOME = Join-Path $RuntimeRoot "cache"
& $python -c @"
from faster_whisper import WhisperModel
from voxcpm import VoxCPM

root = r"$RuntimeRoot"
VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    load_denoiser=False,
    cache_dir=root + r"\cache",
    optimize=True,
    device="cuda",
)
WhisperModel(
    "base.en",
    device="cuda",
    compute_type="float16",
    download_root=root + r"\cache\faster-whisper",
)
print("VoxCPM2 runtime and model cache are ready.")
"@
