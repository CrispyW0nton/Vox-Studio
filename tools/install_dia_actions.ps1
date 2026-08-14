$ErrorActionPreference = "Stop"

$engineRoot = Join-Path $env:LOCALAPPDATA "VoxStudio\engines\dia-actions"
$venv = Join-Path $engineRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"

function Assert-LastExitCode([string]$Activity) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Activity failed with exit code $LASTEXITCODE."
    }
}

New-Item -ItemType Directory -Force -Path $engineRoot | Out-Null

if (-not (Test-Path $python)) {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv --python 3.12 --seed $venv
        Assert-LastExitCode "Creating the Dia virtual environment"
    } else {
        py -3.12 -m venv $venv
        Assert-LastExitCode "Creating the Dia virtual environment"
    }
}

$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
& $python -m pip --version *> $null
$ErrorActionPreference = $savedErrorActionPreference
if ($LASTEXITCODE -ne 0) {
    & $python -m ensurepip --upgrade
    Assert-LastExitCode "Bootstrapping pip in the Dia environment"
}

& $python -m pip install --upgrade pip
Assert-LastExitCode "Updating pip in the Dia environment"
$ErrorActionPreference = "SilentlyContinue"
& $python -c "import dia" *> $null
$diaInstalled = $LASTEXITCODE -eq 0
$ErrorActionPreference = $savedErrorActionPreference
if (-not $diaInstalled) {
    $diaRevision = "876125e461a03b157ec905b0fe8b57a0f8b9e7a0"
    & $python -m pip install "git+https://github.com/nari-labs/dia.git@$diaRevision"
    Assert-LastExitCode "Installing Dia"
}

# Dia currently pins an older CPU wheel on Windows. Install the Blackwell-ready
# CUDA build last so RTX 50-series systems do not silently fall back to CPU.
$ErrorActionPreference = "SilentlyContinue"
& $python -c "import torch, torchaudio; assert torch.__version__.startswith('2.8.0+cu128'); assert torchaudio.__version__.startswith('2.8.0+cu128'); assert torch.cuda.is_available()" *> $null
$cudaRuntimeReady = $LASTEXITCODE -eq 0
$ErrorActionPreference = $savedErrorActionPreference
if (-not $cudaRuntimeReady) {
    & $python -m pip install --upgrade `
        "torch==2.8.0" `
        "torchaudio==2.8.0" `
        --index-url https://download.pytorch.org/whl/cu128
    Assert-LastExitCode "Installing the Dia CUDA runtime"
}

& $python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
Assert-LastExitCode "Validating the Dia CUDA runtime"

Write-Host "Dia vocal-action runtime installed at $engineRoot"
