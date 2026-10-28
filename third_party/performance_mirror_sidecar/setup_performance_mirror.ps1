param(
    [string]$InstallRoot = "$env:LOCALAPPDATA\VoxStudio\engines\seed-vc"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$upstream = "https://github.com/Plachtaa/seed-vc.git"
$revision = "51383efd921027683c89e5348211d93ff12ac2a8"
$python = Join-Path $InstallRoot ".venv\Scripts\python.exe"

Write-Host "Preparing Vox Studio Performance Mirror..."
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "The uv Python installer is required. Install uv from https://docs.astral.sh/uv/."
}

if (-not (Test-Path -LiteralPath (Join-Path $InstallRoot ".git"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $InstallRoot) | Out-Null
    git clone $upstream $InstallRoot
    if ($LASTEXITCODE -ne 0) { throw "Unable to download Seed-VC." }
}

git -C $InstallRoot fetch --depth 1 origin $revision
if ($LASTEXITCODE -ne 0) { throw "Unable to update the Seed-VC checkout." }
git -C $InstallRoot checkout --detach $revision
if ($LASTEXITCODE -ne 0) { throw "Unable to select the tested Seed-VC revision." }

if (-not (Test-Path -LiteralPath $python)) {
    uv venv --python 3.12 (Join-Path $InstallRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Unable to create the Python environment." }
}

uv pip install --python $python `
    torch==2.11.0 torchaudio==2.11.0 `
    --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "Unable to install the CUDA runtime." }
uv pip install --python $python `
    "numpy<2" scipy librosa huggingface-hub munch einops `
    descript-audio-codec "transformers==4.46.3" soundfile pyyaml python-dotenv tqdm
if ($LASTEXITCODE -ne 0) { throw "Unable to install Performance Mirror dependencies." }

& $python -c "import torch; assert torch.cuda.is_available(); print('CUDA ready:', torch.cuda.get_device_name())"
if ($LASTEXITCODE -ne 0) { throw "CUDA is not available to Performance Mirror." }
Write-Host ""
Write-Host "Performance Mirror is installed. You can close this window and start it in Vox Studio."
