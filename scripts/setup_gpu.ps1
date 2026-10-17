param(
    [string]$Python = "python",
    [string]$Environment = ".venv-gpu"
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent
if ($Environment -notmatch '^[A-Za-z0-9_.-]+$' -or $Environment -in @('.', '..')) {
    throw "Environment must be a dedicated directory name inside this repository"
}
$envPath = [IO.Path]::GetFullPath((Join-Path $repoRoot $Environment))
if (-not (Test-Path (Join-Path $envPath "Scripts/python.exe"))) {
    & $Python -m venv $envPath
    if ($LASTEXITCODE -ne 0) { throw "Could not create GPU environment" }
}
# The environment is local, including its own ignore rule.
Set-Content -LiteralPath (Join-Path $envPath ".gitignore") -Value "*"
$gpuPython = Join-Path $envPath "Scripts/python.exe"
& $gpuPython -m pip install "torch==2.13.0+cu130" --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed" }
& $gpuPython -m pip install "numpy==2.5.2" "scipy==1.18.0" "h5py==3.16.0" "pytest==9.1.1"
if ($LASTEXITCODE -ne 0) { throw "Numerical dependencies installation failed" }
& $gpuPython (Join-Path $repoRoot "experiments/train_closure.py") doctor --device cuda
if ($LASTEXITCODE -ne 0) { throw "CUDA execution check failed" }
