param(
    [ValidateSet('cpu','gpu')]
    [string]$FaceService = 'cpu',
    [switch]$InstallDev,
    [switch]$SkipActivation
)

# One-command environment setup for Windows PowerShell
# Usage examples:
#   .\scripts\setup_env.ps1            -> create venv and install runtime deps (cpu face-service)
#   .\scripts\setup_env.ps1 -FaceService gpu -InstallDev -> install GPU face-service and dev deps
# To run from an untrusted session, use: powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1

$ErrorActionPreference = 'Stop'
# Compute repository root as the parent directory of the "scripts" folder where this script lives
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Resolve-Path (Join-Path $scriptDir '..') | Select-Object -ExpandProperty Path
Set-Location $repoRoot

$venvDir = Join-Path $repoRoot ".venv"

Write-Host "Repository root: $repoRoot"
Write-Host "Virtualenv path: $venvDir"

if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venvDir
} else {
    Write-Host "Virtual environment already exists at $venvDir"
}

# Path to pip inside the created venv
$pip = Join-Path $venvDir "Scripts\pip.exe"
$python = Join-Path $venvDir "Scripts\python.exe"

if (-not (Test-Path $pip)) {
    Write-Error "pip not found in venv ($pip). Ensure Python is installed and 'python -m venv .venv' succeeded."
    exit 2
}

Write-Host "Upgrading pip, setuptools and wheel..."
& $python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing runtime requirements from requirements-full.txt..."
& $pip install -r requirements-full.txt

if ($InstallDev) {
    Write-Host "Installing development requirements from requirements-dev.txt..."
    & $pip install -r requirements-dev.txt
}

# Face-service selection
if ($FaceService -eq 'gpu') {
    Write-Host "Installing face-service GPU requirements..."
    if (Test-Path "face-service\requirements-gpu.txt") {
        & $pip install -r "face-service\requirements-gpu.txt"
    } else {
        Write-Warning "face-service/requirements-gpu.txt not found. Using CPU requirements instead."
        if (Test-Path "face-service\requirements-cpu.txt") { & $pip install -r "face-service\requirements-cpu.txt" }
    }
} else {
    if (Test-Path "face-service\requirements-cpu.txt") { 
        Write-Host "Installing face-service CPU requirements..."
        & $pip install -r "face-service\requirements-cpu.txt"
    }
}

Write-Host "Running quick import check to detect missing modules..."
& $python .\scripts\test_imports.py

Write-Host "Setup complete."
if (-not $SkipActivation) {
    Write-Host "To activate the venv for this session run:\n    .\\.venv\\Scripts\\Activate.ps1"
}

exit 0
