<#
.SYNOPSIS
    Sets up the Python virtual environment for this repo (automates the
    "Setup" section of README.md).

.DESCRIPTION
    Creates .\venv (unless it already exists) and installs requirements.txt
    into it -- everything needed for the full pipeline (flake_detection,
    grid_preperation, channels_model) lives in that one file.

    Does NOT touch: your MATLAB install (needed for grid_preperation's
    rgb2ind calls, with the Image Processing Toolbox), the trained
    channels_model.onnx download, or grid_preperation\BaseGrid.cif -- see
    README.md's own Setup section for those.

.PARAMETER Force
    Delete and recreate an existing .\venv instead of reusing it.

.EXAMPLE
    .\setup.ps1
.EXAMPLE
    .\setup.ps1 -Force
#>
[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

function Get-PythonCommand {
    foreach ($candidate in @('python', 'py')) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    throw "No Python interpreter found on PATH (tried 'python', 'py'). Install Python 3.10+ and re-run."
}

$venvPath = Join-Path $repoRoot 'venv'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'

if ((Test-Path $venvPath) -and $Force) {
    Write-Host "Removing existing venv at $venvPath (-Force)..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPath)) {
    $python = Get-PythonCommand
    Write-Host "Creating virtual environment at $venvPath using $python..." -ForegroundColor Cyan
    & $python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw "python -m venv failed (exit $LASTEXITCODE)." }
} else {
    Write-Host "Reusing existing venv at $venvPath (pass -Force to delete and recreate it)." -ForegroundColor Cyan
}

if (-not (Test-Path $venvPython)) {
    throw "Expected venv Python at $venvPython but it doesn't exist -- venv creation may have failed."
}

Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit $LASTEXITCODE)." }

if (-not (Test-Path 'requirements.txt')) { throw "requirements.txt not found in $repoRoot" }

Write-Host "Installing requirements.txt..." -ForegroundColor Cyan
Write-Host "(pins a CUDA 12.4 torch/torchvision wheel via --extra-index-url -- edit that line in requirements.txt first if you're on a different CUDA version, or just accept CPU-only wheels; see README.md)" -ForegroundColor DarkGray
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)." }

Write-Host ""
Write-Host "Done. Remaining manual steps (see README.md's Setup section for detail):" -ForegroundColor Green
Write-Host "  1. Activate the venv in your shell:  venv\Scripts\activate"
Write-Host "  2. MATLAB (with the Image Processing Toolbox) must be installed locally -- grid_preperation calls rgb2ind via the matlabengine package."
Write-Host "  3. Download the trained model from the project's Google Drive folder and place it at channels_model\outputs\channels_model.onnx."
Write-Host "  4. Provide your own grid_preperation\BaseGrid.cif (not tracked in git) before running placement."
