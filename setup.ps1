<#
.SYNOPSIS
    Sets up the Python virtual environment for this repo (automates the
    "Setup" section of README.md).

.DESCRIPTION
    Creates .\venv (unless it already exists) and installs requirements.txt
    into it -- everything needed for the full pipeline (flake_detection,
    grid_preperation, channels_model) lives in that one file, EXCEPT
    matlabengine (see below).

    matlabengine is version-locked to a specific MATLAB release (e.g.
    matlabengine 26.1 only works against a local MATLAB R2026a install --
    MathWorks publishes a distinct matlabengine version per release, not one
    version that works everywhere) -- pip installing whatever's hardcoded in
    requirements.txt silently gives you a matlabengine that can't find/talk
    to a differently-versioned local MATLAB. So this script detects your
    actual installed MATLAB release (scanning Program Files\MATLAB\R*) and
    installs the matlabengine version that matches it instead of
    requirements.txt's own pin -- see Get-MatlabEngineVersion below for the
    release-string -> pip-version mapping this relies on.

    Cannot install MATLAB itself (it's licensed commercial software -- no
    script can do that) or verify the Image Processing Toolbox is actually
    LICENSED under it (only that its files are present on disk, a
    best-effort proxy -- real confirmation only happens when grid_preperation
    actually calls rgb2ind). Also does NOT touch: the trained
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

function Find-MatlabInstalls {
    <#
    Every local MATLAB install found under the default Windows install
    location (Program Files\MATLAB\R20XXx), newest release first. A custom
    install location isn't scanned -- if you installed MATLAB somewhere
    else, install matlabengine by hand instead (see the version-mapping
    note printed at the end of this script).
    #>
    $root = Join-Path ${env:ProgramFiles} 'MATLAB'
    if (-not (Test-Path $root)) { return @() }
    Get-ChildItem -Path $root -Directory -Filter 'R20*' -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName 'bin\matlab.exe') } |
        ForEach-Object {
            [PSCustomObject]@{
                Release          = $_.Name
                Root             = $_.FullName
                HasImageToolbox  = Test-Path (Join-Path $_.FullName 'toolbox\images')
            }
        } |
        Sort-Object Release -Descending
}

function Get-MatlabEngineVersion {
    <#
    MathWorks' matlabengine pip package is versioned "<2-digit release
    year>.<1 for an 'a' release, 2 for a 'b' release>" -- e.g. R2026a ->
    "26.1", R2024b -> "24.2" -- one specific matlabengine version per MATLAB
    release, not a version range (confirmed against this repo's own pin:
    requirements.txt hardcodes matlabengine==26.1, i.e. was set up against a
    local R2026a). Returns $null if `Release` isn't in the expected
    "R20YY[ab]" shape.
    #>
    param([Parameter(Mandatory)][string]$Release)
    if ($Release -notmatch '^R(\d{4})([ab])$') { return $null }
    $yearShort = $Matches[1].Substring(2, 2)
    $half = if ($Matches[2] -eq 'a') { '1' } else { '2' }
    return "$yearShort.$half"
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

# matlabengine is installed separately below (version-matched to whatever
# MATLAB is actually found locally), not from requirements.txt's own pin --
# filter it out of the bulk install so a mismatched local MATLAB doesn't
# silently get the wrong matlabengine.
$reqLines = Get-Content 'requirements.txt' | Where-Object { $_ -notmatch '^\s*matlabengine\s*==' }
$filteredReq = Join-Path $env:TEMP "chip_design_requirements_$([guid]::NewGuid().ToString('N')).txt"
Set-Content -Path $filteredReq -Value $reqLines -Encoding utf8

Write-Host "Installing requirements.txt (matlabengine handled separately below)..." -ForegroundColor Cyan
Write-Host "(pins a CUDA 12.4 torch/torchvision wheel via --extra-index-url -- edit that line in requirements.txt first if you're on a different CUDA version, or just accept CPU-only wheels; see README.md)" -ForegroundColor DarkGray
try {
    & $venvPython -m pip install -r $filteredReq
    if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)." }
} finally {
    Remove-Item $filteredReq -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Locating local MATLAB installs..." -ForegroundColor Cyan
$matlabInstalls = Find-MatlabInstalls
$matlabStatusLines = @()

if (-not $matlabInstalls) {
    Write-Host "  No local MATLAB found under $(Join-Path ${env:ProgramFiles} 'MATLAB')." -ForegroundColor Yellow
    $matlabStatusLines += "MATLAB not found -- install it (with the Image Processing Toolbox), then run:"
    $matlabStatusLines += "  venv\Scripts\python.exe -m pip install matlabengine==<version matching your MATLAB release>"
    $matlabStatusLines += "  (release -> version: R20YYa -> YY.1, R20YYb -> YY.2 -- e.g. R2026a -> 26.1)"
} else {
    $chosen = $matlabInstalls | Where-Object { $_.HasImageToolbox } | Select-Object -First 1
    if (-not $chosen) {
        $chosen = $matlabInstalls | Select-Object -First 1
        Write-Host "  Found $($chosen.Release) at $($chosen.Root), but no toolbox\images folder under it -- Image Processing Toolbox may not be installed (grid_preperation's rgb2ind calls need it)." -ForegroundColor Yellow
        $matlabStatusLines += "$($chosen.Release) found, but Image Processing Toolbox not detected -- install it via MATLAB's Add-Ons before running grid_preperation."
    } else {
        Write-Host "  Found $($chosen.Release) at $($chosen.Root) (Image Processing Toolbox present)." -ForegroundColor Cyan
    }

    $engineVersion = Get-MatlabEngineVersion -Release $chosen.Release
    if (-not $engineVersion) {
        Write-Host "  Could not derive a matlabengine version from release string '$($chosen.Release)' -- install it by hand, see README.md." -ForegroundColor Yellow
        $matlabStatusLines += "Detected MATLAB release '$($chosen.Release)' didn't match the expected R20YY[ab] shape -- install matlabengine by hand for it."
    } else {
        Write-Host "  Installing matlabengine==$engineVersion (matched to $($chosen.Release))..." -ForegroundColor Cyan
        & $venvPython -m pip install "matlabengine==$engineVersion"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  pip install matlabengine==$engineVersion failed -- this version may not be on PyPI yet for a very new MATLAB release, or too old for one this repo's other pins assume. See README.md." -ForegroundColor Yellow
            $matlabStatusLines += "matlabengine==$engineVersion failed to install for detected release $($chosen.Release) -- see README.md."
        } else {
            $matlabStatusLines += "matlabengine==$engineVersion installed, matched to detected MATLAB $($chosen.Release)."
        }
    }
}

Write-Host ""
Write-Host "Done. Remaining manual steps (see README.md's Setup section for detail):" -ForegroundColor Green
Write-Host "  1. Activate the venv in your shell:  venv\Scripts\activate"
Write-Host "  2. MATLAB: $($matlabStatusLines -join "`n     ")"
Write-Host "  3. Download the trained model from the project's Google Drive folder and place it at channels_model\outputs\channels_model.onnx."
Write-Host "  4. Provide your own grid_preperation\BaseGrid.cif (not tracked in git) before running placement."
