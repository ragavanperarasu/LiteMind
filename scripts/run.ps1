<#
.SYNOPSIS
    Builds LiteMind if needed and starts it, reading settings from litemind.json.

.DESCRIPTION
    One command for the whole cycle: compile, then ask for a prompt. Everything
    that can be configured lives in litemind.json, so there are no flags to
    remember - edit that file and run this again.

    The build is incremental. Nothing is recompiled when no source has changed,
    so this is also the right command for the second and every later run.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -Config review.json
#>
[CmdletBinding()]
param(
    [string]$Config = "litemind.json",
    [string]$BuildDirectory = "build",
    [switch]$Clean,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

# Run from the repository root whichever directory this was invoked from, so the
# relative paths in the settings file always mean the same thing.
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    if (-not (Test-Path $Config)) {
        Write-Error "Settings file not found: $Config"
        exit 1
    }

    if (-not $SkipBuild) {
        Write-Host "Building" -ForegroundColor Cyan
        $buildArguments = @("-BuildDirectory", $BuildDirectory)
        if ($Clean) { $buildArguments += "-Clean" }
        & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "build.ps1") @buildArguments
        if ($LASTEXITCODE -ne 0) {
            Write-Error "The build failed. Nothing was run."
            exit $LASTEXITCODE
        }
    }

    $executable = Join-Path $root (Join-Path $BuildDirectory "bin\LiteMind.exe")
    if (-not (Test-Path $executable)) {
        Write-Error "Built, but $executable is missing. Try -Clean."
        exit 1
    }

    Write-Host "`nSettings: $Config  (edit it to change anything)" -ForegroundColor DarkGray
    Write-Host ""
    & $executable --config $Config
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
