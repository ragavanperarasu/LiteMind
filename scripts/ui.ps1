<#
.SYNOPSIS
    Builds the engine and the web interface, then serves both.

.DESCRIPTION
    One command, the same shape as run.ps1: compile what is stale, then start.
    The Node process owns the engine and serves the built interface on one port,
    so there is nothing to start separately and no second window to keep open.

.PARAMETER Port
    Where to listen. Defaults to 5174.

.PARAMETER SkipBuild
    Leave the C++ build alone and only start the interface.

.PARAMETER Dev
    Run the Vite dev server with hot reload alongside the API, for editing the
    interface itself. Without it the pre-built bundle is served.
#>
[CmdletBinding()]
param(
    [int]$Port = 5174,
    [switch]$SkipBuild,
    [switch]$Dev
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repositoryRoot

try {
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        throw "Node.js is not on PATH. Install it from https://nodejs.org and reopen the terminal."
    }
    Write-Host "Node $(node --version)" -ForegroundColor DarkGray

    if (-not $SkipBuild) {
        Write-Host "`nBuilding the engine" -ForegroundColor Cyan
        & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'build.ps1')
        if ($LASTEXITCODE -ne 0) { throw "The engine build failed." }
    }

    $web = Join-Path $repositoryRoot 'ui\web'

    # node_modules is large and not in the repository, so its absence is the
    # normal first-run state rather than an error.
    if (-not (Test-Path (Join-Path $web 'node_modules'))) {
        Write-Host "`nInstalling interface packages (first run only)" -ForegroundColor Cyan
        Push-Location $web
        try {
            & npm install
            if ($LASTEXITCODE -ne 0) { throw "npm install failed." }
        } finally { Pop-Location }
    }

    if (-not $Dev) {
        Write-Host "`nBuilding the interface" -ForegroundColor Cyan
        Push-Location $web
        try {
            & npm run build
            if ($LASTEXITCODE -ne 0) { throw "The interface build failed." }
        } finally { Pop-Location }
    }

    $env:PORT = $Port

    if ($Dev) {
        Write-Host "`nStarting the API on $Port and the dev server on 5173" -ForegroundColor Green
        $api = Start-Process node -ArgumentList (Join-Path $repositoryRoot 'ui\server\server.mjs') -PassThru -NoNewWindow
        try {
            Push-Location $web
            & npm run dev
        } finally {
            Pop-Location
            if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id -Force }
        }
    }
    else {
        Write-Host "`nOpen http://localhost:$Port" -ForegroundColor Green
        Write-Host "Ctrl+C to stop.`n" -ForegroundColor DarkGray
        & node (Join-Path $repositoryRoot 'ui\server\server.mjs')
    }
}
finally {
    Pop-Location
}
