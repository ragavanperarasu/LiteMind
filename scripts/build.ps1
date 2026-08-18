<#
.SYNOPSIS
    Configures and builds LiteMind on Windows.

.DESCRIPTION
    Works with either toolchain:
      - MSYS2 / MinGW-w64 (run from the MINGW64 shell, or let this find it)
      - Visual Studio 2022 with the C++ workload

    LiteMind has no third-party dependencies, so nothing needs installing
    beyond a compiler and CMake.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Toolchain MinGW -RunTests
#>
[CmdletBinding()]
param(
    [ValidateSet("Auto", "MinGW", "MSVC")]
    [string]$Toolchain = "Auto",
    [string]$BuildDirectory = "build",
    [switch]$RunTests,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($Clean -and (Test-Path $BuildDirectory)) {
    Write-Host "Removing $BuildDirectory" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildDirectory
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Error @"
CMake was not found on PATH.

  MSYS2:  pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-gcc
  Or:     winget install Kitware.CMake
"@
    exit 1
}

if ($Toolchain -eq "Auto") {
    $Toolchain = if (Get-Command g++ -ErrorAction SilentlyContinue) { "MinGW" } else { "MSVC" }
    Write-Host "Detected toolchain: $Toolchain" -ForegroundColor Cyan
}

if ($Toolchain -eq "MinGW") {
    $compiler = Get-Command g++ -ErrorAction SilentlyContinue
    if (-not $compiler) {
        Write-Error @"
g++ was not found on PATH.

Open the "MSYS2 MINGW64" shell (not MSYS or MINGW32 - a 32-bit build cannot
memory-map a multi-gigabyte checkpoint) and install the toolchain:

  pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja
"@
        exit 1
    }

    $generator = if (Get-Command ninja -ErrorAction SilentlyContinue) { "Ninja" } else { "MinGW Makefiles" }
    cmake -S . -B $BuildDirectory -G $generator -DCMAKE_BUILD_TYPE=Release
} else {
    # The Visual Studio generator picks the configuration at build time.
    cmake -S . -B $BuildDirectory -G "Visual Studio 17 2022" -A x64
}

if ($LASTEXITCODE -ne 0) { Write-Error "Configuration failed."; exit 1 }

cmake --build $BuildDirectory --config Release --parallel
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed."; exit 1 }

if ($RunTests) {
    Write-Host "`nRunning tests" -ForegroundColor Cyan
    ctest --test-dir $BuildDirectory -C Release --output-on-failure
}

$executable = Get-ChildItem -Path $BuildDirectory -Recurse -Filter "LiteMind.exe" |
    Select-Object -First 1
if ($executable) {
    Write-Host "`nBuilt: $($executable.FullName)" -ForegroundColor Green
    Write-Host "`nNext:" -ForegroundColor Cyan
    Write-Host "  python tools\make_test_model.py models-test   # a few MB, checks the build"
    Write-Host "  $($executable.FullName) models-test -p `"hello`" -n 8"
    Write-Host "  powershell -File scripts\download_model.ps1   # the real 31 GB weights"
} else {
    Write-Warning "The build reported success but LiteMind.exe was not found."
}
