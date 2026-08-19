<#
.SYNOPSIS
    Configures and builds LiteMind on Windows.

.DESCRIPTION
    Works with either toolchain:
      - MSYS2 / MinGW-w64 (found automatically in the usual install locations)
      - Visual Studio 2022 with the C++ workload

    LiteMind has no third-party dependencies, so nothing needs installing
    beyond a compiler and CMake.

    The compiler search deliberately does not just take the first g++ on PATH.
    Windows machines often carry the old MinGW.org toolchain in C:\MinGW, which
    is 32-bit and predates C++20 by years; picking it produces confusing errors.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Clean -RunTests

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -CompilerPath C:\msys64\mingw64\bin\g++.exe
#>
[CmdletBinding()]
param(
    [ValidateSet("Auto", "MinGW", "MSVC")]
    [string]$Toolchain = "Auto",
    [string]$BuildDirectory = "build",
    [string]$CompilerPath = "",
    [switch]$RunTests,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# C++20 needs GCC 10 or newer. GCC 6.3, which MinGW.org still ships, has no
# <span>, no std::bit_cast, and no way to enable the dialect at all.
$minimumGccMajor = 10

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

function Test-GnuCompiler {
    <#
        Returns a description of a g++ candidate, or $null when it cannot build
        this project. -dumpmachine identifies the target architecture and
        -dumpversion the release, which together settle both requirements.
    #>
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
    if (-not (Test-Path $Path)) { return $null }

    try {
        $machine = (& $Path -dumpmachine 2>$null | Select-Object -First 1)
        $version = (& $Path -dumpversion 2>$null | Select-Object -First 1)
    } catch {
        return $null
    }
    if (-not $machine -or -not $version) { return $null }

    $major = 0
    [void][int]::TryParse((($version -split '\.')[0]), [ref]$major)

    return [pscustomobject]@{
        Path      = $Path
        Machine   = $machine
        Version   = $version
        Major     = $major
        Is64Bit   = $machine -match 'x86_64|aarch64'
        IsCurrent = $major -ge $minimumGccMajor
        Usable    = ($machine -match 'x86_64|aarch64') -and ($major -ge $minimumGccMajor)
    }
}

function Find-Msys2Root {
    <#
        Locates an existing MSYS2 installation. Its presence changes the advice
        completely: the toolchain package is missing, not MSYS2 itself, and
        re-running the installer fails with "TargetDirectoryInUse" rather than
        fixing anything.
    #>
    foreach ($candidate in @("C:\msys64", "D:\msys64", "E:\msys64",
                             "$env:SystemDrive\msys64", "$env:USERPROFILE\msys64")) {
        if ($candidate -and (Test-Path (Join-Path $candidate "usr\bin\bash.exe"))) {
            return $candidate
        }
    }
    return $null
}

function Find-GnuCompiler {
    <#
        Collects every g++ this machine offers, preferring known MSYS2 roots
        over whatever PATH happens to resolve first.
    #>
    $candidates = @()

    if ($CompilerPath) { $candidates += $CompilerPath }

    # MSYS2 environments, in the order they are usually preferred.
    foreach ($drive in @("C:", "D:", "E:")) {
        foreach ($environment in @("mingw64", "ucrt64", "clang64")) {
            $candidates += "$drive\msys64\$environment\bin\g++.exe"
        }
    }
    # w64devkit, a common standalone alternative.
    foreach ($drive in @("C:", "D:", "E:")) {
        $candidates += "$drive\w64devkit\bin\g++.exe"
    }

    # Anything already on PATH, last, so a stale MinGW.org install cannot win.
    Get-Command g++ -All -ErrorAction SilentlyContinue |
        Where-Object { $_.Source } |
        ForEach-Object { $candidates += $_.Source }

    $inspected = @()
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        $result = Test-GnuCompiler -Path $candidate
        if ($result) { $inspected += $result }
    }
    return $inspected
}

if ($Toolchain -eq "Auto") {
    $probe = Find-GnuCompiler
    $Toolchain = if ($probe | Where-Object { $_.Usable }) { "MinGW" }
                 elseif (Get-Command cl.exe -ErrorAction SilentlyContinue) { "MSVC" }
                 elseif (Test-Path "${env:ProgramFiles}\Microsoft Visual Studio\2022") { "MSVC" }
                 else { "MinGW" }
    Write-Host "Detected toolchain: $Toolchain" -ForegroundColor Cyan
}

$compilerArgument = @()
$compilerDirectory = ""

if ($Toolchain -eq "MinGW") {
    $inspected = Find-GnuCompiler
    $chosen = $inspected | Where-Object { $_.Usable } | Select-Object -First 1

    if (-not $chosen) {
        Write-Host "`nCompilers found:" -ForegroundColor Yellow
        if ($inspected.Count -eq 0) {
            Write-Host "  none" -ForegroundColor DarkGray
        }
        foreach ($entry in $inspected) {
            $why = if (-not $entry.Is64Bit) { "32-bit" }
                   elseif (-not $entry.IsCurrent) { "GCC $($entry.Version) is older than $minimumGccMajor" }
                   else { "unusable" }
            Write-Host "  $($entry.Path)  [$($entry.Machine), GCC $($entry.Version)] - $why" -ForegroundColor DarkGray
        }

        $msys2 = Find-Msys2Root
        if ($msys2) {
            # MSYS2 is present but its 64-bit toolchain package is not. Running
            # the installer again cannot fix this: it refuses to touch a
            # directory that already holds an installation and exits with 1.
            Write-Host "`nMSYS2 is already installed at $msys2." -ForegroundColor Cyan
            Write-Host "The 64-bit toolchain package is what is missing." -ForegroundColor Cyan
            Write-Error @"

Install the compiler into the MSYS2 you already have. Do not re-run the MSYS2
installer: it refuses to write into an existing installation and exits with
code 1, which is the error you saw.

Run this from PowerShell, no shell switching needed (it takes a few minutes):

  & "$msys2\usr\bin\bash.exe" -lc "pacman -Sy --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-ninja"

Then build again:

  powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Clean -RunTests

If pacman reports a corrupt database or stale keyring, refresh it first:

  & "$msys2\usr\bin\bash.exe" -lc "pacman -Syu --noconfirm"
"@
            exit 1
        }

        Write-Error @"

No usable C++ compiler was found. LiteMind needs 64-bit GCC $minimumGccMajor or newer for C++20.

If C:\MinGW appears above, that is the old MinGW.org toolchain. It is 32-bit
and years too old, and a 32-bit process cannot memory-map a multi-gigabyte
checkpoint in any case. Leave it alone and install MSYS2 alongside it:

  winget install MSYS2.MSYS2

If that fails with exit code 1, MSYS2 is already installed somewhere this
script did not look. Find it and install the toolchain into it:

  & "<msys2>\usr\bin\bash.exe" -lc "pacman -Sy --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-ninja"

Otherwise open "MSYS2 MINGW64" from the Start menu and run:

  pacman -Syu
  pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja

After that this script finds it automatically, from any shell.

If your compiler is somewhere unusual, name it directly:

  powershell -File scripts\build.ps1 -CompilerPath D:\path\to\g++.exe
"@
        exit 1
    }

    Write-Host "Compiler: $($chosen.Path)" -ForegroundColor Green
    Write-Host "          GCC $($chosen.Version), target $($chosen.Machine)" -ForegroundColor DarkGray

    # Put the chosen toolchain first on PATH for this session, so CMake's own
    # probes and the linker see a consistent set of tools.
    $compilerDirectory = Split-Path -Parent $chosen.Path
    $env:PATH = "$compilerDirectory;$env:PATH"

    # CMake wants forward slashes in a path given on the command line.
    $cmakeCompiler = $chosen.Path -replace '\\', '/'
    $compilerArgument = @("-DCMAKE_CXX_COMPILER=$cmakeCompiler")

    $generator = if (Get-Command ninja -ErrorAction SilentlyContinue) { "Ninja" } else { "MinGW Makefiles" }
} else {
    $generator = "Visual Studio 17 2022"
}

# A build directory remembers the generator it was configured with, and CMake
# refuses to reconfigure with a different one. That is a stale-state problem,
# not something the user did wrong, so clear it rather than reporting it.
$cachePath = Join-Path $BuildDirectory "CMakeCache.txt"
if (Test-Path $cachePath) {
    $cachedLine = Select-String -Path $cachePath -Pattern "^CMAKE_GENERATOR:INTERNAL=" |
        Select-Object -First 1
    if ($cachedLine) {
        $cached = $cachedLine.Line -replace "^CMAKE_GENERATOR:INTERNAL=", ""
        if ($cached -ne $generator) {
            Write-Host "The build directory was configured with '$cached', not '$generator'." -ForegroundColor Yellow
            Write-Host "Clearing it and configuring again." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $BuildDirectory
        }
    }
}

Write-Host "`nConfiguring with $generator" -ForegroundColor Cyan
if ($Toolchain -eq "MinGW") {
    cmake -S . -B $BuildDirectory -G $generator -DCMAKE_BUILD_TYPE=Release @compilerArgument
} else {
    # The Visual Studio generator picks the configuration at build time.
    cmake -S . -B $BuildDirectory -G $generator -A x64
}

if ($LASTEXITCODE -ne 0) { Write-Error "Configuration failed."; exit 1 }

Write-Host "`nBuilding" -ForegroundColor Cyan
cmake --build $BuildDirectory --config Release --parallel
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed."; exit 1 }

if ($RunTests) {
    Write-Host "`nRunning tests" -ForegroundColor Cyan
    ctest --test-dir $BuildDirectory -C Release --output-on-failure
    if ($LASTEXITCODE -ne 0) { Write-Error "Tests failed."; exit 1 }
}

$executable = Get-ChildItem -Path $BuildDirectory -Recurse -Filter "LiteMind.exe" |
    Select-Object -First 1
if (-not $executable) {
    Write-Warning "The build reported success but LiteMind.exe was not found."
    exit 1
}

Write-Host "`nBuilt: $($executable.FullName)" -ForegroundColor Green

# Starting the executable is a separate thing from building it. A missing
# runtime DLL makes Windows fail to start the process, and PowerShell reports
# that as silence rather than as an error, so check it here where the cause is
# still obvious.
Write-Host "Checking that it starts... " -NoNewline
$null = & $executable.FullName --help 2>&1
$startCode = $LASTEXITCODE

if ($startCode -eq 0) {
    Write-Host "ok" -ForegroundColor Green
} elseif ($startCode -eq -1073741515) {
    # 0xC0000135, STATUS_DLL_NOT_FOUND.
    Write-Host "failed" -ForegroundColor Red
    Write-Error @"
The executable was built but cannot start: a runtime DLL is missing.

This is why running it prints nothing at all. Windows could not start the
process, and PowerShell does not report that as an error.

The build links the runtime statically to avoid exactly this. If you are
seeing it anyway, rebuild from scratch:

  powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Clean

If it persists, add the compiler's bin directory to PATH for this session:

  `$env:PATH = "$compilerDirectory;`$env:PATH"
"@
    exit 1
} else {
    Write-Host "failed (exit $startCode)" -ForegroundColor Red
    Write-Error "The executable was built but exited with $startCode when asked for --help."
    exit 1
}

Write-Host "`nNext:" -ForegroundColor Cyan
Write-Host "  python tools\make_test_model.py models-test   # a few MB, checks the build"
Write-Host "  $($executable.FullName) models-test -p `"hello`" -n 8"
Write-Host "  powershell -File scripts\download_model.ps1   # the real 31 GB weights"
