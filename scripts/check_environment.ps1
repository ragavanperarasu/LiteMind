<#
.SYNOPSIS
    Reports whether this machine can build and run LiteMind.

.DESCRIPTION
    Checks everything in one pass instead of one failure at a time: compilers
    and their versions, CMake, Python, MSYS2, disk space, RAM and CPU. Changes
    nothing.

    Paste the output when asking for help; it answers most of the questions
    that would otherwise need a round trip.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\check_environment.ps1
#>
[CmdletBinding()]
param()

$minimumGccMajor = 10
$problems = @()
$warnings = @()

function Write-Section {
    param([string]$Title)
    Write-Host "`n$Title" -ForegroundColor Cyan
    Write-Host ("-" * $Title.Length) -ForegroundColor DarkGray
}

Write-Host "LiteMind environment check" -ForegroundColor White

# ── Host ────────────────────────────────────────────────────────────────────
Write-Section "Host"
$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
Write-Host "  Windows      : $($os.Caption) $($os.Version)"
Write-Host "  CPU          : $($cpu.Name.Trim())"
Write-Host "  Cores        : $($cpu.NumberOfCores) physical, $($cpu.NumberOfLogicalProcessors) logical"
$totalRam = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
Write-Host "  RAM          : $totalRam GB"
Write-Host "  PowerShell   : $($PSVersionTable.PSVersion)"

if ($totalRam -lt 6) {
    $warnings += "Only $totalRam GB of RAM. LiteMind streams weights from SSD, but about 2.6 GB stays resident. Use --expert-cache and a small --context."
}

# ── Compilers ───────────────────────────────────────────────────────────────
Write-Section "C++ compilers"

$candidates = @()
foreach ($drive in @("C:", "D:", "E:")) {
    foreach ($environment in @("mingw64", "ucrt64", "clang64")) {
        $candidates += "$drive\msys64\$environment\bin\g++.exe"
    }
    $candidates += "$drive\w64devkit\bin\g++.exe"
}
Get-Command g++ -All -ErrorAction SilentlyContinue |
    Where-Object { $_.Source } |
    ForEach-Object { $candidates += $_.Source }

$usable = @()
$found = @()
foreach ($candidate in ($candidates | Select-Object -Unique)) {
    if ([string]::IsNullOrWhiteSpace($candidate) -or -not (Test-Path $candidate)) { continue }

    $machine = (& $candidate -dumpmachine 2>$null | Select-Object -First 1)
    $version = (& $candidate -dumpversion 2>$null | Select-Object -First 1)
    if (-not $machine) { continue }

    $major = 0
    [void][int]::TryParse((($version -split '\.')[0]), [ref]$major)
    $is64 = $machine -match 'x86_64|aarch64'
    $ok = $is64 -and ($major -ge $minimumGccMajor)

    $found += $candidate
    if ($ok) {
        $usable += $candidate
        Write-Host "  [OK]   $candidate" -ForegroundColor Green
        Write-Host "         GCC $version, target $machine" -ForegroundColor DarkGray
    } else {
        $reason = if (-not $is64) { "32-bit, cannot map a large checkpoint" }
                  else { "GCC $version is older than $minimumGccMajor, no C++20" }
        Write-Host "  [NO]   $candidate" -ForegroundColor Yellow
        Write-Host "         GCC $version, target $machine - $reason" -ForegroundColor DarkGray
    }
}

$hasMsvc = $false
if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
    $hasMsvc = $true
    Write-Host "  [OK]   cl.exe on PATH (Visual Studio)" -ForegroundColor Green
} elseif (Test-Path "${env:ProgramFiles}\Microsoft Visual Studio\2022") {
    $hasMsvc = $true
    Write-Host "  [OK]   Visual Studio 2022 is installed" -ForegroundColor Green
}

if ($found.Count -eq 0 -and -not $hasMsvc) {
    Write-Host "  none found" -ForegroundColor Red
}
if ($usable.Count -eq 0 -and -not $hasMsvc) {
    $problems += "No usable C++ compiler. LiteMind needs 64-bit GCC $minimumGccMajor or newer, or Visual Studio 2019 16.11 or newer."
}

# ── MSYS2 ───────────────────────────────────────────────────────────────────
Write-Section "MSYS2"
$msys2 = $null
foreach ($candidate in @("C:\msys64", "D:\msys64", "E:\msys64", "$env:USERPROFILE\msys64")) {
    if ($candidate -and (Test-Path (Join-Path $candidate "usr\bin\bash.exe"))) {
        $msys2 = $candidate
        break
    }
}

if ($msys2) {
    Write-Host ("  {0,-13}: {1}" -f "installed at", $msys2) -ForegroundColor Green
    foreach ($environment in @("mingw64", "ucrt64", "clang64")) {
        $compiler = Join-Path $msys2 "$environment\bin\g++.exe"
        $present = Test-Path $compiler
        $state = if ($present) { "toolchain present" } else { "toolchain not installed" }
        $colour = if ($present) { "Green" } else { "DarkGray" }
        Write-Host ("  {0,-13}: {1}" -f $environment, $state) -ForegroundColor $colour
    }
    if ($usable.Count -eq 0) {
        $problems += "MSYS2 is installed at $msys2 but has no 64-bit toolchain. Install it with:`n" +
                     "    & `"$msys2\usr\bin\bash.exe`" -lc `"pacman -Sy --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-ninja`"`n" +
                     "  Do not re-run the MSYS2 installer: it refuses to write into an existing installation and exits with code 1."
    }
} else {
    Write-Host "  not found in the usual locations" -ForegroundColor DarkGray
}

# ── Build tools ─────────────────────────────────────────────────────────────
Write-Section "Build tools"

# Search the chosen toolchain's own bin directory as well as PATH: build.ps1
# prepends it, so a tool found here is a tool the build can use.
$toolDirectories = @()
if ($usable.Count -gt 0) {
    $toolDirectories += (Split-Path -Parent $usable[0])
}

foreach ($tool in @("cmake", "ninja", "python", "git")) {
    $resolved = $null
    $command = Get-Command $tool -ErrorAction SilentlyContinue
    if ($command) {
        $resolved = $command.Source
    } else {
        foreach ($directory in $toolDirectories) {
            $candidate = Join-Path $directory "$tool.exe"
            if (Test-Path $candidate) { $resolved = $candidate; break }
        }
    }

    if ($resolved) {
        $version = (& $resolved --version 2>$null | Select-Object -First 1)
        $note = if (-not $command) { "  (in the toolchain directory; build.ps1 adds it to PATH)" } else { "" }
        Write-Host ("  [OK]   {0,-7} {1}{2}" -f $tool, $version, $note) -ForegroundColor Green
    } else {
        $colour = if ($tool -eq "cmake") { "Red" } else { "DarkGray" }
        Write-Host ("  [--]   {0,-7} not found" -f $tool) -ForegroundColor $colour
    }
}
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    $problems += "CMake is not on PATH. Install it with 'winget install Kitware.CMake'."
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    $warnings += "Python is not on PATH. It is only needed for tools\make_test_model.py, which checks the build without a 31 GB download."
}

# ── Disk ────────────────────────────────────────────────────────────────────
Write-Section "Disk space"
foreach ($drive in (Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Free -ne $null })) {
    $free = [math]::Round($drive.Free / 1GB, 1)
    $colour = if ($free -ge 35) { "Green" } else { "DarkGray" }
    Write-Host "  $($drive.Name):  $free GB free" -ForegroundColor $colour
}
$roomy = Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Free -ge 35GB }
if (-not $roomy) {
    $warnings += "No drive has the ~35 GB the DeepSeek-V2-Lite weights need. The build and the tiny test model still work."
}

# ── Existing build ──────────────────────────────────────────────────────────
Write-Section "This checkout"
$root = Split-Path -Parent $PSScriptRoot
Write-Host "  Directory    : $root"
$executable = Get-ChildItem -Path (Join-Path $root "build") -Recurse -Filter "LiteMind.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($executable) {
    Write-Host "  Executable   : $($executable.FullName)"
    $null = & $executable.FullName --help 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Starts       : yes" -ForegroundColor Green
    } elseif ($LASTEXITCODE -eq -1073741515) {
        Write-Host "  Starts       : no, a runtime DLL is missing" -ForegroundColor Red
        $problems += "The built executable cannot start because a runtime DLL is missing. Rebuild with -Clean; the build links the runtime statically."
    } else {
        Write-Host "  Starts       : no, exit $LASTEXITCODE" -ForegroundColor Red
    }
} else {
    Write-Host "  Executable   : not built yet" -ForegroundColor DarkGray
}

# ── Verdict ─────────────────────────────────────────────────────────────────
Write-Section "Verdict"
if ($problems.Count -eq 0) {
    Write-Host "  Ready to build." -ForegroundColor Green
    Write-Host "`n    powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Clean -RunTests"
} else {
    Write-Host "  Not ready. Fix these:" -ForegroundColor Red
    foreach ($problem in $problems) {
        Write-Host "`n  * $problem" -ForegroundColor Yellow
    }
}
foreach ($warning in $warnings) {
    Write-Host "`n  Note: $warning" -ForegroundColor DarkYellow
}
Write-Host ""
