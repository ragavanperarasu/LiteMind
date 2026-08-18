<#
.SYNOPSIS
    Downloads DeepSeek-V2-Lite from Hugging Face into a local directory.

.DESCRIPTION
    Fetches only the files LiteMind needs: config.json, the tokenizer, and the
    BF16 weight shards. About 31 GB, so make sure the target drive has room.

    The download resumes if it is interrupted, so a dropped connection costs
    only the shard that was in flight.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\download_model.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\download_model.ps1 -Destination D:\models\deepseek
#>
[CmdletBinding()]
param(
    [string]$Destination = "models",
    [string]$Repository  = "deepseek-ai/DeepSeek-V2-Lite",
    [string]$Revision    = "main"
)

$ErrorActionPreference = "Stop"

# The shard names come from the repository's file listing. LiteMind reads every
# .safetensors file it finds, so an extra or missing shard is caught at load.
$files = @(
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "model-00001-of-000004.safetensors",
    "model-00002-of-000004.safetensors",
    "model-00003-of-000004.safetensors",
    "model-00004-of-000004.safetensors"
)

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$resolved = (Resolve-Path $Destination).Path

Write-Host "Downloading $Repository into $resolved" -ForegroundColor Cyan
Write-Host "This is roughly 31 GB. Check you have the space before starting." -ForegroundColor Yellow

$drive = (Get-Item $resolved).PSDrive
if ($drive.Free -lt 35GB) {
    Write-Warning ("Drive {0}: has {1:N1} GB free. About 35 GB is needed." -f $drive.Name, ($drive.Free / 1GB))
    $answer = Read-Host "Continue anyway? (y/N)"
    if ($answer -ne "y") { exit 1 }
}

# curl.exe ships with Windows 10 1803 and later and resumes properly with -C -.
# PowerShell's Invoke-WebRequest buffers a whole response in memory, which fails
# on a 9 GB shard.
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    Write-Error "curl.exe was not found. It ships with Windows 10 1803 and later."
    exit 1
}

function Get-RepositoryFile {
    param([string]$Name, [switch]$Optional)

    $url = "https://huggingface.co/$Repository/resolve/$Revision/$Name"
    $target = Join-Path $resolved $Name

    Write-Host "`n-> $Name" -ForegroundColor Green
    # -C - resumes, -L follows the CDN redirect, -f fails loudly on a 404.
    & curl.exe -L -f -C - --retry 5 --retry-delay 5 -o $target $url

    # Exit code 33 means the server refused a resume because the file is already
    # complete, which is success as far as this script is concerned.
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 33) {
        if ($Optional) {
            Write-Host "   (skipped: not present in this repository)" -ForegroundColor DarkGray
            Remove-Item $target -ErrorAction SilentlyContinue
            return $false
        }
        Write-Error "Failed to download $Name (curl exit $LASTEXITCODE)."
        exit 1
    }
    return $true
}

# The small files first, so a wrong repository name fails in seconds rather
# than after several gigabytes.
foreach ($file in @("config.json", "tokenizer.json", "tokenizer_config.json")) {
    [void](Get-RepositoryFile -Name $file)
}
foreach ($file in @("generation_config.json", "model.safetensors.index.json")) {
    [void](Get-RepositoryFile -Name $file -Optional)
}

# Derive the shard list from the index rather than hard-coding names, so this
# keeps working if the repository is ever re-sharded.
$indexPath = Join-Path $resolved "model.safetensors.index.json"
if (Test-Path $indexPath) {
    $index = Get-Content $indexPath -Raw | ConvertFrom-Json
    $shards = $index.weight_map.PSObject.Properties.Value | Sort-Object -Unique
    Write-Host "`nThe index lists $($shards.Count) weight shard(s)." -ForegroundColor Cyan
} else {
    # A single-file repository has no index.
    $shards = @("model.safetensors")
    Write-Host "`nNo index found; assuming a single model.safetensors." -ForegroundColor Cyan
}

foreach ($file in $shards) {
    [void](Get-RepositoryFile -Name $file)
}

Write-Host "`nDone. Check what arrived with:" -ForegroundColor Cyan
Write-Host "  .\build\bin\LiteMind.exe $Destination --inspect"
