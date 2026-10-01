<#
.SYNOPSIS
    Routine integrity check and backup. Install it as a weekly scheduled task.

.DESCRIPTION
    Runs `rag.py verify`, then `rag.py backup`, appending both to a log. Exits
    non-zero if verify found an integrity problem, so a scheduled task shows a
    failure rather than passing quietly.

    The point of the backup is that it holds what cannot be recomputed: the
    registry (titles, page ranges, slots -- the expensive scan) and the extracted
    text. Together they are ~170 MB compressed, against 42 GB of source PDFs. The
    index is not included by default; it rebuilds from the text, so it costs GPU
    time rather than being lost. Pass -WithIndex to snapshot it anyway and make a
    restore immediate.

.EXAMPLE
    .\maintain.ps1 -To E:\rag_backups
    Run once now.

.EXAMPLE
    .\maintain.ps1 -To E:\rag_backups -Install
    Register a weekly task (Sundays 03:00) that does the same. No admin needed.

.EXAMPLE
    .\maintain.ps1 -Uninstall
    Remove the scheduled task.
#>
[CmdletBinding()]
param(
    [string]$To = $env:RAG_BACKUP_DIR,
    [switch]$WithIndex,
    [int]$Keep = 3,
    [switch]$Install,
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$TaskName = 'SignalIntegrityRAG-Maintain'

if ($Uninstall) {
    schtasks /Delete /TN $TaskName /F
    return
}

if ($Install) {
    if (-not $To) { throw "-Install needs -To <backup directory>" }
    $ps = (Get-Command powershell).Source
    $args = "-NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\maintain.ps1`" -To `"$To`" -Keep $Keep"
    if ($WithIndex) { $args += " -WithIndex" }
    # ONWEEK needs no elevation, unlike ONLOGON with /RL HIGHEST
    schtasks /Create /TN $TaskName /TR "`"$ps`" $args" /SC WEEKLY /D SUN /ST 03:00 /F
    if ($LASTEXITCODE -ne 0) { throw "could not register the task" }
    Write-Host "`nregistered '$TaskName' -- Sundays 03:00"
    Write-Host "  run now  : schtasks /Run /TN $TaskName"
    Write-Host "  remove   : .\maintain.ps1 -Uninstall"
    Write-Host "  log      : $PSScriptRoot\data\maintain.log"
    return
}

if (-not $To) { throw "pass -To <directory>, or set RAG_BACKUP_DIR in .env" }

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$log = Join-Path $PSScriptRoot 'data\maintain.log'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

"=== $(Get-Date -Format s) verify ===" | Add-Content $log -Encoding utf8
& $python rag.py verify 2>&1 | Tee-Object -FilePath $log -Append
$verifyFailed = $LASTEXITCODE -ne 0

"=== $(Get-Date -Format s) backup ===" | Add-Content $log -Encoding utf8
$backupArgs = @('rag.py', 'backup', '--to', $To, '--keep', $Keep)
if ($WithIndex) { $backupArgs += '--with-index' }
& $python @backupArgs 2>&1 | Tee-Object -FilePath $log -Append
$backupFailed = $LASTEXITCODE -ne 0

if ($verifyFailed) { Write-Error "verify reported integrity problems - see $log"; exit 1 }
if ($backupFailed) { Write-Error "backup failed - see $log"; exit 1 }
Write-Host "`nverify and backup complete"
