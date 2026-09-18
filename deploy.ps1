<#
.SYNOPSIS
    One-command deploy on Windows: secrets, index, server.

.DESCRIPTION
    Generates the secrets on first run, builds the index from whatever is in
    original/, and serves it. Safe to re-run -- it is how you pick up a newly
    added PDF, and it never regenerates secrets that already exist.

.EXAMPLE
    .\deploy.ps1
    Local, or behind a tunnel.

.EXAMPLE
    .\deploy.ps1 -Tls
    Also run Caddy for a public HTTPS domain (set DOMAIN in .env first).
#>
[CmdletBinding()]
param([switch]$Tls)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker is not installed: https://docs.docker.com/get-docker/"
}

function New-Secret {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    [Convert]::ToBase64String($bytes) -replace '[/+=]', ''
}

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    # A passphrase people have to type gets words, not base64. Everything
    # machines read gets full entropy.
    $chars = 'abcdefghijkmnopqrstuvwxyz'.ToCharArray()
    $words = -join (1..12 | ForEach-Object { $chars | Get-Random })
    $phrase = "$($words.Substring(0,4))-$($words.Substring(4,4))-$($words.Substring(8,4))"
    (Get-Content .env) `
        -replace '^MCP_TEAM_PASSWORD=.*', "MCP_TEAM_PASSWORD=$phrase" `
        -replace '^QDRANT_API_KEY=.*',    "QDRANT_API_KEY=$(New-Secret)" `
        -replace '^MCP_AUTH_TOKEN=.*',    "MCP_AUTH_TOKEN=$(New-Secret)" |
        Set-Content .env -Encoding utf8
    Write-Host "wrote .env with fresh secrets"
}

$profileArgs = if ($Tls) { @('--profile', 'tls') } else { @() }

Write-Host "building the index and starting the stack (first run downloads ~2.8 GB of models)"
& docker compose @profileArgs up -d --build
if ($LASTEXITCODE -ne 0) { throw "docker compose failed" }

Write-Host ""
Write-Host "--- connect Claude to it ----------------------------------------------"
Select-String -Path .env -Pattern '^(MCP_PUBLIC_URL|MCP_TEAM_PASSWORD)=' | ForEach-Object { $_.Line }
Write-Host "Add <MCP_PUBLIC_URL>/mcp as a custom connector; the passphrase is the login."
Write-Host ""
Write-Host "logs:    docker compose logs -f mcp builder"
Write-Host "rebuild: drop a PDF in original\ and run .\deploy.ps1 again"
