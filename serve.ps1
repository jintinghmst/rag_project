<#
.SYNOPSIS
    Start the signal-integrity MCP server, with a Cloudflare tunnel, in one command.

.DESCRIPTION
    Handles the parts that are easy to get wrong by hand: killing stale servers,
    starting the tunnel first so its random hostname can be fed to the server as
    MCP_PUBLIC_URL and MCP_ALLOWED_HOSTS, and printing the URL to paste into
    Claude. Ctrl+C stops both processes.

    The passphrase is remembered in .serve.local.json (gitignored) so it only has
    to be typed once.

.EXAMPLE
    .\serve.ps1
    Start with a tunnel, reusing the saved passphrase.

.EXAMPLE
    .\serve.ps1 -Password "my team phrase"
    Set (and save) a new passphrase.

.EXAMPLE
    .\serve.ps1 -Ngrok -Domain my-lab-rag.ngrok-free.app
    Stable URL via ngrok. The domain is remembered, so later runs are just
    `.\serve.ps1 -Ngrok`. Reserve the domain first at dashboard.ngrok.com and run
    `ngrok config add-authtoken <token>` once.

.EXAMPLE
    .\serve.ps1 -Url rag.mylab.edu
    Use a public URL you already terminate yourself (named Cloudflare tunnel,
    Tailscale Funnel, reverse proxy). Nothing is launched.

.EXAMPLE
    .\serve.ps1 -NoTunnel
    Serve on localhost only -- for running the test scripts, no public URL.

.EXAMPLE
    .\serve.ps1 -Token
    Bearer-token mode instead of OAuth, for check_remote.py. Claude's connector
    UI will NOT accept this mode; it is for scripted checks only.
#>
[CmdletBinding()]
param(
    [string]$Password,
    [int]$Port = 8000,
    [switch]$NoTunnel,
    [switch]$Token,
    [switch]$NoRerank,
    [switch]$Ngrok,
    [string]$Domain,
    [string]$Url,
    [string]$CfTunnel
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$server = Join-Path $root 'scripts\mcp_server.py'
$stateFile = Join-Path $root '.serve.local.json'
$tunnelLog = Join-Path $env:TEMP 'si-mcp-tunnel.log'

if (-not (Test-Path $python)) { throw "venv not found at $python -- create it first" }
if (-not (Test-Path $server)) { throw "server not found at $server" }

# ---- saved settings ---------------------------------------------------------
# One merged store, so saving the passphrase never discards the ngrok domain and
# vice versa.
$state = @{}
if (Test-Path $stateFile) {
    $json = Get-Content $stateFile -Raw | ConvertFrom-Json
    foreach ($prop in $json.PSObject.Properties) { $state[$prop.Name] = $prop.Value }
}
function Save-State {
    $state | ConvertTo-Json | Set-Content $stateFile -Encoding utf8
}

if (-not $Password) {
    if ($state.password) { $Password = $state.password }
    elseif ($env:MCP_TEAM_PASSWORD) { $Password = $env:MCP_TEAM_PASSWORD }
}
if (-not $Password -and -not $Token) {
    $secure = Read-Host 'Team passphrase (saved for next time)' -AsSecureString
    $Password = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
}
if ($Password) {
    $state['password'] = $Password
    Save-State
}

# ---- stop anything already running -----------------------------------------
$stale = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*mcp_server*' }
foreach ($p in $stale) {
    Write-Host "stopping stale server (pid $($p.ProcessId))" -ForegroundColor DarkGray
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 500

# ---- tunnel -----------------------------------------------------------------
$tunnelProc = $null
$publicUrl = "http://127.0.0.1:$Port"
$allowedHost = "127.0.0.1:$Port"

function Resolve-Exe([string]$name, [string[]]$guesses) {
    $cmd = (Get-Command $name -ErrorAction SilentlyContinue).Source
    if ($cmd) { return $cmd }
    foreach ($g in $guesses) { if (Test-Path $g) { return $g } }
    return $null
}

if ($Url) {
    # an already-stable public URL (named cloudflare tunnel, Tailscale Funnel,
    # a reverse proxy): nothing to launch, just point the server at it
    if ($Url -notmatch '^https?://') { $Url = "https://$Url" }
    $publicUrl = $Url.TrimEnd('/')
    $allowedHost = ([Uri]$publicUrl).Host
    Write-Host "using fixed public URL: $publicUrl" -ForegroundColor Cyan
}
elseif ($CfTunnel -or $state.cftunnel) {
    # A *named* Cloudflare tunnel: persistent hostname, because it is a real DNS
    # record in a zone you own. Requires the one-time setup in deploy/README.md
    # (cloudflared tunnel login / create / route dns).
    if (-not $CfTunnel) { $CfTunnel = $state.cftunnel }
    if (-not $Domain -and $state.cfdomain) { $Domain = $state.cfdomain }
    if (-not $Domain) {
        throw "-CfTunnel needs -Domain <hostname> the first time, e.g. rag.yourdomain.edu"
    }
    $Domain = ($Domain -replace '^https?://', '').TrimEnd('/')
    $state['cftunnel'] = $CfTunnel
    $state['cfdomain'] = $Domain
    Save-State

    $cf = Resolve-Exe 'cloudflared' @('C:\Program Files (x86)\cloudflared\cloudflared.exe')
    if (-not $cf) { throw "cloudflared not found. winget install --id Cloudflare.cloudflared" }

    Remove-Item $tunnelLog -ErrorAction SilentlyContinue
    Write-Host "starting named tunnel '$CfTunnel' -> $Domain ..." -ForegroundColor Cyan
    $tunnelProc = Start-Process -FilePath $cf `
        -ArgumentList 'tunnel', 'run', '--url', "http://localhost:$Port", $CfTunnel `
        -RedirectStandardError $tunnelLog -RedirectStandardOutput "$tunnelLog.out" `
        -WindowStyle Hidden -PassThru

    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Path $tunnelLog) {
            if (Select-String -Path $tunnelLog -Pattern 'Registered tunnel connection|Connection .* registered' -Quiet) {
                $ready = $true; break
            }
        }
        if ($tunnelProc.HasExited) { break }
    }
    if (-not $ready) {
        if (-not $tunnelProc.HasExited) {
            Stop-Process -Id $tunnelProc.Id -Force -ErrorAction SilentlyContinue
        }
        $detail = @()
        if (Test-Path $tunnelLog) {
            $detail = Get-Content $tunnelLog |
                Where-Object { $_ -match 'error|ERR|failed|not found|Cannot determine' } |
                Select-Object -First 3
        }
        $msg = "named tunnel did not connect."
        if ($detail) { $msg += "`n`n  " + ($detail -join "`n  ") }
        $msg += @"

  One-time setup, if you have not done it:
    cloudflared tunnel login
    cloudflared tunnel create $CfTunnel
    cloudflared tunnel route dns $CfTunnel $Domain

  Full log: $tunnelLog
"@
        throw $msg
    }
    $publicUrl = "https://$Domain"
    $allowedHost = $Domain
}
elseif ($Ngrok) {
    $ng = Resolve-Exe 'ngrok' @(
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\ngrok.exe"
    )
    if (-not $ng) { throw "ngrok not found. Install with: winget install --id Ngrok.Ngrok" }

    if (-not $Domain -and $state.domain) { $Domain = $state.domain }
    if ($Domain) {
        $Domain = ($Domain -replace '^https?://', '').TrimEnd('/')
        $state['domain'] = $Domain
        Save-State
    } else {
        Write-Host '  no reserved domain -- ngrok will assign a random one.' -ForegroundColor DarkYellow
        Write-Host '  Reserve one at dashboard.ngrok.com, then: .\serve.ps1 -Ngrok -Domain <it>' -ForegroundColor DarkYellow
    }

    Stop-Process -Name ngrok -Force -ErrorAction SilentlyContinue
    Remove-Item $tunnelLog -ErrorAction SilentlyContinue

    $ngArgs = @('http', "$Port", '--log', 'stdout')
    if ($Domain) { $ngArgs += "--domain=$Domain" }

    Write-Host 'starting ngrok...' -ForegroundColor Cyan
    $tunnelProc = Start-Process -FilePath $ng -ArgumentList $ngArgs `
        -RedirectStandardOutput $tunnelLog -RedirectStandardError "$tunnelLog.err" `
        -WindowStyle Hidden -PassThru

    # the local agent API is more reliable than scraping the log
    $found = $null
    $died = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $api = Invoke-RestMethod 'http://127.0.0.1:4040/api/tunnels' -TimeoutSec 2
            $t = $api.tunnels | Where-Object { $_.public_url -like 'https://*' } | Select-Object -First 1
            if ($t) { $found = $t.public_url; break }
        } catch { }
        # ngrok exits immediately on auth or domain errors; don't wait out the loop
        if ($tunnelProc.HasExited) { $died = $true; break }
    }
    if (-not $found) {
        if (-not $died) { Stop-Process -Id $tunnelProc.Id -Force -ErrorAction SilentlyContinue }

        $detail = @()
        foreach ($f in @("$tunnelLog.err", $tunnelLog)) {
            if (Test-Path $f) {
                $lines = Get-Content $f |
                    Where-Object { $_ -match 'ERR_NGROK_\d+|authentication failed|not authenticated|is not available|reserved' } |
                    Select-Object -First 3
                if ($lines) { $detail += $lines }
            }
        }
        $msg = "ngrok did not produce a URL."
        if ($detail) { $msg += "`n`n  " + ($detail -join "`n  ") }
        $msg += @"

  Most likely one of:
    1. No authtoken yet. Sign up at https://dashboard.ngrok.com/signup, then run once:
         ngrok config add-authtoken <your-token>
    2. The -Domain is not reserved on your account. Reserve it under
       Domains at dashboard.ngrok.com, or drop -Domain for a random URL.

  Full log: $tunnelLog
"@
        throw $msg
    }
    $publicUrl = $found.TrimEnd('/')
    $allowedHost = ([Uri]$publicUrl).Host
}
elseif (-not $NoTunnel) {
    $cf = Resolve-Exe 'cloudflared' @('C:\Program Files (x86)\cloudflared\cloudflared.exe')
    if (-not $cf) {
        throw "cloudflared not found. Install it with: winget install --id Cloudflare.cloudflared"
    }

    Remove-Item $tunnelLog -ErrorAction SilentlyContinue
    Write-Host 'starting tunnel (random URL -- use -Ngrok for a stable one)...' -ForegroundColor Cyan
    $tunnelProc = Start-Process -FilePath $cf `
        -ArgumentList 'tunnel', '--url', "http://localhost:$Port" `
        -RedirectStandardError $tunnelLog -RedirectStandardOutput "$tunnelLog.out" `
        -WindowStyle Hidden -PassThru

    $found = $null
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Path $tunnelLog) {
            $m = Select-String -Path $tunnelLog -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' |
                 Select-Object -First 1
            if ($m) { $found = $m.Matches[0].Value; break }
        }
    }
    if (-not $found) {
        if ($tunnelProc) { Stop-Process -Id $tunnelProc.Id -Force -ErrorAction SilentlyContinue }
        throw "tunnel did not produce a URL; see $tunnelLog"
    }
    $publicUrl = $found
    $allowedHost = ([Uri]$found).Host
}

# ---- environment ------------------------------------------------------------
$env:MCP_TRANSPORT     = 'streamable-http'
$env:MCP_HOST          = '127.0.0.1'
$env:MCP_PORT          = "$Port"
$env:MCP_PUBLIC_URL    = $publicUrl
$env:MCP_ALLOWED_HOSTS = $allowedHost
if ($NoRerank) { $env:RAG_RERANK = '0' } else { $env:RAG_RERANK = '1' }

if ($Token) {
    Remove-Item Env:MCP_TEAM_PASSWORD -ErrorAction SilentlyContinue
    if (-not $state.token) {
        $state['token'] = (& $python -c "import secrets;print(secrets.token_urlsafe(32))")
        Save-State
    }
    $env:MCP_AUTH_TOKEN = $state.token
} else {
    Remove-Item Env:MCP_AUTH_TOKEN -ErrorAction SilentlyContinue
    $env:MCP_TEAM_PASSWORD = $Password
}

Write-Host ''
Write-Host '  connector URL for Claude:' -ForegroundColor Green
Write-Host "    $publicUrl/mcp" -ForegroundColor White
if ($Token) {
    Write-Host "  bearer token: $($env:MCP_AUTH_TOKEN)" -ForegroundColor DarkGray
    Write-Host '  (token mode is for check_remote.py; Claude requires OAuth)' -ForegroundColor DarkYellow
} else {
    Write-Host '  sign in with the team passphrase when Claude asks.' -ForegroundColor DarkGray
}
Write-Host ''
Write-Host '  Ctrl+C stops the server and the tunnel.' -ForegroundColor DarkGray
Write-Host ''

# ---- run --------------------------------------------------------------------
try {
    & $python $server
} finally {
    if ($tunnelProc) {
        Stop-Process -Id $tunnelProc.Id -Force -ErrorAction SilentlyContinue
        Write-Host 'tunnel stopped.' -ForegroundColor DarkGray
    }
}
