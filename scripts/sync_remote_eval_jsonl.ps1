param(
    [string]$HostName = "103.207.149.91",
    [int]$Port = 18659,
    [string]$User = "root",
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\id_rsa",
    [string]$DestinationRoot = "artifacts\evals\remote_jsonl",
    [string]$ManifestPath = "artifacts\evals\remote_jsonl_manifest.json"
)

$ErrorActionPreference = "Stop"

function Convert-RemotePathToLocalPath {
    param([string]$RemotePath)
    $trimmed = $RemotePath.TrimStart("/")
    return Join-Path $DestinationRoot ($trimmed -replace "/", "\")
}

function Invoke-WithRetry {
    param(
        [scriptblock]$Action,
        [int]$Attempts = 4,
        [int]$DelaySeconds = 5,
        [string]$Description = "operation"
    )
    $lastError = $null
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            return & $Action
        } catch {
            $lastError = $_
            if ($attempt -lt $Attempts) {
                Write-Warning "${Description} failed on attempt $attempt/${Attempts}: $($_.Exception.Message)"
                Start-Sleep -Seconds $DelaySeconds
            }
        }
    }
    throw $lastError
}

function Get-RemoteInventory {
    $findCommand = "find /workspace/chess_sft_checkpoints /root/chess_sft_checkpoints -type f -name '*.jsonl' -printf '%s`t%p`n' 2>/dev/null | sort -k2"
    $raw = Invoke-WithRetry -Description "remote inventory" -Action {
        $out = & ssh -o ConnectTimeout=30 -o ConnectionAttempts=1 -o BatchMode=yes -o LogLevel=ERROR `
            "$User@$HostName" -p $Port -i $IdentityFile $findCommand
        if ($LASTEXITCODE -ne 0) {
            throw "remote inventory failed with exit code $LASTEXITCODE"
        }
        $out
    }
    foreach ($line in $raw) {
        if (-not $line.Trim()) { continue }
        $parts = $line -split "`t", 2
        if ($parts.Count -ne 2) { continue }
        [pscustomobject]@{
            size = [int64]$parts[0]
            remote_path = $parts[1]
            local_path = Convert-RemotePathToLocalPath $parts[1]
        }
    }
}

function Copy-RemoteFile {
    param(
        [string]$RemotePath,
        [string]$LocalPath
    )
    $localDir = Split-Path -Parent $LocalPath
    New-Item -ItemType Directory -Force -Path $localDir | Out-Null

    try {
        Invoke-WithRetry -Attempts 2 -DelaySeconds 3 -Description "scp $RemotePath" -Action {
            & scp -P $Port -i $IdentityFile -o ConnectTimeout=45 -o BatchMode=yes -o LogLevel=ERROR `
                "$User@$HostName`:$RemotePath" $LocalPath
            if ($LASTEXITCODE -ne 0) {
                throw "scp failed with exit code $LASTEXITCODE"
            }
        }
        return
    } catch {
        Write-Warning "scp fallback for ${RemotePath}: $($_.Exception.Message)"
    }

    $tmp = "$LocalPath.b64"
    if (Test-Path $tmp) {
        Remove-Item $tmp -Force
    }
    Invoke-WithRetry -Description "base64 copy $RemotePath" -Action {
        & ssh -o ConnectTimeout=60 -o ConnectionAttempts=1 -o BatchMode=yes -o LogLevel=ERROR `
            "$User@$HostName" -p $Port -i $IdentityFile "base64 -w0 '$RemotePath'" > $tmp
        if ($LASTEXITCODE -ne 0 -or (Get-Item $tmp).Length -eq 0) {
            Remove-Item $tmp -Force -ErrorAction SilentlyContinue
            throw "copy failed for $RemotePath"
        }
    }
    & certutil -f -decode $tmp $LocalPath | Out-Null
    Remove-Item $tmp -Force
}

$items = @(Get-RemoteInventory)
$results = @()
foreach ($item in $items) {
    $status = "copied"
    try {
        if ((Test-Path $item.local_path) -and ((Get-Item $item.local_path).Length -eq $item.size)) {
            $status = "already_present"
        } else {
            Copy-RemoteFile -RemotePath $item.remote_path -LocalPath $item.local_path
        }
        $actualSize = if (Test-Path $item.local_path) { (Get-Item $item.local_path).Length } else { 0 }
        if ($actualSize -ne $item.size) {
            throw "size mismatch expected=$($item.size) actual=$actualSize"
        }
        $results += [pscustomobject]@{
            status = $status
            size = $item.size
            remote_path = $item.remote_path
            local_path = $item.local_path
        }
        Write-Host "$status $($item.remote_path)"
    } catch {
        $results += [pscustomobject]@{
            status = "failed"
            size = $item.size
            remote_path = $item.remote_path
            local_path = $item.local_path
            error = $_.Exception.Message
        }
        Write-Warning "failed $($item.remote_path): $($_.Exception.Message)"
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ManifestPath) | Out-Null
[pscustomobject]@{
    synced_at = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    host = "$User@$HostName`:$Port"
    destination_root = $DestinationRoot
    total_remote_jsonl = $items.Count
    copied_or_present = @($results | Where-Object { $_.status -ne "failed" }).Count
    failed = @($results | Where-Object { $_.status -eq "failed" }).Count
    files = $results
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $ManifestPath

Write-Host "manifest $ManifestPath"
