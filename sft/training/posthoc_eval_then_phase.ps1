$ErrorActionPreference = "Stop"

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("a", "b", "c")]
    [string]$Phase,

    [Parameter(Mandatory = $true)]
    [ValidateSet("a", "b", "c")]
    [string]$NextPhase,

    [string]$InferenceBackend = "transformers",
    [string]$WandbProject = "chess-sft",
    [string]$RunPrefix = "",
    [switch]$NoWandb,
    [switch]$BuildImage,
    [switch]$SkipEvalOnNextPhase
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$envPath = Join-Path $scriptDir ".env"

$config = @{
    CHESS_SFT_CHECKPOINTS_HOST = "E:/chess_sft_checkpoints"
}

if (Test-Path $envPath) {
    Get-Content $envPath | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }
        $parts = $line.Split("=", 2)
        if ($parts.Count -eq 2) {
            $key = $parts[0].Trim()
            $value = $parts[1].Trim()
            if (-not $config.ContainsKey($key) -or [string]::IsNullOrWhiteSpace($config[$key])) {
                $config[$key] = $value
            }
        }
    }
}

$phaseDir = Join-Path $config["CHESS_SFT_CHECKPOINTS_HOST"] "phase_$Phase"
$sentinelPath = Join-Path $phaseDir "PASSED"
$evalRunName = if ($RunPrefix) { "$RunPrefix-phase-$Phase-posthoc-eval" } else { "phase-$Phase-posthoc-eval" }
$nextRunName = if ($RunPrefix) { "$RunPrefix-phase-$NextPhase-train" } else { "phase-$NextPhase-train" }
$nextContainer = "chess-sft-phase-$NextPhase"

Push-Location $scriptDir
try {
    if ($BuildImage) {
        Write-Host "Building Docker image before post-hoc eval..." -ForegroundColor Cyan
        docker compose -f compose.yaml build
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }

    $evalArgs = @("python", "train.py", "--phase", $Phase, "--eval-only", "--inference-backend", $InferenceBackend)
    if ($NoWandb) {
        $evalArgs += "--no-wandb"
    } else {
        $evalArgs += @("--wandb-project", $WandbProject, "--run-name", $evalRunName)
    }

    Write-Host "Running post-hoc eval for phase $Phase..." -ForegroundColor Cyan
    & (Join-Path $scriptDir "run-docker.ps1") @evalArgs
    $evalRc = $LASTEXITCODE
    if ($evalRc -ne 0) {
        Write-Host "Phase $Phase eval failed with exit code $evalRc. Not launching phase $NextPhase." -ForegroundColor Red
        exit $evalRc
    }

    Set-Content -Path $sentinelPath -Value "Phase $Phase passed post-hoc eval on $(Get-Date -Format s).`n"
    Write-Host "Wrote phase gate sentinel: $sentinelPath" -ForegroundColor Green

    if (docker ps -a --format "{{.Names}}" | Select-String -SimpleMatch $nextContainer) {
        docker rm -f $nextContainer | Out-Null
    }

    $nextArgs = @(
        "compose", "-f", "compose.yaml", "run", "-d", "--name", $nextContainer,
        "trainer", "python", "train.py", "--phase", $NextPhase
    )
    if ($SkipEvalOnNextPhase) {
        $nextArgs += "--skip-eval"
    }
    if ($NoWandb) {
        $nextArgs += "--no-wandb"
    } else {
        $nextArgs += @("--wandb-project", $WandbProject, "--run-name", $nextRunName)
    }

    Write-Host "Launching phase $NextPhase in detached container $nextContainer..." -ForegroundColor Cyan
    & docker @nextArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
