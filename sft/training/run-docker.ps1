$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$defaultRuntimeRoot = Join-Path $repoRoot ".runtime"
$envPath = Join-Path $scriptDir ".env"

function Resolve-HostPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $scriptDir $Value))
}

$config = @{
    CUDA_DEVICE_ID = "0"
    CHESS_SFT_OUTPUT_HOST = Join-Path $defaultRuntimeRoot "chess_sft_data"
    CHESS_SFT_CHECKPOINTS_HOST = Join-Path $defaultRuntimeRoot "chess_sft_checkpoints"
    HF_CACHE_HOST = Join-Path $defaultRuntimeRoot "hf_cache"
    WANDB_HOST_DIR = Join-Path $defaultRuntimeRoot "wandb"
    CHESS_SFT_BASE_MODEL = "Qwen/Qwen3.5-0.8B"
    STOCKFISH_PATH = "/usr/games/stockfish"
    HF_TOKEN = ""
    WANDB_API_KEY = ""
    WANDB_MODE = ""
    WANDB_ENTITY = ""
    WANDB_BASE_URL = ""
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
            if ($config.ContainsKey($key)) {
                $config[$key] = $value
            }
        }
    }
}

foreach ($key in @($config.Keys)) {
    $envValue = [Environment]::GetEnvironmentVariable($key)
    if (-not [string]::IsNullOrWhiteSpace($envValue)) {
        $config[$key] = $envValue
    }
}

foreach ($pathKey in @("CHESS_SFT_OUTPUT_HOST", "CHESS_SFT_CHECKPOINTS_HOST", "HF_CACHE_HOST", "WANDB_HOST_DIR")) {
    $config[$pathKey] = Resolve-HostPath $config[$pathKey]
}

foreach ($pathKey in @("CHESS_SFT_OUTPUT_HOST", "CHESS_SFT_CHECKPOINTS_HOST", "HF_CACHE_HOST", "WANDB_HOST_DIR")) {
    New-Item -ItemType Directory -Force -Path $config[$pathKey] | Out-Null
}

if (-not $args -or $args.Count -eq 0) {
    $CommandArgs = @("bash")
} else {
    $CommandArgs = @($args)
}

$dockerEnvArgs = @(
    "-e", "CUDA_VISIBLE_DEVICES=$($config["CUDA_DEVICE_ID"])",
    "-e", "CHESS_SFT_OUTPUT=/data/chess_sft_data",
    "-e", "CHESS_SFT_CHECKPOINTS=/data/chess_sft_checkpoints",
    "-e", "CHESS_SFT_BASE_MODEL=$($config["CHESS_SFT_BASE_MODEL"])",
    "-e", "HF_HOME=/cache/huggingface",
    "-e", "STOCKFISH_PATH=$($config["STOCKFISH_PATH"])",
    "-e", "WANDB_DIR=/data/wandb"
)

if ($config["HF_TOKEN"]) {
    $dockerEnvArgs += @("-e", "HF_TOKEN=$($config["HF_TOKEN"])")
}

if ($config["WANDB_API_KEY"]) {
    $dockerEnvArgs += @("-e", "WANDB_API_KEY=$($config["WANDB_API_KEY"])")
}
if ($config["WANDB_MODE"]) {
    $dockerEnvArgs += @("-e", "WANDB_MODE=$($config["WANDB_MODE"])")
}
if ($config["WANDB_ENTITY"]) {
    $dockerEnvArgs += @("-e", "WANDB_ENTITY=$($config["WANDB_ENTITY"])")
}
if ($config["WANDB_BASE_URL"]) {
    $dockerEnvArgs += @("-e", "WANDB_BASE_URL=$($config["WANDB_BASE_URL"])")
}

$dockerArgs = @(
    "run",
    "--rm",
    "--gpus", "all"
) + $dockerEnvArgs + @(
    "-v", "${repoRoot}:/workspace",
    "-v", "$($config["CHESS_SFT_OUTPUT_HOST"]):/data/chess_sft_data",
    "-v", "$($config["CHESS_SFT_CHECKPOINTS_HOST"]):/data/chess_sft_checkpoints",
    "-v", "$($config["HF_CACHE_HOST"]):/cache/huggingface",
    "-v", "$($config["WANDB_HOST_DIR"]):/data/wandb",
    "-w", "/workspace",
    "chess-sft-training:latest"
)

if ($CommandArgs.Count -eq 1 -and $CommandArgs[0] -eq "bash") {
    $dockerArgs = @($dockerArgs[0..1] + @("-it") + $dockerArgs[2..($dockerArgs.Count - 1)])
}

Write-Host "Launching Docker training container with CUDA_VISIBLE_DEVICES=$($config["CUDA_DEVICE_ID"])" -ForegroundColor Cyan
$redactedArgs = @()
for ($i = 0; $i -lt $dockerArgs.Count; $i++) {
    $arg = $dockerArgs[$i]
    if ($arg -eq "-e" -and $i + 1 -lt $dockerArgs.Count) {
        $nextArg = $dockerArgs[$i + 1]
        if ($nextArg -like "HF_TOKEN=*") {
            $redactedArgs += @("-e", "HF_TOKEN=***")
            $i++
            continue
        }
        if ($nextArg -like "WANDB_API_KEY=*") {
            $redactedArgs += @("-e", "WANDB_API_KEY=***")
            $i++
            continue
        }
    }
    $redactedArgs += $arg
}
Write-Host "Command: docker $($redactedArgs + $CommandArgs -join ' ')" -ForegroundColor DarkGray

& docker @dockerArgs @CommandArgs
exit $LASTEXITCODE
