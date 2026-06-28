$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$envPath = Join-Path $scriptDir ".env"

$config = @{
    CUDA_DEVICE_ID = $(if ($env:EVAL_CUDA_DEVICE_ID) { $env:EVAL_CUDA_DEVICE_ID } else { "1" })
    CHESS_SFT_OUTPUT_HOST = "E:/chess_sft_data"
    CHESS_SFT_CHECKPOINTS_HOST = "E:/chess_sft_checkpoints"
    HF_CACHE_HOST = "E:/hf_cache"
    WANDB_HOST_DIR = "E:/wandb"
    CHESS_SFT_BASE_MODEL = $(if ([string]::IsNullOrWhiteSpace($env:CHESS_SFT_BASE_MODEL)) { "Qwen/Qwen3-0.6B" } else { $env:CHESS_SFT_BASE_MODEL })
    HF_TOKEN = $env:HF_TOKEN
    WANDB_API_KEY = $env:WANDB_API_KEY
    IMAGE_NAME = "chess-sft-eval-vllm:latest"
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

foreach ($pathKey in @("CHESS_SFT_OUTPUT_HOST", "CHESS_SFT_CHECKPOINTS_HOST", "HF_CACHE_HOST", "WANDB_HOST_DIR")) {
    New-Item -ItemType Directory -Force -Path $config[$pathKey] | Out-Null
}

if (-not $args -or $args.Count -eq 0) {
    $CommandArgs = @("bash")
} else {
    $CommandArgs = @($args)
}

$dockerArgs = @(
    "run",
    "--rm",
    "--gpus", "all",
    "-e", "CUDA_VISIBLE_DEVICES=$($config["CUDA_DEVICE_ID"])",
    "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
    "-e", "CHESS_SFT_OUTPUT=/data/chess_sft_data",
    "-e", "CHESS_SFT_CHECKPOINTS=/data/chess_sft_checkpoints",
    "-e", "CHESS_SFT_BASE_MODEL=$($config["CHESS_SFT_BASE_MODEL"])",
    "-e", "HF_HOME=/cache/huggingface",
    "-e", "STOCKFISH_PATH=/usr/games/stockfish",
    "-e", "WANDB_DIR=/data/wandb",
    "-v", "${repoRoot}:/workspace",
    "-v", "$($config["CHESS_SFT_OUTPUT_HOST"]):/data/chess_sft_data",
    "-v", "$($config["CHESS_SFT_CHECKPOINTS_HOST"]):/data/chess_sft_checkpoints",
    "-v", "$($config["HF_CACHE_HOST"]):/cache/huggingface",
    "-v", "$($config["WANDB_HOST_DIR"]):/data/wandb",
    "-w", "/workspace/sft/training",
    "$($config["IMAGE_NAME"])"
)

if ($config["HF_TOKEN"]) {
    $dockerArgs += @("-e", "HF_TOKEN=$($config["HF_TOKEN"])")
}

if ($config["WANDB_API_KEY"]) {
    $dockerArgs += @("-e", "WANDB_API_KEY=$($config["WANDB_API_KEY"])")
}

if ($CommandArgs.Count -eq 1 -and $CommandArgs[0] -eq "bash") {
    $dockerArgs = @($dockerArgs[0..1] + @("-it") + $dockerArgs[2..($dockerArgs.Count - 1)])
}

Write-Host "Launching Docker eval container with CUDA_VISIBLE_DEVICES=$($config["CUDA_DEVICE_ID"])" -ForegroundColor Cyan
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
