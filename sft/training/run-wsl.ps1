 [CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Distro = "Ubuntu-20.04",
    [string]$WslRepoPath = "/home/vince/code/chess_sft_sdpo",
    [string]$CondaEnv = "chess-sft-wsl",
    [string]$CudaDeviceId = "1",
    [switch]$NoSync,
    [Parameter(ValueFromRemainingArguments = $true, Position = 0)]
    [string[]]$CommandArgs
)

$ErrorActionPreference = "Stop"

function Quote-Bash([string]$Value) {
    return "'" + $Value.Replace("'", "'\''") + "'"
}

function ConvertTo-WslPath([string]$WindowsPath) {
    if ($WindowsPath -notmatch "^([A-Za-z]):\\(.*)$") {
        throw "Only absolute drive-letter paths are supported: $WindowsPath"
    }
    $drive = $Matches[1].ToLowerInvariant()
    $rest = $Matches[2].Replace("\", "/")
    return "/mnt/$drive/$rest"
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$repoRootWsl = ConvertTo-WslPath $repoRoot.Path

if (-not $NoSync) {
    $rsyncCommand = @(
        "rsync -a",
        "--exclude 'sft/make_data/data/'",
        "--exclude 'sft/make_data/polyglot_opening_books/*.bin'",
        "--exclude '**/.pytest_cache/'",
        "--exclude '**/__pycache__/'",
        "--exclude '.claude/'",
        "$(Quote-Bash ($repoRootWsl.TrimEnd('/') + '/'))",
        "$(Quote-Bash ($WslRepoPath.TrimEnd('/') + '/'))"
    ) -join " "
    $syncCommand = "mkdir -p $(Quote-Bash $WslRepoPath) && $rsyncCommand"
    Write-Host "Syncing repo to WSL: $WslRepoPath" -ForegroundColor Cyan
    & wsl.exe -d $Distro -- bash -lc $syncCommand
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

& wsl.exe -d $Distro -- bash -lc "mkdir -p /mnt/e/chess_sft_data /mnt/e/chess_sft_checkpoints /mnt/e/hf_cache /mnt/e/wandb_cache"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $CommandArgs -or $CommandArgs.Count -eq 0) {
    $CommandArgs = @("bash")
}

$command = ($CommandArgs | ForEach-Object { Quote-Bash $_ }) -join " "
$envPairs = @(
    "CUDA_VISIBLE_DEVICES=$(Quote-Bash $CudaDeviceId)",
    "CHESS_SFT_OUTPUT=/mnt/e/chess_sft_data",
    "CHESS_SFT_CHECKPOINTS=/mnt/e/chess_sft_checkpoints",
    "HF_HOME=/mnt/e/hf_cache",
    "WANDB_DIR=/mnt/e/wandb_cache",
    "STOCKFISH_PATH=/mnt/e/stockfish/stockfish"
)

if ($env:CHESS_SFT_BASE_MODEL) {
    $envPairs += "CHESS_SFT_BASE_MODEL=$(Quote-Bash $env:CHESS_SFT_BASE_MODEL)"
}
if ($env:HF_TOKEN) {
    $envPairs += "HF_TOKEN=$(Quote-Bash $env:HF_TOKEN)"
}
if ($env:WANDB_API_KEY) {
    $envPairs += "WANDB_API_KEY=$(Quote-Bash $env:WANDB_API_KEY)"
}

$runCommand = @(
    "cd $(Quote-Bash ($WslRepoPath.TrimEnd('/') + '/sft/training'))",
    "env $($envPairs -join ' ') ~/miniconda3/bin/conda run -n $(Quote-Bash $CondaEnv) $command"
) -join " && "

Write-Host "Running in WSL $Distro with CUDA_VISIBLE_DEVICES=$CudaDeviceId" -ForegroundColor Cyan
& wsl.exe -d $Distro -- bash -lc $runCommand
exit $LASTEXITCODE
