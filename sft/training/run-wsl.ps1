[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Distro = "",
    [string]$WslRepoPath = "/tmp/chess_sft_sdpo",
    [string]$CondaEnv = "chess-sft-wsl",
    [string]$CudaDeviceId = "0",
    [string]$WslDataRoot = "/tmp/chess_sft_data",
    [string]$WslCheckpointRoot = "/tmp/chess_sft_checkpoints",
    [string]$WslHfCache = "/tmp/chess_sft_hf_cache",
    [string]$WslWandbDir = "/tmp/chess_sft_wandb",
    [string]$StockfishPath = "stockfish",
    [string]$CondaExecutable = "conda",
    [string]$VenvPath = "",
    [switch]$NoSync,
    [Parameter(ValueFromRemainingArguments = $true, Position = 0)]
    [string[]]$CommandArgs
)

$ErrorActionPreference = "Stop"
$ScriptBoundParameters = @{} + $PSBoundParameters

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

$wslExecutable = "wsl.exe"
if ($env:CHESS_SFT_WSL_EXE) {
    $wslExecutable = $env:CHESS_SFT_WSL_EXE
}

function Invoke-Wsl([string[]]$Arguments) {
    $wslArgs = @()
    if (-not [string]::IsNullOrWhiteSpace($Distro)) {
        $wslArgs += @("-d", $Distro)
    }
    $wslArgs += $Arguments

    $previousErrorActionPreference = $ErrorActionPreference
    $nativeCommandPreference = Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue
    $hasNativeCommandPreference = $null -ne $nativeCommandPreference
    $previousNativeCommandPreference = $null

    try {
        # WSL commands frequently log progress and warnings to stderr even when
        # they succeed. Keep that output visible without making it terminating.
        $ErrorActionPreference = "Continue"
        if ($hasNativeCommandPreference) {
            $previousNativeCommandPreference = $global:PSNativeCommandUseErrorActionPreference
            $global:PSNativeCommandUseErrorActionPreference = $false
        }
        & $wslExecutable @wslArgs
    } finally {
        if ($hasNativeCommandPreference) {
            $global:PSNativeCommandUseErrorActionPreference = $previousNativeCommandPreference
        }
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Invoke-WslWithRuntimeEnv([string[]]$Arguments, [hashtable]$Environment) {
    $previousValues = @{}
    foreach ($key in $Environment.Keys) {
        $previousValues[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable($key, [string]$Environment[$key], "Process")
    }

    $previousWslEnv = [Environment]::GetEnvironmentVariable("WSLENV", "Process")
    $wslEnvParts = @()
    if (-not [string]::IsNullOrWhiteSpace($previousWslEnv)) {
        $wslEnvParts += $previousWslEnv.Split(":") | Where-Object { $_ }
    }
    foreach ($key in $Environment.Keys) {
        if ($wslEnvParts -notcontains $key) {
            $wslEnvParts += $key
        }
    }
    [Environment]::SetEnvironmentVariable("WSLENV", ($wslEnvParts -join ":"), "Process")

    try {
        Invoke-Wsl $Arguments
    } finally {
        foreach ($key in $Environment.Keys) {
            [Environment]::SetEnvironmentVariable($key, $previousValues[$key], "Process")
        }
        [Environment]::SetEnvironmentVariable("WSLENV", $previousWslEnv, "Process")
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$repoRootWsl = ConvertTo-WslPath $repoRoot.Path
$envPath = Join-Path $scriptDir ".env"

$config = @{
    WSL_DISTRO = ""
    WSL_REPO_PATH = "/tmp/chess_sft_sdpo"
    WSL_CONDA_ENV = "chess-sft-wsl"
    CUDA_DEVICE_ID = "0"
    WSL_DATA_ROOT = "/tmp/chess_sft_data"
    WSL_CHECKPOINT_ROOT = "/tmp/chess_sft_checkpoints"
    WSL_HF_CACHE = "/tmp/chess_sft_hf_cache"
    WSL_WANDB_DIR = "/tmp/chess_sft_wandb"
    WSL_STOCKFISH_PATH = "stockfish"
    WSL_CONDA_EXECUTABLE = "conda"
    WSL_VENV_PATH = ""
    CHESS_SFT_BASE_MODEL = ""
    CHESS_SFT_ENABLE_FLASH_ATTENTION_4_AUTO = ""
    CHESS_SFT_ENABLE_HF_FLASH_ATTN2_AUTO = ""
    CHESS_SFT_DISABLE_HF_FLASH_ATTN2_AUTO = ""
    CUDA_DEVICE_ORDER = ""
    VLLM_USE_V2_MODEL_RUNNER = ""
    VLLM_WORKER_MULTIPROC_METHOD = ""
    VLLM_ENABLE_V1_MULTIPROCESSING = ""
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS = ""
    HF_TOKEN = ""
    HF_HUB_OFFLINE = ""
    HF_DATASETS_OFFLINE = ""
    HF_HUB_ENABLE_HF_TRANSFER = ""
    WANDB_API_KEY = ""
    WANDB_MODE = ""
    WANDB_ENTITY = ""
    WANDB_BASE_URL = ""
    WANDB_DISABLED = ""
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
            if ($value.Length -ge 2) {
                $first = $value.Substring(0, 1)
                $last = $value.Substring($value.Length - 1, 1)
                if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
                    $value = $value.Substring(1, $value.Length - 2)
                }
            }
            if ($config.ContainsKey($key)) {
                $config[$key] = $value
            }
        }
    }
}

function Resolve-ConfigDefault([string]$ParameterName, [string]$Key, [string]$CurrentValue) {
    if ($ScriptBoundParameters.ContainsKey($ParameterName)) {
        return $CurrentValue
    }
    if ($config.ContainsKey($Key) -and -not [string]::IsNullOrWhiteSpace([string]$config[$Key])) {
        return [string]$config[$Key]
    }
    return $CurrentValue
}

function Get-RuntimeValue([string]$Key) {
    switch ($Key) {
        "CHESS_SFT_BASE_MODEL" { if ($env:CHESS_SFT_BASE_MODEL) { return $env:CHESS_SFT_BASE_MODEL } }
        "CHESS_SFT_ENABLE_FLASH_ATTENTION_4_AUTO" { if ($env:CHESS_SFT_ENABLE_FLASH_ATTENTION_4_AUTO) { return $env:CHESS_SFT_ENABLE_FLASH_ATTENTION_4_AUTO } }
        "CHESS_SFT_ENABLE_HF_FLASH_ATTN2_AUTO" { if ($env:CHESS_SFT_ENABLE_HF_FLASH_ATTN2_AUTO) { return $env:CHESS_SFT_ENABLE_HF_FLASH_ATTN2_AUTO } }
        "CHESS_SFT_DISABLE_HF_FLASH_ATTN2_AUTO" { if ($env:CHESS_SFT_DISABLE_HF_FLASH_ATTN2_AUTO) { return $env:CHESS_SFT_DISABLE_HF_FLASH_ATTN2_AUTO } }
        "CUDA_DEVICE_ORDER" { if ($env:CUDA_DEVICE_ORDER) { return $env:CUDA_DEVICE_ORDER } }
        "VLLM_USE_V2_MODEL_RUNNER" { if ($env:VLLM_USE_V2_MODEL_RUNNER) { return $env:VLLM_USE_V2_MODEL_RUNNER } }
        "VLLM_WORKER_MULTIPROC_METHOD" { if ($env:VLLM_WORKER_MULTIPROC_METHOD) { return $env:VLLM_WORKER_MULTIPROC_METHOD } }
        "VLLM_ENABLE_V1_MULTIPROCESSING" { if ($env:VLLM_ENABLE_V1_MULTIPROCESSING) { return $env:VLLM_ENABLE_V1_MULTIPROCESSING } }
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS" { if ($env:VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS) { return $env:VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS } }
        "HF_TOKEN" { if ($env:HF_TOKEN) { return $env:HF_TOKEN } }
        "HF_HUB_OFFLINE" { if ($env:HF_HUB_OFFLINE) { return $env:HF_HUB_OFFLINE } }
        "HF_DATASETS_OFFLINE" { if ($env:HF_DATASETS_OFFLINE) { return $env:HF_DATASETS_OFFLINE } }
        "HF_HUB_ENABLE_HF_TRANSFER" { if ($env:HF_HUB_ENABLE_HF_TRANSFER) { return $env:HF_HUB_ENABLE_HF_TRANSFER } }
        "WANDB_API_KEY" { if ($env:WANDB_API_KEY) { return $env:WANDB_API_KEY } }
        "WANDB_MODE" { if ($env:WANDB_MODE) { return $env:WANDB_MODE } }
        "WANDB_ENTITY" { if ($env:WANDB_ENTITY) { return $env:WANDB_ENTITY } }
        "WANDB_BASE_URL" { if ($env:WANDB_BASE_URL) { return $env:WANDB_BASE_URL } }
        "WANDB_DISABLED" { if ($env:WANDB_DISABLED) { return $env:WANDB_DISABLED } }
        "WANDB_GIT_COMMIT" { if ($env:WANDB_GIT_COMMIT) { return $env:WANDB_GIT_COMMIT } }
    }

    if ($config.ContainsKey($Key)) {
        return [string]$config[$Key]
    }
    return ""
}

$Distro = Resolve-ConfigDefault "Distro" "WSL_DISTRO" $Distro
$WslRepoPath = Resolve-ConfigDefault "WslRepoPath" "WSL_REPO_PATH" $WslRepoPath
$CondaEnv = Resolve-ConfigDefault "CondaEnv" "WSL_CONDA_ENV" $CondaEnv
$CudaDeviceId = Resolve-ConfigDefault "CudaDeviceId" "CUDA_DEVICE_ID" $CudaDeviceId
$WslDataRoot = Resolve-ConfigDefault "WslDataRoot" "WSL_DATA_ROOT" $WslDataRoot
$WslCheckpointRoot = Resolve-ConfigDefault "WslCheckpointRoot" "WSL_CHECKPOINT_ROOT" $WslCheckpointRoot
$WslHfCache = Resolve-ConfigDefault "WslHfCache" "WSL_HF_CACHE" $WslHfCache
$WslWandbDir = Resolve-ConfigDefault "WslWandbDir" "WSL_WANDB_DIR" $WslWandbDir
$StockfishPath = Resolve-ConfigDefault "StockfishPath" "WSL_STOCKFISH_PATH" $StockfishPath
$CondaExecutable = Resolve-ConfigDefault "CondaExecutable" "WSL_CONDA_EXECUTABLE" $CondaExecutable
$VenvPath = Resolve-ConfigDefault "VenvPath" "WSL_VENV_PATH" $VenvPath

if (-not $NoSync) {
    $rsyncCommand = @(
        "rsync -a",
        "--exclude '.git/'",
        "--exclude '.venv/'",
        "--exclude '.tmp/'",
        "--exclude 'build/'",
        "--exclude '*.egg-info/'",
        "--exclude 'chess_sft_data/'",
        "--exclude 'chess_sft_checkpoints/'",
        "--exclude 'sft/make_data/data/'",
        "--exclude 'sft/make_data/output/'",
        "--exclude 'sft/make_data/polyglot_opening_books/*.bin'",
        "--exclude 'ui/node_modules/'",
        "--exclude 'ui/runs/'",
        "--exclude 'ui/dist/'",
        "--exclude '**/.pytest_cache/'",
        "--exclude '**/__pycache__/'",
        "--exclude '.claude/'",
        "$(Quote-Bash ($repoRootWsl.TrimEnd('/') + '/'))",
        "$(Quote-Bash ($WslRepoPath.TrimEnd('/') + '/'))"
    ) -join " "
    $syncCommand = "mkdir -p $(Quote-Bash $WslRepoPath) && $rsyncCommand"
    Write-Host "Syncing repo to WSL: $WslRepoPath" -ForegroundColor Cyan
    Invoke-Wsl @("--", "bash", "-lc", $syncCommand)
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$runtimeDirs = @($WslDataRoot, $WslCheckpointRoot, $WslHfCache, $WslWandbDir)
$mkdirCommand = "mkdir -p " + (($runtimeDirs | ForEach-Object { Quote-Bash $_ }) -join " ")
Invoke-Wsl @("--", "bash", "-lc", $mkdirCommand)
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $CommandArgs -or $CommandArgs.Count -eq 0) {
    $CommandArgs = @("bash")
}

$command = ($CommandArgs | ForEach-Object { Quote-Bash $_ }) -join " "
$runtimeEnv = [ordered]@{
    PYTHONUNBUFFERED = "1"
    CUDA_VISIBLE_DEVICES = $CudaDeviceId
    CHESS_SFT_OUTPUT = $WslDataRoot
    CHESS_SFT_CHECKPOINTS = $WslCheckpointRoot
    HF_HOME = $WslHfCache
    WANDB_DIR = $WslWandbDir
    STOCKFISH_PATH = $StockfishPath
}

foreach ($runtimeKey in @(
    "CHESS_SFT_BASE_MODEL",
    "CHESS_SFT_ENABLE_FLASH_ATTENTION_4_AUTO",
    "CHESS_SFT_ENABLE_HF_FLASH_ATTN2_AUTO",
    "CHESS_SFT_DISABLE_HF_FLASH_ATTN2_AUTO",
    "CUDA_DEVICE_ORDER",
    "VLLM_USE_V2_MODEL_RUNNER",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
    "HF_TOKEN",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "HF_HUB_ENABLE_HF_TRANSFER",
    "WANDB_API_KEY",
    "WANDB_MODE",
    "WANDB_ENTITY",
    "WANDB_BASE_URL",
    "WANDB_DISABLED",
    "WANDB_GIT_COMMIT"
)) {
    $runtimeValue = Get-RuntimeValue $runtimeKey
    if (-not [string]::IsNullOrWhiteSpace($runtimeValue)) {
        $runtimeEnv[$runtimeKey] = $runtimeValue
    }
}

if ([string]::IsNullOrWhiteSpace($VenvPath)) {
    $runnerCommand = "$(Quote-Bash $CondaExecutable) run --no-capture-output -n $(Quote-Bash $CondaEnv) $command"
} else {
    $runnerCommand = ". $(Quote-Bash ($VenvPath.TrimEnd('/') + '/bin/activate')) && $command"
}

$runCommand = @(
    "cd $(Quote-Bash ($WslRepoPath.TrimEnd('/')))",
    $runnerCommand
) -join " && "

$distroLabel = if ([string]::IsNullOrWhiteSpace($Distro)) { "default distro" } else { $Distro }
if ([string]::IsNullOrWhiteSpace($VenvPath)) {
    Write-Host "Running in WSL $distroLabel with conda env $CondaEnv and CUDA_VISIBLE_DEVICES=$CudaDeviceId" -ForegroundColor Cyan
} else {
    Write-Host "Running in WSL $distroLabel with venv $VenvPath and CUDA_VISIBLE_DEVICES=$CudaDeviceId" -ForegroundColor Cyan
}
Invoke-WslWithRuntimeEnv -Arguments @("--", "bash", "-lc", $runCommand) -Environment $runtimeEnv
exit $LASTEXITCODE
