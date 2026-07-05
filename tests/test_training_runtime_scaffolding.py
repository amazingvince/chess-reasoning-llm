from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "sft" / "training"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _powershell_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _run_launcher_with_fake_wsl(tmp_path: Path, exit_code: int) -> subprocess.CompletedProcess[str]:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for executable run-wsl.ps1 tests")

    fake_wsl = tmp_path / "fake-wsl.cmd"
    fake_wsl.write_text(
        "@echo off\r\n"
        "echo fake-wsl-stderr 1>&2\r\n"
        "exit /b %FAKE_WSL_EXIT%\r\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["CHESS_SFT_WSL_EXE"] = str(fake_wsl)
    env["FAKE_WSL_EXIT"] = str(exit_code)

    command = (
        "$global:PSNativeCommandUseErrorActionPreference = $true; "
        f"& {_powershell_quote(TRAINING_DIR / 'run-wsl.ps1')} "
        "-NoSync -WslRepoPath /tmp/chess_sft_sdpo -VenvPath /tmp/chess_sft_sdpo/.venv true; "
        "exit $LASTEXITCODE"
    )
    return subprocess.run(
        [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_training_dockerfile_installs_package_cli_from_repo_root() -> None:
    text = _read(TRAINING_DIR / "Dockerfile")

    assert 'python -m pip install -e ".[data,train,eval]"' in text
    assert "COPY pyproject.toml README.md /workspace/" in text
    assert "COPY src /workspace/src" in text
    assert "WORKDIR /workspace/sft/training" not in text
    assert "COPY sft/training/requirements.txt" not in text
    assert "pip install -r /tmp/requirements.txt" not in text


def test_eval_dockerfile_installs_eval_extra_from_repo_root() -> None:
    text = _read(TRAINING_DIR / "Dockerfile.eval")

    assert "ARG EVAL_BASE_IMAGE=vllm/vllm-openai:latest" in text
    assert "FROM ${EVAL_BASE_IMAGE}" in text
    assert 'python3 -m pip install -e ".[eval]"' in text
    assert "COPY pyproject.toml README.md /workspace/" in text
    assert "COPY src /workspace/src" in text
    assert "WORKDIR /workspace/sft/training" not in text
    assert "COPY sft/training/requirements.eval.txt" not in text
    assert "pip install -r /tmp/requirements.eval.txt" not in text


def test_container_launchers_use_repo_root_workdir() -> None:
    assert "working_dir: /workspace\n" in _read(TRAINING_DIR / "compose.yaml")
    assert '"-w", "/workspace",' in _read(TRAINING_DIR / "run-docker.ps1")
    assert '"-w", "/workspace",' in _read(TRAINING_DIR / "run-eval-docker.ps1")


def test_docker_launchers_put_optional_secrets_before_image_name() -> None:
    training_text = _read(TRAINING_DIR / "run-docker.ps1")
    eval_text = _read(TRAINING_DIR / "run-eval-docker.ps1")

    assert training_text.index('if ($config["HF_TOKEN"])') < training_text.index('"chess-sft-training:latest"')
    assert training_text.index('if ($config["WANDB_API_KEY"])') < training_text.index('"chess-sft-training:latest"')
    assert eval_text.index('if ($config["HF_TOKEN"])') < eval_text.rindex('$($config["IMAGE_NAME"])')
    assert eval_text.index('if ($config["WANDB_API_KEY"])') < eval_text.rindex('$($config["IMAGE_NAME"])')


def test_compose_builds_managed_eval_image() -> None:
    text = _read(TRAINING_DIR / "compose.yaml")

    assert "  evaluator:\n" in text
    assert "dockerfile: sft/training/Dockerfile.eval" in text
    assert "EVAL_BASE_IMAGE: ${EVAL_BASE_IMAGE:-vllm/vllm-openai:latest}" in text
    assert "image: chess-sft-eval-vllm:latest" in text
    assert "CUDA_VISIBLE_DEVICES: ${EVAL_CUDA_DEVICE_ID:-0}" in text


def test_eval_docker_launcher_can_use_compose_evaluator_service() -> None:
    text = _read(TRAINING_DIR / "run-eval-docker.ps1")

    assert "[switch]$BuildImage" in text
    assert "[switch]$UseCompose" in text
    assert "function Invoke-ComposeWithResolvedEnv" in text
    assert 'if (Test-Path $envPath) {' in text
    assert '$composeEnvArgs += @("--env-file", $envPath)' in text
    assert '[Environment]::SetEnvironmentVariable($key, [string]$config[$key], "Process")' in text
    assert '"build", "evaluator"' in text
    assert '"run", "--rm", "evaluator"' in text
    assert "$composeEnvArgs" in text
    assert 'docker compose --env-file .\\sft\\training\\.env -f .\\sft\\training\\compose.yaml run --rm evaluator' in _read(
        TRAINING_DIR / "README.md"
    )


def test_runtime_defaults_are_not_machine_local() -> None:
    files = [
        TRAINING_DIR / ".env.example",
        TRAINING_DIR / "compose.yaml",
        TRAINING_DIR / "run-docker.ps1",
        TRAINING_DIR / "run-eval-docker.ps1",
        TRAINING_DIR / "run-wsl.ps1",
        TRAINING_DIR / "posthoc_eval_then_phase.ps1",
        TRAINING_DIR / "README.md",
    ]
    forbidden_tokens = [
        "E:/",
        "/mnt/e",
        "/home/vince",
        "Ubuntu-20.04",
        "RTX 5090",
        "RTX 4090",
        "CUDA_DEVICE_ID=1",
        'CUDA_DEVICE_ID = "1"',
        'else { "1" }',
    ]

    for path in files:
        text = _read(path)
        for token in forbidden_tokens:
            assert token not in text, f"{path.name} still contains machine-local token {token!r}"

    env_text = _read(TRAINING_DIR / ".env.example")
    assert "CUDA_DEVICE_ID=0" in env_text
    assert "EVAL_CUDA_DEVICE_ID=0" in env_text
    assert "CHESS_SFT_OUTPUT_HOST=../../.runtime/chess_sft_data" in env_text
    assert "Full runs reject offline W&B unless --allow-wandb-offline is passed." in env_text


def test_powershell_env_file_overrides_defaults_and_relative_paths_are_repo_local() -> None:
    for script_name in ["run-docker.ps1", "run-eval-docker.ps1", "posthoc_eval_then_phase.ps1"]:
        text = _read(TRAINING_DIR / script_name)

        assert "$defaultRuntimeRoot = Join-Path $repoRoot \".runtime\"" in text
        if script_name == "posthoc_eval_then_phase.ps1":
            continue
        assert "function Resolve-HostPath" in text
        assert "$config[$key] = $value" in text
        assert "IsNullOrWhiteSpace($config[$key])" not in text
        assert '$config[$pathKey] = Resolve-HostPath $config[$pathKey]' in text


def test_wsl_launcher_runs_from_repo_root() -> None:
    text = _read(TRAINING_DIR / "run-wsl.ps1")

    assert "cd $(Quote-Bash ($WslRepoPath.TrimEnd('/')))" in text
    assert "'/sft/training'" not in text
    assert '[string]$Distro = ""' in text
    assert "function Invoke-Wsl" in text
    assert 'if (-not [string]::IsNullOrWhiteSpace($Distro))' in text
    assert "& wsl.exe -d $Distro" not in text
    assert '[string]$WslRepoPath = "/tmp/chess_sft_sdpo"' in text
    assert '[string]$CudaDeviceId = "0"' in text
    assert '[string]$VenvPath = ""' in text
    assert "$ScriptBoundParameters = @{} + $PSBoundParameters" in text
    assert "$ScriptBoundParameters.ContainsKey($ParameterName)" in text
    assert ". $(Quote-Bash ($VenvPath.TrimEnd('/') + '/bin/activate'))" in text
    assert 'PYTHONUNBUFFERED = "1"' in text
    assert "run --no-capture-output -n" in text
    assert "--exclude '.venv/'" in text
    assert "--exclude '.tmp/'" in text
    assert "--exclude 'chess_sft_data/'" in text
    assert "--exclude 'chess_sft_checkpoints/'" in text
    assert "--exclude 'data/syzygy/'" in text
    assert "--exclude 'polyglot_opening_books/*.bin'" in text
    assert "--exclude 'ui/node_modules/'" in text
    assert "--exclude 'ui/runs/'" in text
    assert "--exclude 'ui/dist/'" in text


def test_wsl_launcher_does_not_treat_native_stderr_as_fatal() -> None:
    text = _read(TRAINING_DIR / "run-wsl.ps1")

    assert 'if ($env:CHESS_SFT_WSL_EXE)' in text
    assert '$wslExecutable = "wsl.exe"' in text
    assert "$previousErrorActionPreference = $ErrorActionPreference" in text
    assert '$ErrorActionPreference = "Continue"' in text
    assert "Get-Variable -Name PSNativeCommandUseErrorActionPreference" in text
    assert "$global:PSNativeCommandUseErrorActionPreference = $false" in text
    assert "$ErrorActionPreference = $previousErrorActionPreference" in text
    assert "$global:PSNativeCommandUseErrorActionPreference = $previousNativeCommandPreference" in text


def test_wsl_launcher_preserves_fake_wsl_exit_codes_with_stderr(tmp_path: Path) -> None:
    success = _run_launcher_with_fake_wsl(tmp_path, exit_code=0)

    assert success.returncode == 0, success.stderr
    assert "fake-wsl-stderr" in success.stderr
    assert "Running in WSL" in success.stdout

    failure = _run_launcher_with_fake_wsl(tmp_path, exit_code=13)

    assert failure.returncode == 13
    assert "fake-wsl-stderr" in failure.stderr


def test_wsl_launcher_forwards_wandb_runtime_settings() -> None:
    text = _read(TRAINING_DIR / "run-wsl.ps1")

    for env_name in [
        "WANDB_API_KEY",
        "WANDB_MODE",
        "WANDB_ENTITY",
        "WANDB_BASE_URL",
        "WANDB_DISABLED",
        "WANDB_GIT_COMMIT",
    ]:
        assert f"$env:{env_name}" in text
        assert f'"{env_name}"' in text
    assert "Invoke-WslWithRuntimeEnv" in text
    assert "[Environment]::SetEnvironmentVariable($key, [string]$Environment[$key], \"Process\")" in text
    assert 'WANDB_API_KEY=$(Quote-Bash $runtimeValue)' not in text


def test_wsl_launcher_forwards_hf_runtime_settings() -> None:
    text = _read(TRAINING_DIR / "run-wsl.ps1")

    for env_name in [
        "HF_TOKEN",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_ENABLE_HF_TRANSFER",
    ]:
        assert f"$env:{env_name}" in text
        assert f'"{env_name}"' in text
    assert "Invoke-WslWithRuntimeEnv" in text
    assert "[Environment]::SetEnvironmentVariable($key, [string]$Environment[$key], \"Process\")" in text
    assert 'HF_TOKEN=$(Quote-Bash $runtimeValue)' not in text


def test_wsl_launcher_forwards_attention_runtime_settings() -> None:
    text = _read(TRAINING_DIR / "run-wsl.ps1")

    for env_name in [
        "CHESS_SFT_ENABLE_FLASH_ATTENTION_4_AUTO",
        "CHESS_SFT_ENABLE_HF_FLASH_ATTN2_AUTO",
        "CHESS_SFT_DISABLE_HF_FLASH_ATTN2_AUTO",
    ]:
        assert f"$env:{env_name}" in text
        assert f'"{env_name}"' in text
    assert "Invoke-WslWithRuntimeEnv" in text


def test_wsl_launcher_reads_training_env_file_for_runtime_config() -> None:
    text = _read(TRAINING_DIR / "run-wsl.ps1")

    assert '$envPath = Join-Path $scriptDir ".env"' in text
    assert 'if (Test-Path $envPath) {' in text
    assert "function Get-RuntimeValue" in text
    assert "function Resolve-ConfigDefault" in text
    for env_name in ["HF_TOKEN", "WANDB_API_KEY", "WANDB_MODE", "CHESS_SFT_BASE_MODEL"]:
        assert f'"{env_name}"' in text


def test_wsl_launcher_docs_use_separator_for_single_dash_command_args() -> None:
    readme_text = _read(TRAINING_DIR / "README.md")

    assert "Use `--` before commands that take single-dash options" in readme_text
    assert ".\\sft\\training\\run-wsl.ps1 -NoSync -- pytest tests -q" in readme_text
    assert ".\\sft\\training\\run-wsl.ps1 -- python -c" in readme_text


def test_docker_launchers_forward_wandb_runtime_settings() -> None:
    compose_text = _read(TRAINING_DIR / "compose.yaml")
    env_text = _read(TRAINING_DIR / ".env.example")

    for env_name in [
        "WANDB_API_KEY",
        "WANDB_MODE",
        "WANDB_ENTITY",
        "WANDB_BASE_URL",
    ]:
        assert f"{env_name}:" in compose_text
        assert f"{env_name}:" in compose_text
        assert f"{env_name}=" in env_text

    for script_name in ["run-docker.ps1", "run-eval-docker.ps1"]:
        text = _read(TRAINING_DIR / script_name)
        for env_name in [
            "WANDB_API_KEY",
            "WANDB_MODE",
            "WANDB_ENTITY",
            "WANDB_BASE_URL",
        ]:
            assert f"{env_name} = \"\"" in text
            assert f'if ($config["{env_name}"])' in text
            assert f'{env_name}=$($config["{env_name}"])' in text


def test_qwen_fast_path_versions_are_consistent_across_runtime_files() -> None:
    env_text = _read(TRAINING_DIR / ".env.example")
    compose_text = _read(TRAINING_DIR / "compose.yaml")
    dockerfile_text = _read(TRAINING_DIR / "Dockerfile")
    readme_text = _read(TRAINING_DIR / "README.md")

    assert "FLASH_LINEAR_ATTENTION_VERSION=0.5.1" in env_text
    assert "CAUSAL_CONV1D_VERSION=1.6.2.post1" in env_text
    assert "FLASH_LINEAR_ATTENTION_VERSION:-0.5.1" in compose_text
    assert "CAUSAL_CONV1D_VERSION:-1.6.2.post1" in compose_text
    assert "ARG FLASH_LINEAR_ATTENTION_VERSION=0.5.1" in dockerfile_text
    assert "ARG CAUSAL_CONV1D_VERSION=1.6.2.post1" in dockerfile_text
    assert "flash-linear-attention[cuda]==0.5.1" in readme_text
    assert "causal-conv1d==1.6.2.post1" in readme_text


def test_posthoc_phase_launcher_uses_package_cli() -> None:
    text = _read(TRAINING_DIR / "posthoc_eval_then_phase.ps1")

    assert '"chess-llm-train",' in text
    assert '"python", "train.py"' not in text


def test_posthoc_phase_launcher_supports_wandb_group() -> None:
    text = _read(TRAINING_DIR / "posthoc_eval_then_phase.ps1")

    assert '[string]$WandbGroup = ""' in text
    assert '$evalArgs += @("--wandb-group", $WandbGroup)' in text
    assert '$nextArgs += @("--wandb-group", $WandbGroup)' in text


def test_training_docs_are_package_cli_first() -> None:
    text = _read(TRAINING_DIR / "README.md")

    for retired_command in [
        "python train.py",
        "python evaluate.py",
        "python run_curriculum.py",
        ".\\run-docker.ps1 python",
        ".\\sft\\training\\run-wsl.ps1 python",
    ]:
        assert retired_command not in text

    assert "docs/runbooks/phase_a_real_run.md" in text
    assert "--num-train-epochs 1" in text
    assert "skip-trainer-eval" in text
    assert 'python -m pip install -e ".[data,train,eval]"' in text
    assert "chess-llm-evaluate" in text
    assert "chess-llm-run-curriculum" in text
    assert "docker compose --env-file .\\sft\\training\\.env -f .\\sft\\training\\compose.yaml build" in text


def test_training_docs_keep_wandb_enabled_for_real_runs() -> None:
    text = _read(TRAINING_DIR / "README.md")

    forbidden_real_run_examples = [
        "Full Phase A training\nchess-llm-train --phase a --no-wandb",
        "Full Phase B training (uses Phase A best/ checkpoint if present)\nchess-llm-train --phase b --no-wandb",
        "Full Phase C training\nchess-llm-train --phase c --no-wandb",
        "Run Full Training\n\n```bash\n.\\sft\\training\\run-docker.ps1 chess-llm-train --phase a --no-wandb",
        "Run commands from the Windows repo root:\n\n```powershell\n.\\sft\\training\\run-wsl.ps1 chess-llm-train --phase a --dry-run --no-wandb",
        "Common First-Run Flow\n\nFor a new machine or fresh environment:\n\n```bash\nchess-llm-train --phase a --dry-run\nchess-llm-train --phase a --smoke-run --no-wandb\nchess-llm-train --phase a --no-wandb",
    ]

    for example in forbidden_real_run_examples:
        assert example not in text


def test_training_docs_explain_explicit_wandb_offline_escape_hatch() -> None:
    text = _read(TRAINING_DIR / "README.md")

    assert "unset `WANDB_MODE=offline`" in text
    assert "--allow-wandb-offline" in text
    assert "Smoke runs allow offline W&B automatically" in text


def test_training_docs_recommend_persistent_wsl_hf_cache() -> None:
    text = _read(TRAINING_DIR / "README.md")

    assert "-WslHfCache /path/to/persistent/chess_sft_hf_cache" in text
    assert "avoid repeated Qwen/Qwen3.5-0.8B downloads" in text


def test_training_docs_explain_wsl_wandb_and_persistent_artifact_paths() -> None:
    text = _read(TRAINING_DIR / "README.md")

    assert "The WSL launcher reads `sft/training/.env`" in text
    assert "`HF_TOKEN` and `WANDB_API_KEY`" in text
    assert "PowerShell environment variables override `.env` values" in text
    assert "run `wandb login` inside the same WSL distro and user" in text
    assert "`WANDB_DISABLED` is forwarded so accidental host-side disables fail fast" in text
    assert "-WslDataRoot /path/to/persistent/chess_sft_data" in text
    assert "-WslCheckpointRoot /path/to/persistent/chess_sft_checkpoints" in text
    assert "-WslWandbDir /path/to/persistent/chess_sft_wandb" in text
    assert "`-WslDataRoot` is the `CHESS_SFT_OUTPUT` root" in text
    assert "do not point it at the `output/` directory itself" in text


def test_training_test_runner_stays_at_repo_root() -> None:
    assert not (TRAINING_DIR / "pyproject.toml").exists()
    assert not (TRAINING_DIR / "tests").exists()
    assert "sft/training/tests" not in _read(REPO_ROOT / "README.md")
    assert "sft/training/tests" not in _read(TRAINING_DIR / "README.md")
