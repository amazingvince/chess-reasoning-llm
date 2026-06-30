#!/usr/bin/env bash
set -Eeuo pipefail

die() {
  echo "error: $*" >&2
  exit 1
}

script_name="$(basename "$0")"

usage() {
  cat <<USAGE
Usage:
  VENV=/path/to/.venv CUDA_HOME=/usr/local/cuda-12.8 ./${script_name}

Builds causal-conv1d from source for one portable WSL wheel that covers the
local Ada/Blackwell pair plus Hopper by default:

  TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"

Environment:
  VENV                    Python virtualenv to modify. Defaults to .venv
                          candidates if one exists.
  CUDA_HOME               CUDA toolkit root. Defaults to /usr/local/cuda.
  TORCH_CUDA_ARCH_LIST    CUDA architectures to compile. Defaults to
                          "8.9;9.0;12.0".
  CAUSAL_CONV1D_VERSION   Defaults to 1.6.2.post1.
  MAX_JOBS                Parallel build jobs. Defaults to 8.
  VERIFY_CUDA             Run causal-conv1d smoke tests after install. Defaults
                          to 1.

Blackwell sm_120 requires a CUDA toolkit whose nvcc supports compute_120.
CUDA 12.3 can build Ada/Hopper, but not Blackwell.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${VENV:-}" ]]; then
  for candidate in "$PWD/.venv" "/home/vince/code/chess_sft_sdpo_codex/.venv"; do
    if [[ -x "$candidate/bin/python" ]]; then
      VENV="$candidate"
      break
    fi
  done
fi

[[ -n "${VENV:-}" ]] || die "set VENV=/path/to/.venv"
[[ -x "$VENV/bin/python" ]] || die "VENV does not contain bin/python: $VENV"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="$CUDA_HOME/bin/nvcc"
[[ -x "$NVCC" ]] || die "nvcc not found at $NVCC; set CUDA_HOME to a CUDA toolkit install"

ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;9.0;12.0}"
CAUSAL_CONV1D_VERSION="${CAUSAL_CONV1D_VERSION:-1.6.2.post1}"
MAX_JOBS="${MAX_JOBS:-8}"
VERIFY_CUDA="${VERIFY_CUDA:-1}"

release="$("$NVCC" --version | sed -nE 's/.*release ([0-9]+)\.([0-9]+).*/\1.\2/p' | head -n 1)"
[[ -n "$release" ]] || die "could not parse nvcc version from $NVCC --version"

major="${release%%.*}"
minor="${release#*.}"

needs_blackwell=0
if [[ "$ARCH_LIST" == *"12.0"* || "$ARCH_LIST" == *"12.0+PTX"* || "$ARCH_LIST" == *"sm_120"* ]]; then
  needs_blackwell=1
fi

if [[ "$needs_blackwell" == "1" ]]; then
  if (( major < 12 || (major == 12 && minor < 8) )); then
    die "Blackwell sm_120 build requested via TORCH_CUDA_ARCH_LIST=$ARCH_LIST, but nvcc is CUDA $release. Install CUDA 12.8+ and rerun with CUDA_HOME=/usr/local/cuda-12.8."
  fi
  if ! "$NVCC" --list-gpu-arch | grep -qx "compute_120"; then
    die "nvcc at $NVCC does not list compute_120; install/select a newer CUDA toolkit"
  fi
fi

echo "Using VENV=$VENV"
echo "Using CUDA_HOME=$CUDA_HOME"
echo "Using nvcc CUDA release $release"
echo "Using TORCH_CUDA_ARCH_LIST=$ARCH_LIST"

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
export MAX_JOBS

"$VENV/bin/python" -m pip install --upgrade wheel packaging ninja
"$VENV/bin/python" -m pip uninstall -y causal-conv1d
"$VENV/bin/python" -m pip install \
  --no-build-isolation \
  --no-binary=causal-conv1d \
  --no-cache-dir \
  "causal-conv1d==$CAUSAL_CONV1D_VERSION"

if [[ "$VERIFY_CUDA" == "1" ]]; then
  "$VENV/bin/python" - <<'PY'
import torch
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")

for device_idx in range(torch.cuda.device_count()):
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)
    dtype = torch.float16
    batch, channels, seqlen, width = 1, 4, 8, 3

    x = torch.randn(batch, channels, seqlen, device=device, dtype=dtype)
    weight = torch.randn(channels, width, device=device, dtype=dtype)
    bias = torch.randn(channels, device=device, dtype=dtype)
    causal_conv1d_fn(x, weight, bias, activation="silu")

    x_step = torch.randn(batch, channels, device=device, dtype=dtype)
    conv_state = torch.zeros(batch, channels, width, device=device, dtype=dtype)
    causal_conv1d_update(x_step, conv_state, weight, bias, activation="silu")

    torch.cuda.synchronize(device)
    capability = ".".join(str(part) for part in torch.cuda.get_device_capability(device_idx))
    print(f"causal-conv1d smoke ok on cuda:{device_idx} {torch.cuda.get_device_name(device_idx)} sm_{capability.replace('.', '')}")
PY
fi
