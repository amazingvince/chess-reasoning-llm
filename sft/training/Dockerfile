ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG INSTALL_VLLM=0
ARG INSTALL_FLASH_ATTN=0
ARG INSTALL_QWEN_FAST_PATH=1
ARG FLASH_LINEAR_ATTENTION_VERSION=0.5.1
ARG CAUSAL_CONV1D_VERSION=1.6.2.post1
ARG FLASH_ATTN_WHEEL_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
ARG STOCKFISH_SOURCE_URL=https://github.com/official-stockfish/Stockfish/archive/refs/tags/sf_18.tar.gz
ARG STOCKFISH_ARCH=x86-64-bmi2

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/cache/huggingface \
    CHESS_SFT_OUTPUT=/data/chess_sft_data \
    CHESS_SFT_CHECKPOINTS=/data/chess_sft_checkpoints \
    STOCKFISH_PATH=/usr/games/stockfish

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgomp1 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    mkdir -p /tmp/stockfish-src; \
    curl -L "${STOCKFISH_SOURCE_URL}" -o /tmp/stockfish.tar.gz; \
    tar -xzf /tmp/stockfish.tar.gz -C /tmp/stockfish-src --strip-components=1; \
    make -C /tmp/stockfish-src/src -j"$(nproc)" build ARCH="${STOCKFISH_ARCH}"; \
    install -m 0755 /tmp/stockfish-src/src/stockfish /usr/games/stockfish; \
    printf "quit\n" | /usr/games/stockfish | head -n 1; \
    rm -rf /tmp/stockfish.tar.gz /tmp/stockfish-src

WORKDIR /workspace

RUN python -m pip install --upgrade pip setuptools wheel

COPY pyproject.toml README.md /workspace/
COPY src /workspace/src

RUN python -m pip install -e ".[data,train,eval]" && \
    if [ "${INSTALL_QWEN_FAST_PATH}" = "1" ]; then \
        python -m pip install wheel packaging ninja && \
        python -m pip install --no-build-isolation "causal-conv1d==${CAUSAL_CONV1D_VERSION}" && \
        python -m pip install "flash-linear-attention[cuda]==${FLASH_LINEAR_ATTENTION_VERSION}"; \
    fi && \
    if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then python -m pip install "${FLASH_ATTN_WHEEL_URL}"; fi && \
    if [ "${INSTALL_VLLM}" = "1" ]; then python -m pip install vllm; fi

COPY . /workspace

CMD ["bash"]
