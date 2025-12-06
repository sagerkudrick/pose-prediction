# syntax=docker/dockerfile:1.4
# CUDA 11.8 (supports A5000 sm_86)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# -----------------------
# Create persistent virtual environment directory (mounted from host)
# -----------------------
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Ensure folder exists even before mount
RUN mkdir -p /app/.venv

# -----------------------
# Install system packages with cache
# -----------------------
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------
# Create venv & upgrade pip
# -----------------------
RUN python3 -m venv /app/.venv && \
    /app/.venv/bin/pip install --upgrade pip setuptools wheel

# -----------------------
# Install PyTorch inside the venv
# -----------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    /app/.venv/bin/pip install \
        torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 \
        --extra-index-url https://download.pytorch.org/whl/cu118

# -----------------------
# Pre-download ResNet18 weights (cached)
# -----------------------
RUN --mount=type=cache,target=/root/.cache/torch \
    mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget https://download.pytorch.org/models/resnet18-f37072fd.pth \
    -O /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth

# -----------------------
# Install project dependencies inside the venv
# -----------------------
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    /app/.venv/bin/pip install -r requirements.txt

# -----------------------
# Copy source code
# -----------------------
COPY . .

# -----------------------
# Default run command
# -----------------------
CMD ["/app/.venv/bin/python", "pose_trainer.py"]
