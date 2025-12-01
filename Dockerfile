# syntax=docker/dockerfile:1.4
# CUDA 11.8 (supports A5000 sm_86)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# -----------------------
# Install system packages with cache
# -----------------------
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------
# Upgrade pip
# -----------------------
RUN python3 -m pip install --upgrade pip

# -----------------------
# Install PyTorch with BuildKit pip cache
# -----------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# -----------------------
# Pre-download ResNet18 weights (cached by BuildKit)
# -----------------------
RUN --mount=type=cache,target=/root/.cache/torch \
    mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget https://download.pytorch.org/models/resnet18-f37072fd.pth \
    -O /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth

# -----------------------
# Install remaining Python dependencies
# -----------------------
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# -----------------------
# Copy source code
# -----------------------
COPY . .

# -----------------------
# Run
# -----------------------
CMD ["python3", "pose_trainer.py"]
