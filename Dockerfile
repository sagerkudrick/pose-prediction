# Use CUDA 11.8 (modern, compatible with PyTorch 2.x)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Install system packages
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8 and sm_86 support
RUN pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Pre-download resnet18 weight file
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget https://download.pytorch.org/models/resnet18-f37072fd.pth \
    -O /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "pose_trainer.py"]
