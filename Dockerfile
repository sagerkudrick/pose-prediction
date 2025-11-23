# Use NVIDIA CUDA 11.4 runtime on Ubuntu 20.04
FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

# Set working directory
WORKDIR /workspace

# Prevent prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y wget && \
    mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget https://download.pytorch.org/models/resnet18-f37072fd.pth -O /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
# Use regular PyTorch wheel (CPU/GPU compatible) to avoid +cu114 issues
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Default command to run your training script
CMD ["python3", "pose_trainer.py"]
