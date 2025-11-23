# Base image with CUDA 11.4 runtime
FROM nvidia/cuda:11.4.3-runtime-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install Python, pip, git, and build essentials
RUN apt-get update && \
    apt-get install -y python3 python3-pip git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip to a safe version
RUN pip3 install --upgrade "pip<24"

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your project files
COPY . .

# Default command to run your trainer
CMD ["python3", "pose_trainer.py"]
