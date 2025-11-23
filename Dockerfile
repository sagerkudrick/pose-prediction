# Base image with CUDA 11.4 (use a valid tag)
FROM nvidia/cuda:11.4.2-runtime-ubuntu22.04

WORKDIR /workspace

# Install Python, pip, git, and build essentials
RUN apt-get update && \
    apt-get install -y python3 python3-pip git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade "pip<24"

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command
CMD ["python3", "pose_trainer.py"]
