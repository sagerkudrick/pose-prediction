# Use NVIDIA CUDA base image with cuDNN for GPU support
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# Set working directory inside container
WORKDIR /workspace

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your code + dataset folder
COPY . .

# Make a folder for checkpoints (persistent volume can be mounted here)
RUN mkdir -p /workspace/checkpoints

# Default command: run training
CMD ["python3", "trainer_rewrite.py"]