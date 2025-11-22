# Base image with CUDA 12.2 runtime
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install Python 3, pip, git
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy code + dataset
COPY . .

# Make folder for checkpoints
RUN mkdir -p /workspace/checkpoints

# Default command: run training
CMD ["python3", "trainer_rewrite.py"]
