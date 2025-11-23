FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

WORKDIR /workspace

# Install Python, pip, git
RUN apt-get update && apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Copy your requirements
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Default command
CMD ["python3", "pose_trainer.py"]
