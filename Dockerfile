# Use lightweight Python 3.8 base image
FROM python:3.8-slim

# Set working directory
WORKDIR /workspace

# Avoid prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy your requirements file
COPY requirements.txt .

# Install PyTorch with CUDA support and other dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    -f https://download.pytorch.org/whl/cu116/torch_stable.html

# Copy the rest of your project
COPY . .

# Default command to run your training script
CMD ["python", "trainer_rewrite.py"]
