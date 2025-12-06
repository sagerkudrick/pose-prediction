# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# -----------------------
# Install system packages
# -----------------------
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y \
        python3 python3-pip python3.12-venv python3-dev \
        git wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------
# Create virtual environment
# -----------------------
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3 -m venv $VIRTUAL_ENV && \
    $VIRTUAL_ENV/bin/pip install --upgrade pip setuptools wheel

# -----------------------
# Copy only requirements first (Docker caching)
# -----------------------
COPY requirements.txt .

# -----------------------
# Install dependencies
# -----------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# -----------------------
# Copy source code only (no dataset/models)
# -----------------------
COPY . .

# -----------------------
# Default run command
# -----------------------
CMD ["python", "pose_trainer.py"]
