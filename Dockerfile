FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y python3 python3-pip git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Use a pip version that works with older wheels
RUN pip3 install "pip<24"

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /workspace/checkpoints

CMD ["python3", "pose_trainer.py"]
