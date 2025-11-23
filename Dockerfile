FROM nvidia/cuda:11.3.1-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install "pip<24"

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /workspace/checkpoints

CMD ["python3", "pose_trainer.py"]
