FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Pin pip below 24 because old torch wheels break on pip 24+
RUN pip3 install "pip<24"

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /workspace/checkpoints

CMD ["python3", "pose_trainer.py"]
