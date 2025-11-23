FROM nvidia/cuda:11.4.3-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "pose_trainer.py"]
