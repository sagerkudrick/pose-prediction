FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# pip 20 is fine as long as we install torch by URL
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 🔥 Install torch 1.12.1 + cu113 via DIRECT WHEEL URL
RUN pip3 install https://download.pytorch.org/whl/cu113/torch-1.12.1%2Bcu113-cp38-cp38-linux_x86_64.whl && \
    pip3 install https://download.pytorch.org/whl/cu113/torchvision-0.13.1%2Bcu113-cp38-cp38-linux_x86_64.whl && \
    pip3 install https://download.pytorch.org/whl/cu113/torchaudio-0.12.1%2Bcu113-cp38-cp38-linux_x86_64.whl

COPY . .

RUN mkdir -p /workspace/checkpoints

CMD ["python3", "pose_trainer.py"]
