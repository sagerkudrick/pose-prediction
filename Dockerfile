# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Tell Python to use the mounted venv
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Just copy your code
COPY . .

# Use the host venv’s python
CMD ["/app/.venv/bin/python", "pose_trainer.py"]
