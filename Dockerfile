# ================================================================
# Dockerfile — Backend Mario RL
# Base: PyTorch 2.3 + CUDA 12.1 (compatible RTX 4060 sm_89)
# ================================================================

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

LABEL maintainer="mario-rl-platform"

# Sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias Python (sin torch: ya viene en la imagen base)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código
COPY . .

# Directorio de datos persistente
ENV DATA_DIR=/data
RUN mkdir -p /data/models /data/logs /data/checkpoints

EXPOSE 5000

CMD ["python", "-u", "app.py"]
