FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# 🔥 CLAVE: fijar numpy < 2
RUN pip install --no-cache-dir \
    "numpy<2" \
    gym==0.26.2 \
    gym[accept-rom-license] \
    gym-super-mario-bros==7.4.0 \
    nes-py==8.2.1 \
    opencv-python \
    matplotlib \
    pyglet

WORKDIR /Super-mario-bros-PPO-pytorch

CMD ["bash"]