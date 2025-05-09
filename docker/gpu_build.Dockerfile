FROM nvidia/cuda:12.5.1-devel-ubuntu22.04 AS colette_gpu_build

LABEL description="LLM application API"
LABEL maintainer="contact@jolibrain.com"

# add missing apt dependencies
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install --no-install-recommends -y \
    python3-pip \
    python3-opencv \
    python3-pytest \
    python3-dev \
    sudo \
    wget \
    unzip \
    git \
    libreoffice \
    libmagic1 \
    poppler-utils \
    g++ \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-luatex \
    ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app
ADD pyproject.toml .
ADD . /app
RUN python3 -m pip install --upgrade pip
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install torch==2.6.0
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install flash-attn --no-build-isolation
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip3 install -e .[dev,trag]
RUN mkdir .cache && mkdir .cache/torch && export TORCH_HOME=/app/.cache/torch
