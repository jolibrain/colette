FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS colette_gpu_build

LABEL description="LLM application API"
LABEL maintainer="contact@jolibrain.com"

# Add deadsnakes PPA and install Python 3.12
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install --no-install-recommends -y \
    software-properties-common \
    curl && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update -y && \
    apt-get install --no-install-recommends -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
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
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libcairo2-dev \
    pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN mkdir /app

WORKDIR /app
ADD . /app

ENV MAX_JOBS=4
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip \
    COLETTE_CUDA_SHORT=126 bash scripts/install_python_deps.sh

# Create cache directories
RUN mkdir -p .cache/torch && export TORCH_HOME=/app/.cache/torch