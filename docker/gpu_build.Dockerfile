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
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN mkdir /app

WORKDIR /app
ADD pyproject.toml .
ADD . /app

# Upgrade pip and install packaging/wheel
RUN python3 -m pip install --upgrade pip
RUN pip install packaging wheel

# Install torch 2.7.0 with CUDA 12.6 support
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip \
    pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip \
    pip3 install -e .[dev,trag]

# Install flash-attn
RUN pip3 uninstall flash-attn -y
RUN pip install flash-attn==2.5.6 --no-build-isolation

# Create cache directories
RUN mkdir -p .cache/torch && export TORCH_HOME=/app/.cache/torch