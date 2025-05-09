FROM ubuntu:22.04 AS colette_cpu_build

LABEL description="LLM application API"
LABEL maintainer="contact@jolibrain.com"

# add missing apt dependencies
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install --no-install-recommends -y \
    python3-pip \
    python3-opencv \
    python3-pytest \
    sudo \
    wget \
    unzip \
    git \
    libreoffice \
    libmagic1 \
    poppler-utils \
    g++ && \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-luatex \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app
ADD requirements.txt .
RUN python3 -m pip install --upgrade pip && pip3 install -r requirements.txt --upgrade && pip3 install llama-cpp-python==0.3.1 uvicorn[standard] fastapi && pip3 cache purge
RUN mkdir .cache && mkdir .cache/torch && export TORCH_HOME=/app/.cache/torch
RUN huggingface-cli login --token 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

ADD . /app