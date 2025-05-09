# Deployment

Colette is most easily deployed using Docker.

### Pre-built Docker images

Docker images are provided on https://docker.jolibrain.com/

To pull an image:

```bash
docker pull docker.jolibrain.com/colette_gpu_build
```

### Building with Docker for GPU

```bash
docker build --no-cache -t colette_gpu_build -f docker/gpu_build.Dockerfile .
docker build --no-cache -t colette_gpu_server -f docker/gpu_server.Dockerfile .
docker run --rm --runtime=nvidia -v /path/to/data:/data -p 1873:1873 colette_gpu_server:latest
```

### Building with Docker for CPU
It is recommended to run on GPU, though for CPU-only platforms, proceed as below:

```bash
docker build --no-cache -t colette_cpu_build -f docker/cpu_build.Dockerfile .
docker build --no-cache -t colette_cpu_server -f docker/cpu_server.Dockerfile .
docker run -v /path/to/data:/data -p 1873:1873 colette_cpu_server:latest
```

### Building the Web User Interface with Docker

```bash
docker build --no-cache -t colette_ui -f docker/ui.Dockerfile .
```
