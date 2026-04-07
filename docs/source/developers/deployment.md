# Deployment

Colette is most easily deployed using Docker.

For image rebuild workflows after source code updates (local compose, registry tagging, GPU/CPU variants), see:

- `container_rebuild.md`

## CI Image Publish Pipeline

Repository contains a dedicated Jenkins pipeline for building and publishing container images:

- `Jenkinsfile.images`

This pipeline builds and tags:

- `docker.jolibrain.com/colette_gpu:<short_sha>` from `docker/gpu_build.Dockerfile`
- `docker.jolibrain.com/colette_gpu_server:<short_sha>` from `docker/gpu_jb_server.Dockerfile`
- `docker.jolibrain.com/colette_ui:<short_sha>` from `docker/ui.Dockerfile`

Push policy:

- Push runs only on `main` and `release/*` branches
- Tags use short Git SHA
- Existing root `Jenkinsfile` remains focused on tests

Jenkins Multibranch setup:

1. Create a dedicated Multibranch Pipeline job for image publication.
2. Set `Script Path` to `Jenkinsfile.images`.
3. Keep your test Multibranch job pointing to `Jenkinsfile`.
4. Optional parameters at runtime:
`NO_CACHE=true` to force clean Docker builds.
`PUSH_ENABLED=false` to build without pushing.
`RUN_INTEGRATION=true` to execute containerized integration tests before push.
`SMOKE_MODE=quick|api` to select smoke depth before push.
`SMOKE_TIMEOUT_SEC=120` to tune API smoke timeout.
`INTEGRATION_PYTEST_ARGS=-q` to pass custom pytest flags for integration runs.
`INTEGRATION_TEST_FILES="/app/tests/test_pipeline_python_api_integration.py /app/tests/test_pipeline_tantivy_integration.py"` to choose integration suites.

Container smoke tests are script-based and reusable outside Jenkins:

- `ci/container_preflight.sh`
- `ci/build_container_images.sh`
- `ci/container_smoke.sh`
- `ci/container_integration.sh`

Run a preflight before long jobs:

```bash
export REGISTRY=docker.jolibrain.com
export IMAGE_TAG=$(git rev-parse --short=12 HEAD)
bash ci/container_preflight.sh build
```

Container image builds are also script-based and reused by `Jenkinsfile.images`:

```bash
export REGISTRY=docker.jolibrain.com
export IMAGE_TAG=$(git rev-parse --short=12 HEAD)
export NO_CACHE=false
bash ci/build_container_images.sh
```

You can build one target only:

```bash
export REGISTRY=docker.jolibrain.com
export IMAGE_TAG=$(git rev-parse --short=12 HEAD)
export TARGET=gpu_server
bash ci/build_container_images.sh
```

Local example (quick mode):

```bash
export IMAGE_TAG=$(git rev-parse --short=12 HEAD)
export GPU_BASE_IMAGE=docker.jolibrain.com/colette_gpu:${IMAGE_TAG}
export GPU_SERVER_IMAGE=docker.jolibrain.com/colette_gpu_server:${IMAGE_TAG}
export UI_IMAGE=docker.jolibrain.com/colette_ui:${IMAGE_TAG}
export SMOKE_MODE=quick
bash ci/container_preflight.sh smoke-quick
bash ci/container_smoke.sh
```

Local example (API mode, requires GPU + HF token):

```bash
export IMAGE_TAG=$(git rev-parse --short=12 HEAD)
export GPU_BASE_IMAGE=docker.jolibrain.com/colette_gpu:${IMAGE_TAG}
export GPU_SERVER_IMAGE=docker.jolibrain.com/colette_gpu_server:${IMAGE_TAG}
export UI_IMAGE=docker.jolibrain.com/colette_ui:${IMAGE_TAG}
export SMOKE_MODE=api
export HF_TOKEN=hf_xxx
export MODELS_PATH=./models
export DATA_PATH=./docs/pdf
export APPS_PATH=./apps
bash ci/container_preflight.sh smoke-api
bash ci/container_smoke.sh
```

Local example (containerized integration tests):

```bash
export IMAGE_TAG=$(git rev-parse --short=12 HEAD)
export GPU_BASE_IMAGE=docker.jolibrain.com/colette_gpu:${IMAGE_TAG}
export HF_TOKEN=hf_xxx
export MODELS_PATH=./models
export TESTS_PATH=./tests
export PYTEST_ARGS=-q
export TEST_FILES="/app/tests/test_pipeline_python_api_integration.py /app/tests/test_pipeline_tantivy_integration.py"
bash ci/container_preflight.sh integration
bash ci/container_integration.sh
```

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
docker run --rm --runtime=nvidia --user $(id -u):$(id -g) -v /path/to/data:/data -p 1873:1873 colette_gpu_server:latest
```

Note: `docker/gpu_server.Dockerfile` depends on a local `colette_gpu:<GTAG>` image.

### Building with Docker for CPU
It is recommended to run on GPU, though for CPU-only platforms, proceed as below:

```bash
docker build --no-cache -t colette_cpu_build -f docker/cpu_build.Dockerfile .
docker build --no-cache -t colette_cpu_server -f docker/cpu_server.Dockerfile .
docker run --user $(id -u):$(id -g) -v /path/to/data:/data -p 1873:1873 colette_cpu_server:latest
```

Note: there is no dedicated CPU compose file in this repository at the moment.

### Building the Web User Interface with Docker

```bash
docker build --no-cache -t colette_ui -f docker/ui.Dockerfile .
```
