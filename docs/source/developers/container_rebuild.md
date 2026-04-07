# Container Rebuild Guide

This guide explains how to rebuild Colette images when code changes.

It covers:

- Local source rebuild with Docker Compose
- Registry-oriented rebuild and push flow
- GPU and CPU variants
- Backend-only and backend+UI scenarios

## Compose Files At A Glance

- `docker-compose-backend-local-container.yml`
  - Local source build, backend only
  - Uses `docker/gpu_server.Dockerfile`
  - `docker/gpu_server.Dockerfile` expects a local base image named `colette_gpu:<GTAG>`
- `docker-compose-backend-ui-local-container.yml`
  - Local source build, backend + UI
  - Backend uses `docker/gpu_server.Dockerfile`
  - UI uses `docker/ui.Dockerfile`
- `docker-compose-backend.yml`
  - Backend only
  - Builds from `docker/gpu_jb_server.Dockerfile`
  - Base image comes from `docker.jolibrain.com/colette_gpu:<GTAG>`
- `docker-compose-backend-ui.yml`
  - Backend + UI
  - Backend uses `docker/gpu_jb_server.Dockerfile`
  - UI uses `docker/ui.Dockerfile`

## 0. Prerequisites

1. Create `.env` from template.

```bash
cp env_template .env
```

2. Set required variables in `.env`.

```env
USER_ID=<id -u>
GROUP_ID=<id -g>
MODELS_PATH=./models
DATA_PATH=./docs/pdf
APPS_PATH=./apps
APP_NAME=app_colette
HF_TOKEN=hf_xxx
```

3. Create host directories (to avoid root-owned files in mounted volumes).

```bash
mkdir -p models apps apps/.cache
```

4. Validate the compose file before running.

```bash
docker compose -f docker-compose-backend-local-container.yml --env-file .env config > /tmp/colette.compose.out
```

## 1. Local Source Rebuild (GPU)

Use this flow when you changed Python code and want to run it immediately from local source context.

### 1.1 Ensure local base image exists

`docker/gpu_server.Dockerfile` uses `FROM colette_gpu:<GTAG>`. Build or tag it locally first.

Option A: build base image from source

```bash
docker build -t colette_gpu:latest -f docker/gpu_build.Dockerfile .
```

Option B: pull official base and retag locally

```bash
docker pull docker.jolibrain.com/colette_gpu:latest
docker tag docker.jolibrain.com/colette_gpu:latest colette_gpu:latest
```

### 1.2 Backend-only rebuild

```bash
docker compose -f docker-compose-backend-local-container.yml --env-file .env build gpu_server
docker compose -f docker-compose-backend-local-container.yml --env-file .env up -d --force-recreate
```

### 1.3 Backend + UI rebuild

```bash
docker compose -f docker-compose-backend-ui-local-container.yml --env-file .env build
docker compose -f docker-compose-backend-ui-local-container.yml --env-file .env up -d --force-recreate
```

### 1.4 Force clean rebuild

Use this when dependencies or Dockerfiles changed.

```bash
docker compose -f docker-compose-backend-local-container.yml --env-file .env build --no-cache gpu_server
docker compose -f docker-compose-backend-local-container.yml --env-file .env up -d --force-recreate
```

## 2. Registry-Oriented Rebuild (GPU)

Use this flow if you need immutable tags for environment promotion.

For CI-based build and publish on protected branches, use `Jenkinsfile.images`.
It builds `colette_gpu`, `colette_gpu_server`, and `colette_ui` and tags them with short Git SHA.

1. Build backend server image from current code using the registry base image.

```bash
export COLETTE_TAG=2026.04.02-dev1
export COLETTE_REGISTRY=<your-registry>

docker build \
  -f docker/gpu_jb_server.Dockerfile \
  --build-arg GTAG=latest \
  -t ${COLETTE_REGISTRY}/colette_gpu_server:${COLETTE_TAG} \
  .
```

2. Build UI image.

```bash
docker build \
  -f docker/ui.Dockerfile \
  -t ${COLETTE_REGISTRY}/colette_ui:${COLETTE_TAG} \
  .
```

3. Push images.

```bash
docker push ${COLETTE_REGISTRY}/colette_gpu_server:${COLETTE_TAG}
docker push ${COLETTE_REGISTRY}/colette_ui:${COLETTE_TAG}
```

4. Deploy with pinned tags.

Use an override compose file, or set `image:` tags explicitly in deployment manifests so runtime does not drift to `latest`.

## 3. CPU Rebuild Path

CPU support exists via Dockerfiles but is not wired into dedicated compose files in this repository.

### 3.1 Build CPU images

```bash
docker build -t colette_cpu_build:latest -f docker/cpu_build.Dockerfile .
docker build -t colette_cpu_server:latest -f docker/cpu_server.Dockerfile .
```

### 3.2 Run CPU backend container

```bash
docker run --rm -p 1873:1873 \
  --user $(id -u):$(id -g) \
  -e HF_TOKEN=${HF_TOKEN} \
  -v ${MODELS_PATH}:/models \
  -v ${DATA_PATH}:/data \
  -v ${APPS_PATH}:/rag \
  colette_cpu_server:latest
```

## 4. Change Impact Matrix

- Python code under `src/`, `server/`, `apps/`
  - Rebuild required: yes
  - Scope: `gpu_server` (and `ui` if UI code changed)
- Dependency files (`pyproject.toml`, `requirements_cpu_build.txt`, Docker install lines)
  - Rebuild required: yes, clean rebuild recommended
  - Scope: affected base + dependent server image
- Compose files or `.env` only
  - Rebuild required: no
  - Scope: `docker compose up -d --force-recreate`
- Model/data volume content only
  - Rebuild required: no
  - Scope: restart service if needed

## 5. Verification Checklist

1. API health:

```bash
curl -fsS http://localhost:1873/v1/info >/dev/null && echo "api-ok"
```

2. UI health (if running):

```bash
curl -I http://localhost:7860
```

3. Confirm active services and image IDs:

```bash
docker compose -f docker-compose-backend-local-container.yml --env-file .env ps
docker images | grep -E 'colette_gpu_server|colette_ui|colette_cpu_server'
```

4. Check logs if healthcheck fails:

```bash
docker compose -f docker-compose-backend-local-container.yml --env-file .env logs --tail=200 gpu_server
```

## 6. Scripted Container Smoke Tests

Dependency installation source of truth:

- `scripts/install_python_deps.sh` is used by both `create_venv_colette.sh` and `docker/gpu_build.Dockerfile`.
- This keeps local and container dependency resolution aligned and is the intended extension point for ARM/DGX variants.

Before smoke tests, build images with the reusable build script:

```bash
export REGISTRY=docker.jolibrain.com
export IMAGE_TAG=$(git rev-parse --short=12 HEAD)
export NO_CACHE=false
bash ci/container_preflight.sh build
bash ci/build_container_images.sh
```

Optional single target build:

```bash
export REGISTRY=docker.jolibrain.com
export IMAGE_TAG=$(git rev-parse --short=12 HEAD)
export TARGET=ui
bash ci/build_container_images.sh
```

Use `ci/container_smoke.sh` to validate built images before push.

Quick mode (fast import and structure checks):

```bash
export IMAGE_TAG=$(git rev-parse --short=12 HEAD)
export GPU_BASE_IMAGE=docker.jolibrain.com/colette_gpu:${IMAGE_TAG}
export GPU_SERVER_IMAGE=docker.jolibrain.com/colette_gpu_server:${IMAGE_TAG}
export UI_IMAGE=docker.jolibrain.com/colette_ui:${IMAGE_TAG}
export SMOKE_MODE=quick
bash ci/container_preflight.sh smoke-quick
bash ci/container_smoke.sh
```

API mode (starts GPU server and checks `/v1/info`):

```bash
export IMAGE_TAG=$(git rev-parse --short=12 HEAD)
export GPU_BASE_IMAGE=docker.jolibrain.com/colette_gpu:${IMAGE_TAG}
export GPU_SERVER_IMAGE=docker.jolibrain.com/colette_gpu_server:${IMAGE_TAG}
export UI_IMAGE=docker.jolibrain.com/colette_ui:${IMAGE_TAG}
export SMOKE_MODE=api
export SMOKE_TIMEOUT_SEC=180
export HF_TOKEN=hf_xxx
export MODELS_PATH=./models
export DATA_PATH=./docs/pdf
export APPS_PATH=./apps
export CONTAINER_UID=$(id -u)
export CONTAINER_GID=$(id -g)
bash ci/container_preflight.sh smoke-api
bash ci/container_smoke.sh
```

Use `ci/container_integration.sh` for heavier end-to-end integration suites in the GPU base image.

```bash
export IMAGE_TAG=$(git rev-parse --short=12 HEAD)
export GPU_BASE_IMAGE=docker.jolibrain.com/colette_gpu:${IMAGE_TAG}
export HF_TOKEN=hf_xxx
export MODELS_PATH=./models
export TESTS_PATH=./tests
export PYTEST_ARGS=-q
export TEST_FILES="/app/tests/test_pipeline_python_api_integration.py /app/tests/test_pipeline_tantivy_integration.py"
export CONTAINER_UID=$(id -u)
export CONTAINER_GID=$(id -g)
bash ci/container_preflight.sh integration
bash ci/container_integration.sh
```

Jenkins integration:

- `Jenkinsfile.images` runs smoke tests after image build and before image push.
- `SMOKE_MODE=quick` does not require HF credentials.
- `SMOKE_MODE=api` uses Jenkins `hf` credential and validates API startup.
- Optional `RUN_INTEGRATION=true` adds containerized integration tests before push.
- `INTEGRATION_PYTEST_ARGS` and `INTEGRATION_TEST_FILES` let Jenkins tune integration scope.

## 7. Common Issues

- `pull access denied for colette_gpu`
  - Build `colette_gpu:latest` locally or retag from `docker.jolibrain.com/colette_gpu:latest`
- Permission errors in mounted folders
  - Ensure `.env` contains correct `USER_ID` and `GROUP_ID`
- `HF_TOKEN` auth failures
  - Regenerate token and restart containers
- Port conflicts on `1873` or `7860`
  - Stop conflicting services or change host ports in compose
- GPU resource mismatch
  - Current compose files reserve `count: 2` GPUs
- `ModuleNotFoundError: No module named 'torch'` while building `flash-attn`
  - Root cause is pip build isolation hiding torch during `pip install -e .[dev,trag]`
  - Docker build now follows `create_venv_colette.sh` order: install torch first, then `flash-attn --no-build-isolation`, then editable install.
  - Rebuild with updated `docker/gpu_build.Dockerfile` and clear stale builder cache:
    `docker builder prune -af && bash ci/build_container_images.sh`
- `Invalid cross-device link` while building `flash-attn`
  - Root cause is pip using a temp directory on a different filesystem than the Docker BuildKit cache mount.
  - `scripts/install_python_deps.sh` now forces `TMPDIR` under `PIP_CACHE_DIR` so wheel build and cache operations stay on the same filesystem.
- `Permission denied` under `models/models--.../.no_exist` or `models/models--.../refs/main`
  - Root cause is mixed ownership in local model cache directories (often created by root from earlier container runs).
  - Current helper scripts (`ci/container_smoke.sh`, `ci/container_integration.sh`) default to running containers as your local UID:GID to avoid creating root-owned cache files.
  - Fix ownership once from a root container:
    `UID_GID="$(id -u):$(id -g)" && docker run --rm --user 0:0 -v "$PWD/models:/models" ubuntu:22.04 bash -lc "chown -R ${UID_GID} /models && chmod -R u+rwX,g+rwX /models"`
  - Keep using `--user $(id -u):$(id -g)` in docker run examples to avoid reintroducing root-owned files.
