# Colette Web User Interface

## Server + Web User Interface via Docker Compose

NOTE: Colette requires a GPU with at least 24GB of VRAM to run the default models. If you have less VRAM, you can try to change the models to lighter ones in the configuration files, but performance may be impacted.

Also the default config file is `vrag_default_lite.json` which is designed to run on 24GB VRAM GPUs. If you have multiples GPUs, you can try `vrag_default.json` which uses larger models and should provide better results and also needs multiple GPUs

### 0. Get the GPU Colette image

```bash
docker pull docker.jolibrain.com/colette_gpu
```

### 1. Clone the repository

```bash
git clone https://github.com/jolibrain/colette.git
cd colette
```

### 2. Create the required directories

```bash
mkdir -p .cache/huggingface
mkdir -p models
mkdir -p app_colette # the name should match APP_NAME in the .env file
```

This will create the directories needed to store the models, application data, and HuggingFace cache under the current directory. Also, in that way, we will make sure that we own these directories with our user and group ids.

### 3. Index your documents

```bash
docker run --gpus all --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -v $PWD:/rag \
  -v $PWD/docs:/data \
  -v $PWD/models:/app/models \
  docker.jolibrain.com/colette_gpu \
  bash -c "git config --global --add safe.directory /app && colette_cli index --app-dir /rag/app_colette --data-dir /data/pdf --config-file src/colette/config/vrag_default_lite.json --models-dir /app/models"
```

### 4. Define your HuggingFace token and paths

#### 4.1 Create your HuggingFace token

Additionally, you may need to create a HuggingFace account and token to download the models:
🔗 [Create a HuggingFace account](https://huggingface.co/join)

#### 4.2 Define your paths

`Colette` uses **three main directories**:

- `models/`: stores all models used by Colette
- `data/`: contains the documents used for indexing
- `apps/`: holds all application-related data indexed with Colette (see Get Started section)

Create a `.env` file at the root of the project with the following content:

```markdown
- **USER_ID**: Your user ID on your machine (get it with `id -u`)

- **GROUP_ID**: Your group ID on your machine (get it with `id -g`)

- **APPS_PATH**: Path to the location where Colette will store its data

- **MODELS_PATH**: Path to the location where Colette will store its models

- **DATA_PATH**: Path to the location where Colette will find the documents used for indexing

- **APP_NAME**: Name of your application (e.g., `app_colette`)

- **HF_TOKEN**: Your HuggingFace token
```

`.env` file example:

```env
USER_ID=100
GROUP_ID=100
APPS_PATH=./
MODELS_PATH=./models
DATA_PATH=./docs/pdf
APP_NAME=app_colette
HF_TOKEN=your_huggingface_token
```

---

### 4. Launch the server and UI

To start the server and UI using the pre-built Docker images:

```bash
docker compose -f docker-compose-backend-ui.yml --env-file .env up
```

To rebuild backend and UI images from local source code before launching:

```bash
docker compose -f docker-compose-backend-ui-local-container.yml --env-file .env build
docker compose -f docker-compose-backend-ui-local-container.yml --env-file .env up -d --force-recreate
```

For full rebuild scenarios (cache strategy, registry tags, CPU flow), see `../developers/container_rebuild.md`.

Once the application is initialized, the UI will be available at [http://localhost:7860](http://localhost:7860).

> ⚠️ **Attention**
> The **first application startup** takes some time, as it downloads the required models from HuggingFace.
> This may take **several minutes** depending on your internet speed. **Be patient**.

To stop the server and UI:

```bash
docker compose -f docker-compose-backend-ui.yml --env-file .env down
```
