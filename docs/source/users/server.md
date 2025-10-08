# Colette Server

## Server via Docker Compose

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
  bash -c "git config --global --add safe.directory /app && colette_cli index --app-dir /rag/app_colette --data-dir /data/pdf --config-file src/colette/config/vrag_default.json --models-dir /app/models"
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

### 4. Launch the server

To start the server using the pre-built Docker images:

```bash
docker compose -f docker-compose-backend.yml --env-file .env up
```

> ⚠️ **Attention**
> The **first application startup** takes some time, as it downloads the required models from HuggingFace.
> This may take **several minutes** depending on your internet speed. **Be patient**.

To stop the server:

```bash
docker compose -f docker-compose-backend.yml --env-file .env down
```

## Interrogate the server via API

### List Available Services

```bash
curl http://localhost:1873/v1/info
```

### Query Colette to get the text answer

```bash
curl -X POST http://localhost:1873/v1/predict/app_colette \
  -H "Content-Type: application/json" \
  -d '{
    "app": {"verbose": "info"},
    "parameters": {
      "input": {
        "message": "What are the identified sources of errors ?",
        "session_id": "test-session-123"
      }
    }
  }' | jq -r '.output'
  ```

#### Output

```
Based on the information provided in the document, the identified sources of errors include:

1. OCR (Optical Character Recognition) errors, which can lead to misinterpretation of tables and images.

2. Layout detection errors, such as incorrect table layouts.

3. Compression errors during text extraction, where approximations based on the model can lead to incorrect results.

4. Indexing errors, where similarity indices may not be accurate enough for complex applications.

5. Hallucinations or approximations made by the LLM (Large Language Model), which can lead to incorrect answers.

6. Biais introduced by the model itself, whether it's compression bias or hallucinations.

The document suggests that these errors accumulate over time, leading to a complex application process despite initial enthusiasm.
```

### Query Colette to get the text sources

```bash
curl -X POST http://localhost:1873/v1/predict/app_colette \
  -H "Content-Type: application/json" \
  -d '{
    "app": {"verbose": "info"},
    "parameters": {
      "input": {
        "message": "What are the identified sources of errors ?",
        "session_id": "test-session-123"
      }
    }
  }' | jq -r '.sources.context'
```

#### Output

```
[{'key': '00912abf-00d8-5ea9-a954-d697a20a4d4f_0004_crop_0002',
  'distance': 0.7925766110420227,
  'content': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABOgAAAbOCAIAAAB....
  'source': '/data/pdf/COLETTEv2_Restitution_2025_03_07_v0.3_JB_light.pdf',
  'crop': 2,
  'crop_label': 'text',
  'label': 'crop',
  'page_number': 5},
{'key': '00912abf-00d8-5ea9-a954-d697a20a4d4f_0004_crop_0000',
  'distance': 0.7736435532569885,
  'content': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAACKsAAAeoCAIAAABvelYrAAEAAEl...
  'page_number': 5,
  'label': 'crop',
  'source': '/data/pdf/COLETTEv2_Restitution_2025_03_07_v0.3_JB_light.pdf',
  'crop_label': 'table',
  'crop': 0},
  .
  .
  .
]
```