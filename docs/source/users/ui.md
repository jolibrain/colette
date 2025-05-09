# Colette Web User Interface

## Server + Web User Interface via Docker Compose

### 1. Clone the repository

```bash
git clone https://git.jolibrain.com/jolibrain/colette_v2.git
cd colette_v2
```

---

### 2. Define your data paths and token

`Colette` uses **three main directories**:

- `models/`: stores all models used by Colette
- `data/`: contains the documents to be indexed
- `apps/`: holds all application-related data indexed with Colette (see Get Started section)

Additionally, you may need a HuggingFace account and token to download the models:
🔗 [Create a HuggingFace account](https://huggingface.co/join)

Create a `.env` file at the root of the project with the following content:

```
APPS_PATH=<path to the location where Colette will store its data>
MODELS_PATH=<path to the location where Colette will store its models>
DATA_PATH=<path to the location where Colette will find the documents to be indexed>
APP_NAME=<name of your application>
HUGGINGFACE_TOKEN=<your_token>
```

For example:

```
APPS_PATH=./
MODELS_PATH=./models
DATA_PATH=./docs/pdf
APP_NAME=app_colette
```

---

### 3. Launch the server and UI

To start the server and UI using the pre-built Docker images:

```bash
docker compose --env-file .env up
```

Once the application is initialized, the UI will be available at [http://localhost:7860](http://localhost:7860).

> ⚠️ **Attention**
> The **first application startup** takes some time, as it downloads the required models from HuggingFace.
> This may take **several minutes** depending on your internet speed. **Be patient**.

To stop the server and UI:

```bash
docker compose --env-file .env down
```
