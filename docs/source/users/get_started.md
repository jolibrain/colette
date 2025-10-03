# Get Started

## Prerequisites

Make sure your system meets the following requirements:

* **Python** >= 3.12
* **CUDA** >= 12.1
* **GPU**: ≥ 24 GB
* **RAM**: ≥ 16 GB
* **Disk space**: ≥ 50 GB
* **Docker**: ≥ 24.0.0
* **Docker Compose**: ≥ v2.26.1
  > 💡 If Docker with GPU support is not installed on your machine (Windows, macOS, or Linux), follow the instructions at [Install Docker Engine](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Command line client via Docker
Easiest way to get started uses Docker. If you with to install from sources, see https://colette.chat/doc/developers/setup.html

1. Pull the Docker image

```bash
docker pull docker.jolibrain.com/colette_gpu
```

2. Create folders for models and app_colette

```bash
mkdir -p models
mkdir -p app_colette
```
This is to ensure that your user is the owner of these folders, as the docker container runs as root.

3. Index your data

```bash
docker run --gpus all --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -v $PWD:/rag \
  -v $PWD/docs:/data \
  -v $PWD/models:/app/models \
  docker.jolibrain.com/colette_gpu \
  bash -c "git config --global --add safe.directory /app && colette_cli index --app-dir /rag/app_colette --data-dir /data/pdf --config-file src/colette/config/vrag_default.json --models-dir /app/models"
```

4. Test by sending a question

```bash
docker run --gpus all --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -v $PWD:/rag \
  -v $PWD/app_colette:/app/app_colette \
  -v $PWD/models:/models \
  docker.jolibrain.com/colette_gpu \
  bash -c "git config --global --add safe.directory /app && colette_cli chat --app-dir app_colette --models-dir /models --msg \"What are the identified sources of errors of a RAG?\""
```

### Activate `venv_colette` for Command line & Developer Setup (Python API)

1. Clone the repo:

```bash
git clone https://github.com/jolibrain/colette.git
```

2. Create a virtual environment and install dependencies

```bash
cd colette
chmod +x create_venv_colette.sh
./create_venv_colette.sh
source venv_colette/bin/activate
```

NOTE: This process may take a while, as there are many dependencies to install and some of them require compilation.

##### Index the data

Let's index a PDF slidedeck from docs/pdf

```bash
colette_cli index --app-dir app_colette --data-dir docs/pdf/ --config-file src/colette/config/vrag_default.json
```

##### Test with a question

```bash
colette_cli chat --app-dir app_colette --msg "What are the identified sources of errors ?"
```

## Python API

(don't forget to activate the virtual environment, see above)

##### Index PDFs and query

```Python
import json
from colette.jsonapi import JSONApi
from colette.apidata import APIData

colette_api = JSONApi()

documents_dir = 'docs/pdf' # where the input documents are located
app_dir = 'app_colette' # where to store the app
app_name = 'colette_doc'

# read the configuration file
config_file = 'src/colette/config/vrag_default.json'
index_file = 'src/colette/config/vrag_default_index.json'
with open(config_file, 'r') as f:
    create_config = json.load(f)
with open(index_file, 'r') as f:
    index_config = json.load(f)
index_config['parameters']['input']['data'] = [documents_dir]
#index_config['parameters']['input']['rag']['reindex'] = False # if True, the RAG will be reindexed

# index the documents
api_data_create = APIData(**create_config)
api_data_index = APIData(**index_config)
colette_api.service_create(app_name, api_data_create)
colette_api.service_index(app_name, api_data_index)
colette_api.start_indexing_loop()

# query the vision RAG
query_api_msg = {
    'parameters': {
        'input': {
            'message': 'What are the identified sources of errors ?'
        }
    }
}
query_data = APIData(**query_api_msg)
response = colette_api.service_predict(app_name, query_data)
print(response)
```
