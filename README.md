<div align="center">
<a href="https://www.colette.chat/">
<img src="https://www.colette.chat/img/colette_logo.png" width="320" alt="colette logo">
</a>
</div>

<p align="center">
<b>Search and interact locally with technical documents of any kind</b>
</p>

## What is Colette?

Colette is an open-source self-hosted RAG and LLM serving software. It is well-suited for searching and interacting with technical documents that cannot be leaked to external APIs.

As the main core feature, Colette embeds a Vision-RAG (V-RAG) that transforms and analyzes all documents as images. This allows to conserve and handle all visual elements such as images, figures, schemas, visual highlights and layouts in documents. This is based on the idea that most documents are targeted at human eyes, and thus can be more thoroughtly analyzed by vision and multimodal LLMs.

Colette was co-financed by [Jolibrain](https://www.jolibrain.com/), [CNES](https://www.cnes.fr/) and [Airbus](https://www.airbus.com/en/products-services/space).

## Demo

https://github.com/user-attachments/assets/7e36b4af-880a-4260-af61-3041b7d60439

## Key Features

- 📊 Vision Retrieval-Augmented Generation (V-RAG) system by combining the Document Screenshot Embedding/ColPali retrievers for document retrieval with Vision Language Model (VLM).

- 📚 Text based RAG system by combining unstructured based text extraction, text embedding and common LLMs

- 🚀 Multi-Model Support for both embedders and inference VLLMs

- 🎨 Image Generation Integration with diffusers

- 🚀 Effortless Setup, dockerized and our tests show decent results on many corpuses, including technical documentations with images, figure and shemas


## System Architecture

![](https://www.colette.chat/img/colette_archi.png)

## Get Started

### Prerequisites

* Python >= 3.12
* CUDA >= 12.1
* GPU >= 24GB
* RAM >= 16GB
* Disk >= 50GB
* Docker >= 24.0.0 & Docker Compose >= v2.26.1
  > If you have not installed Docker on your local machine (Windows, Mac, or Linux),
  > see [Install Docker Engine](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

NOTE: Colette requires a GPU with at least 24GB of VRAM to run the default models. If you have less VRAM, you can try to change the models to lighter ones in the configuration files, but performance may be impacted.

Also the default config file is `vrag_default_lite.json` which is designed to run on 24GB VRAM GPUs. If you have multiples GPUs, you can try `vrag_default.json` which uses larger models and should provide better results and also needs multiple GPUs

### Docker (recommended)
Easiest way to get started uses Docker. If you with to install from sources, see [Developer Setup](https://colette.chat/doc/developers/setup.html)

1. Pull the Docker image

```bash
docker pull docker.jolibrain.com/colette_gpu
```

2. Create folders for models and app_colette

```bash
mkdir -p models
mkdir -p app_colette
```

This is to ensure that your user is the owner of these folders and files that will be created inside when we run the docker containers.

3. Index your data

```bash
docker run --gpus all --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -v $PWD:/rag \
  -v $PWD/docs:/data \
  -v $PWD/models:/app/models \
  docker.jolibrain.com/colette_gpu \
  bash -c "git config --global --add safe.directory /app && colette_cli index --app-dir /rag/app_colette --data-dir /data/pdf --config-file src/colette/config/vrag_default_lite.json --models-dir /app/models"
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

#### Command Line Interface (CLI)

(don't forget to activate the virtual environment, see above)

##### Index the data

Let's index a PDF slidedeck from docs/pdf

```bash
colette_cli index --app-dir app_colette --data-dir docs/pdf/ --config-file src/colette/config/vrag_default_lite.json
```

##### Test with a question

```bash
colette_cli chat --app-dir app_colette --msg "What are the identified sources of errors ?"
```

### Python API

(don't forget to activate the virtual environment, see above)

The example below is also available in `examples/get_start_python_api.py`.
There is also a Jupyter notebook version in `examples/get_start_python_api.ipynb`.

##### Index PDFs and query

```Python
import json
import re
import base64
from io import BytesIO
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from colette.jsonapi import JSONApi
from colette.apidata import APIData

# Get the root path of the colette package
import os
colette_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f'Colette root path: {colette_root}')

colette_api = JSONApi()

documents_dir = os.path.join(colette_root, 'docs/pdf') # where the input documents are located
app_dir = os.path.join(colette_root, 'app_colette') # where to store the app
models_dir = os.path.join(colette_root, 'models') # where the models are located
app_name = 'app_colette' # name of the app

# read the configuration file
config_file = os.path.join(colette_root, 'src/colette/config/vrag_default_lite.json')
index_file = os.path.join(colette_root, 'src/colette/config/vrag_default_index.json')

with open(config_file, 'r') as f:
    create_config = json.load(f)
with open(index_file, 'r') as f:
    index_config = json.load(f)

create_config['app']['repository'] = app_dir
create_config['app']['models_repository'] = models_dir
index_config['parameters']['input']['data'] = [documents_dir]
#index_config['parameters']['input']['rag']['reindex'] = False # if True, the RAG will be reindexed

# Create the service
api_data_create = APIData(**create_config)
colette_api.service_create(app_name, api_data_create)

# Index the documents
api_data_index = APIData(**index_config)
colette_api.service_index(app_name, api_data_index)

# Query the vision RAG
query_api_msg = {
    'parameters': {
        'input': {
            'message': 'What are the identified sources of errors ?'
        }
    }
}
query_data = APIData(**query_api_msg)
response = colette_api.service_predict(app_name, query_data)

# Get the text output
print(response.output)

# Get the image sources
for item in response.sources['context']:
    print(f"Key: {item['key']}, Distance: {item['distance']}")

    # Extract base64 string (remove 'data:image/png;base64,' prefix)
    base64_data = re.sub('^data:image/.+;base64,', '', item['content'])

    # Decode base64 string
    image_data = base64.b64decode(base64_data)
    
    # Create PIL Image
    image = Image.open(BytesIO(image_data))

    # Export image (optional)
    image_filename = f"{item['key']}.png"
    image.save(image_filename)
    print(f"Image saved as: {image_filename}")
```

## Configurations

Colette uses JSON files for every RAG service configuration.

- Example of default V-RAG configuration in `src/colette/config`.
- Full JSON configuration reference: [API](https://colette.chat/doc/api/api.html)

For more details on how to handle and customize configurations, see [Configuration](https://colette.chat/doc/users/configuration.html)

## Documentation

The main documentation is available from [Colette documentation](https://www.colette.chat/doc/)

## FAQ

The main FAQ is [Colette FAQ](https://www.colette.chat/doc/faq)

- What to do if/when Colette returns incorrect answers ?

  First you must know Colette will never work for everything. But there are ways to understand the difficulties, and work around them, most of the time.

  While Colette is designed on the theoritical and technical levels to work `correctly` on average, RAG pipelines suffer from multiple potential error sources (see https://colette.chat/documents/COLETTEv2_Restitution_2025_03_07_v0.3_JB_light.pdf on page 7 for a list).

  Below are a few steps to apply to identify causes of bad answers:

   1. Is the answer in one of the returned document sources ? If not, this is a `retrieval error`. This is the most probable RAG error. To address it:

      a. Make sure the relevant documents are in your corpus, and identify them
      b. Setup an independent RAG by indexing only those relevant documents, and checks whether the answer appears to be correct and the relevant documents are returned as sources
      c. If answer is correct, culprit is either indexing or retrieval, try a larger/different indexing model
      d. If answer is not correct but the relevant documents are returned as sources, culprit is the inference LLM, try a larger/different one
      e. If answer is not correct and the relevant documents are not returned as sources, culprit is the indexing model, try a larger/different one

   2. The returned document sources are correct, the answer is not, culprit is the inference LLM, try a larger/different one

  If this doesn't work, report the issue to us, make sure to be able to share a document if not private, so that we can look at it more closely.
 

- Colette generates errors, or you have difficulties installing it ?

  Look at the list of [issues](https://github.com/jolibrain/colette/issues), and if your problem is not listed already, write a new one.
