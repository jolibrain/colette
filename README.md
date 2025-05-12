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

## Demo

![](https://www.colette.chat/img/screenshot_1.png)


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

* GPU >= 24GB
* RAM >= 16GB
* Disk >= 50GB
* Docker >= 24.0.0 & Docker Compose >= v2.26.1
  > If you have not installed Docker on your local machine (Windows, Mac, or Linux),
  > see [Install Docker Engine](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Docker (recommended)
Easiest way to get started uses Docker. If you with to install from sources, see [Developer Setup](https://colette.chat/doc/developers/setup.html)

1. Pull the Docker image

```bash
docker pull docker.jolibrain.com/colette_gpu
```

2. Index your data

```bash
docker run --runtime=nvidia -v $PWD:/rag -v $PWD/docs:/data -v $PWD/models:/app/models docker.jolibrain.com/colette_gpu colette_cli index --app-dir /rag/app_colette --data-dir /data/pdf --config-file src/colette/config/vrag_default.json --models-dir /app/models
```


3. Test by sending a question

```bash
docker run --runtime=nvidia -v $PWD:/rag -v $PWD/models:/app/models docker.jolibrain.com/colette_gpu colette_cli chat --app-dir app_colette --models-dir /app/models --msg "What are the identified sources of errors of a RAG?"
```

### Command line

1. Clone the repo:

```bash
git clone https://github.com/jolibrain/colette.git
```

2. Install dependencies

```bash
pip install -e .[dev,trag]
```

3. Index the data

Let's index a PDF slidedeck from docs/pdf

```bash
colette_cli index --app-dir app_colette --data-dir docs/pdf/ --config-file src/colette/config/vrag_default.json
```

4. Test with a question

```bash
colette_cli chat --app-dir app_colette --msg "What are the identified sources of errors ?"
```

### Python API

Index PDFs and query them.

```bash
git clone https://github.com/jolibrain/colette.git
pip install -e .[dev,trag]
```

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
create_config['app']['repository'] = app_dir
index_config['parameters']['input']['data'] = [documents_dir]
#index_config['parameters']['input']['rag']['reindex'] = False # if True, the RAG will be reindexed

# index the documents
api_data_create = APIData(**create_config)
api_data_index = APIData(**index_config)
colette_api.service_create(app_name, api_data_create)
colette_api.service_index(app_name, api_data_index)



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
