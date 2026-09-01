# Get Started on DGX Machine

## Prerequisites

Make sure your system meets the following requirements:

* **Python** 3.12 (the bootstrap script requires exactly 3.12)
* **CUDA** >= 12.8, CUDA 13 recommended
* **Memory**: >= 80 GB available to the GPU
* **Disk space**: >= 120 GB
* **Architecture**: x86_64 or aarch64

The memory and disk figures are driven by the DGX default model,
`Qwen/Qwen3.8-27B`: roughly 52 GB of weights to download, ~51 GiB resident once
loaded, plus KV cache. On a DGX Spark (GB10, 121 GB unified memory) the model
occupies 51.1 GiB and leaves 17.1 GiB of KV cache at the config's
`vllm_memory_utilization` of 0.6. The virtual environment itself is ~11 GB, and
FAISS is compiled from source during installation.

To run on less memory, point `source` in `src/colette/config/vrag_default_DGX.json`
at a smaller model of the same family and lower `context_size`.

### A note on ARM

A DGX Spark is aarch64, so it is both "DGX" and "ARM". Use **this** guide and
`create_venv_colette_DGX.sh` for it. `create_venv_colette_ARM.sh` and
[Get Started on ARM Machine](get_started_ARM_machine.md) target other ARM boards
and pin an older dependency set.

## Docker with DGX support

You can use the standard Docker workflow from [Get Started](get_started.md). For DGX-specific source setup (CUDA-aware dependency selection), follow the installation steps below.

## Installation from Source

### Command line & Developer Setup (Python API)

1. Clone the repo:

```bash
git clone https://github.com/jolibrain/colette.git
```

2. Create a virtual environment and install dependencies

```bash
cd colette
chmod +x create_venv_colette_DGX.sh
./create_venv_colette_DGX.sh
source venv_colette/bin/activate
```

NOTE: This process may take a while, as there are many dependencies to install and some of them require compilation.

### Which configuration to use

**Use `src/colette/config/vrag_default_DGX.json` on DGX.** The bootstrap script
writes `COLETTE_VRAG_CONFIG` into the virtual environment's `activate`, so once
the environment is active the examples and the notebook pick it up
automatically. Nothing to export by hand:

```bash
source venv_colette/bin/activate
echo $COLETTE_VRAG_CONFIG
# /path/to/colette/src/colette/config/vrag_default_DGX.json
```

The other shipped configs do **not** work here. `vrag_default.json` and
`vrag_default_lite.json` serve their LLM through the `huggingface` backend, and
the DGX dependency set pins `transformers>=4.56,<5`, which cannot load the
`qwen3_5` architecture those configs use:

```
The checkpoint you are trying to load has model type `qwen3_5`
but Transformers does not recognize this architecture.
```

`vrag_default_DGX.json` serves the same family through vLLM, which does
implement it, and additionally caps the context at 8192, holds vLLM to 60% of
GPU memory, and pins layout detection to CPU so indexing stays off the GPU
memory budget.

To override the default, set `COLETTE_VRAG_CONFIG` yourself or pass
`--config-file` to the CLI.


##### Index the data

Let's index a PDF slidedeck from docs/pdf

```bash
colette_cli index --app-dir app_colette --data-dir docs/pdf/ --config-file src/colette/config/vrag_default_DGX.json
```

##### Test with a question

```bash
colette_cli chat --app-dir app_colette --msg "What are the identified sources of errors ?" #--crop-label "text"
```

## Python API

(don't forget to activate the virtual environment, see above)

The example below is also available in `examples/get_start_python_api.py`.
There is also a Jupyter notebook version in `examples/get_start_python_api.ipynb`.
For text-search-only examples, see `examples/text_search_demo.py` and `examples/text_search_demo.ipynb`.

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
config_file = os.environ.get(
    'COLETTE_VRAG_CONFIG',
    os.path.join(colette_root, 'src/colette/config/vrag_default_DGX.json'),
)
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

# Note the optional 'crop_label' parameter to filter the sources by crop label
# The default crop labels are: 'text', 'table', 'figure'

# Query the vision RAG
query_api_msg = {
    'parameters': {
        'input': {
            'message': 'What are the identified sources of errors ?'
            # 'crop_label': 'text'
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

## Notes and troubleshooting

**The first run downloads ~52 GB of model weights.** Expect a long wait before
anything appears to happen. The download is not resumable in practice — the
Hugging Face `xet` transfer discards partial chunks if the process is
interrupted — so let it finish. Pre-warming the cache once on a machine saves
everyone else the wait.

**Run one instance at a time.** vLLM reserves `vllm_memory_utilization` of GPU
memory up front, so two concurrent services will not fit. Two processes also
share `app_colette/`'s ChromaDB store, which surfaces as:

```
InternalError: Database error: (code: 1032) attempt to write a readonly database
```

**A saved app config overrides `COLETTE_VRAG_CONFIG`.** Once a service has been
created, `<app_dir>/config.json` exists and takes precedence, so query-only
reruns keep the settings they were indexed with. A stale app directory from a
failed run therefore pins you to the wrong model. To start clean:

```bash
rm -rf app_colette
```

**`service_create` does not raise on failure.** It returns an `APIResponse` with
every field set to `None`, and the underlying error appears only in the logs. The
next call then fails with a misleading `service <name> not found`. If you see
that, look at the log output of the *create* step, not the failing call.

**Reasoning traces are not shown.** `Qwen3.8-27B` is a thinking model: it reasons
before answering. Colette keeps the reasoning — it is what grounds the answer —
but strips it from the returned text, so `response.output` holds the final answer
only.

