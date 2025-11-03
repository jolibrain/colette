# Get Started on ARM Machine

## Prerequisites

Make sure your system meets the following requirements:

* **Python** >= 3.12
* **CUDA** >= 12.1
* **GPU**: ≥ 24 GB
* **RAM**: ≥ 16 GB
* **Disk space**: ≥ 50 GB

NOTE: Colette requires a GPU with at least 24GB of VRAM to run the default models. If you have less VRAM, you can try to change the models to lighter ones in the configuration files, but performance may be impacted.

Also the default config file is `vrag_default_lite.json` which is designed to run on 24GB VRAM GPUs. If you have multiples GPUs, you can try `vrag_default.json` which uses larger models and should provide better results and also needs multiple GPUs

## Docker with ARM support

Its working in progress to provide a Docker image with ARM support. Meanwhile, you can install from sources as described below.

## Installation from Source

### Command line & Developer Setup (Python API)

1. Clone the repo:

```bash
git clone https://github.com/jolibrain/colette.git
```

2. Create a virtual environment and install dependencies

```bash
cd colette
chmod +x create_venv_colette_ARM.sh
./create_venv_colette_ARM.sh
source venv_colette/bin/activate
```

NOTE: This process may take a while, as there are many dependencies to install and some of them require compilation.

##### Index the data

Let's index a PDF slidedeck from docs/pdf

```bash
colette_cli index --app-dir app_colette --data-dir docs/pdf/ --config-file src/colette/config/vrag_default_lite.json
```

##### Test with a question

```bash
colette_cli chat --app-dir app_colette --msg "What are the identified sources of errors ?" #--crop-label "text"
```

## Python API

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
