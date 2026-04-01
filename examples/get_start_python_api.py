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
config_file = os.path.join(colette_root, 'src/colette/config/vrag_default.json')
index_file = os.path.join(colette_root, 'src/colette/config/vrag_default_index.json')

with open(config_file, 'r') as f:
    create_config = json.load(f)
with open(index_file, 'r') as f:
    index_config = json.load(f)

create_config['app']['repository'] = app_dir
create_config['app']['models_repository'] = models_dir
index_config['parameters']['input']['data'] = [documents_dir]
# Index once in hybrid mode so vector + text-search data are both available.
# Then per-request retrieval_mode can be 'embedding_retrieval', 'text_search_retrieval', or 'hybrid'
# without a second service_index call.
index_config['parameters']['input']['rag']['retrieval_mode'] = 'hybrid'
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


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval mode examples
# ─────────────────────────────────────────────────────────────────────────────
# Colette supports three retrieval modes:
#   "embedding_retrieval"  - (default) visual embedding retrieval in ChromaDB / ColDB only
#   "text_search_retrieval" - lexical text search (BM25) only
#   "hybrid"    - hybrid retrieval (vector + text search)
#
# The mode can be set:
#   A) In the index config so all queries default to it, or
#   B) Per-request via the predict payload (overrides the service default).

# A) Index mode -----------------------------------------------------------------
# This script already indexed with retrieval_mode='hybrid' above. That mirrors
# production usage and avoids a second service_index call.

# B) Per-request override -------------------------------------------------------
# Even if the service was indexed with mode="embedding_retrieval" you can switch at query time.
# Only the fields you explicitly set in the request 'rag' block override the
# service default; everything else inherits from the service config.

# Query using text search only (keyword/BM25 search)
# Only retrieval_mode is overridden here; text_search_engine_* values are inherited from
# the service/index JSON defaults unless you explicitly set them.
query_text_search_only = {
    'parameters': {
        'input': {
            'message': 'What are the identified sources of errors ?',
            'rag': {
                'retrieval_mode': 'text_search_retrieval',
            }
        }
    }
}
# response_text_search = colette_api.service_predict(app_name, APIData(**query_text_search_only))
#
# Note: for retrieval_mode='text_search_retrieval', embedding context is empty and text hits are
# returned in response_text_search.sources['text_context'].
#
# for text_hit in response_text_search.sources.get('text_context', []):
#     print(f"Source: {text_hit['source']}  page: {text_hit.get('page_number')}")
#     print(f"Score : {text_hit.get('score', 'n/a')}")
#     print(text_hit['content'][:200])
#     print()

# Query using hybrid mode (visual image crops + injected text context)
query_both_modes = {
    'parameters': {
        'input': {
            'message': 'What are the identified sources of errors ?',
            'rag': {
                'retrieval_mode': 'hybrid',
            }
        }
    }
}
# response_both = colette_api.service_predict(app_name, APIData(**query_both_modes))

# Inspect text_context sources --------------------------------------------------
# When retrieval_mode includes "text_search_retrieval", the response sources include a
# 'text_context' list alongside the image 'context' list.
#
# for text_hit in response_both.sources.get('text_context', []):
#     print(f"Source: {text_hit['source']}  page: {text_hit.get('page_number')}")
#     print(f"Score : {text_hit.get('score', 'n/a')}")
#     print(text_hit['content'][:200])
#     print()