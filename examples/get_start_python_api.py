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

documents_dir = os.path.join(colette_root, 'docs/pdf') # where the input documents are located
app_dir = os.path.join(colette_root, 'app_colette') # where to store the app
models_dir = os.path.join(colette_root, 'models') # where the models are located

# If the app already has a saved config (query-only run), use it.
# On a fresh install fall back to the default template.
_app_config = os.path.join(app_dir, 'config.json')
config_file = _app_config if os.path.exists(_app_config) else os.path.join(colette_root, 'src/colette/config/vrag_default.json')
index_file = os.path.join(colette_root, 'src/colette/config/vrag_default_index.json')

app_name = 'app_colette' # name of the app

colette_api = JSONApi()

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


# Clean up any service from a previous cell run in this session, then (re)create
try:
    colette_api.service_delete(app_name)
except Exception:
    pass  # no-op if service doesn't exist yet

# Create the service
api_data_create = APIData(**create_config)
colette_api.service_create(app_name, api_data_create)

# uncomment the following line to index the documents, only if they have not been indexed yet. 
# It may take a while depending on the number of documents and their size.
api_data_index = APIData(**index_config)
colette_api.service_index(app_name, api_data_index)

# Query the service with a question
question = 'What are the identified sources of errors ?'

# Note the optional 'crop_label' parameter to filter the sources by crop label
# The default crop labels are: 'text', 'table', 'figure'

# Query the vision RAG
query_api_msg = {
    'parameters': {
        'input': {
            'message': question
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
            'message': question,
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
            'message': question,
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


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical Indexing (vision_hierarchical retrieval mode)
# ─────────────────────────────────────────────────────────────────────────────
# hindex builds a hierarchical JSON tree from the document's table of contents
# using VLM OCR, then at query time navigates the tree with the VLM to select
# the most relevant sections and returns their full page images.
#
# Two ways to activate hindex:
#   A) retrieval_mode="vision_hierarchical" — hindex is the sole retrieval source.
#   B) retrieval_mode="hybrid" + hindex.enable_hindex=True — hindex results are
#      merged with embedding and BM25 hits for maximum coverage.
#
# NOTE: hindex requires a VLM (vision model) to be configured.  It runs
# automatically during service_index when the mode is set.

# A) Index with vision_hierarchical mode ----------------------------------------
# Re-index (or index for the first time) with hindex active.  The VLM will
# OCR the first N pages to extract the TOC and build a JSON hierarchy tree.

index_config_hindex = {
    'parameters': {
        'input': {
            'data': [documents_dir],
            'rag': {
                'retrieval_mode': 'vision_hierarchical',
                'hindex': {
                    'toc_scan_pages': 20,       # how many leading pages to scan for TOC
                    'add_node_summaries': True,  # ask VLM to summarise each section
                },
            }
        }
    }
}

# api_data_hindex_index = APIData(**index_config_hindex)
# colette_api.service_index(app_name, api_data_hindex_index)

# B) Query with vision_hierarchical mode ----------------------------------------
# The VLM first selects the relevant document (when multiple are indexed), then
# navigates the hierarchy tree and returns the full page images for the chosen
# section — without any text chunking or embedding lookup.

query_vision_hierarchical = {
    'parameters': {
        'input': {
            'message': question,
            'rag': {
                'retrieval_mode': 'vision_hierarchical',
            }
        }
    }
}

# response_hindex = colette_api.service_predict(app_name, APIData(**query_vision_hierarchical))
# print(response_hindex.output)
#
# # hindex page images come back in sources['context'] like any other retrieval mode.
# # Each item also carries retrieval_source='hindex' in its metadata.
# for item in response_hindex.sources.get('context', []):
#     print(f"Key: {item['key']}")
#     print(f"Source: {item.get('source', 'n/a')}")
#     print(f"Page:   {item.get('page_number', 'n/a')}")
#     print(f"Retrieval source: {item.get('retrieval_source', 'n/a')}")  # 'hindex'
#     image = display_image_from_data_uri(item['content'])
#     print("-" * 50)

# C) Hybrid mode with hindex enabled --------------------------------------------
# Index once with hybrid + enable_hindex=True to get all three sources:
#   - visual embedding retrieval (ChromaDB)
#   - BM25 text search (Tantivy)
#   - hierarchical navigation (hindex)

index_config_hybrid_hindex = {
    'parameters': {
        'input': {
            'data': [documents_dir],
            'rag': {
                'retrieval_mode': 'hybrid',
                'hindex': {
                    'enable_hindex': True,
                    'toc_scan_pages': 20,
                },
            }
        }
    }
}

# api_data_hybrid_hindex = APIData(**index_config_hybrid_hindex)
# colette_api.service_index(app_name, api_data_hybrid_hindex)

# At query time, use hybrid retrieval_mode and hindex results are automatically merged.
query_hybrid_with_hindex = {
    'parameters': {
        'input': {
            'message': question,
            'rag': {
                'retrieval_mode': 'hybrid',
            }
        }
    }
}

# response_hybrid_hindex = colette_api.service_predict(
#     app_name, APIData(**query_hybrid_with_hindex)
# )
# print(response_hybrid_hindex.output)
#
# # Distinguish hindex hits from embedding hits via the retrieval_source metadata field.
# for item in response_hybrid_hindex.sources.get('context', []):
#     src = item.get('retrieval_source', 'embedding')
#     print(f"[{src}] {item['key']}  page={item.get('page_number', 'n/a')}")

# D) Inspect the built hierarchy tree -------------------------------------------
# After indexing, the hierarchy trees are stored as JSON files under
# <app_dir>/mm_index/hindex/*.json.  You can read them with DocumentHierarchy.

import os as _os
import json as _json
import sys as _sys
_sys.path.insert(0, _os.path.join(colette_root, 'src'))

from pathlib import Path as _Path
from colette.backends.hf.hindex import DocumentHierarchy

hindex_dir = _Path(app_dir) / 'mm_index' / 'hindex'
if hindex_dir.exists():
    for json_file in sorted(hindex_dir.glob('*.json')):
        data = _json.loads(json_file.read_text(encoding='utf-8'))
        h = DocumentHierarchy.from_dict(data)
        print(f'\n=== {h.doc_name}  ({h.page_count} pages) ===')
        print(h.as_text_tree())