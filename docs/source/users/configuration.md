# Configuration

Colette provides two main interfaces for interaction:

- **JSON Interface**: Accessible via Python for direct integration.
- **HTTP Interface**: Usable with `curl` or any programming language that supports HTTP requests.

Colette operates as an application that is **configured through a JSON file**, allowing users to define its behavior before sending queries.

---

## Pre-configured RAG systems

Colette comes with several pre-configured RAG systems, each tailored for specific use cases. These systems are designed to simplify the setup process and provide a quick start for users.

They can be used using the Colette CLI:

### Indexing

To run the indexing phase.

```bash
colette_cli index [OPTIONS]

Options:

 *  --app-dir         PATH  Specify the application directory [default: None] [required]
 *  --data-dir        PATH  Specify the data directory [default: None] [required]
    --models-dir      PATH  Specify the models directory [default: None]
    --config-file     PATH  Specify the config file [default: None]
    --index-file      PATH  [default: <typer.models.OptionInfo object at 0x7f629528c0b0>]
    --help                  Show this message and exit.
```

### Querying

To start a chat with the application.

```bash
colette_cli chat [OPTIONS]

Options:

*  --app-dir           PATH  Specify the application directory [default: None] [required]
*  --msg               TEXT  Specify the user message [default: None] [required]
   --models-dir        PATH  Specify the models directory [default: None]
   --help                    Show this message and exit.
```

---

## Configuring a RAG-based System

Colette supports **Retrieval-Augmented Generation (RAG)**, enabling the ingestion and retrieval of documents to enhance answer generation. Below is an overview of the RAG pipeline used by Colette:

![RAG Architecture](https://www.colette.chat/img/colette_archi.png "RAG architecture")

### Overview of the RAG Process
The diagram above illustrates the **five key steps** involved in processing queries using a Retrieval-Augmented Generation (RAG) pipeline:

1. **Encoding**
   - Documents (e.g., PDFs, docx, ...) are **converted into embeddings** using an embedding model.
   - When a user submits a **query**, it is also encoded into an embedding.

2. **Indexing & Similarity Search**
   - Encoded document embeddings are **stored in a vector database** (e.g., ChromaDB).
   - When a query embedding is received, the system **searches for similar documents** in the vector database.

3. **Retrieving Relevant Chunks**
   - The **most relevant document chunks** are retrieved from the database.

4. **Generating a Prompt**
   - The retrieved chunks are **formatted into a structured prompt** that provides context to the language model.

5. **Generating the Response**
   - The prompt is passed to the **LLM**, which generates a **final response** based on the retrieved context.

---

### Configuration Structure

The configuration file consists of three main sections:

1. **`app` Section** - Defines the application repository and logging level.
2. **`parameters` Section** - Contains sub-sections related to data input, preprocessing, retrieval settings, and prompt templates.
3. **`llm` Section** - Specifies the LLM model and inference settings.

Below is a detailed breakdown of each section.

---

#### 1. Application Settings (`app`)
```json
{
    "app": {
        "repository": "/path/to/rag",
        "verbose": "info"
    }
}
```
This section controls **Colette’s internal configurations**, including:
- **`repository`**: The directory where Colette will store its internal files, configurations, and data.
- **`verbose`**: The logging level (e.g., `info`, `debug`).

---

#### **2. Parameters (`parameters`)**

The `parameters` section defines how **data is processed, indexed, and retrieved** before being passed to the LLM for answer generation.

##### 2.1 Input Settings (`input`)
```json
"input": {
    "preprocessing": {
        "files": ["all"],
        "lib": "unstructured",
        "save_output": false,
        "filters": ["\/~[^\/]*$"]
    },
    "rag": {
        "indexdb_lib": "chromadb",
        "retrieval_mode": "embedding_retrieval",
        "embedding_lib": "huggingface",
        "embedding_model": "intfloat/multilingual-e5-small",
        "gpu_id": 0,
        "text_search_engine_top_k": 4,
        "text_search_engine_fields": ["content", "source"],
        "text_search_engine_max_chars_per_doc": 1500,
        "text_search_engine_max_total_chars": 6000,
        "search": true,
        "reindex": false
    },
    "template": {
        "template_prompt": "Tu es un assistant expert dans le management des systèmes spatiaux et orbitaux. Réponds en francais en utilisant les informations du contexte qui suit. Contexte: {context}. Question: {question}. Réponse: ",
        "template_prompt_variables": ["context", "question"]
    },
    "data": ["/path/to/data/"]
}
```
###### Preprocessing (`preprocessing`)
This section controls how Colette **filters and processes files** before they are indexed in the RAG system:
- **`files`**: Determines which files to include (`["all"]` means all files are considered).
- **`lib`**: Specifies the library used for preprocessing (`unstructured` for document parsing).
- **`save_output`**: Boolean flag (`false` here) indicating whether to store preprocessed files.
- **`filters`**: Regex patterns to exclude files based on their names.

###### RAG Configuration (`rag`)
Controls settings for **retrieval and vector database indexing**:
- **`indexdb_lib`**: The vector database used (`chromadb`).
- **`retrieval_mode`**: Retrieval strategy (`embedding_retrieval`, `text_search_retrieval`, `hybrid`).
- **`embedding_lib`**: The library used to compute text embeddings (`huggingface`).
- **`embedding_model`**: The embedding model (`intfloat/multilingual-e5-small`).
- **`gpu_id`**: Specifies which GPU to use (`0` means the first available GPU).
- **`text_search_engine_top_k`**: Number of text-search hits to fetch.
- **`text_search_engine_fields`**: Indexed fields queried by the text search engine.
- **`text_search_engine_max_chars_per_doc`**: Max chars per retrieved text hit added to prompt context.
- **`text_search_engine_max_total_chars`**: Global char cap added to prompt context from text search hits.
- **`search`**: If `true`, retrieval is enabled.
- **`reindex`**: If `false`, existing indexes are used instead of re-processing everything.

###### Prompt Template (`template`)
Defines the **system prompt** used when querying the LLM:
- **`template_prompt`**: The structured prompt instructing the assistant to respond as an expert in space systems management.
- **`template_prompt_variables`**: The placeholders used in the prompt (`context`, `question`).

###### Data Sources (`data`)
Defines **where Colette retrieves documents** for processing:
- **`data`**: The directory containing documents (`/path/to/data/`).

---

#### 3. LLM Configuration (`llm`)
```json
"llm": {
    "source": "llama3.1:latest",
    "inference": {
        "lib": "ollama"
    }
}
```
This section defines **the language model used for answering queries**:
- **`source`**: The LLM model version (`llama3.1:latest`).
- **`inference`**:
  - **`lib`**: The inference library (`ollama`).

---

### How Colette Processes RAG Queries
1. **Preprocessing**: Colette filters and processes input documents based on the defined rules.
2. **Indexing**: The documents are embedded using the `huggingface` embedding model (`intfloat/multilingual-e5-small`) and stored in `chromadb`.
3. **Retrieval**: When a user submits a query, Colette searches the document embeddings to retrieve relevant context.
4. **Prompt Construction**: The retrieved context is inserted into the `template_prompt`.
5. **LLM Query**: The constructed prompt is sent to `llama3.1:latest` using `ollama` for inference.
6. **Response Generation**: The model generates an answer using both the retrieved documents and its own knowledge.

---

## Example API Usage
### Querying via JSON Interface
```python
import requests

url = "http://localhost:1873/predict"
data = {
    "question": "Quels sont les principes de gestion des débris spatiaux ?"
}

response = requests.post(url, json=data)
print(response.json())
```

### Querying via cURL
```bash
curl -X POST http://localhost:1873/predict -H "Content-Type: application/json" -d '{
  "question": "Quels sont les principes de gestion des débris spatiaux ?"
}'
```

---
