# Building a chatbot from a website

## Preamble

In this tutorial, we will show how to create a collette application (chatbot) from the content of  the following [website](https://www.eoportal.org/satellite-missions?Mission+type=EO)
We will run the Colette server on a machine having multiple GPUs and the UI on a simple laptop.
On the server, choose a location where you can clone the Colette software. Store this location in an environment variable:

```
export COLETTE_DIR=your_preferred_location
```

## Install Colette on the server

To install Colette, connect to your server and execute the following commands:

```
cd $COLETTE_DIR
git clone https://git.jolibrain.com/jolibrain/colette_v2.git
cd colette_v2
pyenv virtualenv 3.10 rag_demo
pyenv local rag_demo
pip install -r requirements.txt
```

Next, create a directory to host all your Colette applications:

```
mkdir $COLETTE_DIR/colette_v2/colette_applications
```

## Starting the Server

To start the Colette server, you may specify which GPU to use if your machine has multiple GPUs. For instance, to utilize GPU with ID `3`, use the following command. If you are using a job manager like SLURM and have already allocated a GPU, you can omit `CUDA_VISIBLE_DEVICES`.

```bash
[CUDA_VISIBLE_DEVICES=3] server/run.sh --host=my_gpu_public_ip --port=8888 --reload-exclude='apps/*:colette_applications/*'
```

This command launches the [Uvicorn](https://www.uvicorn.org/) server on the server's public interface while excluding certain directories from the watchdog for automatic reloading.

Colette supports managing multiple applications simultaneously, each identified by a unique name. In this case, we'll name our application `rag_demo`.


## Extracting Satellite URLs

To scrape only the pages corresponding to satellites, use BeautifulSoup via the following command:

```bash
python apps/demo/demo_preprocess.py --url https://www.eoportal.org/satellite-missions?Mission+type=EO --app-dir $COLETTE_DIR/colette_v2/colette_applications/rag_demo/
```

This command will save an HTML file for each satellite in `$COLETTE_DIR/colette_v2/colette_applications/rag_demo/data`.

## Fetching and Indexing the Data

Next, create a Colette application based on the extracted URLs by crafting a JSON payload:

```json
{
    "app": {
        "repository": "$COLETTE_DIR/colette_v2/colette_applications/rag_demo",
        "verbose": "info"
    },
    "parameters": {
        "input": {
            "preprocessing": {
                "files": ["all"],
                "lib": "unstructured",
                "save_output": false,
                "filters": ["\\/~[^\\/]*$"]
            },
            "rag": {
                "indexdb_lib": "chromadb",
                "embedding_lib": "huggingface",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "gpu_id": 0,
                "search": true,
                "reindex": true
            },
            "template": {
                "template_prompt": "You are a useful assistant expert in space systems management. Question: {question} Contexte: {context} Réponse: ",
                "template_prompt_variables": ["context", "question"]
            },
            "data": ["$COLETTE_DIR/colette_v2/colette_applications/rag_demo/data"]
        },
        "llm": {
            "source": "llama3.1:latest",
            "inference": {
                "lib": "ollama"
            }
        }
    }
}
```

### Payload Breakdown

- **`app` block**: Defines the application repository path and sets the verbosity level (`info`).

- **`parameters` block**:
  - **`input` block**: 
    - **Preprocessing Parameters**: Specifies that all files in the `data` directory will be processed using the `unstructured` library, with file names filtered by the given regex.
    - **`rag` Parameters**: Uses [ChromaDB](https://www.trychroma.com/) for indexing. Embeddings are generated using the `sentence-transformers/all-MiniLM-L6-v2` model from HuggingFace. The `gpu_id` specifies which GPU to use; use `-1` to run on CPU. Set `reindex` to `false` to skip recreating the index, and `search` to `true` to enable searching.
    - **`template` block**: Defines the prompt template and the variables to be replaced for prompt generation.

- **`llm` block**: Specifies using the [Ollama](https://ollama.com/) library for LLM inference and indicates the model source.

To create the application, execute the following command:

```bash
curl -X POST -H 'Content-Type: application/json' http://my_gpu_public_ip:8888/v2/app/rag_demo -d @/path/to/json
```

This process will take a few minutes to index the 1000+ satellites, and you should see the following in your terminal:

```
2024-08-20 14:09:31,147 - rag_demo - INFO - init:
	RAG: True
	Preprocessing: unstructured
2024-08-20 14:09:31,156 - rag_demo - INFO - get_data: read 1084 files (inputconnector.py:66)
2024-08-20 14:09:31,156 - rag_demo - INFO - preprocessing started using 32 threads (langchaininputconn.py:283)
Processing documents: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1084/1084 [15:44<00:00,  1.15it/s]
2024-08-20 14:25:16,143 - rag_demo - INFO - preprocessing completed (langchaininputconn.py:299)
Indexing documents: 2it [04:10, 125.31s/it]
2024-08-20 14:30:38,870 - rag_demo - INFO - New index has completed (langchaininputconn.py:344)
2024-08-20 14:30:38,878 - rag_demo - INFO - embedding index has 45046 elements (langchaininputconn.py:345)
2024-08-20 14:30:44,200 - rag_demo - INFO - sucessfully persisted bm25 index /tmp/colette_v2/colette_applications/rag_demo/index/bm25.pkl (langchaininputconn.py:356)
INFO:     10.10.77.106:42796 - "PUT /v2/app/rag_demo HTTP/1.1" 200 OK
```

## Starting the chat application

On any other machine, you can now start the front end application like this:

```
pip install -r requirements.txt
streamlit run frontend/app.py
```

You should get a new tab in your browser that looks like the following image

![this](images/streamlit_demo.png "Chatbot demo")

Just change the name of the host in the upper left part of the side menu and start chatting.

## Limitations

This demo has been optimized for speed and not for performance. You might want to tweak some parameters, for example:
  - we use "sentence-transformers/all-MiniLM-L6-v2" as an embedding model. This one has a small memory footprint and is fast but its performance is probably slightly less than "hkunlp/instructor-large" for example. If you want more information on embedding models, visit the [sentence-transformers](https://sbert.net/docs/sentence_transformer/pretrained_models.html) website or its companion [Sentence Transformers Hugging Face organization](https://huggingface.co/models?library=sentence-transformers&author=sentence-transformers).