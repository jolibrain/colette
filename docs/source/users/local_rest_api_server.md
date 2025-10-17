# Local REST API Server

Colette includes a local REST API server that allows you to interact with Colette programmatically over HTTP. This can be useful for integrating Colette into your own applications or workflows.

## Prerequisites

Make sure you have all the pre-requisites listed in the [Get Started](get_started.md) guide.
The docker part is optional if you plan to run the server directly using Python.

## Starting the Local REST API Server

### Clone the Repository
If you haven't already, clone the Colette repository from GitHub:

```bash
git clone https://github.com/jolibrain/colette.git
cd colette
```

### Create and Activate Virtual Environment

Create a virtual environment and activate it:

```bash
chmod +x create_venv_colette.sh
./create_venv_colette.sh
source venv_colette/bin/activate
```

NOTES: 
- This process may take a while, as there are many dependencies to install and some of them require compilation.
- There are scripts available for ARM machines as well. Check [Get Started ARM](get_started_ARM_machine.md) for more details.

### Index Your Data

There are multiple ways to index your data. You can use the CLI as shown in the [Get Started](get_started.md) guide, or you can use the REST API.

To index data using the colette CLI, run:

```bash
mkdir -p models
mkdir -p app_colette
```
This is to ensure that your user is the owner of these folders.

Then run:

```bash
colette_cli index --app-dir app_colette --data-dir docs/pdf/ --config-file src/colette/config/vrag_default.json
```

Make sure to replace `app_colette`, `docs/pdf/`, and the config file path with your actual application directory, data directory, and configuration file path.

### Start the REST API Server

To start the REST API server, run the following command, from the root of the Colette repository:

```bash
python server/run_local_rest_api_server.py
```

If you want to run in the background, you can use:

```bash
nohup python server/run_local_rest_api_server.py > api_server.log 2>&1 &
```

Don't forget to get the process ID (PID) if you want to stop it later:

```bash
echo $!
```

You can check the logs in `api_server.log`. Also you can change where the logs are stored by changing the output file.

### Create the Service

```bash
curl -X POST http://localhost:8000/service/create \
  -H "Content-Type: application/json" \
  -d '{
    "documents_dir": "docs/pdf",
    "app_dir": "app_colette",
    "models_dir": "models"
  }'
```

NOTE: Make sure to replace the paths with your actual directories.

### Query - OUTPUT: textual data

You can query the service using the following command:

```bash
curl -X POST http://localhost:8000/service/query \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the identified sources of errors ?"
  }' | jq -r '.output'
```

### Query - OUTPUT: persisted answer (text) and context (images)
```bash
curl -X POST http://localhost:8000/service/query \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the identified sources of errors ?",
    "return_images": true
  }' | tee >(jq -r '.output' > response.txt) | jq -r '.sources[] | select(.image_base64 and .key) | "\(.key)|\(.source)|\(.image_base64)"' | \
  while IFS='|' read -r image_name source_name image_data; do
    source_basename="${source_name##*/}"
    echo "$image_data" | sed 's/data:image\/[^;]*;base64,//' | base64 -d > "${source_basename}_${image_name}.png"
  done
```

Note that the images will be saved in the current directory with names based on their source and the query image name. The output text will be saved in `response.txt`.

### Other Endpoints

#### Index documents

```bash
curl -X POST http://localhost:8000/service/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents_dir": "docs/pdf",
    "app_dir": "app_colette",
    "models_dir": "models"
  }'
```

#### Index documents (with reindex flag)

```bash
curl -X POST http://localhost:8000/service/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents_dir": "docs/pdf",
    "app_dir": "app_colette",
    "models_dir": "models",
    "reindex": true
  }'    
```

#### Query the RAG (with base64 images in response)
```bash
curl -X POST http://localhost:8000/service/query \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the identified sources of errors ?",
    "return_images": true
  }'