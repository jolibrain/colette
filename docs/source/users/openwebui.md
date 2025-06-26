# Colette With OpenWebUI

## Launch Colette

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/jolibrain/colette.git
cd colette
pip install -e .[dev,trag]
```

It is strongly recommanded to create a virtual environnement before installing dependencies

---

### 2. Launch the server

```bash
bash server/run_openwebui.sh
```

---

### 3. Create one or several service(s) and index it/them :

You can do it either with a python script :

```bash
import json
import requests
from src.colette.jsonapi import JSONApi
from src.colette.apidata import APIData

colette_api = JSONApi()

documents_dir = 'docs/pdf' # where the input documents are located
app_dir = 'app_colette' # where to store the app
app_name = 'colette_doc' # the name of your application

url_base = "http://127.0.0.1:8889/v1" # The adress of your backend server 

config_file = 'src/colette/config/vrag_default.json'
index_file = 'src/colette/config/vrag_default_index.json'
with open(config_file, 'r') as f:
    create_config = json.load(f)
with open(index_file, 'r') as f:
    index_config = json.load(f)
create_config['app']['repository'] = app_dir
index_config['parameters']['input']['data'] = [documents_dir]

# Create the app
    print(f"Creating app {app_name}...")
    response = requests.post(url_base + "/app/" + app_name, json=create_config)
    if response.status_code == 200:
        print(f"App {app_name} created successfully.")
    else:
        print(f"Failed to create app {app_name}. Status code: {response.status_code}")
        print(f"Error message: {response.text}")

# Index the data
print(f"Indexing data for app {app_name}...")
response = requests.put(url_base + "/index/" + app_name, json=index_config)
if response.status_code == 200:
    print(f"Data indexing successfully launched for {app_name}.")
else:
    print(f"Failed to index data for app {app_name}. Status code: {response.status_code}")
    print(f"Error message: {response.text}")
```

You can also create your own json and create your service with a curl request :

```bash
curl -X POST -H 'Content-Type: application/json' http://localhost:8889/v1/app/your_app_name --data @your_json.json
```

Or directly with the web API on ```http://localhost:8889/docs```

Now your backend is ready, let's move to creating the user interface.

---

## OpenWebUI Instance

Install OpenWebUI with : 
```bash
pip install open-webui
```

Launch the server :
```bash
open-webui serve --host 0.0.0.0 --port 4321
```
**Warning, by default OpenWebUI download several models, if you don't want them do this command before launching server : ```export HF_HUB_OFFLINE=1```**

Now your UI is ready on http://localhost:4321

On your first login you'll need to create an account.

---

## Link OpenWebUI with your backend

In order to link your backend with your frontend, click on your name on the bottom left of the UI, then Admin Panel --> Parameters --> Connections 

Here you can manage OpenAI API connections, add a new one and specify the link to your backend (in our case it should be http://localhost:8889)

Now you can interact with the backend via OpenWebUI.