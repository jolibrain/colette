This is an example coder LLM (qwen2.5-coder or deepseek-coder-v2) via ollama, served by Colette.

# Setup

- Setup Ollama, see https://ollama.com/download

- Get coder model

```
ollama pull qwen2.5-coder
```

# Quickstart

- Start a colette server:
```
cd colette_v2
./server/run.sh
```

- Create the service:
```
cd apps/coder
./create_coder.sh
```

- Test the service
```
cd apps/coder
./send_msg.sh
```
