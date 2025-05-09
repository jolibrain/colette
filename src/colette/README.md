## Command Line Interface

### Indexing Documents

To index documents located in path/to/data/dir, execute the following command:

```bash

python path/to/cli.py index --app-dir path/to/index/dir --data-dir path/to/data/dir [--models-dir path/to/models] [--config-file path/to/config/file]
```
Where:

- `--app-dir` specifies the directory to store the index.
- `--data-dir` indicates the directory containing the documents to be indexed.
- `--models-dir` (optional) defines the cache directory for downloaded models.
- `--config-file` (optional) provides a configuration file; if omitted, indexing defaults to using a multimodal LLM.


### Interacting the Index

To interact with indexed documents, use the following command:

```bash
python path/to/cli.py index --app-dir path/to/index/dir --model-dirs /path/to/models --msg "some question"
```

where:
  - `--app-dir` specifies the directory containing the index.
  - `--models-dir` (optional) defines the cache directory for downloaded models.
  - `--msg` is your query directed at the indexed documents.

**Note:** Ensure that the paths provided are absolute or relative to the current working directory to avoid potential path resolution issues.

