# Pipeline Utility Scripts

This folder contains standalone diagnostics for manual end-to-end pipeline checks.

These scripts are intentionally **not** collected by `pytest`.

## Available scripts

- `full_pipeline_qwen3vl.py`
- `full_pipeline_emb_gme_qwen2vl.py`

## Run from repository root

```bash
export PYTHONPATH=$PWD/src
python -u scripts/pipeline/full_pipeline_qwen3vl.py
python -u scripts/pipeline/full_pipeline_emb_gme_qwen2vl.py
```

## Outputs

- Logs: `scripts/pipeline/logs/`
- Extracted output images: `scripts/pipeline/output/`
