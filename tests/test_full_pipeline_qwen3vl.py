#!/usr/bin/env python3
"""
Run a minimal end-to-end pipeline using the colette Python API.

Usage (from repo root):
  export PYTHONPATH=$PWD/src
  python -u tests/test_full_pipeline_qwen3vl.py

This is a conservative smoke test for the Qwen3 VL embedding pipeline.
It forces the index config to use `Qwen/Qwen3-VL-Embedding-2B` and will
print errors instead of raising, so it is safe for exploratory runs.
"""
import json
import os
import time
import re
import base64
import logging
import subprocess
import shutil
from io import BytesIO
from datetime import datetime
from PIL import Image

import pytest

try:
    import psutil
except Exception:
    psutil = None

from colette.jsonapi import JSONApi
from colette.apidata import APIData


def main():
    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    print("Colette repo root:", root)
    t0 = time.time()

    # setup debug logger
    log_path = os.path.join(root, 'tests', 'pipeline_debug.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = logging.getLogger('pipeline_debug')

    def log_system_state(stage: str):
        ts = datetime.utcnow().isoformat() + 'Z'
        info = [f"stage={stage}", f"ts={ts}"]
        # process memory
        try:
            if psutil:
                p = psutil.Process()
                mem = p.memory_info().rss
                info.append(f"proc_rss_bytes={mem}")
            else:
                with open(f"/proc/{os.getpid()}/status", 'r') as f:
                    for l in f:
                        if l.startswith('VmRSS:'):
                            info.append(l.strip())
                            break
        except Exception as e:
            info.append(f"proc_mem_err={e}")

        # system mem
        try:
            if os.path.exists('/proc/meminfo'):
                with open('/proc/meminfo', 'r') as f:
                    lines = f.read().splitlines()[:5]
                for l in lines:
                    info.append(l.strip())
        except Exception as e:
            info.append(f"meminfo_err={e}")

        # GPU info via nvidia-smi if available
        try:
            if shutil.which('nvidia-smi'):
                res = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                info.append('nvidia_smi:\n' + res.stdout.strip())
        except Exception as e:
            info.append(f"nvidia_smi_err={e}")

        # disk space
        try:
            st = os.statvfs(root)
            free_bytes = st.f_bavail * st.f_frsize
            info.append(f"root_fs_free_bytes={free_bytes}")
        except Exception as e:
            info.append(f"statvfs_err={e}")

        logger.info(' | '.join(info))

    logger.info('Starting pipeline smoke test')
    log_system_state('start')

    colette_api = JSONApi()

    documents_dir = os.path.join(root, 'docs', 'pdf')
    app_dir = os.path.join(root, 'app_colette')
    models_dir = os.path.join(root, 'models')
    app_name = 'ci_test_app_qwen3'

    cfg_create = os.path.join(root, 'src', 'colette', 'config', 'vrag_default.json')
    cfg_index = os.path.join(root, 'src', 'colette', 'config', 'vrag_default_index.json')

    with open(cfg_create, 'r') as f:
        create_config = json.load(f)
    with open(cfg_index, 'r') as f:
        index_config = json.load(f)

    create_config['app']['repository'] = app_dir
    create_config['app']['models_repository'] = models_dir

    try:
        index_config['parameters']['input']['rag']['embedding_model'] = 'Qwen/Qwen3-VL-Embedding-2B'
    except Exception:
        pass

    index_config['parameters']['input']['data'] = [documents_dir]

    logger.info('Creating service (this may download models)')
    api_data_create = APIData(**create_config)
    try:
        colette_api.service_create(app_name, api_data_create)
    except Exception as e:
        logger.exception('service_create failed')
        log_system_state('after_service_create_failed')
        return
    else:
        logger.info('service_create completed')
        log_system_state('after_service_create')

    logger.info('Indexing service (this may be long)')
    api_data_index = APIData(**index_config)
    try:
        colette_api.service_index(app_name, api_data_index)
    except Exception as e:
        logger.exception('service_index failed')
        log_system_state('after_service_index_failed')
        return
    else:
        logger.info('service_index completed')
        log_system_state('after_service_index')

    logger.info('Querying the vision RAG')
    query_api_msg = {
        'parameters': {
            'input': {
                'message': 'What are the identified sources of errors ?'
            }
        }
    }
    query_data = APIData(**query_api_msg)
    try:
        response = colette_api.service_predict(app_name, query_data)
    except Exception as e:
        logger.exception('service_predict failed')
        log_system_state('after_service_predict_failed')
        return
    else:
        logger.info('service_predict completed')
        log_system_state('after_service_predict')

    logger.info('\n--- Response output ---')
    try:
        logger.info(f"response.output={response.output}")
    except Exception:
        logger.info('(no textual output)')

    out_dir = os.path.join(root, 'tests', 'pipeline_output')
    os.makedirs(out_dir, exist_ok=True)
    logger.info('\nSaving image sources to %s', out_dir)
    try:
        for item in response.sources.get('context', []):
            key = item.get('key', 'no_key')
            base64_data = re.sub(r'^data:image/.+;base64,', '', item.get('content', ''))
            if not base64_data:
                continue
            image_data = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_data))
            fname = os.path.join(out_dir, f"{key}.png")
            image.save(fname)
            logger.info('Saved %s', fname)
    except Exception as e:
        logger.exception('Saving sources failed')
        log_system_state('after_saving_sources_failed')

    # final snapshot and elapsed
    log_system_state('end')
    logger.info('Elapsed %s', time.time() - t0)


if __name__ == '__main__':
    main()
