#!/bin/bash

uvicorn --app-dir=src colette.httpjsonapi:app --port 1873 $*
