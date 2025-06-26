#!/bin/bash

uvicorn --app-dir=src colette.openwebuiapi:app --port 8889 $*
