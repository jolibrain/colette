#!/usr/bin/env bash
set -e

# always log in
huggingface-cli login --token "$HF_TOKEN"

# then exec the user-specified command (or the default from CMD)
exec "$@"
