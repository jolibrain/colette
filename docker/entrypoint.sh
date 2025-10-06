#!/usr/bin/env bash
set -e

# Set HOME to a writable directory
export HOME=/tmp

# Configure git to trust /app directory
git config --global --add safe.directory /app

# always log in
huggingface-cli login --token "$HF_TOKEN"

# then exec the user-specified command (or the default from CMD)
exec "$@"
