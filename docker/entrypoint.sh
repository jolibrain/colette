#!/usr/bin/env bash
set -e

# Set HOME to a writable directory
export HOME=/tmp

# Configure git to trust /app directory
git config --global --add safe.directory /app

# Log in to HuggingFace if a token is provided.
# Strip whitespace to handle copy-paste artifacts (e.g. trailing newlines from Jenkins secrets).
if [ -n "${HF_TOKEN:-}" ]; then
    HF_TOKEN="$(printf '%s' "${HF_TOKEN}" | tr -d '\n\r\t ')"
    export HF_TOKEN
    hf auth login --token "${HF_TOKEN}"
fi

# then exec the user-specified command (or the default from CMD)
exec "$@"
