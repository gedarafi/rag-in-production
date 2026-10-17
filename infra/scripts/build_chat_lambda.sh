#!/usr/bin/env bash
# Build the chat Lambda's folder: src/chat/handler.py plus its dependencies for the Lambda runtime
# (Linux x86_64, Python 3.13). Terragrunt runs it before every plan, apply and destroy of chat-api.
set -euo pipefail
root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
out="$root/infra/.build/chat"
rm -rf "$out"
uv pip install --quiet --target "$out" -r "$root/src/chat/requirements.txt" \
  --python-platform x86_64-manylinux2014 --python-version 3.13 --only-binary :all:
cp "$root/src/chat/handler.py" "$out/"
