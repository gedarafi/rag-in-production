#!/usr/bin/env bash
# Build the chat Lambda's folder: src/chat/handler.py plus its dependencies for the Lambda runtime
# (Linux x86_64, Python 3.13). Terragrunt runs it before every plan, apply and destroy of chat-api.
set -euo pipefail
root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
out="$root/infra/.build/chat"
rm -rf "$out"
# The chat dependency group, pinned by uv.lock.
uv export --quiet --project "$root" --frozen --only-group chat --no-hashes --no-header --no-annotate |
  uv pip install --quiet --target "$out" -r - \
  --python-platform x86_64-manylinux2014 --python-version 3.13 --only-binary :all:
cp "$root/src/chat/handler.py" "$out/"
