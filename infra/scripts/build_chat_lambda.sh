#!/usr/bin/env bash
# Build the chat API Lambda's folder: src/chat (handler and store) and src/tracing.py, as the package
# src, plus their dependencies for the Lambda runtime (Linux x86_64, Python 3.13). Terragrunt runs it
# before every plan, apply and destroy of chat-api.
set -euo pipefail
root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
out="$root/infra/.build/chat"
rm -rf "$out"
# The chat dependency group, pinned by uv.lock.
uv export --quiet --project "$root" --frozen --only-group chat --no-hashes --no-header --no-annotate |
  uv pip install --quiet --target "$out" -r - \
  --python-platform x86_64-manylinux2014 --python-version 3.13 --only-binary :all:
mkdir -p "$out/src/chat"
cp "$root/src/__init__.py" "$root/src/tracing.py" "$out/src/"
cp "$root/src/chat/__init__.py" "$root/src/chat/handler.py" "$root/src/chat/store.py" "$out/src/chat/"
