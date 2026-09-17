#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python || ! -f frontend/dist/index.html ]]; then
  printf 'Run ./setup.sh once while online, then start again.\n' >&2
  exit 1
fi
if [[ ! -f data/demo/network.json ]]; then
  printf 'Missing bundled dataset. Restore data/demo from the repository.\n' >&2
  exit 1
fi
exec .venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port "${FLOW_PORT:-8000}"
