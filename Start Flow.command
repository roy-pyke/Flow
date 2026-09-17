#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python || ! -f frontend/dist/index.html ]]; then
  printf 'Run ./setup.sh once while online before launching Flow.\n'
  exit 1
fi
(
  for attempt in {1..50}; do
    if curl --silent --fail "http://127.0.0.1:${FLOW_PORT:-8000}/api/health" >/dev/null; then
      open "http://127.0.0.1:${FLOW_PORT:-8000}"
      exit 0
    fi
    sleep 0.2
  done
) &
./start.sh
