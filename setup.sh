#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${FLOW_PYTHON:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null; then PYTHON_BIN=python3; fi
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,12), "Python 3.12 or newer is required"'
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -c "import duckdb; duckdb.connect().execute('INSTALL spatial')"
npm --prefix frontend ci
npm --prefix frontend run build
printf '\nReady. Run ./start.sh, then open http://127.0.0.1:8000\n'
