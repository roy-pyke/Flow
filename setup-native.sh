#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  printf 'Run ./setup.sh first to create the Python environment.\n' >&2
  exit 1
fi
.venv/bin/python -m pip install ./cpp
FLOW_REQUIRE_NATIVE=1 .venv/bin/python -m pytest -q tests/test_native_parity.py
printf '\nNative extension built and parity checked. Restart Flow to load the new binary.\n'
