#!/usr/bin/env bash
# Reproduce the whole study: data, MNL, local LLM, figures, tables.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

PY="${PYTHON:-python3}"
if [[ ! -x .venv/bin/python ]]; then
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install -q -U pip
.venv/bin/python -m pip install -q -r requirements.txt
exec .venv/bin/python -m src.run_all "$@"
