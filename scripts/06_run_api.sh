#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src SENSOR_DATA_PATH=data/processed/cmapss_FD001.csv \
  .venv/bin/python -m uvicorn apm.api.main:app \
  --reload \
  --host 127.0.0.1 \
  --port 8000

