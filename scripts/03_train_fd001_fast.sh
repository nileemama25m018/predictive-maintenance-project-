#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src SENSOR_DATA_PATH=data/processed/cmapss_FD001.csv \
  .venv/bin/python -m apm.models.train \
  --data data/processed/cmapss_FD001.csv \
  --target models \
  --windows 5,10 \
  --no-slopes
