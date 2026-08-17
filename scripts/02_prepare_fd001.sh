#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src .venv/bin/python scripts/prepare_cmapss.py \
  --input data/raw/train_FD001.txt \
  --output data/processed/cmapss_FD001.csv

