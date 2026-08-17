#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src SENSOR_DATA_PATH=data/processed/cmapss_FD001.csv \
  .venv/bin/python -m streamlit run app/streamlit_app.py \
  --server.address 127.0.0.1 \
  --server.port 8501

