#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src .venv/bin/python -m apm.database.db --seed-demo

