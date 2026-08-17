#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src .venv/bin/python -m apm.rag.indexer \
  --docs knowledge_base \
  --index rag_index

