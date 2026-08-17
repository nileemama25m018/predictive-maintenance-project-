# Agentic Predictive Maintenance Copilot

An end-to-end **ML + RAG + AI agent** project for predictive maintenance. It predicts Remaining Useful Life (RUL) from machine sensor time series, retrieves maintenance knowledge, checks structured history, and produces evidence-grounded maintenance recommendations.

## Why This Is Resume-Strong

This is not a PDF chatbot. The system combines:

- RUL prediction using classical ML and optional LSTM sequence modeling
- Strong time-series feature engineering: rolling statistics, deltas, slopes, sensor stability filtering, correlation pruning
- RAG over maintenance manuals and failure-mode documents
- Agent tools for prediction, machine history, document search, and maintenance prioritization
- FastAPI backend, Streamlit dashboard, tests, Docker-ready structure

## Architecture

```text
Sensor data -> Feature pipeline -> ML/LSTM RUL model -> Risk engine
                                  -> Agent tools
Manuals/docs -> RAG index --------/
Machine DB ----------------------/

User -> Streamlit/FastAPI -> Agent -> Evidence-grounded recommendation
```

## Quick Start

```bash
cd agentic-predictive-maintenance
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python scripts/make_demo_data.py
python -m apm.models.train --data data/processed/demo_sensor_data.csv --target models
python -m apm.rag.indexer --docs knowledge_base --index rag_index
python -m apm.database.db --seed-demo
uvicorn apm.api.main:app --reload
```

In another terminal:

```bash
streamlit run app/streamlit_app.py
```

## Run In VS Code

Open this folder directly in VS Code:

```text
agentic-predictive-maintenance
```

Then run these VS Code tasks in order:

```text
Terminal -> Run Task -> Install dependencies
Terminal -> Run Task -> Prepare FD001 data
Terminal -> Run Task -> Train FD001 model fast
Terminal -> Run Task -> Build RAG index
Terminal -> Run Task -> Seed database
```

After that, use Run and Debug:

```text
FastAPI API
Streamlit Dashboard
```

Open the dashboard at:

```text
http://127.0.0.1:8501
```

If training seems stuck, first run `Train model fast` without LSTM. LSTM training is slower and needs PyTorch to be installed correctly.

## Using NASA C-MAPSS

Download C-MAPSS and place files such as `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in `data/raw/`. Then train:

```bash
python -m apm.models.train --cmapss data/raw/train_FD001.txt --target models --use-lstm
```

## Example Questions

- Which machines should be inspected first?
- Why is Machine 7 high risk?
- What is the predicted RUL of Machine 3?
- What procedure should the technician follow?
- What happens if we operate this machine for 20 more cycles?

## Resume Bullet

Built an agentic predictive-maintenance decision-support system combining RUL prediction, anomaly scoring, SQL-based machine history, and RAG over maintenance manuals; implemented an agent that selects tools and produces evidence-grounded risk explanations and maintenance actions.
