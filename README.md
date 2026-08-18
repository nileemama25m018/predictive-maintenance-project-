# Agentic Predictive Maintenance Copilot

An end-to-end **ML + RAG + AI agent** project for predictive maintenance. It predicts Remaining Useful Life (RUL) from machine sensor time series, retrieves maintenance knowledge, checks structured history, and produces evidence-grounded maintenance recommendations.


## Architecture

```text
Sensor data -> Feature pipeline -> ML/LSTM RUL model -> Risk engine
                                  -> Agent tools
Manuals/docs -> RAG index --------/
Machine DB ----------------------/

User -> Streamlit/FastAPI -> Agent -> Evidence-grounded recommendation
```


## Using NASA C-MAPSS

Download C-MAPSS and place files such as `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in `data/raw/`. Then train:


## Example Questions

- Which machines should be inspected first?
- Why is Machine 7 high risk?
- What is the predicted RUL of Machine 3?
- What procedure should the technician follow?
- What happens if we operate this machine for 20 more cycles?

