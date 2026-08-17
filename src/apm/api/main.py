from __future__ import annotations

from pydantic import BaseModel
from fastapi import FastAPI

from apm.agent.agent import MaintenanceAgent
from apm.agent.tools import MaintenanceTools


app = FastAPI(title="Agentic Predictive Maintenance Copilot", version="0.1.0")


class ChatRequest(BaseModel):
    query: str


class PredictRequest(BaseModel):
    machine_id: int


def get_agent() -> MaintenanceAgent:
    return MaintenanceAgent(MaintenanceTools())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    response = get_agent().answer(request.query)
    return {
        "answer": response.answer,
        "tool_calls": response.tool_calls,
        "evidence": response.evidence,
    }


@app.post("/predict-rul")
def predict_rul(request: PredictRequest) -> dict:
    return MaintenanceTools().predict_rul(request.machine_id)


@app.get("/machines/{machine_id}/history")
def machine_history(machine_id: int) -> dict:
    return MaintenanceTools().get_machine_history(machine_id)


@app.get("/maintenance-priority")
def maintenance_priority(top_n: int = 5) -> list[dict]:
    return MaintenanceTools().rank_maintenance_priority(top_n=top_n)

