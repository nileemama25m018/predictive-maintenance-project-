from __future__ import annotations

import re
from dataclasses import dataclass

from apm.agent.tools import MaintenanceTools


@dataclass
class AgentAnswer:
    answer: str
    tool_calls: list[str]
    evidence: dict


class MaintenanceAgent:
    """A deterministic, auditable agent loop with tool selection.

    You can later swap the planner/summarizer with LangGraph + an LLM, while keeping
    these same safe tools and evidence contracts.
    """

    def __init__(self, tools: MaintenanceTools | None = None):
        self.tools = tools or MaintenanceTools()

    def answer(self, query: str) -> AgentAnswer:
        q = query.lower()
        machine_id = self._extract_machine_id(query)
        tool_calls: list[str] = []
        evidence: dict = {}

        if "which" in q and ("first" in q or "priority" in q or "inspect" in q):
            tool_calls.append("rank_maintenance_priority")
            evidence["priority"] = self.tools.rank_maintenance_priority(top_n=5)
            docs_query = "maintenance inspection priority low remaining useful life anomaly sensor degradation"
            tool_calls.append("search_maintenance_docs")
            evidence["documents"] = self.tools.search_maintenance_docs(docs_query)
            return AgentAnswer(self._priority_answer(evidence), tool_calls, evidence)

        if machine_id is None:
            tool_calls.append("search_maintenance_docs")
            evidence["documents"] = self.tools.search_maintenance_docs(query)
            return AgentAnswer(self._doc_answer(query, evidence), tool_calls, evidence)

        if any(word in q for word in ["rul", "risk", "why", "service", "maintain", "inspect", "what should"]):
            tool_calls.extend(["predict_rul", "get_machine_history", "search_maintenance_docs"])
            evidence["prediction"] = self.tools.predict_rul(machine_id)
            evidence["history"] = self.tools.get_machine_history(machine_id)
            evidence["documents"] = self.tools.search_maintenance_docs(
                f"machine maintenance procedure {query} risk degradation inspection"
            )
            return AgentAnswer(self._machine_answer(machine_id, evidence), tool_calls, evidence)

        tool_calls.append("get_machine_history")
        evidence["history"] = self.tools.get_machine_history(machine_id)
        return AgentAnswer(self._history_answer(machine_id, evidence), tool_calls, evidence)

    @staticmethod
    def _extract_machine_id(text: str) -> int | None:
        match = re.search(r"(?:machine|engine)\s*#?\s*(\d+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"\bM(?:achine)?(\d+)\b", text, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    @staticmethod
    def _machine_answer(machine_id: int, evidence: dict) -> str:
        prediction = evidence.get("prediction", {})
        history = evidence.get("history", {})
        docs = evidence.get("documents", [])
        if prediction.get("error"):
            return f"I could not produce a RUL prediction for Machine {machine_id}: {prediction['error']}"

        latest = history.get("latest_sensor") or {}
        failures = history.get("failures") or []
        doc_lines = [
            f"- {d.get('source', 'unknown')} (score={d.get('score', 0):.3f})"
            for d in docs
            if not d.get("error")
        ]
        failure_note = "A previous related failure exists." if failures else "No previous failure is recorded in the demo database."
        return (
            f"Machine {machine_id} is currently classified as {prediction.get('risk_level')} risk. "
            f"The latest predicted RUL is {prediction.get('predicted_rul')} cycles at cycle {prediction.get('cycle')}. "
            f"The latest sensor summary is {latest.get('sensor_summary', 'not available')}. {failure_note}\n\n"
            "Recommended action: prioritize inspection if the risk is HIGH or CRITICAL, verify the sensors driving degradation, "
            "and follow the retrieved maintenance procedure before continued operation.\n\n"
            "Evidence used:\n"
            f"- RUL model prediction: {prediction}\n"
            f"- Maintenance/failure history records: {len(history.get('maintenance_history', []))} maintenance events, {len(failures)} failures\n"
            + ("\n".join(doc_lines) if doc_lines else "- No document evidence available; build the RAG index.")
        )

    @staticmethod
    def _priority_answer(evidence: dict) -> str:
        priority = evidence.get("priority", [])
        if not priority:
            return "I cannot rank maintenance priority until the RUL model and sensor data are available."
        lines = [
            f"{i}. Machine {row['machine_id']} - RUL {row['predicted_rul']} cycles - {row['risk_level']}"
            for i, row in enumerate(priority, start=1)
        ]
        return "Recommended inspection priority:\n" + "\n".join(lines)

    @staticmethod
    def _doc_answer(query: str, evidence: dict) -> str:
        docs = evidence.get("documents", [])
        if not docs or docs[0].get("error"):
            return "I could not search the maintenance knowledge base yet. Build the RAG index first."
        return "Relevant maintenance guidance:\n" + "\n\n".join(
            f"Source: {d['source']}\n{d['text'][:500]}" for d in docs[:3]
        )

    @staticmethod
    def _history_answer(machine_id: int, evidence: dict) -> str:
        history = evidence.get("history", {})
        records = history.get("maintenance_history", [])
        if not records:
            return f"No maintenance history found for Machine {machine_id}."
        last = records[0]
        return (
            f"Machine {machine_id} was last serviced on {last['maintenance_date']} for "
            f"{last['component']} ({last['maintenance_type']}). Notes: {last['description']}"
        )

