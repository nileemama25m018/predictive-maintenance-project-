from apm.agent.agent import MaintenanceAgent


class DummyTools:
    def rank_maintenance_priority(self, top_n=5):
        return [{"machine_id": 1, "cycle": 100, "predicted_rul": 12.0, "risk_level": "CRITICAL"}]

    def search_maintenance_docs(self, query, k=4):
        return [{"source": "manual.md", "score": 0.9, "text": "Inspect bearing before continued operation."}]

    def predict_rul(self, machine_id):
        return {"machine_id": machine_id, "cycle": 100, "predicted_rul": 12.0, "risk_level": "CRITICAL"}

    def get_machine_history(self, machine_id):
        return {"latest_sensor": {"sensor_summary": 0.7}, "maintenance_history": [], "failures": []}


def test_agent_machine_answer():
    response = MaintenanceAgent(DummyTools()).answer("Why is Machine 1 high risk?")
    assert "Machine 1" in response.answer
    assert "predict_rul" in response.tool_calls

