from __future__ import annotations

from pathlib import Path

import pandas as pd

from apm.config import get_settings
from apm.database.db import latest_machine_snapshot
from apm.models.predictor import RULPredictor, classify_risk
from apm.rag.indexer import RagIndex


class MaintenanceTools:
    def __init__(
        self,
        database_url: str | None = None,
        sensor_data_path: str | None = None,
        rag_index_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ):
        settings = get_settings()
        self.database_url = database_url or settings.database_url
        self.sensor_data_path = Path(sensor_data_path or settings.sensor_data_path)
        self.rag = RagIndex(rag_index_dir or settings.rag_index_dir)
        self.predictor = None
        model_path = model_path or settings.rul_model_path
        metadata_path = metadata_path or settings.feature_metadata_path
        if Path(model_path).exists() and Path(metadata_path).exists():
            self.predictor = RULPredictor(model_path, metadata_path)

    def get_machine_history(self, machine_id: int) -> dict:
        return latest_machine_snapshot(self.database_url, machine_id)

    def predict_rul(self, machine_id: int) -> dict:
        if self.predictor is None:
            return {"machine_id": machine_id, "error": "RUL model is not trained yet."}
        if not self.sensor_data_path.exists():
            return {"machine_id": machine_id, "error": f"Missing sensor data: {self.sensor_data_path}"}
        df = pd.read_csv(self.sensor_data_path)
        machine_df = df[df["engine_id"] == machine_id].copy()
        if machine_df.empty:
            return {"machine_id": machine_id, "error": "Machine not found in sensor data."}
        pred = self.predictor.predict_frame(machine_df).sort_values("cycle").tail(1).iloc[0]
        return {
            "machine_id": machine_id,
            "cycle": int(pred["cycle"]),
            "predicted_rul": round(float(pred["predicted_rul"]), 2),
            "risk_level": str(pred["risk_level"]),
        }

    def rank_maintenance_priority(self, top_n: int = 5) -> list[dict]:
        if self.predictor is None or not self.sensor_data_path.exists():
            return []
        df = pd.read_csv(self.sensor_data_path)
        latest_rows = []
        for machine_id, group in df.groupby("engine_id"):
            prediction = self.predictor.predict_frame(group).sort_values("cycle").tail(1).iloc[0]
            latest_rows.append(
                {
                    "machine_id": int(machine_id),
                    "cycle": int(prediction["cycle"]),
                    "predicted_rul": round(float(prediction["predicted_rul"]), 2),
                    "risk_level": classify_risk(float(prediction["predicted_rul"])),
                }
            )
        return sorted(latest_rows, key=lambda row: row["predicted_rul"])[:top_n]

    def search_maintenance_docs(self, query: str, k: int = 4) -> list[dict]:
        try:
            return self.rag.search(query, k=k)
        except Exception as exc:
            return [{"error": str(exc), "source": None, "text": "RAG index not built yet."}]
