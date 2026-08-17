from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from apm.features.engineering import add_temporal_features, load_feature_metadata


class RULPredictor:
    def __init__(
        self,
        model_path: str | Path = "models/best_rul_model.joblib",
        metadata_path: str | Path = "models/feature_metadata.json",
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.model = joblib.load(self.model_path)
        self.metadata = load_feature_metadata(self.metadata_path)
        self.feature_cols = self.metadata["selected_features"]

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = add_temporal_features(frame)
        missing = [c for c in self.feature_cols if c not in df.columns]
        for col in missing:
            df[col] = 0.0
        pred = self.model.predict(df[self.feature_cols])
        out = df[["engine_id", "cycle"]].copy()
        out["predicted_rul"] = pred.clip(min=0)
        out["risk_level"] = out["predicted_rul"].map(classify_risk)
        return out


def classify_risk(rul: float) -> str:
    if rul <= 20:
        return "CRITICAL"
    if rul <= 40:
        return "HIGH"
    if rul <= 80:
        return "MEDIUM"
    return "LOW"

