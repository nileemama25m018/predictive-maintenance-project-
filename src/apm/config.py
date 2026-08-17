from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///maintenance.db")
    llm_provider: str = os.getenv("LLM_PROVIDER", "local")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    sensor_data_path: Path = Path(os.getenv("SENSOR_DATA_PATH", "data/processed/cmapss_FD001.csv"))
    rul_model_path: Path = Path(os.getenv("RUL_MODEL_PATH", "models/best_rul_model.joblib"))
    feature_metadata_path: Path = Path(os.getenv("FEATURE_METADATA_PATH", "models/feature_metadata.json"))
    rag_index_dir: Path = Path(os.getenv("RAG_INDEX_DIR", "rag_index"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
