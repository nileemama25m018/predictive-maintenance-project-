from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from apm.data.cmapss import feature_base_columns, sensor_columns


def _rolling_slope(values: np.ndarray) -> float:
    if len(values) < 2 or np.all(np.isnan(values)):
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.nan_to_num(values.astype(float), nan=float(np.nanmean(values)))
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def add_temporal_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Create high-signal time-series features per engine without future leakage."""
    df = df.sort_values(["engine_id", "cycle"]).copy()
    sensors = sensor_columns(df)
    grouped = df.groupby("engine_id", group_keys=False)
    new_features: dict[str, pd.Series] = {}

    for col in sensors:
        new_features[f"{col}_delta_1"] = grouped[col].diff().fillna(0.0)
        for window in windows:
            roll = grouped[col].rolling(window=window, min_periods=1)
            new_features[f"{col}_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
            new_features[f"{col}_std_{window}"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
            new_features[f"{col}_min_{window}"] = roll.min().reset_index(level=0, drop=True)
            new_features[f"{col}_max_{window}"] = roll.max().reset_index(level=0, drop=True)
            new_features[f"{col}_slope_{window}"] = (
                grouped[col]
                .rolling(window=window, min_periods=2)
                .apply(_rolling_slope, raw=True)
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )
    if new_features:
        df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()


def select_features(
    train_df: pd.DataFrame,
    target_col: str = "rul_capped",
    variance_threshold: float = 1e-8,
    corr_threshold: float = 0.985,
) -> tuple[list[str], dict]:
    ignore = {"engine_id", "rul", "rul_capped", target_col}
    candidate_cols = [
        c for c in train_df.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    X = train_df[candidate_cols]

    vt = VarianceThreshold(threshold=variance_threshold)
    vt.fit(X)
    kept = list(X.columns[vt.get_support()])

    corr = train_df[kept].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = {column for column in upper.columns if any(upper[column] > corr_threshold)}
    selected = [c for c in kept if c not in to_drop]

    metadata = {
        "target_col": target_col,
        "selected_features": selected,
        "dropped_low_variance": [c for c in candidate_cols if c not in kept],
        "dropped_high_correlation": sorted(to_drop),
        "base_features": feature_base_columns(train_df),
        "sensor_features": sensor_columns(train_df),
    }
    return selected, metadata


def save_feature_metadata(metadata: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_feature_metadata(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
