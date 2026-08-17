from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


CMAPSS_COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_cmapss_train(path: str | Path) -> pd.DataFrame:
    """Load a NASA C-MAPSS train file and add uncapped/capped RUL labels."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
    return add_rul_targets(df)


def add_rul_targets(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["rul"] = max_cycle - df["cycle"]
    df["rul_capped"] = np.minimum(df["rul"], cap)
    return df


def group_train_valid_test_split(
    df: pd.DataFrame,
    valid_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by engine_id to avoid leaking future engine behavior across sets."""
    groups = df["engine_id"]
    first = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_valid_idx, test_idx = next(first.split(df, groups=groups))
    train_valid = df.iloc[train_valid_idx].copy()
    test = df.iloc[test_idx].copy()

    adjusted_valid = valid_size / (1.0 - test_size)
    second = GroupShuffleSplit(n_splits=1, test_size=adjusted_valid, random_state=random_state + 1)
    train_idx, valid_idx = next(second.split(train_valid, groups=train_valid["engine_id"]))
    return train_valid.iloc[train_idx].copy(), train_valid.iloc[valid_idx].copy(), test


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("sensor_")]


def feature_base_columns(df: pd.DataFrame) -> list[str]:
    cols = ["cycle"] + [c for c in df.columns if c.startswith("setting_") or c.startswith("sensor_")]
    return [c for c in cols if c in df.columns]

