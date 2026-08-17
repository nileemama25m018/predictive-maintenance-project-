from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from apm.data.cmapss import add_rul_targets, group_train_valid_test_split, load_cmapss_train
from apm.features.engineering import add_temporal_features, save_feature_metadata, select_features


def _optional_models(random_state: int) -> dict:
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }
    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=600,
            learning_rate=0.035,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor

        models["catboost"] = CatBoostRegressor(
            iterations=900,
            learning_rate=0.035,
            depth=7,
            loss_function="RMSE",
            verbose=False,
            random_seed=random_state,
        )
    except Exception:
        pass
    return models


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def near_failure_metrics(df: pd.DataFrame, y_pred, threshold: int = 30) -> dict:
    mask = df["rul"] <= threshold
    if not mask.any():
        return {}
    return {f"near_failure_{k}": v for k, v in regression_metrics(df.loc[mask, "rul_capped"], np.asarray(y_pred)[mask.to_numpy()]).items()}


def load_training_frame(args) -> pd.DataFrame:
    if args.cmapss:
        return load_cmapss_train(args.cmapss)
    df = pd.read_csv(args.data)
    if "rul_capped" not in df.columns:
        df = add_rul_targets(df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/demo_sensor_data.csv")
    parser.add_argument("--cmapss", default=None)
    parser.add_argument("--target", default="models")
    parser.add_argument("--use-lstm", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    target_dir = Path(args.target)
    target_dir.mkdir(parents=True, exist_ok=True)

    raw = load_training_frame(args)
    engineered = add_temporal_features(raw)
    train_df, valid_df, test_df = group_train_valid_test_split(engineered, random_state=args.random_state)
    feature_cols, metadata = select_features(train_df)

    X_train, y_train = train_df[feature_cols], train_df["rul_capped"]
    X_valid, y_valid = valid_df[feature_cols], valid_df["rul_capped"]
    X_test, y_test = test_df[feature_cols], test_df["rul_capped"]

    results = []
    best_name = None
    best_model = None
    best_valid_mae = float("inf")

    for name, model in _optional_models(args.random_state).items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(with_centering=False)),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        valid_pred = pipe.predict(X_valid)
        test_pred = pipe.predict(X_test)
        valid_metrics = regression_metrics(y_valid, valid_pred)
        test_metrics = regression_metrics(y_test, test_pred) | near_failure_metrics(test_df, test_pred)
        row = {"model": name, "valid": valid_metrics, "test": test_metrics}
        results.append(row)
        if valid_metrics["mae"] < best_valid_mae:
            best_valid_mae = valid_metrics["mae"]
            best_name = name
            best_model = pipe

    if args.use_lstm:
        try:
            from apm.models.lstm import make_sequences, train_lstm_regressor
            import torch

            seq_features = metadata["base_features"]
            train_seq = make_sequences(train_df, seq_features)
            valid_seq = make_sequences(valid_df, seq_features)
            test_seq = make_sequences(test_df, seq_features)
            if len(train_seq.y) and len(valid_seq.y):
                lstm_model, history = train_lstm_regressor(train_seq, valid_seq)
                with torch.no_grad():
                    valid_pred = lstm_model(torch.tensor(valid_seq.X)).numpy()
                    test_pred = lstm_model(torch.tensor(test_seq.X)).numpy()
                row = {
                    "model": "lstm",
                    "valid": regression_metrics(valid_seq.y, valid_pred),
                    "test": regression_metrics(test_seq.y, test_pred),
                    "history": history,
                    "sequence_features": seq_features,
                }
                results.append(row)
                torch.save(
                    {
                        "state_dict": lstm_model.state_dict(),
                        "sequence_features": seq_features,
                        "window_size": 30,
                    },
                    target_dir / "lstm_rul_model.pt",
                )
        except Exception as exc:
            results.append({"model": "lstm", "error": str(exc)})

    if best_model is None:
        raise RuntimeError("No model trained. Install scikit-learn dependencies and retry.")

    metadata["best_model"] = best_name
    metadata["engineered_feature_count"] = len(feature_cols)
    joblib.dump(best_model, target_dir / "best_rul_model.joblib")
    save_feature_metadata(metadata, target_dir / "feature_metadata.json")
    (target_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps({"best_model": best_name, "results": results}, indent=2))


if __name__ == "__main__":
    main()
