import pandas as pd

from apm.data.cmapss import add_rul_targets
from apm.features.engineering import add_temporal_features, select_features


def test_temporal_features_and_selection():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 1, 2, 3],
            "setting_1": [0, 0, 0, 1, 1, 1],
            "setting_2": [0, 0, 0, 1, 1, 1],
            "setting_3": [0, 0, 0, 1, 1, 1],
            "sensor_1": [1, 2, 3, 2, 3, 4],
            "sensor_2": [5, 5, 5, 6, 6, 6],
        }
    )
    df = add_rul_targets(df)
    out = add_temporal_features(df, windows=(2,))
    assert "sensor_1_mean_2" in out.columns
    features, metadata = select_features(out)
    assert features
    assert metadata["selected_features"] == features

