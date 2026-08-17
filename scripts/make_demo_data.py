from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from apm.data.cmapss import add_rul_targets


def main() -> None:
    rng = np.random.default_rng(42)
    rows = []
    for engine_id in range(1, 31):
        max_cycle = int(rng.integers(90, 190))
        degradation_rate = rng.uniform(0.004, 0.012)
        for cycle in range(1, max_cycle + 1):
            progress = cycle / max_cycle
            row = {
                "engine_id": engine_id,
                "cycle": cycle,
                "setting_1": rng.normal(0, 0.2),
                "setting_2": rng.normal(0, 0.2),
                "setting_3": rng.normal(0, 0.2),
            }
            for s in range(1, 22):
                base = rng.normal(0, 0.03)
                if s in {2, 3, 4, 7, 11, 12, 15, 20, 21}:
                    signal = progress * degradation_rate * s * 8
                elif s in {8, 13, 16}:
                    signal = -progress * degradation_rate * s * 5
                else:
                    signal = rng.normal(0, 0.01)
                row[f"sensor_{s}"] = base + signal + rng.normal(0, 0.02)
            rows.append(row)
    df = add_rul_targets(pd.DataFrame(rows))
    out = Path("data/processed/demo_sensor_data.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df):,} rows")


if __name__ == "__main__":
    main()

