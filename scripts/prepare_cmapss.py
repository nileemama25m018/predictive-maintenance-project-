from __future__ import annotations

import argparse
from pathlib import Path

from apm.data.cmapss import load_cmapss_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NASA C-MAPSS train file to processed CSV with RUL labels.")
    parser.add_argument("--input", default="data/raw/train_FD001.txt", help="Path to train_FD001.txt")
    parser.add_argument("--output", default="data/processed/cmapss_FD001.csv", help="Output processed CSV path")
    args = parser.parse_args()

    df = load_cmapss_train(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {output} with {len(df):,} rows and {df['engine_id'].nunique()} engines")


if __name__ == "__main__":
    main()

