from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_preprocess import split_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/validation/test CSV splits.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--stratify_cols", nargs="+", default=["structure_type"])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    train, val, test = split_dataset(frame, args.stratify_cols, seed=args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    print(f"Saved splits to {output_dir}")


if __name__ == "__main__":
    main()

