from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate import load_model, predict_frame
from data_preprocess import damage_state_from_midr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch or single-sample MIDR prediction.")
    parser.add_argument("--checkpoint", default="outputs/model.pt")
    parser.add_argument("--preprocessor", default="outputs/preprocessor.joblib")
    parser.add_argument("--input_csv", default="examples/demo_input.csv")
    parser.add_argument("--output_csv", default="outputs/predictions.csv")
    args = parser.parse_args()

    model, preprocessor, checkpoint = load_model(Path(args.checkpoint), Path(args.preprocessor))
    cfg = checkpoint["config"]
    frame = pd.read_csv(args.input_csv)
    midr = predict_frame(model, preprocessor, checkpoint, frame)
    result = frame.copy()
    result["pred_MIDR"] = midr
    result["pred_damage_state"] = [
        damage_state_from_midr(structure, value, cfg["damage_thresholds"])
        for structure, value in zip(frame[cfg["data"]["structure_col"]].astype(str), midr)
    ]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()

