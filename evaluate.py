from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from data_preprocess import add_physics_features, damage_state_from_midr
from models.factory import build_model


def inverse_target(values: np.ndarray, mode: str, mean: float, std: float) -> np.ndarray:
    transformed = values * std + mean
    if mode == "log1p_1000":
        return np.clip(np.expm1(np.clip(transformed, -20, 20)) / 1000.0, 0.0, None)
    return np.clip(transformed, 0.0, None)


def load_model(checkpoint_path: Path, preprocessor_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    preprocessor = joblib.load(preprocessor_path)
    cfg = checkpoint["config"]
    model = build_model(
        cfg["model"]["name"],
        num_continuous=len(checkpoint["continuous_cols"]),
        category_cardinalities=preprocessor.category_cardinalities,
        model_config=cfg["model"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, preprocessor, checkpoint


def predict_frame(model, preprocessor, checkpoint: dict, frame: pd.DataFrame) -> np.ndarray:
    frame = add_physics_features(frame, checkpoint["config"]["physics"].get("alpha_tg", 6.283185307))
    continuous, categorical = preprocessor.transform(frame)
    with torch.no_grad():
        outputs = model(torch.tensor(continuous, dtype=torch.float32), torch.tensor(categorical, dtype=torch.long))
    return inverse_target(
        outputs["midr"].cpu().numpy(),
        checkpoint["target_transform"],
        checkpoint["target_mean"],
        checkpoint["target_std"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PI-RKAN-FTT model.")
    parser.add_argument("--checkpoint", default="outputs/model.pt")
    parser.add_argument("--preprocessor", default="outputs/preprocessor.joblib")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_csv", default="outputs/test_predictions.csv")
    args = parser.parse_args()

    model, preprocessor, checkpoint = load_model(Path(args.checkpoint), Path(args.preprocessor))
    cfg = checkpoint["config"]
    frame = pd.read_csv(args.test_csv)
    prediction = predict_frame(model, preprocessor, checkpoint, frame)
    target_col = cfg["data"]["target_col"]
    structure_col = cfg["data"]["structure_col"]
    y_true = frame[target_col].to_numpy(dtype=float)
    true_state = [
        damage_state_from_midr(structure, value, cfg["damage_thresholds"])
        for structure, value in zip(frame[structure_col].astype(str), y_true)
    ]
    pred_state = [
        damage_state_from_midr(structure, value, cfg["damage_thresholds"])
        for structure, value in zip(frame[structure_col].astype(str), prediction)
    ]
    metrics = {
        "R2": r2_score(y_true, prediction),
        "RMSE": mean_squared_error(y_true, prediction) ** 0.5,
        "MAE": mean_absolute_error(y_true, prediction),
        "Accuracy": accuracy_score(true_state, pred_state),
        "Macro-F1": f1_score(true_state, pred_state, labels=["DS1", "DS2", "DS3", "DS4", "DS5"], average="macro", zero_division=0),
    }
    output = frame.copy()
    output["pred_MIDR"] = prediction
    output["true_damage_state"] = true_state
    output["pred_damage_state"] = pred_state
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()

