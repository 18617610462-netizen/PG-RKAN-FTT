from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_preprocess import DamageDataset, TabularPreprocessor, add_physics_features, load_config
from losses import physics_informed_loss
from models.factory import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def transform_target(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log1p_1000":
        return np.log1p(np.clip(values, 0.0, None) * 1000.0)
    return np.asarray(values, dtype=float)


def threshold_tensor(thresholds: dict[str, list[float]], structure_order: list[str], mode: str, mean: float, std: float) -> torch.Tensor:
    raw = np.asarray([thresholds[name] for name in structure_order], dtype=float)
    transformed = (transform_target(raw, mode) - mean) / std
    return torch.tensor(transformed, dtype=torch.float32)


def make_dataset(frame: pd.DataFrame, preprocessor: TabularPreprocessor, target_col: str, mean: float, std: float, mode: str) -> DamageDataset:
    continuous, categorical = preprocessor.transform(frame)
    target = ((transform_target(frame[target_col].to_numpy(dtype=float), mode) - mean) / std).astype(np.float32)
    resonance = frame.get("resonance_factor", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=np.float32)
    return DamageDataset(continuous, categorical, target, resonance)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FT-Transformer/RKAN/PI-RKAN-FTT models.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 2026)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    train_df = pd.read_csv(data_cfg["train_csv"])
    val_df = pd.read_csv(data_cfg["val_csv"])
    train_df = add_physics_features(train_df, cfg["physics"].get("alpha_tg", 6.283185307))
    val_df = add_physics_features(val_df, cfg["physics"].get("alpha_tg", 6.283185307))

    continuous_cols = data_cfg["continuous_cols"] + ["Tg", "resonance_factor"]
    preprocessor = TabularPreprocessor(continuous_cols, data_cfg["categorical_cols"]).fit(train_df)
    target_mode = cfg["training"].get("target_transform", "log1p_1000")
    transformed = transform_target(train_df[data_cfg["target_col"]].to_numpy(dtype=float), target_mode)
    target_mean = float(transformed.mean())
    target_std = float(transformed.std() or 1.0)

    train_ds = make_dataset(train_df, preprocessor, data_cfg["target_col"], target_mean, target_std, target_mode)
    val_ds = make_dataset(val_df, preprocessor, data_cfg["target_col"], target_mean, target_std, target_mode)
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)

    model = build_model(
        cfg["model"]["name"],
        num_continuous=len(continuous_cols),
        category_cardinalities=preprocessor.category_cardinalities,
        model_config=cfg["model"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    structure_order = list(preprocessor.category_mapping[data_cfg["structure_col"]])
    thresholds = threshold_tensor(cfg["damage_thresholds"], structure_order, target_mode, target_mean, target_std)
    intensity_index = continuous_cols.index("PGA")
    similarity_indices = [continuous_cols.index(name) for name in ["Vs30", "num_stories", "T1", "total_height", "PGV_PGA", "Sa"] if name in continuous_cols]

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        losses = []
        for continuous, categorical, target, resonance in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(continuous, categorical)
            loss, _ = physics_informed_loss(
                outputs,
                target,
                continuous,
                categorical,
                resonance,
                thresholds,
                intensity_index,
                similarity_indices,
                cfg["physics"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:03d} train_loss={np.mean(losses):.6f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "target_mean": target_mean,
            "target_std": target_std,
            "target_transform": target_mode,
            "continuous_cols": continuous_cols,
            "structure_order": structure_order,
            "thresholds": cfg["damage_thresholds"],
        },
        output_dir / "model.pt",
    )
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    (output_dir / "training_metadata.json").write_text(json.dumps({"epochs": cfg["training"]["epochs"]}, indent=2), encoding="utf-8")
    print(f"Saved model to {output_dir / 'model.pt'}")


if __name__ == "__main__":
    main()

