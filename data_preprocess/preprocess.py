from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


DAMAGE_LABELS = ["DS1", "DS2", "DS3", "DS4", "DS5"]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def damage_state_from_midr(structure_type: str, midr: float, thresholds: dict[str, list[float]]) -> str:
    t1, t2, t3, t4 = thresholds[str(structure_type)]
    value = max(float(midr), 0.0)
    if value <= t1:
        return "DS1"
    if value <= t2:
        return "DS2"
    if value <= t3:
        return "DS3"
    if value <= t4:
        return "DS4"
    return "DS5"


def add_physics_features(frame: pd.DataFrame, alpha_tg: float = 6.283185307, eps: float = 1e-8) -> pd.DataFrame:
    result = frame.copy()
    result["Tg"] = alpha_tg * result["PGV"].astype(float) / (result["PGA"].astype(float) + eps)
    result["resonance_factor"] = np.exp(
        -np.abs(result["T1"].astype(float) - result["Tg"].astype(float))
        / (np.abs(result["Tg"].astype(float)) + eps)
    )
    return result


def split_dataset(
    frame: pd.DataFrame,
    stratify_cols: list[str],
    seed: int = 2026,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify = frame[stratify_cols].astype(str).agg("_".join, axis=1)
    train, temp = train_test_split(frame, train_size=train_ratio, random_state=seed, stratify=stratify)
    temp_stratify = temp[stratify_cols].astype(str).agg("_".join, axis=1)
    val_fraction = val_ratio / (1.0 - train_ratio)
    val, test = train_test_split(temp, train_size=val_fraction, random_state=seed, stratify=temp_stratify)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


@dataclass
class TabularPreprocessor:
    continuous_cols: list[str]
    categorical_cols: list[str]
    category_mapping: dict[str, dict[str, int]] | None = None
    scaler: StandardScaler | None = None

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        self.scaler = StandardScaler().fit(frame[self.continuous_cols].to_numpy(dtype=float))
        self.category_mapping = {}
        for column in self.categorical_cols:
            values = sorted(frame[column].astype(str).unique())
            self.category_mapping[column] = {value: index for index, value in enumerate(values)}
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.scaler is None or self.category_mapping is None:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        continuous = self.scaler.transform(frame[self.continuous_cols].to_numpy(dtype=float)).astype(np.float32)
        categorical_columns = []
        for column in self.categorical_cols:
            mapping = self.category_mapping[column]
            categorical_columns.append(
                frame[column].astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64)
            )
        categorical = np.vstack(categorical_columns).T.astype(np.int64)
        return continuous, categorical

    @property
    def category_cardinalities(self) -> list[int]:
        if self.category_mapping is None:
            raise RuntimeError("Preprocessor has not been fitted.")
        return [len(self.category_mapping[column]) for column in self.categorical_cols]


class DamageDataset(Dataset):
    def __init__(
        self,
        continuous: np.ndarray,
        categorical: np.ndarray,
        target: np.ndarray,
        resonance_factor: np.ndarray | None = None,
    ) -> None:
        self.continuous = torch.tensor(continuous, dtype=torch.float32)
        self.categorical = torch.tensor(categorical, dtype=torch.long)
        self.target = torch.tensor(target, dtype=torch.float32)
        if resonance_factor is None:
            resonance_factor = np.zeros(len(target), dtype=np.float32)
        self.resonance_factor = torch.tensor(resonance_factor, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.continuous[index],
            self.categorical[index],
            self.target[index],
            self.resonance_factor[index],
        )

