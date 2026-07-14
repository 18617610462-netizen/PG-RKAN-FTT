from __future__ import annotations

import torch
import torch.nn.functional as F


def threshold_neighborhood_weights(
    target: torch.Tensor,
    structure_index: torch.Tensor,
    thresholds: torch.Tensor,
    weight_scale: float,
    distance_scale: float,
) -> torch.Tensor:
    sample_thresholds = thresholds[structure_index.long()]
    distance = torch.min(torch.abs(target.unsqueeze(1) - sample_thresholds), dim=1).values
    return 1.0 + weight_scale * torch.exp(-distance / max(distance_scale, 1e-6))


def period_spectrum_weights(resonance_factor: torch.Tensor, weight_scale: float) -> torch.Tensor:
    return 1.0 + weight_scale * torch.clamp(resonance_factor, min=0.0)


def intensity_trend_constraint(
    prediction: torch.Tensor,
    continuous: torch.Tensor,
    categorical: torch.Tensor,
    intensity_index: int,
    similarity_indices: list[int],
    margin: float = 0.01,
    similarity_radius: float = 2.5,
) -> torch.Tensor:
    """Soft pairwise penalty for nearby samples with stronger IM but lower MIDR."""
    if prediction.shape[0] < 2:
        return prediction.new_tensor(0.0)
    descriptor = continuous[:, similarity_indices]
    distance = torch.cdist(descriptor, descriptor, p=2)
    same_structure = categorical[:, 0].unsqueeze(0) == categorical[:, 0].unsqueeze(1)
    intensity = continuous[:, intensity_index]
    stronger = intensity.unsqueeze(0) > intensity.unsqueeze(1) + margin
    mask = same_structure & (distance < similarity_radius) & stronger
    if not bool(mask.any()):
        return prediction.new_tensor(0.0)
    pred_lower_im = prediction.unsqueeze(0).expand_as(distance)
    pred_higher_im = prediction.unsqueeze(1).expand_as(distance)
    violation = torch.relu(pred_lower_im - pred_higher_im + margin)
    return violation[mask].pow(2).mean()


def ordinal_damage_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    structure_index: torch.Tensor,
    thresholds: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    sample_thresholds = thresholds[structure_index.long()]
    exceedance = (target.unsqueeze(1) > sample_thresholds).to(prediction.dtype)
    logits = (prediction.unsqueeze(1) - sample_thresholds) / max(temperature, 1e-6)
    return F.binary_cross_entropy_with_logits(logits, exceedance)


def physics_informed_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    continuous: torch.Tensor,
    categorical: torch.Tensor,
    resonance_factor: torch.Tensor,
    thresholds: torch.Tensor,
    intensity_index: int,
    similarity_indices: list[int],
    config: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = outputs["midr"]
    structure_index = categorical[:, 0]
    reg = F.smooth_l1_loss(prediction, target, beta=0.5, reduction="none")
    threshold_w = threshold_neighborhood_weights(
        target,
        structure_index,
        thresholds,
        config.get("threshold_sample_weight", 1.0),
        config.get("threshold_distance_scale", 0.35),
    )
    resonance_w = period_spectrum_weights(
        resonance_factor,
        config.get("resonance_sample_weight", 0.5),
    )
    weights = threshold_w * resonance_w
    weights = weights / weights.mean().clamp_min(1e-6)
    regression_loss = (weights * reg).mean()
    cls_loss = ordinal_damage_loss(
        prediction,
        target,
        structure_index,
        thresholds,
        temperature=config.get("ordinal_temperature", 0.12),
    )
    trend_loss = intensity_trend_constraint(
        prediction,
        continuous,
        categorical,
        intensity_index=intensity_index,
        similarity_indices=similarity_indices,
        margin=config.get("intensity_pair_margin", 0.01),
        similarity_radius=config.get("similarity_radius", 2.5),
    )
    total = (
        regression_loss
        + config.get("classification_weight", 0.2) * cls_loss
        + config.get("intensity_trend_weight", 0.02) * trend_loss
    )
    return total, {
        "regression": regression_loss.detach(),
        "classification": cls_loss.detach(),
        "intensity_trend": trend_loss.detach(),
        "threshold_weight_mean": threshold_w.detach().mean(),
        "resonance_weight_mean": resonance_w.detach().mean(),
    }

