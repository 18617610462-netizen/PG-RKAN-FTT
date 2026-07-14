from __future__ import annotations

from .ft_transformer import FTTransformer, PIRKANFTT, RKANFTT


def build_model(name: str, num_continuous: int, category_cardinalities: list[int], model_config: dict):
    kwargs = {
        "d_token": model_config.get("d_token", 24),
        "n_blocks": model_config.get("n_blocks", 2),
        "n_heads": model_config.get("n_heads", 4),
        "basis_size": model_config.get("basis_size", 8),
        "hidden_multiplier": model_config.get("hidden_multiplier", 1.5),
        "attention_dropout": model_config.get("attention_dropout", 0.05),
        "residual_dropout": model_config.get("residual_dropout", 0.10),
        "ffn_dropout": model_config.get("ffn_dropout", 0.10),
    }
    if name == "FT-Transformer":
        return FTTransformer(num_continuous, category_cardinalities, **kwargs)
    if name == "RKAN-FFN":
        return RKANFTT(num_continuous, category_cardinalities, module_mode="rkan_ffn", **kwargs)
    if name == "RKAN-Attn":
        return RKANFTT(num_continuous, category_cardinalities, module_mode="rkan_attn", **kwargs)
    if name == "RKAN-FTT":
        return RKANFTT(num_continuous, category_cardinalities, module_mode="rkan_ftt", **kwargs)
    if name in {"PI-RKAN-FTT", "PG-RKAN-FTT"}:
        return PIRKANFTT(num_continuous, category_cardinalities, **kwargs)
    raise ValueError(f"Unknown model name: {name}")
