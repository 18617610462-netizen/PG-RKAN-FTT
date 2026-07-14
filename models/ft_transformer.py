from __future__ import annotations

import torch
import torch.nn as nn

from .rkan import RBFKANLinear, RKANFeedForward


class FeatureTokenizer(nn.Module):
    def __init__(self, num_continuous: int, category_cardinalities: list[int], d_token: int) -> None:
        super().__init__()
        self.continuous_weight = nn.Parameter(torch.randn(num_continuous, d_token) * 0.02)
        self.continuous_bias = nn.Parameter(torch.zeros(num_continuous, d_token))
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_token) for cardinality in category_cardinalities]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

    def forward(self, continuous: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        continuous_tokens = (
            continuous.unsqueeze(-1) * self.continuous_weight.unsqueeze(0)
            + self.continuous_bias.unsqueeze(0)
        )
        categorical_tokens = [
            embedding(categorical[:, index].long()).unsqueeze(1)
            for index, embedding in enumerate(self.category_embeddings)
        ]
        cls = self.cls_token.expand(continuous.shape[0], -1, -1)
        return torch.cat([cls, continuous_tokens, *categorical_tokens], dim=1)


class MultiheadAttention(nn.Module):
    def __init__(self, d_token: int, n_heads: int, dropout: float, use_rkan: bool, basis_size: int) -> None:
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError("d_token must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_token // n_heads
        self.scale = self.head_dim**-0.5
        layer = RBFKANLinear if use_rkan else nn.Linear
        kwargs = {"basis_size": basis_size} if use_rkan else {}
        self.q_projection = layer(d_token, d_token, **kwargs)
        self.k_projection = layer(d_token, d_token, **kwargs)
        self.v_projection = layer(d_token, d_token, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_token, d_token)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        return x.reshape(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.split_heads(self.q_projection(x))
        k = self.split_heads(self.k_projection(x))
        v = self.split_heads(self.v_projection(x))
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        context = torch.matmul(self.dropout(weights), v)
        context = context.transpose(1, 2).reshape(x.shape)
        return self.output_projection(context)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        hidden_multiplier: float,
        basis_size: int,
        attention_dropout: float,
        residual_dropout: float,
        ffn_dropout: float,
        use_rkan_attention: bool,
        use_rkan_ffn: bool,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_token)
        self.attention = MultiheadAttention(
            d_token=d_token,
            n_heads=n_heads,
            dropout=attention_dropout,
            use_rkan=use_rkan_attention,
            basis_size=basis_size,
        )
        self.residual_dropout = nn.Dropout(residual_dropout)
        self.norm_ffn = nn.LayerNorm(d_token)
        hidden_dim = max(d_token, int(d_token * hidden_multiplier))
        if use_rkan_ffn:
            self.ffn = RKANFeedForward(d_token, hidden_dim, basis_size=basis_size, dropout=ffn_dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_token, hidden_dim),
                nn.GELU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(hidden_dim, d_token),
                nn.Dropout(ffn_dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_dropout(self.attention(self.norm_attention(x)))
        return x + self.ffn(self.norm_ffn(x))


class FTTransformerBackbone(nn.Module):
    def __init__(
        self,
        num_continuous: int,
        category_cardinalities: list[int],
        d_token: int = 24,
        n_blocks: int = 2,
        n_heads: int = 4,
        basis_size: int = 8,
        hidden_multiplier: float = 1.5,
        attention_dropout: float = 0.05,
        residual_dropout: float = 0.10,
        ffn_dropout: float = 0.10,
        module_mode: str = "ft",
    ) -> None:
        super().__init__()
        if module_mode not in {"ft", "rkan_ffn", "rkan_attn", "rkan_ftt"}:
            raise ValueError(f"Unknown module_mode: {module_mode}")
        self.tokenizer = FeatureTokenizer(num_continuous, category_cardinalities, d_token)
        self.encoder = nn.ModuleList(
            [
                EncoderLayer(
                    d_token=d_token,
                    n_heads=n_heads,
                    hidden_multiplier=hidden_multiplier,
                    basis_size=basis_size,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                    ffn_dropout=ffn_dropout,
                    use_rkan_attention=module_mode in {"rkan_attn", "rkan_ftt"},
                    use_rkan_ffn=module_mode in {"rkan_ffn", "rkan_ftt"},
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, continuous: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(continuous, categorical)
        for layer in self.encoder:
            tokens = layer(tokens)
        return tokens[:, 0]


class MultiTaskHead(nn.Module):
    """Joint MIDR regression and damage-state logits."""

    def __init__(self, d_token: int, n_damage_states: int = 5) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.LayerNorm(d_token), nn.GELU())
        self.regression = nn.Linear(d_token, 1)
        self.classification = nn.Linear(d_token, n_damage_states)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(features)
        return {
            "midr": self.regression(h).squeeze(-1),
            "damage_logits": self.classification(h),
        }


class FTTransformer(nn.Module):
    def __init__(self, num_continuous: int, category_cardinalities: list[int], **kwargs) -> None:
        super().__init__()
        d_token = kwargs.get("d_token", 24)
        self.backbone = FTTransformerBackbone(
            num_continuous,
            category_cardinalities,
            module_mode="ft",
            **kwargs,
        )
        self.head = MultiTaskHead(d_token=d_token)

    def forward(self, continuous: torch.Tensor, categorical: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.backbone(continuous, categorical))


class RKANFTT(nn.Module):
    def __init__(
        self,
        num_continuous: int,
        category_cardinalities: list[int],
        module_mode: str = "rkan_ftt",
        **kwargs,
    ) -> None:
        super().__init__()
        d_token = kwargs.get("d_token", 24)
        self.backbone = FTTransformerBackbone(
            num_continuous,
            category_cardinalities,
            module_mode=module_mode,
            **kwargs,
        )
        self.head = MultiTaskHead(d_token=d_token)

    def forward(self, continuous: torch.Tensor, categorical: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.backbone(continuous, categorical))


class PIRKANFTT(RKANFTT):
    """PI-RKAN-FTT uses the same architecture as RKAN-FTT and physics-informed losses."""

    def __init__(self, num_continuous: int, category_cardinalities: list[int], **kwargs) -> None:
        super().__init__(
            num_continuous=num_continuous,
            category_cardinalities=category_cardinalities,
            module_mode="rkan_ftt",
            **kwargs,
        )

