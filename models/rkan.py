from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBFKANLinear(nn.Module):
    """RKAN Linear layer using Gaussian radial basis functions.

    The layer keeps a conventional smooth base branch and adds a local RBF-KAN
    branch. It is used for Q/K/V attention projections and feed-forward blocks.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        basis_size: int = 8,
        grid_range: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.basis_size = basis_size
        grid = torch.linspace(grid_range[0], grid_range[1], basis_size)
        self.register_buffer("grid", grid)
        self.denominator = (grid_range[1] - grid_range[0]) / max(basis_size - 1, 1)
        self.base_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.rbf_weight = nn.Parameter(torch.empty(output_dim, input_dim, basis_size))
        self.rbf_scaler = nn.Parameter(torch.ones(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.normal_(self.rbf_weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dimension {self.input_dim}, got {x.shape[-1]}.")
        output_shape = x.shape[:-1] + (self.output_dim,)
        flat = x.reshape(-1, self.input_dim)
        bounded = 2.0 * torch.tanh(flat / 2.0)
        grid = self.grid.to(device=x.device, dtype=x.dtype)
        basis = torch.exp(-((bounded.unsqueeze(-1) - grid) / self.denominator) ** 2)
        base_output = F.linear(F.silu(flat), self.base_weight)
        scaled_weight = self.rbf_weight * self.rbf_scaler.unsqueeze(-1)
        rbf_output = torch.einsum("bik,oik->bo", basis, scaled_weight)
        return (base_output + rbf_output + self.bias).reshape(output_shape)


class RKANFeedForward(nn.Module):
    """Transformer feed-forward block with RKAN Linear mappings."""

    def __init__(
        self,
        d_token: int,
        hidden_dim: int,
        basis_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            RBFKANLinear(d_token, hidden_dim, basis_size=basis_size),
            nn.Dropout(dropout),
            RBFKANLinear(hidden_dim, d_token, basis_size=basis_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

