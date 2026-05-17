"""SubLN (Sub-Layer Normalization) for BitLinear gradient stabilization.

Inserted before attention output projection and FFN output projection.
Initialized to identity so it has no effect at init but learns to
stabilize gradient flow during quantization-aware training.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SubLayerNorm(nn.Module):
    """RMSNorm variant inserted before output projections.

    Initializes gamma to 1.0 (identity passthrough at init) so the model
    behaves identically to a non-SubLN model at the start of training.
    During training, it learns to rescale pre-projection activations to
    prevent gradient collapse under ternary quantization.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6, enabled: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.enabled = enabled
        self.gamma = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.gamma

    def extra_repr(self) -> str:
        return f"dim={self.normalized_shape}, enabled={self.enabled}"
