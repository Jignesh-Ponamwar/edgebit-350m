"""Low-bit quantization primitives for EdgeBit.

Provides:
  - Straight-through estimator (STE) rounding
  - Absmean group-wise ternary quantization
  - NF4 embedding quantization
  - INT8 symmetric quantization (for KV cache)
  - Quantization statistics collection
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


# -- STE helpers --------------------------------------------------------------

def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through estimator gradient."""
    return (x.round() - x).detach() + x


def ste_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Clamp with straight-through estimator gradient."""
    return (x.clamp(lo, hi) - x).detach() + x


# -- Ternary quantization -----------------------------------------------------

def ternary_quantize_absmean(
    weight: torch.Tensor,
    group_size: int = 128,
    stochastic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight tensor to {-1, 0, +1} using per-group absmean scaling.

    Args:
        weight: (..., K) weight tensor. Last dim is grouped.
        group_size: number of elements per quantization group.
        stochastic: if True, use stochastic rounding during training.

    Returns:
        (quantized_weight, scale) where quantized is in {-1, 0, +1}
        and scale is per-group absolute-mean.
    """
    orig_shape = weight.shape
    K = orig_shape[-1]

    if group_size > 0 and K % group_size == 0:
        weight_grouped = weight.reshape(-1, group_size)
    else:
        weight_grouped = weight.reshape(-1, K)

    scale = weight_grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
    normalized = weight_grouped / scale

    if stochastic and weight.requires_grad:
        noise = torch.rand_like(normalized) - 0.5
        quantized = (normalized + noise).round()
    else:
        quantized = ste_round(normalized)

    quantized = ste_clamp(quantized, -1.0, 1.0)
    quantized = quantized.reshape(orig_shape)
    scale = scale.reshape(*orig_shape[:-1], -1)

    return quantized, scale


# -- INT8 symmetric quantization (KV cache) -----------------------------------

def int8_symmetric_quantize(
    x: torch.Tensor,
    per_token: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 range [-127, 127] with symmetric scaling.

    Args:
        x: input tensor, typically (batch, seq, dim).
        per_token: if True, compute scale per token (last dim). Otherwise per-tensor.

    Returns:
        (quantized, scale) where quantized is in [-127, 127].
    """
    if per_token:
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    else:
        amax = x.abs().amax().clamp(min=1e-5)

    scale = 127.0 / amax
    quantized = ste_clamp(ste_round(x * scale), -127.0, 127.0)
    return quantized, scale


def int8_dequantize(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 tensor back to float."""
    return quantized / scale


# -- NF4 quantization (for embeddings) ----------------------------------------

NF4_LEVELS = torch.tensor([
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0,
], dtype=torch.float32)


def nf4_quantize(
    weight: torch.Tensor,
    block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight to NF4 (4-bit NormalFloat) per block.

    Args:
        weight: tensor to quantize.
        block_size: elements per quantization block.

    Returns:
        (indices, scale) where indices are 0-15 NF4 codes.
    """
    orig_shape = weight.shape
    flat = weight.reshape(-1)
    pad_len = (block_size - flat.numel() % block_size) % block_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))

    blocks = flat.reshape(-1, block_size)
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    normalized = blocks / scale

    levels = NF4_LEVELS.to(weight.device)
    diffs = (normalized.unsqueeze(-1) - levels.unsqueeze(0).unsqueeze(0)).abs()
    indices = diffs.argmin(dim=-1)

    return indices.reshape(-1)[:math.prod(orig_shape)].reshape(orig_shape), scale.squeeze(-1)


def nf4_dequantize(
    indices: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 64,
    orig_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Dequantize NF4 indices back to float."""
    levels = NF4_LEVELS.to(indices.device)
    flat_idx = indices.reshape(-1)
    flat_vals = levels[flat_idx.long()]

    n_blocks = (flat_vals.numel() + block_size - 1) // block_size
    pad_len = n_blocks * block_size - flat_vals.numel()
    if pad_len > 0:
        flat_vals = F.pad(flat_vals, (0, pad_len))

    blocks = flat_vals.reshape(-1, block_size)
    blocks = blocks * scale.unsqueeze(-1)
    result = blocks.reshape(-1)

    if orig_shape is not None:
        result = result[:math.prod(orig_shape)].reshape(orig_shape)
    return result


# -- Quantization statistics ---------------------------------------------------

@dataclass
class QuantStats:
    """Collects quantization statistics for monitoring."""
    name: str = ""
    mean_abs_weight: float = 0.0
    weight_std: float = 0.0
    pct_zero: float = 0.0
    pct_plus_one: float = 0.0
    pct_minus_one: float = 0.0
    scale_mean: float = 0.0
    scale_std: float = 0.0
    grad_norm: float = 0.0
    _count: int = field(default=0, repr=False)

    def update(self, weight: torch.Tensor, quantized: torch.Tensor,
               scale: torch.Tensor, grad: Optional[torch.Tensor] = None) -> None:
        with torch.no_grad():
            w = weight.float()
            q = quantized.float()
            self.mean_abs_weight = w.abs().mean().item()
            self.weight_std = w.std().item()
            self.pct_zero = (q.abs() < 0.5).float().mean().item()
            self.pct_plus_one = (q > 0.5).float().mean().item()
            self.pct_minus_one = (q < -0.5).float().mean().item()
            self.scale_mean = scale.float().mean().item()
            self.scale_std = scale.float().std().item()
            if grad is not None:
                self.grad_norm = grad.float().norm().item()
            self._count += 1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean_abs_weight": round(self.mean_abs_weight, 6),
            "weight_std": round(self.weight_std, 6),
            "pct_zero": round(self.pct_zero, 4),
            "pct_plus_one": round(self.pct_plus_one, 4),
            "pct_minus_one": round(self.pct_minus_one, 4),
            "scale_mean": round(self.scale_mean, 6),
            "scale_std": round(self.scale_std, 6),
            "grad_norm": round(self.grad_norm, 6),
        }
