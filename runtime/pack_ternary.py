#!/usr/bin/env python3
"""Ternary weight packing for efficient storage and inference.

Packs ternary weights {-1, 0, +1} into 2-bit representation:
  00 = 0, 01 = +1, 10 = -1, 11 = unused

This reduces storage from 32 bits/param to 2 bits/param (16x compression).
Provides pack/unpack kernels for CPU inference.

Storage format:
  - Packed weights: uint8 tensor (4 ternary values per byte)
  - Scales: float16 tensor (one per quantization group)
"""
from __future__ import annotations

import math
from typing import Optional

import torch


TERNARY_MAP_PACK = {-1: 2, 0: 0, 1: 1}
TERNARY_MAP_UNPACK = {0: 0, 1: 1, 2: -1, 3: 0}


def pack_ternary_weights(
    weights: torch.Tensor,
    group_size: int = 128,
) -> dict[str, torch.Tensor]:
    """Pack ternary weight tensor into 2-bit representation.

    Args:
        weights: float tensor with values in {-1, 0, +1}.
        group_size: size of quantization groups for scale factors.

    Returns:
        dict with 'packed' (uint8), 'scales' (float16), 'shape' (original shape).
    """
    orig_shape = weights.shape
    flat = weights.reshape(-1).float()

    n_groups = math.ceil(flat.numel() / group_size)
    pad_len = n_groups * group_size - flat.numel()
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len)])

    groups = flat.reshape(n_groups, group_size)
    scales = groups.abs().sum(dim=-1) / (groups != 0).float().sum(dim=-1).clamp(min=1)

    ternary = flat.round().long().clamp(-1, 1)
    mapped = torch.where(ternary == -1, torch.tensor(2, dtype=torch.long),
             torch.where(ternary == 1, torch.tensor(1, dtype=torch.long),
                         torch.tensor(0, dtype=torch.long)))

    n_vals = mapped.numel()
    n_bytes = math.ceil(n_vals / 4)
    pad_pack = n_bytes * 4 - n_vals
    if pad_pack > 0:
        mapped = torch.cat([mapped, torch.zeros(pad_pack, dtype=torch.long)])

    mapped = mapped.reshape(-1, 4)
    packed = (mapped[:, 0] | (mapped[:, 1] << 2) | (mapped[:, 2] << 4) | (mapped[:, 3] << 6))
    packed = packed.to(torch.uint8)

    return {
        "packed": packed,
        "scales": scales.half(),
        "shape": torch.tensor(list(orig_shape), dtype=torch.long),
        "group_size": torch.tensor([group_size], dtype=torch.long),
        "numel": torch.tensor([weights.numel()], dtype=torch.long),
    }


def unpack_ternary_weights(
    packed_data: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Unpack 2-bit ternary weights back to float tensor.

    Args:
        packed_data: dict from pack_ternary_weights.

    Returns:
        float32 tensor with values in {-1, 0, +1} * scale.
    """
    packed = packed_data["packed"]
    scales = packed_data["scales"].float()
    orig_shape = tuple(packed_data["shape"].tolist())
    group_size = packed_data["group_size"].item()
    numel = packed_data["numel"].item()

    vals0 = (packed & 0x03).long()
    vals1 = ((packed >> 2) & 0x03).long()
    vals2 = ((packed >> 4) & 0x03).long()
    vals3 = ((packed >> 6) & 0x03).long()

    flat = torch.stack([vals0, vals1, vals2, vals3], dim=-1).reshape(-1)[:numel]

    result = torch.where(flat == 2, torch.tensor(-1.0),
             torch.where(flat == 1, torch.tensor(1.0),
                         torch.tensor(0.0)))

    pad_len = len(scales) * group_size - numel
    if pad_len > 0:
        result = torch.cat([result, torch.zeros(pad_len)])

    groups = result.reshape(-1, group_size)
    groups = groups * scales.unsqueeze(-1)
    result = groups.reshape(-1)[:numel]

    return result.reshape(orig_shape)


def compute_packing_stats(weights: torch.Tensor) -> dict[str, float]:
    """Compute storage statistics for a weight tensor."""
    numel = weights.numel()
    fp32_bytes = numel * 4
    fp16_bytes = numel * 2
    packed_bytes = math.ceil(numel / 4)
    scale_bytes = math.ceil(numel / 128) * 2

    return {
        "numel": numel,
        "fp32_mb": fp32_bytes / 1e6,
        "fp16_mb": fp16_bytes / 1e6,
        "packed_mb": (packed_bytes + scale_bytes) / 1e6,
        "compression_ratio": fp16_bytes / max(packed_bytes + scale_bytes, 1),
        "bits_per_param": (packed_bytes + scale_bytes) * 8 / max(numel, 1),
    }


def pack_model_weights(
    model_state_dict: dict[str, torch.Tensor],
    group_size: int = 128,
    skip_keys: Optional[set[str]] = None,
) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
    """Pack all eligible weight tensors in a model state dict.

    Packs tensors that look like BitLinear weights (2D, values near {-1,0,+1}).
    Skips embeddings, norms, and biases.
    """
    skip_keys = skip_keys or set()
    packed = {}
    total_original = 0
    total_packed = 0

    for key, tensor in model_state_dict.items():
        if key in skip_keys:
            packed[key] = tensor
            continue

        is_weight = "weight" in key and tensor.ndim == 2
        is_norm = "norm" in key or "layernorm" in key
        is_embed = "embed" in key
        is_bias = "bias" in key
        is_head = "lm_head" in key

        if is_weight and not is_norm and not is_embed and not is_bias and not is_head:
            vals = tensor.float()
            ternary_pct = ((vals.abs() - 1.0).abs() < 0.3).float().mean() + (vals.abs() < 0.3).float().mean()
            if ternary_pct > 0.5:
                pack_data = pack_ternary_weights(vals.round().clamp(-1, 1), group_size)
                packed[key] = pack_data
                stats = compute_packing_stats(tensor)
                total_original += stats["fp16_mb"]
                total_packed += stats["packed_mb"]
                continue

        packed[key] = tensor
        if tensor.ndim >= 1:
            total_original += tensor.numel() * 2 / 1e6

    print(f"Packing: {total_original:.1f}MB -> {total_packed:.1f}MB "
          f"({total_original/max(total_packed,0.001):.1f}x compression on ternary layers)")
    return packed
