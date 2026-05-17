"""Packed EdgeBit checkpoint helpers.

This module stores BitLinear-style weights as 2-bit ternary tensors plus fp16
group scales. Loading reconstructs regular tensors so the existing PyTorch
model can run them. It is a verified packed storage path, not a custom packed
matmul runtime.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from modeling.quant_utils import ternary_quantize_absmean
from runtime.pack_ternary import pack_ternary_weights, unpack_ternary_weights


def is_packable_weight(name: str, tensor: torch.Tensor) -> bool:
    if tensor.ndim != 2 or "weight" not in name:
        return False
    skipped = ("embed", "norm", "layernorm", "lm_head", "bias")
    return not any(part in name for part in skipped)


def pack_edgebit_state_dict(
    state_dict: dict[str, torch.Tensor],
    group_size: int = 128,
) -> dict[str, Any]:
    packed: dict[str, Any] = {
        "__format__": "edgebit_packed_state_dict_v1",
        "__group_size__": group_size,
        "tensors": {},
    }

    for name, tensor in state_dict.items():
        cpu_tensor = tensor.detach().cpu()
        if is_packable_weight(name, cpu_tensor):
            ternary, _ = ternary_quantize_absmean(cpu_tensor.float(), group_size=group_size)
            packed["tensors"][name] = {
                "packed_ternary": pack_ternary_weights(ternary, group_size=group_size),
                "dtype": str(cpu_tensor.dtype).replace("torch.", ""),
            }
        else:
            packed["tensors"][name] = {"tensor": cpu_tensor}

    return packed


def unpack_edgebit_state_dict(packed_state: dict[str, Any]) -> dict[str, torch.Tensor]:
    if packed_state.get("__format__") != "edgebit_packed_state_dict_v1":
        raise ValueError("Unsupported packed checkpoint format")

    state_dict: dict[str, torch.Tensor] = {}
    for name, payload in packed_state["tensors"].items():
        if "packed_ternary" in payload:
            state_dict[name] = unpack_ternary_weights(payload["packed_ternary"])
        else:
            state_dict[name] = payload["tensor"]
    return state_dict


def save_packed_checkpoint(
    checkpoint: dict[str, Any],
    output_path: str,
    group_size: int = 128,
) -> str:
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    packed = pack_edgebit_state_dict(state_dict, group_size=group_size)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(packed, output_path)
    return output_path


def load_packed_checkpoint(path: str) -> dict[str, torch.Tensor]:
    packed = torch.load(path, map_location="cpu", weights_only=False)
    return unpack_edgebit_state_dict(packed)


def estimate_packed_state_size_bytes(packed_state: dict[str, Any]) -> int:
    total = 0
    for payload in packed_state["tensors"].values():
        if "packed_ternary" in payload:
            data = payload["packed_ternary"]
            total += data["packed"].numel() * data["packed"].element_size()
            total += data["scales"].numel() * data["scales"].element_size()
            total += data["shape"].numel() * data["shape"].element_size()
            total += data["group_size"].numel() * data["group_size"].element_size()
            total += data["numel"].numel() * data["numel"].element_size()
        else:
            tensor = payload["tensor"]
            total += tensor.numel() * tensor.element_size()
    return total
