#!/usr/bin/env python3
"""Memory profiling for EdgeBit models.

Tracks:
  - Model weight memory (by layer type)
  - Activation memory during forward pass
  - KV cache memory at various sequence lengths
  - Peak RSS / GPU memory
  - Packed vs unpacked comparison
"""
from __future__ import annotations

import gc
import os
import json
import argparse
from dataclasses import dataclass, asdict, field
from typing import Optional

import torch

from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM
from runtime.pack_ternary import pack_model_weights, compute_packing_stats


@dataclass
class LayerMemProfile:
    name: str
    param_count: int
    param_bytes: int
    dtype: str


@dataclass
class MemoryProfile:
    model_name: str
    total_params: int
    weight_memory_mb: float
    packed_weight_memory_mb: float
    compression_ratio: float
    embedding_memory_mb: float
    norm_memory_mb: float
    bitlinear_memory_mb: float
    lm_head_memory_mb: float
    kv_cache_memory_mb: dict[int, float] = field(default_factory=dict)
    activation_memory_mb: float = 0.0
    peak_rss_mb: float = 0.0
    layer_details: list[LayerMemProfile] = field(default_factory=list)


def profile_weight_memory(model: torch.nn.Module) -> dict[str, float]:
    categories = {"embedding": 0, "norm": 0, "bitlinear": 0, "lm_head": 0, "other": 0}

    for name, param in model.named_parameters():
        size_bytes = param.nelement() * param.element_size()
        if "embed" in name:
            categories["embedding"] += size_bytes
        elif "norm" in name or "layernorm" in name:
            categories["norm"] += size_bytes
        elif "lm_head" in name:
            categories["lm_head"] += size_bytes
        elif "weight" in name and param.ndim == 2:
            categories["bitlinear"] += size_bytes
        else:
            categories["other"] += size_bytes

    return {k: v / 1e6 for k, v in categories.items()}


def profile_kv_cache_memory(config: EdgeBitConfig, seq_lengths: list[int]) -> dict[int, float]:
    results = {}
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    n_layers = config.n_layers
    batch_size = 1

    for seq_len in seq_lengths:
        if config.kv_cache_quant:
            bytes_per_element = 1
            scale_overhead = (seq_len * n_kv_heads * head_dim) / 128 * 2
        else:
            bytes_per_element = 2

        kv_bytes = 2 * n_layers * batch_size * seq_len * n_kv_heads * head_dim * bytes_per_element
        if config.kv_cache_quant:
            kv_bytes += 2 * n_layers * scale_overhead

        results[seq_len] = kv_bytes / 1e6

    return results


def profile_activation_memory(
    model: EdgeBitForCausalLM,
    config: EdgeBitConfig,
    seq_len: int = 512,
    batch_size: int = 1,
    device: str = "cpu",
) -> float:
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    gc.collect()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
    else:
        try:
            import psutil
            mem_before = psutil.Process().memory_info().rss
        except ImportError:
            return 0.0

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    with torch.no_grad():
        model(input_ids=input_ids)

    if device.type == "cuda":
        torch.cuda.synchronize()
        mem_after = torch.cuda.max_memory_allocated()
    else:
        try:
            import psutil
            mem_after = psutil.Process().memory_info().rss
        except ImportError:
            return 0.0

    return (mem_after - mem_before) / 1e6


def profile_packed_memory(model: EdgeBitForCausalLM) -> tuple[float, float]:
    state_dict = model.state_dict()

    unpacked_bytes = 0
    packed_bytes = 0

    for key, tensor in state_dict.items():
        size = tensor.nelement() * tensor.element_size()
        unpacked_bytes += size

        is_weight = "weight" in key and tensor.ndim == 2
        is_norm = "norm" in key
        is_embed = "embed" in key
        is_head = "lm_head" in key

        if is_weight and not is_norm and not is_embed and not is_head:
            stats = compute_packing_stats(tensor)
            packed_bytes += stats["packed_mb"] * 1e6
        else:
            packed_bytes += size

    return unpacked_bytes / 1e6, packed_bytes / 1e6


def run_profile(
    config_path: Optional[str] = None,
    config_preset: str = "tiny",
    seq_lengths: Optional[list[int]] = None,
    device: str = "cpu",
) -> MemoryProfile:
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048]

    if config_path:
        import yaml
        with open(config_path) as f:
            cfg_dict = yaml.safe_load(f)
        config = EdgeBitConfig(**cfg_dict.get("model", cfg_dict))
    else:
        preset_fn = getattr(EdgeBitConfig, config_preset, None)
        if preset_fn and callable(preset_fn):
            config = preset_fn()
        else:
            config = EdgeBitConfig.tiny()

    model = EdgeBitForCausalLM(config)
    model.eval()

    total_params = model.count_parameters()["total"]
    weight_mem = profile_weight_memory(model)
    total_weight_mb = sum(weight_mem.values())

    unpacked_mb, packed_mb = profile_packed_memory(model)
    compression = unpacked_mb / max(packed_mb, 0.001)

    kv_cache_mem = profile_kv_cache_memory(config, seq_lengths)

    activation_mb = profile_activation_memory(model, config, seq_len=512, device=device)

    peak_rss = 0.0
    try:
        import psutil
        peak_rss = psutil.Process().memory_info().rss / 1e6
    except ImportError:
        pass

    layer_details = []
    for name, param in model.named_parameters():
        layer_details.append(LayerMemProfile(
            name=name,
            param_count=param.nelement(),
            param_bytes=param.nelement() * param.element_size(),
            dtype=str(param.dtype).replace("torch.", ""),
        ))

    return MemoryProfile(
        model_name=f"edgebit-{config_preset}",
        total_params=total_params,
        weight_memory_mb=round(total_weight_mb, 2),
        packed_weight_memory_mb=round(packed_mb, 2),
        compression_ratio=round(compression, 2),
        embedding_memory_mb=round(weight_mem.get("embedding", 0), 2),
        norm_memory_mb=round(weight_mem.get("norm", 0), 2),
        bitlinear_memory_mb=round(weight_mem.get("bitlinear", 0), 2),
        lm_head_memory_mb=round(weight_mem.get("lm_head", 0), 2),
        kv_cache_memory_mb={k: round(v, 2) for k, v in kv_cache_mem.items()},
        activation_memory_mb=round(activation_mb, 2),
        peak_rss_mb=round(peak_rss, 2),
        layer_details=layer_details,
    )


def format_profile(p: MemoryProfile) -> str:
    lines = [
        f"=== Memory Profile: {p.model_name} ===",
        f"Total parameters: {p.total_params:,}",
        f"",
        f"Weight Memory (fp32):    {p.weight_memory_mb:.2f} MB",
        f"Weight Memory (packed):  {p.packed_weight_memory_mb:.2f} MB",
        f"Compression ratio:       {p.compression_ratio:.1f}x",
        f"",
        f"  Embeddings:  {p.embedding_memory_mb:.2f} MB",
        f"  Norms:       {p.norm_memory_mb:.2f} MB",
        f"  BitLinear:   {p.bitlinear_memory_mb:.2f} MB",
        f"  LM Head:     {p.lm_head_memory_mb:.2f} MB",
        f"",
        f"KV Cache Memory (batch=1):",
    ]
    for sl, mem in sorted(p.kv_cache_memory_mb.items()):
        lines.append(f"  seq_len={sl:>5}: {mem:.2f} MB")

    lines.extend([
        f"",
        f"Activation memory (seq=512): {p.activation_memory_mb:.2f} MB",
        f"Peak RSS: {p.peak_rss_mb:.1f} MB",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="EdgeBit Memory Profiler")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--preset", type=str, default="tiny")
    parser.add_argument("--seq_lengths", nargs="+", type=int, default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--show_layers", action="store_true")
    args = parser.parse_args()

    profile = run_profile(
        config_path=args.config,
        config_preset=args.preset,
        seq_lengths=args.seq_lengths,
        device=args.device,
    )

    print(format_profile(profile))

    if args.show_layers:
        print(f"\n{'Layer':<60} {'Params':>12} {'Bytes':>12} {'Dtype':<10}")
        print("-" * 96)
        for ld in profile.layer_details:
            print(f"{ld.name:<60} {ld.param_count:>12,} {ld.param_bytes:>12,} {ld.dtype:<10}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        data = asdict(profile)
        with open(args.output_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nProfile saved to {args.output_json}")


if __name__ == "__main__":
    main()
