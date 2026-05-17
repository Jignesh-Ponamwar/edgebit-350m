#!/usr/bin/env python3
"""CPU inference benchmarking for EdgeBit models.

Measures:
  - Tokens per second (prefill + decode)
  - Time to first token (TTFT)
  - Per-token latency (decode)
  - Peak memory usage
  - Comparison: packed ternary vs fp16 vs fp32
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch

from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM


@dataclass
class BenchResult:
    model_name: str
    device: str
    dtype: str
    quant_mode: str
    n_params: int
    prompt_tokens: int
    gen_tokens: int
    prefill_ms: float
    decode_ms: float
    total_ms: float
    ttft_ms: float
    tokens_per_sec: float
    decode_tok_per_sec: float
    peak_mem_mb: float
    model_size_mb: float


def get_model_size_mb(model: torch.nn.Module) -> float:
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total / 1e6


def bench_prefill(
    model: EdgeBitForCausalLM,
    input_ids: torch.Tensor,
    device: torch.device,
    warmup: int = 2,
    repeats: int = 5,
) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids=input_ids)

        times = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids=input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return sum(times) / len(times)


def bench_generation(
    model: EdgeBitForCausalLM,
    input_ids: torch.Tensor,
    gen_tokens: int,
    device: torch.device,
    warmup: int = 1,
    repeats: int = 3,
) -> tuple[float, float, float]:
    with torch.no_grad():
        for _ in range(warmup):
            model.generate(input_ids, max_new_tokens=min(gen_tokens, 8), temperature=0.0)

        ttfts, decode_times, totals = [], [], []
        for _ in range(repeats):
            gc.collect()
            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            cur = input_ids.clone()

            logits = model(input_ids=cur)["logits"]
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, next_tok], dim=-1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_first = time.perf_counter()
            ttfts.append((t_first - t0) * 1000)

            for _ in range(gen_tokens - 1):
                logits = model(input_ids=cur)["logits"]
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cur = torch.cat([cur, next_tok], dim=-1)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            decode_times.append((t_end - t_first) * 1000)
            totals.append((t_end - t0) * 1000)

    avg_ttft = sum(ttfts) / len(ttfts)
    avg_decode = sum(decode_times) / len(decode_times)
    avg_total = sum(totals) / len(totals)
    return avg_ttft, avg_decode, avg_total


def run_benchmark(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    config_preset: str = "tiny",
    prompt_tokens: int = 64,
    gen_tokens: int = 32,
    device_str: str = "cpu",
    quant_mode: str = "ternary",
) -> BenchResult:
    device = torch.device(device_str)

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

    config.quant_mode = quant_mode
    model = EdgeBitForCausalLM(config)

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    model_size = get_model_size_mb(model)
    n_params = model.count_parameters()["total"]

    input_ids = torch.randint(0, config.vocab_size, (1, prompt_tokens), device=device)

    prefill_ms = bench_prefill(model, input_ids, device)
    ttft_ms, decode_ms, total_ms = bench_generation(model, input_ids, gen_tokens, device)

    tokens_per_sec = (prompt_tokens + gen_tokens) / (total_ms / 1000)
    decode_tok_per_sec = (gen_tokens - 1) / (decode_ms / 1000) if decode_ms > 0 else 0

    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
    else:
        try:
            import psutil
            peak_mem = psutil.Process().memory_info().rss / 1e6
        except ImportError:
            pass

    dtype_str = str(next(model.parameters()).dtype).replace("torch.", "")

    return BenchResult(
        model_name=f"edgebit-{config_preset}",
        device=device_str,
        dtype=dtype_str,
        quant_mode=quant_mode,
        n_params=n_params,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        prefill_ms=round(prefill_ms, 2),
        decode_ms=round(decode_ms, 2),
        total_ms=round(total_ms, 2),
        ttft_ms=round(ttft_ms, 2),
        tokens_per_sec=round(tokens_per_sec, 2),
        decode_tok_per_sec=round(decode_tok_per_sec, 2),
        peak_mem_mb=round(peak_mem, 2),
        model_size_mb=round(model_size, 2),
    )


def format_results(results: list[BenchResult]) -> str:
    lines = [
        f"{'Model':<20} {'Device':<8} {'Quant':<10} {'Params':<10} "
        f"{'Prefill':<10} {'TTFT':<10} {'Decode':<12} {'Tok/s':<10} {'Mem MB':<10}",
        "-" * 110,
    ]
    for r in results:
        params_str = f"{r.n_params/1e6:.1f}M"
        lines.append(
            f"{r.model_name:<20} {r.device:<8} {r.quant_mode:<10} {params_str:<10} "
            f"{r.prefill_ms:>7.1f}ms {r.ttft_ms:>7.1f}ms "
            f"{r.decode_tok_per_sec:>8.1f}t/s {r.tokens_per_sec:>7.1f}t/s "
            f"{r.peak_mem_mb:>8.1f}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="EdgeBit CPU Inference Benchmark")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--preset", type=str, default="tiny", choices=["tiny", "small_125m", "base"])
    parser.add_argument("--prompt_tokens", type=int, default=64)
    parser.add_argument("--gen_tokens", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quant_modes", nargs="+", default=["none", "int8", "ternary"])
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    results = []
    for qm in args.quant_modes:
        print(f"\nBenchmarking quant_mode={qm}...")
        r = run_benchmark(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            config_preset=args.preset,
            prompt_tokens=args.prompt_tokens,
            gen_tokens=args.gen_tokens,
            device_str=args.device,
            quant_mode=qm,
        )
        results.append(r)
        print(f"  prefill={r.prefill_ms:.1f}ms  ttft={r.ttft_ms:.1f}ms  "
              f"decode={r.decode_tok_per_sec:.1f}t/s  total={r.tokens_per_sec:.1f}t/s  "
              f"mem={r.peak_mem_mb:.1f}MB")

    print("\n" + format_results(results))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
