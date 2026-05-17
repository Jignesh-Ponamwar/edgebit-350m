#!/usr/bin/env python3
"""Token throughput and latency benchmarking for EdgeBit models.

Measures:
  - Tokens/second at various batch sizes and sequence lengths
  - RAM usage per configuration
  - Latency percentiles (p50, p90, p99)
  - Throughput scaling across batch sizes
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import numpy as np

from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM

logger = logging.getLogger(__name__)


@dataclass
class ThroughputResult:
    batch_size: int
    seq_len: int
    quant_mode: str
    n_iterations: int
    total_tokens: int
    total_time_s: float
    tokens_per_sec: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    mem_mb: float


def measure_throughput(
    model: EdgeBitForCausalLM,
    config: EdgeBitConfig,
    batch_size: int,
    seq_len: int,
    n_iterations: int = 20,
    warmup: int = 3,
    device: str = "cpu",
) -> ThroughputResult:
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=dev)

    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids=input_ids)

    gc.collect()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(n_iterations):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids=input_ids)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    latencies_np = np.array(latencies)
    total_time = latencies_np.sum() / 1000
    total_tokens = batch_size * seq_len * n_iterations

    mem_mb = 0.0
    if dev.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        try:
            import psutil
            mem_mb = psutil.Process().memory_info().rss / 1e6
        except ImportError:
            pass

    return ThroughputResult(
        batch_size=batch_size,
        seq_len=seq_len,
        quant_mode=config.quant_mode,
        n_iterations=n_iterations,
        total_tokens=total_tokens,
        total_time_s=round(total_time, 4),
        tokens_per_sec=round(total_tokens / total_time, 2),
        latency_p50_ms=round(float(np.percentile(latencies_np, 50)), 2),
        latency_p90_ms=round(float(np.percentile(latencies_np, 90)), 2),
        latency_p99_ms=round(float(np.percentile(latencies_np, 99)), 2),
        mem_mb=round(mem_mb, 2),
    )


def run_sweep(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    config_preset: str = "tiny",
    batch_sizes: Optional[list[int]] = None,
    seq_lengths: Optional[list[int]] = None,
    quant_modes: Optional[list[str]] = None,
    n_iterations: int = 20,
    device: str = "cpu",
) -> list[ThroughputResult]:
    if batch_sizes is None:
        batch_sizes = [1, 2, 4]
    if seq_lengths is None:
        seq_lengths = [128, 256, 512]
    if quant_modes is None:
        quant_modes = ["none", "ternary"]

    results = []

    for qm in quant_modes:
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

        config.quant_mode = qm
        model = EdgeBitForCausalLM(config)

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            sd = state.get("model_state_dict", state)
            model.load_state_dict(sd)

        model.eval()

        for bs in batch_sizes:
            for sl in seq_lengths:
                logger.info("Benchmarking: qm=%s bs=%d sl=%d", qm, bs, sl)
                try:
                    r = measure_throughput(
                        model, config, bs, sl,
                        n_iterations=n_iterations, device=device,
                    )
                    results.append(r)
                    logger.info("  -> %.0f tok/s, p50=%.1fms, mem=%.1fMB",
                                r.tokens_per_sec, r.latency_p50_ms, r.mem_mb)
                except RuntimeError as e:
                    logger.warning("  -> OOM or error: %s", e)

        del model
        gc.collect()

    return results


def format_results(results: list[ThroughputResult]) -> str:
    lines = [
        f"{'Quant':<10} {'BS':<5} {'SeqLen':<8} {'Tok/s':<12} "
        f"{'P50ms':<10} {'P90ms':<10} {'P99ms':<10} {'Mem MB':<10}",
        "-" * 80,
    ]
    for r in results:
        lines.append(
            f"{r.quant_mode:<10} {r.batch_size:<5} {r.seq_len:<8} "
            f"{r.tokens_per_sec:>10.1f} {r.latency_p50_ms:>8.1f} "
            f"{r.latency_p90_ms:>8.1f} {r.latency_p99_ms:>8.1f} "
            f"{r.mem_mb:>8.1f}"
        )
    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="EdgeBit Token Throughput Benchmark")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--preset", type=str, default="tiny")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--seq_lengths", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--quant_modes", nargs="+", default=["none", "ternary"])
    parser.add_argument("--n_iterations", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    results = run_sweep(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        config_preset=args.preset,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        quant_modes=args.quant_modes,
        n_iterations=args.n_iterations,
        device=args.device,
    )

    print("\n" + format_results(results))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
