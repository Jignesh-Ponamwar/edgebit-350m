#!/usr/bin/env python3
"""Minimal edge deployment example.

A self-contained script demonstrating EdgeBit deployment on
resource-constrained devices. Includes model loading, memory
reporting, warmup, and benchmarking in a single file.

Designed for copy-paste deployment on Raspberry Pi 5, IoT devices,
or any CPU-only environment.

Usage:
    # Quick test with random weights:
    python demos/edge_deploy.py

    # With trained checkpoint:
    python demos/edge_deploy.py --checkpoint /path/to/ckpt

    # Raspberry Pi optimized:
    OMP_NUM_THREADS=4 python demos/edge_deploy.py --preset tiny
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def get_memory_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="EdgeBit Edge Deployment")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--preset", type=str, default="tiny")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", type=str, default="The future of edge AI is")
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--benchmark_rounds", type=int, default=3)
    args = parser.parse_args()

    print("=" * 60)
    print("EdgeBit Edge Deployment")
    print("=" * 60)

    mem_start = get_memory_mb()
    print(f"\n[Memory] Baseline: {mem_start:.1f} MB")

    print("\n[1/5] Loading model configuration...")
    from modeling.config import EdgeBitConfig
    from modeling.model import EdgeBitForCausalLM

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)
        config = EdgeBitConfig(**cfg_dict.get("model", cfg_dict))
    else:
        preset_fn = getattr(EdgeBitConfig, args.preset, None)
        config = preset_fn() if preset_fn else EdgeBitConfig.tiny()

    print(f"  Config: {config.hidden_dim}h, {config.n_layers}L, {config.n_heads}H")
    print(f"  Estimated params: {config.n_params_estimate:,}")

    print("\n[2/5] Loading model weights...")
    t0 = time.perf_counter()
    model = EdgeBitForCausalLM(config)

    if args.checkpoint:
        ckpt_file = args.checkpoint
        if os.path.isdir(ckpt_file):
            ckpt_file = os.path.join(ckpt_file, "training_state.pt")
        state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
        print(f"  Loaded checkpoint: {ckpt_file}")
    else:
        print("  Using random weights (no checkpoint)")

    model.eval()
    load_time = time.perf_counter() - t0
    mem_model = get_memory_mb()
    params = model.count_parameters()
    print(f"  Params: {params['total']:,} total, {params['bitlinear']:,} BitLinear")
    print(f"  Load time: {load_time:.2f}s")
    print(f"  [Memory] After model load: {mem_model:.1f} MB (+{mem_model - mem_start:.1f} MB)")

    print("\n[3/5] Loading tokenizer...")
    if args.tokenizer == "simple":
        from training.simple_tokenizer import SimpleHashTokenizer
        tokenizer = SimpleHashTokenizer(vocab_size=config.vocab_size)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if len(tokenizer) != config.vocab_size:
            print(
                f"  Warning: tokenizer vocab ({len(tokenizer):,}) does not match "
                f"model vocab ({config.vocab_size:,})"
            )
    print(f"  Tokenizer: {args.tokenizer} (vocab={len(tokenizer):,})")

    print("\n[4/5] Warmup inference...")
    dummy = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        for _ in range(2):
            model(input_ids=dummy)
    mem_warmup = get_memory_mb()
    print(f"  [Memory] After warmup: {mem_warmup:.1f} MB (+{mem_warmup - mem_model:.1f} MB)")

    print(f"\n[5/5] Generation benchmark ({args.benchmark_rounds} rounds)...")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    prompt_len = input_ids.shape[1]
    print(f"  Prompt: \"{args.prompt}\" ({prompt_len} tokens)")
    print(f"  Generating: {args.max_tokens} tokens\n")

    latencies = []
    last_output = ""

    for r in range(args.benchmark_rounds):
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=args.max_tokens, temperature=0.7, top_k=50,
            )
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        generated = output[0, prompt_len:]
        last_output = tokenizer.decode(generated, skip_special_tokens=True)
        n_tokens = len(generated)
        tps = n_tokens / elapsed if elapsed > 0 else 0
        print(f"  Round {r+1}: {n_tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")

    mem_final = get_memory_mb()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  Generated text: \"{last_output[:100]}{'...' if len(last_output) > 100 else ''}\"")
    print(f"\n  Model: EdgeBit-{args.preset} ({params['total']:,} params)")
    print(f"  Quant mode: {config.quant_mode}")
    print(f"  Avg latency: {sum(latencies)/len(latencies):.2f}s")
    avg_tps = args.max_tokens / (sum(latencies) / len(latencies))
    print(f"  Avg throughput: {avg_tps:.1f} tok/s")
    print(f"  Peak memory: {mem_final:.1f} MB")
    print(f"  Model load: {load_time:.2f}s")

    print("\n  Device suitability:")
    if mem_final < 200:
        print("    Raspberry Pi 5 (2GB): SUITABLE")
    elif mem_final < 500:
        print("    Raspberry Pi 5 (4GB): SUITABLE")
    elif mem_final < 1000:
        print("    Laptop (8GB): SUITABLE")
    else:
        print("    Cloud CPU recommended")

    print()


if __name__ == "__main__":
    main()
