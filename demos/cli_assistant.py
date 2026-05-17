#!/usr/bin/env python3
"""Interactive CLI assistant demo for EdgeBit models.

Usage:
    python demos/cli_assistant.py --preset tiny
    python demos/cli_assistant.py --checkpoint /path/to/ckpt --config configs/model_350m.yaml

Features:
    - Interactive multi-turn conversation
    - Token-by-token streaming output
    - Configurable generation parameters
    - Memory and latency reporting
"""
from __future__ import annotations

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM


def load_model(args) -> tuple[EdgeBitForCausalLM, EdgeBitConfig, object]:
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)
        config = EdgeBitConfig(**cfg_dict.get("model", cfg_dict))
    else:
        preset_fn = getattr(EdgeBitConfig, args.preset, None)
        config = preset_fn() if preset_fn else EdgeBitConfig.tiny()

    config.vocab_size = 151936
    model = EdgeBitForCausalLM(config)

    if args.checkpoint:
        ckpt_file = args.checkpoint
        if os.path.isdir(ckpt_file):
            ckpt_file = os.path.join(ckpt_file, "training_state.pt")
        state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        sd = state.get("model_state_dict", state)
        model.load_state_dict(sd)

    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, config, tokenizer


def generate_streaming(
    model: EdgeBitForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
) -> tuple[str, float, int]:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    t0 = time.perf_counter()
    generated = []

    with torch.no_grad():
        cur_ids = input_ids
        for _ in range(max_new_tokens):
            logits = model(input_ids=cur_ids)["logits"][:, -1, :]

            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    topk_vals, topk_idx = torch.topk(logits, top_k)
                    logits_filtered = torch.full_like(logits, float("-inf"))
                    logits_filtered.scatter_(1, topk_idx, topk_vals)
                    probs = torch.softmax(logits_filtered, dim=-1)
                    next_id = torch.multinomial(probs, 1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, 1)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)

            token_str = tokenizer.decode(next_id[0], skip_special_tokens=True)
            print(token_str, end="", flush=True)
            generated.append(next_id.item())

            if next_id.item() == tokenizer.eos_token_id:
                break

            cur_ids = torch.cat([cur_ids, next_id], dim=-1)

    elapsed = time.perf_counter() - t0
    print()
    return tokenizer.decode(generated, skip_special_tokens=True), elapsed, len(generated)


def main():
    parser = argparse.ArgumentParser(description="EdgeBit CLI Assistant")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--preset", type=str, default="tiny")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    print("Loading model...")
    model, config, tokenizer = load_model(args)
    params = model.count_parameters()
    print(f"Model: {params['total']:,} params ({config.quant_mode} mode)")
    print(f"Type 'quit' or 'exit' to stop. Type 'stats' for session stats.\n")

    total_tokens = 0
    total_time = 0.0
    turn = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "stats":
            avg_tps = total_tokens / total_time if total_time > 0 else 0
            print(f"  Turns: {turn}")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Avg throughput: {avg_tps:.1f} tok/s")
            try:
                import psutil
                mem = psutil.Process().memory_info().rss / 1e6
                print(f"  Memory: {mem:.1f} MB")
            except ImportError:
                pass
            continue

        print("Assistant: ", end="", flush=True)
        response, elapsed, n_tokens = generate_streaming(
            model, tokenizer, user_input,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        tps = n_tokens / elapsed if elapsed > 0 else 0
        print(f"  [{n_tokens} tokens, {elapsed:.2f}s, {tps:.1f} tok/s]\n")

        total_tokens += n_tokens
        total_time += elapsed
        turn += 1


if __name__ == "__main__":
    main()
