#!/usr/bin/env python3
"""Short text summarization demo.

Generates concise summaries of input text using EdgeBit model
with instruction prompting. Designed for edge deployment where
latency and memory are constrained.

Usage:
    python demos/summarizer.py --preset tiny
    python demos/summarizer.py --checkpoint /path/to/ckpt --text "Your text here"
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM

SUMMARIZE_PROMPT = """### Instruction:
Summarize the following text in 1-2 sentences.

### Input:
{text}

### Summary:
"""

SAMPLE_TEXTS = [
    (
        "Machine learning models are increasingly being deployed on edge devices "
        "such as smartphones, IoT sensors, and embedded systems. These devices "
        "typically have limited computational resources, including restricted memory "
        "and processing power. To address these constraints, researchers have developed "
        "various model compression techniques, including quantization, pruning, and "
        "knowledge distillation. Quantization reduces the precision of model weights "
        "and activations from 32-bit floating point to lower bit-widths, significantly "
        "reducing model size and inference latency."
    ),
    (
        "The Raspberry Pi 5 features a Broadcom BCM2712 SoC with a quad-core "
        "ARM Cortex-A76 processor clocked at 2.4GHz. It comes with either 4GB or "
        "8GB of LPDDR4X RAM. The board includes dual 4Kp60 HDMI display output, "
        "a PCIe 2.0 interface, and USB 3.0 ports. It is significantly faster than "
        "its predecessor, the Raspberry Pi 4, with roughly 2-3x improvement in "
        "single-threaded CPU performance."
    ),
    (
        "Transformers have become the dominant architecture for natural language "
        "processing tasks. The self-attention mechanism allows the model to weigh "
        "the importance of different words in a sequence when making predictions. "
        "However, the quadratic complexity of self-attention with respect to sequence "
        "length remains a significant challenge for processing long documents."
    ),
]


def load_model(args):
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
        model.load_state_dict(state.get("model_state_dict", state))

    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, config, tokenizer


def summarize(
    model: EdgeBitForCausalLM,
    tokenizer,
    text: str,
    max_new_tokens: int = 64,
    temperature: float = 0.3,
) -> tuple[str, float, int]:
    prompt = SUMMARIZE_PROMPT.format(text=text)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=30,
        )
    elapsed = time.perf_counter() - t0

    generated = output[0, input_ids.shape[1]:]
    summary = tokenizer.decode(generated, skip_special_tokens=True).strip()

    n_tokens = len(generated)
    return summary, elapsed, n_tokens


def main():
    parser = argparse.ArgumentParser(description="EdgeBit Text Summarizer")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--preset", type=str, default="tiny")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    print("Loading model...")
    model, config, tokenizer = load_model(args)
    params = model.count_parameters()
    print(f"Model: {params['total']:,} params ({config.quant_mode} mode)\n")

    if args.text:
        texts = [args.text]
    elif args.input_file:
        with open(args.input_file) as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = SAMPLE_TEXTS

    for i, text in enumerate(texts):
        display = text[:100] + "..." if len(text) > 100 else text
        print(f"--- Input {i+1} ---")
        print(f"{display}\n")

        summary, elapsed, n_tokens = summarize(
            model, tokenizer, text,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        tps = n_tokens / elapsed if elapsed > 0 else 0
        print(f"Summary: {summary}")
        print(f"  [{n_tokens} tokens, {elapsed:.2f}s, {tps:.1f} tok/s]\n")


if __name__ == "__main__":
    main()
