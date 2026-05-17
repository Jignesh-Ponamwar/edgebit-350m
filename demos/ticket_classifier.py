#!/usr/bin/env python3
"""Support ticket classification demo.

Classifies support tickets into categories using EdgeBit model
with few-shot prompting. Demonstrates practical edge AI deployment
for customer service automation.

Usage:
    python demos/ticket_classifier.py --preset tiny
    python demos/ticket_classifier.py --checkpoint /path/to/ckpt

Categories:
    - billing: Payment, invoice, subscription issues
    - technical: Bugs, errors, crashes
    - account: Login, password, profile issues
    - feature: Feature requests, suggestions
    - general: Other inquiries
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM

CATEGORIES = ["billing", "technical", "account", "feature", "general"]

FEW_SHOT_TEMPLATE = """Classify the following support ticket into one of these categories: billing, technical, account, feature, general.

Ticket: "I can't log into my account, it says my password is wrong"
Category: account

Ticket: "My invoice shows the wrong amount for last month"
Category: billing

Ticket: "The app crashes when I try to upload a file larger than 10MB"
Category: technical

Ticket: "It would be great if you could add dark mode to the app"
Category: feature

Ticket: "{ticket}"
Category:"""

SAMPLE_TICKETS = [
    "I was charged twice for my subscription this month",
    "The search feature returns no results when I use special characters",
    "Can you add support for exporting data to CSV format?",
    "I need to update my email address but can't find the setting",
    "Your service has been great, just wanted to say thanks!",
    "My payment failed but the order still went through",
    "The mobile app is very slow on Android 14",
    "How do I reset my two-factor authentication?",
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


def classify_ticket(
    model: EdgeBitForCausalLM,
    tokenizer,
    ticket: str,
    temperature: float = 0.1,
) -> tuple[str, float, float]:
    prompt = FEW_SHOT_TEMPLATE.format(ticket=ticket)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=5, temperature=temperature, top_k=10,
        )
    elapsed = (time.perf_counter() - t0) * 1000

    generated = output[0, input_ids.shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip().lower()

    category = "general"
    for cat in CATEGORIES:
        if cat in response:
            category = cat
            break

    with torch.no_grad():
        logits = model(input_ids=input_ids)["logits"][0, -1, :]
        probs = torch.softmax(logits / max(temperature, 0.01), dim=-1)
        confidence = probs.max().item()

    return category, confidence, elapsed


def run_batch(model, tokenizer, tickets: list[str], temperature: float = 0.1):
    print(f"\n{'Ticket':<60} {'Category':<12} {'Conf':<8} {'Time':<10}")
    print("-" * 90)

    results = []
    for ticket in tickets:
        cat, conf, elapsed = classify_ticket(model, tokenizer, ticket, temperature)
        display = ticket[:57] + "..." if len(ticket) > 57 else ticket
        print(f"{display:<60} {cat:<12} {conf:.3f}    {elapsed:.0f}ms")
        results.append({"ticket": ticket, "category": cat, "confidence": conf, "latency_ms": elapsed})

    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    print(f"\nAverage latency: {avg_latency:.0f}ms")
    return results


def main():
    parser = argparse.ArgumentParser(description="EdgeBit Ticket Classifier")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--preset", type=str, default="tiny")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    print("Loading model...")
    model, config, tokenizer = load_model(args)
    params = model.count_parameters()
    print(f"Model: {params['total']:,} params ({config.quant_mode} mode)")

    if args.input_file:
        with open(args.input_file) as f:
            tickets = [json.loads(line)["text"] for line in f if line.strip()]
        results = run_batch(model, tokenizer, tickets, args.temperature)
    elif args.interactive:
        print("\nEnter tickets to classify (one per line). Type 'quit' to exit.\n")
        while True:
            try:
                ticket = input("Ticket: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not ticket or ticket.lower() in ("quit", "exit"):
                break
            cat, conf, elapsed = classify_ticket(model, tokenizer, ticket, args.temperature)
            print(f"  -> {cat} (confidence: {conf:.3f}, {elapsed:.0f}ms)\n")
    else:
        print("\nRunning on sample tickets:")
        results = run_batch(model, tokenizer, SAMPLE_TICKETS, args.temperature)

        if args.output_file:
            os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
