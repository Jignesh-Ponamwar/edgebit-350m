#!/usr/bin/env python3
"""Evaluation harness for EdgeBit models on standard benchmarks.

Wraps lm-evaluation-harness for:
  - MMLU (5-shot)
  - HellaSwag (10-shot)
  - GSM8K (5-shot, chain-of-thought)
  - Winogrande (5-shot)
  - ARC-Easy / ARC-Challenge (25-shot)

Also supports internal perplexity evaluation on custom datasets.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import math
from typing import Optional

import torch
from torch.utils.data import DataLoader

from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM

logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = "cpu",
) -> tuple[EdgeBitForCausalLM, EdgeBitConfig]:
    if config_path:
        import yaml
        with open(config_path) as f:
            cfg_dict = yaml.safe_load(f)
        config = EdgeBitConfig(**cfg_dict.get("model", cfg_dict))
    else:
        config_json = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_json):
            with open(config_json) as f:
                cfg_dict = json.load(f)
            config = EdgeBitConfig(**(cfg_dict.get("edgebit_config", cfg_dict)))
        else:
            config = EdgeBitConfig()

    model = EdgeBitForCausalLM(config)

    state_path = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
    else:
        for fname in os.listdir(checkpoint_path):
            if fname.endswith((".pt", ".bin", ".safetensors")):
                if fname.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state = load_file(os.path.join(checkpoint_path, fname))
                else:
                    state = torch.load(
                        os.path.join(checkpoint_path, fname),
                        map_location="cpu", weights_only=False,
                    )
                    if "model_state_dict" in state:
                        state = state["model_state_dict"]
                model.load_state_dict(state)
                break

    model = model.to(device)
    model.eval()
    return model, config


def eval_perplexity(
    model: EdgeBitForCausalLM,
    data_path: str,
    tokenizer,
    max_length: int = 2048,
    max_samples: int = 500,
    batch_size: int = 1,
    device: str = "cpu",
) -> dict[str, float]:
    from training.data import PretrainingDataset

    dataset = PretrainingDataset(data_path, tokenizer, max_length=max_length, max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs["loss"]

            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))

    return {
        "perplexity": round(perplexity, 4),
        "avg_loss": round(avg_loss, 6),
        "total_tokens": total_tokens,
        "n_samples": len(dataset),
    }


def run_lm_eval_harness(
    checkpoint_path: str,
    tasks: list[str],
    config_path: Optional[str] = None,
    num_fewshot: Optional[int] = None,
    device: str = "cpu",
    batch_size: int = 1,
    output_path: Optional[str] = None,
) -> dict:
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logger.error(
            "lm-evaluation-harness not installed. "
            "Install with: pip install lm-eval"
        )
        return {"error": "lm-eval not installed"}

    model, config = load_model(checkpoint_path, config_path, device)

    task_configs = {
        "mmlu": {"num_fewshot": 5},
        "hellaswag": {"num_fewshot": 10},
        "gsm8k": {"num_fewshot": 5},
        "winogrande": {"num_fewshot": 5},
        "arc_easy": {"num_fewshot": 25},
        "arc_challenge": {"num_fewshot": 25},
    }

    results = {}
    for task in tasks:
        fewshot = num_fewshot if num_fewshot is not None else task_configs.get(task, {}).get("num_fewshot", 0)
        logger.info("Evaluating %s (%d-shot)...", task, fewshot)

        try:
            task_results = lm_eval.simple_evaluate(
                model=model,
                tasks=[task],
                num_fewshot=fewshot,
                batch_size=batch_size,
                device=device,
            )
            results[task] = task_results.get("results", {}).get(task, {})
            logger.info("%s: %s", task, json.dumps(results[task], indent=2))
        except Exception as e:
            logger.error("Failed to evaluate %s: %s", task, e)
            results[task] = {"error": str(e)}

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", output_path)

    return results


def format_results(results: dict) -> str:
    lines = [
        f"{'Benchmark':<20} {'Metric':<20} {'Score':<10}",
        "-" * 50,
    ]
    for task, metrics in results.items():
        if isinstance(metrics, dict) and "error" not in metrics:
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    lines.append(f"{task:<20} {metric_name:<20} {value:>8.4f}")
        elif isinstance(metrics, dict) and "error" in metrics:
            lines.append(f"{task:<20} {'ERROR':<20} {metrics['error']}")
    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="EdgeBit LM Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tasks", nargs="+", default=["mmlu", "hellaswag", "gsm8k", "winogrande"])
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)

    sub = parser.add_subparsers(dest="mode")
    ppl = sub.add_parser("perplexity")
    ppl.add_argument("--data_path", type=str, required=True)
    ppl.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B")
    ppl.add_argument("--max_samples", type=int, default=500)

    args = parser.parse_args()

    if args.mode == "perplexity":
        model, config = load_model(args.checkpoint, args.config, args.device)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        results = eval_perplexity(
            model, args.data_path, tokenizer,
            max_samples=args.max_samples, device=args.device,
        )
        print(json.dumps(results, indent=2))
    else:
        results = run_lm_eval_harness(
            checkpoint_path=args.checkpoint,
            tasks=args.tasks,
            config_path=args.config,
            num_fewshot=args.num_fewshot,
            device=args.device,
            batch_size=args.batch_size,
            output_path=args.output,
        )
        print(format_results(results))


if __name__ == "__main__":
    main()
