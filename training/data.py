"""Data loading utilities for EdgeBit training.

Supports:
  - JSONL pretraining data ({"text": "..."})
  - JSONL instruction data ({"instruction": "...", "response": "..."})
  - Streaming for large datasets
  - Efficient tokenization with packing
"""
from __future__ import annotations

import json
import os
import logging
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


class PretrainingDataset(Dataset):
    """Memory-mapped JSONL dataset for pretraining.

    Each line is {"text": "..."} format. Tokenizes and packs
    sequences to max_length for efficient training.
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[str] = []

        logger.info("Loading pretraining data from %s", path)
        with open(path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                obj = json.loads(line.strip())
                text = obj.get("text", "")
                if len(text) > 20:
                    self.samples.append(text)

        logger.info("Loaded %d samples", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class StreamingPretrainingDataset(IterableDataset):
    """Streaming JSONL dataset for large pretraining corpora.

    Reads files lazily to avoid loading entire dataset into memory.
    Supports multiple JSONL files via directory path.
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if os.path.isdir(path):
            self.files = sorted(
                os.path.join(path, f) for f in os.listdir(path)
                if f.endswith(".jsonl")
            )
        else:
            self.files = [path]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        files = self.files
        if worker_info is not None:
            per_worker = len(files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(files)
            files = files[start:end]

        for filepath in files:
            with open(filepath) as f:
                for line in f:
                    obj = json.loads(line.strip())
                    text = obj.get("text", "")
                    if len(text) < 20:
                        continue

                    enc = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.max_length,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    input_ids = enc["input_ids"].squeeze(0)
                    attention_mask = enc["attention_mask"].squeeze(0)
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100
                    yield {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }


class InstructionDataset(Dataset):
    """JSONL instruction-following dataset for SFT/distillation.

    Format: {"instruction": "...", "response": "..."}
    or:     {"text": "<formatted prompt+response>"}
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[str] = []

        with open(path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                obj = json.loads(line.strip())
                if "text" in obj:
                    self.samples.append(obj["text"])
                elif "instruction" in obj and "response" in obj:
                    text = f"### Instruction:\n{obj['instruction']}\n\n### Response:\n{obj['response']}"
                    self.samples.append(text)

        logger.info("Loaded %d instruction samples from %s", len(self.samples), path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_synthetic_data(path: str, n_samples: int = 1000, min_len: int = 50,
                          max_len: int = 500) -> None:
    """Generate synthetic JSONL data for smoke testing."""
    import random
    import string

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "machine", "learning", "model", "training", "data", "edge",
             "deployment", "inference", "quantization", "transformer", "attention",
             "neural", "network", "optimization", "gradient", "weight", "bias",
             "layer", "hidden", "embedding", "token", "sequence", "batch"]

    with open(path, "w") as f:
        for _ in range(n_samples):
            length = random.randint(min_len, max_len)
            text = " ".join(random.choices(words, k=length))
            f.write(json.dumps({"text": text}) + "\n")

    logger.info("Generated %d synthetic samples at %s", n_samples, path)
