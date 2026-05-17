"""Small offline tokenizer for smoke tests and CI.

This tokenizer is intentionally simple: it hashes whitespace-delimited tokens
into a fixed vocabulary so training and export smoke tests do not require a
network download from HuggingFace.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import torch


class SimpleHashTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2

    def __init__(self, vocab_size: int = 4096):
        if vocab_size < 128:
            raise ValueError("vocab_size must be at least 128")
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.vocab_size

    def _token_to_id(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        value = int.from_bytes(digest, "little")
        return 3 + (value % (self.vocab_size - 3))

    def encode(self, text: str, return_tensors: str | None = None):
        tokens = re.findall(r"\w+|[^\w\s]", text.lower())
        ids = [self._token_to_id(tok) for tok in tokens] or [self.eos_token_id]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        if skip_special_tokens:
            ids = [i for i in ids if i not in {self.pad_token_id, self.eos_token_id}]
        return " ".join(f"tok{i}" for i in ids)

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = 2048,
        padding: str = "max_length",
        return_tensors: str | None = None,
    ) -> dict[str, torch.Tensor]:
        ids = self.encode(text)
        if truncation:
            ids = ids[:max_length]
        attention = [1] * len(ids)

        if padding == "max_length" and len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_len
            attention = attention + [0] * pad_len

        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.tensor(attention, dtype=torch.long)
        if return_tensors == "pt":
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "simple_tokenizer.json"), "w") as f:
            json.dump({"type": "SimpleHashTokenizer", "vocab_size": self.vocab_size}, f, indent=2)
