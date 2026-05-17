"""Tests for the offline smoke-test tokenizer."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.simple_tokenizer import SimpleHashTokenizer


def test_simple_tokenizer_call_shapes():
    tokenizer = SimpleHashTokenizer(vocab_size=256)
    enc = tokenizer(
        "edge quantization test",
        max_length=8,
        padding="max_length",
        return_tensors="pt",
    )
    assert enc["input_ids"].shape == (1, 8)
    assert enc["attention_mask"].shape == (1, 8)
    assert enc["input_ids"].max() < len(tokenizer)
    assert enc["attention_mask"].sum() == 3


def test_simple_tokenizer_encode_tensor():
    tokenizer = SimpleHashTokenizer(vocab_size=256)
    ids = tokenizer.encode("hello edge", return_tensors="pt")
    assert isinstance(ids, torch.Tensor)
    assert ids.ndim == 2
