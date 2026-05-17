"""Tests for packed EdgeBit checkpoint storage."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM
from runtime.packed_checkpoint import (
    estimate_packed_state_size_bytes,
    pack_edgebit_state_dict,
    unpack_edgebit_state_dict,
)


def test_packed_checkpoint_roundtrip_loads_into_model():
    config = EdgeBitConfig(
        vocab_size=128,
        hidden_dim=32,
        ffn_dim=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        max_seq_len=32,
        group_size=8,
    )
    model = EdgeBitForCausalLM(config)
    packed = pack_edgebit_state_dict(model.state_dict(), group_size=config.group_size)
    unpacked = unpack_edgebit_state_dict(packed)

    reloaded = EdgeBitForCausalLM(config)
    reloaded.load_state_dict(unpacked, strict=True)

    x = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        out = reloaded(input_ids=x)
    assert out["logits"].shape == (1, 8, config.vocab_size)


def test_packed_checkpoint_is_smaller_for_bitlinear_heavy_state():
    state = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
        "model.embed_tokens.weight": torch.randn(128, 128),
        "model.norm.weight": torch.randn(128),
    }
    packed = pack_edgebit_state_dict(state, group_size=32)
    packed_bytes = estimate_packed_state_size_bytes(packed)
    original_bytes = sum(t.numel() * t.element_size() for t in state.values())
    assert packed_bytes < original_bytes
