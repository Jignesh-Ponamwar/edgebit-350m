"""Tests for ternary weight packing/unpacking."""
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.pack_ternary import (
    pack_ternary_weights,
    unpack_ternary_weights,
    compute_packing_stats,
    pack_model_weights,
)


class TestPackTernary:
    def test_roundtrip_exact(self):
        w = torch.tensor([[-1, 0, 1, 0, -1, 1, 0, 0]], dtype=torch.float)
        packed = pack_ternary_weights(w, group_size=8)
        unpacked = unpack_ternary_weights(packed)
        assert torch.allclose(unpacked, w, atol=1e-3)

    def test_roundtrip_larger(self):
        w = torch.zeros(64, 128)
        w[::3] = 1
        w[1::3] = -1
        packed = pack_ternary_weights(w, group_size=128)
        unpacked = unpack_ternary_weights(packed)
        error = (unpacked.sign() - w.sign()).abs().mean()
        assert error < 0.01

    def test_packed_size(self):
        w = torch.zeros(256, 512)
        packed = pack_ternary_weights(w, group_size=128)
        n_bytes = packed["packed"].numel()
        expected = (256 * 512) // 4
        assert n_bytes == expected

    def test_scales_shape(self):
        w = torch.zeros(128, 256)
        packed = pack_ternary_weights(w, group_size=128)
        n_groups = (128 * 256) // 128
        assert packed["scales"].numel() == n_groups

    def test_metadata_preserved(self):
        w = torch.randn(32, 64).round().clamp(-1, 1)
        packed = pack_ternary_weights(w, group_size=64)
        assert tuple(packed["shape"].tolist()) == (32, 64)
        assert packed["numel"].item() == 32 * 64
        assert packed["group_size"].item() == 64


class TestPackingStats:
    def test_compression_ratio(self):
        w = torch.randn(256, 512)
        stats = compute_packing_stats(w)
        assert stats["compression_ratio"] > 7
        assert stats["bits_per_param"] < 3

    def test_correct_sizes(self):
        w = torch.randn(100, 200)
        stats = compute_packing_stats(w)
        assert stats["numel"] == 20000
        assert stats["fp32_mb"] == pytest.approx(20000 * 4 / 1e6)
        assert stats["fp16_mb"] == pytest.approx(20000 * 2 / 1e6)


class TestPackModelWeights:
    def test_packs_eligible_layers(self):
        state = {
            "layer.0.weight": torch.zeros(64, 128),
            "layer.0.bias": torch.zeros(64),
            "embed.weight": torch.zeros(1000, 64),
            "norm.weight": torch.zeros(64),
        }
        packed = pack_model_weights(state, group_size=128)
        assert isinstance(packed["layer.0.weight"], dict)
        assert "packed" in packed["layer.0.weight"]
        assert isinstance(packed["layer.0.bias"], torch.Tensor)
        assert isinstance(packed["embed.weight"], torch.Tensor)
        assert isinstance(packed["norm.weight"], torch.Tensor)
