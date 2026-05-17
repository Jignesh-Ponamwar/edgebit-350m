"""Tests for BitLinear layer and quantization utilities."""
import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.bitlinear import BitLinear
from modeling.quant_utils import (
    ternary_quantize_absmean,
    int8_symmetric_quantize,
    nf4_quantize,
    nf4_dequantize,
    ste_round,
    ste_clamp,
)


class TestSTE:
    def test_ste_round_forward(self):
        x = torch.tensor([0.3, 0.7, 1.4, -0.6])
        y = ste_round(x)
        assert torch.equal(y, torch.tensor([0.0, 1.0, 1.0, -1.0]))

    def test_ste_round_gradient(self):
        x = torch.tensor([0.3, 0.7, 1.4], requires_grad=True)
        y = ste_round(x)
        y.sum().backward()
        assert torch.equal(x.grad, torch.ones(3))

    def test_ste_clamp_forward(self):
        x = torch.tensor([-2.0, -0.5, 0.5, 2.0])
        y = ste_clamp(x, -1.0, 1.0)
        assert torch.equal(y, torch.tensor([-1.0, -0.5, 0.5, 1.0]))

    def test_ste_clamp_gradient(self):
        x = torch.tensor([-2.0, 0.5, 2.0], requires_grad=True)
        y = ste_clamp(x, -1.0, 1.0)
        y.sum().backward()
        assert torch.equal(x.grad, torch.ones(3))


class TestTernaryQuantize:
    def test_output_values(self):
        w = torch.randn(128, 256)
        q, scale = ternary_quantize_absmean(w, group_size=128)
        unique = q.unique()
        assert all(v in [-1, 0, 1] for v in unique.tolist())

    def test_scale_positive(self):
        w = torch.randn(64, 128)
        _, scale = ternary_quantize_absmean(w, group_size=64)
        assert (scale > 0).all()

    def test_shape_preserved(self):
        w = torch.randn(32, 64)
        q, scale = ternary_quantize_absmean(w, group_size=32)
        assert q.shape == w.shape

    def test_gradient_flows(self):
        w = torch.randn(32, 64, requires_grad=True)
        q, scale = ternary_quantize_absmean(w, group_size=32)
        loss = q.sum() + scale.sum()
        loss.backward()
        assert w.grad is not None
        assert w.grad.shape == w.shape


class TestINT8Quantize:
    def test_output_range(self):
        x = torch.randn(4, 128)
        q, scale = int8_symmetric_quantize(x)
        assert q.abs().max() <= 127

    def test_dequantize_close(self):
        x = torch.randn(4, 128) * 2
        q, scale = int8_symmetric_quantize(x, per_token=True)
        recon = q.float() / scale
        rel_error = (recon - x).abs().mean() / x.abs().mean()
        assert rel_error < 0.15


class TestNF4:
    def test_roundtrip(self):
        w = torch.randn(128)
        codes, absmax = nf4_quantize(w, block_size=64)
        recon = nf4_dequantize(codes, absmax, block_size=64)
        assert recon.shape == w.shape
        rel_error = (recon - w).abs().mean() / w.abs().mean()
        assert rel_error < 0.3

    def test_codes_in_range(self):
        w = torch.randn(256)
        codes, _ = nf4_quantize(w, block_size=64)
        assert codes.min() >= 0
        assert codes.max() <= 15


class TestBitLinear:
    def test_forward_shape(self):
        bl = BitLinear(64, 128, quant_mode="ternary")
        x = torch.randn(2, 16, 64)
        y = bl(x)
        assert y.shape == (2, 16, 128)

    def test_forward_none_mode(self):
        bl = BitLinear(64, 128, quant_mode="none")
        x = torch.randn(2, 16, 64)
        y = bl(x)
        assert y.shape == (2, 16, 128)

    def test_forward_int8_mode(self):
        bl = BitLinear(64, 128, quant_mode="int8")
        x = torch.randn(2, 16, 64)
        y = bl(x)
        assert y.shape == (2, 16, 128)

    def test_gradient_flows(self):
        bl = BitLinear(32, 64, quant_mode="ternary")
        x = torch.randn(1, 8, 32, requires_grad=True)
        y = bl(x)
        y.sum().backward()
        assert x.grad is not None
        assert bl.weight.grad is not None

    def test_set_quant_mode(self):
        bl = BitLinear(32, 64, quant_mode="none")
        bl.set_quant_mode("ternary")
        assert bl.quant_mode == "ternary"
        x = torch.randn(1, 8, 32)
        y = bl(x)
        assert y.shape == (1, 8, 64)

    def test_from_linear(self):
        linear = nn.Linear(64, 128, bias=False)
        bl = BitLinear.from_linear(linear, quant_mode="ternary")
        assert bl.in_features == 64
        assert bl.out_features == 128
        assert torch.equal(bl.weight.data, linear.weight.data)

    def test_bias_support(self):
        bl = BitLinear(32, 64, bias=True, quant_mode="ternary")
        assert bl.bias is not None
        x = torch.randn(1, 8, 32)
        y = bl(x)
        assert y.shape == (1, 8, 64)

    def test_collect_stats(self):
        bl = BitLinear(32, 64, quant_mode="ternary", collect_stats=True)
        x = torch.randn(1, 8, 32)
        bl(x)
        assert bl.stats is not None
        assert 0 <= bl.stats.pct_zero <= 1
        assert 0 <= bl.stats.pct_plus_one <= 1
        assert 0 <= bl.stats.pct_minus_one <= 1
