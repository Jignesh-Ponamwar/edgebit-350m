"""Tests for EdgeBit model forward pass, generation, and parameter counting."""
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM, EdgeBitModel


@pytest.fixture
def tiny_config():
    config = EdgeBitConfig.tiny()
    config.vocab_size = 1000
    return config


@pytest.fixture
def tiny_model(tiny_config):
    return EdgeBitForCausalLM(tiny_config)


class TestEdgeBitConfig:
    def test_tiny_config(self):
        c = EdgeBitConfig.tiny()
        assert c.hidden_dim == 512
        assert c.n_layers == 8
        assert c.n_heads == 8

    def test_small_config(self):
        c = EdgeBitConfig.small_125m()
        assert c.hidden_dim == 768
        assert c.n_layers == 12

    def test_default_config(self):
        c = EdgeBitConfig()
        assert c.hidden_dim == 1024
        assert c.n_layers == 24
        assert c.n_heads == 16
        assert c.n_kv_heads == 4

    def test_base_alias(self):
        c1 = EdgeBitConfig.base()
        c2 = EdgeBitConfig.base_350m()
        assert c1.hidden_dim == c2.hidden_dim
        assert c1.n_layers == c2.n_layers

    def test_to_dict_roundtrip(self):
        c = EdgeBitConfig.tiny()
        d = c.to_dict()
        c2 = EdgeBitConfig(**d)
        assert c.hidden_dim == c2.hidden_dim
        assert c.n_layers == c2.n_layers

    def test_n_params_estimate(self):
        c = EdgeBitConfig.tiny()
        est = c.n_params_estimate
        assert est > 30_000_000
        assert est < 100_000_000


class TestEdgeBitModel:
    def test_forward_output_shape(self, tiny_config):
        model = EdgeBitModel(tiny_config)
        x = torch.randint(0, tiny_config.vocab_size, (2, 32))
        out = model(x)
        assert out.shape == (2, 32, tiny_config.hidden_dim)

    def test_forward_dtype(self, tiny_config):
        model = EdgeBitModel(tiny_config)
        x = torch.randint(0, tiny_config.vocab_size, (1, 16))
        out = model(x)
        assert out.dtype == torch.float32


class TestEdgeBitForCausalLM:
    def test_forward_shape(self, tiny_model, tiny_config):
        x = torch.randint(0, tiny_config.vocab_size, (2, 32))
        out = tiny_model(input_ids=x)
        assert out["logits"].shape == (2, 32, tiny_config.vocab_size)

    def test_forward_with_labels(self, tiny_model, tiny_config):
        x = torch.randint(0, tiny_config.vocab_size, (2, 32))
        out = tiny_model(input_ids=x, labels=x)
        assert "loss" in out
        assert out["loss"].ndim == 0
        assert out["loss"].item() > 0

    def test_loss_backward(self, tiny_model, tiny_config):
        x = torch.randint(0, tiny_config.vocab_size, (1, 16))
        out = tiny_model(input_ids=x, labels=x)
        out["loss"].backward()
        has_grad = False
        for p in tiny_model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad

    def test_generate(self, tiny_model, tiny_config):
        x = torch.randint(0, tiny_config.vocab_size, (1, 8))
        with torch.no_grad():
            gen = tiny_model.generate(x, max_new_tokens=10, temperature=1.0)
        assert gen.shape[0] == 1
        assert gen.shape[1] == 18

    def test_generate_greedy(self, tiny_model, tiny_config):
        x = torch.randint(0, tiny_config.vocab_size, (1, 4))
        with torch.no_grad():
            gen = tiny_model.generate(x, max_new_tokens=5, temperature=0.0)
        assert gen.shape == (1, 9)

    def test_greedy_matches_argmax(self, tiny_model, tiny_config):
        tiny_model.eval()
        x = torch.randint(0, tiny_config.vocab_size, (1, 4))
        with torch.no_grad():
            logits = tiny_model(input_ids=x)["logits"][:, -1, :]
            gen = tiny_model.generate(x, max_new_tokens=1, temperature=0.0)
        assert torch.equal(gen[:, -1], logits.argmax(dim=-1))

    def test_kv_cache_matches_full_forward(self, tiny_config):
        from modeling.attention_utils import INT8KVCache

        tiny_config.quant_mode = "none"
        tiny_config.kv_cache_quant = False
        model = EdgeBitForCausalLM(tiny_config)
        model.eval()

        x = torch.randint(0, tiny_config.vocab_size, (1, 6))
        with torch.no_grad():
            full_logits = model(input_ids=x)["logits"]
            cache = INT8KVCache(enabled=False)
            step_logits = []
            for idx in range(x.shape[1]):
                out = model(input_ids=x[:, idx:idx + 1], kv_cache=cache)
                step_logits.append(out["logits"])
            cached_logits = torch.cat(step_logits, dim=1)

        assert torch.allclose(cached_logits, full_logits, atol=1e-4, rtol=1e-4)

    def test_padding_attention_mask_changes_logits(self, tiny_config):
        tiny_config.quant_mode = "none"
        model = EdgeBitForCausalLM(tiny_config)
        model.eval()

        x = torch.randint(1, tiny_config.vocab_size, (1, 8))
        mask = torch.ones_like(x)
        mask[:, -3:] = 0
        with torch.no_grad():
            masked = model(input_ids=x, attention_mask=mask)["logits"]
            unmasked = model(input_ids=x)["logits"]

        assert not torch.allclose(masked[:, -1, :], unmasked[:, -1, :])

    def test_count_parameters(self, tiny_model):
        counts = tiny_model.count_parameters()
        assert "total" in counts
        assert "trainable" in counts
        assert "bitlinear" in counts
        assert counts["total"] > 0
        assert counts["trainable"] == counts["total"]

    def test_set_quant_mode(self, tiny_model):
        tiny_model.set_quant_mode("none")
        x = torch.randint(0, 1000, (1, 8))
        out1 = tiny_model(input_ids=x)

        tiny_model.set_quant_mode("ternary")
        out2 = tiny_model(input_ids=x)

        assert out1["logits"].shape == out2["logits"].shape

    def test_different_seq_lengths(self, tiny_model, tiny_config):
        for sl in [4, 16, 64, 128]:
            x = torch.randint(0, tiny_config.vocab_size, (1, sl))
            out = tiny_model(input_ids=x)
            assert out["logits"].shape == (1, sl, tiny_config.vocab_size)

    def test_batch_consistency(self, tiny_model, tiny_config):
        tiny_model.eval()
        x = torch.randint(0, tiny_config.vocab_size, (1, 16))
        with torch.no_grad():
            out1 = tiny_model(input_ids=x)
            out2 = tiny_model(input_ids=x)
        assert torch.allclose(out1["logits"], out2["logits"], atol=1e-5)


class TestAttentionUtils:
    def test_qk_rmsnorm(self):
        from modeling.attention_utils import QKRMSNorm
        norm = QKRMSNorm(64)
        q = torch.randn(1, 8, 16, 64)
        k = torch.randn(1, 2, 16, 64)
        q_n, k_n = norm(q, k)
        assert q_n.shape == q.shape
        assert k_n.shape == k.shape

    def test_int8_kv_cache(self):
        from modeling.attention_utils import INT8KVCache
        cache = INT8KVCache(enabled=True)
        k = torch.randn(1, 2, 16, 64)
        v = torch.randn(1, 2, 16, 64)
        k_out, v_out = cache.update(0, k, v)
        assert cache.seq_len == 16
        assert k_out.shape == (1, 2, 16, 64)

    def test_int8_kv_cache_append(self):
        from modeling.attention_utils import INT8KVCache
        cache = INT8KVCache(enabled=True)
        k1 = torch.randn(1, 2, 8, 64)
        v1 = torch.randn(1, 2, 8, 64)
        cache.update(0, k1, v1)
        assert cache.seq_len == 8

        k2 = torch.randn(1, 2, 4, 64)
        v2 = torch.randn(1, 2, 4, 64)
        k_out, v_out = cache.update(0, k2, v2)
        assert cache.seq_len == 12
        assert k_out.shape == (1, 2, 12, 64)


class TestSubLN:
    def test_subln_preserves_shape(self):
        from modeling.subln import SubLayerNorm
        sln = SubLayerNorm(64)
        x = torch.randn(2, 8, 64)
        y = sln(x)
        assert y.shape == x.shape

    def test_subln_disabled(self):
        from modeling.subln import SubLayerNorm
        sln = SubLayerNorm(64, enabled=False)
        x = torch.randn(2, 8, 64)
        y = sln(x)
        assert torch.equal(x, y)
