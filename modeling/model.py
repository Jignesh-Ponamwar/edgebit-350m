"""EdgeBit transformer model with ternary weights and GQA.

Architecture:
  - RoPE positional encoding
  - Grouped Query Attention (GQA) with configurable KV heads
  - SwiGLU FFN (gate + up + down)
  - SubLN before output projections
  - QK normalization for attention stability
  - BitLinear layers for ternary quantization
  - INT8 KV cache for inference
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.config import EdgeBitConfig
from modeling.bitlinear import BitLinear
from modeling.subln import SubLayerNorm
from modeling.attention_utils import QKRMSNorm, CosineAttentionScorer, INT8KVCache


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


def precompute_rope(dim: int, max_seq: int, theta: float = 10000.0,
                    device: Optional[torch.device] = None) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_offset: int = 0,
) -> torch.Tensor:
    """Apply rotary position encoding. x shape: (batch, n_heads, seq, head_dim)."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    seq_len = x.shape[2]
    end = position_offset + seq_len
    if end > cos.shape[0]:
        raise ValueError(
            f"RoPE position {end} exceeds configured max_seq_len={cos.shape[0]}"
        )
    cos = cos[position_offset:end].unsqueeze(0).unsqueeze(0)
    sin = sin[position_offset:end].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class EdgeBitAttention(nn.Module):
    """Grouped Query Attention with BitLinear projections."""

    def __init__(self, config: EdgeBitConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = BitLinear(
            config.hidden_dim, self.n_heads * self.head_dim,
            group_size=config.group_size, quant_mode=config.quant_mode,
        )
        self.k_proj = BitLinear(
            config.hidden_dim, self.n_kv_heads * self.head_dim,
            group_size=config.group_size, quant_mode=config.quant_mode,
        )
        self.v_proj = BitLinear(
            config.hidden_dim, self.n_kv_heads * self.head_dim,
            group_size=config.group_size, quant_mode=config.quant_mode,
        )

        self.subln = SubLayerNorm(config.hidden_dim, enabled=config.use_subln)
        self.o_proj = BitLinear(
            self.n_heads * self.head_dim, config.hidden_dim,
            group_size=config.group_size, quant_mode=config.quant_mode,
        )

        if config.use_qk_norm:
            if config.qk_norm_type == "rmsnorm":
                self.qk_norm = QKRMSNorm(self.head_dim)
            elif config.qk_norm_type == "cosine":
                self.qk_scorer = CosineAttentionScorer(self.head_dim)
            else:
                raise ValueError(f"Unknown qk_norm_type: {config.qk_norm_type}")

        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        bs, n_kv, seq, hd = x.shape
        return x[:, :, None, :, :].expand(bs, n_kv, self.n_rep, seq, hd).reshape(
            bs, self.n_heads, seq, hd
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[INT8KVCache] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin, position_offset=position_offset)
        k = apply_rope(k, rope_cos, rope_sin, position_offset=position_offset)

        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        if self.config.use_qk_norm:
            if self.config.qk_norm_type == "cosine":
                attn_weights = self.qk_scorer(q, k)
            else:
                q, k = self.qk_norm(q, k)
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(self.subln(attn_output))


class EdgeBitMLP(nn.Module):
    """SwiGLU FFN with BitLinear layers and SubLN."""

    def __init__(self, config: EdgeBitConfig):
        super().__init__()
        self.gate_proj = BitLinear(
            config.hidden_dim, config.ffn_dim,
            group_size=config.group_size, quant_mode=config.quant_mode,
        )
        self.up_proj = BitLinear(
            config.hidden_dim, config.ffn_dim,
            group_size=config.group_size, quant_mode=config.quant_mode,
        )
        self.subln = SubLayerNorm(config.ffn_dim, enabled=config.use_subln)
        self.down_proj = BitLinear(
            config.ffn_dim, config.hidden_dim,
            group_size=config.group_size, quant_mode=config.quant_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(self.subln(gate * up))


class EdgeBitDecoderLayer(nn.Module):
    def __init__(self, config: EdgeBitConfig, layer_idx: int = 0):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.self_attn = EdgeBitAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.mlp = EdgeBitMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[INT8KVCache] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, rope_cos, rope_sin, attention_mask, kv_cache, position_offset
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class EdgeBitModel(nn.Module):
    """EdgeBit decoder-only transformer backbone."""

    def __init__(self, config: EdgeBitConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList(
            [EdgeBitDecoderLayer(config, i) for i in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

        rope_cos, rope_sin = precompute_rope(
            config.head_dim, config.max_seq_len, config.rope_theta
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        self.apply(self._init_weights)

    def _build_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_seq_len: int,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype

        causal = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )
        if past_seq_len > 0:
            prefix = torch.zeros((seq_len, past_seq_len), device=device, dtype=dtype)
            causal = torch.cat([prefix, causal], dim=-1)
        additive_mask = causal.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, -1)

        if attention_mask is None:
            return additive_mask

        if attention_mask.ndim == 4:
            return additive_mask + attention_mask.to(device=device, dtype=dtype)
        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask must be shape (batch, seq) or additive shape "
                "(batch, 1, query, key)"
            )

        key_padding = attention_mask.to(device=device)
        if key_padding.shape != (bsz, seq_len):
            raise ValueError(
                f"attention_mask shape {tuple(key_padding.shape)} does not match "
                f"input_ids shape {(bsz, seq_len)}"
            )
        key_padding = torch.where(
            key_padding > 0,
            torch.zeros((), device=device, dtype=dtype),
            torch.full((), float("-inf"), device=device, dtype=dtype),
        ).view(bsz, 1, 1, seq_len)

        if past_seq_len > 0:
            cached_padding = torch.zeros((bsz, 1, 1, past_seq_len), device=device, dtype=dtype)
            key_padding = torch.cat([cached_padding, key_padding], dim=-1)

        return additive_mask + key_padding

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, BitLinear)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[INT8KVCache] = None,
    ) -> torch.Tensor:
        past_seq_len = kv_cache.seq_len if kv_cache is not None else 0
        hidden_states = self.embed_tokens(input_ids)

        attention_mask = self._build_attention_mask(input_ids, attention_mask, past_seq_len)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                self.rope_cos,
                self.rope_sin,
                attention_mask,
                kv_cache,
                past_seq_len,
            )

        return self.norm(hidden_states)


class EdgeBitForCausalLM(nn.Module):
    """EdgeBit causal language model with tied embeddings."""

    def __init__(self, config: EdgeBitConfig):
        super().__init__()
        self.config = config
        self.model = EdgeBitModel(config)
        if config.tie_embeddings:
            self.lm_head_weight = self.model.embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[INT8KVCache] = None,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.model(input_ids, attention_mask, kv_cache)

        if self.config.tie_embeddings:
            logits = F.linear(hidden_states, self.lm_head_weight)
        else:
            logits = self.lm_head(hidden_states)

        result = {"logits": logits, "hidden_states": hidden_states}

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        kv_cache: Optional[INT8KVCache] = None,
    ) -> torch.Tensor:
        """Simple autoregressive generation with INT8 KV cache."""
        if kv_cache is None:
            kv_cache = INT8KVCache(enabled=self.config.kv_cache_quant)

        generated = input_ids
        for _ in range(max_new_tokens):
            out = self.forward(generated[:, -1:] if kv_cache.seq_len > 0 else generated,
                               kv_cache=kv_cache)
            logits = out["logits"][:, -1, :]

            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                continue

            logits = logits / temperature

            if top_k > 0:
                top_k = min(top_k, logits.shape[-1])
                topk_vals, _ = logits.topk(top_k)
                logits[logits < topk_vals[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def set_quant_mode(self, mode: str) -> None:
        """Set quantization mode on all BitLinear layers."""
        for module in self.modules():
            if isinstance(module, BitLinear):
                module.set_quant_mode(mode)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        bitlinear = sum(
            p.numel() for m in self.modules() if isinstance(m, BitLinear)
            for p in m.parameters()
        )
        return {"total": total, "trainable": trainable, "bitlinear": bitlinear}
