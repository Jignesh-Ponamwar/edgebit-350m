"""QK stabilization utilities for low-bit attention.

Provides:
  - RMSNorm on Q/K projections (default)
  - Cosine attention alternative
  - INT8 KV cache wrapper

Low-bit models are prone to attention logit explosion because quantized
Q/K vectors have irregular magnitude distributions. Normalizing Q and K
before dot-product scoring stabilizes training.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.quant_utils import int8_symmetric_quantize, int8_dequantize


class QKRMSNorm(nn.Module):
    """Apply RMSNorm independently to Q and K before attention scoring.

    This prevents attention logit explosion in low-bit models by ensuring
    Q and K have unit RMS regardless of quantization noise.
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.head_dim = head_dim
        self.eps = eps
        self.q_scale = nn.Parameter(torch.ones(head_dim))
        self.k_scale = nn.Parameter(torch.ones(head_dim))

    def _rms_norm(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * scale

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rms_norm(q, self.q_scale), self._rms_norm(k, self.k_scale)


class CosineAttentionScorer(nn.Module):
    """Cosine similarity attention scoring as alternative to dot-product.

    Replaces Q @ K^T / sqrt(d) with cos(Q, K) * tau, where tau is a
    learned temperature. More stable under aggressive quantization.
    """

    def __init__(self, head_dim: int, init_tau: float = 10.0):
        super().__init__()
        self.head_dim = head_dim
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau)))

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute cosine attention scores.

        Args:
            q: (batch, n_heads, seq_q, head_dim)
            k: (batch, n_kv_heads, seq_k, head_dim)

        Returns:
            scores: (batch, n_heads, seq_q, seq_k)
        """
        q_norm = F.normalize(q.float(), dim=-1)
        k_norm = F.normalize(k.float(), dim=-1)
        tau = self.log_tau.exp().clamp(max=100.0)
        scores = torch.matmul(q_norm, k_norm.transpose(-2, -1)) * tau
        return scores.to(q.dtype)


class INT8KVCache:
    """INT8 quantized KV cache for memory-efficient inference.

    Stores keys and values in INT8 with per-token scaling factors.
    Dequantizes on the fly during attention computation.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._k_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._v_cache: list[tuple[torch.Tensor, torch.Tensor]] = []

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V to cache and return full sequence K/V."""
        if not self.enabled:
            if layer_idx < len(self._k_cache):
                old_k = self._k_cache[layer_idx]
                old_v = self._v_cache[layer_idx]
                new_k = torch.cat([old_k, key], dim=2)
                new_v = torch.cat([old_v, value], dim=2)
                self._k_cache[layer_idx] = new_k
                self._v_cache[layer_idx] = new_v
                return new_k, new_v
            else:
                self._k_cache.append(key)
                self._v_cache.append(value)
                return key, value

        k_q, k_s = int8_symmetric_quantize(key, per_token=True)
        v_q, v_s = int8_symmetric_quantize(value, per_token=True)

        if layer_idx < len(self._k_cache):
            old_kq, old_ks = self._k_cache[layer_idx]
            old_vq, old_vs = self._v_cache[layer_idx]
            k_q = torch.cat([old_kq, k_q], dim=2)
            k_s = torch.cat([old_ks, k_s], dim=2)
            v_q = torch.cat([old_vq, v_q], dim=2)
            v_s = torch.cat([old_vs, v_s], dim=2)
            self._k_cache[layer_idx] = (k_q, k_s)
            self._v_cache[layer_idx] = (v_q, v_s)
        else:
            self._k_cache.append((k_q, k_s))
            self._v_cache.append((v_q, v_s))

        return int8_dequantize(k_q, k_s), int8_dequantize(v_q, v_s)

    def reset(self) -> None:
        self._k_cache.clear()
        self._v_cache.clear()

    @property
    def seq_len(self) -> int:
        if not self._k_cache:
            return 0
        item = self._k_cache[0]
        if isinstance(item, tuple):
            return item[0].shape[2]
        return item.shape[2]
