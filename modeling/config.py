"""EdgeBit model configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EdgeBitConfig:
    """Configuration for EdgeBit transformer models.

    All sizes are chosen for runtime efficiency on edge hardware.
    The architecture uses GQA (grouped query attention) with separate
    KV heads for memory efficiency.
    """

    # --- Architecture ---
    vocab_size: int = 32000
    hidden_dim: int = 1024
    ffn_dim: int = 2816
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 64
    max_seq_len: int = 2048
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    rope_theta: float = 10000.0

    # --- Quantization ---
    group_size: int = 128
    quant_mode: str = "ternary"
    embedding_quant: str = "nf4"
    kv_cache_quant: bool = True
    stochastic_round: bool = False

    # --- Stabilization ---
    use_subln: bool = True
    use_qk_norm: bool = True
    qk_norm_type: str = "rmsnorm"
    attention_dropout: float = 0.0

    # --- Training ---
    initializer_range: float = 0.02

    @property
    def n_params_estimate(self) -> int:
        """Rough parameter count estimate."""
        emb = self.vocab_size * self.hidden_dim
        kv_dim = self.n_kv_heads * self.head_dim
        per_layer = (
            self.hidden_dim * self.hidden_dim  # q_proj
            + self.hidden_dim * kv_dim           # k_proj
            + self.hidden_dim * kv_dim           # v_proj
            + self.hidden_dim * self.hidden_dim  # o_proj
            + self.hidden_dim * self.ffn_dim * 3  # gate + up + down
            + self.hidden_dim * 4                 # norms
        )
        total = emb + per_layer * self.n_layers + self.hidden_dim
        if not self.tie_embeddings:
            total += emb
        return total

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def tiny(cls) -> "EdgeBitConfig":
        """~50M parameter tiny model for validation."""
        return cls(
            hidden_dim=512,
            ffn_dim=1408,
            n_layers=8,
            n_heads=8,
            n_kv_heads=2,
            head_dim=64,
        )

    @classmethod
    def small_125m(cls) -> "EdgeBitConfig":
        """~125M parameter model for convergence proof."""
        return cls(
            hidden_dim=768,
            ffn_dim=2048,
            n_layers=12,
            n_heads=12,
            n_kv_heads=4,
            head_dim=64,
        )

    @classmethod
    def base_350m(cls) -> "EdgeBitConfig":
        """~350M parameter model -- production target."""
        return cls()

    @classmethod
    def base(cls) -> "EdgeBitConfig":
        """Alias for base_350m()."""
        return cls.base_350m()
