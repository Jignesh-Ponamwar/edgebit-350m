"""Grouped Ternary BitLinear layer for EdgeBit.

Maintains latent FP master weights. During forward pass, quantizes to
{-1, 0, +1} using per-group absmean scaling. Activations are quantized
to INT8 per-token. Gradients flow through quantization via STE.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.quant_utils import (
    ternary_quantize_absmean,
    int8_symmetric_quantize,
    QuantStats,
)


class BitLinear(nn.Module):
    """Linear layer with ternary weight quantization and INT8 activation quantization.

    Args:
        in_features: input dimension.
        out_features: output dimension.
        bias: whether to include bias.
        group_size: quantization group size for weights.
        stochastic_round: use stochastic rounding during training.
        quant_mode: "ternary" | "int8" | "int4" | "none". Controls precision.
        collect_stats: log quantization statistics.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 128,
        stochastic_round: bool = False,
        quant_mode: str = "ternary",
        collect_stats: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.stochastic_round = stochastic_round
        self.quant_mode = quant_mode
        self.collect_stats = collect_stats

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.stats: Optional[QuantStats] = QuantStats() if collect_stats else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _quantize_weight_ternary(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return ternary_quantize_absmean(
            w,
            group_size=self.group_size,
            stochastic=self.stochastic_round and self.training,
        )

    def _quantize_weight_int8(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return int8_symmetric_quantize(w, per_token=False)

    def _quantize_weight_int4(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = w.shape
        K = orig_shape[-1]
        gs = self.group_size if K % self.group_size == 0 else K
        w_grouped = w.reshape(-1, gs)
        amax = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        scale = 7.0 / amax
        normalized = w_grouped * scale
        from modeling.quant_utils import ste_round, ste_clamp
        quantized = ste_clamp(ste_round(normalized), -8.0, 7.0)
        return quantized.reshape(orig_shape), scale.reshape(*orig_shape[:-1], -1)

    def _quantize_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return int8_symmetric_quantize(x, per_token=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_mode == "none":
            return F.linear(x, self.weight, self.bias)

        if self.quant_mode == "ternary":
            w_q, w_scale = self._quantize_weight_ternary(self.weight)
        elif self.quant_mode == "int8":
            w_q, w_scale = self._quantize_weight_int8(self.weight)
        elif self.quant_mode == "int4":
            w_q, w_scale = self._quantize_weight_int4(self.weight)
        else:
            raise ValueError(f"Unknown quant_mode: {self.quant_mode}")

        x_q, x_scale = self._quantize_activations(x)
        y = F.linear(x_q, w_q, self.bias)
        y = y / x_scale

        if self.quant_mode == "ternary":
            w_scale_broad = w_scale.mean(dim=-1)
            y = y * w_scale_broad.unsqueeze(0).unsqueeze(0)
        else:
            y = y / w_scale.mean()

        if self.collect_stats and self.stats is not None:
            self.stats.update(
                self.weight.data, w_q, w_scale,
                grad=self.weight.grad if self.weight.grad is not None else None,
            )

        return y

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128,
                    quant_mode: str = "ternary", **kwargs) -> "BitLinear":
        bl = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            quant_mode=quant_mode,
            **kwargs,
        )
        bl.weight.data.copy_(linear.weight.data)
        if linear.bias is not None and bl.bias is not None:
            bl.bias.data.copy_(linear.bias.data)
        return bl

    def set_quant_mode(self, mode: str) -> None:
        """Switch quantization precision. Used by progressive curriculum."""
        assert mode in ("none", "int8", "int4", "ternary")
        self.quant_mode = mode

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"group={self.group_size}, mode={self.quant_mode}"
        )
