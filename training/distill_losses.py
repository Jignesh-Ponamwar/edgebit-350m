"""Distillation loss functions for EdgeBit training.

Provides:
  - Logit-level KL divergence distillation
  - Hidden-state alignment distillation (intermediate layers)
  - Combined distillation + CE loss
  - Configurable layer mapping for teacher-student alignment
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HiddenStateProjector(nn.Module):
    """Projects student hidden states to teacher hidden state space.

    Used when teacher and student have different hidden dimensions.
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DistillationLoss(nn.Module):
    """Combined distillation loss with logit KD + hidden-state alignment.

    Args:
        temperature: softmax temperature for KL divergence.
        alpha_kd: weight for KL divergence loss on logits.
        alpha_ce: weight for cross-entropy loss on hard labels.
        alpha_hidden: weight for hidden-state alignment loss.
        student_dim: student model hidden dimension.
        teacher_dim: teacher model hidden dimension.
        layer_mapping: dict mapping student layer indices to teacher layer indices.
            Example: {2: 4, 5: 8, 8: 12} maps student layer 2 to teacher layer 4.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha_kd: float = 0.7,
        alpha_ce: float = 0.3,
        alpha_hidden: float = 0.1,
        student_dim: int = 1024,
        teacher_dim: int = 1024,
        layer_mapping: Optional[dict[int, int]] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha_kd = alpha_kd
        self.alpha_ce = alpha_ce
        self.alpha_hidden = alpha_hidden
        self.layer_mapping = layer_mapping or {}

        if student_dim != teacher_dim and self.layer_mapping:
            self.projectors = nn.ModuleDict({
                str(s_idx): HiddenStateProjector(student_dim, teacher_dim)
                for s_idx in self.layer_mapping
            })
        else:
            self.projectors = nn.ModuleDict()

    def logit_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence between student and teacher output distributions."""
        s_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        t_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean")
        return kl * (self.temperature ** 2)

    def hidden_state_loss(
        self,
        student_hiddens: dict[int, torch.Tensor],
        teacher_hiddens: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """MSE loss between projected student and teacher hidden states."""
        if not self.layer_mapping:
            return torch.tensor(0.0, device=next(iter(student_hiddens.values())).device)

        total = torch.tensor(0.0, device=next(iter(student_hiddens.values())).device)
        count = 0

        for s_idx, t_idx in self.layer_mapping.items():
            if s_idx not in student_hiddens or t_idx not in teacher_hiddens:
                continue

            s_h = student_hiddens[s_idx]
            t_h = teacher_hiddens[t_idx]

            if str(s_idx) in self.projectors:
                s_h = self.projectors[str(s_idx)](s_h)

            total = total + F.mse_loss(s_h.float(), t_h.float().detach())
            count += 1

        return total / max(count, 1)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_hiddens: Optional[dict[int, torch.Tensor]] = None,
        teacher_hiddens: Optional[dict[int, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined distillation loss.

        Returns dict with total loss and components for logging.
        """
        shift_s = student_logits[..., :-1, :].contiguous()
        shift_t = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        kd_loss = self.logit_kd_loss(
            shift_s.view(-1, shift_s.size(-1)),
            shift_t.view(-1, shift_t.size(-1)),
        )

        ce_loss = F.cross_entropy(
            shift_s.view(-1, shift_s.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        total = self.alpha_kd * kd_loss + self.alpha_ce * ce_loss

        hidden_loss = torch.tensor(0.0, device=student_logits.device)
        if (student_hiddens is not None and teacher_hiddens is not None
                and self.alpha_hidden > 0):
            hidden_loss = self.hidden_state_loss(student_hiddens, teacher_hiddens)
            total = total + self.alpha_hidden * hidden_loss

        return {
            "loss": total,
            "kd_loss": kd_loss.detach(),
            "ce_loss": ce_loss.detach(),
            "hidden_loss": hidden_loss.detach(),
        }
