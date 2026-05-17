"""Tests for distillation loss functions."""
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.distill_losses import DistillationLoss, HiddenStateProjector


class TestHiddenStateProjector:
    def test_projection_shape(self):
        proj = HiddenStateProjector(512, 1024)
        x = torch.randn(2, 16, 512)
        y = proj(x)
        assert y.shape == (2, 16, 1024)

    def test_same_dim_identity(self):
        proj = HiddenStateProjector(256, 256)
        x = torch.randn(1, 8, 256)
        y = proj(x)
        assert y.shape == x.shape


class TestDistillationLoss:
    @pytest.fixture
    def loss_fn(self):
        return DistillationLoss(
            temperature=4.0,
            alpha_kd=0.7,
            alpha_ce=0.3,
            alpha_hidden=0.0,
        )

    @pytest.fixture
    def loss_fn_with_hidden(self):
        return DistillationLoss(
            temperature=4.0,
            alpha_kd=0.7,
            alpha_ce=0.3,
            alpha_hidden=0.1,
            student_dim=256,
            teacher_dim=512,
            layer_mapping={2: 4, 5: 8},
        )

    def test_logit_kd_loss(self, loss_fn):
        s_logits = torch.randn(4, 100)
        t_logits = torch.randn(4, 100)
        kd = loss_fn.logit_kd_loss(s_logits, t_logits)
        assert kd.ndim == 0
        assert kd.item() >= 0

    def test_logit_kd_identical(self, loss_fn):
        logits = torch.randn(4, 100)
        kd = loss_fn.logit_kd_loss(logits, logits)
        assert kd.item() < 0.01

    def test_forward_basic(self, loss_fn):
        s_logits = torch.randn(2, 32, 100)
        t_logits = torch.randn(2, 32, 100)
        labels = torch.randint(0, 100, (2, 32))
        labels[:, -5:] = -100

        result = loss_fn(s_logits, t_logits, labels)
        assert "loss" in result
        assert "kd_loss" in result
        assert "ce_loss" in result
        assert "hidden_loss" in result
        assert result["loss"].ndim == 0
        assert result["loss"].item() > 0

    def test_forward_gradient(self, loss_fn):
        s_logits = torch.randn(2, 16, 50, requires_grad=True)
        t_logits = torch.randn(2, 16, 50)
        labels = torch.randint(0, 50, (2, 16))

        result = loss_fn(s_logits, t_logits, labels)
        result["loss"].backward()
        assert s_logits.grad is not None

    def test_hidden_state_loss(self, loss_fn_with_hidden):
        student_h = {
            2: torch.randn(2, 16, 256),
            5: torch.randn(2, 16, 256),
        }
        teacher_h = {
            4: torch.randn(2, 16, 512),
            8: torch.randn(2, 16, 512),
        }
        h_loss = loss_fn_with_hidden.hidden_state_loss(student_h, teacher_h)
        assert h_loss.ndim == 0
        assert h_loss.item() > 0

    def test_forward_with_hidden(self, loss_fn_with_hidden):
        s_logits = torch.randn(2, 16, 50)
        t_logits = torch.randn(2, 16, 50)
        labels = torch.randint(0, 50, (2, 16))
        student_h = {2: torch.randn(2, 16, 256), 5: torch.randn(2, 16, 256)}
        teacher_h = {4: torch.randn(2, 16, 512), 8: torch.randn(2, 16, 512)}

        result = loss_fn_with_hidden(
            s_logits, t_logits, labels, student_h, teacher_h,
        )
        assert result["hidden_loss"].item() > 0
        assert result["loss"].item() > 0

    def test_temperature_effect(self):
        s_logits = torch.randn(2, 16, 50)
        t_logits = torch.randn(2, 16, 50)
        labels = torch.randint(0, 50, (2, 16))

        loss_low_t = DistillationLoss(temperature=1.0, alpha_kd=1.0, alpha_ce=0.0)
        loss_high_t = DistillationLoss(temperature=8.0, alpha_kd=1.0, alpha_ce=0.0)

        r1 = loss_low_t(s_logits, t_logits, labels)
        r2 = loss_high_t(s_logits, t_logits, labels)
        assert r1["loss"].item() != r2["loss"].item()

    def test_alpha_weights(self):
        s_logits = torch.randn(2, 16, 50)
        t_logits = torch.randn(2, 16, 50)
        labels = torch.randint(0, 50, (2, 16))

        loss_kd = DistillationLoss(alpha_kd=1.0, alpha_ce=0.0)
        loss_ce = DistillationLoss(alpha_kd=0.0, alpha_ce=1.0)

        r_kd = loss_kd(s_logits, t_logits, labels)
        r_ce = loss_ce(s_logits, t_logits, labels)

        assert abs(r_kd["loss"].item() - r_kd["kd_loss"].item()) < 0.01
        assert abs(r_ce["loss"].item() - r_ce["ce_loss"].item()) < 0.01
