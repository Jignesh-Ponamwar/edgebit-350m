"""Tests for QuantizationScheduler and progressive curriculum."""
import pytest
import torch
import os
import yaml

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.quant_scheduler import QuantizationScheduler, CurriculumPhase
from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM


@pytest.fixture
def tiny_model():
    config = EdgeBitConfig.tiny()
    config.vocab_size = 500
    return EdgeBitForCausalLM(config)


@pytest.fixture
def curriculum_yaml(tmp_path):
    data = {
        "phases": [
            {"name": "warmup", "quant_mode": "none", "start_step": 0, "end_step": 200},
            {"name": "int8", "quant_mode": "int8", "start_step": 200, "end_step": 500},
            {"name": "ternary", "quant_mode": "ternary", "start_step": 500, "end_step": 1000},
        ]
    }
    path = tmp_path / "curriculum.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


class TestCurriculumPhase:
    def test_duration(self):
        p = CurriculumPhase(name="test", quant_mode="none", start_step=0, end_step=100)
        assert p.duration == 100

    def test_step_in_range(self):
        p = CurriculumPhase(name="test", quant_mode="none", start_step=10, end_step=50)
        assert p.start_step == 10
        assert p.end_step == 50
        assert p.duration == 40


class TestQuantizationScheduler:
    def test_from_total_steps(self, tiny_model):
        sched = QuantizationScheduler.from_total_steps(1000, tiny_model)
        assert len(sched.phases) == 4
        assert sched.phases[0].quant_mode == "none"
        assert sched.phases[-1].quant_mode == "ternary"

    def test_from_yaml(self, curriculum_yaml, tiny_model):
        sched = QuantizationScheduler.from_yaml(curriculum_yaml, tiny_model)
        assert len(sched.phases) == 3
        assert sched.phases[0].name == "warmup"
        assert sched.phases[0].end_step == 200

    def test_step_transitions(self, tiny_model):
        sched = QuantizationScheduler.from_total_steps(100, tiny_model)

        phase = sched.step(0)
        assert phase == "bf16_warmup"

        phase = sched.step(50)
        assert phase is not None

        phase = sched.step(99)
        assert phase is not None

    def test_model_mode_changes(self, tiny_model):
        sched = QuantizationScheduler.from_total_steps(100, tiny_model)

        sched.step(0)

        sched.step(95)
        from modeling.bitlinear import BitLinear
        for m in tiny_model.modules():
            if isinstance(m, BitLinear):
                assert m.quant_mode == "ternary"

    def test_state_dict_roundtrip(self, tiny_model):
        sched = QuantizationScheduler.from_total_steps(100, tiny_model)
        sched.step(50)

        state = sched.state_dict()
        assert "phases" in state
        assert "current_mode" in state

        sched2 = QuantizationScheduler.from_total_steps(100, tiny_model)
        sched2.load_state_dict(state)
        assert sched2._current_mode == sched._current_mode

    def test_summary(self, tiny_model):
        sched = QuantizationScheduler.from_total_steps(1000, tiny_model)
        summary = sched.summary()
        assert "bf16" in summary.lower() or "none" in summary.lower()
        assert "ternary" in summary.lower()

    def test_get_phase(self, tiny_model):
        sched = QuantizationScheduler.from_total_steps(1000, tiny_model)
        phase = sched.get_phase(0)
        assert phase.quant_mode == "none"

        phase = sched.get_phase(999)
        assert phase.quant_mode == "ternary"
