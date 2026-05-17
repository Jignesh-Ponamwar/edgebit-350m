"""Progressive Quantization Curriculum Scheduler.

Manages staged precision transitions during training:
  Phase A: BF16 warmup (no quantization, stabilize weights)
  Phase B: INT8 QAT (gentle quantization, learn to handle noise)
  Phase C: 4-bit QAT (aggressive quantization, approach ternary)
  Phase D: Ternary convergence (final target precision)

The scheduler is checkpoint-safe: it can reconstruct its state from
the current training step and the curriculum config.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import yaml

from modeling.bitlinear import BitLinear

logger = logging.getLogger(__name__)


@dataclass
class CurriculumPhase:
    """A single phase in the quantization curriculum."""
    name: str
    quant_mode: str
    start_step: int
    end_step: int

    @property
    def duration(self) -> int:
        return self.end_step - self.start_step


class QuantizationScheduler:
    """Manages progressive quantization transitions.

    Args:
        phases: list of CurriculumPhase instances.
        model: the model whose BitLinear layers will be switched.
    """

    def __init__(self, phases: list[CurriculumPhase], model: object):
        self.phases = sorted(phases, key=lambda p: p.start_step)
        self.model = model
        self._current_mode: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str, model: object, total_steps: int = 0) -> "QuantizationScheduler":
        with open(path) as f:
            cfg = yaml.safe_load(f)

        phases = []
        for p in cfg["phases"]:
            if "start_step" in p and "end_step" in p:
                start = p["start_step"]
                end = p["end_step"]
            elif "start_pct" in p and "end_pct" in p:
                if total_steps <= 0:
                    raise ValueError(
                        "Curriculum YAML uses percentage-based phases but total_steps was not provided. "
                        "Pass total_steps to from_yaml() or use start_step/end_step in the YAML."
                    )
                start = int(p["start_pct"] * total_steps)
                end = int(p["end_pct"] * total_steps)
            else:
                raise ValueError(
                    f"Phase '{p['name']}' must have either start_step/end_step or start_pct/end_pct"
                )
            phases.append(CurriculumPhase(
                name=p["name"],
                quant_mode=p["quant_mode"],
                start_step=start,
                end_step=end,
            ))
        return cls(phases, model)

    @classmethod
    def from_total_steps(cls, total_steps: int, model: object) -> "QuantizationScheduler":
        """Create a default curriculum from total training steps."""
        warmup = int(total_steps * 0.10)
        int8_end = int(total_steps * 0.30)
        int4_end = int(total_steps * 0.55)

        phases = [
            CurriculumPhase("bf16_warmup", "none", 0, warmup),
            CurriculumPhase("int8_qat", "int8", warmup, int8_end),
            CurriculumPhase("int4_qat", "int4", int8_end, int4_end),
            CurriculumPhase("ternary", "ternary", int4_end, total_steps),
        ]
        return cls(phases, model)

    def get_phase(self, step: int) -> CurriculumPhase:
        for phase in self.phases:
            if phase.start_step <= step < phase.end_step:
                return phase
        return self.phases[-1]

    def step(self, global_step: int) -> str:
        """Update model quantization mode for the current step.

        Returns the name of the active phase.
        """
        phase = self.get_phase(global_step)
        if phase.quant_mode != self._current_mode:
            logger.info(
                "Step %d: transitioning to phase '%s' (quant_mode=%s)",
                global_step, phase.name, phase.quant_mode,
            )
            self._set_quant_mode(phase.quant_mode)
            self._current_mode = phase.quant_mode
        return phase.name

    def _set_quant_mode(self, mode: str) -> None:
        count = 0
        for module in self.model.modules():
            if isinstance(module, BitLinear):
                module.set_quant_mode(mode)
                count += 1
        logger.info("Set %d BitLinear layers to quant_mode='%s'", count, mode)

    def state_dict(self) -> dict:
        return {
            "current_mode": self._current_mode,
            "phases": [
                {"name": p.name, "quant_mode": p.quant_mode,
                 "start_step": p.start_step, "end_step": p.end_step}
                for p in self.phases
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        self._current_mode = state.get("current_mode")

    def summary(self) -> str:
        lines = ["Quantization Curriculum:"]
        for p in self.phases:
            lines.append(f"  {p.name}: steps {p.start_step}-{p.end_step} ({p.quant_mode})")
        return "\n".join(lines)
