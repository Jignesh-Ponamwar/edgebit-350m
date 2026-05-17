#!/usr/bin/env python3
"""EdgeBit training loop with progressive quantization curriculum.

Supports:
  - Pretraining (causal LM)
  - Distillation (teacher-student KD)
  - SFT (instruction tuning)
  - Progressive quantization (BF16 -> INT8 -> INT4 -> ternary)
  - DeepSpeed / FSDP integration via Accelerate
  - Checkpoint resume
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM
from training.quant_scheduler import QuantizationScheduler
from training.data import PretrainingDataset, create_synthetic_data
from training.distill_losses import DistillationLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("edgebit.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EdgeBit Training")
    p.add_argument("--config", type=str, default="configs/model_350m.yaml")
    p.add_argument("--data_path", type=str, default="/mnt/data/pretrain/train.jsonl")
    p.add_argument("--output_dir", type=str, default="/mnt/ckpts/edgebit-350m")
    p.add_argument("--curriculum", type=str, default="configs/quant_curriculum.yaml")
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--use_accelerate", action="store_true")
    p.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    p.add_argument("--teacher_model", type=str, default=None)
    p.add_argument("--distill_temperature", type=float, default=4.0)
    p.add_argument("--alpha_kd", type=float, default=0.7)
    p.add_argument("--alpha_ce", type=float, default=0.3)
    return p.parse_args()


def load_model_config(path: str) -> EdgeBitConfig:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return EdgeBitConfig(**cfg.get("model", {}))


def load_tokenizer(name: str):
    if name == "simple":
        from training.simple_tokenizer import SimpleHashTokenizer
        return SimpleHashTokenizer()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def save_checkpoint(
    model: EdgeBitForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    quant_scheduler: QuantizationScheduler,
    step: int,
    loss: float,
    output_dir: str,
) -> str:
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "quant_scheduler_state_dict": quant_scheduler.state_dict(),
        "step": step,
        "loss": loss,
    }, os.path.join(ckpt_dir, "training_state.pt"))

    model.config.to_dict()
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(model.config.to_dict(), f, indent=2)

    logger.info("Saved checkpoint at step %d to %s", step, ckpt_dir)
    return ckpt_dir


def load_checkpoint(
    path: str,
    model: EdgeBitForCausalLM,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    quant_scheduler: Optional[QuantizationScheduler] = None,
) -> int:
    state_path = os.path.join(path, "training_state.pt")
    if not os.path.exists(state_path):
        logger.warning("No training state found at %s", state_path)
        return 0

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and state.get("scheduler_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if quant_scheduler and state.get("quant_scheduler_state_dict"):
        quant_scheduler.load_state_dict(state["quant_scheduler_state_dict"])

    step = state.get("step", 0)
    logger.info("Resumed from %s at step %d (loss=%.4f)", path, step, state.get("loss", -1))
    return step


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    accelerator = None
    if args.use_accelerate:
        try:
            from accelerate import Accelerator
        except ImportError as exc:
            raise RuntimeError(
                "Install accelerate or omit --use_accelerate."
            ) from exc
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
        )
        device = accelerator.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    config = load_model_config(args.config)

    if args.smoke_test and args.tokenizer == "Qwen/Qwen3-0.6B":
        args.tokenizer = "simple"
    if args.smoke_test:
        args.max_seq_length = min(args.max_seq_length, config.max_seq_len)

    tokenizer = load_tokenizer(args.tokenizer)
    config.vocab_size = len(tokenizer)

    logger.info("Model config: %s", config.to_dict())

    model = EdgeBitForCausalLM(config).to(device)
    param_counts = model.count_parameters()
    logger.info("Parameters: total=%d trainable=%d bitlinear=%d",
                param_counts["total"], param_counts["trainable"], param_counts["bitlinear"])

    if args.smoke_test:
        data_path = os.path.join(args.output_dir, "smoke_data.jsonl")
        os.makedirs(args.output_dir, exist_ok=True)
        create_synthetic_data(data_path, n_samples=200)
        args.data_path = data_path
        if args.max_steps < 0:
            args.max_steps = 50
        args.save_steps = min(args.save_steps, max(args.max_steps, 1))
        args.logging_steps = min(args.logging_steps, max(args.max_steps, 1))
        logger.info("SMOKE TEST: 200 samples, %d steps", args.max_steps)

    dataset = PretrainingDataset(
        args.data_path, tokenizer, max_length=args.max_seq_length,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    teacher_model = None
    distill_loss = None
    if args.teacher_model:
        from transformers import AutoModelForCausalLM
        teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model).to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        distill_loss = DistillationLoss(
            temperature=args.distill_temperature,
            alpha_kd=args.alpha_kd,
            alpha_ce=args.alpha_ce,
            alpha_hidden=0.0,
        )

    total_steps = args.max_steps if args.max_steps > 0 else (
        len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_steps, T_mult=1)

    if os.path.exists(args.curriculum):
        quant_scheduler = QuantizationScheduler.from_yaml(args.curriculum, model, total_steps=total_steps)
    else:
        quant_scheduler = QuantizationScheduler.from_total_steps(total_steps, model)
    logger.info(quant_scheduler.summary())

    global_step = 0
    if args.resume_from:
        global_step = load_checkpoint(args.resume_from, model, optimizer, scheduler, quant_scheduler)

    if accelerator is not None:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
        if teacher_model is not None:
            teacher_model = accelerator.prepare(teacher_model)

    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    accum_loss = 0.0
    step_time = time.time()

    logger.info("Starting training: %d total steps, %d epochs", total_steps, args.num_epochs)

    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            phase_name = quant_scheduler.step(global_step)

            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            if teacher_model is not None and distill_loss is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                if teacher_outputs.logits.shape[-1] != outputs["logits"].shape[-1]:
                    raise ValueError(
                        "Teacher and student vocab sizes differ; logit distillation "
                        "requires matching tokenizers/vocabularies."
                    )
                loss_dict = distill_loss(outputs["logits"], teacher_outputs.logits, labels)
                raw_loss = loss_dict["loss"]
            else:
                raw_loss = outputs["loss"]

            loss = raw_loss / args.gradient_accumulation_steps
            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if accelerator is not None:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    elapsed = time.time() - step_time
                    tok_per_sec = (args.batch_size * args.max_seq_length *
                                   args.gradient_accumulation_steps * args.logging_steps) / elapsed
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        "step=%d epoch=%d loss=%.4f lr=%.2e tok/s=%.0f phase=%s",
                        global_step, epoch, accum_loss / args.logging_steps,
                        lr, tok_per_sec, phase_name,
                    )
                    accum_loss = 0.0
                    step_time = time.time()

                if global_step % args.save_steps == 0:
                    model_to_save = accelerator.unwrap_model(model) if accelerator else model
                    save_checkpoint(model_to_save, optimizer, scheduler, quant_scheduler,
                                    global_step, raw_loss.item(), args.output_dir)

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    model_to_save = accelerator.unwrap_model(model) if accelerator else model
    save_checkpoint(model_to_save, optimizer, scheduler, quant_scheduler,
                    global_step, accum_loss, args.output_dir)
    logger.info("Training complete. Final step: %d", global_step)


if __name__ == "__main__":
    main()
