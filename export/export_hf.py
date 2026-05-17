#!/usr/bin/env python3
"""Export EdgeBit model to HuggingFace-compatible format.

Produces:
  - config.json (HF-style config)
  - model.safetensors (weights in safetensors format)
  - tokenizer files (copied from source tokenizer)
  - generation_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import logging
from typing import Optional

import torch

from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM

logger = logging.getLogger(__name__)


def build_hf_config(config: EdgeBitConfig) -> dict:
    return {
        "architectures": ["EdgeBitForCausalLM"],
        "model_type": "edgebit",
        "hidden_size": config.hidden_dim,
        "intermediate_size": config.ffn_dim,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.n_kv_heads,
        "head_dim": config.head_dim,
        "max_position_embeddings": config.max_seq_len,
        "vocab_size": config.vocab_size,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": config.tie_embeddings,
        "torch_dtype": "float32",
        "quantization_config": {
            "quant_method": "edgebit_ternary",
            "quant_mode": config.quant_mode,
            "group_size": config.group_size,
            "embedding_quant": config.embedding_quant,
            "kv_cache_quant": config.kv_cache_quant,
        },
        "edgebit_config": config.to_dict(),
    }


def build_generation_config() -> dict:
    return {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    }


def export_to_hf(
    checkpoint_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
    tokenizer_name: str = "Qwen/Qwen3-0.6B",
    use_safetensors: bool = True,
    save_packed: bool = False,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    if config_path:
        import yaml
        with open(config_path) as f:
            cfg_dict = yaml.safe_load(f)
        config = EdgeBitConfig(**cfg_dict.get("model", cfg_dict))
    else:
        config_json = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_json):
            with open(config_json) as f:
                cfg_dict = json.load(f)
            if "edgebit_config" in cfg_dict:
                config = EdgeBitConfig(**cfg_dict["edgebit_config"])
            else:
                config = EdgeBitConfig(**cfg_dict)
        else:
            config = EdgeBitConfig()

    model = EdgeBitForCausalLM(config)

    state_path = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        logger.info("Loaded from training checkpoint: %s", state_path)
    else:
        pt_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".pt") or f.endswith(".bin")]
        if pt_files:
            state = torch.load(
                os.path.join(checkpoint_path, pt_files[0]),
                map_location="cpu", weights_only=False,
            )
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
            logger.info("Loaded weights from: %s", pt_files[0])

    state_dict = model.state_dict()
    if config.tie_embeddings and "lm_head_weight" in state_dict:
        state_dict = dict(state_dict)
        state_dict["lm_head_weight"] = state_dict["lm_head_weight"].clone()

    if use_safetensors:
        try:
            from safetensors.torch import save_file
            save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
            logger.info("Saved weights in safetensors format")
        except ImportError:
            logger.warning("safetensors not installed, falling back to .bin")
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    else:
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        logger.info("Saved weights in pytorch .bin format")

    hf_config = build_hf_config(config)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)

    gen_config = build_generation_config()
    with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
        json.dump(gen_config, f, indent=2)

    if tokenizer_name == "simple":
        from training.simple_tokenizer import SimpleHashTokenizer
        tokenizer = SimpleHashTokenizer(vocab_size=config.vocab_size)
        tokenizer.save_pretrained(output_dir)
        logger.info("Saved simple tokenizer metadata")
    else:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            tokenizer.save_pretrained(output_dir)
            logger.info("Saved tokenizer from %s", tokenizer_name)
        except Exception as e:
            logger.warning("Could not save tokenizer: %s", e)

    if save_packed:
        from runtime.packed_checkpoint import save_packed_checkpoint
        save_packed_checkpoint(
            {"model_state_dict": state_dict},
            os.path.join(output_dir, "edgebit_packed_state.pt"),
            group_size=config.group_size,
        )
        logger.info("Saved packed EdgeBit storage artifact")

    param_count = sum(p.numel() for p in model.parameters())
    weight_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    meta = {
        "model_type": "edgebit",
        "param_count": param_count,
        "weight_size_mb": round(weight_mb, 2),
        "quant_mode": config.quant_mode,
        "source_checkpoint": checkpoint_path,
        "export_format": "safetensors" if use_safetensors else "pytorch",
        "packed_storage_artifact": save_packed,
        "runtime_note": (
            "Packed storage reloads into the PyTorch model; custom packed "
            "ternary matmul kernels are not included in this export."
        ),
    }
    with open(os.path.join(output_dir, "export_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Export complete: %s (%.1f MB, %d params)", output_dir, weight_mb, param_count)
    return output_dir


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Export EdgeBit to HuggingFace format")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--no_safetensors", action="store_true")
    parser.add_argument("--save_packed", action="store_true")
    args = parser.parse_args()

    export_to_hf(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
        tokenizer_name=args.tokenizer,
        use_safetensors=not args.no_safetensors,
        save_packed=args.save_packed,
    )


if __name__ == "__main__":
    main()
