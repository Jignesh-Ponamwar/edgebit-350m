#!/usr/bin/env python3
"""Export EdgeBit model to a minimal GGUF-like container.

GGUF (GPT-Generated Unified Format) is the standard format for
llama.cpp and other edge inference engines. This exporter writes the
EdgeBit tensor names and metadata in a minimal GGUF layout, but it does
not yet emit a llama.cpp-loadable custom ternary tensor type. Quantizable
weights are stored as fp16 when --no_quantize is not supplied.

GGUF file layout:
  1. Header (magic, version, tensor count, metadata count)
  2. Metadata KV pairs
  3. Tensor info entries
  4. Alignment padding
  5. Tensor data
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import struct
from typing import Optional

import torch
import numpy as np

from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM
logger = logging.getLogger(__name__)

GGUF_MAGIC = 0x46475547  # "GGUF"
GGUF_VERSION = 3

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q2_K = 10
GGML_TYPE_I8 = 24


class GGUFWriter:
    def __init__(self, path: str):
        self.path = path
        self.metadata: list[tuple[str, int, bytes]] = []
        self.tensors: list[tuple[str, np.ndarray, int]] = []

    def add_string(self, key: str, value: str):
        encoded = value.encode("utf-8")
        data = struct.pack("<Q", len(encoded)) + encoded
        self.metadata.append((key, GGUF_TYPE_STRING, data))

    def add_uint32(self, key: str, value: int):
        data = struct.pack("<I", value)
        self.metadata.append((key, GGUF_TYPE_UINT32, data))

    def add_uint64(self, key: str, value: int):
        data = struct.pack("<Q", value)
        self.metadata.append((key, GGUF_TYPE_UINT64, data))

    def add_int32(self, key: str, value: int):
        data = struct.pack("<i", value)
        self.metadata.append((key, GGUF_TYPE_INT32, data))

    def add_float32(self, key: str, value: float):
        data = struct.pack("<f", value)
        self.metadata.append((key, GGUF_TYPE_FLOAT32, data))

    def add_bool(self, key: str, value: bool):
        data = struct.pack("<B", 1 if value else 0)
        self.metadata.append((key, GGUF_TYPE_BOOL, data))

    def add_tensor(self, name: str, data: np.ndarray, ggml_type: int = GGML_TYPE_F32):
        self.tensors.append((name, data, ggml_type))

    def _write_string(self, f, s: str):
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)

    def write(self):
        with open(self.path, "wb") as f:
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.metadata)))

            for key, vtype, data in self.metadata:
                self._write_string(f, key)
                f.write(struct.pack("<I", vtype))
                f.write(data)

            tensor_offsets = []
            current_offset = 0
            for name, data, ggml_type in self.tensors:
                self._write_string(f, name)
                n_dims = len(data.shape)
                f.write(struct.pack("<I", n_dims))
                for dim in data.shape:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", ggml_type))
                f.write(struct.pack("<Q", current_offset))

                tensor_offsets.append(current_offset)
                size = data.nbytes
                aligned_size = (size + 31) & ~31
                current_offset += aligned_size

            alignment = 32
            pos = f.tell()
            pad = (alignment - (pos % alignment)) % alignment
            f.write(b"\x00" * pad)

            for name, data, _ in self.tensors:
                raw = data.tobytes()
                f.write(raw)
                pad = (alignment - (len(raw) % alignment)) % alignment
                f.write(b"\x00" * pad)

        logger.info("Wrote GGUF: %s (%.2f MB)", self.path, os.path.getsize(self.path) / 1e6)


def map_weight_name(name: str) -> str:
    name = name.replace("model.embed_tokens.weight", "token_embd.weight")
    name = name.replace("model.norm.weight", "output_norm.weight")
    name = name.replace("lm_head.weight", "output.weight")
    name = name.replace("model.layers.", "blk.")
    name = name.replace(".self_attn.", ".attn.")
    name = name.replace(".mlp.", ".ffn.")
    name = name.replace(".input_layernorm.", ".attn_norm.")
    name = name.replace(".post_attention_layernorm.", ".ffn_norm.")
    name = name.replace("q_proj", "q")
    name = name.replace("k_proj", "k")
    name = name.replace("v_proj", "v")
    name = name.replace("o_proj", "o")
    name = name.replace("gate_proj", "gate")
    name = name.replace("up_proj", "up")
    name = name.replace("down_proj", "down")
    return name


def export_to_gguf(
    checkpoint_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    quantize_weights: bool = True,
) -> str:
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
    else:
        pt_files = [f for f in os.listdir(checkpoint_path) if f.endswith((".pt", ".bin"))]
        if pt_files:
            state = torch.load(
                os.path.join(checkpoint_path, pt_files[0]),
                map_location="cpu", weights_only=False,
            )
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)

    writer = GGUFWriter(output_path)

    writer.add_string("general.architecture", "edgebit")
    writer.add_string("general.name", "EdgeBit-350M")
    writer.add_string("general.quantization", config.quant_mode)
    writer.add_string(
        "edgebit.export.note",
        "minimal experimental GGUF-like export; custom packed ternary GGML kernels are not included",
    )
    writer.add_uint32("edgebit.block_count", config.n_layers)
    writer.add_uint32("edgebit.embedding_length", config.hidden_dim)
    writer.add_uint32("edgebit.feed_forward_length", config.ffn_dim)
    writer.add_uint32("edgebit.attention.head_count", config.n_heads)
    writer.add_uint32("edgebit.attention.head_count_kv", config.n_kv_heads)
    writer.add_uint32("edgebit.context_length", config.max_seq_len)
    writer.add_uint32("edgebit.vocab_size", config.vocab_size)
    writer.add_float32("edgebit.attention.layer_norm_rms_epsilon", config.rms_norm_eps)
    writer.add_float32("edgebit.rope.freq_base", config.rope_theta)
    writer.add_uint32("edgebit.quantization.group_size", config.group_size)
    writer.add_bool("edgebit.quantization.kv_cache_quant", config.kv_cache_quant)

    state_dict = model.state_dict()
    for name, tensor in state_dict.items():
        gguf_name = map_weight_name(name)
        data = tensor.detach().cpu()

        is_weight_2d = tensor.ndim == 2 and "weight" in name
        is_quantizable = not any(skip in name for skip in ["embed", "norm", "lm_head", "bias"])

        if quantize_weights and is_weight_2d and is_quantizable:
            data_np = data.to(torch.float16).numpy().astype(np.float16)
            writer.add_tensor(gguf_name, data_np, GGML_TYPE_F16)
        else:
            data_np = data.float().numpy().astype(np.float32)
            writer.add_tensor(gguf_name, data_np, GGML_TYPE_F32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer.write()

    file_size = os.path.getsize(output_path) / 1e6
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("GGUF export: %s (%.1f MB, %d params)", output_path, file_size, param_count)
    return output_path


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Export EdgeBit to GGUF format")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--no_quantize", action="store_true")
    args = parser.parse_args()

    export_to_gguf(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config_path=args.config,
        quantize_weights=not args.no_quantize,
    )


if __name__ == "__main__":
    main()
