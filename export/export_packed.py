#!/usr/bin/env python3
"""Export an EdgeBit training checkpoint to verified packed storage."""
from __future__ import annotations

import argparse
import logging
import os

import torch

from runtime.packed_checkpoint import (
    estimate_packed_state_size_bytes,
    save_packed_checkpoint,
)

logger = logging.getLogger(__name__)


def export_packed(checkpoint_path: str, output_path: str, group_size: int = 128) -> str:
    state_path = checkpoint_path
    if os.path.isdir(state_path):
        state_path = os.path.join(state_path, "training_state.pt")
    checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
    output = save_packed_checkpoint(checkpoint, output_path, group_size=group_size)

    packed = torch.load(output, map_location="cpu", weights_only=False)
    logical_mb = estimate_packed_state_size_bytes(packed) / 1e6
    file_mb = os.path.getsize(output) / 1e6
    logger.info(
        "Packed checkpoint saved to %s (logical %.2f MB, file %.2f MB)",
        output,
        logical_mb,
        file_mb,
    )
    return output


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Export EdgeBit packed checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--group_size", type=int, default=128)
    args = parser.parse_args()
    export_packed(args.checkpoint, args.output, args.group_size)


if __name__ == "__main__":
    main()
