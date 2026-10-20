#!/usr/bin/env python3
"""Generate a small pinned PyTorch MPS fixture for the tiled video linear op."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import struct
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
import torch.nn.functional as F


def bf16_truncate(values: list[float]) -> list[float]:
    """Model the raw resident BF16 upper-word encoding used by the spec."""
    result = []
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0] & 0xFFFF0000
        result.append(struct.unpack("<f", struct.pack("<I", bits))[0])
    return result


def tensor_on_mps(values: list[float], shape: tuple[int, ...], use_bf16: bool) -> torch.Tensor:
    source = values
    if use_bf16:
        source = bf16_truncate(values)
    return torch.tensor(source, dtype=torch.float32, device="mps").reshape(shape)


def floats(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().to("cpu").reshape(-1).tolist()


def case_outputs(x: list[float], w: list[float], b: list[float],
                 rows: int, in_dim: int, out_dim: int) -> list[dict[str, object]]:
    cases = []
    for input_dtype in ("f32", "bf16"):
        for weight_dtype in ("f32", "bf16"):
            for bias_dtype in ("f32", "bf16"):
                x_mps = tensor_on_mps(x, (rows, in_dim), input_dtype == "bf16")
                w_mps = tensor_on_mps(w, (out_dim, in_dim), weight_dtype == "bf16")
                b_mps = tensor_on_mps(b, (out_dim,), bias_dtype == "bf16")
                with torch.no_grad():
                    y_mps = F.linear(x_mps, w_mps, b_mps)
                cases.append({
                    "input_dtype": input_dtype,
                    "weight_dtype": weight_dtype,
                    "bias_dtype": bias_dtype,
                    "output": floats(y_mps),
                })
    return cases


def cancellation_values(rows: int, in_dim: int, out_dim: int) -> tuple[list[float], list[float], list[float]]:
    # Binary fractions make the source values deterministic while alternating
    # weight signs force a long reduction to retain small residual terms.
    x = [(((row + k) % 13) - 6) * 0.0625 + ((row * 3 + k % 5) % 3) * 0.03125
         for row in range(rows) for k in range(in_dim)]
    w = []
    for column in range(out_dim):
        for k in range(in_dim):
            major = 0.125 if (k + column) % 2 == 0 else -0.125
            residual = (((k * 17 + column * 13) % 9) - 4) * (1.0 / 65536.0)
            w.append(major + residual)
    b = [((column % 3) - 1) * 0.03125 for column in range(out_dim)]
    return x, w, b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=19)
    parser.add_argument("--in-dim", type=int, default=19)
    parser.add_argument("--out-dim", type=int, default=13)
    args = parser.parse_args()
    if min(args.rows, args.in_dim, args.out_dim) <= 0:
        parser.error("shape dimensions must be positive")
    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is required; fallback is disabled")

    rng = random.Random(0x51A7)
    tiny_x = [rng.uniform(-0.75, 0.75) for _ in range(args.rows * args.in_dim)]
    tiny_w = [rng.uniform(-0.25, 0.25) for _ in range(args.out_dim * args.in_dim)]
    tiny_b = [rng.uniform(-0.2, 0.2) for _ in range(args.out_dim)]
    groups = [{
        "shape": {"rows": args.rows, "in_dim": args.in_dim, "out_dim": args.out_dim},
        "input": tiny_x,
        "weight": tiny_w,
        "bias": tiny_b,
        "cases": case_outputs(tiny_x, tiny_w, tiny_b, args.rows, args.in_dim, args.out_dim),
    }]
    for in_dim in (5120, 13824):
        rows, out_dim = 3, 5
        x, w, b = cancellation_values(rows, in_dim, out_dim)
        groups.append({
            "shape": {"rows": rows, "in_dim": in_dim, "out_dim": out_dim},
            "input": x,
            "weight": w,
            "bias": b,
            "cases": case_outputs(x, w, b, rows, in_dim, out_dim),
        })

    payload = {
        "schema": "video-linear-tiled-mps-v2",
        "upstream": {
            "oracle_device": "mps",
            "operation": "torch.nn.functional.linear",
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "mps_fallback_disabled": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0",
        },
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "shapes": [group["shape"] for group in groups],
        "case_count": sum(len(group["cases"]) for group in groups),
        "f32_flops": sum(2 * group["shape"]["rows"] * group["shape"]["in_dim"] * group["shape"]["out_dim"]
                         for group in groups),
        "torch_version": torch.__version__,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
