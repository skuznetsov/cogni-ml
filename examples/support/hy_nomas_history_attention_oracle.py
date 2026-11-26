#!/usr/bin/env python3
"""Prepare a deterministic CPU F32 attention oracle for the native probe.

This is fixture preparation only: it loads no model weights and never selects a
CUDA device. The operation mirrors the pinned upstream `attn_mode == "torch"`
path in `hymm/models/dit/nomas_cpt.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
UPSTREAM_RELATIVE_SOURCE = "hymm/models/dit/nomas_cpt.py"
UPSTREAM_SOURCE_SHA256 = "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee"
QUERY_ROWS = 257
HISTORY_ROWS = 23
HEADS = 16
HEAD_DIM = 128


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_tensor(directory: Path, name: str, tensor: torch.Tensor) -> dict[str, object]:
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise RuntimeError(f"{name} must be a CPU F32 tensor")
    contiguous = tensor.detach().contiguous()
    data = contiguous.numpy().astype("<f4", copy=False).tobytes(order="C")
    filename = f"{name}.f32le"
    (directory / filename).write_bytes(data)
    return {
        "file": filename,
        "shape": list(contiguous.shape),
        "dtype": "F32",
        "byte_order": "little",
        "layout": "contiguous BTHD" if contiguous.ndim == 4 else "contiguous THD",
        "nbytes": len(data),
        "sha256": sha256(data),
    }


def deterministic_tensor(shape: tuple[int, ...], salt: int) -> torch.Tensor:
    count = 1
    for dimension in shape:
        count *= dimension
    index = torch.arange(count, dtype=torch.int64, device="cpu")
    numerator = torch.remainder(index * 37 + salt * 997, 4093) - 2046
    return (numerator.to(torch.float32) / 4096.0).reshape(shape).contiguous()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--upstream-source", required=True, type=Path)
    args = parser.parse_args()

    source = args.upstream_source.read_bytes()
    source_sha = sha256(source)
    if source_sha != UPSTREAM_SOURCE_SHA256:
        raise SystemExit(
            "upstream source hash mismatch: expected pinned nomas_cpt.py "
            f"{UPSTREAM_SOURCE_SHA256}, got {source_sha}"
        )
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"refusing to replace existing fixture directory: {args.output_dir}")

    torch.set_num_threads(1)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    query = deterministic_tensor((1, QUERY_ROWS, HEADS, HEAD_DIM), 11)
    current_key = deterministic_tensor((1, QUERY_ROWS, HEADS, HEAD_DIM), 23)
    current_value = deterministic_tensor((1, QUERY_ROWS, HEADS, HEAD_DIM), 41)
    history_key = deterministic_tensor((1, HISTORY_ROWS, HEADS, HEAD_DIM), 59)
    history_value = deterministic_tensor((1, HISTORY_ROWS, HEADS, HEAD_DIM), 73)

    tensors = {
        "query": write_tensor(output_dir, "query", query),
        "current_key": write_tensor(output_dir, "current_key", current_key),
        "current_value": write_tensor(output_dir, "current_value", current_value),
        "history_key": write_tensor(output_dir, "history_key", history_key),
        "history_value": write_tensor(output_dir, "history_value", history_value),
    }

    cases: list[dict[str, object]] = []
    q_bhsd = query.transpose(1, 2).contiguous()
    for history_rows in (0, HISTORY_ROWS):
        if history_rows:
            key = torch.cat((history_key, current_key), dim=1)
            value = torch.cat((history_value, current_value), dim=1)
        else:
            key = current_key
            value = current_value
        # Same operation and layout conversion as the pinned upstream torch path.
        output = F.scaled_dot_product_attention(
            q_bhsd,
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).contiguous()
        expected = write_tensor(output_dir, f"expected_p{history_rows}", output)
        cases.append({"history_rows": history_rows, "expected": expected})

    manifest = {
        "schema": "cogni_ml.hy_nomas_history_attention_oracle",
        "schema_version": 1,
        "upstream": {
            "repository": "video-scene-20261019/upstream/hy-nomas-20261024",
            "revision": UPSTREAM_REVISION,
            "source_file": UPSTREAM_RELATIVE_SOURCE,
            "source_file_sha256": source_sha,
            "route": "hymm/models/dit/nomas_cpt.py attn_mode == 'torch' branch",
            "equation": "F.scaled_dot_product_attention(Q^T,K^T,V^T, dropout_p=0.0, is_causal=False), then transpose output to BTHD",
        },
        "oracle": {
            "script_sha256": sha256(Path(__file__).resolve().read_bytes()),
            "torch_version": torch.__version__,
            "device": "cpu",
            "dtype": "torch.float32",
            "mask": "none; every provided K/V row is valid",
        },
        "geometry": {
            "batch": 1,
            "query_rows": QUERY_ROWS,
            "history_rows": HISTORY_ROWS,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "layout": "BTHD contiguous, history prefix followed by current rows",
        },
        "tensors": tensors,
        "cases": cases,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={sha256(manifest_bytes)}")
    print("oracle_device=cpu model_weights=none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
