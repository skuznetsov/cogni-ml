#!/usr/bin/env python3
"""Create a bounded full-geometry Wan S2V cross-attention MPS reference.

Q/K/V use a deterministic dyadic formula reproduced by the native benchmark.
The reference runs the same contiguous F32 SDPA adapter used by the Wan S2V
real-stack oracle, in independent query chunks to keep score workspace bounded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# Match the oracle policy before importing torch: unsupported MPS operators
# must fail instead of silently running on CPU.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
import torch.nn.functional as F


Q_ROWS = 36_608
K_ROWS = 512
HEADS = 40
HEAD_DIM = 128
FORMULA = "avalanche32(native_flat_index + salt*0x9e3779b9); ((hash & 1023) - 512) / 256"
SCHEMA = "wan_s2v_cross_attention_reference_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest(path: Path) -> str:
    return sha256_file(path) if path.is_file() else "unavailable"


def dyadic_bhqd(rows: int, salt: int, chunk_rows: int = 512) -> np.ndarray:
    """Return [H, rows, D] matching native [rows, H, D] row-major values."""
    output = np.empty((HEADS, rows, HEAD_DIM), dtype=np.float32)
    d = np.arange(HEAD_DIM, dtype=np.uint64)[None, :]
    for head in range(HEADS):
        for start in range(0, rows, chunk_rows):
            stop = min(rows, start + chunk_rows)
            q = np.arange(start, stop, dtype=np.uint64)[:, None]
            native_index = ((q * HEADS + np.uint64(head)) * HEAD_DIM + d).astype(np.uint32)
            salted = native_index + np.uint32((salt * 0x9E3779B9) & 0xFFFFFFFF)
            mixed = salted ^ (salted >> np.uint32(16))
            mixed = mixed * np.uint32(0x7FEB352D)
            mixed = mixed ^ (mixed >> np.uint32(15))
            mixed = mixed * np.uint32(0x846CA68B)
            mixed = mixed ^ (mixed >> np.uint32(16))
            value = (mixed & np.uint32(1023)).astype(np.int16) - np.int16(512)
            output[head, start:stop, :] = value.astype(np.float32) / np.float32(256.0)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="new or empty fixture directory")
    parser.add_argument(
        "--wan-attention-source",
        type=Path,
        default=Path("/opt/mldata_ssd/cogni-ml/video-scene-20261019/wan-audit/wan_modules_attention.py"),
    )
    parser.add_argument("--query-chunk-rows", type=int, default=512)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for this oracle")
    if args.query_chunk_rows <= 0:
        raise ValueError("query chunk rows must be positive")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("MPS fallback must remain disabled")
    if not args.wan_attention_source.is_file():
        raise FileNotFoundError(f"Pinned Wan attention source is missing: {args.wan_attention_source}")
    reference_adapter = Path(__file__).resolve().parents[2] / "spec/support/wan_s2v_real_stack_reference.py"
    if not reference_adapter.is_file():
        raise FileNotFoundError(f"Wan S2V SDPA adapter source is missing: {reference_adapter}")
    args.output.mkdir(parents=True, exist_ok=True)
    expected_path = args.output / "expected_bhqd.f32"
    if expected_path.exists():
        raise FileExistsError(expected_path)

    device = torch.device("mps")
    # The arrays are created on CPU in bounded row tiles, then transferred as
    # contiguous [H, Q/K, D]. This avoids an O(Q*K) oracle score allocation.
    q_cpu = dyadic_bhqd(Q_ROWS, 1)
    k_cpu = dyadic_bhqd(K_ROWS, 2)
    v_cpu = dyadic_bhqd(K_ROWS, 3)
    q = torch.from_numpy(q_cpu).to(device=device, dtype=torch.float32)
    k = torch.from_numpy(k_cpu).to(device=device, dtype=torch.float32)
    v = torch.from_numpy(v_cpu).to(device=device, dtype=torch.float32)
    del q_cpu, k_cpu, v_cpu

    result = torch.empty((HEADS, Q_ROWS, HEAD_DIM), device=device, dtype=torch.float32)
    for start in range(0, Q_ROWS, args.query_chunk_rows):
        stop = min(Q_ROWS, start + args.query_chunk_rows)
        # This is the exact transpose/contiguous/F.scaled_dot_product_attention/
        # output-layout adapter used by wan_s2v_real_stack_reference.py.
        q_chunk = q[:, start:stop, :].unsqueeze(0).contiguous()
        output = F.scaled_dot_product_attention(
            q_chunk,
            k.unsqueeze(0),
            v.unsqueeze(0),
            attn_mask=None,
            is_causal=False,
            dropout_p=0.0,
        )
        if output.device.type != "mps" or output.dtype != torch.float32:
            raise RuntimeError("SDPA left MPS F32")
        result[:, start:stop, :].copy_(output[0])
        del q_chunk, output

    if not torch.isfinite(result).all().item():
        raise RuntimeError("non-finite upstream SDPA output")
    torch.mps.synchronize()
    result.detach().cpu().contiguous().numpy().tofile(expected_path)
    expected_bytes = Q_ROWS * HEADS * HEAD_DIM * 4
    if expected_path.stat().st_size != expected_bytes:
        raise RuntimeError("unexpected reference output byte count")

    script_path = Path(__file__).resolve()
    manifest = {
        "schema": SCHEMA,
        "geometry": {
            "batches": 1,
            "query_rows": Q_ROWS,
            "key_rows": K_ROWS,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "native_layout": "B,Q,H,D",
            "reference_layout": "B,H,Q,D",
        },
        "input_formula": FORMULA,
        "input_salts": {"q": 1, "k": 2, "v": 3},
        "reference": {
            "operator": "torch.nn.functional.scaled_dot_product_attention",
            "torch_version": torch.__version__,
            "device": "mps",
            "dtype": "float32",
            "mps_fallback_disabled": True,
            "query_chunk_rows": args.query_chunk_rows,
            "chunking_semantics": "independent query rows with full K/V; no mask, causal=false, dropout=0",
            "source_adapter": "transpose to contiguous B,H,Q,D, PyTorch MPS SDPA, return B,H,Q,D",
            "comparison_tolerance": "2e-5 + 2e-5 * abs(reference)",
        },
        "expected": {
            "file": expected_path.name,
            "dtype": "F32_LE",
            "shape": [1, HEADS, Q_ROWS, HEAD_DIM],
            "bytes": expected_bytes,
            "sha256": sha256_file(expected_path),
        },
        "source": {
            "wan_attention_path": str(args.wan_attention_source),
            "wan_attention_sha256": sha256_file(args.wan_attention_source),
            "wan_s2v_reference_adapter_path": str(reference_adapter),
            "wan_s2v_reference_adapter_sha256": sha256_file(reference_adapter),
            "adapter_script": script_path.name,
            "adapter_sha256": sha256_file(script_path),
        },
        "scope": "full-geometry F32 cross-attention oracle only; not a full block or video-quality result",
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fixture": str(args.output), "manifest": str(manifest_path), "sha256": manifest["expected"]["sha256"]}))


if __name__ == "__main__":
    main()
