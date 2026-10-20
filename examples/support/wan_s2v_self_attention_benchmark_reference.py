#!/usr/bin/env python3
"""Build a full-geometry F32 Wan self-attention MPS reference fixture.

Q/K/V values follow the native Metal benchmark's flat-index avalanche formula.
The oracle uses the Wan real-stack SDPA adapter in independent query chunks so
the quadratic score workspace stays bounded. All SDPA math runs on MPS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

# Set this before importing torch: unsupported MPS operators must fail closed.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
import torch.nn.functional as F


Q_ROWS = 36_608
K_ROWS = 36_608
HEADS = 40
HEAD_DIM = 128
CHANNELS = HEADS * HEAD_DIM
FORMULA = (
    "avalanche32(native_flat_index + salt*0x9e3779b9); ((hash & 1023) - 512) / 256"
)
SCHEMA = "wan_s2v_self_attention_reference_v1"
DEFAULT_WAN_ATTENTION_SOURCE = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/wan-audit/wan_modules_attention.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dyadic_bqhd(rows: int, salt: int, tile_rows: int = 256) -> np.ndarray:
    """Return [rows,H,D] values indexed exactly like native B,Q,H,D storage."""
    output = np.empty((rows, HEADS, HEAD_DIM), dtype=np.float32)
    channel = np.arange(CHANNELS, dtype=np.uint64)[None, :]
    salt32 = np.uint32((salt * 0x9E3779B9) & 0xFFFFFFFF)
    for start in range(0, rows, tile_rows):
        stop = min(rows, start + tile_rows)
        query = np.arange(start, stop, dtype=np.uint64)[:, None]
        native_index = (query * CHANNELS + channel).astype(np.uint32)
        mixed = native_index + salt32
        mixed ^= mixed >> np.uint32(16)
        mixed *= np.uint32(0x7FEB352D)
        mixed ^= mixed >> np.uint32(15)
        mixed *= np.uint32(0x846CA68B)
        mixed ^= mixed >> np.uint32(16)
        quantized = (mixed & np.uint32(1023)).astype(np.int16) - np.int16(512)
        output[start:stop] = quantized.astype(np.float32).reshape(
            stop - start, HEADS, HEAD_DIM
        ) / np.float32(256.0)
    return output


def upload_input(
    rows: int, salt: int, device: torch.device
) -> tuple[torch.Tensor, float, float]:
    started = time.perf_counter()
    cpu_values = dyadic_bqhd(rows, salt)
    construction_seconds = time.perf_counter() - started
    upload_started = time.perf_counter()
    native = (
        torch.from_numpy(cpu_values).unsqueeze(0).to(device=device, dtype=torch.float32)
    )
    upload_enqueue_seconds = time.perf_counter() - upload_started
    del cpu_values
    return native, construction_seconds, upload_enqueue_seconds


def wan_sdpa_query_chunk(
    q_bqhd: torch.Tensor,
    k_bhkd: torch.Tensor,
    v_bhkd: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    """Apply the real-stack transpose/F32 MPS SDPA adapter to one query span."""
    q_chunk = q_bqhd[:, start:stop, :, :].transpose(1, 2).contiguous()
    if (
        q_chunk.device.type != "mps"
        or q_chunk.dtype != torch.float32
        or not q_chunk.is_contiguous()
    ):
        raise RuntimeError("Wan self-attention query chunk must be contiguous MPS F32")
    output_bhqd = F.scaled_dot_product_attention(
        q_chunk,
        k_bhkd,
        v_bhkd,
        attn_mask=None,
        is_causal=False,
        dropout_p=0.0,
    )
    if output_bhqd.device.type != "mps" or output_bhqd.dtype != torch.float32:
        raise RuntimeError("MPS SDPA left MPS F32")
    return output_bhqd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new or empty reference fixture directory",
    )
    parser.add_argument(
        "--wan-attention-source", type=Path, default=DEFAULT_WAN_ATTENTION_SOURCE
    )
    parser.add_argument(
        "--query-chunk-rows",
        type=int,
        choices=(64, 128, 256),
        default=128,
        help="independent query rows per SDPA call (default: 128)",
    )
    args = parser.parse_args()

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("MPS fallback must remain disabled")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for this oracle")
    if not torch.__version__.split("+", maxsplit=1)[0].startswith("2.8."):
        raise RuntimeError(
            f"PyTorch 2.8.x is required for this oracle, got {torch.__version__}"
        )
    if not args.wan_attention_source.is_file():
        raise FileNotFoundError(
            f"Wan attention source for provenance is missing: {args.wan_attention_source}"
        )

    repo_root = Path(__file__).resolve().parents[2]
    adapter_source = repo_root / "spec/support/wan_s2v_real_stack_reference.py"
    if not adapter_source.is_file():
        raise FileNotFoundError(
            f"Wan S2V SDPA adapter source is missing: {adapter_source}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    expected_path = args.output / "expected_bhqd.f32"
    manifest_path = args.output / "manifest.json"
    if expected_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Reference fixture output already exists; use a new or empty directory"
        )

    device = torch.device("mps")
    input_setup_started = time.perf_counter()
    q, q_cpu_seconds, q_upload_enqueue_seconds = upload_input(Q_ROWS, 1, device)
    k, k_cpu_seconds, k_upload_enqueue_seconds = upload_input(K_ROWS, 2, device)
    v, v_cpu_seconds, v_upload_enqueue_seconds = upload_input(K_ROWS, 3, device)

    # Match the upstream adapter's B,Q,H,D -> contiguous B,H,Q,D conversion.
    # Reuse the converted K/V across independent query chunks to avoid repeated
    # full-key copies or score tensors on the host.
    layout_conversion_started = time.perf_counter()
    k_bhkd = k.transpose(1, 2).contiguous()
    v_bhkd = v.transpose(1, 2).contiguous()
    layout_conversion_enqueue_seconds = time.perf_counter() - layout_conversion_started
    del k, v
    input_wait_started = time.perf_counter()
    torch.mps.synchronize()
    input_terminal_wait_seconds = time.perf_counter() - input_wait_started
    input_setup_seconds = time.perf_counter() - input_setup_started

    result_bhqd = torch.empty(
        (1, HEADS, Q_ROWS, HEAD_DIM), device=device, dtype=torch.float32
    )
    oracle_started = time.perf_counter()
    for start in range(0, Q_ROWS, args.query_chunk_rows):
        stop = min(Q_ROWS, start + args.query_chunk_rows)
        output_bhqd = wan_sdpa_query_chunk(q, k_bhkd, v_bhkd, start, stop)
        result_bhqd[:, :, start:stop, :].copy_(output_bhqd)
        del output_bhqd
    finite = torch.isfinite(result_bhqd).all()
    torch.mps.synchronize()
    oracle_wall_seconds = time.perf_counter() - oracle_started
    if not bool(finite.item()):
        raise RuntimeError("non-finite upstream SDPA output")

    export_started = time.perf_counter()
    result_bhqd.detach().cpu().contiguous().numpy().tofile(expected_path)
    output_export_seconds = time.perf_counter() - export_started
    expected_bytes = Q_ROWS * HEADS * HEAD_DIM * 4
    if expected_path.stat().st_size != expected_bytes:
        raise RuntimeError("unexpected reference output byte count")

    generator_path = Path(__file__).resolve()
    manifest = {
        "schema": SCHEMA,
        "provenance_validation": "recorded source and output hashes are consistency checks, not a trusted digest allowlist",
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
            "source_adapter": "wan_s2v_real_stack_reference.py: transpose contiguous B,H,Q,D, PyTorch MPS SDPA, return B,H,Q,D",
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
            "wan_attention_source_path": str(args.wan_attention_source.resolve()),
            "wan_attention_source_sha256": sha256_file(args.wan_attention_source),
            "wan_s2v_reference_adapter_path": str(adapter_source.resolve()),
            "wan_s2v_reference_adapter_sha256": sha256_file(adapter_source),
            "generator_path": str(generator_path),
            "generator_sha256": sha256_file(generator_path),
        },
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "device": str(device),
            "mps_available": True,
            "mps_fallback_disabled": True,
        },
        "timing_seconds": {
            "q_cpu_fixture_construction": q_cpu_seconds,
            "k_cpu_fixture_construction": k_cpu_seconds,
            "v_cpu_fixture_construction": v_cpu_seconds,
            "mps_qkv_transfer_enqueue": q_upload_enqueue_seconds
            + k_upload_enqueue_seconds
            + v_upload_enqueue_seconds,
            "mps_kv_layout_conversion_enqueue": layout_conversion_enqueue_seconds,
            "mps_input_terminal_wait": input_terminal_wait_seconds,
            "input_setup_host_wall_including_one_wait": input_setup_seconds,
            "mps_oracle_including_one_terminal_wait": oracle_wall_seconds,
            "final_output_readback_and_export": output_export_seconds,
        },
        "scope": "full-geometry F32 self-attention SDPA oracle only; not a full S2V block or video-quality result",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "fixture": str(args.output.resolve()),
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "expected_sha256": manifest["expected"]["sha256"],
                "query_chunk_rows": args.query_chunk_rows,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
