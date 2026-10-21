#!/usr/bin/env python3
"""Build a full-geometry F32 Wan text-cross-attention MPS reference fixture.

Inputs use the native B,Q,H,D flat-index avalanche fixture. Query rows are
generated and uploaded in bounded chunks; the full K/V tensors are converted
to contiguous B,H,K,D once and reused. SDPA runs only on MPS. The full output
stays on MPS until one final synchronization, then is streamed to a native
B,Q,H,D little-endian F32 file in bounded CPU chunks.
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

# Unsupported MPS operators must fail closed. Set this before importing torch.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
import torch.nn.functional as F


QUERY_ROWS = 36_608
KEY_ROWS = 512
HEADS = 40
HEAD_DIM = 128
CHANNELS = HEADS * HEAD_DIM
FORMULA = (
    "avalanche32(native_flat_index + salt*0x9e3779b9); "
    "((hash & 1023) - 512) / 256"
)
SCHEMA = "wan_s2v_text_cross_cadence_reference_v1"
DEFAULT_WAN_ATTENTION_SOURCE = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/wan-audit/wan_modules_attention.py"
)
WAN_ATTENTION_SOURCE_SHA256 = (
    "23fe7c6f6e4065242d95e5e188cb2d1a16bd283f05dca7e7e878977158fcfdbc"
)
ADAPTER_SOURCE_SHA256 = (
    "863f0020418f3da40caf6f2863dbef9ed40d147127c34b0748cf3415433333a3"
)
EXPECTED_BYTES = QUERY_ROWS * HEADS * HEAD_DIM * 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dyadic_bqhd(
    rows: int, salt: int, row_start: int = 0
) -> np.ndarray:
    """Generate [rows,H,D] by native global B,Q,H,D flat index."""
    output = np.empty((rows, HEADS, HEAD_DIM), dtype=np.float32)
    channel = np.arange(CHANNELS, dtype=np.uint64)[None, :]
    query = np.arange(row_start, row_start + rows, dtype=np.uint64)[:, None]
    native_index = (query * CHANNELS + channel).astype(np.uint32)
    mixed = native_index + np.uint32((salt * 0x9E3779B9) & 0xFFFFFFFF)
    mixed ^= mixed >> np.uint32(16)
    mixed *= np.uint32(0x7FEB352D)
    mixed ^= mixed >> np.uint32(15)
    mixed *= np.uint32(0x846CA68B)
    mixed ^= mixed >> np.uint32(16)
    quantized = (mixed & np.uint32(1023)).astype(np.int16) - np.int16(512)
    output[:] = quantized.astype(np.float32).reshape(rows, HEADS, HEAD_DIM) / np.float32(
        256.0
    )
    return output


def upload_bqhd(rows: int, salt: int, device: torch.device, row_start: int = 0):
    started = time.perf_counter()
    cpu_values = dyadic_bqhd(rows, salt, row_start)
    construction_seconds = time.perf_counter() - started
    upload_started = time.perf_counter()
    native = torch.from_numpy(cpu_values).unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    upload_enqueue_seconds = time.perf_counter() - upload_started
    del cpu_values
    return native, construction_seconds, upload_enqueue_seconds


def wan_sdpa_query_chunk(
    q_bqhd: torch.Tensor, k_bhkd: torch.Tensor, v_bhkd: torch.Tensor
) -> torch.Tensor:
    """Match the pinned real-stack transpose/F32 MPS SDPA adapter."""
    q_bhqd = q_bqhd.transpose(1, 2).contiguous()
    if (
        q_bhqd.device.type != "mps"
        or q_bhqd.dtype != torch.float32
        or not q_bhqd.is_contiguous()
        or k_bhkd.device.type != "mps"
        or k_bhkd.dtype != torch.float32
        or not k_bhkd.is_contiguous()
        or v_bhkd.device.type != "mps"
        or v_bhkd.dtype != torch.float32
        or not v_bhkd.is_contiguous()
    ):
        raise RuntimeError("Wan cross-attention SDPA operands must be contiguous MPS F32")

    output_bhqd = F.scaled_dot_product_attention(
        q_bhqd,
        k_bhkd,
        v_bhkd,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
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
        "--wan-attention-source",
        type=Path,
        default=DEFAULT_WAN_ATTENTION_SOURCE,
    )
    parser.add_argument(
        "--query-chunk-rows",
        type=int,
        choices=(128, 256),
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
            f"Pinned Wan attention source is missing: {args.wan_attention_source}"
        )
    upstream_sha256 = sha256_file(args.wan_attention_source)
    if upstream_sha256 != WAN_ATTENTION_SOURCE_SHA256:
        raise RuntimeError(
            "Pinned Wan attention source SHA-256 mismatch: "
            f"{args.wan_attention_source} ({upstream_sha256})"
        )

    repo_root = Path(__file__).resolve().parents[2]
    adapter_source = repo_root / "spec/support/wan_s2v_real_stack_reference.py"
    if not adapter_source.is_file():
        raise FileNotFoundError(f"Wan S2V SDPA adapter source is missing: {adapter_source}")
    adapter_sha256 = sha256_file(adapter_source)
    if adapter_sha256 != ADAPTER_SOURCE_SHA256:
        raise RuntimeError(
            "Pinned Wan S2V SDPA adapter SHA-256 mismatch: "
            f"{adapter_source} ({adapter_sha256})"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    expected_path = args.output / "expected_bqhd.f32"
    manifest_path = args.output / "manifest.json"
    if expected_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Reference fixture output already exists; use a new or empty directory"
        )

    device = torch.device("mps")
    input_setup_started = time.perf_counter()

    k_bqhd, k_cpu_seconds, k_upload_enqueue_seconds = upload_bqhd(
        KEY_ROWS, 2, device
    )
    v_bqhd, v_cpu_seconds, v_upload_enqueue_seconds = upload_bqhd(
        KEY_ROWS, 3, device
    )
    layout_conversion_started = time.perf_counter()
    k_bhkd = k_bqhd.transpose(1, 2).contiguous()
    v_bhkd = v_bqhd.transpose(1, 2).contiguous()
    layout_conversion_enqueue_seconds = time.perf_counter() - layout_conversion_started
    del k_bqhd, v_bqhd

    result_bqhd = torch.empty(
        (1, QUERY_ROWS, HEADS, HEAD_DIM), device=device, dtype=torch.float32
    )
    input_setup_seconds = time.perf_counter() - input_setup_started

    query_cpu_seconds = 0.0
    query_upload_enqueue_seconds = 0.0
    oracle_started = time.perf_counter()
    finite = torch.ones((), device=device, dtype=torch.bool)
    for start in range(0, QUERY_ROWS, args.query_chunk_rows):
        stop = min(QUERY_ROWS, start + args.query_chunk_rows)
        q_bqhd, cpu_seconds, upload_enqueue_seconds = upload_bqhd(
            stop - start, 1, device, row_start=start
        )
        query_cpu_seconds += cpu_seconds
        query_upload_enqueue_seconds += upload_enqueue_seconds
        output_bhqd = wan_sdpa_query_chunk(q_bqhd, k_bhkd, v_bhkd)
        result_bqhd[:, start:stop, :, :].copy_(output_bhqd.transpose(1, 2))
        finite = finite & torch.isfinite(output_bhqd).all()
        del q_bqhd, output_bhqd

    # Retire all SDPA and copy work together; export follows this one explicit wait.
    torch.mps.synchronize()
    oracle_wall_seconds = time.perf_counter() - oracle_started
    finite_gate_started = time.perf_counter()
    if not bool(finite.item()):
        raise RuntimeError("non-finite upstream SDPA output")
    finite_gate_seconds = time.perf_counter() - finite_gate_started

    export_started = time.perf_counter()
    with expected_path.open("xb") as stream:
        for start in range(0, QUERY_ROWS, args.query_chunk_rows):
            stop = min(QUERY_ROWS, start + args.query_chunk_rows)
            host_chunk = (
                result_bqhd[0, start:stop]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
            )
            host_chunk.astype(np.dtype("<f4"), copy=False).tofile(stream)
            del host_chunk
    output_export_seconds = time.perf_counter() - export_started
    if expected_path.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(
            f"unexpected reference output byte count: {expected_path.stat().st_size}"
        )

    generator_path = Path(__file__).resolve()
    manifest = {
        "schema": SCHEMA,
        "kind": "wan_s2v_text_cross_cadence",
        "geometry": {
            "batches": 1,
            "query_rows": QUERY_ROWS,
            "key_rows": KEY_ROWS,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "native_layout": "B,Q,H,D",
            "reference_layout": "B,Q,H,D",
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
            "chunking_semantics": (
                "independent query spans with full K/V; no mask, noncausal, dropout=0"
            ),
            "source_adapter": (
                "transpose Q/K/V to contiguous B,H,Q,D; PyTorch MPS SDPA; "
                "return contiguous B,Q,H,D"
            ),
            "comparison_tolerance": "2e-5 + 2e-5 * abs(reference)",
            "terminal_sync_semantics": (
                "one explicit torch.mps.synchronize() after all chunks; finite-scalar "
                ".item() and streamed CPU readbacks follow as diagnostic export work "
                "and may entail implicit waits outside native measured arms"
            ),
        },
        "experiment": {
            "dispatches_per_command_root": [1, 2, 4],
            "groups_per_segment": 320,
            "kernel": "attention_tiled_range",
            "kernel_unchanged": True,
        },
        "expected": {
            "file": expected_path.name,
            "dtype": "F32_LE",
            "shape": [1, QUERY_ROWS, HEADS, HEAD_DIM],
            "bytes": EXPECTED_BYTES,
            "sha256": sha256_file(expected_path),
        },
        "source": {
            "wan_attention_source_path": str(args.wan_attention_source.resolve()),
            "wan_attention_source_sha256": upstream_sha256,
            "wan_s2v_reference_adapter_path": str(adapter_source.resolve()),
            "wan_s2v_reference_adapter_sha256": adapter_sha256,
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
            "k_cpu_fixture_construction": k_cpu_seconds,
            "v_cpu_fixture_construction": v_cpu_seconds,
            "q_cpu_fixture_construction": query_cpu_seconds,
            "mps_kv_transfer_enqueue": k_upload_enqueue_seconds
            + v_upload_enqueue_seconds,
            "mps_q_transfer_enqueue": query_upload_enqueue_seconds,
            "mps_kv_layout_conversion_enqueue": layout_conversion_enqueue_seconds,
            "input_setup_host_wall": input_setup_seconds,
            "mps_oracle_including_one_terminal_wait": oracle_wall_seconds,
            "mps_finite_scalar_readback": finite_gate_seconds,
            "streamed_output_readback_and_export": output_export_seconds,
        },
        "scope": (
            "full-geometry F32 text cross-attention SDPA oracle only; "
            "not a full S2V block, benchmark result, or video-quality result"
        ),
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

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
