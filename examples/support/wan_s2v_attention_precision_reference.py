#!/usr/bin/env python3
"""Build a bounded MPS SDPA reference for Wan attention input dtypes.

Q/K/V follow the native B,Q,H,D avalanche fixture. Query chunks are always
128 rows; a short final chunk is zero-padded and trimmed after attention. The
pinned Wan ``attention`` fallback runs on MPS, and its output is widened to
F32 on MPS before the BHQD fixture is streamed to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import ModuleType

# Unsupported MPS operators must fail instead of silently running on CPU.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch


SCHEMA = "wan_s2v_attention_precision_reference_v1"
K_ROWS_DEFAULT = 36_608
HEADS_DEFAULT = 40
HEAD_DIM_DEFAULT = 128
QUERY_CHUNK_ROWS = 128
GENERATOR_CHUNK_ROWS = 512
F32_BYTES = 4
EXPLICIT_BUFFER_LIMIT = 3 * 1024**3
FORMULA = (
    "avalanche32(native_flat_index + salt*0x9e3779b9); "
    "((hash & 1023) - 512) / 256"
)
DEFAULT_WAN_ATTENTION_SOURCE = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/wan-audit/wan_modules_attention.py"
)
WAN_ATTENTION_SOURCE_SHA256 = (
    "23fe7c6f6e4065242d95e5e188cb2d1a16bd283f05dca7e7e878977158fcfdbc"
)
DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}
DTYPE_BYTES = {"f32": 4, "f16": 2, "bf16": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dyadic_bqhd(
    rows: int,
    heads: int,
    head_dim: int,
    salt: int,
    row_start: int = 0,
) -> np.ndarray:
    """Generate [rows,H,D] values from global native B,Q,H,D flat indices."""
    channels = heads * head_dim
    output = np.empty((rows, heads, head_dim), dtype=np.float32)
    channel = np.arange(channels, dtype=np.uint64)[None, :]
    query = np.arange(row_start, row_start + rows, dtype=np.uint64)[:, None]
    native_index = (query * channels + channel).astype(np.uint32)
    mixed = native_index + np.uint32((salt * 0x9E3779B9) & 0xFFFFFFFF)
    mixed ^= mixed >> np.uint32(16)
    mixed *= np.uint32(0x7FEB352D)
    mixed ^= mixed >> np.uint32(15)
    mixed *= np.uint32(0x846CA68B)
    mixed ^= mixed >> np.uint32(16)
    quantized = (mixed & np.uint32(1023)).astype(np.int16) - np.int16(512)
    output[:] = quantized.astype(np.float32).reshape(rows, heads, head_dim) / np.float32(
        256.0
    )
    return output


def upload_bqhd(
    rows: int,
    heads: int,
    head_dim: int,
    salt: int,
    row_start: int,
    device: torch.device,
    target_dtype: torch.dtype,
) -> tuple[torch.Tensor, float, float]:
    """Stream bounded CPU fixture tiles into one resident MPS target tensor."""
    upload_started = time.perf_counter()
    native = torch.empty(
        (1, rows, heads, head_dim), device=device, dtype=target_dtype
    )
    construction_seconds = 0.0
    for chunk_start in range(0, rows, GENERATOR_CHUNK_ROWS):
        chunk_rows = min(GENERATOR_CHUNK_ROWS, rows - chunk_start)
        construct_started = time.perf_counter()
        cpu_values = dyadic_bqhd(
            chunk_rows,
            heads,
            head_dim,
            salt,
            row_start + chunk_start,
        )
        construction_seconds += time.perf_counter() - construct_started

        cpu_tensor = torch.from_numpy(cpu_values).unsqueeze(0)
        gpu_f32 = cpu_tensor.to(device=device, dtype=torch.float32)
        if target_dtype == torch.float32:
            gpu_target = gpu_f32
        else:
            gpu_target = gpu_f32.to(dtype=target_dtype)
        native[:, chunk_start : chunk_start + chunk_rows, :, :].copy_(gpu_target)
        del cpu_values, cpu_tensor, gpu_f32, gpu_target
    upload_cast_seconds = time.perf_counter() - upload_started
    return native, construction_seconds, upload_cast_seconds


def load_pinned_attention_source(path: Path) -> ModuleType:
    """Import the pinned module with both optional FlashAttention imports blocked."""
    source_sha256 = sha256_file(path)
    if source_sha256 != WAN_ATTENTION_SOURCE_SHA256:
        raise RuntimeError(
            "Pinned Wan attention source SHA-256 mismatch: "
            f"expected {WAN_ATTENTION_SOURCE_SHA256}, got {source_sha256}"
        )

    spec = importlib.util.spec_from_file_location(
        "_wan_s2v_pinned_attention_precision", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pinned Wan attention source: {path}")
    module = importlib.util.module_from_spec(spec)

    blocked_names = ("flash_attn_interface", "flash_attn")
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in blocked_names}
    sys.modules[spec.name] = module
    try:
        for name in blocked_names:
            sys.modules[name] = None  # type: ignore[assignment]
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(f"Failed to import pinned Wan attention source: {exc}") from exc
    finally:
        sys.modules.pop(spec.name, None)
        for name, prior in saved.items():
            if prior is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior  # type: ignore[assignment]

    # Keep the fallback route explicit even if the module's import behavior changes.
    module.FLASH_ATTN_2_AVAILABLE = False
    module.FLASH_ATTN_3_AVAILABLE = False
    if not callable(getattr(module, "attention", None)):
        raise RuntimeError("Pinned Wan source does not expose attention()")
    if module.FLASH_ATTN_2_AVAILABLE or module.FLASH_ATTN_3_AVAILABLE:
        raise RuntimeError("FlashAttention must be unavailable for this MPS oracle")
    return module


def estimated_explicit_peak_bytes(
    query_rows: int,
    key_rows: int,
    heads: int,
    head_dim: int,
    input_dtype: str,
) -> int:
    """Conservative user-tensor estimate; excludes opaque MPSGraph workspace."""
    channels = heads * head_dim
    target_kv = key_rows * channels * DTYPE_BYTES[input_dtype]
    resident_kv = 2 * target_kv
    output = query_rows * channels * F32_BYTES
    q_chunk = QUERY_CHUNK_ROWS * channels * DTYPE_BYTES[input_dtype]
    output_chunk = QUERY_CHUNK_ROWS * channels * F32_BYTES
    output_native_chunk = QUERY_CHUNK_ROWS * channels * DTYPE_BYTES[input_dtype]
    transfer_rows = min(key_rows, GENERATOR_CHUNK_ROWS)
    transfer_f32 = transfer_rows * channels * F32_BYTES
    transfer_target = transfer_rows * channels * DTYPE_BYTES[input_dtype]
    q_transfer_f32 = QUERY_CHUNK_ROWS * channels * F32_BYTES
    q_transfer_target = QUERY_CHUNK_ROWS * channels * DTYPE_BYTES[input_dtype]
    # K/V are resident at target dtype while one bounded V tile is transferred.
    input_setup_peak = 2 * target_kv + transfer_f32 + transfer_target
    # A short tail may briefly hold its valid Q tensor and the padded Q tensor.
    query_setup_peak = (
        resident_kv
        + output
        + 2 * q_chunk
        + q_transfer_f32
        + q_transfer_target
    )
    attention_peak = (
        resident_kv + output + q_chunk + output_native_chunk + output_chunk
    )
    return max(input_setup_peak, query_setup_peak, attention_peak)


def estimated_cpu_generator_peak_bytes(
    chunk_rows: int, heads: int, head_dim: int
) -> int:
    """Bound explicit NumPy arrays; allow 24 bytes per generated scalar."""
    channels = heads * head_dim
    # Covers output, native index, mixed hash, a mask temporary, quantized
    # values, and the F32 conversion temporary, plus row/channel index vectors.
    return chunk_rows * channels * 24 + (chunk_rows + channels) * 8


def _dtype_name(dtype_name: str) -> str:
    return {
        "f32": "float32",
        "f16": "float16",
        "bf16": "bfloat16",
    }[dtype_name]


def run_reference(args: argparse.Namespace) -> dict[str, object]:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("MPS fallback must remain disabled")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for this oracle")
    torch_version = torch.__version__.split("+", maxsplit=1)[0]
    if not torch_version.startswith("2.8."):
        raise RuntimeError(f"PyTorch 2.8.x is required, got {torch.__version__}")
    if args.query_rows <= 0 or args.query_start < 0:
        raise ValueError("query rows must be positive and query start nonnegative")
    if args.key_rows <= 0 or args.heads <= 0 or args.head_dim <= 0:
        raise ValueError("key rows, heads, and head dimension must be positive")
    if args.query_start + args.query_rows > args.key_rows:
        raise ValueError("selected query range must fit within the self-attention extent")
    if args.q_salt < 0 or args.q_salt > 0xFFFFFFFF:
        raise ValueError("q salt must be an unsigned 32-bit integer")
    peak_bytes = estimated_explicit_peak_bytes(
        args.query_rows, args.key_rows, args.heads, args.head_dim, args.dtype
    )
    if peak_bytes > EXPLICIT_BUFFER_LIMIT:
        raise ValueError(
            "estimated explicit MPS tensors exceed the 3 GiB cap "
            f"({peak_bytes} bytes; opaque MPSGraph workspace is excluded)"
        )

    source_path = args.wan_attention_source.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Pinned Wan attention source is missing: {source_path}")
    source_sha256 = sha256_file(source_path)
    upstream = load_pinned_attention_source(source_path)

    output_dir = args.output.resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError("Output directory must be new or empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    expected_path = output_dir / "expected_bhqd.f32"
    expected_tmp_path = output_dir / ".expected_bhqd.f32.tmp"
    manifest_path = output_dir / "manifest.json"
    manifest_tmp_path = output_dir / ".manifest.json.tmp"
    if any(path.exists() for path in (expected_path, expected_tmp_path, manifest_path, manifest_tmp_path)):
        raise FileExistsError("Reference fixture output already exists")

    device = torch.device("mps")
    target_dtype = DTYPES[args.dtype]
    input_setup_started = time.perf_counter()
    k_bqhd, k_cpu_seconds, k_transfer_cast_seconds = upload_bqhd(
        args.key_rows, args.heads, args.head_dim, 2, 0, device, target_dtype
    )
    v_bqhd, v_cpu_seconds, v_transfer_cast_seconds = upload_bqhd(
        args.key_rows, args.heads, args.head_dim, 3, 0, device, target_dtype
    )
    input_setup_seconds = time.perf_counter() - input_setup_started

    result_bhqd = torch.empty(
        (1, args.heads, args.query_rows, args.head_dim),
        device=device,
        dtype=torch.float32,
    )
    finite = torch.ones((), device=device, dtype=torch.bool)
    query_cpu_seconds = 0.0
    query_transfer_cast_seconds = 0.0
    oracle_started = time.perf_counter()

    for chunk_start in range(0, args.query_rows, QUERY_CHUNK_ROWS):
        valid_rows = min(QUERY_CHUNK_ROWS, args.query_rows - chunk_start)
        q_valid, cpu_seconds, transfer_seconds = upload_bqhd(
            valid_rows,
            args.heads,
            args.head_dim,
            args.q_salt,
            args.query_start + chunk_start,
            device,
            target_dtype,
        )
        query_cpu_seconds += cpu_seconds
        query_transfer_cast_seconds += transfer_seconds
        if valid_rows == QUERY_CHUNK_ROWS:
            q_chunk = q_valid
        else:
            q_chunk = torch.zeros(
                (1, QUERY_CHUNK_ROWS, args.heads, args.head_dim),
                device=device,
                dtype=target_dtype,
            )
            q_chunk[:, :valid_rows, :, :].copy_(q_valid)
            del q_valid

        if (
            q_chunk.device.type != "mps"
            or k_bqhd.device.type != "mps"
            or v_bqhd.device.type != "mps"
            or q_chunk.dtype != target_dtype
            or k_bqhd.dtype != target_dtype
            or v_bqhd.dtype != target_dtype
            or not q_chunk.is_contiguous()
            or not k_bqhd.is_contiguous()
            or not v_bqhd.is_contiguous()
        ):
            raise RuntimeError("Wan attention inputs left contiguous MPS at the requested dtype")

        output_bqhd = upstream.attention(
            q_chunk,
            k_bqhd,
            v_bqhd,
            q_lens=None,
            k_lens=None,
            dropout_p=0.0,
            softmax_scale=None,
            q_scale=None,
            causal=False,
            window_size=(-1, -1),
            deterministic=False,
            dtype=target_dtype,
            fa_version=None,
        )
        if (
            not isinstance(output_bqhd, torch.Tensor)
            or output_bqhd.device.type != "mps"
            or output_bqhd.dtype != target_dtype
            or tuple(output_bqhd.shape)
            != (1, QUERY_CHUNK_ROWS, args.heads, args.head_dim)
        ):
            raise RuntimeError("Pinned Wan attention returned an unexpected device/dtype/shape")

        # The official fallback returns B,Q,H,D in q.dtype. Widen on MPS,
        # transpose to the fixture's BHQD order, and discard padded rows.
        output_chunk_bhqd = (
            output_bqhd[:, :valid_rows, :, :]
            .transpose(1, 2)
            .to(dtype=torch.float32)
        )
        result_bhqd[:, :, chunk_start : chunk_start + valid_rows, :].copy_(
            output_chunk_bhqd
        )
        finite = finite & torch.isfinite(output_chunk_bhqd).all()
        del q_chunk, output_bqhd, output_chunk_bhqd

    # Fixture-generation synchronization only. The probe's measured arms own
    # their separate timing and synchronization boundary.
    torch.mps.synchronize()
    oracle_wall_seconds = time.perf_counter() - oracle_started
    if not bool(finite.item()):
        raise RuntimeError("non-finite pinned Wan attention output")

    expected_bytes = args.query_rows * args.heads * args.head_dim * F32_BYTES
    cpu_generator_peak_bytes = estimated_cpu_generator_peak_bytes(
        min(GENERATOR_CHUNK_ROWS, max(args.key_rows, args.query_rows)),
        args.heads,
        args.head_dim,
    )
    export_started = time.perf_counter()
    try:
        with expected_tmp_path.open("xb") as stream:
            # BHQD is head-major, so serialize one complete head at a time.
            for head in range(args.heads):
                host_head = (
                    result_bhqd[0, head, :, :]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                    .numpy()
                )
                host_head.astype(np.dtype("<f4"), copy=False).tofile(stream)
                del host_head
        if expected_tmp_path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"unexpected output byte count: {expected_tmp_path.stat().st_size}"
            )
        os.replace(expected_tmp_path, expected_path)
    except BaseException:
        expected_tmp_path.unlink(missing_ok=True)
        raise
    output_export_seconds = time.perf_counter() - export_started

    generator_path = Path(__file__).resolve()
    channels = args.heads * args.head_dim
    manifest = {
        "schema": SCHEMA,
        "kind": "wan_s2v_self_attention_precision",
        "query_rows": args.query_rows,
        "query_start": args.query_start,
        "key_rows": args.key_rows,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "query_chunk_rows": QUERY_CHUNK_ROWS,
        "geometry": {
            "batches": 1,
            "query_rows": args.query_rows,
            "query_start": args.query_start,
            "query_extent_rows": args.key_rows,
            "key_rows": args.key_rows,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "native_layout": "B,Q,H,D",
            "reference_layout": "B,H,Q,D",
        },
        "input_formula": FORMULA,
        "input_salts": {"q": args.q_salt, "k": 2, "v": 3},
        "reference": {
            "operator": "pinned wan_modules_attention.py::attention SDPA fallback",
            "torch_version": torch.__version__,
            "device": "mps",
            "dtype": _dtype_name(args.dtype),
            "mps_fallback_disabled": True,
            "flash_attention_2_available": False,
            "flash_attention_3_available": False,
            "wan_attention_kwargs": {
                "q_lens": None,
                "k_lens": None,
                "dropout_p": 0.0,
                "softmax_scale": None,
                "q_scale": None,
                "causal": False,
                "window_size": [-1, -1],
                "deterministic": False,
                "fa_version": None,
            },
            "sdpa_semantics": "no mask, noncausal, dropout=0, default scale",
            "query_chunk_rows": QUERY_CHUNK_ROWS,
            "query_chunk_padding": (
                "short final Q chunk is zero-padded to 128 rows; padded outputs are discarded"
            ),
            "input_precision": (
                "F32 dyadic CPU fixture values transferred to MPS as F32, then cast on MPS "
                "to the requested Q/K/V dtype before the official function"
            ),
            "output_precision": (
                "official function output retains Q dtype; each valid chunk is widened to "
                "F32 on MPS for expected_bhqd.f32 serialization"
            ),
            "internal_accumulator_precision": "not inferred or claimed",
            "comparison_tolerance": "2e-5 + 2e-5 * abs(reference)",
            "terminal_sync_semantics": (
                "one explicit torch.mps.synchronize() after all fixture chunks; the finite "
                "scalar read and streamed CPU export are fixture-generation work and may "
                "entail implicit waits outside native benchmark arms"
            ),
        },
        "memory": {
            "estimated_explicit_peak_bytes": peak_bytes,
            "explicit_buffer_limit_bytes": EXPLICIT_BUFFER_LIMIT,
            "cpu_generator_chunk_rows": GENERATOR_CHUNK_ROWS,
            "estimated_explicit_cpu_numpy_peak_bytes_per_generator_chunk": (
                cpu_generator_peak_bytes
            ),
            "cpu_numpy_peak_model": (
                "24 bytes per generated scalar plus row/channel index vectors; excludes "
                "allocator overhead and transfer staging"
            ),
            "mps_private_workspace_included": False,
            "note": "opaque MPSGraph workspace is excluded from the explicit tensor estimate",
        },
        "expected": {
            "file": expected_path.name,
            "dtype": "F32_LE",
            "shape": [1, args.heads, args.query_rows, args.head_dim],
            "bytes": expected_bytes,
            "sha256": sha256_file(expected_path),
        },
        "source": {
            "wan_attention_source_path": str(source_path),
            "wan_attention_source_sha256": source_sha256,
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
            "mps_kv_transfer_and_cast": k_transfer_cast_seconds
            + v_transfer_cast_seconds,
            "mps_q_transfer_and_cast": query_transfer_cast_seconds,
            "input_setup_host_wall": input_setup_seconds,
            "mps_attention_including_final_sync": oracle_wall_seconds,
            "streamed_output_readback_and_export": output_export_seconds,
        },
        "scope": (
            "selected-query pinned Wan self-attention SDPA fixture for one input dtype; "
            "no internal accumulator precision, performance, full-block, or video-quality claim"
        ),
    }
    try:
        with manifest_tmp_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(manifest_tmp_path, manifest_path)
    except BaseException:
        manifest_tmp_path.unlink(missing_ok=True)
        expected_path.unlink(missing_ok=True)
        raise
    return {
        "status": "complete",
        "fixture": str(output_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "expected_sha256": manifest["expected"]["sha256"],
        "query_rows": args.query_rows,
        "query_start": args.query_start,
        "query_chunk_rows": QUERY_CHUNK_ROWS,
        "dtype": args.dtype,
        "q_salt": args.q_salt,
        "estimated_explicit_peak_bytes": peak_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, required=True, help="new or empty fixture directory"
    )
    parser.add_argument(
        "--wan-attention-source",
        type=Path,
        default=DEFAULT_WAN_ATTENTION_SOURCE,
        help="pinned upstream wan_modules_attention.py source",
    )
    parser.add_argument("--query-rows", type=int, default=128)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--key-rows", type=int, default=K_ROWS_DEFAULT)
    parser.add_argument("--heads", type=int, default=HEADS_DEFAULT)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM_DEFAULT)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="f32")
    parser.add_argument("--q-salt", type=int, default=1)
    args = parser.parse_args()

    try:
        result = run_reference(args)
    except (FileExistsError, FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
