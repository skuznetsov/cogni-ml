#!/usr/bin/env python3
"""Create bounded synthetic PyTorch-MPS references for compact attention.

This producer has no model or checkpoint dependency. It writes one manifest and
one little-endian F32 output sidecar for either a full T=257 control or selected
T=6560 query chunks. It intentionally never evaluates full T=6560 attention.
"""

from __future__ import annotations

import argparse
from array import array
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any


SCHEMA = "hy_nomas_compact_attention_reference_v1"
PINNED_TORCH_VERSION = "2.8.0"
PINNED_TORCH_GIT_COMMIT = "a1cb3cc05d46d198467bebbb6e8fba50a325d4e7"
TORCH_ATTENTION_SOURCE = "aten/src/ATen/native/mps/operations/Attention.mm"
TORCH_ATTENTION_SOURCE_SHA256 = "167ee89bddb489c154b5d5ca72cf042745443f557d624fc48175f2d29f45a351"
TORCH_ATTENTION_SOURCE_URL = (
    "https://github.com/pytorch/pytorch/blob/v2.8.0/" + TORCH_ATTENTION_SOURCE
)
FORMULA = (
    "avalanche32(native_flat_index + salt*0x9e3779b9); "
    "((hash & 65535) - 32768) / 65536"
)
SALTS = {"q": 1, "k": 2, "v": 3}
HEADS = 16
HEAD_DIM = 128
J_TOKENS = 6560
J_CHUNK = 128
J_REFERENCE_HEADS = HEADS


class ReferenceError(Exception):
    """A bounded rejection of an unsupported or unsafe producer input."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fresh(path: Path) -> None:
    if path.exists():
        raise ReferenceError(f"output path already exists: {path}")
    if not path.parent.is_dir():
        raise ReferenceError(f"output parent directory does not exist: {path.parent}")


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("t257-full", "j6560-sampled"),
        required=True,
        help="bounded attention shape; full J=6560 is deliberately unavailable",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument(
        "--input-bf16-boundary",
        action="store_true",
        help="round synthetic Q/K/V through MPS BF16 and back to F32 before attention",
    )
    args = parser.parse_args(argv)
    manifest = args.manifest.expanduser().absolute()
    outputs = args.outputs.expanduser().absolute()
    if manifest == outputs:
        parser.error("--manifest and --outputs must be different paths")
    _fresh(manifest)
    _fresh(outputs)
    args.manifest = manifest
    args.outputs = outputs
    return args


def _avalanche32(index: int, salt: int) -> int:
    value = (index + salt * 0x9E3779B9) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value


def _make_mps_tensor(
    torch: Any, tokens: int, salt: int, device: Any, input_bf16_boundary: bool
) -> Any:
    count = tokens * HEADS * HEAD_DIM
    host_values = array(
        "f",
        (
            ((_avalanche32(index, salt) & 0xFFFF) - 32768) / 65536.0
            for index in range(count)
        ),
    )
    if host_values.itemsize != 4:
        raise ReferenceError("host array('f') is not IEEE-sized F32")
    host_bthd = torch.frombuffer(host_values, dtype=torch.float32).view(
        1, tokens, HEADS, HEAD_DIM
    )
    mps_bthd = host_bthd.to(device=device)
    del host_bthd, host_values
    if input_bf16_boundary:
        mps_bthd = mps_bthd.to(dtype=torch.bfloat16).to(dtype=torch.float32)
    return mps_bthd.permute(0, 2, 1, 3).contiguous()


def _tensor_bytes(torch: Any, tensor: Any) -> bytes:
    if tensor.device.type != "mps" or tensor.dtype != torch.float32:
        raise ReferenceError("attention output did not remain F32 on MPS")
    host = tensor.detach().to(device="cpu").contiguous()
    if host.dtype != torch.float32 or not host.is_contiguous():
        raise ReferenceError("reference output must be contiguous F32")
    return ctypes.string_at(host.data_ptr(), host.numel() * 4)


def _record(label: str, query_start: int, rows: int, head_start: int,
            group_heads: int, output: bytes, offset: int) -> tuple[dict[str, Any], bytes]:
    record = {
        "label": label,
        "query_start": query_start,
        "query_rows": rows,
        "head_start": head_start,
        "group_heads": group_heads,
        "layout": "B,T,H,D",
        "shape": [1, rows, group_heads, HEAD_DIM],
        "offset_bytes": offset,
        "bytes": len(output),
        "sha256": _sha256(output),
    }
    return record, output


def _check_runtime() -> tuple[Any, Any, dict[str, Any]]:
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if fallback not in (None, "0"):
        raise ReferenceError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    if sys.platform != "darwin" or sys.byteorder != "little":
        raise ReferenceError("the pinned MPS reference requires little-endian macOS")
    try:
        macos = platform.mac_ver()[0]
        if not macos or int(macos.split(".", 1)[0]) < 15:
            raise ReferenceError("the pinned F32 MPS path requires macOS 15 or newer")
        import torch
    except ReferenceError:
        raise
    except Exception as exc:  # pragma: no cover - platform-specific diagnostics
        raise ReferenceError(f"could not import the pinned PyTorch runtime: {exc}") from None

    if torch.__version__ != PINNED_TORCH_VERSION:
        raise ReferenceError(
            f"PyTorch version is {torch.__version__!r}, expected {PINNED_TORCH_VERSION!r}"
        )
    if torch.version.git_version != PINNED_TORCH_GIT_COMMIT:
        raise ReferenceError("PyTorch source commit does not match the pinned v2.8.0 tag")
    if not torch.backends.mps.is_available():
        raise ReferenceError("PyTorch MPS is unavailable; CPU fallback is forbidden")
    device = torch.device("mps")
    producer_hash = _sha256(Path(__file__).resolve().read_bytes())
    provenance = {
        "producer": Path(__file__).name,
        "producer_sha256": producer_hash,
        "torch_version": torch.__version__,
        "torch_git_commit": torch.version.git_version,
        "torch_tag": "v2.8.0",
        "torch_attention_source": TORCH_ATTENTION_SOURCE,
        "torch_attention_source_url": TORCH_ATTENTION_SOURCE_URL,
        "torch_attention_source_sha256": TORCH_ATTENTION_SOURCE_SHA256,
        "reference_device": "mps",
        "mps_fallback_disabled": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0",
        "macos_version": macos,
        "machine": platform.machine(),
    }
    return torch, device, provenance


def _produce_t257(
    torch: Any, device: Any, input_bf16_boundary: bool
) -> tuple[list[dict[str, Any]], bytes]:
    tokens = 257
    query = _make_mps_tensor(torch, tokens, SALTS["q"], device, input_bf16_boundary)
    key = _make_mps_tensor(torch, tokens, SALTS["k"], device, input_bf16_boundary)
    value = _make_mps_tensor(torch, tokens, SALTS["v"], device, input_bf16_boundary)
    attended = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )
    output = _tensor_bytes(torch, attended.permute(0, 2, 1, 3).contiguous())
    record, payload = _record("full", 0, tokens, 0, HEADS, output, 0)
    return [record], payload


def _produce_j6560(
    torch: Any, device: Any, input_bf16_boundary: bool
) -> tuple[list[dict[str, Any]], bytes]:
    query = _make_mps_tensor(torch, J_TOKENS, SALTS["q"], device, input_bf16_boundary)
    key = _make_mps_tensor(torch, J_TOKENS, SALTS["k"], device, input_bf16_boundary)
    value = _make_mps_tensor(torch, J_TOKENS, SALTS["v"], device, input_bf16_boundary)
    chunk_count = (J_TOKENS + J_CHUNK - 1) // J_CHUNK
    chunk_indices = (0, 25, chunk_count - 1)
    offset = 0
    records: list[dict[str, Any]] = []
    payloads: list[bytes] = []
    for chunk_index in chunk_indices:
        query_start = min(chunk_index * J_CHUNK, J_TOKENS - J_CHUNK)
        rows = J_CHUNK
        query_slice = query[:, :, query_start : query_start + rows, :].contiguous()
        attended = torch.nn.functional.scaled_dot_product_attention(
            query_slice,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        output = _tensor_bytes(torch, attended.permute(0, 2, 1, 3).contiguous())
        label = f"h0_q{query_start}_r{rows}"
        record, payload = _record(
            label, query_start, rows, 0, J_REFERENCE_HEADS, output, offset
        )
        record["chunk_index"] = chunk_index
        records.append(record)
        payloads.append(payload)
        offset += len(payload)
        del query_slice, attended, output
    return records, b"".join(payloads)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    torch, device, provenance = _check_runtime()
    if args.case == "t257-full":
        tokens = 257
        records, output_bytes = _produce_t257(torch, device, args.input_bf16_boundary)
        reference_shape = {"query_rows": tokens, "heads_per_group": HEADS}
    else:
        tokens = J_TOKENS
        records, output_bytes = _produce_j6560(torch, device, args.input_bf16_boundary)
        reference_shape = {"query_rows": J_CHUNK, "heads_per_group": J_REFERENCE_HEADS}

    if sys.byteorder != "little":
        raise ReferenceError("output sidecar byte order is not little-endian")
    sidecar = {
        "filename": args.outputs.name,
        "byte_order": "little",
        "dtype": "float32",
        "bytes": len(output_bytes),
        "sha256": _sha256(output_bytes),
    }
    manifest = {
        "schema": SCHEMA,
        "case": args.case,
        "geometry": {
            "batches": 1,
            "tokens": tokens,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "native_input_layout": "B,T,H,D",
            "pytorch_qkv_layout": "B,H,T,D",
            "reference_output_layout": "B,T,H,D",
        },
        "input": {
            "dtype": "float32",
            "layout": "B,T,H,D",
            "formula": FORMULA,
            "salts": SALTS,
            "checkpoint": "none; direct deterministic synthetic Q/K/V",
            "bf16_boundary_roundtrip": args.input_bf16_boundary,
        },
        "reference": {
            "operator": "torch.nn.functional.scaled_dot_product_attention",
            "device": "mps",
            "dtype": "float32",
            "attention_mask": "none",
            "dropout_p": 0.0,
            "is_causal": False,
            "scale": "default 1/sqrt(head_dim)",
            "mps_fallback_disabled": True,
            "comparison_gate": "2e-5 + 2e-5*abs(reference); BF16 RNE bitwise",
            "query_slice_scope": (
                "selected mathematical query rows only; no full-T6560 parity claim"
                if args.case == "j6560-sampled"
                else "full T=257 query shape"
            ),
        },
        "provenance": provenance,
        "reference_shape": reference_shape,
        "outputs": sidecar,
        "records": records,
    }

    sidecar_created = False
    manifest_created = False
    try:
        with args.outputs.open("xb") as stream:
            sidecar_created = True
            stream.write(output_bytes)
            stream.flush()
        manifest_bytes = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with args.manifest.open("xb") as stream:
            manifest_created = True
            stream.write(manifest_bytes)
            stream.flush()
    except OSError as exc:
        if manifest_created:
            args.manifest.unlink(missing_ok=True)
        if sidecar_created:
            args.outputs.unlink(missing_ok=True)
        raise ReferenceError(f"could not create fresh fixture files: {exc}") from None

    print(
        json.dumps(
            {
                "status": "created",
                "case": args.case,
                "manifest": str(args.manifest),
                "manifest_sha256": _sha256(manifest_bytes),
                "outputs": str(args.outputs),
                "outputs_sha256": sidecar["sha256"],
                "records": len(records),
                "checkpoint": None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReferenceError as exc:
        raise SystemExit(str(exc)) from None
