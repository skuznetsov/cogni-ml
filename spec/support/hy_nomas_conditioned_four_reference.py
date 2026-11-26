#!/usr/bin/env python3
"""Produce a bounded conditioned full-J reference for pretrained blocks 0..3.

The producer consumes the authenticated Qwen state-26 crop, runs the complete
1280-row masked text refiner and time embedder on MPS, then hands their outputs
through four pinned MMDoubleStreamBlock instances. Attention uses an explicit
MPS unpadding adapter: only the 5280 image rows and first 23 valid text rows
participate in attention, and padded query outputs are zero-filled. This is a
bounded reference probe, not full media generation or Magi/CUDA parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import hy_nomas_four_block_reference as four_reference
import hy_nomas_full_j_reference as full_j_reference
import hy_nomas_pretrained_serial_reference as serial_reference
import hy_nomas_reference as block_reference
import hy_nomas_text_conditioning_reference as text_reference


SCHEMA = "hy-nomas-conditioned-four-reference-v1"
PINNED_REVISION = serial_reference.PINNED_REVISION
DEFAULT_UPSTREAM = serial_reference.DEFAULT_UPSTREAM
DEFAULT_TEXT_WEIGHTS = text_reference.DEFAULT_TEXT_WEIGHTS
DEFAULT_TEXT_MANIFEST = text_reference.DEFAULT_TEXT_MANIFEST
DEFAULT_QWEN_REFERENCE = text_reference.DEFAULT_QWEN_REFERENCE
DEFAULT_QWEN_SIDECAR = text_reference.DEFAULT_QWEN_SIDECAR
ATOL = serial_reference.ATOL
RTOL = serial_reference.RTOL
VIDEO_TOKENS = full_j_reference.VIDEO_TOKENS
TEXT_TOKENS = full_j_reference.TEXT_TOKENS
VALID_TEXT_TOKENS = 23
PADDED_TEXT_TOKENS = TEXT_TOKENS - VALID_TEXT_TOKENS
TOTAL_TOKENS = VIDEO_TOKENS + TEXT_TOKENS
VALID_JOINT_TOKENS = VIDEO_TOKENS + VALID_TEXT_TOKENS
QUERY_CHUNK = full_j_reference.QUERY_CHUNK
ATTENTION_CHUNKS_PER_BLOCK = math.ceil(VALID_JOINT_TOKENS / QUERY_CHUNK)
REFINER_ATTENTION_CALLS = 2
JSON_LIMIT = 2 * 1024 * 1024
SIDE_CAR_LIMIT = 512 * 1024 * 1024
QWEN_SIDECAR_BYTES = 268_627_968

GEOMETRY = {
    **full_j_reference.GEOMETRY,
    "valid_text_tokens": VALID_TEXT_TOKENS,
    "padded_text_tokens": PADDED_TEXT_TOKENS,
    "valid_joint_tokens": VALID_JOINT_TOKENS,
    "refiner_input_dim": 3584,
    "refiner_depth": 2,
}
PRECISION = dict(full_j_reference.PRECISION)
INPUT_GENERATION = {
    "kind": "deterministic synthetic MPS F32 video input plus pinned upstream ND RoPE",
    "video_formula": full_j_reference.INPUT_GENERATION["formula"],
    "video_phase": full_j_reference.INPUT_GENERATION["phases"]["img"],
    "rope": dict(full_j_reference.INPUT_GENERATION["rope"]),
}
PINNED_UPSTREAM_SOURCE_SHA256 = {
    **block_reference.PINNED_SOURCE_SHA256,
    **text_reference.PINNED_SOURCE_SHA256,
}
if any(
    relative in block_reference.PINNED_SOURCE_SHA256
    and block_reference.PINNED_SOURCE_SHA256[relative] != digest
    for relative, digest in text_reference.PINNED_SOURCE_SHA256.items()
):
    raise RuntimeError("conflicting pinned upstream source digests")

EXPECTED_TENSOR_SHAPES: dict[str, list[int]] = {
    "input.img": [1, VIDEO_TOKENS, 2048],
    "input.rope_cos": [VIDEO_TOKENS, 128],
    "input.rope_sin": [VIDEO_TOKENS, 128],
    "conditioning.projected_text": [1, TEXT_TOKENS, 2048],
    "conditioning.vec": [1, 2048],
}
for _index in range(4):
    EXPECTED_TENSOR_SHAPES[f"block{_index}.img.output"] = [1, VIDEO_TOKENS, 2048]
    EXPECTED_TENSOR_SHAPES[f"block{_index}.txt.output"] = [1, TEXT_TOKENS, 2048]


class ReferenceError(RuntimeError):
    """A fail-closed input, provenance, execution, or fixture error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_arg(value: str, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReferenceError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _element_count(shape: list[int] | tuple[int, ...]) -> int:
    if (not isinstance(shape, (list, tuple)) or not shape or len(shape) > 4
            or any(type(value) is not int or value <= 0 for value in shape)):
        raise ReferenceError("invalid tensor shape")
    return math.prod(shape)


def _same_json_types(expected: Any, actual: Any) -> bool:
    """Compare JSON values without Python's bool/int equality ambiguity."""
    if isinstance(expected, dict):
        return (type(actual) is dict and set(actual) == set(expected)
                and all(_same_json_types(value, actual[key])
                        for key, value in expected.items()))
    if isinstance(expected, list):
        return (type(actual) is list and len(actual) == len(expected)
                and all(_same_json_types(left, right)
                        for left, right in zip(expected, actual)))
    return type(actual) is type(expected) and actual == expected


def attention_geometry(video_tokens: int, text_tokens: int,
                       valid_text_tokens: int, query_chunk: int) -> dict[str, int]:
    """Pure admission math shared by CPU tests and the MPS adapter."""
    if (any(type(value) is not int for value in
            (video_tokens, text_tokens, valid_text_tokens, query_chunk))
            or video_tokens <= 0 or text_tokens <= 0 or query_chunk <= 0
            or not 0 < valid_text_tokens <= text_tokens):
        raise ReferenceError("invalid conditioned attention geometry")
    return {
        "tensor_tokens": video_tokens + text_tokens,
        "valid_joint_tokens": video_tokens + valid_text_tokens,
        "padded_text_tokens": text_tokens - valid_text_tokens,
        "query_chunks": math.ceil((video_tokens + valid_text_tokens) / query_chunk),
    }


def validate_tensor_layout(tensors: dict[str, Any], sidecar_bytes: int) -> None:
    if type(sidecar_bytes) is not int or not 0 < sidecar_bytes <= SIDE_CAR_LIMIT:
        raise ReferenceError("tensor sidecar size outside 512 MiB limit")
    if not isinstance(tensors, dict) or set(tensors) != set(EXPECTED_TENSOR_SHAPES):
        raise ReferenceError("conditioned tensor inventory is incomplete or unexpected")
    cursor = 0
    for name in sorted(EXPECTED_TENSOR_SHAPES):
        metadata = tensors.get(name)
        shape = EXPECTED_TENSOR_SHAPES[name]
        nbytes = _element_count(shape) * 4
        if (not isinstance(metadata, dict) or metadata.get("dtype") != "F32"
                or type(metadata.get("shape")) is not list
                or not _same_json_types(shape, metadata.get("shape"))
                or type(metadata.get("offset_bytes")) is not int
                or metadata["offset_bytes"] != cursor
                or type(metadata.get("nbytes")) is not int
                or metadata["nbytes"] != nbytes):
            raise ReferenceError(f"conditioned tensor range/shape mismatch: {name}")
        if nbytes > sidecar_bytes - cursor:
            raise ReferenceError(f"conditioned tensor range exceeds sidecar: {name}")
        cursor += nbytes
    if cursor != sidecar_bytes:
        raise ReferenceError("conditioned sidecar has trailing or missing bytes")


def _validate_text_source(value: Any) -> None:
    if not isinstance(value, dict):
        raise ReferenceError("text source metadata is missing")
    expected = {
        "nbytes": 583_316_840,
        "sha256": text_reference.TEXT_WEIGHTS_SHA256,
        "manifest_sha256": text_reference.TEXT_MANIFEST_SHA256,
        "tensor_count": 42,
        "dtype": "F32",
        "repository": text_reference.TEXT_SOURCE["repository"],
        "revision": text_reference.TEXT_SOURCE["revision"],
        "checkpoint_path": text_reference.TEXT_SOURCE["checkpoint_path"],
        "source_git_commit": text_reference.TEXT_SOURCE["source_git_commit"],
        "source_payload_sha256": "8b60f66d65d54332ecd96aa89ba4fb6c4565cb11cefccee50f4f68138301b1a5",
        "source_header_sha256": "cbfa83dc81af315b425fd783e0ce42d8ff09df772952ef2a8aff48493e21b200",
    }
    for key, expected_value in expected.items():
        if (type(value.get(key)) is not type(expected_value)
                or value.get(key) != expected_value):
            raise ReferenceError(f"text source pin mismatch: {key}")
    for key, expected_name in (
        ("path", DEFAULT_TEXT_WEIGHTS.name),
        ("manifest_path", DEFAULT_TEXT_MANIFEST.name),
    ):
        path = value.get(key)
        if (not isinstance(path, str) or not Path(path).is_absolute()
                or Path(path).name != expected_name):
            raise ReferenceError(f"text source path is invalid: {key}")


def _validate_block_sources(source: Any) -> None:
    order = [f"block{index}" for index in range(4)]
    if (not isinstance(source, dict) or source.get("block_order") != order
            or not isinstance(source.get("blocks"), dict)
            or set(source["blocks"]) != set(order)):
        raise ReferenceError("conditioned block order/inventory mismatch")
    for name in order[:2]:
        try:
            serial_reference._validate_block_document(name, source["blocks"][name])
        except (serial_reference.ReferenceError, KeyError, TypeError) as exc:
            raise ReferenceError(f"{name} source metadata mismatch: {exc}") from None
    for name in order[2:]:
        try:
            four_reference._validate_dynamic_block_document(name, source["blocks"][name])
        except (four_reference.ReferenceError, KeyError, TypeError) as exc:
            raise ReferenceError(f"{name} source metadata mismatch: {exc}") from None
    seen: set[str] = set()
    seen_manifests: set[str] = set()
    seen_payloads: set[str] = set()
    for name in order:
        block = source["blocks"][name]
        digest = block.get("file_sha256")
        manifest_digest = block.get("manifest_sha256")
        payload_digest = block.get("source_payload_sha256")
        if (not isinstance(digest, str) or digest in seen
                or not isinstance(manifest_digest, str) or manifest_digest in seen_manifests
                or not isinstance(payload_digest, str) or payload_digest in seen_payloads):
            raise ReferenceError("conditioned block source alias or digest mismatch")
        seen.add(digest)
        seen_manifests.add(manifest_digest)
        seen_payloads.add(payload_digest)


def validate_reference_document(document: dict[str, Any]) -> None:
    """Validate the conditioned fixture envelope without importing torch."""
    if (not isinstance(document, dict) or document.get("schema") != SCHEMA
            or type(document.get("schema_version")) is not int
            or document.get("schema_version") != 1):
        raise ReferenceError("missing or invalid conditioned four-block schema")
    upstream = document.get("upstream")
    if (not isinstance(upstream, dict)
            or upstream.get("revision") != PINNED_REVISION
            or upstream.get("source_sha256") != PINNED_UPSTREAM_SOURCE_SHA256):
        raise ReferenceError("conditioned pinned upstream source inventory mismatch")
    if not _same_json_types(GEOMETRY, document.get("geometry")):
        raise ReferenceError("conditioned full-J geometry mismatch")
    if not _same_json_types(INPUT_GENERATION, document.get("input_generation")):
        raise ReferenceError("conditioned video/RoPE generation recipe mismatch")
    if not _same_json_types(PRECISION, document.get("precision")):
        raise ReferenceError("conditioned precision policy mismatch")
    qwen_source = document.get("qwen_source")
    if not _same_json_types(text_reference.QWEN_SOURCE, qwen_source):
        raise ReferenceError("conditioned Qwen source pin mismatch")
    mask = document.get("mask")
    if (not isinstance(mask, list) or len(mask) != TEXT_TOKENS
            or any(type(value) is not int for value in mask)
            or mask != [1] * VALID_TEXT_TOKENS + [0] * PADDED_TEXT_TOKENS):
        raise ReferenceError("conditioned Qwen crop mask mismatch")
    timestep = document.get("timestep")
    if type(timestep) not in (int, float) or float(timestep) != 500.0:
        raise ReferenceError("conditioned timestep mismatch")
    _validate_text_source(document.get("text_source"))
    _validate_block_sources(document.get("source"))

    padding = document.get("padding_policy")
    expected_padding = {
        "kind": "unpad_zero_tokens",
        "valid_joint_prefix": VALID_JOINT_TOKENS,
        "valid_text_rows": VALID_TEXT_TOKENS,
        "padded_text_rows": PADDED_TEXT_TOKENS,
        "refiner_rows": TEXT_TOKENS,
        "padded_query_outputs": "attention result explicitly zero-filled",
    }
    if not _same_json_types(expected_padding, padding):
        raise ReferenceError("conditioned padding/unpadding policy mismatch")
    tolerance = document.get("tolerance")
    if (not isinstance(tolerance, dict)
            or type(tolerance.get("atol")) not in (int, float)
            or type(tolerance.get("rtol")) not in (int, float)
            or float(tolerance["atol"]) != ATOL or float(tolerance["rtol"]) != RTOL):
        raise ReferenceError("conditioned unchanged tolerance mismatch")

    receipt = document.get("producer")
    if not isinstance(receipt, dict):
        raise ReferenceError("conditioned producer receipt is missing")
    expected_receipt = {
        "device": "mps",
        "activation_readback": "one packed final F32 tensor sidecar readback",
        "conditioning_invocations": 1,
        "refiner_rows": TEXT_TOKENS,
        "refiner_attention_calls": REFINER_ATTENTION_CALLS,
        "block_invocations": [f"block{index}" for index in range(4)],
        "attention_adapter": "PyTorch MPS SDPA F32; valid joint prefix 5303; query rows chunked at 256",
        "attention_adapter_calls": 4,
        "attention_query_chunks_per_block": ATTENTION_CHUNKS_PER_BLOCK,
        "attention_query_chunk_calls": 4 * ATTENTION_CHUNKS_PER_BLOCK,
        "padding_poison_control_calls": 1,
        "padding_poison_control_query_chunks": ATTENTION_CHUNKS_PER_BLOCK,
        "padding_poison_control_passed": True,
        "block_handoff": {
            "img": "each block output passed directly as next block input",
            "txt": "each block output passed directly as next block input",
            "same_mps_tensor_objects": True,
        },
        "captures_per_block": ["img.output", "txt.output"],
        "terminal_tensor_count": len(EXPECTED_TENSOR_SHAPES),
        "max_attention_query_chunk": QUERY_CHUNK,
    }
    for key, expected in expected_receipt.items():
        if not _same_json_types(expected, receipt.get(key)):
            raise ReferenceError(f"conditioned producer receipt mismatch: {key}")

    tensor_file = document.get("tensor_file")
    if (not isinstance(tensor_file, dict)
            or not isinstance(tensor_file.get("name"), str)
            or re.fullmatch(r"[A-Za-z0-9_.-]+\.tensors\.f32le", tensor_file["name"]) is None
            or tensor_file.get("encoding") != "f32-le"
            or type(tensor_file.get("nbytes")) is not int
            or not isinstance(tensor_file.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", tensor_file["sha256"]) is None):
        raise ReferenceError("conditioned tensor sidecar metadata is invalid")
    validate_tensor_layout(document.get("tensors"), tensor_file["nbytes"])
    if (type(receipt.get("terminal_sidecar_bytes")) is not int
            or receipt["terminal_sidecar_bytes"] != tensor_file["nbytes"]):
        raise ReferenceError("conditioned producer sidecar byte count mismatch")


def _load_text_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    expected = _digest_arg(expected_sha256, "text manifest digest")
    if path.is_symlink():
        raise ReferenceError("text manifest symlink is not admitted")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0 or resolved.stat().st_size > JSON_LIMIT:
        raise ReferenceError("text manifest must be a bounded regular file")
    if sha256_file(resolved) != expected:
        raise ReferenceError("text manifest digest mismatch")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ReferenceError("text manifest is invalid bounded UTF-8 JSON") from None
    if not isinstance(value, dict):
        raise ReferenceError("text manifest root must be an object")
    return value


def _verify_upstream(upstream: Path) -> dict[str, str]:
    source_hashes = block_reference._verify_upstream(upstream)
    for relative, expected in text_reference.PINNED_SOURCE_SHA256.items():
        path = upstream / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ReferenceError(f"pinned text/refiner source digest mismatch: {relative}")
        source_hashes[relative] = expected
    if source_hashes != PINNED_UPSTREAM_SOURCE_SHA256:
        raise ReferenceError("conditioned source files differ from the pinned inventory")
    return source_hashes


def _verify_text_weights(weights: Path, manifest_path: Path,
                         weights_sha256: str, manifest_sha256: str) -> tuple[Path, Path, dict[str, Any]]:
    expected_weights = _digest_arg(weights_sha256, "text weights digest")
    expected_manifest = _digest_arg(manifest_sha256, "text manifest digest")
    if (expected_weights != text_reference.TEXT_WEIGHTS_SHA256
            or expected_manifest != text_reference.TEXT_MANIFEST_SHA256):
        raise ReferenceError("text weights/manifest are not the pinned 42-tensor source")
    if weights.is_symlink():
        raise ReferenceError("text weights symlink is not admitted")
    path = weights.resolve(strict=True)
    if not path.is_file() or path.suffix != ".safetensors":
        raise ReferenceError("text weights must be a regular safetensors file")
    if path.stat().st_size != 583_316_840 or sha256_file(path) != expected_weights:
        raise ReferenceError("text weights size/digest mismatch")
    manifest = _load_text_manifest(manifest_path, expected_manifest)
    text_reference._verify_text_manifest(manifest, path, expected_weights)
    return path, manifest_path.resolve(strict=True), manifest


def _build_chunked_unpad_adapter(torch: Any, receipt: dict[str, Any]) -> Any:
    functional = torch.nn.functional

    def _attend(query: Any, key: Any, value: Any) -> Any:
        query_valid = query[:, :VALID_JOINT_TOKENS]
        key_valid = key[:, :VALID_JOINT_TOKENS]
        value_valid = value[:, :VALID_JOINT_TOKENS]
        query_sequence = query_valid.permute(0, 2, 1, 3)
        key_heads = key_valid.permute(0, 2, 1, 3).contiguous()
        value_heads = value_valid.permute(0, 2, 1, 3).contiguous()
        output_heads = torch.zeros(
            (1, 16, TOTAL_TOKENS, 128), dtype=torch.float32, device=query.device
        )
        chunks = 0
        for start in range(0, VALID_JOINT_TOKENS, QUERY_CHUNK):
            stop = min(start + QUERY_CHUNK, VALID_JOINT_TOKENS)
            query_chunk = query_sequence[:, :, start:stop, :].contiguous()
            attended = functional.scaled_dot_product_attention(
                query_chunk, key_heads, value_heads,
                attn_mask=None, dropout_p=0.0, is_causal=False,
            )
            if attended.device.type != "mps" or attended.dtype != torch.float32:
                raise RuntimeError("conditioned MPS SDPA returned unexpected device/dtype")
            output_heads[:, :, start:stop, :].copy_(attended)
            chunks += 1
            del query_chunk, attended
        output = output_heads.permute(0, 2, 1, 3).contiguous().flatten(2)
        return output, chunks

    def parallel_attention(
        q: tuple[Any, Any],
        k: tuple[Any, Any],
        v: tuple[Any, Any],
        img_q_len: int,
        img_kv_len: int,
        attn_mode: str | None = None,
        text_mask: Any = None,
        attn_param: dict[str, Any] | None = None,
    ) -> Any:
        if (attn_mode != "magi_mp_causal" or text_mask is not None
                or attn_param is None):
            raise RuntimeError("conditioned adapter admits only one unmasked segment")
        if (attn_param.get("num_prompts") != 1
                or attn_param.get("chunk_size_tokens", 0) != 0
                or attn_param.get("prompt_video_boundaries_tokens") != [0, VIDEO_TOKENS]
                or attn_param.get("prompt_video_boundaries") != [0, VIDEO_TOKENS]
                or attn_param.get("txt_tokens_per_block") != TEXT_TOKENS
                or attn_param.get("seg_ar_attention_backend") != "magi"
                or img_q_len != VIDEO_TOKENS or img_kv_len != VIDEO_TOKENS):
            raise RuntimeError("conditioned adapter received unsupported segment geometry")
        query = torch.cat((q[0], q[1]), dim=1)
        key = torch.cat((k[0], k[1]), dim=1)
        value = torch.cat((v[0], v[1]), dim=1)
        expected_shape = (1, TOTAL_TOKENS, 16, 128)
        if any(tuple(tensor.shape) != expected_shape for tensor in (query, key, value)):
            raise RuntimeError("conditioned adapter Q/K/V geometry mismatch")
        if any(tensor.device.type != "mps" or tensor.dtype != torch.float32
               for tensor in (query, key, value)):
            raise RuntimeError("conditioned adapter requires MPS F32 Q/K/V")

        output, chunks = _attend(query, key, value)
        receipt["calls"] += 1
        receipt["query_chunks"] += chunks
        receipt["max_query_chunk"] = max(receipt["max_query_chunk"], QUERY_CHUNK)

        if receipt["padding_control"] is None:
            # Poison only masked text suffixes in copies of the first block's
            # Q/K/V. If the adapter accidentally reads a padded query/key/value,
            # NaNs reach a valid attention result and fail the terminal check.
            poisoned = [tensor.clone() for tensor in (query, key, value)]
            for tensor in poisoned:
                tensor[:, VALID_JOINT_TOKENS:, :, :].fill_(float("nan"))
            poisoned_output, poison_chunks = _attend(*poisoned)
            finite = torch.isfinite(poisoned_output).all() & torch.isfinite(output).all()
            close = (poisoned_output - output).abs() <= (
                ATOL + RTOL * output.abs()
            )
            receipt["padding_control"] = (finite & close.all()).to(
                dtype=torch.float32
            ).reshape(1)
            receipt["padding_control_chunks"] = poison_chunks
            del poisoned, poisoned_output, finite, close
        del query, key, value
        return output

    return parallel_attention


def _capture_projected_text(module: Any, records: dict[str, Any]) -> Any:
    def save(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        import torch
        if not isinstance(output, torch.Tensor):
            raise ReferenceError("refiner projected-text output is not a tensor")
        records["conditioning.projected_text"] = output.detach()
    return module.register_forward_hook(save)


def _serialize_tensors(torch: Any, records: dict[str, Any],
                       padding_control: Any) -> tuple[bytes, dict[str, Any], bool]:
    if set(records) != set(EXPECTED_TENSOR_SHAPES):
        raise ReferenceError("terminal conditioned tensor inventory mismatch")
    flat: list[Any] = []
    finite: list[Any] = []
    metadata: dict[str, Any] = {}
    offset = 0
    for name in sorted(EXPECTED_TENSOR_SHAPES):
        value = records[name]
        expected_shape = EXPECTED_TENSOR_SHAPES[name]
        if (value.device.type != "mps" or value.dtype != torch.float32
                or list(value.shape) != expected_shape):
            raise ReferenceError(f"terminal conditioned tensor device/dtype/shape mismatch: {name}")
        contiguous = value.detach().contiguous()
        nbytes = _element_count(expected_shape) * 4
        metadata[name] = {
            "dtype": "F32",
            "shape": expected_shape,
            "offset_bytes": offset,
            "nbytes": nbytes,
        }
        offset += nbytes
        flat.append(contiguous.reshape(-1))
        finite.append(torch.isfinite(contiguous).all().to(dtype=torch.float32).reshape(1))
    if offset > SIDE_CAR_LIMIT or padding_control is None:
        raise ReferenceError("conditioned sidecar bound or padding control missing")
    packed = torch.cat(flat + finite + [padding_control]).to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    array = packed.numpy()
    value_count = offset // 4
    finite_start = value_count
    finite_stop = finite_start + len(finite)
    if not bool((array[finite_start:finite_stop] == 1.0).all()):
        raise ReferenceError("non-finite conditioned terminal tensor")
    padding_passed = bool(array[finite_stop] == 1.0)
    if not padding_passed:
        raise ReferenceError("poisoned masked Q/K/V suffix changed valid attention")
    raw = array[:value_count].astype("<f4", copy=False).tobytes()
    if len(raw) != offset:
        raise ReferenceError("conditioned F32 sidecar extent mismatch")
    validate_tensor_layout(metadata, len(raw))
    return raw, metadata, padding_passed


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new conditioned reference JSON in an existing directory")
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--qwen-reference", type=Path, default=DEFAULT_QWEN_REFERENCE)
    parser.add_argument("--qwen-sidecar", type=Path, default=DEFAULT_QWEN_SIDECAR)
    parser.add_argument("--text-weights", type=Path, default=DEFAULT_TEXT_WEIGHTS)
    parser.add_argument("--text-weights-sha256", required=True)
    parser.add_argument("--text-manifest", type=Path, default=DEFAULT_TEXT_MANIFEST)
    parser.add_argument("--text-manifest-sha256", required=True)
    block_paths = {
        0: (serial_reference.DEFAULT_BLOCK0_WEIGHTS, serial_reference.DEFAULT_BLOCK0_MANIFEST),
        1: (serial_reference.DEFAULT_BLOCK1_WEIGHTS, serial_reference.DEFAULT_BLOCK1_MANIFEST),
        2: (four_reference.DEFAULT_BLOCK2_WEIGHTS, four_reference.DEFAULT_BLOCK2_MANIFEST),
        3: (four_reference.DEFAULT_BLOCK3_WEIGHTS, four_reference.DEFAULT_BLOCK3_MANIFEST),
    }
    for index, (weights, manifest) in block_paths.items():
        parser.add_argument(f"--block{index}-weights", type=Path, default=weights)
        parser.add_argument(f"--block{index}-weights-sha256", required=True)
        parser.add_argument(f"--block{index}-manifest", type=Path, default=manifest)
        parser.add_argument(f"--block{index}-manifest-sha256", required=True)
    return parser


def produce_fixture(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise ReferenceError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    output = args.output.absolute()
    if (output.suffix.lower() != ".json" or os.path.lexists(output)
            or not output.parent.is_dir()):
        raise ReferenceError("--output must be a new .json file in an existing directory")
    tensor_name = f"{output.stem}.tensors.f32le"
    tensor_path = output.parent / tensor_name
    if os.path.lexists(tensor_path):
        raise ReferenceError("conditioned sidecar already exists")

    text_path, text_manifest_path, _text_manifest = _verify_text_weights(
        args.text_weights, args.text_manifest,
        args.text_weights_sha256, args.text_manifest_sha256,
    )
    qwen_reference_path = args.qwen_reference
    qwen_sidecar_path = args.qwen_sidecar
    for label, path in (("Qwen reference", qwen_reference_path),
                        ("Qwen sidecar", qwen_sidecar_path)):
        if path.is_symlink():
            raise ReferenceError(f"{label} symlink is not admitted")
    qwen_reference_path = qwen_reference_path.resolve(strict=True)
    qwen_sidecar_path = qwen_sidecar_path.resolve(strict=True)
    if (not qwen_reference_path.is_file() or qwen_reference_path.stat().st_size > JSON_LIMIT
            or not qwen_sidecar_path.is_file()
            or qwen_sidecar_path.stat().st_size != QWEN_SIDECAR_BYTES):
        raise ReferenceError("Qwen fixture file type or extent mismatch")
    qwen_reference_sha = sha256_file(qwen_reference_path)
    qwen_sidecar_sha = sha256_file(qwen_sidecar_path)
    if (qwen_reference_sha != text_reference.QWEN_REFERENCE_SHA256
            or qwen_sidecar_sha != text_reference.QWEN_SIDECAR_SHA256):
        raise ReferenceError("Qwen reference or BF16 sidecar digest mismatch")
    try:
        qwen_document = json.loads(qwen_reference_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ReferenceError("Qwen reference is invalid bounded UTF-8 JSON") from None
    valid_indices = text_reference.validate_qwen_reference(
        qwen_document,
        reference_sha256=qwen_reference_sha,
        sidecar_bytes=qwen_sidecar_path.stat().st_size,
        sidecar_sha256=qwen_sidecar_sha,
    )

    block_weights: dict[str, tuple[Path, str, Path, dict[str, Any]]] = {}
    for index in (0, 1):
        name = f"block{index}"
        block_weights[name] = serial_reference._trusted_weight_pair(
            name=name,
            weights=getattr(args, f"block{index}_weights"),
            weights_sha256=getattr(args, f"block{index}_weights_sha256"),
            manifest_path=getattr(args, f"block{index}_manifest"),
            manifest_sha256=getattr(args, f"block{index}_manifest_sha256"),
        )
    for index in (2, 3):
        name = f"block{index}"
        block_weights[name] = four_reference._trusted_dynamic_pair(
            name=name,
            weights=getattr(args, f"block{index}_weights"),
            weights_sha256=getattr(args, f"block{index}_weights_sha256"),
            manifest_path=getattr(args, f"block{index}_manifest"),
            manifest_sha256=getattr(args, f"block{index}_manifest_sha256"),
        )
    file_digests = [block_weights[f"block{index}"][1] for index in range(4)]
    manifest_digests = [sha256_file(block_weights[f"block{index}"][2]) for index in range(4)]
    if len(set(file_digests)) != 4 or len(set(manifest_digests)) != 4:
        raise ReferenceError("four distinct block file and manifest pins are required")

    upstream = args.upstream.resolve(strict=True)
    source_hashes = _verify_upstream(upstream)

    import numpy as np
    import torch
    from safetensors import safe_open

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ReferenceError("MPS is required; CPU fallback is forbidden")
    if torch.get_default_dtype() != torch.float32:
        torch.set_default_dtype(torch.float32)
    device = torch.device("mps")
    runtime_started = False
    try:
        runtime_started = True
        helpers = block_reference._install_upstream_modules(upstream)
        text_dir = upstream / "hymm/models/text_encoder"
        block_reference._namespace("hymm.models.text_encoder", text_dir)
        attention_module, _ = text_reference._install_text_attention(
            torch, valid_indices=valid_indices, precision="f32", capture={},
        )
        text_module = __import__("importlib").import_module(
            "hymm.models.text_encoder.token_refiner"
        )
        embed_layers = __import__("importlib").import_module(
            "hymm.models.dit.modules.embed_layers"
        )

        qwen_map = np.memmap(
            qwen_sidecar_path, mode="r", dtype="<u2", shape=(27, 1, 1388, 3584)
        )
        raw_rows = np.array(qwen_map[26, 0, 108:1388], copy=True)
        del qwen_map
        qwen_bf16 = torch.from_numpy(raw_rows).view(torch.bfloat16)
        del raw_rows
        qwen_full = qwen_bf16.unsqueeze(0).to(device=device, dtype=torch.float32)
        del qwen_bf16
        mask_values = [1] * VALID_TEXT_TOKENS + [0] * PADDED_TEXT_TOKENS
        full_mask = torch.tensor([mask_values], dtype=torch.bool, device=device)
        timestep = torch.tensor([500.0], dtype=torch.float32, device=device)

        text_state = text_reference._text_state(torch, text_path)
        time_state = text_reference._time_state(torch, text_path)
        refiner = text_module.SingleTokenRefinerPerBlock(
            in_channels=3584, hidden_size=2048, heads_num=16, depth=2,
            mlp_width_ratio=4.0, mlp_drop_rate=0.0, act_type="silu",
            qk_norm=False, qkv_bias=True, attn_mode="torch", seg_pos_emb_num=0,
            dtype=torch.float32, device=torch.device("meta"),
        )
        refiner.load_state_dict(text_state, strict=True, assign=True)
        refiner.to(device=device, dtype=torch.float32).eval()
        del text_state
        time_in = embed_layers.TimestepEmbedder(
            2048, torch.nn.SiLU, dtype=torch.float32, device=torch.device("meta")
        )
        time_in.load_state_dict(time_state, strict=True, assign=True)
        time_in.to(device=device, dtype=torch.float32).eval()
        del time_state
        block_reference._round_parameters_bf16_in_place(torch, refiner, device)
        block_reference._round_parameters_bf16_in_place(torch, time_in, device)

        records: dict[str, Any] = {}
        projected_hook = _capture_projected_text(refiner.individual_token_refiner, records)
        try:
            with torch.inference_mode():
                vec = time_in(timestep)
                refiner_output = refiner(qwen_full, timestep, full_mask)
        finally:
            projected_hook.remove()
        projected_text = records.get("conditioning.projected_text")
        if (projected_text is None or tuple(projected_text.shape) != (1, TEXT_TOKENS, 2048)
                or tuple(vec.shape) != (1, 2048)
                or any(value.device.type != "mps" or value.dtype != torch.float32
                       for value in (projected_text, vec))):
            raise ReferenceError("full masked refiner conditioning output geometry/device mismatch")
        records["conditioning.vec"] = vec

        input_set = full_j_reference._generate_mps_inputs(torch, helpers[-1], device)
        img = input_set["img"]
        rope_cos = input_set["rope_cos"]
        rope_sin = input_set["rope_sin"]
        del input_set
        records["input.img"] = img
        records["input.rope_cos"] = rope_cos
        records["input.rope_sin"] = rope_sin

        receipt: dict[str, Any] = {
            "calls": 0, "query_chunks": 0, "max_query_chunk": 0,
            "padding_control": None, "padding_control_chunks": 0,
        }
        adapter = _build_chunked_unpad_adapter(torch, receipt)
        block_inputs = {
            "img": img,
            "txt": projected_text,
            "vec": vec,
            "vec_txt": vec,
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
        }
        model_roots: list[Any] = []
        for index in range(4):
            state = serial_reference._load_block_state(
                torch, block_weights[f"block{index}"][0], index
            )
            img_output, txt_output, model = four_reference._run_one_block(
                torch, upstream, helpers, state, index, device, adapter,
                block_inputs, receipt,
            )
            records[f"block{index}.img.output"] = img_output
            records[f"block{index}.txt.output"] = txt_output
            model_roots.append(model)
            block_inputs = {
                **block_inputs,
                "img": img_output,
                "txt": txt_output,
            }
        expected_chunks = 4 * ATTENTION_CHUNKS_PER_BLOCK
        if (receipt["calls"] != 4 or receipt["query_chunks"] != expected_chunks
                or receipt["padding_control_chunks"] != ATTENTION_CHUNKS_PER_BLOCK):
            raise ReferenceError("conditioned four-block attention receipt mismatch")
        raw_tensor_bytes, tensor_metadata, padding_passed = _serialize_tensors(
            torch, records, receipt["padding_control"]
        )
        if not padding_passed:
            raise ReferenceError("masked Q/K/V poison falsifier failed")
    except BaseException:
        if runtime_started:
            try:
                torch.mps.synchronize()
            except BaseException:
                pass
        raise

    tensor_sha = hashlib.sha256(raw_tensor_bytes).hexdigest()
    text_source = {
        "path": str(text_path),
        "nbytes": text_path.stat().st_size,
        "sha256": text_reference.TEXT_WEIGHTS_SHA256,
        "manifest_path": str(text_manifest_path),
        "manifest_sha256": text_reference.TEXT_MANIFEST_SHA256,
        "tensor_count": 42,
        "dtype": "F32",
        "repository": text_reference.TEXT_SOURCE["repository"],
        "revision": text_reference.TEXT_SOURCE["revision"],
        "checkpoint_path": text_reference.TEXT_SOURCE["checkpoint_path"],
        "source_git_commit": text_reference.TEXT_SOURCE["source_git_commit"],
        "source_payload_sha256": "8b60f66d65d54332ecd96aa89ba4fb6c4565cb11cefccee50f4f68138301b1a5",
        "source_header_sha256": "cbfa83dc81af315b425fd783e0ce42d8ff09df772952ef2a8aff48493e21b200",
    }
    source_blocks = four_reference._source_block_metadata(block_weights)
    document = {
        "schema": SCHEMA,
        "schema_version": 1,
        "upstream": {"revision": PINNED_REVISION, "source_sha256": source_hashes},
        "qwen_source": dict(text_reference.QWEN_SOURCE),
        "text_source": text_source,
        "source": {
            "block_order": [f"block{index}" for index in range(4)],
            "blocks": source_blocks,
        },
        "geometry": dict(GEOMETRY),
        "input_generation": dict(INPUT_GENERATION),
        "precision": dict(PRECISION),
        "mask": mask_values,
        "timestep": 500.0,
        "padding_policy": {
            "kind": "unpad_zero_tokens",
            "valid_joint_prefix": VALID_JOINT_TOKENS,
            "valid_text_rows": VALID_TEXT_TOKENS,
            "padded_text_rows": PADDED_TEXT_TOKENS,
            "refiner_rows": TEXT_TOKENS,
            "padded_query_outputs": "attention result explicitly zero-filled",
        },
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "producer": {
            "path": str(Path(__file__).resolve()),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "device": "mps",
            "activation_readback": "one packed final F32 tensor sidecar readback",
            "conditioning_invocations": 1,
            "refiner_rows": TEXT_TOKENS,
            "refiner_attention_calls": REFINER_ATTENTION_CALLS,
            "block_invocations": [f"block{index}" for index in range(4)],
            "attention_adapter": "PyTorch MPS SDPA F32; valid joint prefix 5303; query rows chunked at 256",
            "attention_adapter_calls": receipt["calls"],
            "attention_query_chunks_per_block": ATTENTION_CHUNKS_PER_BLOCK,
            "attention_query_chunk_calls": receipt["query_chunks"],
            "max_attention_query_chunk": receipt["max_query_chunk"],
            "padding_poison_control_calls": 1,
            "padding_poison_control_query_chunks": receipt["padding_control_chunks"],
            "padding_poison_control_passed": padding_passed,
            "block_handoff": {
                "img": "each block output passed directly as next block input",
                "txt": "each block output passed directly as next block input",
                "same_mps_tensor_objects": True,
            },
            "captures_per_block": ["img.output", "txt.output"],
            "terminal_tensor_count": len(records),
            "terminal_sidecar_bytes": len(raw_tensor_bytes),
        },
        "limitations": [
            "Qwen state 26 and the 108:1388 crop are loaded from the authenticated saved BF16 capture; this producer does not rerun Qwen",
            "the initial video tensor is deterministic synthetic MPS F32 input and the reference stops after four DiT blocks",
            "the full 1280-row refiner is evaluated, while its text-attention adapter unpads the known 23-row valid prefix",
            "MPS SDPA with explicit prefix unpadding and zero-filled padded query outputs is not Magi/CUDA attention parity",
            "weights use F32 storage containing BF16-rounded values; physical BF16 storage and native BF16 compute are not tested",
            "block 2 and block 3 source files are authenticated range subsets; the original whole-checkpoint LFS digest is declared but not verified",
        ],
        "tensor_file": {
            "name": tensor_name,
            "nbytes": len(raw_tensor_bytes),
            "sha256": tensor_sha,
            "encoding": "f32-le",
        },
        "tensors": tensor_metadata,
    }
    validate_reference_document(document)
    json_bytes = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(json_bytes) > JSON_LIMIT:
        raise ReferenceError("conditioned four-block JSON exceeds 2 MiB")
    four_reference._publish_noreplace(tensor_path, raw_tensor_bytes)
    try:
        four_reference._publish_noreplace(output, json_bytes)
    except BaseException:
        tensor_path.unlink(missing_ok=True)
        raise
    return document


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    document = produce_fixture(args)
    print(json.dumps({
        "fixture": str(args.output),
        "schema": SCHEMA,
        "sidecar": document["tensor_file"],
        "blocks": document["source"]["block_order"],
        "padding_poison_control_passed": document["producer"]["padding_poison_control_passed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
