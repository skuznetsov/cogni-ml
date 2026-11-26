#!/usr/bin/env python3
"""Compare pinned upstream controls against one authenticated native capture.

This bounded oracle consumes one saved first-segment, first-step, block-0
capture. It runs the unchanged pinned MMDoubleStreamBlock on MPS with the
capture's BF16 checkpoint values as either F32 parameters/activations or true
BF16 parameters/activations, then compares checkpoint summaries against the
native F32 capture. It also replays the captured native joint Q/K/V through
chunked PyTorch MPS SDPA to isolate attention from preceding block operations.

An optional prelude mode independently recomputes the pinned patch, timestep,
and full 1280-row text refiner from the authenticated latent and Qwen bundle.
It reports those results separately from the block controls that start from
the native captured patch/refiner outputs.

The SDPA adapter is a diagnostic control, not Magi/CUDA parity. It performs no
Qwen inference, diffusion, history, full-DiT, or media generation.
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
import tempfile
import types
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import hy_nomas_continuation_inputs as continuation_inputs  # noqa: E402
import hy_nomas_generation_inputs as generation_inputs  # noqa: E402
import hy_nomas_pretrained_serial_reference as serial_reference  # noqa: E402
import hy_nomas_q4_dit_reference as cache_reference  # noqa: E402
import hy_nomas_reference as block_reference  # noqa: E402
import hy_nomas_text_conditioning_reference as text_reference  # noqa: E402


SCHEMA = "cogni_ml.hy_nomas_real_block_reference/v1"
CAPTURE_SCHEMA = "cogni_ml.hy_nomas_real_block_capture/v1"
PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
PINNED_CACHE_INDEX_SHA256 = "a235ca592b47161472c7fd94cd3d886fce164480cd0571179c8860002063cbb0"
ATOL = 2e-5
RTOL = 2e-5
QUERY_CHUNK = 256
HASH_CHUNK = 8 * 1024 * 1024
COMPARE_CHUNK_ELEMENTS = 1_048_576
GEOMETRY = {
    "batches": 1,
    "video_tokens": 4320,
    "text_tokens": 576,
    "valid_text_tokens": 557,
    "heads": 16,
    "head_dim": 128,
    "dim": 2048,
    "ffn_dim": 8192,
    "text_rms_tokens": 1280,
}
EXPECTED_CAPTURE_KEYS = {
    "schema", "conditioning_manifest", "conditioning_sha256", "weight_index_sha256",
    "cache_index", "cache_index_sha256", "block_weights", "block_weights_sha256",
    "global_weights", "global_weights_sha256", "geometry", "timestep", "scope",
    "precision", "named_bytes", "named_allocations_during_encode",
    "activation_readbacks_before_terminal_wait", "terminal_wait_count", "encode_seconds",
    "gpu_elapsed_seconds", "payload", "payload_bytes", "payload_sha256", "tensors",
}
EXPECTED_CAPTURE_SCOPE = (
    "first segment, first step, patch/refiner/block0; no history, full DiT or media generation"
)
CAPTURE_INPUT_NAMES = {
    "input.latent", "input.img", "input.txt", "input.vec", "input.rope_cos", "input.rope_sin",
}
_HEX = re.compile(r"[0-9a-f]{64}\Z")


class ReferenceError(RuntimeError):
    """A fail-closed source, cache, capture, or execution error."""


def _json_no_duplicates(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ReferenceError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except ReferenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ReferenceError(f"invalid {label} JSON") from None
    if type(value) is not dict:
        raise ReferenceError(f"{label} root must be a JSON object")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ReferenceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _regular_file(path_arg: str | Path, label: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise ReferenceError(f"{label} does not exist: {path}") from None
    if path.is_symlink() or not resolved.is_file():
        raise ReferenceError(f"{label} must be a canonical regular file: {path}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _authenticate_ranges(path: Path, expected_bytes: int, expected_sha256: str,
                         ranges: list[tuple[str, int, int, str]]) -> None:
    """Hash one payload and all contiguous tensor ranges in a single pass."""
    if path.stat().st_size != expected_bytes:
        raise ReferenceError(f"payload byte extent mismatch: {path}")
    whole = hashlib.sha256()
    range_hashes = [hashlib.sha256() for _ in ranges]
    cursor = 0
    range_index = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(HASH_CHUNK)
            if not chunk:
                break
            whole.update(chunk)
            end = cursor + len(chunk)
            while range_index < len(ranges):
                _name, start, stop, _sha = ranges[range_index]
                if start >= end:
                    break
                if stop <= cursor:
                    range_index += 1
                    continue
                left = max(start, cursor) - cursor
                right = min(stop, end) - cursor
                if left < right:
                    range_hashes[range_index].update(memoryview(chunk)[left:right])
                if stop <= end:
                    range_index += 1
                else:
                    break
            cursor = end
    if cursor != expected_bytes or whole.hexdigest() != expected_sha256:
        raise ReferenceError(f"aggregate payload digest/extent mismatch: {path}")
    if range_index != len(ranges):
        raise ReferenceError("tensor range inventory did not cover the full payload")
    for (name, _start, _stop, expected), actual in zip(ranges, range_hashes):
        if actual.hexdigest() != expected:
            raise ReferenceError(f"payload tensor digest mismatch: {name}")


def _bf16_rne_roundtrip_f32_bytes(source_f32_le: bytes) -> bytes:
    """Return little-endian F32 bytes after an exact BF16 RNE roundtrip.

    This uses integer bit operations only, so source provenance checks do not
    need torch or a device. It matches casting finite F32 values to BF16 and
    expanding those BF16 values back to F32.
    """
    try:
        import numpy as np
    except ImportError as error:
        raise ReferenceError(f"NumPy is required for CPU provenance checks: {error}") from error
    if len(source_f32_le) % 4:
        raise ReferenceError("F32 provenance bytes are not word aligned")
    words = np.frombuffer(source_f32_le, dtype="<u4")
    if np.any((words & np.uint32(0x7F800000)) == np.uint32(0x7F800000)):
        raise ReferenceError("source F32 provenance contains nonfinite values")
    words64 = words.astype(np.uint64)
    retained_lsb = (words >> np.uint32(16)) & np.uint32(1)
    rounded_bf16 = (words64 + np.uint64(0x7FFF) + retained_lsb.astype(np.uint64)) >> np.uint64(16)
    if np.any((rounded_bf16 & np.uint64(0x7F80)) == np.uint64(0x7F80)):
        raise ReferenceError("finite source F32 overflows to nonfinite BF16")
    expanded = (rounded_bf16.astype(np.uint32) << np.uint32(16)).astype("<u4", copy=False)
    return expanded.tobytes()


def _first_segment_latent_roundtrip_sha256(payload: Path, record: dict[str, Any]) -> str:
    """Hash the authenticated latent-bank prefix after its step-zero BF16 roundtrip."""
    expected_shape = [1, 32, 180, 9, 16]
    expected_nbytes = math.prod(expected_shape) * 4
    if (type(record) is not dict or record.get("dtype") != "F32"
            or record.get("shape") != expected_shape
            or type(record.get("offset")) is not int or record["offset"] < 0
            or record.get("nbytes") != expected_nbytes
            or not isinstance(record.get("sha256"), str)
            or _HEX.fullmatch(record["sha256"]) is None):
        raise ReferenceError("authenticated continuation latent bank has the wrong dtype or geometry")
    try:
        import numpy as np
    except ImportError as error:
        raise ReferenceError(f"NumPy is required for CPU provenance checks: {error}") from error
    if record["offset"] + expected_nbytes > payload.stat().st_size:
        raise ReferenceError("authenticated continuation latent bank is outside its payload")
    bank = np.memmap(
        payload, dtype="<f4", mode="r", offset=record["offset"], shape=tuple(expected_shape),
    )
    prefix = np.ascontiguousarray(bank[:, :, :30, :, :], dtype="<f4").tobytes()
    transformed = _bf16_rne_roundtrip_f32_bytes(prefix)
    return hashlib.sha256(transformed).hexdigest()


def _validate_flat_capture(capture: dict[str, Any], capture_path: Path,
                           capture_sha256: str) -> tuple[Path, list[str]]:
    if _sha256_file(capture_path) != _digest(capture_sha256, "capture SHA-256"):
        raise ReferenceError("capture manifest digest differs from the supplied digest")
    if set(capture) != EXPECTED_CAPTURE_KEYS or capture.get("schema") != CAPTURE_SCHEMA:
        raise ReferenceError("native capture schema or key inventory mismatch")
    if capture.get("geometry") != GEOMETRY:
        raise ReferenceError("native capture geometry mismatch")
    if (capture.get("timestep") != 1000
            or capture.get("terminal_wait_count") != 1
            or capture.get("named_allocations_during_encode") != 0
            or capture.get("activation_readbacks_before_terminal_wait") != 0
            or capture.get("scope") != EXPECTED_CAPTURE_SCOPE
            or capture.get("precision") != "BF16 resident weights; F32 activations and MPS attention control"):
        raise ReferenceError("native capture execution scope or precision contract mismatch")
    records = capture.get("tensors")
    if type(records) is not dict:
        raise ReferenceError("native capture tensors must be an object")
    required = CAPTURE_INPUT_NAMES | {
        f"block.{name}" for name in serial_reference.BLOCK_CAPTURE_NAMES
    }
    conditioning = {name for name in records if name.startswith("conditioning.")}
    if len(conditioning) != 53 or required - set(records):
        raise ReferenceError(
            "native tensor inventory is incomplete: "
            f"missing={sorted(required - set(records))}, conditioning={len(conditioning)}"
        )
    if set(records) != required | conditioning:
        raise ReferenceError("native tensor inventory contains unknown names")
    payload_name = capture.get("payload")
    if (not isinstance(payload_name, str) or not payload_name
            or Path(payload_name).name != payload_name or not payload_name.endswith(".tensors.bin")):
        raise ReferenceError("capture payload must be a sibling .tensors.bin basename")
    payload = _regular_file(capture_path.parent / payload_name, "capture payload")
    payload_bytes = capture.get("payload_bytes")
    payload_sha = _digest(capture.get("payload_sha256"), "capture payload SHA-256")
    if type(payload_bytes) is not int or payload_bytes <= 0:
        raise ReferenceError("capture payload byte count is malformed")

    shapes = _capture_shapes()
    expected_names = set(records)
    if set(shapes) != expected_names:
        raise ReferenceError(
            f"capture shape map differs from tensor inventory: "
            f"missing={sorted(expected_names - set(shapes))}, extra={sorted(set(shapes) - expected_names)}"
        )
    ordered = sorted(records.items(), key=lambda pair: pair[1].get("offset", -1))
    cursor = 0
    ranges: list[tuple[str, int, int, str]] = []
    for name, record in ordered:
        if type(record) is not dict or set(record) != {"dtype", "shape", "offset", "nbytes", "sha256"}:
            raise ReferenceError(f"capture tensor descriptor malformed: {name}")
        count = math.prod(shapes[name])
        if (record.get("dtype") != "F32" or record.get("shape") != [count]
                or type(record.get("offset")) is not int or record["offset"] != cursor
                or type(record.get("nbytes")) is not int or record["nbytes"] != count * 4):
            raise ReferenceError(f"capture tensor extent/layout mismatch: {name}")
        sha = _digest(record.get("sha256"), f"{name} SHA-256")
        ranges.append((name, cursor, cursor + count * 4, sha))
        cursor += count * 4
    if cursor != payload_bytes:
        raise ReferenceError("native tensor extents do not cover payload bytes exactly")
    _authenticate_ranges(payload, payload_bytes, payload_sha, ranges)
    return payload, sorted(records)


def _capture_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "input.latent": (1, 32, 30, 9, 16),
        "input.img": (1, 4320, 2048),
        "input.txt": (1, 576, 2048),
        "input.vec": (1, 2048),
        "input.rope_cos": (4320, 128),
        "input.rope_sin": (4320, 128),
    }
    for name, shape in serial_reference._CAPTURE_SHAPES.items():
        dynamic = list(shape)
        if name.startswith("joint."):
            dynamic[1] = GEOMETRY["video_tokens"] + GEOMETRY["text_tokens"]
        elif name.startswith("img.") and name != "img.modulation" and len(dynamic) >= 2:
            dynamic[1] = GEOMETRY["video_tokens"]
        elif name.startswith("txt.") and name != "txt.modulation" and len(dynamic) >= 2:
            dynamic[1] = GEOMETRY["text_tokens"]
        shapes[f"block.{name}"] = tuple(dynamic)

    # The refiner capture registry has a fixed, authenticated inventory. Its
    # native descriptors deliberately carry flat element counts; these shapes
    # establish the dimensions needed to validate every element count.
    fixed = {
        "input_compact": (1, 1280, 3584), "masked_mean": (1, 3584),
        "timestep": (1,), "refiner_timestep_sinusoidal": (1, 256),
        "time_in.sinusoidal": (1, 256), "refiner_timestep_mlp1": (1, 2048),
        "refiner_timestep_silu": (1, 2048), "refiner_timestep": (1, 2048),
        "context_linear1": (1, 2048), "context_silu": (1, 2048),
        "context_embed": (1, 2048), "c": (1, 2048), "time_in.mlp1": (1, 2048),
        "time_in.silu": (1, 2048), "vec": (1, 2048), "input_embed": (1, 1280, 2048),
    }
    for block in range(2):
        prefix = f"block{block}."
        fixed.update({
            prefix + "condition_silu": (1, 2048), prefix + "adaLN": (1, 4096),
            prefix + "gate_msa": (1, 2048), prefix + "gate_mlp": (1, 2048),
            prefix + "norm1": (1, 1280, 2048), prefix + "qkv": (1, 1280, 3, 2048),
            prefix + "q": (1, 1280, 2048), prefix + "k": (1, 1280, 2048),
            prefix + "v": (1, 1280, 2048), prefix + "attention": (1, 1280, 2048),
            prefix + "proj": (1, 1280, 2048), prefix + "residual1": (1, 1280, 2048),
            prefix + "norm2": (1, 1280, 2048), prefix + "mlp_fc1": (1, 1280, 8192),
            prefix + "mlp_silu": (1, 1280, 8192), prefix + "mlp_fc2": (1, 1280, 2048),
            prefix + "output": (1, 1280, 2048), prefix + "residual2": (1, 1280, 2048),
        })
    fixed["projected_text"] = (1, 1280, 2048)
    for name, shape in fixed.items():
        shapes[f"conditioning.{name}"] = shape
    return shapes


def _authenticate_sources(capture: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the continuation base, cache index, and exact BF16 files."""
    conditioning_path = _regular_file(capture["conditioning_manifest"], "conditioning manifest")
    conditioning_sha = _digest(capture["conditioning_sha256"], "conditioning manifest SHA-256")
    if _sha256_file(conditioning_path) != conditioning_sha:
        raise ReferenceError("conditioning manifest digest mismatch")
    condition = _json_no_duplicates(conditioning_path.read_bytes(), "conditioning manifest")
    try:
        continuation_inputs.validate_document(condition)
    except Exception as error:
        raise ReferenceError(f"continuation manifest rejected: {error}") from error
    condition_payload = _regular_file(
        conditioning_path.parent / condition["payload"], "continuation tensor payload"
    )
    condition_ranges = [
        (name, rec["offset"], rec["offset"] + rec["nbytes"], rec["sha256"])
        for name, rec in sorted(condition["tensors"].items(), key=lambda pair: pair[1]["offset"])
    ]
    _authenticate_ranges(condition_payload, condition["payload_bytes"],
                         condition["payload_sha256"], condition_ranges)
    latent_record = condition["tensors"].get("input.latent_bank")
    first_segment_latent_sha = _first_segment_latent_roundtrip_sha256(
        condition_payload, latent_record,
    )
    bundle = condition["qwen_input_bundle"]
    base_manifest = _regular_file(bundle["manifest_path"], "base Qwen input manifest")
    if _sha256_file(base_manifest) != bundle["manifest_sha256"]:
        raise ReferenceError("base Qwen input manifest SHA-256 mismatch")
    base = _json_no_duplicates(base_manifest.read_bytes(), "base Qwen input manifest")
    try:
        generation_inputs.validate_document(base)
    except Exception as error:
        raise ReferenceError(f"base Qwen input manifest rejected: {error}") from error
    base_payload = _regular_file(bundle["payload_path"], "base Qwen input payload")
    if base_payload.parent != base_manifest.parent:
        raise ReferenceError("base Qwen input payload must be a sibling of its manifest")
    if base["payload"] != base_payload.name:
        raise ReferenceError("base Qwen payload basename differs from manifest")
    base_ranges = [
        (name, rec["offset"], rec["offset"] + rec["nbytes"], rec["sha256"])
        for name, rec in sorted(base["tensors"].items(), key=lambda pair: pair[1]["offset"])
    ]
    _authenticate_ranges(base_payload, base["payload_bytes"], base["payload_sha256"], base_ranges)
    if base["payload_sha256"] != bundle["payload_sha256"]:
        raise ReferenceError("base Qwen payload digest differs from continuation bundle binding")
    if base["tensors"]["input.qwen_bf16"]["sha256"] != bundle["qwen_tensor_sha256"]:
        raise ReferenceError("base Qwen tensor differs from continuation bundle binding")
    try:
        # Reuse the repository's full provenance validator, including the
        # original Qwen reference/sidecar and state-26 crop binding.
        continuation_inputs._admit_qwen_inputs(base_manifest, bundle["manifest_sha256"])
    except Exception as error:
        raise ReferenceError(f"authenticated Qwen source bundle rejected: {error}") from error
    condition_qwen = condition.get("qwen_source")
    base_qwen = base.get("qwen_source")
    if type(condition_qwen) is not dict or type(base_qwen) is not dict:
        raise ReferenceError("Qwen source metadata is missing from the continuation bundle")
    valid_count = condition_qwen.get("valid_tokens")
    mask = condition_qwen.get("cropped_attention_mask")
    base_mask = base_qwen.get("cropped_attention_mask")
    if (type(valid_count) is not int or valid_count != GEOMETRY["valid_text_tokens"]
            or type(mask) is not list or type(base_mask) is not list
            or mask != base_mask
            or mask != [1] * valid_count + [0] * (GEOMETRY["text_rms_tokens"] - valid_count)):
        raise ReferenceError("authenticated Qwen mask is not the captured 557-row valid prefix")
    qwen_record = base["tensors"]["input.qwen_bf16"]
    if (qwen_record.get("dtype") != "BF16"
            or qwen_record.get("shape") != [1, GEOMETRY["text_rms_tokens"], 3584]):
        raise ReferenceError("authenticated Qwen input does not match the full retained-padding extent")
    # The continuation bundle stores its first-segment RoPE digests as source
    # attestations. Verify those against the first 4320 rows of the base bank.
    rope_digests = {}
    for key in ("cos", "sin"):
        record = base["tensors"][f"input.rope_{key}"]
        segment_bytes = GEOMETRY["video_tokens"] * 128 * 4
        path = base_payload
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            stream.seek(record["offset"])
            remaining = segment_bytes
            while remaining:
                chunk = stream.read(min(remaining, HASH_CHUNK))
                if not chunk:
                    raise ReferenceError("base RoPE bank truncated while verifying first segment")
                digest.update(chunk)
                remaining -= len(chunk)
        rope_digests[f"base_first_rope_{key}_sha256"] = digest.hexdigest()
        if rope_digests[f"base_first_rope_{key}_sha256"] != bundle[f"first_rope_{key}_sha256"]:
            raise ReferenceError(f"base first-segment RoPE digest mismatch: {key}")

    cache_path = _regular_file(capture["cache_index"], "cache index")
    cache_sha = _digest(capture["cache_index_sha256"], "cache index SHA-256")
    if cache_sha != PINNED_CACHE_INDEX_SHA256 or _sha256_file(cache_path) != cache_sha:
        raise ReferenceError("cache index differs from pinned BF16 cache index")
    index = _json_no_duplicates(cache_path.read_bytes(), "cache index")
    try:
        cache_reference.validate_cache_index_document(index)
    except Exception as error:
        raise ReferenceError(f"cache index rejected: {error}") from error
    if capture.get("weight_index_sha256") != cache_sha:
        raise ReferenceError("capture weight index digest differs from pinned cache index")
    block_entry = index["blocks"][0]
    expected_block = (cache_path.parent / block_entry["bf16"]).resolve(strict=True)
    expected_global = (cache_path.parent / index["globals"]["bf16"]).resolve(strict=True)
    block_path = _regular_file(capture["block_weights"], "block-0 BF16 weights")
    global_path = _regular_file(capture["global_weights"], "global BF16 weights")
    block_sha = _digest(capture["block_weights_sha256"], "block-0 weights SHA-256")
    global_sha = _digest(capture["global_weights_sha256"], "global weights SHA-256")
    if (block_path != expected_block or block_sha != block_entry["bf16_sha256"]
            or _sha256_file(block_path) != block_sha):
        raise ReferenceError("block-0 BF16 checkpoint is not the exact index-pinned file")
    if (global_path != expected_global or global_sha != index["globals"]["sha256"]
            or _sha256_file(global_path) != global_sha):
        raise ReferenceError("global BF16 checkpoint is not the exact index-pinned file")
    source_info = {
        "conditioning_manifest_sha256": conditioning_sha,
        "conditioning_payload_sha256": condition["payload_sha256"],
        "base_qwen_manifest_sha256": bundle["manifest_sha256"],
        "base_qwen_payload_sha256": bundle["payload_sha256"],
        "cache_index_sha256": cache_sha,
        "block0_weights_sha256": block_sha,
        "global_weights_sha256": global_sha,
        "block0_weights": str(block_path),
        "global_weights": str(global_path),
        "continuation_prompt_sha256": condition["qwen_source"]["prompt_sha256"],
        "first_segment_latent_bf16_roundtrip_sha256": first_segment_latent_sha,
        **rope_digests,
    }
    prelude_inputs = {
        "qwen_payload": base_payload,
        "qwen_record": qwen_record,
        "valid_text_mask": list(mask),
        "qwen_manifest_sha256": bundle["manifest_sha256"],
        "qwen_payload_sha256": bundle["payload_sha256"],
        "qwen_tensor_sha256": bundle["qwen_tensor_sha256"],
    }
    return source_info, prelude_inputs


def _range_sha256(path: Path, offset: int, nbytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = nbytes
        while remaining:
            chunk = stream.read(min(remaining, HASH_CHUNK))
            if not chunk:
                raise ReferenceError("payload tensor truncated while checking an alias")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _validate_capture_aliases(capture: dict[str, Any], payload: Path,
                              source_info: dict[str, Any]) -> None:
    records = capture["tensors"]
    if (records["input.latent"]["sha256"]
            != source_info["first_segment_latent_bf16_roundtrip_sha256"]):
        raise ReferenceError(
            "captured first-segment latent differs from the authenticated latent-bank BF16 roundtrip"
        )
    if records["input.vec"]["sha256"] != records["conditioning.vec"]["sha256"]:
        raise ReferenceError("block vec does not alias the captured refiner vec")
    projected = records["conditioning.projected_text"]
    compact_bytes = GEOMETRY["text_tokens"] * GEOMETRY["dim"] * 4
    if (_range_sha256(payload, projected["offset"], compact_bytes)
            != records["input.txt"]["sha256"]):
        raise ReferenceError("block text is not the captured refiner's first 576 projected rows")
    for axis in ("cos", "sin"):
        if records[f"input.rope_{axis}"]["sha256"] != source_info[f"base_first_rope_{axis}_sha256"]:
            raise ReferenceError(f"captured first-segment RoPE differs from its authenticated base: {axis}")


class _Comparison:
    """Chunked MPS-vs-memmap metric accumulator; no activation arrays retained."""

    def __init__(self, torch: Any, device: Any, name: str, expected_count: int):
        self.torch = torch
        self.device = device
        self.name = name
        self.expected_count = expected_count
        self.seen = 0
        self.max_abs = torch.zeros((), dtype=torch.float32, device=device)
        self.sum_abs = torch.zeros((), dtype=torch.float32, device=device)
        self.sum_sq = torch.zeros((), dtype=torch.float32, device=device)
        self.max_rel = torch.zeros((), dtype=torch.float32, device=device)
        self.bad_count = torch.zeros((), dtype=torch.int64, device=device)
        self.first_bad = torch.full((), -1, dtype=torch.int64, device=device)
        self.all_finite = torch.ones((), dtype=torch.bool, device=device)

    def update(self, actual: Any, flat_offset: int, expected_cpu: Any) -> None:
        torch = self.torch
        actual = actual.detach().reshape(-1)
        if actual.device.type != "mps":
            raise ReferenceError(f"{self.name} comparison input left MPS")
        if flat_offset < 0 or flat_offset + actual.numel() > self.expected_count:
            raise ReferenceError(f"{self.name} comparison range is outside its record")
        actual_f32 = actual.to(dtype=torch.float32)
        expected = torch.from_numpy(expected_cpu).to(device=self.device, dtype=torch.float32)
        finite = torch.isfinite(actual_f32) & torch.isfinite(expected)
        diff = (actual_f32 - expected).abs()
        safe_diff = torch.nan_to_num(diff, nan=0.0, posinf=float("inf"), neginf=float("inf"))
        tolerance = ATOL + RTOL * expected.abs()
        bad = (~finite) | (diff > tolerance)
        self.max_abs = torch.maximum(self.max_abs, safe_diff.max())
        self.sum_abs = self.sum_abs + safe_diff.sum()
        self.sum_sq = self.sum_sq + (safe_diff * safe_diff).sum()
        relative = safe_diff / expected.abs().clamp_min(1e-12)
        self.max_rel = torch.maximum(self.max_rel, torch.nan_to_num(relative, nan=0.0).max())
        self.bad_count = self.bad_count + bad.sum(dtype=torch.int64)
        candidate = torch.argmax(bad.to(dtype=torch.int64)) + flat_offset
        choose = (self.first_bad < 0) & bad.any()
        self.first_bad = torch.where(choose, candidate, self.first_bad)
        self.all_finite = self.all_finite & finite.all()
        self.seen += actual.numel()

    def finish(self) -> dict[str, Any]:
        if self.seen != self.expected_count:
            raise ReferenceError(f"{self.name} compared {self.seen} of {self.expected_count} values")
        count = max(1, self.seen)
        float_summary = self.torch.stack((
            self.max_abs,
            self.sum_abs / count,
            (self.sum_sq / count).sqrt(),
            self.max_rel,
        )).detach().cpu().tolist()
        integer_summary = self.torch.stack((self.bad_count, self.first_bad)).detach().cpu().tolist()
        failures = int(integer_summary[0])
        first_bad = int(integer_summary[1]) if integer_summary[1] >= 0 else None
        all_finite = bool(self.all_finite.detach().cpu().item())
        def finite_or_none(value: float) -> float | None:
            return float(value) if math.isfinite(float(value)) else None
        return {
            "checkpoint": self.name,
            "elements": self.seen,
            "max_abs": finite_or_none(float_summary[0]),
            "mean_abs": finite_or_none(float_summary[1]),
            "rms_abs": finite_or_none(float_summary[2]),
            "max_relative": finite_or_none(float_summary[3]),
            "outside_atol_rtol": failures,
            "first_failure_flat_index": first_bad,
            "all_finite": all_finite,
            "allclose_atol_2e-5_rtol_2e-5": failures == 0 and all_finite,
        }


class _ComparisonSet:
    def __init__(self, torch: Any, device: Any, records: dict[str, Any],
                 capture_shapes: dict[str, tuple[int, ...]], payload_map: Any):
        self.torch = torch
        self.device = device
        self.records = records
        self.capture_shapes = capture_shapes
        self.payload_map = payload_map
        self.results: dict[str, dict[str, Any]] = {}
        self.accumulators: dict[str, _Comparison] = {}

    def update(self, capture_name: str, actual: Any, flat_offset: int = 0) -> None:
        if capture_name not in self.records:
            raise ReferenceError(f"upstream produced an unregistered checkpoint: {capture_name}")
        shape = self.capture_shapes[capture_name]
        if (flat_offset == 0 and actual.numel() == math.prod(shape)
                and tuple(actual.shape) != shape):
            raise ReferenceError(
                f"{capture_name} shape mismatch: {tuple(actual.shape)} != {shape}"
            )
        record = self.records[capture_name]
        count = record["nbytes"] // 4
        accumulator = self.accumulators.get(capture_name)
        if accumulator is not None and accumulator.seen == accumulator.expected_count:
            # q/k/v projections share one identical modulated input; the
            # native checkpoint registry stores that input once.
            return
        if accumulator is None:
            if flat_offset != 0:
                raise ReferenceError(f"{capture_name} comparison starts after an unobserved prefix")
            accumulator = _Comparison(self.torch, self.device, capture_name, count)
            self.accumulators[capture_name] = accumulator
        if flat_offset != accumulator.seen:
            raise ReferenceError(
                f"{capture_name} comparison ranges are not contiguous: "
                f"next={flat_offset}, seen={accumulator.seen}"
            )
        flat = actual.detach().reshape(-1)
        offset = 0
        while offset < flat.numel():
            stop = min(offset + COMPARE_CHUNK_ELEMENTS, flat.numel())
            record_start = record["offset"] // 4 + flat_offset + offset
            expected = self.payload_map[record_start:record_start + stop - offset].copy()
            accumulator.update(flat[offset:stop], flat_offset + offset, expected)
            offset = stop

    def finish(self, expected_names: set[str]) -> list[dict[str, Any]]:
        if set(self.accumulators) != expected_names:
            raise ReferenceError(
                "upstream checkpoint inventory mismatch: "
                f"missing={sorted(expected_names - set(self.accumulators))}, "
                f"extra={sorted(set(self.accumulators) - expected_names)}"
            )
        summaries = []
        # dict insertion order is hook/forward execution order; preserve it so
        # first-failure reporting identifies the earliest observed checkpoint.
        for name in self.accumulators:
            result = self.accumulators[name].finish()
            self.results[name] = result
            summaries.append(result)
        return summaries


def _build_attention_adapter(torch: Any, mode_state: dict[str, Any]) -> Any:
    functional = torch.nn.functional

    def parallel_attention(q: tuple[Any, Any], k: tuple[Any, Any], v: tuple[Any, Any],
                           img_q_len: int, img_kv_len: int,
                           attn_mode: str | None = None, text_mask: Any = None,
                           attn_param: dict[str, Any] | None = None) -> Any:
        expected_tokens = GEOMETRY["video_tokens"] + GEOMETRY["text_tokens"]
        valid_tokens = GEOMETRY["video_tokens"] + GEOMETRY["valid_text_tokens"]
        if (attn_mode != "magi_mp_causal" or text_mask is not None or attn_param is None
                or attn_param.get("num_prompts") != 1
                or attn_param.get("chunk_size_tokens", 0) != 0
                or attn_param.get("prompt_video_boundaries_tokens") != [0, GEOMETRY["video_tokens"]]
                or attn_param.get("prompt_video_boundaries") != [0, GEOMETRY["video_tokens"]]
                or attn_param.get("txt_tokens_per_block") != GEOMETRY["text_tokens"]
                or attn_param.get("seg_ar_attention_backend") != "magi"
                or img_q_len != GEOMETRY["video_tokens"] or img_kv_len != img_q_len):
            raise ReferenceError("pinned block called attention outside captured one-segment geometry")
        query, key, value = (torch.cat(parts, dim=1) for parts in (q, k, v))
        expected_shape = (1, expected_tokens, GEOMETRY["heads"], GEOMETRY["head_dim"])
        if any(tuple(tensor.shape) != expected_shape for tensor in (query, key, value)):
            raise ReferenceError("pinned block Q/K/V shape differs from capture geometry")
        expected_dtype = torch.float32 if mode_state["mode"] == "f32" else torch.bfloat16
        if any(t.device.type != "mps" or t.dtype != expected_dtype for t in (query, key, value)):
            raise ReferenceError("attention Q/K/V escaped the requested MPS precision control")
        compare = mode_state["compare"]
        compare.update("block.joint.query", query)
        compare.update("block.joint.key", key)
        compare.update("block.joint.value", value)

        q_heads = query[:, :valid_tokens].permute(0, 2, 1, 3).contiguous()
        k_heads = key[:, :valid_tokens].permute(0, 2, 1, 3).contiguous()
        v_heads = value[:, :valid_tokens].permute(0, 2, 1, 3).contiguous()
        output_heads = torch.zeros(
            (1, GEOMETRY["heads"], expected_tokens, GEOMETRY["head_dim"]),
            dtype=expected_dtype, device=query.device,
        )
        chunks = 0
        for start in range(0, valid_tokens, QUERY_CHUNK):
            stop = min(start + QUERY_CHUNK, valid_tokens)
            attended = functional.scaled_dot_product_attention(
                q_heads[:, :, start:stop], k_heads, v_heads,
                attn_mask=None, dropout_p=0.0, is_causal=False,
            )
            output_heads[:, :, start:stop].copy_(attended)
            chunks += 1
        output = output_heads.permute(0, 2, 1, 3).contiguous().flatten(2)
        compare.update("block.joint.attention", output)
        compare.update("block.img.attn.raw", output[:, :img_q_len])
        compare.update("block.txt.attn.raw", output[:, img_q_len:])
        receipt = mode_state["receipt"]
        receipt["attention_calls"] += 1
        receipt["attention_query_chunks"] += chunks
        receipt["attention_dtype"] = str(output.dtype).removeprefix("torch.")
        return output

    return parallel_attention


def _install_compare_hooks(model: Any, compare: _ComparisonSet,
                           heads: int, head_dim: int) -> list[Any]:
    hooks: list[Any] = []
    torch = __import__("torch")

    def save_output(module: Any, key: str, transform: Any = None) -> None:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise ReferenceError(f"expected tensor checkpoint at block.{key}")
            value = transform(output) if transform is not None else output
            compare.update(f"block.{key}", value)
        hooks.append(module.register_forward_hook(hook))

    def save_input(module: Any, key: str) -> None:
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise ReferenceError(f"expected tensor input checkpoint at block.{key}")
            compare.update(f"block.{key}", inputs[0])
        hooks.append(module.register_forward_pre_hook(hook))

    for stream in ("img", "txt"):
        save_output(getattr(model, f"{stream}_mod"), f"{stream}.modulation")
        save_output(getattr(model, f"{stream}_norm1"), f"{stream}.attn.norm")
        for projection in ("q", "k", "v"):
            linear = getattr(model, f"{stream}_attn_{projection}")
            save_input(linear, f"{stream}.attn.modulated")
            save_output(
                linear, f"{stream}.attn.{projection}",
                lambda output, h=heads, d=head_dim: output.reshape(
                    output.shape[0], output.shape[1], h, d
                ),
            )
        for norm in ("q_norm", "k_norm"):
            save_output(getattr(model, f"{stream}_attn_{norm}"), f"{stream}.attn.{norm}")
        save_output(getattr(model, f"{stream}_attn_proj"), f"{stream}.attn.projected")
        norm2 = getattr(model, f"{stream}_norm2")
        save_input(norm2, f"{stream}.after_attention")
        save_output(norm2, f"{stream}.ff.norm")
        mlp = getattr(model, f"{stream}_mlp")
        save_input(mlp.fc1, f"{stream}.ff.modulated")
        save_output(mlp.fc1, f"{stream}.ff.hidden")
        save_output(mlp.act, f"{stream}.ff.activated")
        save_output(mlp.fc2, f"{stream}.ff.projected")
    return hooks


def _load_capture_tensor(torch: Any, np: Any, payload_map: Any,
                         record: dict[str, Any], shape: tuple[int, ...],
                         device: Any, dtype: Any) -> Any:
    start = record["offset"] // 4
    stop = start + record["nbytes"] // 4
    cpu = torch.from_numpy(payload_map[start:stop].copy()).reshape(shape)
    return cpu.to(device=device, dtype=dtype)


def _load_block_weights(torch: Any, path: Path, target_dtype: Any) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ReferenceError("safetensors is required for the pinned BF16 block") from error
    expected = block_reference._expected_block_keys(2048, 8192, 16)
    source_to_relative = {f"double_blocks.0.{name}": name for name in expected}
    with safe_open(str(path), framework="pt", device="cpu") as source:
        if set(source.keys()) != set(source_to_relative):
            raise ReferenceError("pinned block-0 checkpoint must contain exactly its 32 tensors")
        state = {}
        for source_name, relative_name in source_to_relative.items():
            tensor = source.get_tensor(source_name)
            if tensor.dtype != torch.bfloat16 or list(tensor.shape) != expected[relative_name]:
                raise ReferenceError(f"pinned BF16 block tensor dtype/shape mismatch: {source_name}")
            state[relative_name] = tensor.to(dtype=target_dtype).contiguous()
    if len(state) != 32:
        raise ReferenceError("pinned BF16 block checkpoint tensor count mismatch")
    return state


def _verify_prelude_sources(upstream: Path) -> dict[str, str]:
    """Verify every pinned source file used by PatchEmbed and the text refiner."""
    actual = {}
    for relative, expected in text_reference.PINNED_SOURCE_SHA256.items():
        path = upstream / relative
        if not path.is_file() or path.is_symlink():
            raise ReferenceError(f"pinned prelude source is missing or non-canonical: {relative}")
        digest = _sha256_file(path)
        if digest != expected:
            raise ReferenceError(f"pinned prelude source digest mismatch: {relative}")
        actual[relative] = digest
    return actual


def _install_refiner_attention(torch: Any, state: dict[str, Any]) -> tuple[Any, Any]:
    """Install a bounded MPS adapter for the pinned refiner attention call."""
    module_name = "hymm.models.dit.modules.attention"
    attention_module = types.ModuleType(module_name)
    functional = torch.nn.functional

    def attention(q: Any, k: Any, v: Any, drop_rate: float = 0,
                  attn_mask: Any = None, causal: bool = False) -> Any:
        expected_dtype = torch.float32 if state["mode"] == "f32" else torch.bfloat16
        expected_shape = (
            1, GEOMETRY["text_rms_tokens"], GEOMETRY["heads"], GEOMETRY["head_dim"],
        )
        if (drop_rate != 0 or causal or attn_mask is None
                or tuple(attn_mask.shape) != (1, GEOMETRY["text_rms_tokens"])
                or attn_mask.dtype != torch.bool or attn_mask.device.type != "mps"):
            raise ReferenceError("pinned refiner attention escaped the full-mask contract")
        if (tuple(q.shape) != expected_shape or tuple(k.shape) != expected_shape
                or tuple(v.shape) != expected_shape
                or any(t.device.type != "mps" or t.dtype != expected_dtype for t in (q, k, v))):
            raise ReferenceError("pinned refiner Q/K/V escaped the selected MPS precision control")

        compare = state["compare"]
        index = int(state["attention_block"])
        prefix = f"conditioning.block{index}."
        if index not in (0, 1):
            raise ReferenceError("pinned refiner executed more than two attention blocks")
        compare.update(prefix + "q", q.reshape(1, GEOMETRY["text_rms_tokens"], -1))
        compare.update(prefix + "k", k.reshape(1, GEOMETRY["text_rms_tokens"], -1))
        compare.update(prefix + "v", v.reshape(1, GEOMETRY["text_rms_tokens"], -1))

        valid = int(state["valid_text_tokens"])
        q_valid = q[:, :valid].permute(0, 2, 1, 3).contiguous()
        k_valid = k[:, :valid].permute(0, 2, 1, 3).contiguous()
        v_valid = v[:, :valid].permute(0, 2, 1, 3).contiguous()
        attended = torch.empty_like(q_valid)
        chunks = 0
        for start in range(0, valid, QUERY_CHUNK):
            stop = min(start + QUERY_CHUNK, valid)
            attended[:, :, start:stop].copy_(functional.scaled_dot_product_attention(
                q_valid[:, :, start:stop], k_valid, v_valid,
                attn_mask=None, dropout_p=0.0, is_causal=False,
            ))
            chunks += 1
        padded = torch.zeros(
            (1, GEOMETRY["text_rms_tokens"], GEOMETRY["heads"], GEOMETRY["head_dim"]),
            dtype=expected_dtype, device=q.device,
        )
        padded[:, :valid].copy_(attended.permute(0, 2, 1, 3))
        output = padded.flatten(2)
        compare.update(prefix + "attention", output)
        state["attention_block"] = index + 1
        state["attention_query_chunks"] += chunks
        return output

    attention_module.attention = attention  # type: ignore[attr-defined]
    sys.modules[module_name] = attention_module
    parent = sys.modules.get("hymm.models.dit.modules")
    if parent is not None:
        setattr(parent, "attention", attention_module)
    return attention_module, attention


def _build_prelude_modules(torch: Any, upstream: Path, helpers: tuple[Any, ...],
                           dtype: Any) -> dict[str, Any]:
    """Construct the unchanged pinned global modules on meta with chosen dtype."""
    import importlib

    block_reference._namespace(
        "hymm.models.text_encoder", upstream / "hymm/models/text_encoder",
    )
    text_module = importlib.import_module("hymm.models.text_encoder.token_refiner")
    embed_layers = importlib.import_module("hymm.models.dit.modules.embed_layers")
    mlp_layers = sys.modules["hymm.models.dit.modules.mlp_layers"]
    meta = torch.device("meta")
    modules = {
        "txt_in": text_module.SingleTokenRefinerPerBlock(
            in_channels=3584, hidden_size=GEOMETRY["dim"],
            heads_num=GEOMETRY["heads"], depth=2,
            mlp_width_ratio=4.0, mlp_drop_rate=0.0, act_type="silu",
            qk_norm=False, qkv_bias=True, attn_mode="torch", seg_pos_emb_num=0,
            dtype=dtype, device=meta,
        ),
        "time_in": embed_layers.TimestepEmbedder(
            GEOMETRY["dim"], torch.nn.SiLU, dtype=dtype, device=meta,
        ),
        "img_in": embed_layers.PatchEmbed(
            patch_size=(1, 1, 1), in_chans=32, embed_dim=GEOMETRY["dim"],
            dtype=dtype, device=meta,
        ),
        "final_layer": mlp_layers.FinalLayerPerBlock(
            GEOMETRY["dim"], 1, 32, helpers[0]("silu"), device=meta, dtype=dtype,
        ),
    }
    for prefix, module in modules.items():
        expected = cache_reference._GLOBAL_EXPECTED_COUNTS[f"{prefix}."]
        if len(module.state_dict()) != expected:
            raise ReferenceError(f"pinned global module inventory mismatch: {prefix}")
    return modules


def _load_prelude_weights(torch: Any, path: Path, upstream: Path,
                          helpers: tuple[Any, ...], device: Any,
                          dtype: Any) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ReferenceError("safetensors is required for the pinned BF16 globals") from error
    modules = _build_prelude_modules(torch, upstream, helpers, dtype)
    expected_names = {
        f"{prefix}.{name}"
        for prefix, module in modules.items()
        for name in module.state_dict()
    }
    loaded = {}
    with safe_open(str(path), framework="pt", device="cpu") as source:
        if set(source.keys()) != expected_names:
            raise ReferenceError("pinned global checkpoint must contain exactly 48 registered tensors")
        for prefix, module in modules.items():
            state = {}
            for relative_name, target in module.state_dict().items():
                source_name = f"{prefix}.{relative_name}"
                tensor = source.get_tensor(source_name)
                if (tensor.dtype != torch.bfloat16
                        or tuple(tensor.shape) != tuple(target.shape)):
                    raise ReferenceError(f"pinned BF16 global tensor dtype/shape mismatch: {source_name}")
                state[relative_name] = tensor.to(dtype=dtype).contiguous()
            try:
                module.load_state_dict(state, strict=True, assign=True)
            except (RuntimeError, TypeError) as error:
                raise ReferenceError(f"could not bind pinned global module {prefix}: {error}") from None
            module.to(device=device, dtype=dtype).eval()
            loaded[prefix] = module
            del state
    return loaded


def _install_prelude_hooks(torch: Any, modules: dict[str, Any],
                           compare: _ComparisonSet) -> list[Any]:
    hooks: list[Any] = []

    def save_output(module: Any, key: str, transform: Any = None) -> None:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise ReferenceError(f"expected tensor checkpoint at {key}")
            compare.update(key, transform(output) if transform is not None else output)
        hooks.append(module.register_forward_hook(hook))

    def save_input(module: Any, key: str, input_index: int = 0) -> None:
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if len(inputs) <= input_index or not isinstance(inputs[input_index], torch.Tensor):
                raise ReferenceError(f"expected tensor input checkpoint at {key}")
            compare.update(key, inputs[input_index])
        hooks.append(module.register_forward_pre_hook(hook))

    time_in = modules["time_in"]
    for module, key in (
        (time_in.mlp[0], "conditioning.time_in.mlp1"),
        (time_in.mlp[1], "conditioning.time_in.silu"),
        (time_in.mlp[2], "conditioning.vec"),
    ):
        save_output(module, key)

    refiner = modules["txt_in"]
    refiner_timestep: dict[str, Any] = {"value": None}

    def remember_refiner_timestep(_module: Any, _inputs: tuple[Any, ...],
                                  output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise ReferenceError("expected tensor checkpoint at conditioning.refiner_timestep")
        refiner_timestep["value"] = output

    def save_context_sum(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        timestep = refiner_timestep["value"]
        if not isinstance(output, torch.Tensor) or not isinstance(timestep, torch.Tensor):
            raise ReferenceError("refiner timestep/context sum inputs were not captured")
        # The pinned forward adds these two module outputs immediately after
        # c_embedder returns and before input_embedder runs. Observe that sum
        # at this hook to retain the source execution order in first-failure
        # reporting.
        compare.update("conditioning.c", timestep + output)

    hooks.append(refiner.t_embedder.register_forward_hook(remember_refiner_timestep))
    hooks.append(refiner.c_embedder.register_forward_hook(save_context_sum))
    for module, key in (
        (refiner.t_embedder.mlp[0], "conditioning.refiner_timestep_mlp1"),
        (refiner.t_embedder.mlp[1], "conditioning.refiner_timestep_silu"),
        (refiner.t_embedder.mlp[2], "conditioning.refiner_timestep"),
        (refiner.c_embedder.linear_1, "conditioning.context_linear1"),
        (refiner.c_embedder.act_1, "conditioning.context_silu"),
        (refiner.c_embedder.linear_2, "conditioning.context_embed"),
        (refiner.input_embedder, "conditioning.input_embed"),
        (refiner.individual_token_refiner, "conditioning.projected_text"),
    ):
        save_output(module, key)
    save_input(refiner.c_embedder, "conditioning.masked_mean")

    for index, block in enumerate(refiner.individual_token_refiner.blocks):
        prefix = f"conditioning.block{index}."
        save_output(block.adaLN_modulation[0], prefix + "condition_silu")
        save_output(block.adaLN_modulation[1], prefix + "adaLN")

        def save_gates(_module: Any, _inputs: tuple[Any, ...], output: Any,
                       block_prefix: str = prefix) -> None:
            gate_msa, gate_mlp = output.chunk(2, dim=-1)
            compare.update(block_prefix + "gate_msa", gate_msa)
            compare.update(block_prefix + "gate_mlp", gate_mlp)

        hooks.append(block.adaLN_modulation.register_forward_hook(save_gates))
        for module, key in (
            (block.norm1, prefix + "norm1"),
            (block.self_attn_proj, prefix + "proj"),
            (block.norm2, prefix + "norm2"),
            (block.mlp.fc1, prefix + "mlp_fc1"),
            (block.mlp.act, prefix + "mlp_silu"),
            (block.mlp.fc2, prefix + "mlp_fc2"),
        ):
            save_output(module, key)
        save_output(
            block.self_attn_qkv, prefix + "qkv",
            lambda output: output.reshape(
                output.shape[0], output.shape[1], 3, GEOMETRY["dim"],
            ),
        )
        save_input(block.norm2, prefix + "residual1")

        def save_block_output(_module: Any, _inputs: tuple[Any, ...], output: Any,
                              block_prefix: str = prefix) -> None:
            if not isinstance(output, torch.Tensor):
                raise ReferenceError(f"expected tensor checkpoint at {block_prefix}output")
            compare.update(block_prefix + "output", output)
            compare.update(block_prefix + "residual2", output)

        hooks.append(block.register_forward_hook(save_block_output))
    return hooks


def _load_prelude_qwen(torch: Any, np: Any, source: dict[str, Any],
                       device: Any, dtype: Any) -> Any:
    record = source["qwen_record"]
    payload = source["qwen_payload"]
    if _range_sha256(payload, record["offset"], record["nbytes"]) != record["sha256"]:
        raise ReferenceError("Qwen tensor changed after source authentication")
    raw = np.memmap(
        payload, mode="r", dtype="<u2", offset=record["offset"],
        shape=tuple(record["shape"]),
    )
    cpu = np.array(raw, copy=True)
    del raw
    qwen_bf16 = torch.from_numpy(cpu).view(torch.bfloat16)
    if tuple(qwen_bf16.shape) != (1, GEOMETRY["text_rms_tokens"], 3584):
        raise ReferenceError("authenticated Qwen tensor has an unexpected retained-padding shape")
    return qwen_bf16.to(device=device, dtype=dtype)


def _run_prelude_control(torch: Any, np: Any, capture: dict[str, Any],
                         payload_map: Any, source: dict[str, Any], upstream: Path,
                         helpers: tuple[Any, ...], attention_state: dict[str, Any],
                         mode: str) -> dict[str, Any]:
    device = torch.device("mps")
    dtype = torch.float32 if mode == "f32" else torch.bfloat16
    attention_state.update({
        "mode": mode,
        "valid_text_tokens": GEOMETRY["valid_text_tokens"],
        "attention_block": 0,
        "attention_query_chunks": 0,
        "compare": None,
    })
    compare = _ComparisonSet(
        torch, device, capture["tensors"], _capture_shapes(), payload_map,
    )
    attention_state["compare"] = compare
    modules = _load_prelude_weights(
        torch, Path(capture["global_weights"]), upstream, helpers, device, dtype,
    )
    hooks = _install_prelude_hooks(torch, modules, compare)
    embed_layers = sys.modules["hymm.models.dit.modules.embed_layers"]
    original_timestep_embedding = embed_layers.timestep_embedding
    active_sinusoidal_checkpoint: dict[str, str | None] = {"name": None}

    def compare_timestep_embedding(t: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_timestep_embedding(t, *args, **kwargs)
        name = active_sinusoidal_checkpoint["name"]
        if name is not None:
            compare.update(name, result)
        return result

    embed_layers.timestep_embedding = compare_timestep_embedding
    try:
        with torch.inference_mode():
            latent = _load_capture_tensor(
                torch, np, payload_map, capture["tensors"]["input.latent"],
                (1, 32, 30, 9, 16), device, dtype,
            )
            # Match the pinned cached-generation path: original PatchEmbed
            # receives [B,C,T,H,W], uses its pointwise Conv3d, then flattens
            # T,H,W into token order.
            patch_tokens = modules["img_in"](latent)
            compare.update("input.img", patch_tokens)
            del patch_tokens, latent

            timestep = torch.tensor(
                [float(capture["timestep"])], dtype=torch.float32, device=device,
            )
            compare.update("conditioning.timestep", timestep)
            active_sinusoidal_checkpoint["name"] = "conditioning.time_in.sinusoidal"
            modules["time_in"](timestep)
            active_sinusoidal_checkpoint["name"] = None

            qwen = _load_prelude_qwen(torch, np, source, device, dtype)
            mask = torch.tensor(
                [source["valid_text_mask"]], dtype=torch.bool, device=device,
            )
            if tuple(mask.shape) != (1, GEOMETRY["text_rms_tokens"]):
                raise ReferenceError("authenticated refiner mask has the wrong full extent")
            compare.update("conditioning.input_compact", qwen)
            active_sinusoidal_checkpoint["name"] = "conditioning.refiner_timestep_sinusoidal"
            modules["txt_in"](qwen, timestep, mask)
            active_sinusoidal_checkpoint["name"] = None
    finally:
        embed_layers.timestep_embedding = original_timestep_embedding
        for hook in hooks:
            hook.remove()
        attention_state["compare"] = None

    if attention_state["attention_block"] != 2:
        raise ReferenceError("pinned full text refiner did not execute exactly two attention blocks")
    expected_names = {
        name for name in _capture_shapes() if name.startswith("conditioning.")
    } | {"input.img"}
    if len(expected_names) != 54:
        raise ReferenceError(f"prelude checkpoint registry is malformed: {len(expected_names)}")
    summaries = compare.finish(expected_names)
    first_failure = next(
        (item for item in summaries if not item["allclose_atol_2e-5_rtol_2e-5"]), None
    )
    return {
        "status": "completed",
        "mode": mode,
        "parameter_and_activation_dtype": "F32" if mode == "f32" else "BF16",
        "source_weight_values": "exact pinned BF16 global checkpoint values",
        "device": "mps",
        "patch_call": "pinned PatchEmbed.forward using original pointwise Conv3d, then [T,H,W] flatten",
        "refiner_input": "authenticated state-26 Qwen BF16 tensor, full 1280 rows and 557-row prefix mask",
        "timestep_frequency_path": "pinned F32 timestep_embedding at t=1000; module linear dtype follows weights",
        "attention": "unpadded PyTorch MPS SDPA with zero padded query outputs; not Magi/CUDA parity",
        "attention_calls": attention_state["attention_block"],
        "attention_query_chunks": attention_state["attention_query_chunks"],
        "checkpoint_order": "pinned forward/hook execution order; patch, DiT timestep, then refiner",
        "checkpoint_count": len(summaries),
        "first_checkpoint_outside_tolerance": first_failure,
        "checkpoints": summaries,
    }


def _run_control(torch: Any, np: Any, capture: dict[str, Any], payload_map: Any,
                 payload: Path, upstream: Path, adapter: Any, mode_state: dict[str, Any],
                 mode: str) -> dict[str, Any]:
    device = torch.device("mps")
    dtype = torch.float32 if mode == "f32" else torch.bfloat16
    mode_state["mode"] = mode
    records = capture["tensors"]
    shapes = _capture_shapes()
    compare = _ComparisonSet(torch, device, records, shapes, payload_map)
    receipt = {"attention_calls": 0, "attention_query_chunks": 0}
    mode_state["compare"] = compare
    mode_state["receipt"] = receipt
    helpers = block_reference._install_upstream_modules(upstream)
    posemb = helpers[-1]

    def apply_rotary_capture(q: Any, k: Any, freqs_cis: Any,
                             head_first: bool = False) -> tuple[Any, Any]:
        if q.device.type != "mps" or k.device.type != "mps":
            raise ReferenceError("pinned RoPE escaped MPS")
        q_rot, k_rot = posemb.apply_rotary_emb(q, k, freqs_cis, head_first=head_first)
        compare.update("block.img.attn.q_rope", q_rot)
        compare.update("block.img.attn.k_rope", k_rot)
        return q_rot, k_rot

    # The module's global adapter symbol is fixed at class compilation; this
    # closure dispatches according to the one active precision mode.
    block_type = block_reference._extract_pinned_block(
        upstream, torch, adapter, apply_rotary_capture, helpers,
    )
    block = block_type(
        hidden_size=2048, heads_num=16, mlp_width_ratio=4.0,
        mlp_act_type="gelu_tanh", attn_mode="magi_mp_causal", qk_norm=True,
        qk_norm_type="rms", qkv_bias=True, dtype=dtype, device=torch.device("meta"),
    )
    block.eval()
    state = _load_block_weights(torch, Path(capture["block_weights"]), dtype)
    block.load_state_dict(state, strict=True, assign=True)
    block.to(device=device, dtype=dtype).eval()
    del state

    inputs = {}
    for key, shape in (
        ("img", (1, 4320, 2048)), ("txt", (1, 576, 2048)),
        ("vec", (1, 2048)), ("rope_cos", (4320, 128)),
        ("rope_sin", (4320, 128)),
    ):
        input_dtype = torch.float32 if key.startswith("rope_") else dtype
        inputs[key] = _load_capture_tensor(
            torch, np, payload_map, records[f"input.{key}"], shape, device, input_dtype,
        )
    inputs["vec_txt"] = inputs["vec"]
    hooks = _install_compare_hooks(block, compare, 16, 128)
    attn_param = {
        "num_prompts": 1,
        "txt_tokens_per_block": GEOMETRY["text_tokens"],
        "prompt_video_boundaries_tokens": [0, GEOMETRY["video_tokens"]],
        "prompt_video_boundaries": [0, GEOMETRY["video_tokens"]],
        "chunk_size_tokens": 0,
        "seg_ar_attention_backend": "magi",
    }
    try:
        with torch.inference_mode():
            img_output, txt_output = block(
                img=inputs["img"], txt=inputs["txt"], vec=inputs["vec"],
                vec_txt=inputs["vec_txt"],
                freqs_cis=(inputs["rope_cos"], inputs["rope_sin"]),
                text_mask=None, attn_param=attn_param, is_flash=False,
                txt_tokens_per_block=GEOMETRY["text_tokens"],
                prompt_video_boundaries=[0, GEOMETRY["video_tokens"]],
            )
        compare.update("block.img.output", img_output)
        compare.update("block.txt.output", txt_output)
    finally:
        for hook in hooks:
            hook.remove()
    if receipt["attention_calls"] != 1:
        raise ReferenceError("upstream block did not execute exactly one joint attention call")
    expected_names = {f"block.{name}" for name in serial_reference.BLOCK_CAPTURE_NAMES}
    summaries = compare.finish(expected_names)
    first_failure = next(
        (item for item in summaries if not item["allclose_atol_2e-5_rtol_2e-5"]), None
    )
    return {
        "status": "completed",
        "mode": mode,
        "parameter_and_activation_dtype": "F32" if mode == "f32" else "BF16",
        "rope_table_dtype": "F32 (pinned apply_rotary_emb promotes Q/K to F32 and casts its result back)",
        "device": "mps",
        "checkpoint_order": "upstream forward/hook insertion order; img/txt are interleaved by source execution",
        "attention": "chunked PyTorch MPS SDPA; not Magi/CUDA parity",
        "attention_calls": receipt["attention_calls"],
        "attention_query_chunks": receipt["attention_query_chunks"],
        "checkpoint_count": len(summaries),
        "first_checkpoint_outside_tolerance": first_failure,
        "checkpoints": summaries,
    }


def _replay_native_qkv(torch: Any, np: Any, records: dict[str, Any],
                       payload_map: Any, device: Any) -> dict[str, Any]:
    """Replay authenticated native joint Q/K/V, isolating attention dispatch."""
    valid = GEOMETRY["video_tokens"] + GEOMETRY["valid_text_tokens"]
    total = GEOMETRY["video_tokens"] + GEOMETRY["text_tokens"]
    dim = GEOMETRY["dim"]
    heads = GEOMETRY["heads"]
    head_dim = GEOMETRY["head_dim"]
    qkv = {}
    for axis, capture_name in (("q", "block.joint.query"),
                               ("k", "block.joint.key"),
                               ("v", "block.joint.value")):
        shape = (1, total, heads, head_dim)
        record = records[capture_name]
        qkv[axis] = _load_capture_tensor(
            torch, np, payload_map, record, shape, device, torch.float32,
        )
    query = qkv["q"][:, :valid].permute(0, 2, 1, 3).contiguous()
    key = qkv["k"][:, :valid].permute(0, 2, 1, 3).contiguous()
    value = qkv["v"][:, :valid].permute(0, 2, 1, 3).contiguous()
    compare = _ComparisonSet(
        torch, device, records, _capture_shapes(), payload_map,
    )
    functional = torch.nn.functional
    for start in range(0, valid, QUERY_CHUNK):
        stop = min(start + QUERY_CHUNK, valid)
        output = functional.scaled_dot_product_attention(
            query[:, :, start:stop], key, value,
            attn_mask=None, dropout_p=0.0, is_causal=False,
        )
        rows = output.permute(0, 2, 1, 3).contiguous().reshape(-1)
        compare.update("block.joint.attention", rows, flat_offset=start * dim)
    if valid < total:
        padding = torch.zeros((total - valid) * dim, dtype=torch.float32, device=device)
        compare.update("block.joint.attention", padding, flat_offset=valid * dim)
    summary = compare.finish({"block.joint.attention"})[0]
    return {
        "status": "completed",
        "source": "authenticated native joint Q/K/V tensors",
        "device": "mps",
        "compute_dtype": "F32",
        "attention": "chunked PyTorch MPS SDPA; not Magi/CUDA parity",
        "valid_joint_tokens": valid,
        "padded_query_rows_zeroed": total - valid,
        "query_chunk_tokens": QUERY_CHUNK,
        "comparison": summary,
    }


def _write_json_fresh(path_arg: str | Path, document: dict[str, Any]) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    parent = path.parent.resolve(strict=True)
    path = parent / path.name
    if path.exists() or path.is_symlink():
        raise ReferenceError(f"refusing to overwrite oracle report: {path}")
    encoded = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Linking publishes the completed inode only if the destination is
        # still absent. os.replace() would overwrite a file created after the
        # initial existence check, losing another writer's report.
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, help="authenticated native capture JSON")
    parser.add_argument("--capture-sha256", required=True, help="expected capture manifest SHA-256")
    parser.add_argument("--output", required=True, help="fresh JSON output path (never overwritten)")
    parser.add_argument("--upstream", required=True, help="pinned upstream checkout root")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="authenticate capture and source bindings without importing torch or using MPS",
    )
    parser.add_argument(
        "--include-prelude", action="store_true",
        help="also recompute pinned patch and the full 1280-row text-refiner prelude on MPS",
    )
    parser.add_argument(
        "--prelude-only", action="store_true",
        help="run only the optional patch/refiner controls (implies --include-prelude)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    include_prelude = args.include_prelude or args.prelude_only
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "pinned_upstream_revision": PINNED_REVISION,
        "comparison_tolerance": {"atol": ATOL, "rtol": RTOL},
        "capture_manifest": str(Path(args.capture).expanduser().resolve()),
        "capture_manifest_sha256": _digest(args.capture_sha256, "capture SHA-256"),
        "controls": {},
        "attention_replay": None,
        "prelude_controls": {},
        "prelude_only": args.prelude_only,
        "scope_note": (
            "First-segment, first-step, block-0 only. F32 expands exact BF16 checkpoint values "
            "to F32; BF16 uses BF16 parameters and activations. Both use the unchanged pinned "
            "MMDoubleStreamBlock with a chunked PyTorch MPS SDPA adapter. That adapter is not "
            "Magi/CUDA parity. The common-QKV replay isolates this adapter from upstream "
            "pre-attention block operations. Block controls start from authenticated native "
            "patch/refiner outputs. Optional prelude controls independently recompute the pinned "
            "PatchEmbed and full retained-padding 1280-row text refiner from authenticated first-"
            "chunk latent/Qwen inputs; they remain separate from the common block-input controls. "
            "Pinned block Q/K RMSNorm is rowwise over the final head dimension; text rows at block "
            "input remain the captured compact 576."
        ),
    }
    exit_code = 1
    try:
        capture_path = _regular_file(args.capture, "native capture manifest")
        capture = _json_no_duplicates(capture_path.read_bytes(), "native capture manifest")
        payload, tensor_names = _validate_flat_capture(
            capture, capture_path, report["capture_manifest_sha256"],
        )
        sources, prelude_inputs = _authenticate_sources(capture)
        _validate_capture_aliases(capture, payload, sources)
        upstream_path = Path(args.upstream).expanduser()
        if not upstream_path.is_absolute():
            upstream_path = Path.cwd() / upstream_path
        upstream = upstream_path.resolve(strict=True)
        if upstream_path.is_symlink() or not upstream.is_dir():
            raise ReferenceError("pinned upstream path is not a directory")
        source_hashes = block_reference._verify_upstream(upstream)
        prelude_source_hashes = (
            _verify_prelude_sources(upstream) if include_prelude else {}
        )
        report.update({
            "capture_payload_bytes": capture["payload_bytes"],
            "capture_payload_sha256": capture["payload_sha256"],
            "tensor_record_count": len(tensor_names),
            "authenticated_sources": sources,
            "upstream_source_sha256": source_hashes,
            "prelude_source_sha256": prelude_source_hashes,
        })
        if args.validate_only:
            report["status"] = "validated_only"
            exit_code = 0
        else:
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
                raise ReferenceError("MPS fallback must be disabled for this oracle")
            if os.environ.get("HY_TEXT_MLP_FC2_FIXED_M", "0") not in ("", "0"):
                raise ReferenceError("fixed-M text MLP path must be disabled for this oracle")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
            os.environ["HY_TEXT_MLP_FC2_FIXED_M"] = "0"
            try:
                import numpy as np
                import torch
            except ImportError as error:
                raise ReferenceError(f"MPS oracle dependency unavailable: {error}") from error
            if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
                raise ReferenceError("this oracle requires an available PyTorch MPS device")
            device = torch.device("mps")
            payload_map = np.memmap(payload, dtype="<f4", mode="r", shape=(capture["payload_bytes"] // 4,))
            control_errors = []
            if not args.prelude_only:
                mode_state: dict[str, Any] = {}
                adapter = _build_attention_adapter(torch, mode_state)
                for mode in ("f32", "bf16"):
                    try:
                        report["controls"][mode] = _run_control(
                            torch, np, capture, payload_map, payload, upstream, adapter,
                            mode_state, mode,
                        )
                    except Exception as error:
                        report["controls"][mode] = {
                            "status": "failed", "error_type": type(error).__name__,
                            "error": str(error),
                        }
                        control_errors.append(mode)
                    finally:
                        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                            torch.mps.empty_cache()
                try:
                    report["attention_replay"] = _replay_native_qkv(
                        torch, np, capture["tensors"], payload_map, device,
                    )
                except Exception as error:
                    report["attention_replay"] = {
                        "status": "failed", "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    control_errors.append("attention_replay")
            if include_prelude:
                helpers = block_reference._install_upstream_modules(upstream)
                attention_state: dict[str, Any] = {}
                _install_refiner_attention(torch, attention_state)
                for mode in ("f32", "bf16"):
                    try:
                        report["prelude_controls"][mode] = _run_prelude_control(
                            torch, np, capture, payload_map, prelude_inputs, upstream,
                            helpers, attention_state, mode,
                        )
                    except Exception as error:
                        report["prelude_controls"][mode] = {
                            "status": "failed", "error_type": type(error).__name__,
                            "error": str(error),
                        }
                        control_errors.append(f"prelude_{mode}")
                    finally:
                        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                            torch.mps.empty_cache()
            del payload_map
            report["status"] = "completed" if not control_errors else "partial"
            report["failed_controls"] = control_errors
            exit_code = 0 if not control_errors else 2
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
    try:
        output = _write_json_fresh(args.output, report)
    except Exception as error:
        print(f"oracle report error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": report.get("status"), "report": str(output), "exit_code": exit_code,
        "failed_controls": report.get("failed_controls", []),
        "error": report.get("error"),
    }, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
