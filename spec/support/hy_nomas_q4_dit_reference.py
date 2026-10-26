#!/usr/bin/env python3
"""Produce a bounded MPS reference for one full first-segment Hy-Nomas DiT pass.

The input is the saved Qwen state-26 crop plus a deterministic latent. Global
modules and the BF16 control use the authenticated full F32 source rounded to
BF16 on MPS. The Q4 path independently decodes each packed block on MPS,
rounds it to the cache's BF16 resident values, then runs the same pinned
upstream block. Attention is an explicit MPS valid-prefix SDPA adapter; this
is not Magi/CUDA parity or diffusion generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SPEC_SUPPORT = ROOT / "spec" / "support"
EXAMPLES_SUPPORT = ROOT / "examples" / "support"
for _support in (SPEC_SUPPORT, EXAMPLES_SUPPORT):
    if str(_support) not in sys.path:
        sys.path.insert(0, str(_support))

import hy_nomas_conditioned_four_reference as conditioned_reference  # noqa: E402
import hy_nomas_full_checkpoint as checkpoint_reference  # noqa: E402
import hy_nomas_full_j_reference as full_j_reference  # noqa: E402
import hy_nomas_pretrained_serial_reference as serial_reference  # noqa: E402
import hy_nomas_reference as block_reference  # noqa: E402
import hy_nomas_text_conditioning_reference as text_reference  # noqa: E402


SCHEMA = "hy-nomas-q4-dit-reference-v1"
SCHEMA_VERSION = 1
CACHE_INDEX_SCHEMA = "hy-nomas-q4-dit-cache-index-v1"
CACHE_MANIFEST_SCHEMA = "hy-nomas-q4-block-cache-v1"
PINNED_REVISION = serial_reference.PINNED_REVISION
DEFAULT_UPSTREAM = conditioned_reference.DEFAULT_UPSTREAM
DEFAULT_WEIGHTS = checkpoint_reference.DEFAULT_OUTPUT
DEFAULT_CACHE_INDEX = checkpoint_reference.ALLOWED_OUTPUT_ROOT / "weights" / "q4cache" / "index.json"
DEFAULT_QWEN_REFERENCE = text_reference.DEFAULT_QWEN_REFERENCE
DEFAULT_QWEN_SIDECAR = text_reference.DEFAULT_QWEN_SIDECAR
SOURCE_SHA256 = checkpoint_reference.PINNED.sha256
SOURCE_BYTES = checkpoint_reference.PINNED.size_bytes
SOURCE_HEADER_SHA256 = checkpoint_reference.SOURCE_HEADER_SHA256
ATOL = serial_reference.ATOL
RTOL = serial_reference.RTOL
JSON_LIMIT = 2 * 1024 * 1024
MAX_CACHE_MANIFEST_BYTES = 1024 * 1024
SIDE_CAR_LIMIT = 512 * 1024 * 1024
BLOCK_COUNT = 54
VIDEO_TOKENS = 5280
TEXT_TOKENS = 1280
VALID_TEXT_TOKENS = 23
PADDED_TEXT_TOKENS = TEXT_TOKENS - VALID_TEXT_TOKENS
VALID_JOINT_TOKENS = VIDEO_TOKENS + VALID_TEXT_TOKENS
TOTAL_TOKENS = VIDEO_TOKENS + TEXT_TOKENS
QUERY_CHUNK = 256
ATTENTION_CHUNKS_PER_BLOCK = math.ceil(VALID_JOINT_TOKENS / QUERY_CHUNK)
CAPTURE_BLOCKS = (0, 7, 53)
MASK = [1] * VALID_TEXT_TOKENS + [0] * PADDED_TEXT_TOKENS
GEOMETRY = {
    "batches": 1,
    "video_tokens": VIDEO_TOKENS,
    "text_tokens": TEXT_TOKENS,
    "valid_text_tokens": VALID_TEXT_TOKENS,
    "padded_text_tokens": PADDED_TEXT_TOKENS,
    "valid_joint_tokens": VALID_JOINT_TOKENS,
    "total_tokens": TOTAL_TOKENS,
    "dim": 2048,
    "heads": 16,
    "head_dim": 128,
    "ffn_dim": 8192,
    "double_blocks": BLOCK_COUNT,
    "single_blocks": 0,
    "rope_grid": [6, 22, 40],
    "rope_axis_dims": [16, 56, 56],
    "rope_theta": 256.0,
    "query_chunk": QUERY_CHUNK,
    "segments": 1,
    "chunk_size_tokens": 0,
}
INPUT_GENERATION = {
    "kind": "deterministic MPS F32 latent formula plus pinned upstream ND RoPE",
    "latent_formula": (
        "x[i]=0.1*sin(float32(i)*0.017+0.11)+"
        "0.05*cos(float32(i)*0.0031+0.077)"
    ),
    "latent_shape": [1, 32, 6, 22, 40],
    "rope": {
        "implementation": "pinned upstream posemb_layers.get_nd_rotary_pos_embed",
        "grid": [6, 22, 40],
        "grid_axis_order": ["temporal", "height", "width"],
        "axis_dims": [16, 56, 56],
        "theta": 256.0,
        "use_real": True,
        "device": "mps",
    },
}
PRECISION = {
    "global_source_dtype": "F32",
    "global_values": "F32->BF16->F32 on MPS",
    "global_storage": "F32",
    "block_source_dtype": "F32",
    "block_values_q4": "Q4_K 8 ranges decoded on MPS, rounded to BF16, represented F32",
    "block_values_bf16": "F32->BF16->F32 on MPS",
    "block_storage": "F32",
    "compute": "F32",
    "attention": "PyTorch MPS SDPA F32 valid-prefix adapter",
    "native_bf16_parameter_storage": False,
    "native_bf16_compute": False,
    "cuda_magi_parity": False,
}

_BLOCK_RELATIVE_SHAPES = block_reference._expected_block_keys(2048, 8192, 16)
_GLOBAL_EXPECTED_COUNTS = {"txt_in.": 38, "time_in.": 4, "img_in.": 2, "final_layer.": 4}
SOURCE_DATA_OFFSET = 8 + checkpoint_reference.PINNED.header_bytes
_Q4_GEOMETRY = {
    key: GEOMETRY[key]
    for key in ("batches", "video_tokens", "text_tokens", "dim", "heads", "head_dim", "ffn_dim")
}
_CONDITIONING = {
    "qwen_state_index": 26,
    "crop_start": 108,
    "crop_tokens": TEXT_TOKENS,
    "valid_text_tokens": VALID_TEXT_TOKENS,
    "padded_text_tokens": PADDED_TEXT_TOKENS,
    "refiner_rows": TEXT_TOKENS,
}
_PADDING_POLICY = {
    "kind": "valid_prefix_attention_zero_padded_query_outputs",
    "valid_joint_prefix": VALID_JOINT_TOKENS,
    "valid_text_rows": VALID_TEXT_TOKENS,
    "padded_text_rows": PADDED_TEXT_TOKENS,
    "refiner_rows": TEXT_TOKENS,
    "padded_query_outputs": "attention output zero-filled; residual stream retained",
}


class ReferenceError(RuntimeError):
    """A fail-closed source, cache, execution, or fixture error."""


def _same_json_types(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return (type(actual) is dict and set(actual) == set(expected)
                and all(_same_json_types(value, actual[key]) for key, value in expected.items()))
    if isinstance(expected, list):
        return (type(actual) is list and len(actual) == len(expected)
                and all(_same_json_types(left, right) for left, right in zip(expected, actual)))
    return type(actual) is type(expected) and actual == expected


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ReferenceError(f"duplicate key in {label}")
            result[key] = value
        return result

    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except ReferenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ReferenceError(f"invalid {label} JSON") from None
    if type(result) is not dict:
        raise ReferenceError(f"{label} root must be an object")
    return result


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReferenceError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _emit_stage_telemetry(torch: Any, stage: str, **fields: Any) -> None:
    """Emit CPU-side progress plus non-synchronizing MPS allocator counters."""
    mps = getattr(torch, "mps", None)
    memory: dict[str, int | None] = {}
    for method, label in (
        ("current_allocated_memory", "mps_current_allocated_bytes"),
        ("driver_allocated_memory", "mps_driver_allocated_bytes"),
    ):
        counter = getattr(mps, method, None)
        try:
            memory[label] = int(counter()) if callable(counter) else None
        except Exception:
            # Telemetry must not change whether an otherwise valid oracle run
            # succeeds. Unsupported counters remain explicitly unavailable.
            memory[label] = None
    marker = {
        "stage": stage,
        "monotonic_ns": time.monotonic_ns(),
        **fields,
        **memory,
    }
    print(json.dumps(marker, sort_keys=True), file=sys.stderr, flush=True)


def _basename(value: Any, suffix: str, label: str) -> str:
    if (not isinstance(value, str) or not value or Path(value).name != value
            or value in (".", "..") or not value.endswith(suffix)):
        raise ReferenceError(f"{label} must be a sibling basename ending in {suffix}")
    return value


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ReferenceError(f"{label} symlink is not admitted")
    try:
        resolved = path.resolve(strict=True)
        info = resolved.stat()
    except OSError:
        raise ReferenceError(f"{label} does not exist") from None
    if not resolved.is_file() or not os.path.isfile(resolved) or info.st_nlink <= 0:
        raise ReferenceError(f"{label} must be a regular file")
    return resolved


def _identity(path: Path) -> tuple[int, int, int, int, int]:
    return _stat_identity(path.stat())


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _element_count(shape: Any) -> int:
    if (not isinstance(shape, (list, tuple)) or not shape or len(shape) > 8
            or any(type(value) is not int or value <= 0 for value in shape)):
        raise ReferenceError("invalid tensor shape")
    return math.prod(shape)


MAX_SOURCE_TENSOR_BYTES = max(
    _element_count(shape) * 4 for shape in _BLOCK_RELATIVE_SHAPES.values()
)
MAX_SOURCE_BLOCK_BYTES = sum(
    _element_count(shape) * 4 for shape in _BLOCK_RELATIVE_SHAPES.values()
)


def expected_block_shapes(block_index: int) -> dict[str, list[int]]:
    if type(block_index) is not int or not 0 <= block_index < BLOCK_COUNT:
        raise ReferenceError("block index must be an integer in 0...54")
    return {
        f"double_blocks.{block_index}.{name}": list(shape)
        for name, shape in _BLOCK_RELATIVE_SHAPES.items()
    }


def expected_tensor_shapes() -> dict[str, list[int]]:
    expected = {
        "input.latent": [1, 32, 6, 22, 40],
        "input.qwen_hidden": [1, TEXT_TOKENS, 3584],
        "input.rope_cos": [VIDEO_TOKENS, 128],
        "input.rope_sin": [VIDEO_TOKENS, 128],
        "img_in": [1, VIDEO_TOKENS, 2048],
        "conditioning.projected_text": [1, TEXT_TOKENS, 2048],
        "conditioning.vec": [1, 2048],
        "final.norm": [1, VIDEO_TOKENS, 2048],
        "final.modulated": [1, VIDEO_TOKENS, 2048],
        "final.modulation": [1, 4096],
        "final.linear": [1, VIDEO_TOKENS, 32],
        "output.latent": [1, 32, 6, 22, 40],
    }
    for index in CAPTURE_BLOCKS:
        expected[f"block{index}.img.output"] = [1, VIDEO_TOKENS, 2048]
        expected[f"block{index}.txt.output"] = [1, TEXT_TOKENS, 2048]
    return expected


EXPECTED_TENSOR_SHAPES = expected_tensor_shapes()


def expected_q4_tensor_layout(block_index: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    cursor = 0
    for name, shape in sorted(expected_block_shapes(block_index).items()):
        elements = _element_count(shape)
        if len(shape) == 2:
            if elements % 256:
                raise ReferenceError(f"matrix is not Q4_K aligned: {name}")
            encoding = "q4_k_8range"
            nbytes = (elements // 256) * 144
        elif len(shape) == 1:
            encoding = "bf16-le"
            nbytes = elements * 2
        else:
            raise ReferenceError(f"unsupported source tensor rank: {name}")
        result[name] = {
            "shape": shape,
            "source_dtype": "F32",
            "resident_dtype": "BF16",
            "encoding": encoding,
            "offset_bytes": cursor,
            "nbytes": nbytes,
        }
        cursor += nbytes
    return result


def validate_cache_index_document(document: dict[str, Any]) -> None:
    """Validate cache identities and exact block order without importing torch."""
    if (type(document) is not dict or document.get("schema") != CACHE_INDEX_SCHEMA
            or type(document.get("schema_version")) is not int
            or document.get("schema_version") != 1
            or set(document) != {"schema", "schema_version", "source", "blocks", "globals"}):
        raise ReferenceError("cache index schema or field inventory mismatch")
    source = document.get("source")
    if (type(source) is not dict or set(source) != {"sha256", "bytes"}
            or source.get("sha256") != SOURCE_SHA256
            or type(source.get("bytes")) is not int or source.get("bytes") != SOURCE_BYTES):
        raise ReferenceError("cache index source checkpoint pin mismatch")
    blocks = document.get("blocks")
    if type(blocks) is not list or len(blocks) != BLOCK_COUNT:
        raise ReferenceError("cache index must contain all 54 ordered blocks")
    seen_names: set[str] = set()
    seen_hashes: dict[str, set[str]] = {
        field: set() for field in ("manifest_sha256", "payload_sha256", "bf16_sha256")
    }
    for index, entry in enumerate(blocks):
        if (type(entry) is not dict or set(entry) != {
                "block_index", "manifest", "manifest_sha256", "payload_sha256",
                "bf16", "bf16_sha256"}):
            raise ReferenceError(f"cache index block {index} field inventory mismatch")
        if type(entry.get("block_index")) is not int or entry["block_index"] != index:
            raise ReferenceError("cache index block order/index mismatch")
        manifest = _basename(entry.get("manifest"), ".json", f"block {index} manifest")
        bf16 = _basename(entry.get("bf16"), ".safetensors", f"block {index} BF16 file")
        payload = f"{Path(manifest).stem}.bin"
        for name in (manifest, bf16, payload):
            if name in seen_names:
                raise ReferenceError("cache index contains aliased block files")
            seen_names.add(name)
        for field in seen_hashes:
            digest = _digest(entry.get(field), f"block {index} {field}")
            if digest in seen_hashes[field]:
                raise ReferenceError(f"cache index contains duplicate {field}")
            seen_hashes[field].add(digest)
    globals_doc = document.get("globals")
    if (type(globals_doc) is not dict or set(globals_doc) != {"bf16", "sha256"}):
        raise ReferenceError("cache index global BF16 record mismatch")
    global_name = _basename(globals_doc.get("bf16"), ".safetensors", "global BF16 file")
    if global_name in seen_names:
        raise ReferenceError("global BF16 cache aliases a block file")
    _digest(globals_doc.get("sha256"), "global BF16 cache digest")


def validate_q4_manifest(manifest: dict[str, Any], block_entry: dict[str, Any],
                         block_index: int) -> None:
    """Validate one exact native Q4 block manifest, including offset layout."""
    expected_top = {
        "schema", "schema_version", "source", "block_index", "geometry",
        "codec", "payload", "tensors",
    }
    if type(manifest) is not dict or set(manifest) != expected_top:
        raise ReferenceError(f"block {block_index} Q4 manifest fields are not exact")
    if (manifest.get("schema") != CACHE_MANIFEST_SCHEMA
            or type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != 1
            or type(manifest.get("block_index")) is not int
            or manifest.get("block_index") != block_index):
        raise ReferenceError(f"block {block_index} Q4 manifest schema/index mismatch")
    source = manifest.get("source")
    if (type(source) is not dict or set(source) != {"sha256", "bytes"}
            or source.get("sha256") != SOURCE_SHA256
            or type(source.get("bytes")) is not int or source.get("bytes") != SOURCE_BYTES):
        raise ReferenceError(f"block {block_index} Q4 source pin mismatch")
    if not _same_json_types(_Q4_GEOMETRY, manifest.get("geometry")):
        raise ReferenceError(f"block {block_index} Q4 geometry mismatch")
    expected_codec = {
        "matrix": "q4_k", "ranges": 8, "block_elements": 256,
        "block_bytes": 144, "vector": "bf16-le",
    }
    if not _same_json_types(expected_codec, manifest.get("codec")):
        raise ReferenceError(f"block {block_index} Q4 codec mismatch")
    payload = manifest.get("payload")
    if (type(payload) is not dict or set(payload) != {"basename", "nbytes", "sha256"}
            or not isinstance(payload.get("basename"), str)
            or type(payload.get("nbytes")) is not int or payload["nbytes"] <= 0
            or _digest(payload.get("sha256"), f"block {block_index} payload")
            != block_entry.get("payload_sha256")):
        raise ReferenceError(f"block {block_index} Q4 payload descriptor mismatch")
    manifest_name = _basename(block_entry.get("manifest"), ".json", "manifest")
    if payload["basename"] != f"{Path(manifest_name).stem}.bin":
        raise ReferenceError(f"block {block_index} payload sibling basename mismatch")

    expected = expected_q4_tensor_layout(block_index)
    tensors = manifest.get("tensors")
    if type(tensors) is not dict or set(tensors) != set(expected):
        raise ReferenceError(f"block {block_index} Q4 tensor inventory mismatch")
    cursor = 0
    for name in sorted(expected):
        desc = tensors.get(name)
        wanted = expected[name]
        if (type(desc) is not dict or set(desc) != {
                "shape", "source_dtype", "resident_dtype", "encoding", "offset_bytes", "nbytes"}
                or not _same_json_types(wanted["shape"], desc.get("shape"))
                or desc.get("source_dtype") != wanted["source_dtype"]
                or desc.get("resident_dtype") != wanted["resident_dtype"]
                or desc.get("encoding") != wanted["encoding"]
                or type(desc.get("offset_bytes")) is not int
                or desc["offset_bytes"] != cursor
                or type(desc.get("nbytes")) is not int
                or desc["nbytes"] != wanted["nbytes"]):
            raise ReferenceError(f"block {block_index} Q4 tensor layout mismatch: {name}")
        cursor += wanted["nbytes"]
    if cursor != payload["nbytes"]:
        raise ReferenceError(f"block {block_index} Q4 payload has gaps or trailing bytes")


def validate_mask(mask: Any) -> None:
    if (type(mask) is not list or len(mask) != TEXT_TOKENS
            or any(type(value) is not int for value in mask) or mask != MASK):
        raise ReferenceError("Qwen full-row mask must be the pinned 23-valid-row prefix")


def validate_unpatchified_shape(shape: Any) -> None:
    if not _same_json_types([1, 32, 6, 22, 40], shape):
        raise ReferenceError("final unpatchified latent shape mismatch")


def validate_tensor_layout(tensors: dict[str, Any], sidecar_bytes: int) -> None:
    if type(sidecar_bytes) is not int or not 0 < sidecar_bytes <= SIDE_CAR_LIMIT:
        raise ReferenceError("Q4 DiT tensor sidecar size outside 512 MiB bound")
    if type(tensors) is not dict or set(tensors) != set(EXPECTED_TENSOR_SHAPES):
        raise ReferenceError("Q4 DiT tensor inventory mismatch")
    cursor = 0
    for name in sorted(EXPECTED_TENSOR_SHAPES):
        meta = tensors[name]
        shape = EXPECTED_TENSOR_SHAPES[name]
        nbytes = _element_count(shape) * 4
        if (type(meta) is not dict or set(meta) != {"dtype", "shape", "offset_bytes", "nbytes"}
                or meta.get("dtype") != "F32"
                or not _same_json_types(shape, meta.get("shape"))
                or type(meta.get("offset_bytes")) is not int or meta["offset_bytes"] != cursor
                or type(meta.get("nbytes")) is not int or meta["nbytes"] != nbytes):
            raise ReferenceError(f"Q4 DiT tensor range/shape mismatch: {name}")
        if nbytes > sidecar_bytes - cursor:
            raise ReferenceError(f"Q4 DiT tensor range exceeds sidecar: {name}")
        cursor += nbytes
    if cursor != sidecar_bytes:
        raise ReferenceError("Q4 DiT tensor sidecar has trailing or missing bytes")


def _expected_sidecar_bytes() -> int:
    return sum(_element_count(shape) * 4 for shape in EXPECTED_TENSOR_SHAPES.values())


def _validated_weight_source_loading(producer: dict[str, Any], precision_mode: str) -> dict[str, str]:
    if "weight_source_loading" not in producer:
        # Schema 1 predates this optional receipt. Its established behavior was
        # mapped source tensors for BF16, and Q4 payloads for Q4 blocks.
        loading = {
            "block_weights": (
                "q4_cache_payloads" if precision_mode == "q4"
                else "safetensors_mapped_tensor_views"
            ),
            "global_weights": "safetensors_mapped_tensor_views",
        }
    else:
        loading = producer["weight_source_loading"]

    expected_block_loading = (
        "q4_cache_payloads" if precision_mode == "q4"
        else loading.get("block_weights") if type(loading) is dict else None
    )
    if (precision_mode == "bf16"
            and expected_block_loading not in (
                "safetensors_mapped_tensor_views", "owned_raw_f32_extents"
            )):
        raise ReferenceError("Q4 DiT original block source-loading mode mismatch")
    expected_loading = {
        "block_weights": expected_block_loading,
        "global_weights": "safetensors_mapped_tensor_views",
    }
    if not _same_json_types(expected_loading, loading):
        raise ReferenceError("Q4 DiT weight source-loading receipt mismatch")
    return expected_loading


def validate_reference_document(document: dict[str, Any]) -> None:
    """Validate an emitted fixture's source, geometry, precision, and captures."""
    expected_fields = {
        "schema", "schema_version", "precision_mode", "upstream", "source",
        "qwen_source", "qwen_reference", "qwen_sidecar", "geometry",
        "input_generation", "precision", "mask", "timestep", "conditioning",
        "padding_policy", "tolerance", "producer", "tensor_file", "tensors",
    }
    if (type(document) is not dict or set(document) != expected_fields
            or document.get("schema") != SCHEMA
            or type(document.get("schema_version")) is not int
            or document.get("schema_version") != SCHEMA_VERSION):
        raise ReferenceError("missing or invalid Q4 DiT reference schema")
    if document.get("precision_mode") not in ("q4", "bf16"):
        raise ReferenceError("unsupported Q4 DiT precision mode")
    if not _same_json_types(GEOMETRY, document.get("geometry")):
        raise ReferenceError("Q4 DiT geometry mismatch")
    if not _same_json_types(INPUT_GENERATION, document.get("input_generation")):
        raise ReferenceError("Q4 DiT input/RoPE generation mismatch")
    if not _same_json_types(MASK, document.get("mask")):
        raise ReferenceError("Qwen attention mask mismatch")
    upstream = document.get("upstream")
    if (type(upstream) is not dict or set(upstream) != {"revision", "source_sha256"}
            or upstream.get("revision") != PINNED_REVISION
            or not _same_json_types(
                conditioned_reference.PINNED_UPSTREAM_SOURCE_SHA256,
                upstream.get("source_sha256"),
            )):
        raise ReferenceError("Q4 DiT pinned upstream inventory mismatch")
    source = document.get("source")
    if (type(source) is not dict or set(source) != {
            "checkpoint_path", "checkpoint_sha256", "checkpoint_bytes",
            "checkpoint_header_sha256", "cache_index", "cache_block_order",
            "cache_blocks", "precision_source",
    } or source.get("checkpoint_sha256") != SOURCE_SHA256
            or type(source.get("checkpoint_path")) is not str
            or type(source.get("checkpoint_bytes")) is not int
            or source.get("checkpoint_bytes") != SOURCE_BYTES
            or source.get("checkpoint_header_sha256") != SOURCE_HEADER_SHA256):
        raise ReferenceError("Q4 DiT source checkpoint pin mismatch")
    cache_index = source.get("cache_index")
    if (type(cache_index) is not dict or type(cache_index.get("path")) is not str
            or type(cache_index.get("sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", cache_index["sha256"]) is None):
        raise ReferenceError("Q4 DiT cache-index source record mismatch")
    if not _same_json_types(list(range(BLOCK_COUNT)), source.get("cache_block_order")):
        raise ReferenceError("Q4 DiT cache block order mismatch")
    cache_blocks = source.get("cache_blocks")
    if type(cache_blocks) is not list or len(cache_blocks) != BLOCK_COUNT:
        raise ReferenceError("Q4 DiT cache block inventory mismatch")
    seen_cache_names: set[str] = set()
    for index, entry in enumerate(cache_blocks):
        if type(entry) is not dict or set(entry) != {
                "block_index", "manifest", "manifest_sha256", "payload_sha256",
                "bf16", "bf16_sha256"}:
            raise ReferenceError(f"Q4 DiT cache block {index} record mismatch")
        if type(entry.get("block_index")) is not int or entry["block_index"] != index:
            raise ReferenceError("Q4 DiT cache block index/order mismatch")
        names = (
            _basename(entry.get("manifest"), ".json", "cache manifest"),
            _basename(entry.get("bf16"), ".safetensors", "cache BF16"),
            f"{Path(entry['manifest']).stem}.bin",
        )
        if any(name in seen_cache_names for name in names):
            raise ReferenceError("Q4 DiT cache block files alias")
        seen_cache_names.update(names)
        for field in ("manifest_sha256", "payload_sha256", "bf16_sha256"):
            _digest(entry.get(field), f"cache block {index} {field}")
    if source["precision_source"] != (
            "original full F32 checkpoint" if document["precision_mode"] == "bf16"
            else "pinned Q4_K block payloads plus original global F32 checkpoint"):
        raise ReferenceError("Q4 DiT precision source description mismatch")
    precision = dict(PRECISION)
    precision["mode"] = document["precision_mode"]
    if not _same_json_types(precision, document.get("precision")):
        raise ReferenceError("Q4 DiT precision description mismatch")
    if not _same_json_types(text_reference.QWEN_SOURCE, document.get("qwen_source")):
        raise ReferenceError("Q4 DiT Qwen source pin mismatch")
    qwen_reference = document.get("qwen_reference")
    if (type(qwen_reference) is not dict or set(qwen_reference) != {"path", "sha256"}
            or type(qwen_reference.get("path")) is not str
            or qwen_reference.get("sha256") != text_reference.QWEN_REFERENCE_SHA256):
        raise ReferenceError("Q4 DiT Qwen reference identity mismatch")
    qwen_sidecar = document.get("qwen_sidecar")
    if (type(qwen_sidecar) is not dict
            or set(qwen_sidecar) != {"path", "sha256", "nbytes"}
            or type(qwen_sidecar.get("path")) is not str
            or qwen_sidecar.get("sha256") != text_reference.QWEN_SIDECAR_SHA256
            or type(qwen_sidecar.get("nbytes")) is not int
            or qwen_sidecar.get("nbytes") != 268_627_968):
        raise ReferenceError("Q4 DiT Qwen sidecar identity mismatch")
    if not _same_json_types(500.0, document.get("timestep")):
        raise ReferenceError("Q4 DiT timestep mismatch")
    if not _same_json_types(_CONDITIONING, document.get("conditioning")):
        raise ReferenceError("Q4 DiT conditioning crop/valid-prefix mismatch")
    if not _same_json_types(_PADDING_POLICY, document.get("padding_policy")):
        raise ReferenceError("Q4 DiT padded-query policy mismatch")
    if not _same_json_types({"atol": ATOL, "rtol": RTOL}, document.get("tolerance")):
        raise ReferenceError("Q4 DiT numerical tolerance mismatch")
    producer = document.get("producer")
    producer_fields_without_loading = {
        "path", "source_sha256", "device", "activation_readback",
        "terminal_activation_readbacks", "attention_adapter",
        "attention_adapter_calls", "query_chunks_per_block",
        "attention_query_chunk_calls", "padding_poison_control_chunks",
        "padding_poison_control_passed", "block_invocations",
        "captured_block_outputs", "block_handoff", "final_layer",
        "camera_conditioning", "action_conditioning", "full_generation",
    }
    if (type(producer) is not dict
            or set(producer) not in (
                producer_fields_without_loading,
                producer_fields_without_loading | {"weight_source_loading"},
            )
            or type(producer.get("path")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", producer.get("source_sha256", "")) is None
            or producer.get("device") != "mps"
            or not _same_json_types(1, producer.get("terminal_activation_readbacks"))
            or not _same_json_types(list(range(BLOCK_COUNT)), producer.get("block_invocations"))
            or not _same_json_types(BLOCK_COUNT, producer.get("attention_adapter_calls"))
            or not _same_json_types(ATTENTION_CHUNKS_PER_BLOCK,
                                    producer.get("query_chunks_per_block"))
            or not _same_json_types(
                BLOCK_COUNT * ATTENTION_CHUNKS_PER_BLOCK,
                producer.get("attention_query_chunk_calls"),
            )
            or not _same_json_types(
                ATTENTION_CHUNKS_PER_BLOCK, producer.get("padding_poison_control_chunks")
            )
            or producer.get("padding_poison_control_passed") is not True):
        raise ReferenceError("Q4 DiT producer execution receipt mismatch")
    if (not _same_json_types(list(CAPTURE_BLOCKS), producer.get("captured_block_outputs"))
            or producer.get("activation_readback")
            != "one packed terminal F32 tensor sidecar readback"
            or producer.get("attention_adapter")
            != "PyTorch MPS SDPA F32, valid joint prefix only"
            or producer.get("block_handoff") != "img/text outputs passed directly to the next block"
            or producer.get("final_layer")
            != "pinned FinalLayerPerBlock forward plus pinned pointwise unpatchify layout"
            or producer.get("camera_conditioning") != "off"
            or producer.get("action_conditioning") != "off"
            or producer.get("full_generation") is not False):
        raise ReferenceError("Q4 DiT producer scope/capture receipt mismatch")
    _validated_weight_source_loading(producer, document["precision_mode"])
    tensor_file = document.get("tensor_file")
    if (type(tensor_file) is not dict or set(tensor_file) != {"name", "nbytes", "sha256", "encoding"}
            or _basename(tensor_file.get("name"), ".tensors.f32le", "tensor sidecar")
            != tensor_file.get("name")
            or type(tensor_file.get("nbytes")) is not int
            or tensor_file["nbytes"] != _expected_sidecar_bytes()
            or tensor_file.get("encoding") != "f32-le"):
        raise ReferenceError("Q4 DiT tensor-file metadata mismatch")
    _digest(tensor_file.get("sha256"), "Q4 DiT tensor sidecar")
    validate_tensor_layout(document.get("tensors"), tensor_file["nbytes"])


def _validate_source_checkpoint(
    path_arg: Path, supplied_sha: str
) -> tuple[Path, str, tuple[int, int, int, int, int],
           dict[str, tuple[str, tuple[int, ...], int, int]]]:
    expected_sha = _digest(supplied_sha, "full checkpoint SHA-256")
    if expected_sha != SOURCE_SHA256:
        raise ReferenceError("full checkpoint digest is not the pinned source")
    path = _regular_file(path_arg, "full checkpoint")
    identity = _identity(path)
    if identity[2] != SOURCE_BYTES:
        raise ReferenceError("full checkpoint byte count mismatch")
    actual_sha = sha256_file(path)
    if actual_sha != SOURCE_SHA256 or _identity(path) != identity:
        raise ReferenceError("full checkpoint bytes differ from the pinned source")
    with path.open("rb") as stream:
        length = stream.read(8)
        if len(length) != 8 or struct.unpack("<Q", length)[0] != checkpoint_reference.PINNED.header_bytes:
            raise ReferenceError("full checkpoint safetensors header length mismatch")
        header = stream.read(checkpoint_reference.PINNED.header_bytes)
    try:
        checkpoint_reference.validate_checkpoint_header(header, identity[2])
    except checkpoint_reference.DownloadError as exc:
        raise ReferenceError(f"full checkpoint header admission failed: {exc}") from None
    if _identity(path) != identity:
        raise ReferenceError("full checkpoint changed while reading its authenticated header")

    # Keep only the authenticated named extents needed by the block-source
    # reader. Safetensors offsets are relative to the start of its data section.
    try:
        header_document = checkpoint_reference._json_object(
            header, label="authenticated safetensors header"
        )
    except checkpoint_reference.DownloadError as exc:
        raise ReferenceError(f"full checkpoint header admission failed: {exc}") from None
    header_document.pop("__metadata__", None)
    source_layout: dict[str, tuple[str, tuple[int, ...], int, int]] = {}
    for name, descriptor in header_document.items():
        if name.startswith("double_blocks."):
            source_layout[name] = (
                descriptor["dtype"], tuple(descriptor["shape"]),
                descriptor["data_offsets"][0], descriptor["data_offsets"][1],
            )
    if _identity(path) != identity:
        raise ReferenceError("full checkpoint changed while retaining authenticated extents")
    return path, actual_sha, identity, source_layout


def _validate_cache_index(path_arg: Path, supplied_sha: str,
                          *, precision: str) -> tuple[Path, str, dict[str, Any], dict[int, dict[str, Any]]]:
    expected_sha = _digest(supplied_sha, "Q4 cache-index SHA-256")
    path = _regular_file(path_arg, "Q4 cache index")
    if path.stat().st_size <= 0 or path.stat().st_size > JSON_LIMIT:
        raise ReferenceError("Q4 cache index exceeds its size bound")
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise ReferenceError("Q4 cache-index SHA-256 mismatch")
    document = _json_object(raw, "Q4 cache index")
    validate_cache_index_document(document)
    admitted: dict[int, dict[str, Any]] = {}
    if precision == "q4":
        for index, entry in enumerate(document["blocks"]):
            manifest_path = path.parent / entry["manifest"]
            payload_name = f"{Path(entry['manifest']).stem}.bin"
            manifest_path = _regular_file(manifest_path, f"block {index} Q4 manifest")
            if manifest_path.stat().st_size <= 0 or manifest_path.stat().st_size > MAX_CACHE_MANIFEST_BYTES:
                raise ReferenceError(f"block {index} Q4 manifest exceeds its size bound")
            manifest_raw = manifest_path.read_bytes()
            manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
            if manifest_sha != entry["manifest_sha256"]:
                raise ReferenceError(f"block {index} Q4 manifest hash mismatch")
            manifest = _json_object(manifest_raw, f"block {index} Q4 manifest")
            validate_q4_manifest(manifest, entry, index)
            payload_path = _regular_file(path.parent / payload_name, f"block {index} Q4 payload")
            if payload_path.stat().st_size != manifest["payload"]["nbytes"]:
                raise ReferenceError(f"block {index} Q4 payload byte count mismatch")
            payload_sha = sha256_file(payload_path)
            if payload_sha != entry["payload_sha256"]:
                raise ReferenceError(f"block {index} Q4 payload SHA-256 mismatch")
            admitted[index] = {
                "entry": entry, "manifest": manifest,
                "manifest_path": manifest_path, "payload_path": payload_path,
            }
    return path, actual_sha, document, admitted


def _validate_qwen_source(reference_arg: Path, sidecar_arg: Path
                          ) -> tuple[dict[str, Any], Path, str, str,
                                     tuple[int, int, int, int, int],
                                     tuple[int, int, int, int, int]]:
    reference_path = _regular_file(reference_arg, "Qwen reference")
    sidecar_path = _regular_file(sidecar_arg, "Qwen sidecar")
    if reference_path.stat().st_size <= 0 or reference_path.stat().st_size > JSON_LIMIT:
        raise ReferenceError("Qwen reference exceeds its size bound")
    reference_identity = _identity(reference_path)
    reference_raw = reference_path.read_bytes()
    if _identity(reference_path) != reference_identity:
        raise ReferenceError("Qwen reference changed during byte admission")
    reference_sha = hashlib.sha256(reference_raw).hexdigest()
    sidecar_identity = _identity(sidecar_path)
    if sidecar_identity[2] != 268_627_968:
        raise ReferenceError("Qwen BF16 sidecar byte count mismatch")
    sidecar_sha = sha256_file(sidecar_path)
    if _identity(sidecar_path) != sidecar_identity:
        raise ReferenceError("Qwen BF16 sidecar changed during byte admission")
    if reference_sha != text_reference.QWEN_REFERENCE_SHA256:
        raise ReferenceError("Qwen reference is not the authenticated saved fixture")
    if sidecar_sha != text_reference.QWEN_SIDECAR_SHA256:
        raise ReferenceError("Qwen BF16 sidecar digest mismatch")
    try:
        document = _json_object(reference_raw, "Qwen reference")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError,
            ReferenceError):
        raise ReferenceError("Qwen reference is not bounded valid JSON") from None
    try:
        valid_indices = text_reference.validate_qwen_reference(
            document, reference_sha256=reference_sha,
            sidecar_bytes=sidecar_path.stat().st_size,
            sidecar_sha256=sidecar_sha,
        )
    except text_reference.ReferenceError as exc:
        raise ReferenceError(f"Qwen source validation failed: {exc}") from None
    if not valid_indices or valid_indices != list(range(VALID_TEXT_TOKENS)):
        raise ReferenceError("Qwen crop is not the exact valid 23-token prefix")
    return (document, sidecar_path, reference_sha, sidecar_sha,
            reference_identity, sidecar_identity)


def _module_state_shapes(module: Any) -> dict[str, list[int]]:
    return {name: list(value.shape) for name, value in module.state_dict().items()}


def _load_module_from_full_source(torch: Any, source: Any, module: Any,
                                  prefix: str, device: Any) -> Any:
    expected = _module_state_shapes(module)
    expected_source = {f"{prefix}{name}": (name, shape) for name, shape in expected.items()}
    actual_names = {name for name in source.keys() if name.startswith(prefix)}
    if actual_names != set(expected_source):
        raise ReferenceError(f"full checkpoint {prefix} tensor inventory mismatch")
    state: dict[str, Any] = {}
    for source_name, (target_name, shape) in expected_source.items():
        value = source.get_tensor(source_name)
        if value.dtype != torch.float32 or list(value.shape) != shape:
            raise ReferenceError(f"full checkpoint source tensor shape/dtype mismatch: {source_name}")
        if not bool(torch.isfinite(value).all().item()):
            raise ReferenceError(f"full checkpoint source tensor is non-finite: {source_name}")
        state[target_name] = value
    try:
        module.load_state_dict(state, strict=True, assign=True)
    except (RuntimeError, TypeError) as exc:
        raise ReferenceError(f"could not bind full checkpoint state {prefix}: {exc}") from None
    module.to(device=device, dtype=torch.float32).eval()
    del state
    block_reference._round_parameters_bf16_in_place(torch, module, device)
    return module


def _global_modules(torch: Any, upstream: Path, helpers: tuple[Any, ...]) -> dict[str, Any]:
    import importlib

    block_reference._namespace("hymm.models.text_encoder", upstream / "hymm/models/text_encoder")
    text_module, _ = text_reference._install_text_attention(
        torch, valid_indices=list(range(VALID_TEXT_TOKENS)), precision="f32", capture={},
    )
    del text_module
    refiner_module = importlib.import_module("hymm.models.text_encoder.token_refiner")
    embed_layers = importlib.import_module("hymm.models.dit.modules.embed_layers")
    mlp_layers = sys.modules["hymm.models.dit.modules.mlp_layers"]
    meta = torch.device("meta")
    factory_dtype = torch.float32
    modules = {
        "txt_in": refiner_module.SingleTokenRefinerPerBlock(
            in_channels=3584, hidden_size=2048, heads_num=16, depth=2,
            mlp_width_ratio=4.0, mlp_drop_rate=0.0, act_type="silu",
            qk_norm=False, qkv_bias=True, attn_mode="torch", seg_pos_emb_num=0,
            dtype=factory_dtype, device=meta,
        ),
        "time_in": embed_layers.TimestepEmbedder(
            2048, torch.nn.SiLU, dtype=factory_dtype, device=meta,
        ),
        "img_in": embed_layers.PatchEmbed(
            patch_size=(1, 1, 1), in_chans=32, embed_dim=2048,
            dtype=factory_dtype, device=meta,
        ),
        "final_layer": mlp_layers.FinalLayerPerBlock(
            2048, 1, 32, helpers[0]("silu"), device=meta, dtype=factory_dtype,
        ),
    }
    for prefix, module in modules.items():
        if len(module.state_dict()) != _GLOBAL_EXPECTED_COUNTS[f"{prefix}."]:
            raise ReferenceError(f"upstream global module inventory mismatch: {prefix}")
    return modules


def _build_global_modules(torch: Any, source: Any, upstream: Path,
                          helpers: tuple[Any, ...], device: Any) -> dict[str, Any]:
    modules = _global_modules(torch, upstream, helpers)
    loaded: dict[str, Any] = {}
    for prefix, module in modules.items():
        loaded[prefix] = _load_module_from_full_source(
            torch, source, module, f"{prefix}.", device,
        )
    return loaded


def _deterministic_latent(torch: Any, device: Any) -> Any:
    shape = [1, 32, 6, 22, 40]
    count = math.prod(shape)
    values = torch.arange(count, dtype=torch.float32, device=device)
    values.mul_(0.017).add_(0.11).sin_().mul_(0.1)
    cosines = torch.arange(count, dtype=torch.float32, device=device)
    cosines.mul_(0.0031).add_(0.077).cos_().mul_(0.05)
    values.add_(cosines)
    del cosines
    return values.reshape(shape)


def _read_q4_payload_state(torch: Any, np: Any, admitted: dict[str, Any],
                           block_index: int, device: Any) -> dict[str, Any]:
    manifest = admitted["manifest"]
    entry = admitted["entry"]
    payload_path = admitted["payload_path"]
    raw = payload_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry["payload_sha256"] or len(raw) != manifest["payload"]["nbytes"]:
        raise ReferenceError(f"block {block_index} Q4 payload changed after admission")
    state: dict[str, Any] = {}
    for name in sorted(manifest["tensors"]):
        descriptor = manifest["tensors"][name]
        start = descriptor["offset_bytes"]
        stop = start + descriptor["nbytes"]
        tensor_bytes = raw[start:stop]
        shape = descriptor["shape"]
        if descriptor["encoding"] == "bf16-le":
            value = _decode_bf16_le_mps(torch, np, tensor_bytes, device)
        else:
            value = _decode_q4_k_mps(torch, np, tensor_bytes, shape, device)
        if list(value.shape) != shape or value.device.type != "mps" or value.dtype != torch.float32:
            raise ReferenceError(f"block {block_index} decoded Q4 tensor mismatch: {name}")
        state[name.removeprefix(f"double_blocks.{block_index}.")] = value
    if set(state) != set(_BLOCK_RELATIVE_SHAPES):
        raise ReferenceError(f"block {block_index} decoded Q4 state inventory mismatch")
    del raw
    return state


def _decode_q4_k_mps(torch: Any, np: Any, payload: bytes,
                     shape: list[int], device: Any) -> Any:
    elements = _element_count(shape)
    if elements % 256 or len(payload) != (elements // 256) * 144:
        raise ReferenceError("Q4_K payload extent does not match its matrix shape")
    # Upload the authenticated bytes unchanged. All unpacking and numeric
    # reconstruction below runs on MPS so this oracle does not decode weights
    # numerically on the CPU or reuse the native Metal decoder.
    raw_u8 = np.frombuffer(payload, dtype=np.uint8).copy()
    packed = torch.from_numpy(raw_u8).to(device=device)
    blocks = packed.reshape(-1, 144).to(dtype=torch.int32)
    block_count = blocks.shape[0]

    header = blocks[:, :4]
    d_word = header[:, 0] | (header[:, 1] << 8)
    dmin_word = header[:, 2] | (header[:, 3] << 8)
    half_words = torch.stack((d_word, dmin_word), dim=1).to(dtype=torch.int16)
    half_values = half_words.view(torch.float16).to(dtype=torch.float32)

    packed_scales = blocks[:, 4:16]
    scales = []
    mins = []
    for index in range(8):
        if index < 4:
            scales.append(packed_scales[:, index] & 0x3F)
            mins.append(packed_scales[:, index + 4] & 0x3F)
        else:
            scales.append((packed_scales[:, index + 4] & 0x0F) | (
                (packed_scales[:, index - 4] >> 6) << 4
            ))
            mins.append((packed_scales[:, index + 4] >> 4) | (
                (packed_scales[:, index] >> 6) << 4
            ))
    scale_mps = torch.stack(scales, dim=1).to(dtype=torch.float32)
    min_mps = torch.stack(mins, dim=1).to(dtype=torch.float32)

    packed_quants = blocks[:, 16:].reshape(block_count, 4, 32)
    codes = torch.cat((packed_quants & 0x0F, packed_quants >> 4), dim=2)
    code_mps = codes.reshape(block_count, 256).to(dtype=torch.float32)
    d = half_values[:, 0]
    dmin = half_values[:, 1]
    q = code_mps.reshape(block_count, 8, 32)
    decoded = (d[:, None, None] * scale_mps[:, :, None]) * q - (
        dmin[:, None, None] * min_mps[:, :, None]
    )
    return decoded.reshape(shape).to(dtype=torch.bfloat16).to(dtype=torch.float32)


def _decode_bf16_le_mps(torch: Any, np: Any, payload: bytes, device: Any) -> Any:
    if not payload or len(payload) % 2:
        raise ReferenceError("BF16 payload must contain complete little-endian words")
    raw_u8 = np.frombuffer(payload, dtype=np.uint8).copy()
    uploaded = torch.from_numpy(raw_u8).to(device=device)
    return uploaded.view(torch.int16).view(torch.bfloat16).to(dtype=torch.float32)


def _q4_decoder_probe_fixture() -> tuple[bytes, list[float], bytes, list[float]]:
    """Construct one hand-checkable Q4_K block and a four-word BF16 vector."""
    scales = [1, 2, 3, 4, 17, 18, 19, 20]
    mins = [10, 12, 14, 16, 18, 20, 22, 24]
    packed = bytearray(144)
    packed[0:4] = bytes((0x00, 0x30, 0x00, 0x30))  # d=dmin=0.125
    for index in range(4):
        packed[4 + index] = scales[index] | ((scales[index + 4] >> 4) << 6)
        packed[8 + index] = mins[index] | ((mins[index + 4] >> 4) << 6)
    for index in range(4, 8):
        packed[8 + index] = (scales[index] & 0x0F) | ((mins[index] & 0x0F) << 4)

    expected_q4: list[float] = []
    for group in range(4):
        for half in range(2):
            scale_index = group * 2 + half
            for column in range(32):
                code = (2 * (column + group + (4 if half else 0))) % 16
                expected_q4.append(
                    float(0.125 * scales[scale_index] * code - 0.125 * mins[scale_index])
                )
        for column in range(32):
            low = (2 * (column + group)) % 16
            high = (2 * (column + group + 4)) % 16
            packed[16 + group * 32 + column] = low | (high << 4)

    vector_bits = (0x3F80, 0xC000, 0x3F00, 0x4050)  # 1.0, -2.0, 0.5, 3.25
    vector = b"".join(struct.pack("<H", bits) for bits in vector_bits)
    return bytes(packed), expected_q4, vector, [1.0, -2.0, 0.5, 3.25]


def run_q4_decoder_probe(torch: Any, np: Any) -> dict[str, Any]:
    """Run the isolated decoder discriminator; this is not a model run."""
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ReferenceError("MPS is required for the Q4 decoder probe")
    device = torch.device("mps")
    q4_payload, expected_q4, vector_payload, expected_vector = _q4_decoder_probe_fixture()
    decoded_q4 = _decode_q4_k_mps(torch, np, q4_payload, [1, 256], device)
    decoded_vector = _decode_bf16_le_mps(torch, np, vector_payload, device)
    packed = torch.cat((decoded_q4.reshape(-1), decoded_vector.reshape(-1))).to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    actual = packed.numpy()
    expected = np.asarray(expected_q4 + expected_vector, dtype=np.float32)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        mismatch = np.flatnonzero(actual != expected)
        first = int(mismatch[0]) if mismatch.size else -1
        raise ReferenceError(f"MPS Q4/BF16 decoder probe mismatch at packed value {first}")
    return {
        "status": "PASS",
        "device": "mps",
        "q4_k_values": len(expected_q4),
        "bf16_values": len(expected_vector),
        "terminal_readbacks": 1,
        "expected_exact": True,
    }


def _load_source_block_state(torch: Any, source: Any, block_index: int) -> dict[str, Any]:
    expected = expected_block_shapes(block_index)
    keys = set(source.keys())
    prefix = f"double_blocks.{block_index}."
    names = {name for name in keys if name.startswith(prefix)}
    if names != set(expected):
        raise ReferenceError(f"full checkpoint block {block_index} tensor inventory mismatch")
    state: dict[str, Any] = {}
    for source_name, shape in expected.items():
        value = source.get_tensor(source_name)
        if value.dtype != torch.float32 or list(value.shape) != shape:
            raise ReferenceError(f"full checkpoint block tensor shape/dtype mismatch: {source_name}")
        if not bool(torch.isfinite(value).all().item()):
            raise ReferenceError(f"full checkpoint block tensor is non-finite: {source_name}")
        state[source_name.removeprefix(prefix)] = value
    return state


def _validated_source_block_extents(
    source_layout: dict[str, tuple[str, tuple[int, ...], int, int]], block_index: int
) -> list[tuple[str, tuple[int, ...], int, int]]:
    """Validate exact named F32 source extents before reading any block bytes."""
    expected = expected_block_shapes(block_index)
    prefix = f"double_blocks.{block_index}."
    names = {name for name in source_layout if name.startswith(prefix)}
    if names != set(expected):
        raise ReferenceError(f"full checkpoint block {block_index} tensor inventory mismatch")

    selected: list[tuple[str, tuple[int, ...], int, int]] = []
    total_bytes = 0
    data_bytes = SOURCE_BYTES - SOURCE_DATA_OFFSET
    for name, shape_list in expected.items():
        descriptor = source_layout.get(name)
        if type(descriptor) is not tuple or len(descriptor) != 4:
            raise ReferenceError(f"full checkpoint block source descriptor mismatch: {name}")
        dtype, shape, start, end = descriptor
        shape = tuple(shape)
        expected_shape = tuple(shape_list)
        tensor_bytes = _element_count(expected_shape) * 4
        if (dtype != "F32" or shape != expected_shape
                or type(start) is not int or type(end) is not int
                or start < 0 or end <= start or end - start != tensor_bytes
                or tensor_bytes > MAX_SOURCE_TENSOR_BYTES or end > data_bytes):
            raise ReferenceError(f"full checkpoint block tensor extent mismatch: {name}")
        total_bytes += tensor_bytes
        selected.append((name, expected_shape, start, tensor_bytes))

    if total_bytes != MAX_SOURCE_BLOCK_BYTES or total_bytes > data_bytes:
        raise ReferenceError(f"full checkpoint block {block_index} exceeds its source byte bound")
    selected.sort(key=lambda item: (item[2], item[0]))
    previous_end = -1
    for name, _shape, start, tensor_bytes in selected:
        if start < previous_end:
            raise ReferenceError(f"full checkpoint block source extents overlap: {name}")
        previous_end = start + tensor_bytes
    return selected


def _read_exact_into(stream: Any, byte_count: int, tensor_name: str) -> bytearray:
    if type(byte_count) is not int or not 0 < byte_count <= MAX_SOURCE_TENSOR_BYTES:
        raise ReferenceError(f"full checkpoint tensor exceeds its owned-buffer bound: {tensor_name}")
    buffer = bytearray(byte_count)
    view = memoryview(buffer)
    cursor = 0
    try:
        while cursor < byte_count:
            count = stream.readinto(view[cursor:])
            if count is None or count <= 0:
                raise ReferenceError(f"short read for full checkpoint tensor: {tensor_name}")
            cursor += count
    finally:
        view.release()
    if cursor != byte_count:
        raise ReferenceError(f"short read for full checkpoint tensor: {tensor_name}")
    return buffer


def _assert_source_file_identity(stream: Any, path: Path,
                                 expected_identity: tuple[int, int, int, int, int]) -> None:
    try:
        descriptor_identity = _stat_identity(os.fstat(stream.fileno()))
        path_identity = _identity(path)
    except OSError:
        raise ReferenceError("full checkpoint became unavailable during owned read") from None
    if descriptor_identity != expected_identity or path_identity != expected_identity:
        raise ReferenceError("full checkpoint identity changed during owned read")


def _read_owned_source_block_buffers(
    path: Path,
    expected_identity: tuple[int, int, int, int, int],
    source_layout: dict[str, tuple[str, tuple[int, ...], int, int]],
    block_index: int,
) -> dict[str, bytearray]:
    """Copy one authenticated block's exact original F32 bytes into owned CPU buffers."""
    extents = _validated_source_block_extents(source_layout, block_index)
    buffers: dict[str, bytearray] = {}
    try:
        with path.open("rb", buffering=0) as stream:
            _assert_source_file_identity(stream, path, expected_identity)
            for name, _shape, data_offset, tensor_bytes in extents:
                stream.seek(SOURCE_DATA_OFFSET + data_offset)
                buffers[name] = _read_exact_into(stream, tensor_bytes, name)
            _assert_source_file_identity(stream, path, expected_identity)
    except OSError:
        raise ReferenceError("could not read owned full checkpoint block bytes") from None
    return buffers


def _load_owned_source_block_state(
    torch: Any,
    path: Path,
    expected_identity: tuple[int, int, int, int, int],
    source_layout: dict[str, tuple[str, tuple[int, ...], int, int]],
    block_index: int,
) -> dict[str, Any]:
    """Build CPU F32 tensors over owned raw bytes, with no CPU value conversion."""
    if sys.byteorder != "little":
        raise ReferenceError("owned source loading requires a little-endian host")
    prefix = f"double_blocks.{block_index}."
    buffers = _read_owned_source_block_buffers(
        path, expected_identity, source_layout, block_index
    )
    state: dict[str, Any] = {}
    for name, shape, _offset, tensor_bytes in _validated_source_block_extents(
        source_layout, block_index
    ):
        buffer = buffers[name]
        if len(buffer) != tensor_bytes:
            raise ReferenceError(f"owned source tensor byte count mismatch: {name}")
        value = torch.frombuffer(
            buffer, dtype=torch.float32, count=_element_count(shape)
        ).reshape(shape)
        if (value.dtype != torch.float32 or list(value.shape) != list(shape)
                or not bool(torch.isfinite(value).all().item())):
            raise ReferenceError(f"full checkpoint source tensor is invalid: {name}")
        state[name.removeprefix(prefix)] = value
    return state


def _build_block(torch: Any, helpers: tuple[Any, ...],
                 block_type: Any, state: dict[str, Any], device: Any,
                 *, round_bf16: bool) -> Any:
    block = block_type(
        hidden_size=2048, heads_num=16, mlp_width_ratio=4.0,
        mlp_act_type="gelu_tanh", attn_mode="magi_mp_causal",
        qk_norm=True, qk_norm_type="rms", qkv_bias=True,
        dtype=torch.float32, device=torch.device("meta"),
    )
    try:
        block.load_state_dict(state, strict=True, assign=True)
    except (RuntimeError, TypeError) as exc:
        raise ReferenceError(f"pinned block state binding failed: {exc}") from None
    block.to(device=device, dtype=torch.float32).eval()
    del state
    if round_bf16:
        block_reference._round_parameters_bf16_in_place(torch, block, device)
    return block


def _build_chunked_unpad_adapter(torch: Any, receipt: dict[str, Any]) -> Any:
    functional = torch.nn.functional

    def attend(query: Any, key: Any, value: Any) -> tuple[Any, int]:
        query_valid = query[:, :VALID_JOINT_TOKENS]
        key_valid = key[:, :VALID_JOINT_TOKENS]
        value_valid = value[:, :VALID_JOINT_TOKENS]
        q_heads = query_valid.permute(0, 2, 1, 3)
        k_heads = key_valid.permute(0, 2, 1, 3).contiguous()
        v_heads = value_valid.permute(0, 2, 1, 3).contiguous()
        output_heads = torch.zeros(
            (1, 16, TOTAL_TOKENS, 128), dtype=torch.float32, device=query.device
        )
        chunks = 0
        for start in range(0, VALID_JOINT_TOKENS, QUERY_CHUNK):
            stop = min(start + QUERY_CHUNK, VALID_JOINT_TOKENS)
            attended = functional.scaled_dot_product_attention(
                q_heads[:, :, start:stop, :].contiguous(), k_heads, v_heads,
                attn_mask=None, dropout_p=0.0, is_causal=False,
            )
            output_heads[:, :, start:stop, :].copy_(attended)
            chunks += 1
            del attended
        output = output_heads.permute(0, 2, 1, 3).contiguous().flatten(2)
        return output, chunks

    def parallel_attention(q: tuple[Any, Any], k: tuple[Any, Any], v: tuple[Any, Any],
                           img_q_len: int, img_kv_len: int,
                           attn_mode: str | None = None, text_mask: Any = None,
                           attn_param: dict[str, Any] | None = None) -> Any:
        if (attn_mode != "magi_mp_causal" or text_mask is not None or attn_param is None
                or attn_param.get("num_prompts") != 1
                or attn_param.get("chunk_size_tokens", 0) != 0
                or attn_param.get("prompt_video_boundaries_tokens") != [0, VIDEO_TOKENS]
                or attn_param.get("prompt_video_boundaries") != [0, VIDEO_TOKENS]
                or attn_param.get("txt_tokens_per_block") != TEXT_TOKENS
                or attn_param.get("seg_ar_attention_backend") != "magi"
                or img_q_len != VIDEO_TOKENS or img_kv_len != VIDEO_TOKENS):
            raise ReferenceError("Q4 DiT MPS attention adapter received unsupported geometry")
        query, key, value = (torch.cat((parts[0], parts[1]), dim=1) for parts in (q, k, v))
        expected = (1, TOTAL_TOKENS, 16, 128)
        if any(tuple(tensor.shape) != expected or tensor.device.type != "mps"
               or tensor.dtype != torch.float32 for tensor in (query, key, value)):
            raise ReferenceError("Q4 DiT attention Q/K/V shape/device/dtype mismatch")
        output, chunks = attend(query, key, value)
        receipt["calls"] += 1
        receipt["query_chunks"] += chunks
        if receipt["padding_control"] is None:
            poisoned = [tensor.clone() for tensor in (query, key, value)]
            for tensor in poisoned:
                tensor[:, VALID_JOINT_TOKENS:, :, :].fill_(float("nan"))
            poisoned_output, poison_chunks = attend(*poisoned)
            finite = torch.isfinite(poisoned_output).all() & torch.isfinite(output).all()
            close = (poisoned_output - output).abs() <= (ATOL + RTOL * output.abs())
            receipt["padding_control"] = (finite & close.all()).to(
                dtype=torch.float32
            ).reshape(1)
            receipt["padding_control_chunks"] = poison_chunks
            del poisoned, poisoned_output, finite, close
        del query, key, value
        return output

    return parallel_attention


def _capture_projected_text(module: Any, records: dict[str, Any]) -> Any:
    def capture(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        import torch
        if not isinstance(output, torch.Tensor):
            raise ReferenceError("projected text refiner output is not a tensor")
        records["conditioning.projected_text"] = output.detach()
    return module.register_forward_hook(capture)


def _unpatchify(torch: Any, flat: Any) -> Any:
    if tuple(flat.shape) != (1, VIDEO_TOKENS, 32):
        raise ReferenceError("final linear output shape does not match pointwise patches")
    tokens = flat.reshape(1, 6, 22, 40, 32, 1, 1, 1)
    return torch.einsum("nthwcopq->nctohpwq", tokens).reshape(1, 32, 6, 22, 40)


def _serialize_tensors(torch: Any, records: dict[str, Any],
                       terminal_controls: tuple[Any, ...] = ()) -> tuple[bytes, dict[str, Any]]:
    if set(records) != set(EXPECTED_TENSOR_SHAPES):
        raise ReferenceError("terminal tensor inventory differs from the admitted Q4 DiT capture")
    flat: list[Any] = []
    finite: list[Any] = []
    metadata: dict[str, Any] = {}
    offset = 0
    for name in sorted(EXPECTED_TENSOR_SHAPES):
        value = records[name]
        shape = EXPECTED_TENSOR_SHAPES[name]
        if (value.device.type != "mps" or value.dtype != torch.float32
                or list(value.shape) != shape):
            raise ReferenceError(f"terminal MPS F32 tensor shape/device mismatch: {name}")
        contiguous = value.detach().contiguous()
        nbytes = _element_count(shape) * 4
        metadata[name] = {
            "dtype": "F32", "shape": shape,
            "offset_bytes": offset, "nbytes": nbytes,
        }
        offset += nbytes
        flat.append(contiguous.reshape(-1))
        finite.append(torch.isfinite(contiguous).all().to(dtype=torch.float32).reshape(1))
    if offset > SIDE_CAR_LIMIT:
        raise ReferenceError("Q4 DiT sidecar exceeds the 512 MiB bound")
    controls = [value.detach().to(dtype=torch.float32).reshape(-1)
                for value in terminal_controls]
    packed = torch.cat(flat + finite + controls).to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    array = packed.numpy()
    value_count = offset // 4
    finite_count = len(finite)
    finite_flags = array[value_count:value_count + finite_count]
    control_flags = array[value_count + finite_count:]
    if not bool((finite_flags == 1.0).all()):
        raise ReferenceError("non-finite value in one packed terminal MPS readback")
    if not bool((control_flags == 1.0).all()):
        raise ReferenceError("MPS terminal control failed in packed activation readback")
    raw = array[:value_count].astype("<f4", copy=False).tobytes()
    if len(raw) != offset:
        raise ReferenceError("Q4 DiT F32 sidecar extent mismatch")
    validate_tensor_layout(metadata, len(raw))
    return raw, metadata


def _publish_noreplace(path: Path, data: bytes) -> None:
    if not path.parent.is_dir() or os.path.lexists(path):
        raise ReferenceError(f"output path must be new in an existing directory: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    created = False
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def _publish_bundle(output: Path, document: dict[str, Any], sidecar: bytes) -> None:
    sidecar_path = output.parent / document["tensor_file"]["name"]
    if os.path.lexists(sidecar_path) or os.path.lexists(output):
        raise ReferenceError("Q4 DiT reference output or sidecar already exists")
    sidecar_published = False
    try:
        _publish_noreplace(sidecar_path, sidecar)
        sidecar_published = True
        encoded = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if len(encoded) > JSON_LIMIT:
            raise ReferenceError("Q4 DiT reference JSON exceeds its size bound")
        _publish_noreplace(output, encoded)
        sidecar_published = False
    finally:
        if sidecar_published:
            sidecar_path.unlink(missing_ok=True)


def _source_record(path: Path, cache_index: Path, cache_index_sha: str,
                   precision: str, cache_doc: dict[str, Any]) -> dict[str, Any]:
    blocks = []
    for entry in cache_doc["blocks"]:
        blocks.append({
            "block_index": entry["block_index"],
            "manifest": entry["manifest"],
            "manifest_sha256": entry["manifest_sha256"],
            "payload_sha256": entry["payload_sha256"],
            "bf16": entry["bf16"],
            "bf16_sha256": entry["bf16_sha256"],
        })
    return {
        "checkpoint_path": str(path),
        "checkpoint_sha256": SOURCE_SHA256,
        "checkpoint_bytes": SOURCE_BYTES,
        "checkpoint_header_sha256": SOURCE_HEADER_SHA256,
        "cache_index": {"path": str(cache_index), "sha256": cache_index_sha},
        "cache_block_order": list(range(BLOCK_COUNT)),
        "cache_blocks": blocks,
        "precision_source": "original full F32 checkpoint" if precision == "bf16"
        else "pinned Q4_K block payloads plus original global F32 checkpoint",
    }


def _fixture_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new reference JSON in an existing scratch directory")
    parser.add_argument("--precision", choices=("q4", "bf16"), required=True)
    parser.add_argument(
        "--source-loading", choices=("mapped", "owned"), default="mapped",
        help="original F32 block weights: mapped safetensors views or owned raw tensor buffers (BF16 mode)",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--cache-index", type=Path, default=DEFAULT_CACHE_INDEX)
    parser.add_argument("--cache-index-sha256", required=True)
    parser.add_argument("--qwen-reference", type=Path, default=DEFAULT_QWEN_REFERENCE)
    parser.add_argument("--qwen-sidecar", type=Path, default=DEFAULT_QWEN_SIDECAR)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    return parser


def produce_fixture(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise ReferenceError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    if args.precision not in ("q4", "bf16"):
        raise ReferenceError("precision must be q4 or bf16")
    if args.source_loading not in ("mapped", "owned"):
        raise ReferenceError("source-loading must be mapped or owned")
    if args.precision == "q4" and args.source_loading == "owned":
        raise ReferenceError("owned source loading is only available for the BF16 original-block path")
    output = args.output.absolute()
    if (output.suffix.lower() != ".json" or os.path.lexists(output)
            or not output.parent.is_dir()):
        raise ReferenceError("--output must be a new .json path in an existing directory")
    tensor_path = output.parent / f"{output.stem}.tensors.f32le"
    if os.path.lexists(tensor_path):
        raise ReferenceError("Q4 DiT tensor sidecar already exists")

    # Complete byte, header, cache and Qwen admission before importing torch or
    # constructing any MPS object. These files and identities are snapshots.
    weights_path, weights_sha, weights_identity, source_layout = _validate_source_checkpoint(
        args.weights, args.weights_sha256
    )
    cache_path, cache_sha, cache_doc, q4_blocks = _validate_cache_index(
        args.cache_index, args.cache_index_sha256, precision=args.precision
    )
    (_qwen_doc, qwen_sidecar_path, qwen_ref_sha, qwen_sidecar_sha,
     qwen_reference_identity, qwen_sidecar_identity) = _validate_qwen_source(
        args.qwen_reference, args.qwen_sidecar
    )
    upstream = args.upstream.resolve(strict=True)
    try:
        source_hashes = conditioned_reference._verify_upstream(upstream)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ReferenceError(f"pinned upstream source validation failed: {exc}") from None

    import numpy as np
    import torch
    from safetensors import safe_open

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ReferenceError("MPS is required; CPU fallback is forbidden")
    if torch.get_default_dtype() != torch.float32:
        torch.set_default_dtype(torch.float32)
    device = torch.device("mps")
    records: dict[str, Any] = {}
    receipt: dict[str, Any] = {
        "calls": 0, "query_chunks": 0, "padding_control": None,
        "padding_control_chunks": 0,
    }
    runtime_started = False
    try:
        runtime_started = True
        helpers = block_reference._install_upstream_modules(upstream)
        modules = _global_modules(torch, upstream, helpers)
        _emit_stage_telemetry(torch, "global_modules_ready")
        # _global_modules installs the full-row attention adapter before this
        # module import; avoid replacing its imported function afterward.
        import importlib
        token_refiner = importlib.import_module("hymm.models.text_encoder.token_refiner")
        if token_refiner.SingleTokenRefinerPerBlock is not type(modules["txt_in"]):
            raise ReferenceError("text refiner attention adapter was not installed before module import")

        if (_identity(args.qwen_reference.resolve()) != qwen_reference_identity
                or _identity(qwen_sidecar_path) != qwen_sidecar_identity):
            raise ReferenceError("Qwen source changed after admission")
        with qwen_sidecar_path.open("rb") as qwen_stream:
            qwen_map = np.memmap(
                qwen_stream, mode="r", dtype="<u2", shape=(27, 1, 1388, 3584)
            )
            qwen_rows = np.array(qwen_map[26, 0, 108:1388], copy=True)
            del qwen_map
            stream_identity = os.fstat(qwen_stream.fileno())
            if ((stream_identity.st_dev, stream_identity.st_ino, stream_identity.st_size,
                 stream_identity.st_mtime_ns, stream_identity.st_ctime_ns)
                    != qwen_sidecar_identity):
                raise ReferenceError("Qwen sidecar changed while loading the saved crop")
        qwen_cpu_bf16 = torch.from_numpy(qwen_rows).view(torch.bfloat16)
        del qwen_rows
        qwen_input = qwen_cpu_bf16.unsqueeze(0).to(device=device, dtype=torch.float32)
        qwen_for_refiner = qwen_input.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        full_mask = torch.tensor([MASK], dtype=torch.bool, device=device)
        timestep = torch.tensor([500.0], dtype=torch.float32, device=device)

        with safe_open(str(weights_path), framework="pt", device="cpu") as source:
            if _identity(weights_path) != weights_identity:
                raise ReferenceError("full checkpoint changed after byte admission")
            global_modules = _build_global_modules(torch, source, upstream, helpers, device)
            projected_hook = _capture_projected_text(
                global_modules["txt_in"].individual_token_refiner, records
            )
            try:
                with torch.inference_mode():
                    vec = global_modules["time_in"](timestep)
                    global_modules["txt_in"](qwen_for_refiner, timestep, full_mask)
            finally:
                projected_hook.remove()
            projected = records.get("conditioning.projected_text")
            if (projected is None or list(projected.shape) != [1, TEXT_TOKENS, 2048]
                    or vec.device.type != "mps" or vec.dtype != torch.float32):
                raise ReferenceError("full-row refiner or timestep output shape/device mismatch")
            records["conditioning.vec"] = vec
            records["input.qwen_hidden"] = qwen_input

            latent = _deterministic_latent(torch, device)
            records["input.latent"] = latent
            with torch.inference_mode():
                img = global_modules["img_in"](latent)
            if list(img.shape) != [1, VIDEO_TOKENS, 2048]:
                raise ReferenceError("upstream img_in output shape mismatch")
            records["img_in"] = img

            rope_cos, rope_sin = helpers[-1].get_nd_rotary_pos_embed(
                [16, 56, 56], tuple(GEOMETRY["rope_grid"]), theta=GEOMETRY["rope_theta"],
                use_real=True, theta_rescale_factor=1, device=device,
            )
            if list(rope_cos.shape) != [VIDEO_TOKENS, 128] or list(rope_sin.shape) != [VIDEO_TOKENS, 128]:
                raise ReferenceError("pinned upstream ND RoPE output shape mismatch")
            records["input.rope_cos"] = rope_cos
            records["input.rope_sin"] = rope_sin

            adapter = _build_chunked_unpad_adapter(torch, receipt)
            block_type = block_reference._extract_pinned_block(
                upstream, torch, adapter, helpers[-1].apply_rotary_emb, helpers
            )
            _emit_stage_telemetry(torch, "global_inputs_ready")
            block_inputs = {
                "img": img, "txt": projected, "vec": vec, "vec_txt": vec,
                "rope_cos": rope_cos, "rope_sin": rope_sin,
            }
            for index in range(BLOCK_COUNT):
                _emit_stage_telemetry(torch, "block_begin", block_index=index)
                if args.precision == "q4":
                    state = _read_q4_payload_state(torch, np, q4_blocks[index], index, device)
                    round_bf16 = False
                else:
                    if args.source_loading == "owned":
                        state = _load_owned_source_block_state(
                            torch, weights_path, weights_identity, source_layout, index
                        )
                    else:
                        state = _load_source_block_state(torch, source, index)
                    round_bf16 = True
                _emit_stage_telemetry(
                    torch, "block_source_loaded", block_index=index,
                    source_tensor_count=len(state),
                    source_loading=(
                        "q4_cache_payloads" if args.precision == "q4"
                        else ("owned_raw_f32_extents" if args.source_loading == "owned"
                              else "safetensors_mapped_tensor_views")
                    ),
                )
                block = _build_block(
                    torch, helpers, block_type, state, device,
                    round_bf16=round_bf16,
                )
                _emit_stage_telemetry(torch, "block_weights_ready", block_index=index)
                attn_param = {
                    "num_prompts": 1,
                    "txt_tokens_per_block": TEXT_TOKENS,
                    "prompt_video_boundaries_tokens": [0, VIDEO_TOKENS],
                    "prompt_video_boundaries": [0, VIDEO_TOKENS],
                    "chunk_size_tokens": 0,
                    "seg_ar_attention_backend": "magi",
                }
                before_calls = receipt["calls"]
                with torch.inference_mode():
                    img_out, txt_out = block(
                        img=block_inputs["img"], txt=block_inputs["txt"],
                        vec=block_inputs["vec"], vec_txt=block_inputs["vec_txt"],
                        freqs_cis=(rope_cos, rope_sin), text_mask=None,
                        attn_param=attn_param, is_flash=False,
                        txt_tokens_per_block=TEXT_TOKENS,
                        prompt_video_boundaries=[0, VIDEO_TOKENS],
                    )
                if receipt["calls"] != before_calls + 1:
                    raise ReferenceError(f"block {index} did not make one MPS attention call")
                _emit_stage_telemetry(
                    torch, "block_forward_returned", block_index=index,
                    attention_calls=receipt["calls"],
                )
                for label, value, shape in (
                    ("img", img_out, [1, VIDEO_TOKENS, 2048]),
                    ("txt", txt_out, [1, TEXT_TOKENS, 2048]),
                ):
                    if value.device.type != "mps" or value.dtype != torch.float32 or list(value.shape) != shape:
                        raise ReferenceError(f"block {index} {label} output shape/device/dtype mismatch")
                if index in CAPTURE_BLOCKS:
                    records[f"block{index}.img.output"] = img_out.detach()
                    records[f"block{index}.txt.output"] = txt_out.detach()
                block_inputs = {**block_inputs, "img": img_out, "txt": txt_out}
                del block, state
                if index not in CAPTURE_BLOCKS:
                    del img_out, txt_out
                _emit_stage_telemetry(torch, "block_released", block_index=index)

            if (receipt["calls"] != BLOCK_COUNT
                    or receipt["query_chunks"] != BLOCK_COUNT * ATTENTION_CHUNKS_PER_BLOCK
                    or receipt["padding_control_chunks"] != ATTENTION_CHUNKS_PER_BLOCK
                    or receipt["padding_control"] is None):
                raise ReferenceError("full DiT attention/padding control receipt mismatch")
            final_layer = global_modules["final_layer"]
            final_handles = []

            def save_output(name: str):
                def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                    records[name] = output.detach()
                return hook

            def save_input(name: str):
                def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                    records[name] = inputs[0].detach()
                return hook

            final_handles.extend([
                final_layer.norm_final.register_forward_hook(save_output("final.norm")),
                final_layer.adaLN_modulation.register_forward_hook(save_output("final.modulation")),
                final_layer.linear.register_forward_pre_hook(save_input("final.modulated")),
                final_layer.linear.register_forward_hook(save_output("final.linear")),
            ])
            try:
                with torch.inference_mode():
                    final_tokens = final_layer(block_inputs["img"], vec)
            finally:
                for handle in final_handles:
                    handle.remove()
            if list(final_tokens.shape) != [1, VIDEO_TOKENS, 32]:
                raise ReferenceError("upstream final layer output shape mismatch")
            output_latent = _unpatchify(torch, final_tokens)
            validate_unpatchified_shape(list(output_latent.shape))
            records["output.latent"] = output_latent

            if _identity(weights_path) != weights_identity:
                raise ReferenceError("full checkpoint changed during reference execution")
            del global_modules, block_inputs, block_type
        _emit_stage_telemetry(torch, "terminal_readback_begin")
        raw_sidecar, tensor_metadata = _serialize_tensors(
            torch, records, terminal_controls=(receipt["padding_control"],)
        )
        _emit_stage_telemetry(torch, "terminal_readback_complete",
                              sidecar_bytes=len(raw_sidecar))
    except BaseException:
        if runtime_started:
            try:
                torch.mps.synchronize()
            except BaseException:
                pass
        raise

    tensor_sha = hashlib.sha256(raw_sidecar).hexdigest()
    precision = dict(PRECISION)
    precision["mode"] = args.precision
    source_record = _source_record(weights_path, cache_path, cache_sha, args.precision, cache_doc)
    document = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "precision_mode": args.precision,
        "upstream": {"revision": PINNED_REVISION,
                     "source_sha256": source_hashes},
        "source": source_record,
        "qwen_source": dict(text_reference.QWEN_SOURCE),
        "qwen_reference": {"path": str(args.qwen_reference.resolve()), "sha256": qwen_ref_sha},
        "qwen_sidecar": {"path": str(qwen_sidecar_path), "sha256": qwen_sidecar_sha,
                          "nbytes": qwen_sidecar_path.stat().st_size},
        "geometry": dict(GEOMETRY),
        "input_generation": dict(INPUT_GENERATION),
        "precision": precision,
        "mask": list(MASK),
        "timestep": 500.0,
        "conditioning": dict(_CONDITIONING),
        "padding_policy": dict(_PADDING_POLICY),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "producer": {
            "path": str(Path(__file__).resolve()),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "device": "mps",
            "activation_readback": "one packed terminal F32 tensor sidecar readback",
            "terminal_activation_readbacks": 1,
            "attention_adapter": "PyTorch MPS SDPA F32, valid joint prefix only",
            "attention_adapter_calls": receipt["calls"],
            "query_chunks_per_block": ATTENTION_CHUNKS_PER_BLOCK,
            "attention_query_chunk_calls": receipt["query_chunks"],
            "padding_poison_control_chunks": receipt["padding_control_chunks"],
            "padding_poison_control_passed": True,
            "block_invocations": list(range(BLOCK_COUNT)),
            "captured_block_outputs": list(CAPTURE_BLOCKS),
            "block_handoff": "img/text outputs passed directly to the next block",
            "final_layer": "pinned FinalLayerPerBlock forward plus pinned pointwise unpatchify layout",
            "camera_conditioning": "off",
            "action_conditioning": "off",
            "full_generation": False,
            "weight_source_loading": {
                "block_weights": (
                    "q4_cache_payloads" if args.precision == "q4"
                    else ("owned_raw_f32_extents" if args.source_loading == "owned"
                          else "safetensors_mapped_tensor_views")
                ),
                "global_weights": "safetensors_mapped_tensor_views",
            },
        },
        "tensor_file": {
            "name": tensor_path.name,
            "nbytes": len(raw_sidecar),
            "sha256": tensor_sha,
            "encoding": "f32-le",
        },
        "tensors": tensor_metadata,
    }
    validate_reference_document(document)
    _publish_bundle(output, document, raw_sidecar)
    return document


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--decoder-probe"]:
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
            print("ERROR: PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0", file=sys.stderr)
            return 2
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        try:
            import numpy as np
            import torch
            result = run_q4_decoder_probe(torch, np)
        except (ReferenceError, OSError, RuntimeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0

    args = _fixture_parser().parse_args(argv)
    try:
        document = produce_fixture(args)
    except (ReferenceError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "schema": document["schema"],
        "precision_mode": document["precision_mode"],
        "output": str(args.output),
        "tensor_file": document["tensor_file"],
        "block_count": BLOCK_COUNT,
        "terminal_readbacks": document["producer"]["terminal_activation_readbacks"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
