#!/usr/bin/env python3
"""Build a fresh, offline Hy-Nomas Cinematic LoRA BF16 cache.

The builder authenticates the pinned F32 base, Q4 cache index, adapter config,
and adapter before importing PyTorch or touching MPS. It maps safetensors files
and handles one active matrix at a time: F32 base + (B @ A) * (alpha / rank),
computed on MPS in F32 and rounded once to BF16. Camera-off action targets are
explicitly listed in the lineage and are not merged. The output preserves the
base Q4 manifests/payloads and globals by hard link, and copies every unchanged
per-block BF16 tensor byte-for-byte.

This is an offline cache producer only. It does not instantiate or run a model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import time
from typing import Any, BinaryIO, Iterable, Mapping
import uuid


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path("/opt/mldata_ssd/cogni-ml")
BASE_CACHE_DIR = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/weights/q4-cache-v2")
BASE_INDEX_PATH = BASE_CACHE_DIR / "index.json"
BASE_SOURCE_PATH = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/weights/"
    "diffusion_pytorch_model_ema_0_995.safetensors"
)
ADAPTER_PATH = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-quality-next-20261031/"
    "cinematic-v1/adapter.safetensors"
)
ADAPTER_CONFIG_PATH = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-quality-next-20261031/"
    "cinematic-v1/adapter_config.json"
)

BASE_SOURCE_SHA256 = "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70"
BASE_SOURCE_BYTES = 34_213_991_280
BASE_SOURCE_HEADER_SHA256 = "cbfa83dc81af315b425fd783e0ce42d8ff09df772952ef2a8aff48493e21b200"
BASE_INDEX_SHA256 = "a235ca592b47161472c7fd94cd3d886fce164480cd0571179c8860002063cbb0"
UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
UPSTREAM_REPO = "tencent/Hy-Nomas"
UPSTREAM_MERGE_SCRIPT_SHA256 = "92e24f385660a2d37c2e23c23fd4f66f3ba1b0f8195a9e59ba500ff15a7bbee4"
ADAPTER_REVISION = "bb6a8ca451bfe84947f0ea1c6d51957f983ffffe"
ADAPTER_HUB_PATH = "DiT/lora_weights/cinematic/adapter.safetensors"
ADAPTER_SHA256 = "765426a6c417eed70a57d72f67cfe002076cfc92b6b35fe185091012ba80c7c2"
ADAPTER_BYTES = 6_122_591_632
ADAPTER_CONFIG_SHA256 = "f656c2866953e7a90a785a5c51ce127a79ea61c4217ef30e3cd2d69afcd3d998"
ADAPTER_CONFIG_BYTES = 704
ADAPTER_RANK = 256
ADAPTER_ALPHA = 512.0
MERGE_SCALE = ADAPTER_ALPHA / ADAPTER_RANK

BLOCK_COUNT = 54
SOURCE_TENSOR_COUNT = 2268
ADAPTER_PAIR_COUNT = 918
ACTIVE_TARGET_COUNT = 756
SKIPPED_ACTION_TARGET_COUNT = 162
BLOCK_TENSOR_COUNT = 32
INDEX_SCHEMA = "hy-nomas-q4-dit-cache-index-v1"
LINEAGE_SCHEMA = "cogni_ml.hy_nomas_cinematic_bf16_cache/v1"
Q4_MANIFEST_SCHEMA = "hy-nomas-q4-block-cache-v1"
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 8 * 1024 * 1024
HASH_CHUNK_BYTES = 8 * 1024 * 1024

ACTIVE_TARGET_SHAPES: dict[str, tuple[int, int]] = {
    "img_attn_q": (2048, 2048),
    "img_attn_k": (2048, 2048),
    "img_attn_v": (2048, 2048),
    "img_attn_proj": (2048, 2048),
    "txt_attn_q": (2048, 2048),
    "txt_attn_k": (2048, 2048),
    "txt_attn_v": (2048, 2048),
    "txt_attn_proj": (2048, 2048),
    "img_mlp.fc1": (8192, 2048),
    "img_mlp.fc2": (2048, 8192),
    "txt_mlp.fc1": (8192, 2048),
    "txt_mlp.fc2": (2048, 8192),
    "img_mod.linear": (12288, 2048),
    "txt_mod.linear": (12288, 2048),
}
ACTION_TARGET_SHAPES: dict[str, tuple[int, int]] = {
    "img_attn_q": (1024, 2048),
    "keyboard_attn_kv": (2048, 128),
    "proj_keyboard": (2048, 1024),
}
CONFIG_TARGETS = (
    "double_blocks.*.img_attn_q",
    "double_blocks.*.img_attn_k",
    "double_blocks.*.img_attn_v",
    "double_blocks.*.img_attn_proj",
    "double_blocks.*.txt_attn_q",
    "double_blocks.*.txt_attn_k",
    "double_blocks.*.txt_attn_v",
    "double_blocks.*.txt_attn_proj",
    "double_blocks.*.img_mlp.fc1",
    "double_blocks.*.img_mlp.fc2",
    "double_blocks.*.txt_mlp.fc1",
    "double_blocks.*.txt_mlp.fc2",
    "double_blocks.*.img_mod.linear",
    "double_blocks.*.txt_mod.linear",
    "action_modules.*.img_attn_q",
    "action_modules.*.keyboard_attn_kv",
    "action_modules.*.proj_keyboard",
)


class CinematicCacheError(ValueError):
    """An input or output failed the pinned offline cache contract."""


@dataclass(frozen=True)
class TensorRecord:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class SafetensorsHeader:
    path: Path
    file_bytes: int
    raw_header: bytes
    header_prefix: bytes
    data_start: int
    tensors: dict[str, TensorRecord]


@dataclass(frozen=True)
class AdapterPair:
    target: str
    a_key: str
    b_key: str
    a: TensorRecord
    b: TensorRecord
    active: bool


@dataclass(frozen=True)
class Preflight:
    source_header: SafetensorsHeader
    adapter_header: SafetensorsHeader
    adapter_pairs: dict[str, AdapterPair]
    base_index: dict[str, Any]
    block_headers: tuple[SafetensorsHeader, ...]
    globals_header: SafetensorsHeader
    q4_manifests: tuple[tuple[Path, Path, str], ...]
    external_snapshots: tuple[tuple[Path, tuple[int, int, int, int], str], ...]


def _check_deadline(deadline_unix: float | None, where: str) -> None:
    if deadline_unix is not None and time.time() >= deadline_unix:
        raise CinematicCacheError(f"absolute build deadline reached during {where}")


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _safe_basename(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value not in (".", "..")
        and not value.startswith(".")
        and "/" not in value
        and "\\" not in value
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,250}", value) is not None
        and ".." not in value
    )


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CinematicCacheError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_json_bytes(raw: bytes, *, label: str, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    if not raw or len(raw) > max_bytes:
        raise CinematicCacheError(f"{label} JSON byte size is invalid")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, CinematicCacheError) as exc:
        raise CinematicCacheError(f"{label} is not valid duplicate-free UTF-8 JSON: {exc}") from None
    if not isinstance(value, dict):
        raise CinematicCacheError(f"{label} JSON root must be an object")
    return value


def _file_snapshot(path: Path, label: str) -> tuple[int, int, int, int]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise CinematicCacheError(f"{label} is unavailable: {path}: {exc}") from None
    if path.is_symlink() or not path.is_file():
        raise CinematicCacheError(f"{label} must be a non-symlink regular file: {path}")
    return before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns


def _same_file_snapshot(path: Path, before: tuple[int, int, int, int], label: str) -> None:
    after = _file_snapshot(path, label)
    if after != before:
        raise CinematicCacheError(f"{label} changed while it was being read: {path}")


def _sha256_file(path: Path, *, label: str, deadline_unix: float | None = None) -> str:
    before = _file_snapshot(path, label)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != before:
                raise CinematicCacheError(f"{label} changed before hashing: {path}")
            while True:
                _check_deadline(deadline_unix, f"hashing {label}")
                chunk = stream.read(HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
            after_fd = os.fstat(stream.fileno())
            if (after_fd.st_dev, after_fd.st_ino, after_fd.st_size,
                    after_fd.st_mtime_ns) != before:
                raise CinematicCacheError(f"{label} changed while hashing: {path}")
    except OSError as exc:
        raise CinematicCacheError(f"could not hash {label}: {path}: {exc}") from None
    _same_file_snapshot(path, before, label)
    return digest.hexdigest()


def _read_regular_bytes(path: Path, *, label: str, max_bytes: int) -> bytes:
    before = _file_snapshot(path, label)
    if before[2] <= 0 or before[2] > max_bytes:
        raise CinematicCacheError(f"{label} size is invalid: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CinematicCacheError(f"could not read {label}: {path}: {exc}") from None
    _same_file_snapshot(path, before, label)
    if len(raw) != before[2]:
        raise CinematicCacheError(f"{label} changed size while reading: {path}")
    return raw


def _dtype_nbytes(dtype: str) -> int:
    if dtype == "F32":
        return 4
    if dtype == "BF16":
        return 2
    raise CinematicCacheError(f"unsupported safetensors dtype: {dtype}")


def _element_count(shape: tuple[int, ...]) -> int:
    result = 1
    if not shape:
        raise CinematicCacheError("scalar safetensors tensors are outside this cache contract")
    for dimension in shape:
        if type(dimension) is not int or dimension <= 0:
            raise CinematicCacheError("safetensors tensor dimensions must be positive integers")
        result *= dimension
    return result


def read_safetensors_header(path: Path, *, label: str) -> SafetensorsHeader:
    """Read and validate only a bounded safetensors header, never tensor payloads."""
    before = _file_snapshot(path, label)
    file_bytes = before[2]
    if file_bytes < 8:
        raise CinematicCacheError(f"{label} is shorter than the safetensors prefix")
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != before:
                raise CinematicCacheError(f"{label} changed before its header read")
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise CinematicCacheError(f"{label} has a short safetensors prefix")
            header_bytes = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_bytes <= MAX_SAFETENSORS_HEADER_BYTES:
                raise CinematicCacheError(f"{label} safetensors header length is outside the limit")
            raw_header = stream.read(header_bytes)
            if len(raw_header) != header_bytes:
                raise CinematicCacheError(f"{label} has a short safetensors header")
            after_fd = os.fstat(stream.fileno())
            if (after_fd.st_dev, after_fd.st_ino, after_fd.st_size,
                    after_fd.st_mtime_ns) != before:
                raise CinematicCacheError(f"{label} changed during its header read")
    except OSError as exc:
        raise CinematicCacheError(f"could not read {label} header: {exc}") from None
    _same_file_snapshot(path, before, label)
    header = _load_json_bytes(raw_header.rstrip(b" "), label=f"{label} safetensors header",
                              max_bytes=MAX_SAFETENSORS_HEADER_BYTES)
    data_bytes = file_bytes - 8 - header_bytes
    metadata = header.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(not isinstance(key, str) or not isinstance(value, str)
               for key, value in metadata.items())
    ):
        raise CinematicCacheError(f"{label} safetensors metadata must map strings to strings")
    if not header:
        raise CinematicCacheError(f"{label} safetensors tensor inventory is empty")
    tensors: dict[str, TensorRecord] = {}
    extents: list[tuple[int, int, str]] = []
    for name, value in header.items():
        if not isinstance(name, str) or not name or name == "__metadata__" or not isinstance(value, dict):
            raise CinematicCacheError(f"{label} safetensors tensor record is malformed")
        if set(value) != {"dtype", "shape", "data_offsets"}:
            raise CinematicCacheError(f"{label} safetensors tensor fields are invalid: {name}")
        dtype = value["dtype"]
        shape_raw = value["shape"]
        offsets = value["data_offsets"]
        if not isinstance(dtype, str) or not isinstance(shape_raw, list):
            raise CinematicCacheError(f"{label} safetensors dtype or shape is malformed: {name}")
        shape = tuple(shape_raw)
        expected_nbytes = _element_count(shape) * _dtype_nbytes(dtype)
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(item) is not int for item in offsets)):
            raise CinematicCacheError(f"{label} safetensors offsets are malformed: {name}")
        start, end = offsets
        if start < 0 or end <= start or end > data_bytes or end - start != expected_nbytes:
            raise CinematicCacheError(f"{label} safetensors byte extent mismatch: {name}")
        tensors[name] = TensorRecord(dtype, shape, start, end)
        extents.append((start, end, name))
    cursor = 0
    for start, end, name in sorted(extents):
        if start != cursor:
            raise CinematicCacheError(f"{label} safetensors ranges overlap or have a gap at {name}")
        cursor = end
    if cursor != data_bytes:
        raise CinematicCacheError(f"{label} safetensors data section has unclaimed bytes")
    return SafetensorsHeader(path, file_bytes, prefix + raw_header, prefix + raw_header,
                             8 + header_bytes, tensors)


def _lora_shapes_for_target(module: str, *, action: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    if action:
        out_features, in_features = ACTION_TARGET_SHAPES[module]
    else:
        out_features, in_features = ACTIVE_TARGET_SHAPES[module]
    return (ADAPTER_RANK, in_features), (out_features, ADAPTER_RANK)


def expected_lora_key_shapes() -> dict[str, tuple[str, tuple[int, int]]]:
    result: dict[str, tuple[str, tuple[int, int]]] = {}
    for index in range(BLOCK_COUNT):
        for module in ACTIVE_TARGET_SHAPES:
            target = f"double_blocks.{index}.{module}.weight"
            a_shape, b_shape = _lora_shapes_for_target(module, action=False)
            result[f"{target[:-7]}.lora_A.weight"] = ("F32", a_shape)
            result[f"{target[:-7]}.lora_B.weight"] = ("F32", b_shape)
        for module in ACTION_TARGET_SHAPES:
            target = f"action_modules.{index}.{module}.weight"
            a_shape, b_shape = _lora_shapes_for_target(module, action=True)
            result[f"{target[:-7]}.lora_A.weight"] = ("F32", a_shape)
            result[f"{target[:-7]}.lora_B.weight"] = ("F32", b_shape)
    return result


def normalize_lora_key(name: str) -> tuple[str, str]:
    if name.endswith(".lora_A.weight"):
        suffix = "A"
        stem = name[:-len(".lora_A.weight")]
    elif name.endswith(".lora_B.weight"):
        suffix = "B"
        stem = name[:-len(".lora_B.weight")]
    else:
        raise CinematicCacheError(f"unexpected adapter tensor suffix: {name}")
    stem = stem.replace("._checkpoint_wrapped_module.", ".")
    stem = stem.replace("_checkpoint_wrapped_module.", "")
    target = stem + ".weight"
    if (not stem or target.startswith(".") or target.endswith("..weight")
            or ".." in target or "/" in target or "\\" in target):
        raise CinematicCacheError(f"malformed normalized adapter target: {name}")
    return target, suffix


def validate_adapter_inventory(
    records: Iterable[tuple[str, TensorRecord]],
) -> dict[str, AdapterPair]:
    expected = expected_lora_key_shapes()
    seen_keys: set[str] = set()
    pairs: dict[str, dict[str, tuple[str, TensorRecord]]] = {}
    for name, record in records:
        if not isinstance(name, str) or not isinstance(record, TensorRecord):
            raise CinematicCacheError("adapter tensor inventory records are malformed")
        target, suffix = normalize_lora_key(name)
        if name in seen_keys or suffix in pairs.setdefault(target, {}):
            raise CinematicCacheError(f"duplicate adapter tensor or normalized target: {name}")
        seen_keys.add(name)
        if name not in expected:
            raise CinematicCacheError(f"unexpected adapter tensor target: {name}")
        expected_dtype, expected_shape = expected[name]
        if record.dtype != expected_dtype:
            raise CinematicCacheError(f"adapter tensors must be F32: {name}")
        if record.shape != expected_shape:
            raise CinematicCacheError(f"adapter tensor rank/shape mismatch: {name}")
        active = target.startswith("double_blocks.")
        pairs[target][suffix] = (name, record)
        pairs[target]["_active"] = ("1" if active else "0", record)

    missing = sorted(set(expected) - seen_keys)
    if missing:
        raise CinematicCacheError(f"adapter inventory is missing {len(missing)} expected tensors")
    if len(seen_keys) != 2 * ADAPTER_PAIR_COUNT:
        raise CinematicCacheError("adapter tensor count is not exactly 1836")
    if len(pairs) != ADAPTER_PAIR_COUNT:
        raise CinematicCacheError("adapter target count is not exactly 918")
    result: dict[str, AdapterPair] = {}
    for target, values in pairs.items():
        if set(values) != {"A", "B", "_active"}:
            raise CinematicCacheError(f"adapter A/B pair is incomplete: {target}")
        a_key, a_record = values["A"]
        b_key, b_record = values["B"]
        active = values["_active"][0] == "1"
        result[target] = AdapterPair(target, a_key, b_key, a_record, b_record, active)
    active_count = sum(pair.active for pair in result.values())
    if active_count != ACTIVE_TARGET_COUNT or len(result) - active_count != SKIPPED_ACTION_TARGET_COUNT:
        raise CinematicCacheError("adapter active/action pair counts differ from pinned route")
    return result


def skipped_action_targets() -> list[str]:
    return sorted(
        f"action_modules.{index}.{module}.weight"
        for index in range(BLOCK_COUNT)
        for module in ACTION_TARGET_SHAPES
    )


def active_target_names() -> list[str]:
    return sorted(
        f"double_blocks.{index}.{module}.weight"
        for index in range(BLOCK_COUNT)
        for module in ACTIVE_TARGET_SHAPES
    )


def validate_config_document(
    document: dict[str, Any], *, actual_sha256: str, actual_bytes: int,
    expected_sha256: str = ADAPTER_CONFIG_SHA256,
    expected_bytes: int = ADAPTER_CONFIG_BYTES,
) -> None:
    if actual_sha256 != expected_sha256 or actual_bytes != expected_bytes:
        raise CinematicCacheError("adapter config SHA-256 or byte pin mismatch")
    if set(document) != {"format_version", "rank", "alpha", "dropout", "targets"}:
        raise CinematicCacheError("adapter config fields differ from pinned contract")
    if (type(document["format_version"]) is not int or document["format_version"] != 1
            or type(document["rank"]) is not int or document["rank"] != ADAPTER_RANK
            or type(document["alpha"]) not in (int, float) or document["alpha"] != ADAPTER_ALPHA
            or type(document["dropout"]) not in (int, float) or document["dropout"] != 0.0):
        raise CinematicCacheError("adapter config rank, alpha, format, or dropout mismatch")
    targets = document["targets"]
    if (not isinstance(targets, list) or any(not isinstance(item, str) for item in targets)
            or len(targets) != len(CONFIG_TARGETS) or len(set(targets)) != len(targets)
            or tuple(targets) != CONFIG_TARGETS):
        raise CinematicCacheError("adapter config target patterns differ from pinned target inventory")


def validate_base_source_inventory(records: Mapping[str, TensorRecord]) -> None:
    if len(records) != SOURCE_TENSOR_COUNT:
        raise CinematicCacheError("base F32 source tensor inventory is not exactly 2268")
    if any(record.dtype != "F32" for record in records.values()):
        raise CinematicCacheError("base checkpoint inventory must contain only F32 tensors")
    expected: dict[str, tuple[int, int]] = {}
    for index in range(BLOCK_COUNT):
        for module, shape in ACTIVE_TARGET_SHAPES.items():
            expected[f"double_blocks.{index}.{module}.weight"] = shape
        for module, shape in ACTION_TARGET_SHAPES.items():
            expected[f"action_modules.{index}.{module}.weight"] = shape
    if len(expected) != ACTIVE_TARGET_COUNT + SKIPPED_ACTION_TARGET_COUNT:
        raise CinematicCacheError("internal active/action target inventory count mismatch")
    missing = sorted(set(expected) - set(records))
    if missing:
        raise CinematicCacheError(f"base F32 source is missing {len(missing)} pinned target weights")
    for name, shape in expected.items():
        if records[name].shape != shape:
            raise CinematicCacheError(f"base F32 target shape mismatch: {name}")


def validate_cache_index_document(
    document: dict[str, Any], *, expected_source_sha256: str, expected_source_bytes: int,
) -> None:
    if set(document) != {"schema", "schema_version", "source", "blocks", "globals"}:
        raise CinematicCacheError("Q4 cache index fields differ from schema v1")
    if (document.get("schema") != INDEX_SCHEMA
            or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1):
        raise CinematicCacheError("Q4 cache index schema or version mismatch")
    source = document["source"]
    if (not isinstance(source, dict) or set(source) != {"sha256", "bytes"}
            or source.get("sha256") != expected_source_sha256
            or type(source.get("bytes")) is not int
            or source.get("bytes") != expected_source_bytes):
        raise CinematicCacheError("Q4 cache index source pin mismatch")
    entries = document["blocks"]
    if not isinstance(entries, list) or len(entries) != BLOCK_COUNT:
        raise CinematicCacheError("Q4 cache index must contain exactly 54 ordered blocks")
    manifest_names: set[str] = set()
    manifest_hashes: set[str] = set()
    payload_hashes: set[str] = set()
    bf16_names: set[str] = set()
    for expected_index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "block_index", "manifest", "manifest_sha256", "payload_sha256", "bf16", "bf16_sha256"
        }:
            raise CinematicCacheError("Q4 cache block index record has invalid fields")
        if type(entry["block_index"]) is not int or entry["block_index"] != expected_index:
            raise CinematicCacheError("Q4 cache index blocks must be exactly ordered 0...53")
        for name_field in ("manifest", "bf16"):
            if not _safe_basename(entry[name_field]):
                raise CinematicCacheError(f"Q4 cache {name_field} is not a safe sibling basename")
        for hash_field in ("manifest_sha256", "payload_sha256", "bf16_sha256"):
            if not _valid_sha256(entry[hash_field]):
                raise CinematicCacheError(f"Q4 cache {hash_field} is not lowercase SHA-256")
        if (entry["manifest"] in manifest_names or entry["manifest_sha256"] in manifest_hashes
                or entry["payload_sha256"] in payload_hashes or entry["bf16"] in bf16_names):
            raise CinematicCacheError("Q4 cache index contains duplicate file identities")
        manifest_names.add(entry["manifest"])
        manifest_hashes.add(entry["manifest_sha256"])
        payload_hashes.add(entry["payload_sha256"])
        bf16_names.add(entry["bf16"])
    globals_record = document["globals"]
    if (not isinstance(globals_record, dict) or set(globals_record) != {"bf16", "sha256"}
            or not _safe_basename(globals_record.get("bf16"))
            or not _valid_sha256(globals_record.get("sha256"))):
        raise CinematicCacheError("Q4 cache global BF16 pin is invalid")
    if globals_record["bf16"] in bf16_names or globals_record["bf16"] in manifest_names:
        raise CinematicCacheError("Q4 cache index repeats a sibling basename")


def validate_base_index_document(
    document: dict[str, Any], *, actual_sha256: str,
    expected_index_sha256: str = BASE_INDEX_SHA256,
    expected_source_sha256: str = BASE_SOURCE_SHA256,
    expected_source_bytes: int = BASE_SOURCE_BYTES,
) -> None:
    if actual_sha256 != expected_index_sha256:
        raise CinematicCacheError("base cache index SHA-256 pin mismatch")
    validate_cache_index_document(document, expected_source_sha256=expected_source_sha256,
                                  expected_source_bytes=expected_source_bytes)


def validate_lineage_document(
    document: dict[str, Any], *, expected_index_sha256: str,
) -> None:
    expected_fields = {
        "schema", "upstream_revision", "base_index_sha256", "base_source_sha256",
        "adapter", "merge", "index", "producer",
    }
    if set(document) != expected_fields or document.get("schema") != LINEAGE_SCHEMA:
        raise CinematicCacheError("cinematic lineage schema or field set mismatch")
    if (document.get("upstream_revision") != UPSTREAM_REVISION
            or document.get("base_index_sha256") != BASE_INDEX_SHA256
            or document.get("base_source_sha256") != BASE_SOURCE_SHA256):
        raise CinematicCacheError("cinematic lineage base/upstream pin mismatch")
    adapter = document["adapter"]
    if not isinstance(adapter, dict) or set(adapter) != {
        "repo", "revision", "path", "sha256", "bytes", "rank", "alpha", "config_sha256"
    }:
        raise CinematicCacheError("cinematic lineage adapter fields mismatch")
    if adapter != {
        "repo": UPSTREAM_REPO,
        "revision": ADAPTER_REVISION,
        "path": ADAPTER_HUB_PATH,
        "sha256": ADAPTER_SHA256,
        "bytes": ADAPTER_BYTES,
        "rank": ADAPTER_RANK,
        "alpha": ADAPTER_ALPHA,
        "config_sha256": ADAPTER_CONFIG_SHA256,
    }:
        raise CinematicCacheError("cinematic lineage adapter pin mismatch")
    merge = document["merge"]
    expected_merge = {
        "formula": "bf16(base_f32 + (B_f32 @ A_f32) * 2)",
        "active_targets": ACTIVE_TARGET_COUNT,
        "skipped_action_targets": SKIPPED_ACTION_TARGET_COUNT,
        "action_conditioning": False,
    }
    if not isinstance(merge, dict) or merge != expected_merge:
        raise CinematicCacheError("cinematic lineage merge contract mismatch")
    index = document["index"]
    if (not isinstance(index, dict) or set(index) != {"basename", "sha256"}
            or index.get("basename") != "index.json"
            or index.get("sha256") != expected_index_sha256
            or not _valid_sha256(index.get("sha256"))):
        raise CinematicCacheError("cinematic lineage candidate index pin mismatch")
    producer = document["producer"]
    if not isinstance(producer, dict) or not producer:
        raise CinematicCacheError("cinematic lineage producer must be a non-empty object")
    if producer.get("skipped_action_targets") != skipped_action_targets():
        raise CinematicCacheError("cinematic lineage must enumerate all 162 skipped action targets")
    if producer.get("active_target_names") != active_target_names():
        raise CinematicCacheError("cinematic lineage active target inventory mismatch")


def build_lineage_document(*, index_sha256: str, producer: dict[str, Any]) -> dict[str, Any]:
    if not _valid_sha256(index_sha256):
        raise CinematicCacheError("candidate index SHA-256 is invalid")
    if not isinstance(producer, dict):
        raise CinematicCacheError("producer evidence must be an object")
    producer_doc = dict(producer)
    producer_doc["active_target_names"] = active_target_names()
    producer_doc["skipped_action_targets"] = skipped_action_targets()
    return {
        "schema": LINEAGE_SCHEMA,
        "upstream_revision": UPSTREAM_REVISION,
        "base_index_sha256": BASE_INDEX_SHA256,
        "base_source_sha256": BASE_SOURCE_SHA256,
        "adapter": {
            "repo": UPSTREAM_REPO,
            "revision": ADAPTER_REVISION,
            "path": ADAPTER_HUB_PATH,
            "sha256": ADAPTER_SHA256,
            "bytes": ADAPTER_BYTES,
            "rank": ADAPTER_RANK,
            "alpha": ADAPTER_ALPHA,
            "config_sha256": ADAPTER_CONFIG_SHA256,
        },
        "merge": {
            "formula": "bf16(base_f32 + (B_f32 @ A_f32) * 2)",
            "active_targets": ACTIVE_TARGET_COUNT,
            "skipped_action_targets": SKIPPED_ACTION_TARGET_COUNT,
            "action_conditioning": False,
        },
        "index": {"basename": "index.json", "sha256": index_sha256},
        "producer": producer_doc,
    }


def create_output_directory(output: Path, *, allowed_root: Path = ARTIFACT_ROOT) -> Path:
    """Reserve a fresh output directory without following output symlinks."""
    if output.name in ("", ".", "..") or output.name.startswith("."):
        raise CinematicCacheError("output directory must have a visible basename")
    if not allowed_root.exists():
        raise CinematicCacheError(f"external artifact root does not exist: {allowed_root}")
    canonical_root = allowed_root.resolve(strict=True)
    parent = output.parent.resolve(strict=True)
    try:
        if os.path.commonpath((str(canonical_root), str(parent))) != str(canonical_root):
            raise CinematicCacheError("output directory must be under the canonical external SSD root")
    except ValueError:
        raise CinematicCacheError("output directory must be under the canonical external SSD root") from None
    canonical_output = parent / output.name
    if os.path.lexists(canonical_output):
        raise FileExistsError(f"refusing to replace existing output directory: {canonical_output}")
    os.mkdir(canonical_output, 0o700)
    info = canonical_output.lstat()
    if not canonical_output.is_dir() or canonical_output.is_symlink():
        raise CinematicCacheError("new output path is not a regular directory")
    if info.st_dev != parent.stat().st_dev:
        raise CinematicCacheError("output directory is not on the external artifact filesystem")
    return canonical_output


def _create_output_directory(output: Path) -> Path:
    return create_output_directory(output, allowed_root=ARTIFACT_ROOT)


def _validate_q4_manifest(path: Path, entry: dict[str, Any], block_index: int) -> tuple[Path, str]:
    raw = _read_regular_bytes(path, label=f"Q4 manifest block {block_index}", max_bytes=MAX_JSON_BYTES)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != entry["manifest_sha256"]:
        raise CinematicCacheError(f"Q4 manifest pin mismatch for block {block_index}")
    manifest = _load_json_bytes(raw, label=f"Q4 manifest block {block_index}")
    if set(manifest) != {
        "schema", "schema_version", "source", "block_index", "geometry", "codec", "payload", "tensors"
    }:
        raise CinematicCacheError(f"Q4 manifest fields mismatch for block {block_index}")
    if (manifest.get("schema") != Q4_MANIFEST_SCHEMA
            or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
            or type(manifest.get("block_index")) is not int
            or manifest["block_index"] != block_index):
        raise CinematicCacheError(f"Q4 manifest identity mismatch for block {block_index}")
    source = manifest["source"]
    if (not isinstance(source, dict) or set(source) != {"sha256", "bytes"}
            or source.get("sha256") != BASE_SOURCE_SHA256
            or type(source.get("bytes")) is not int
            or source.get("bytes") != BASE_SOURCE_BYTES):
        raise CinematicCacheError(f"Q4 manifest source pin mismatch for block {block_index}")
    payload = manifest["payload"]
    expected_payload_basename = path.stem + ".bin"
    if (not isinstance(payload, dict) or set(payload) != {"basename", "nbytes", "sha256"}
            or payload.get("basename") != expected_payload_basename
            or not _safe_basename(payload.get("basename"))
            or type(payload.get("nbytes")) is not int or payload["nbytes"] <= 0
            or payload.get("sha256") != entry["payload_sha256"]):
        raise CinematicCacheError(f"Q4 manifest payload pin mismatch for block {block_index}")
    return path.parent / payload["basename"], actual


def _expected_block_names(source_tensors: Mapping[str, TensorRecord], block_index: int) -> set[str]:
    prefix = f"double_blocks.{block_index}."
    names = {name for name in source_tensors if name.startswith(prefix)}
    if len(names) != BLOCK_TENSOR_COUNT:
        raise CinematicCacheError(f"base F32 block {block_index} does not contain exactly 32 weights")
    return names


def _validate_bf16_block_header(
    header: SafetensorsHeader, source_tensors: Mapping[str, TensorRecord], block_index: int,
) -> None:
    expected = _expected_block_names(source_tensors, block_index)
    if set(header.tensors) != expected or len(header.tensors) != BLOCK_TENSOR_COUNT:
        raise CinematicCacheError(f"base BF16 block {block_index} inventory differs from F32 source")
    for name in expected:
        record = header.tensors[name]
        source = source_tensors[name]
        if record.dtype != "BF16" or record.shape != source.shape:
            raise CinematicCacheError(f"base BF16 block {block_index} dtype/shape mismatch: {name}")


def _validate_source_target_coverage(
    header: SafetensorsHeader, adapter_pairs: Mapping[str, AdapterPair],
) -> None:
    for target in adapter_pairs:
        if not target.startswith("double_blocks."):
            continue
        base = header.tensors.get(target)
        if base is None or base.dtype != "F32":
            raise CinematicCacheError(f"base F32 tensor missing active adapter target: {target}")
        shape = ACTIVE_TARGET_SHAPES[target.split(".", 2)[2][:-7]]
        if base.shape != shape:
            raise CinematicCacheError(f"base F32 active target shape mismatch: {target}")


def _load_base_index(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_bytes(path, label="base Q4 cache index", max_bytes=MAX_JSON_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    document = _load_json_bytes(raw, label="base Q4 cache index")
    validate_base_index_document(document, actual_sha256=digest)
    return document, raw


def _make_preflight(*, verify_cache_assets: bool, deadline_unix: float | None = None) -> Preflight:
    external_snapshots: list[tuple[Path, tuple[int, int, int, int], str]] = []

    def capture(path: Path, label: str) -> tuple[int, int, int, int]:
        snapshot = _file_snapshot(path, label)
        external_snapshots.append((path, snapshot, label))
        return snapshot

    config_snapshot = capture(ADAPTER_CONFIG_PATH, "adapter config")
    config_raw = _read_regular_bytes(ADAPTER_CONFIG_PATH, label="adapter config", max_bytes=16 * 1024)
    _same_file_snapshot(ADAPTER_CONFIG_PATH, config_snapshot, "adapter config")
    config_digest = hashlib.sha256(config_raw).hexdigest()
    config = _load_json_bytes(config_raw, label="adapter config", max_bytes=16 * 1024)
    validate_config_document(config, actual_sha256=config_digest, actual_bytes=len(config_raw))

    source_snapshot = capture(BASE_SOURCE_PATH, "base F32 checkpoint")
    source_header = read_safetensors_header(BASE_SOURCE_PATH, label="base F32 checkpoint")
    if source_header.file_bytes != BASE_SOURCE_BYTES:
        raise CinematicCacheError("base F32 checkpoint byte count mismatch")
    source_header_sha = hashlib.sha256(source_header.raw_header[8:]).hexdigest()
    if source_header_sha != BASE_SOURCE_HEADER_SHA256:
        raise CinematicCacheError("base F32 checkpoint header SHA-256 mismatch")
    validate_base_source_inventory(source_header.tensors)
    base_sha = _sha256_file(BASE_SOURCE_PATH, label="base F32 checkpoint", deadline_unix=deadline_unix)
    if base_sha != BASE_SOURCE_SHA256:
        raise CinematicCacheError("base F32 checkpoint full SHA-256 mismatch")
    _same_file_snapshot(BASE_SOURCE_PATH, source_snapshot, "base F32 checkpoint")

    adapter_snapshot = capture(ADAPTER_PATH, "Cinematic LoRA adapter")
    adapter_header = read_safetensors_header(ADAPTER_PATH, label="Cinematic LoRA adapter")
    if adapter_header.file_bytes != ADAPTER_BYTES:
        raise CinematicCacheError("Cinematic adapter byte count mismatch")
    adapter_sha = _sha256_file(ADAPTER_PATH, label="Cinematic LoRA adapter", deadline_unix=deadline_unix)
    if adapter_sha != ADAPTER_SHA256:
        raise CinematicCacheError("Cinematic adapter full SHA-256 mismatch")
    _same_file_snapshot(ADAPTER_PATH, adapter_snapshot, "Cinematic LoRA adapter")
    adapter_pairs = validate_adapter_inventory(adapter_header.tensors.items())
    _validate_source_target_coverage(source_header, adapter_pairs)

    index_snapshot = capture(BASE_INDEX_PATH, "base Q4 cache index")
    base_index, _ = _load_base_index(BASE_INDEX_PATH)
    _same_file_snapshot(BASE_INDEX_PATH, index_snapshot, "base Q4 cache index")
    block_headers: list[SafetensorsHeader] = []
    manifests: list[tuple[Path, Path, str]] = []
    seen_global_names = {base_index["globals"]["bf16"]}
    for expected_index, entry in enumerate(base_index["blocks"]):
        _check_deadline(deadline_unix, f"preflighting block {expected_index}")
        manifest_path = BASE_CACHE_DIR / entry["manifest"]
        manifest_snapshot = capture(manifest_path, f"Q4 manifest block {expected_index}")
        payload_path, manifest_sha = _validate_q4_manifest(manifest_path, entry, expected_index)
        _same_file_snapshot(manifest_path, manifest_snapshot,
                            f"Q4 manifest block {expected_index}")
        payload_snapshot = capture(payload_path, f"Q4 payload block {expected_index}")
        manifest_payload = _load_json_bytes(
            _read_regular_bytes(manifest_path, label=f"Q4 manifest block {expected_index}",
                                max_bytes=MAX_JSON_BYTES),
            label=f"Q4 manifest block {expected_index}",
        )["payload"]
        if payload_snapshot[2] != manifest_payload["nbytes"]:
            raise CinematicCacheError(f"Q4 payload byte count mismatch for block {expected_index}")
        if verify_cache_assets:
            payload_sha = _sha256_file(payload_path, label=f"Q4 payload block {expected_index}",
                                       deadline_unix=deadline_unix)
            if payload_sha != entry["payload_sha256"]:
                raise CinematicCacheError(f"Q4 payload SHA-256 mismatch for block {expected_index}")
        _same_file_snapshot(payload_path, payload_snapshot,
                            f"Q4 payload block {expected_index}")
        if entry["bf16"] in seen_global_names or not _safe_basename(entry["bf16"]):
            raise CinematicCacheError("Q4 cache block BF16 basename collides with another artifact")
        seen_global_names.add(entry["bf16"])
        block_path = BASE_CACHE_DIR / entry["bf16"]
        block_snapshot = capture(block_path, f"base BF16 block {expected_index}")
        block_header = read_safetensors_header(block_path,
                                               label=f"base BF16 block {expected_index}")
        _validate_bf16_block_header(block_header, source_header.tensors, expected_index)
        _same_file_snapshot(block_path, block_snapshot, f"base BF16 block {expected_index}")
        block_headers.append(block_header)
        manifests.append((manifest_path, payload_path, manifest_sha))

    globals_record = base_index["globals"]
    globals_path = BASE_CACHE_DIR / globals_record["bf16"]
    globals_snapshot = capture(globals_path, "base globals BF16 cache")
    globals_header = read_safetensors_header(globals_path, label="base globals BF16 cache")
    if any(record.dtype != "BF16" for record in globals_header.tensors.values()):
        raise CinematicCacheError("base globals cache contains a non-BF16 tensor")
    if verify_cache_assets:
        globals_sha = _sha256_file(globals_path, label="base globals BF16 cache",
                                   deadline_unix=deadline_unix)
        if globals_sha != globals_record["sha256"]:
            raise CinematicCacheError("base globals BF16 cache SHA-256 mismatch")
    _same_file_snapshot(globals_path, globals_snapshot, "base globals BF16 cache")

    for expected_index, (header, entry) in enumerate(zip(block_headers, base_index["blocks"], strict=True)):
        if header.file_bytes <= 0 or entry["block_index"] != expected_index:
            raise CinematicCacheError("base BF16 block header/index identity mismatch")
    return Preflight(source_header, adapter_header, adapter_pairs, base_index,
                     tuple(block_headers), globals_header, tuple(manifests),
                     tuple(external_snapshots))


def _validate_external_source_snapshots(preflight: Preflight) -> None:
    """Require every hashed/consumed input to retain its preflight file identity."""
    if not preflight.external_snapshots:
        raise CinematicCacheError("preflight source snapshot inventory is empty")
    for path, expected, label in preflight.external_snapshots:
        _same_file_snapshot(path, expected, label)
    if len(preflight.block_headers) != BLOCK_COUNT or len(preflight.adapter_pairs) != ADAPTER_PAIR_COUNT:
        raise CinematicCacheError("preflight inventory changed before MPS execution")


def _expected_source_snapshot(
    snapshots: tuple[tuple[Path, tuple[int, int, int, int], str], ...], path: Path,
) -> tuple[int, int, int, int]:
    for snapshot_path, snapshot, _label in snapshots:
        if snapshot_path == path:
            return snapshot
    raise CinematicCacheError(f"preflight has no source snapshot for {path}")


def _check_mps_device() -> Any:
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if fallback not in (None, "0"):
        raise CinematicCacheError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        import torch
    except ImportError as exc:
        raise CinematicCacheError("cache building requires PyTorch with MPS support") from exc
    try:
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise CinematicCacheError("MPS is required for LoRA merge; CPU fallback is forbidden")
        device = torch.device("mps")
    except (AttributeError, RuntimeError) as exc:
        raise CinematicCacheError(f"MPS runtime is unavailable: {exc}") from None
    return torch, device


def _base_target_to_pair_key(target: str) -> str:
    if not target.endswith(".weight"):
        raise CinematicCacheError(f"LoRA target is not a base weight: {target}")
    return target[:-len(".weight")]


def _build_block_header(
    ordered_names: list[str], records: Mapping[str, TensorRecord], *, block_index: int,
) -> tuple[bytes, int]:
    offset = 0
    document: dict[str, Any] = {}
    for name in ordered_names:
        record = records[name]
        nbytes = _element_count(record.shape) * 2
        document[name] = {"dtype": "BF16", "shape": list(record.shape),
                          "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    document["__metadata__"] = {
        "schema": "hy-nomas-cinematic-bf16-block-v1",
        "base_source_sha256": BASE_SOURCE_SHA256,
        "adapter_sha256": ADAPTER_SHA256,
        "block_index": str(block_index),
    }
    raw = json.dumps(document, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    raw += b" " * ((-len(raw)) % 8)
    if len(raw) > MAX_SAFETENSORS_HEADER_BYTES:
        raise CinematicCacheError(f"cinematic block {block_index} safetensors header is too large")
    return struct.pack("<Q", len(raw)) + raw, offset


def _mapped_tensor(mmap_file: mmap.mmap, header: SafetensorsHeader,
                   record: TensorRecord, torch: Any, dtype: Any) -> tuple[Any, memoryview]:
    start = header.data_start + record.start
    end = header.data_start + record.end
    view = memoryview(mmap_file)[start:end]
    tensor = torch.frombuffer(view, dtype=dtype).reshape(record.shape)
    return tensor, view


def _merge_one_target_to_bf16(
    *, target: str, base_file: mmap.mmap, adapter_file: mmap.mmap,
    source_header: SafetensorsHeader, adapter_header: SafetensorsHeader,
    pair: AdapterPair, torch: Any, device: Any,
) -> bytes:
    base_record = source_header.tensors.get(target)
    if base_record is None or base_record.dtype != "F32":
        raise CinematicCacheError(f"active F32 base weight disappeared: {target}")
    expected_a = (ADAPTER_RANK, base_record.shape[1])
    expected_b = (base_record.shape[0], ADAPTER_RANK)
    if pair.a.shape != expected_a or pair.b.shape != expected_b:
        raise CinematicCacheError(f"LoRA A/B dimensions no longer match base target: {target}")

    base_cpu = a_cpu = b_cpu = base_mps = a_mps = b_mps = delta = output_mps = output_cpu = None
    base_view = a_view = b_view = None
    synchronized = False
    try:
        base_cpu, base_view = _mapped_tensor(base_file, source_header, base_record,
                                             torch, torch.float32)
        a_cpu, a_view = _mapped_tensor(adapter_file, adapter_header, pair.a,
                                       torch, torch.float32)
        b_cpu, b_view = _mapped_tensor(adapter_file, adapter_header, pair.b,
                                       torch, torch.float32)
        with torch.no_grad():
            base_mps = base_cpu.to(device=device, dtype=torch.float32)
            a_mps = a_cpu.to(device=device, dtype=torch.float32)
            b_mps = b_cpu.to(device=device, dtype=torch.float32)
            # Do not keep CPU tensor wrappers or file-backed views alive beyond
            # the ordered MPS copies. The source mappings remain owned by caller.
            del base_cpu, a_cpu, b_cpu
            base_cpu = a_cpu = b_cpu = None
            base_view.release()
            a_view.release()
            b_view.release()
            base_view = a_view = b_view = None
            delta = torch.matmul(b_mps, a_mps)
            delta.mul_(MERGE_SCALE)
            base_mps.add_(delta)
            output_mps = base_mps.to(dtype=torch.bfloat16).contiguous()
            torch.mps.synchronize()
            synchronized = True
            if bool(torch.logical_not(torch.isfinite(output_mps)).any().to(device="cpu").item()):
                raise CinematicCacheError(f"merged BF16 tensor is non-finite: {target}")
            output_cpu = output_mps.to(device="cpu").contiguous()
            if output_cpu.dtype != torch.bfloat16 or tuple(output_cpu.shape) != base_record.shape:
                raise CinematicCacheError(f"merged BF16 output metadata changed: {target}")
            raw = output_cpu.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
            expected_bytes = _element_count(base_record.shape) * 2
            if len(raw) != expected_bytes:
                raise CinematicCacheError(f"merged BF16 byte extent mismatch: {target}")
            return raw
    except CinematicCacheError:
        raise
    except Exception as exc:
        raise CinematicCacheError(
            f"MPS F32 LoRA merge failed for {target}: {type(exc).__name__}: {exc}"
        ) from None
    finally:
        for view in (base_view, a_view, b_view):
            if view is not None:
                try:
                    view.release()
                except ValueError:
                    pass
        del base_cpu, a_cpu, b_cpu, base_mps, a_mps, b_mps, delta, output_mps, output_cpu
        try:
            if not synchronized:
                torch.mps.synchronize()
            torch.mps.empty_cache()
        except Exception as exc:
            raise CinematicCacheError(
                f"MPS tensor cleanup could not complete for {target}: {type(exc).__name__}: {exc}"
            ) from None


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise CinematicCacheError(f"could not fsync output directory metadata: {path}: {exc}") from None


def _link_noreplace(source: Path, destination: Path, *, label: str) -> None:
    if destination.is_symlink() or os.path.lexists(destination):
        raise FileExistsError(f"refusing to replace {label}: {destination}")
    if source.is_symlink() or not source.is_file():
        raise CinematicCacheError(f"{label} source must be a regular non-symlink file: {source}")
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError:
        raise FileExistsError(f"refusing to replace {label}: {destination}") from None
    except OSError as exc:
        raise CinematicCacheError(f"could not hard-link {label}: {exc}") from None
    source_stat = source.stat(follow_symlinks=False)
    destination_stat = destination.lstat()
    if (destination.is_symlink() or not destination.is_file()
            or source_stat.st_dev != destination_stat.st_dev
            or source_stat.st_ino != destination_stat.st_ino):
        destination.unlink(missing_ok=True)
        raise CinematicCacheError(f"hard-linked {label} is not the same regular sibling")


def _publish_bytes_noreplace(path: Path, raw: bytes, *, label: str) -> None:
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    created_temp = False
    try:
        with temp_path.open("xb") as stream:
            created_temp = True
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _link_noreplace(temp_path, path, label=label)
    finally:
        if created_temp:
            temp_path.unlink(missing_ok=True)


def _validate_output_candidate(
    output: Path, index_document: dict[str, Any], index_bytes: bytes,
    index_sha256: str, lineage: dict[str, Any], lineage_bytes: bytes,
    base_index: dict[str, Any], block_hashes: list[str], created: set[str],
) -> None:
    validate_cache_index_document(index_document, expected_source_sha256=BASE_SOURCE_SHA256,
                                  expected_source_bytes=BASE_SOURCE_BYTES)
    if hashlib.sha256(index_bytes).hexdigest() != index_sha256:
        raise CinematicCacheError("candidate index bytes changed after hash")
    if len(block_hashes) != BLOCK_COUNT or any(not _valid_sha256(value) for value in block_hashes):
        raise CinematicCacheError("candidate block BF16 pin inventory is incomplete")
    for index, (entry, base_entry) in enumerate(zip(index_document["blocks"],
                                                    base_index["blocks"], strict=True)):
        if (entry["manifest"] != base_entry["manifest"]
                or entry["manifest_sha256"] != base_entry["manifest_sha256"]
                or entry["payload_sha256"] != base_entry["payload_sha256"]
                or entry["bf16"] != f"block-{index}-cinematic-bf16.safetensors"
                or entry["bf16_sha256"] != block_hashes[index]):
            raise CinematicCacheError(f"candidate index does not preserve pinned Q4 ancestry at block {index}")
    if index_document["globals"] != base_index["globals"]:
        raise CinematicCacheError("candidate index globals must remain the exact pinned base control")
    validate_lineage_document(lineage, expected_index_sha256=index_sha256)
    if hashlib.sha256(lineage_bytes).hexdigest() != hashlib.sha256(
            (json.dumps(lineage, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest():
        raise CinematicCacheError("candidate lineage serialization changed")
    for basename in created:
        item = output / basename
        if item.is_symlink() or not item.is_file():
            raise CinematicCacheError(f"candidate output contains a non-regular artifact: {basename}")


def _cleanup_owned_output(output: Path, owned_names: set[str], identity: tuple[int, int]) -> None:
    try:
        info = output.lstat()
    except OSError:
        return
    if output.is_symlink() or not output.is_dir() or (info.st_dev, info.st_ino) != identity:
        return
    for name in sorted(owned_names, reverse=True):
        if _safe_basename(name):
            (output / name).unlink(missing_ok=True)
    try:
        output.rmdir()
        _fsync_directory(output.parent)
    except OSError:
        # Keep anything that appeared concurrently; cleanup owns only paths it created.
        pass


def _copy_base_q4_siblings(
    output: Path, preflight: Preflight, *, deadline_unix: float | None,
    owned_names: set[str],
) -> None:
    for block_index, (manifest_path, payload_path, _) in enumerate(preflight.q4_manifests):
        _check_deadline(deadline_unix, f"linking Q4 cache block {block_index}")
        for source in (manifest_path, payload_path):
            destination = output / source.name
            _link_noreplace(source, destination, label=f"Q4 cache block {block_index} sibling")
            owned_names.add(source.name)
    globals_name = preflight.base_index["globals"]["bf16"]
    source = BASE_CACHE_DIR / globals_name
    _link_noreplace(source, output / globals_name, label="base global BF16 cache")
    owned_names.add(globals_name)
    _fsync_directory(output)


def _producer_evidence() -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    try:
        relative_script = script_path.relative_to(ROOT).as_posix()
    except ValueError:
        relative_script = script_path.name
    return {
        "path": relative_script,
        "source_sha256": _sha256_file(script_path, label="cache builder source"),
        "upstream_merge_script": "scripts/train/merge_lora.py",
        "upstream_merge_script_sha256": UPSTREAM_MERGE_SCRIPT_SHA256,
        "upstream_lora_key_normalization": (
            "strip exact .lora_A.weight or .lora_B.weight; replace "
            "._checkpoint_wrapped_module. with .; remove _checkpoint_wrapped_module.; append .weight"
        ),
        "device": "mps",
        "base_dtype": "F32",
        "adapter_dtype": "F32",
        "merge_compute_dtype": "F32",
        "output_dtype": "BF16",
        "scale": MERGE_SCALE,
        "adapter_pairs": ADAPTER_PAIR_COUNT,
        "active_double_block_pairs": ACTIVE_TARGET_COUNT,
        "skipped_action_pairs": SKIPPED_ACTION_TARGET_COUNT,
        "block_count": BLOCK_COUNT,
        "block_tensor_count": BLOCK_TENSOR_COUNT,
        "copy_policy": (
            "Q4 manifests, Q4 payloads and globals are hard-linked from the pinned base cache; "
            "unadapted block BF16 payloads are copied byte-for-byte"
        ),
        "model_execution": False,
        "skipped_action_reason": "action conditioning is disabled on the admitted camera-off route",
    }


def _plan_summary(preflight: Preflight) -> dict[str, Any]:
    block_output_bytes = sum(header.file_bytes for header in preflight.block_headers)
    linked_bytes = sum(path.stat().st_size for pair in preflight.q4_manifests for path in pair[:2])
    globals_bytes = (BASE_CACHE_DIR / preflight.base_index["globals"]["bf16"]).stat().st_size
    return {
        "status": "plan_validated",
        "base_source_sha256": BASE_SOURCE_SHA256,
        "base_index_sha256": BASE_INDEX_SHA256,
        "adapter_sha256": ADAPTER_SHA256,
        "adapter_config_sha256": ADAPTER_CONFIG_SHA256,
        "source_tensor_count": SOURCE_TENSOR_COUNT,
        "adapter_pair_count": ADAPTER_PAIR_COUNT,
        "active_target_count": ACTIVE_TARGET_COUNT,
        "skipped_action_target_count": SKIPPED_ACTION_TARGET_COUNT,
        "block_count": BLOCK_COUNT,
        "block_tensor_count": BLOCK_TENSOR_COUNT,
        "output_block_bf16_bytes": block_output_bytes,
        "hard_linked_q4_manifest_payload_bytes": linked_bytes,
        "hard_linked_globals_bytes": globals_bytes,
        "device_touched": False,
    }


def _write_bf16_block(
    *, output: Path, block_index: int, source_header: SafetensorsHeader,
    base_bf16_header: SafetensorsHeader, adapter_header: SafetensorsHeader,
    adapter_pairs: Mapping[str, AdapterPair], base_index: dict[str, Any],
    torch: Any, device: Any, deadline_unix: float | None,
    owned_names: set[str],
    external_snapshots: tuple[tuple[Path, tuple[int, int, int, int], str], ...],
) -> str:
    """Block implementation with all metadata explicit for guarded callers."""
    source_bf16_path = base_bf16_header.path
    base_record = base_index["blocks"][block_index]
    expected_base_file_sha = base_record["bf16_sha256"]
    source_stat = _expected_source_snapshot(external_snapshots, source_bf16_path)
    _same_file_snapshot(source_bf16_path, source_stat, f"base BF16 block {block_index}")
    output_name = f"block-{block_index}-cinematic-bf16.safetensors"
    output_path = output / output_name
    temp_path = output / f".{output_name}.{uuid.uuid4().hex}.partial"
    base_map = adapter_map = source_map = None
    source_stream: BinaryIO | None = None
    base_stream: BinaryIO | None = None
    adapter_stream: BinaryIO | None = None
    created_temp = False
    try:
        source_stream = source_bf16_path.open("rb")
        opened = os.fstat(source_stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != source_stat:
            raise CinematicCacheError(f"base BF16 block {block_index} changed before reading")
        source_map = mmap.mmap(source_stream.fileno(), 0, access=mmap.ACCESS_COPY)
        ordered_names = [
            name for name, _ in sorted(base_bf16_header.tensors.items(), key=lambda item: item[1].start)
        ]
        output_header, expected_payload_bytes = _build_block_header(
            ordered_names, base_bf16_header.tensors, block_index=block_index,
        )
        output_hash = hashlib.sha256()
        source_hash = hashlib.sha256(base_bf16_header.header_prefix)
        with temp_path.open("xb") as stream:
            created_temp = True
            stream.write(output_header)
            output_hash.update(output_header)
            for name in ordered_names:
                _check_deadline(deadline_unix, f"merging block {block_index} tensor {name}")
                record = base_bf16_header.tensors[name]
                start = base_bf16_header.data_start + record.start
                end = base_bf16_header.data_start + record.end
                source_view = memoryview(source_map)[start:end]
                source_hash.update(source_view)
                pair = adapter_pairs.get(name)
                if pair is not None:
                    if not pair.active:
                        raise CinematicCacheError(f"camera-off action target reached BF16 block merge: {name}")
                    if base_map is None:
                        base_stream = BASE_SOURCE_PATH.open("rb")
                        base_expected = _expected_source_snapshot(external_snapshots, BASE_SOURCE_PATH)
                        base_opened = os.fstat(base_stream.fileno())
                        if (base_opened.st_dev, base_opened.st_ino, base_opened.st_size,
                                base_opened.st_mtime_ns) != base_expected:
                            raise CinematicCacheError("base F32 checkpoint changed before MPS mapping")
                        base_map = mmap.mmap(base_stream.fileno(), 0, access=mmap.ACCESS_COPY)
                    if adapter_map is None:
                        adapter_stream = ADAPTER_PATH.open("rb")
                        adapter_expected = _expected_source_snapshot(external_snapshots, ADAPTER_PATH)
                        adapter_opened = os.fstat(adapter_stream.fileno())
                        if (adapter_opened.st_dev, adapter_opened.st_ino, adapter_opened.st_size,
                                adapter_opened.st_mtime_ns) != adapter_expected:
                            raise CinematicCacheError("Cinematic adapter changed before MPS mapping")
                        adapter_map = mmap.mmap(adapter_stream.fileno(), 0, access=mmap.ACCESS_COPY)
                    output_bytes = _merge_one_target_to_bf16(
                        target=name, base_file=base_map, adapter_file=adapter_map,
                        source_header=source_header, adapter_header=adapter_header,
                        pair=pair, torch=torch, device=device,
                    )
                else:
                    output_bytes = source_view
                expected_nbytes = _element_count(record.shape) * 2
                if len(output_bytes) != expected_nbytes:
                    source_view.release()
                    raise CinematicCacheError(f"output BF16 extent mismatch: {name}")
                stream.write(output_bytes)
                output_hash.update(output_bytes)
                if output_bytes is not source_view:
                    del output_bytes
                source_view.release()
            stream.flush()
            os.fsync(stream.fileno())
        if source_hash.hexdigest() != expected_base_file_sha:
            raise CinematicCacheError(f"base BF16 block {block_index} SHA-256 mismatch")
        if os.stat(temp_path).st_size != len(output_header) + expected_payload_bytes:
            raise CinematicCacheError(f"output BF16 block {block_index} byte count mismatch")
        _same_file_snapshot(source_bf16_path, source_stat, f"base BF16 block {block_index}")
        if os.path.lexists(output_path):
            raise FileExistsError(f"refusing to replace candidate block cache: {output_path}")
        os.link(temp_path, output_path, follow_symlinks=False)
        owned_names.add(output_name)
        temp_path.unlink()
        created_temp = False
        _fsync_directory(output)
        written_header = read_safetensors_header(output_path, label=f"candidate BF16 block {block_index}")
        if (set(written_header.tensors) != set(base_bf16_header.tensors)
                or any(written_header.tensors[name].dtype != "BF16"
                       or written_header.tensors[name].shape != base_bf16_header.tensors[name].shape
                       for name in ordered_names)):
            raise CinematicCacheError(f"candidate BF16 block {block_index} header validation failed")
        return output_hash.hexdigest()
    except OSError as exc:
        raise CinematicCacheError(f"block {block_index} cache write failed: {exc}") from None
    finally:
        if created_temp:
            temp_path.unlink(missing_ok=True)
        if source_map is not None:
            source_map.close()
        if base_map is not None:
            base_map.close()
        if adapter_map is not None:
            adapter_map.close()
        for stream in (source_stream, base_stream, adapter_stream):
            if stream is not None:
                stream.close()


def build_cache(
    output_arg: Path, *, deadline_unix: float | None = None,
) -> dict[str, Any]:
    output = _output_path_under_artifact_root(output_arg)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to replace existing output directory: {output}")
    preflight = _make_preflight(verify_cache_assets=True, deadline_unix=deadline_unix)
    _validate_external_source_snapshots(preflight)
    producer = _producer_evidence()
    producer_path = Path(__file__).resolve()
    producer_snapshot = _file_snapshot(producer_path, "cache builder source")
    if producer["source_sha256"] != _sha256_file(producer_path, label="cache builder source"):
        raise CinematicCacheError("cache builder source changed while freezing producer evidence")
    _same_file_snapshot(producer_path, producer_snapshot, "cache builder source")
    torch, device = _check_mps_device()
    output = _create_output_directory(output)
    output_stat = output.lstat()
    output_identity = (output_stat.st_dev, output_stat.st_ino)
    owned_names: set[str] = set()
    try:
        _copy_base_q4_siblings(output, preflight, deadline_unix=deadline_unix,
                               owned_names=owned_names)
        block_hashes: list[str] = []
        for index, block_header in enumerate(preflight.block_headers):
            _check_deadline(deadline_unix, f"starting block {index}")
            block_sha = _write_bf16_block(
                output=output, block_index=index,
                source_header=preflight.source_header,
                base_bf16_header=block_header,
                adapter_header=preflight.adapter_header,
                adapter_pairs=preflight.adapter_pairs,
                base_index=preflight.base_index,
                torch=torch, device=device, deadline_unix=deadline_unix,
                owned_names=owned_names,
                external_snapshots=preflight.external_snapshots,
            )
            block_hashes.append(block_sha)
            sys.stderr.write(f"packed Cinematic block {index + 1}/{BLOCK_COUNT}\n")

        _validate_external_source_snapshots(preflight)
        _same_file_snapshot(producer_path, producer_snapshot, "cache builder source")
        candidate_index = json.loads(json.dumps(preflight.base_index))
        for index, entry in enumerate(candidate_index["blocks"]):
            entry["bf16"] = f"block-{index}-cinematic-bf16.safetensors"
            entry["bf16_sha256"] = block_hashes[index]
        validate_cache_index_document(candidate_index,
                                      expected_source_sha256=BASE_SOURCE_SHA256,
                                      expected_source_bytes=BASE_SOURCE_BYTES)
        index_bytes = (json.dumps(candidate_index, indent=2, sort_keys=True) + "\n").encode("utf-8")
        index_sha = hashlib.sha256(index_bytes).hexdigest()
        lineage = build_lineage_document(index_sha256=index_sha, producer=producer)
        lineage_bytes = (json.dumps(lineage, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _validate_output_candidate(output, candidate_index, index_bytes, index_sha,
                                   lineage, lineage_bytes, preflight.base_index,
                                   block_hashes, owned_names)

        _check_deadline(deadline_unix, "candidate index publication")
        _validate_external_source_snapshots(preflight)
        _same_file_snapshot(producer_path, producer_snapshot, "cache builder source")
        _publish_bytes_noreplace(output / "index.json", index_bytes, label="candidate index")
        owned_names.add("index.json")
        _fsync_directory(output)
        published_index = _read_regular_bytes(
            output / "index.json", label="candidate index", max_bytes=MAX_JSON_BYTES,
        )
        if hashlib.sha256(published_index).hexdigest() != index_sha:
            raise CinematicCacheError("published candidate index readback SHA-256 mismatch")
        published_index_doc = _load_json_bytes(published_index, label="published candidate index")
        validate_cache_index_document(
            published_index_doc, expected_source_sha256=BASE_SOURCE_SHA256,
            expected_source_bytes=BASE_SOURCE_BYTES,
        )
        _validate_external_source_snapshots(preflight)
        _same_file_snapshot(producer_path, producer_snapshot, "cache builder source")
        _publish_bytes_noreplace(output / "cinematic-lineage.json", lineage_bytes,
                                 label="cinematic lineage")
        owned_names.add("cinematic-lineage.json")
        _fsync_directory(output)
        published_lineage = _read_regular_bytes(
            output / "cinematic-lineage.json", label="cinematic lineage",
            max_bytes=MAX_JSON_BYTES,
        )
        if hashlib.sha256(published_lineage).hexdigest() != hashlib.sha256(lineage_bytes).hexdigest():
            raise CinematicCacheError("published cinematic lineage readback SHA-256 mismatch")
        validate_lineage_document(
            _load_json_bytes(published_lineage, label="published cinematic lineage"),
            expected_index_sha256=index_sha,
        )
        return {
            "status": "created",
            "output_dir": str(output),
            "index": str(output / "index.json"),
            "index_sha256": index_sha,
            "lineage": str(output / "cinematic-lineage.json"),
            "lineage_sha256": hashlib.sha256(lineage_bytes).hexdigest(),
            "blocks": BLOCK_COUNT,
            "merged_active_targets": ACTIVE_TARGET_COUNT,
            "skipped_action_targets": SKIPPED_ACTION_TARGET_COUNT,
            "action_conditioning": False,
        }
    except Exception:
        # Only paths linked by this invocation are removed. Any path that
        # existed before reservation is rejected and never enters this set.
        _cleanup_owned_output(output, owned_names, output_identity)
        raise
    finally:
        try:
            torch.mps.synchronize()
            torch.mps.empty_cache()
        except Exception:
            # The build result is already rejected if a tensor merge's cleanup
            # failed; this final idempotent purge never publishes a new cache.
            pass
        del preflight, torch, device


def _output_path_under_artifact_root(output: Path) -> Path:
    if not ARTIFACT_ROOT.exists():
        raise CinematicCacheError(f"external artifact root is unavailable: {ARTIFACT_ROOT}")
    canonical_root = ARTIFACT_ROOT.resolve(strict=True)
    parent = output.parent.resolve(strict=True)
    try:
        if os.path.commonpath((str(canonical_root), str(parent))) != str(canonical_root):
            raise CinematicCacheError("output directory must be under the canonical external SSD root")
    except ValueError:
        raise CinematicCacheError("output directory must be under the canonical external SSD root") from None
    canonical = parent / output.name
    if output.name in ("", ".", "..") or output.name.startswith("."):
        raise CinematicCacheError("output directory must have a visible basename")
    if os.path.lexists(canonical):
        raise FileExistsError(f"refusing to replace existing output directory: {canonical}")
    return canonical


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="fresh output directory under /opt/mldata_ssd/cogni-ml")
    parser.add_argument("--plan-only", action="store_true",
                        help="authenticate pinned source/config/index/header inventory without importing PyTorch or MPS")
    parser.add_argument("--deadline-unix", type=float,
                        help="optional absolute Unix deadline supplied by the parent run guard")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.deadline_unix is not None and args.deadline_unix <= 0:
            raise CinematicCacheError("--deadline-unix must be a positive absolute Unix timestamp")
        if args.plan_only:
            preflight = _make_preflight(verify_cache_assets=False,
                                        deadline_unix=args.deadline_unix)
            print(json.dumps(_plan_summary(preflight), indent=2, sort_keys=True))
            return 0
        if args.output_dir is None:
            raise CinematicCacheError("--output-dir is required unless --plan-only is selected")
        result = build_cache(args.output_dir, deadline_unix=args.deadline_unix)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (CinematicCacheError, FileExistsError) as exc:
        print(f"hy_nomas_cinematic_cache: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
