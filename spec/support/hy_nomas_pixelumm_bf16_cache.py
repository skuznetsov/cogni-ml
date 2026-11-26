#!/usr/bin/env python3
"""Prepare and authenticate a bounded PixelUMM BF16 startup weight cache.

Only decoder blocks zero and one, zero through three, zero through seven,
zero through fifteen, or zero through thirty-five are loaded. F32 checkpoint weights are moved to MPS and cast with PyTorch's
BF16 conversion before their raw little-endian uint16 words are persisted.
The cache is a startup artifact, not an inference timing path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Mapping

import torch

import hy_nomas_pixelumm_checkpoint as checkpoint
import hy_nomas_pixelumm_reference as reference

SCHEMA = "cogni_ml.pixelumm_bf16_weight_cache/v1"
ROOT = checkpoint.ROOT
CACHE_DIR = (ROOT / "source").resolve()
SIXTEEN_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261027-sixteen")
SIXTEEN_CACHE_DIR = (SIXTEEN_ROOT / "source").resolve()
THIRTY_SIX_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-36-20261027")
THIRTY_SIX_CACHE_DIR = (THIRTY_SIX_ROOT / "source").resolve()
UPSTREAM_MODEL_SHA256 = "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec"
CONVERSION = {
    "method": "torch.Tensor.to(torch.bfloat16)",
    "device": "mps",
    "rounding": "round_to_nearest_even",
}
CHUNK_BYTES = 8 * 1024 * 1024
MAX_CACHE_PAYLOAD_BYTES = 16 * 1024 * 1024 * 1024


class CacheError(ValueError):
    """A cache, source receipt, or cache destination did not pass validation."""


def _digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _shape_bytes(shape: object, width: int) -> int:
    if (not isinstance(shape, list) or not shape
            or any(type(value) is not int or value <= 0 for value in shape)):
        raise CacheError("invalid tensor shape in cache manifest")
    product = 1
    for value in shape:
        product *= value
    return product * width


def _tensor_record(stream, aggregate, key: str, tensor: torch.Tensor, offset: int):
    if tensor.dtype != torch.bfloat16 or tensor.numel() == 0:
        raise CacheError(f"cache tensor must be nonempty BF16: {key}")
    cpu = tensor.detach().contiguous().cpu()
    array = cpu.view(torch.uint8).reshape(-1).numpy()
    view = memoryview(array).cast("B")
    nbytes = len(view)
    if nbytes != cpu.numel() * 2 or nbytes != _shape_bytes(list(cpu.shape), 2):
        raise CacheError(f"unexpected BF16 byte extent: {key}")
    record_hash = hashlib.sha256()
    raw_hash = hashlib.sha256()
    cursor = 0
    while cursor < nbytes:
        end = min(cursor + CHUNK_BYTES, nbytes)
        chunk = view[cursor:end]
        stream.write(chunk)
        aggregate.update(chunk)
        record_hash.update(chunk)
        raw_hash.update(chunk)
        cursor = end
    stream.flush()

    # Read back each serialized range in bounded chunks and compare it with
    # the source-loaded MPS parameter's raw uint16 view, without a second
    # tensor-sized CPU allocation.
    stream.seek(offset)
    cursor = 0
    equal = True
    while cursor < nbytes:
        end = min(cursor + CHUNK_BYTES, nbytes)
        expected = view[cursor:end]
        observed = stream.read(end - cursor)
        if len(observed) != end - cursor or observed != expected:
            equal = False
        cursor = end
    stream.seek(offset + nbytes)
    source_sha = raw_hash.hexdigest()
    payload_sha = record_hash.hexdigest()
    record = {
        "dtype": "BF16", "shape": list(cpu.shape), "offset": offset,
        "nbytes": nbytes, "sha256": payload_sha,
    }
    comparison = {
        "shape": list(cpu.shape),
        "source_raw_uint16_sha256": source_sha,
        "payload_sha256": payload_sha,
        "raw_uint16_equal": equal and source_sha == payload_sha,
        "payload_bytes_equal": equal,
    }
    del view, array, cpu
    return record, comparison


def _expected_shapes(blocks: Mapping[int, Mapping[str, tuple[int, ...]]] | None = None):
    if blocks is None:
        schemas = {index: checkpoint.block_schema(index) for index in (0, 1)}
    else:
        schemas = blocks
    count = len(schemas)
    if count not in {2, 4, 8, 16, 36} or set(schemas) != set(range(count)):
        raise CacheError("cache requires consecutive block schemas 0-1, 0-3, 0-7, 0-15, or 0-35")
    result = {}
    for index in range(count):
        expected = schemas[index]
        if not isinstance(expected, Mapping) or len(expected) != 13:
            raise CacheError(f"block {index} schema must contain 13 weights")
        for name, shape in expected.items():
            if not isinstance(name, str) or not name or not isinstance(shape, (list, tuple)):
                raise CacheError("invalid block cache schema entry")
            result[f"block{index}.{name}"] = list(shape)
    return result


def _source_dir_for_block(block_count: int, index: int) -> Path:
    """Return the immutable source root for one admitted cache block."""
    if (type(block_count) is not int or block_count not in {2, 4, 8, 16, 36}
            or type(index) is not int or not 0 <= index < block_count):
        raise CacheError("cache block source route is outside an admitted prefix")
    if block_count == 36 and index >= 16:
        return THIRTY_SIX_CACHE_DIR
    return CACHE_DIR


def validate_manifest(
    manifest: object,
    payload_path: Path | str,
    *,
    expected_shapes: Mapping[str, list[int]] | None = None,
    require_pinned_identity: bool = True,
    payload_basename: str | None = None,
) -> dict:
    """Validate cache metadata, tensor ownership and every byte/hash binding.

    ``expected_shapes`` is injectable only to keep the structural unit tests
    tiny. Production callers use the pinned 26-, 52-, 104-, or 468-tensor schema.
    """
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise CacheError("unsupported BF16 cache schema")
    if sys.byteorder != "little":
        raise CacheError("BF16 safetensors-compatible cache requires little-endian host bytes")
    blocks = manifest.get("blocks") if isinstance(manifest, dict) else None
    valid_block_sets = (
        {"0", "1"},
        {"0", "1", "2", "3"},
        {str(index) for index in range(8)},
        {str(index) for index in range(16)},
        {str(index) for index in range(36)},
    )
    if not isinstance(blocks, dict) or set(blocks) not in valid_block_sets:
        raise CacheError("BF16 cache must authenticate blocks 0-1, 0-3, 0-7, 0-15, or 0-35 exactly")
    block_count = len(blocks)
    if expected_shapes is None:
        expected_shapes = _expected_shapes({
            index: checkpoint.block_schema(index) for index in range(block_count)
        })
    else:
        expected_shapes = dict(expected_shapes)
    expected_keys = set(expected_shapes)
    expected_prefixes = {f"block{index}." for index in range(block_count)}
    if (len(expected_keys) != block_count * 13 or len(expected_shapes) != block_count * 13
            or any(sum(key.startswith(prefix) for key in expected_keys) != 13
                   for prefix in expected_prefixes)
        or any(not any(key.startswith(prefix) for prefix in expected_prefixes)
                   for key in expected_keys)):
        raise CacheError(f"cache schema must describe exactly {block_count * 13} block-owned tensors")
    expected_cache_bytes = sum(_shape_bytes(shape, 2) for shape in expected_shapes.values())
    if expected_cache_bytes > MAX_CACHE_PAYLOAD_BYTES:
        raise CacheError("BF16 cache exceeds the 16 GiB payload cap")

    if require_pinned_identity:
        identities = {
            "upstream_revision": reference.UPSTREAM_REVISION,
            "upstream_model_path": reference.UPSTREAM_MODEL_PATH,
            "source_commit": checkpoint.SOURCE_COMMIT,
            "hf_revision": checkpoint.REVISION,
            "hf_source_path": checkpoint.SOURCE_PATH,
            "checkpoint_header_sha256": checkpoint.HEADER_SHA256,
            "stored_dtype": "F32", "runtime_dtype": "BF16",
            "full_file_hash_verified": False,
        }
        for field, expected in identities.items():
            if manifest.get(field) != expected:
                raise CacheError(f"BF16 cache pinned identity mismatch: {field}")
        if (type(manifest.get("source_file_bytes")) is not int
                or manifest["source_file_bytes"] != checkpoint.SOURCE_BYTES):
            raise CacheError("BF16 cache source file size mismatch")
        if (manifest.get("upstream_model_sha256") != UPSTREAM_MODEL_SHA256
                or manifest.get("python_version") != platform.python_version()):
            raise CacheError("BF16 cache upstream/runtime identity is malformed")

    conversion = manifest.get("conversion")
    if not isinstance(conversion, dict):
        raise CacheError("BF16 cache is missing conversion provenance")
    for field, expected in CONVERSION.items():
        if conversion.get(field) != expected:
            raise CacheError(f"BF16 cache conversion mismatch: {field}")
    torch_version = conversion.get("torch_version")
    if not isinstance(torch_version, str) or not torch_version:
        raise CacheError("BF16 cache torch version is missing")

    for index in range(block_count):
        block = blocks[str(index)]
        prefix = f"block{index}."
        keys = sorted(key for key in expected_keys if key.startswith(prefix))
        if not isinstance(block, dict):
            raise CacheError("invalid BF16 cache block entry")
        if (type(block.get("selected_block")) is not int
                or block.get("selected_block") != index
                or block.get("weight_keys") != keys
                or block.get("bf16_bytes") != sum(_shape_bytes(expected_shapes[k], 2) for k in keys)
                or not _digest(block.get("subset_sha256"))
                or not _digest(block.get("source_receipt_sha256"))):
            raise CacheError(f"BF16 cache block {index} source/ownership mismatch")
        if require_pinned_identity:
            if block.get("subset_file") != f"block{index}-f32.safetensors":
                raise CacheError(f"BF16 cache block {index} subset filename mismatch")
            if block.get("source_receipt_file") != f"block{index}-f32.receipt.json":
                raise CacheError(f"BF16 cache block {index} receipt filename mismatch")
            if type(block.get("subset_bytes")) is not int or block["subset_bytes"] < 8:
                raise CacheError("BF16 cache source subset byte count is malformed")
            extent, chunks = block.get("source_extent"), block.get("source_range_chunks")
            if (not isinstance(extent, list) or len(extent) != 2
                    or any(type(value) is not int for value in extent)
                    or extent[1] - extent[0] != block["bf16_bytes"] * 2
                    or block.get("source_f32_bytes") != block["bf16_bytes"] * 2
                    or not isinstance(chunks, list) or not chunks):
                raise CacheError(f"BF16 cache block {index} source range is malformed")
            expected_chunks = []
            for start in range(extent[0], extent[1], 64 * 1024 * 1024):
                end = min(start + 64 * 1024 * 1024, extent[1])
                expected_chunks.append((
                    start + 8 + checkpoint.HEADER_BYTES,
                    end + 8 + checkpoint.HEADER_BYTES - 1,
                ))
            if len(chunks) != len(expected_chunks):
                raise CacheError(f"BF16 cache block {index} source chunk count mismatch")
            for chunk, (start, end_inclusive) in zip(chunks, expected_chunks):
                if (not isinstance(chunk, dict) or type(chunk.get("start")) is not int
                        or type(chunk.get("end_inclusive")) is not int
                        or chunk.get("start") != start
                        or chunk.get("end_inclusive") != end_inclusive
                        or not _digest(chunk.get("sha256"))):
                    raise CacheError(f"BF16 cache block {index} source chunk is malformed")

    weights = manifest.get("weights")
    if not isinstance(weights, dict) or set(weights) != expected_keys:
        raise CacheError("BF16 cache tensor ownership mismatch")
    comparison = manifest.get("comparison")
    if (not isinstance(comparison, dict) or type(comparison.get("tensor_count")) is not int
            or comparison.get("tensor_count") != block_count * 13
            or comparison.get("all_raw_uint16_equal") is not True
            or comparison.get("method") != "source MPS BF16 tensor raw uint16 bytes equal persisted payload"):
        raise CacheError("BF16 source-versus-cache comparison did not pass")
    comparisons = comparison.get("weights")
    if not isinstance(comparisons, dict) or set(comparisons) != expected_keys:
        raise CacheError("BF16 cache per-tensor comparison inventory mismatch")

    path = Path(payload_path)
    if not path.is_file() or path.is_symlink():
        raise CacheError("BF16 cache payload is not a regular file")
    if manifest.get("payload") != (payload_basename or path.name):
        raise CacheError("BF16 cache payload basename mismatch")
    payload_bytes = path.stat().st_size
    if payload_bytes > MAX_CACHE_PAYLOAD_BYTES:
        raise CacheError("BF16 cache payload exceeds the 16 GiB cap")
    if type(manifest.get("payload_bytes")) is not int or manifest["payload_bytes"] != payload_bytes:
        raise CacheError("BF16 cache payload byte count mismatch")
    payload_hash = manifest.get("payload_sha256")
    if not _digest(payload_hash):
        raise CacheError("BF16 cache payload digest is malformed")

    # The producer serializes one numeric block at a time. Lexicographic key
    # ordering would place block10 before block2 and reject a valid 16-block
    # payload despite its contiguous, authenticated ranges.
    ordered_keys = [
        key
        for index in range(block_count)
        for key in sorted(key for key in expected_keys if key.startswith(f"block{index}."))
    ]
    records = []
    cursor = 0
    for key in ordered_keys:
        record = weights[key]
        if not isinstance(record, dict):
            raise CacheError(f"invalid BF16 cache record: {key}")
        shape = expected_shapes[key]
        nbytes = _shape_bytes(shape, 2)
        if (record.get("dtype") != "BF16" or record.get("shape") != shape
                or record.get("offset") != cursor or record.get("nbytes") != nbytes
                or not _digest(record.get("sha256"))):
            raise CacheError(f"BF16 cache tensor record mismatch: {key}")
        item = comparisons[key]
        if (not isinstance(item, dict) or item.get("shape") != shape
                or item.get("source_raw_uint16_sha256") != record["sha256"]
                or item.get("payload_sha256") != record["sha256"]
                or item.get("raw_uint16_equal") is not True
                or item.get("payload_bytes_equal") is not True):
            raise CacheError(f"BF16 cache raw-word comparison mismatch: {key}")
        records.append((key, cursor, nbytes, record["sha256"]))
        cursor += nbytes
    if cursor != payload_bytes:
        raise CacheError("BF16 cache records do not cover the entire payload")

    aggregate = hashlib.sha256()
    with path.open("rb") as stream:
        for key, offset, nbytes, expected_sha in records:
            if stream.tell() != offset:
                raise CacheError("BF16 cache tensor payload has a gap")
            digest = hashlib.sha256()
            remaining = nbytes
            while remaining:
                chunk = stream.read(min(CHUNK_BYTES, remaining))
                if not chunk:
                    raise CacheError("BF16 cache payload is truncated")
                aggregate.update(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != expected_sha:
                raise CacheError(f"BF16 cache payload tensor digest mismatch: {key}")
    if aggregate.hexdigest() != payload_hash:
        raise CacheError("BF16 cache aggregate payload digest mismatch")
    return {"payload_bytes": payload_bytes, "payload_sha256": payload_hash,
            "tensor_count": len(records), "all_raw_uint16_equal": True}


def validate_cache_file(path: Path | str) -> dict:
    """Load a production cache and bind it to the currently pinned receipts."""
    original_path = Path(path).expanduser()
    if original_path.is_symlink() or not original_path.is_file():
        raise CacheError("cache manifest must be a regular non-symlink file")
    path = original_path.resolve()
    if path.parent not in {CACHE_DIR, SIXTEEN_CACHE_DIR, THIRTY_SIX_CACHE_DIR}:
        raise CacheError("cache manifest must be in a pinned task source directory")
    try:
        raw = path.read_bytes()
        if len(raw) > 2 * 1024 * 1024:
            raise CacheError("cache manifest exceeds the bounded metadata limit")
        manifest = checkpoint.parse_header(raw)
    except (OSError, checkpoint.InventoryError) as error:
        raise CacheError("cannot parse cache manifest") from error
    block_count = len(manifest.get("blocks", {}))
    expected_dir = {
        16: SIXTEEN_CACHE_DIR,
        36: THIRTY_SIX_CACHE_DIR,
    }.get(block_count, CACHE_DIR)
    if path.parent != expected_dir:
        raise CacheError("cache schema and artifact directory do not match")
    payload_name = manifest.get("payload")
    if (not isinstance(payload_name, str) or Path(payload_name).name != payload_name
            or payload_name in {"", ".", ".."}):
        raise CacheError("cache payload must be a basename in the manifest directory")
    payload = path.parent / payload_name
    if payload.is_symlink():
        raise CacheError("cache payload must be a regular non-symlink file")
    result = validate_manifest(manifest, payload)
    raw_header = (CACHE_DIR / "pixelumm-header.json").read_bytes()
    for index in range(len(manifest["blocks"])):
        block = manifest["blocks"][str(index)]
        source_dir = _source_dir_for_block(block_count, index)
        subset = source_dir / block["subset_file"]
        receipt_path = source_dir / block["source_receipt_file"]
        receipt_raw = receipt_path.read_bytes()
        if hashlib.sha256(receipt_raw).hexdigest() != block["source_receipt_sha256"]:
            raise CacheError(f"block {index} source receipt digest changed")
        receipt = checkpoint.parse_header(receipt_raw)
        checked = checkpoint.validate_block_receipt(raw_header, receipt, subset, index)
        for key, field in (("subset_sha256", "subset_sha256"),
                           ("subset_bytes", "subset_bytes"),
                           ("source_extent", "block_data_offsets")):
            if block[key] != checked[field]:
                raise CacheError(f"block {index} cached source provenance changed: {key}")
        if (block["source_f32_bytes"] != checked["block_f32_bytes"]
                or block["source_range_chunks"] != receipt.get("range_chunks")):
            raise CacheError(f"block {index} cached source range evidence changed")
    return result


def _source_block_metadata(index: int, subset_path: Path, receipt_path: Path,
                           checked: dict, receipt: dict, receipt_sha: str,
                           names: list[str]) -> dict:
    return {
        "selected_block": index,
        "subset_file": subset_path.name,
        "subset_bytes": checked["subset_bytes"],
        "subset_sha256": checked["subset_sha256"],
        "source_receipt_file": receipt_path.name,
        "source_receipt_sha256": receipt_sha,
        "source_extent": checked["block_data_offsets"],
        "source_f32_bytes": checked["block_f32_bytes"],
        "source_range_chunks": receipt["range_chunks"],
        "weight_keys": names,
        "bf16_bytes": sum(_shape_bytes(list(shape), 2)
                          for shape in checkpoint.block_schema(index).values()),
    }


def make_cache(
    *, upstream: Path | str, header: Path | str,
    block0: Path | str, receipt0: Path | str,
    block1: Path | str, receipt1: Path | str,
    output: Path | str, device_name: str = "mps",
    block2: Path | str | None = None, receipt2: Path | str | None = None,
    block3: Path | str | None = None, receipt3: Path | str | None = None,
    block4: Path | str | None = None, receipt4: Path | str | None = None,
    block5: Path | str | None = None, receipt5: Path | str | None = None,
    block6: Path | str | None = None, receipt6: Path | str | None = None,
    block7: Path | str | None = None, receipt7: Path | str | None = None,
    block8: Path | str | None = None, receipt8: Path | str | None = None,
    block9: Path | str | None = None, receipt9: Path | str | None = None,
    block10: Path | str | None = None, receipt10: Path | str | None = None,
    block11: Path | str | None = None, receipt11: Path | str | None = None,
    block12: Path | str | None = None, receipt12: Path | str | None = None,
    block13: Path | str | None = None, receipt13: Path | str | None = None,
    block14: Path | str | None = None, receipt14: Path | str | None = None,
    block15: Path | str | None = None, receipt15: Path | str | None = None,
    block16: Path | str | None = None, receipt16: Path | str | None = None,
    block17: Path | str | None = None, receipt17: Path | str | None = None,
    block18: Path | str | None = None, receipt18: Path | str | None = None,
    block19: Path | str | None = None, receipt19: Path | str | None = None,
    block20: Path | str | None = None, receipt20: Path | str | None = None,
    block21: Path | str | None = None, receipt21: Path | str | None = None,
    block22: Path | str | None = None, receipt22: Path | str | None = None,
    block23: Path | str | None = None, receipt23: Path | str | None = None,
    block24: Path | str | None = None, receipt24: Path | str | None = None,
    block25: Path | str | None = None, receipt25: Path | str | None = None,
    block26: Path | str | None = None, receipt26: Path | str | None = None,
    block27: Path | str | None = None, receipt27: Path | str | None = None,
    block28: Path | str | None = None, receipt28: Path | str | None = None,
    block29: Path | str | None = None, receipt29: Path | str | None = None,
    block30: Path | str | None = None, receipt30: Path | str | None = None,
    block31: Path | str | None = None, receipt31: Path | str | None = None,
    block32: Path | str | None = None, receipt32: Path | str | None = None,
    block33: Path | str | None = None, receipt33: Path | str | None = None,
    block34: Path | str | None = None, receipt34: Path | str | None = None,
    block35: Path | str | None = None, receipt35: Path | str | None = None,
) -> dict:
    """Convert and persist an admitted consecutive prefix of upstream BF16 blocks."""
    if device_name != "mps":
        raise CacheError("production cache conversion must use the pinned PyTorch MPS path")
    if not torch.backends.mps.is_available():
        raise CacheError("PyTorch MPS is not available")
    if sys.byteorder != "little":
        raise CacheError("BF16 cache serialization requires a little-endian host")
    source_pairs = [(block0, receipt0), (block1, receipt1)]
    optional_pairs = (
        (block2, receipt2), (block3, receipt3),
        (block4, receipt4), (block5, receipt5),
        (block6, receipt6), (block7, receipt7),
        (block8, receipt8), (block9, receipt9),
        (block10, receipt10), (block11, receipt11),
        (block12, receipt12), (block13, receipt13),
        (block14, receipt14), (block15, receipt15),
        (block16, receipt16), (block17, receipt17),
        (block18, receipt18), (block19, receipt19),
        (block20, receipt20), (block21, receipt21),
        (block22, receipt22), (block23, receipt23),
        (block24, receipt24), (block25, receipt25),
        (block26, receipt26), (block27, receipt27),
        (block28, receipt28), (block29, receipt29),
        (block30, receipt30), (block31, receipt31),
        (block32, receipt32), (block33, receipt33),
        (block34, receipt34), (block35, receipt35),
    )
    missing_suffix = False
    for index, (subset, receipt) in enumerate(optional_pairs, start=2):
        if (subset is None) != (receipt is None):
            raise CacheError(f"block {index} subset and receipt must be supplied together")
        if subset is None:
            missing_suffix = True
            continue
        if missing_suffix:
            raise CacheError("block subset and receipt inputs must form a consecutive prefix")
        source_pairs.append((subset, receipt))
    if len(source_pairs) not in {2, 4, 8, 16, 36}:
        raise CacheError("cache requires exactly 2, 4, 8, 16, or 36 consecutive blocks")
    subset_inputs = tuple(Path(subset).expanduser() for subset, _ in source_pairs)
    receipt_inputs = tuple(Path(receipt).expanduser() for _, receipt in source_pairs)
    if any(path.is_symlink() for path in (*subset_inputs, *receipt_inputs)):
        raise CacheError("cache source subsets and receipts must not be symlinks")
    paths = tuple(path.resolve() for path in subset_inputs)
    receipts = tuple(path.resolve() for path in receipt_inputs)
    output_path = Path(output).expanduser().absolute()
    expected_output_dir = {
        16: SIXTEEN_CACHE_DIR,
        36: THIRTY_SIX_CACHE_DIR,
    }.get(len(paths), CACHE_DIR)
    if output_path.parent.resolve() != expected_output_dir:
        raise CacheError("cache output must be in its pinned task source directory")
    for index, (subset_path, receipt_path) in enumerate(zip(paths, receipts)):
        expected_source_dir = _source_dir_for_block(len(paths), index)
        if (subset_path.parent != expected_source_dir
                or receipt_path.parent != expected_source_dir):
            raise CacheError(f"block {index} inputs must be in their pinned source directory")
    expected_bytes = len(paths) * 385_892_864
    if expected_bytes > MAX_CACHE_PAYLOAD_BYTES:
        raise CacheError("BF16 cache request exceeds the 16 GiB payload cap")
    payload_path = output_path.with_suffix(".bin")
    payload_partial = payload_path.with_suffix(payload_path.suffix + ".partial")
    manifest_partial = output_path.with_suffix(output_path.suffix + ".partial")
    if any(path.exists() or path.is_symlink() for path in
           (output_path, payload_path, payload_partial, manifest_partial)):
        raise CacheError("refusing to replace an existing BF16 cache artifact")

    source, model_sha = reference.load_upstream(upstream)
    header_path = Path(header).expanduser().resolve()
    raw_header = header_path.read_bytes()
    checkpoint.parse_pinned_header(raw_header)
    if header_path.parent != CACHE_DIR:
        raise CacheError("pinned PixelUMM header must be in the task source directory")
    device = torch.device("mps")
    blocks = {}
    records = {}
    comparisons = {}
    aggregate = hashlib.sha256()
    offset = 0
    linked_payload = False
    try:
        with payload_partial.open("xb+") as stream:
            for index, (subset_path, receipt_path) in enumerate(zip(paths, receipts)):
                layer, checked, receipt_sha = reference._load_layer_for_block(
                    source, subset_path, receipt_path, header_path, device, index
                )
                receipt = checkpoint.parse_header(receipt_path.read_bytes())
                weight_keys = []
                with torch.inference_mode():
                    # Synchronization is confined to startup conversion, not the
                    # timed native inference path.
                    torch.mps.synchronize()
                    for name, parameter in sorted(layer.named_parameters()):
                        key = f"block{index}.{name}"
                        if key in records:
                            raise CacheError("duplicate source parameter name")
                        record, observation = _tensor_record(stream, aggregate, key, parameter, offset)
                        records[key] = record
                        comparisons[key] = observation
                        weight_keys.append(key)
                        offset += record["nbytes"]
                expected_keys = sorted(
                    f"block{index}.{name}" for name in checkpoint.block_schema(index)
                )
                if sorted(weight_keys) != expected_keys:
                    raise CacheError(f"source-loaded block {index} parameter inventory mismatch")
                blocks[str(index)] = _source_block_metadata(
                    index, subset_path, receipt_path, checked, receipt, receipt_sha,
                    expected_keys,
                )
                del parameter
                del layer
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            stream.flush()
            os.fsync(stream.fileno())
        if not all(item["raw_uint16_equal"] and item["payload_bytes_equal"]
                   for item in comparisons.values()):
            raise CacheError("source MPS BF16 words differ from persisted cache bytes")
        expected_count = len(paths) * 13
        if len(records) != expected_count:
            raise CacheError(f"cache conversion did not produce all {expected_count} source tensors")

        manifest = {
            "schema": SCHEMA,
            "upstream_revision": reference.UPSTREAM_REVISION,
            "upstream_model_path": reference.UPSTREAM_MODEL_PATH,
            "upstream_model_sha256": model_sha,
            "source_commit": checkpoint.SOURCE_COMMIT,
            "hf_revision": checkpoint.REVISION,
            "hf_source_path": checkpoint.SOURCE_PATH,
            "source_file_bytes": checkpoint.SOURCE_BYTES,
            "checkpoint_header_sha256": checkpoint.HEADER_SHA256,
            "stored_dtype": "F32", "runtime_dtype": "BF16",
            "full_file_hash_verified": False,
            "python_version": platform.python_version(),
            "conversion": {**CONVERSION, "torch_version": torch.__version__},
            "blocks": blocks,
            "weights": records,
            "comparison": {
                "method": "source MPS BF16 tensor raw uint16 bytes equal persisted payload",
                "tensor_count": len(comparisons),
                "all_raw_uint16_equal": True,
                "weights": comparisons,
            },
            "payload": payload_path.name,
            "payload_bytes": offset,
            "payload_sha256": aggregate.hexdigest(),
        }
        validate_manifest(
            manifest, payload_partial, payload_basename=payload_path.name
        )
        with manifest_partial.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(payload_partial, payload_path)
        linked_payload = True
        os.link(manifest_partial, output_path)
        payload_partial.unlink()
        manifest_partial.unlink()
        return {
            "manifest": str(output_path), "payload": str(payload_path),
            "payload_bytes": offset, "payload_sha256": manifest["payload_sha256"],
            "tensor_count": len(records), "all_raw_uint16_equal": True,
        }
    except Exception:
        for candidate in (payload_partial, manifest_partial):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        if linked_payload:
            try:
                payload_path.unlink()
            except FileNotFoundError:
                pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--block0", required=True, type=Path)
    parser.add_argument("--receipt0", required=True, type=Path)
    parser.add_argument("--block1", required=True, type=Path)
    parser.add_argument("--receipt1", required=True, type=Path)
    parser.add_argument("--block2", type=Path)
    parser.add_argument("--receipt2", type=Path)
    parser.add_argument("--block3", type=Path)
    parser.add_argument("--receipt3", type=Path)
    parser.add_argument("--block4", type=Path)
    parser.add_argument("--receipt4", type=Path)
    parser.add_argument("--block5", type=Path)
    parser.add_argument("--receipt5", type=Path)
    parser.add_argument("--block6", type=Path)
    parser.add_argument("--receipt6", type=Path)
    parser.add_argument("--block7", type=Path)
    parser.add_argument("--receipt7", type=Path)
    for index in range(8, 16):
        parser.add_argument(f"--block{index}", type=Path)
        parser.add_argument(f"--receipt{index}", type=Path)
    for index in range(16, 36):
        parser.add_argument(f"--block{index}", type=Path)
        parser.add_argument(f"--receipt{index}", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("mps",), default="mps")
    args = parser.parse_args(argv)
    try:
        result = make_cache(
            upstream=args.upstream, header=args.header,
            block0=args.block0, receipt0=args.receipt0,
            block1=args.block1, receipt1=args.receipt1,
            block2=args.block2, receipt2=args.receipt2,
            block3=args.block3, receipt3=args.receipt3,
            block4=args.block4, receipt4=args.receipt4,
            block5=args.block5, receipt5=args.receipt5,
            block6=args.block6, receipt6=args.receipt6,
            block7=args.block7, receipt7=args.receipt7,
            **{f"block{index}": getattr(args, f"block{index}") for index in range(8, 16)},
            **{f"receipt{index}": getattr(args, f"receipt{index}") for index in range(8, 16)},
            **{f"block{index}": getattr(args, f"block{index}") for index in range(16, 36)},
            **{f"receipt{index}": getattr(args, f"receipt{index}") for index in range(16, 36)},
            output=args.output, device_name=args.device,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (CacheError, checkpoint.InventoryError, reference.ReferenceError) as error:
        print(f"PixelUMM BF16 cache rejected: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
