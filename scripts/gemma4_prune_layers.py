#!/usr/bin/env python3
"""Inspect GGUF tensor metadata or create an immutable Gemma 4 layer prefix.

The writer copies tensor payload bytes without decoding or requantizing them.
It deliberately refuses Gemma 4 per-layer embeddings (PLE), unknown
architecture arrays that look layer-indexed, and unrecognized GGML tensor
types, because those cases require layout-aware pruning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional


SCHEMA = "cogni_ml.gemma4_prefix_prune.v1"
COPY_CHUNK_BYTES = 4 * 1024 * 1024
MAX_HEADER_ENTRIES = 10_000_000

# (elements per encoded block, bytes per encoded block). Keep this deliberately
# limited to the target Gemma 4 file's official b11714 types; unknown types fail
# closed rather than risking a corrupt payload-size calculation.
GGML_BLOCKS = {
    0: (1, 4),       # GGML_TYPE_F32
    12: (256, 144),  # GGML_TYPE_Q4_K: 2*FP16 + 12 scales + 128 quants
    14: (256, 210),  # GGML_TYPE_Q6_K: FP16 + 16 scales + 192 quants
}

META_WIDTHS = {
    0: 1,    # UINT8
    1: 1,    # INT8
    2: 2,    # UINT16
    3: 2,    # INT16
    4: 4,    # UINT32
    5: 4,    # INT32
    6: 4,    # FLOAT32
    7: 1,    # BOOL
    10: 8,   # UINT64
    11: 8,   # INT64
    12: 8,   # FLOAT64
}
META_NAMES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

LAYER_ARRAY_SUFFIXES = {
    ".attention.head_count_kv",
    ".attention.sliding_window_pattern",
}
PER_LAYER_TENSOR_MARKERS = ("per_layer", "per-layer", ".ple")
BLOCK_NAME = re.compile(r"^blk\.(\d+)(?:\.|$)")


class GGUFError(ValueError):
    """Raised when a GGUF file is malformed or unsafe to prune."""


@dataclass
class Metadata:
    key: str
    value_type: int
    raw: bytes
    array_type: Optional[int] = None
    array_count: Optional[int] = None


@dataclass
class Tensor:
    name: str
    dimensions: list[int]
    tensor_type: int
    offset: int
    elements: int
    payload_bytes: int
    block_index: Optional[int]


@dataclass
class GGUFFile:
    path: Path
    version: int
    metadata: list[Metadata]
    metadata_by_key: dict[str, Metadata]
    tensors: list[Tensor]
    architecture: str
    block_count: int
    alignment: int
    data_start: int
    file_bytes: int


def _read_exact(stream: BinaryIO, count: int, *, what: str) -> bytes:
    if count < 0:
        raise GGUFError(f"negative byte count while reading {what}")
    value = stream.read(count)
    if len(value) != count:
        raise GGUFError(f"truncated GGUF while reading {what}")
    return value


def _u32(stream: BinaryIO, *, what: str) -> int:
    return struct.unpack("<I", _read_exact(stream, 4, what=what))[0]


def _u64(stream: BinaryIO, *, what: str) -> int:
    return struct.unpack("<Q", _read_exact(stream, 8, what=what))[0]


def _read_string(stream: BinaryIO, file_bytes: int, *, what: str) -> str:
    length = _u64(stream, what=f"{what} length")
    if length > file_bytes or stream.tell() + length > file_bytes:
        raise GGUFError(f"invalid {what} length {length}")
    try:
        return _read_exact(stream, length, what=what).decode("utf-8")
    except UnicodeDecodeError as error:
        raise GGUFError(f"{what} is not valid UTF-8") from error


def _skip_meta_value(stream: BinaryIO, value_type: int, file_bytes: int) -> None:
    if value_type in META_WIDTHS:
        _read_exact(stream, META_WIDTHS[value_type], what="metadata scalar")
        return
    if value_type == 8:
        length = _u64(stream, what="metadata string length")
        if length > file_bytes or stream.tell() + length > file_bytes:
            raise GGUFError(f"invalid metadata string length {length}")
        _read_exact(stream, length, what="metadata string")
        return
    if value_type == 9:
        item_type = _u32(stream, what="metadata array element type")
        if item_type == 9:
            raise GGUFError("nested metadata arrays are unsupported")
        count = _u64(stream, what="metadata array count")
        if count > MAX_HEADER_ENTRIES:
            raise GGUFError(f"metadata array is unreasonably large: {count}")
        for _ in range(count):
            _skip_meta_value(stream, item_type, file_bytes)
        return
    raise GGUFError(f"unsupported GGUF metadata type {value_type}")


def _parse_metadata(stream: BinaryIO, file_bytes: int) -> Metadata:
    key = _read_string(stream, file_bytes, what="metadata key")
    value_type = _u32(stream, what=f"type for metadata {key}")
    value_start = stream.tell()
    array_type = None
    array_count = None
    if value_type == 9:
        array_type = _u32(stream, what=f"array type for {key}")
        if array_type == 9:
            raise GGUFError(f"nested array metadata is unsupported for {key}")
        array_count = _u64(stream, what=f"array count for {key}")
        if array_count > MAX_HEADER_ENTRIES:
            raise GGUFError(f"metadata array {key} is unreasonably large: {array_count}")
        for _ in range(array_count):
            _skip_meta_value(stream, array_type, file_bytes)
    else:
        _skip_meta_value(stream, value_type, file_bytes)
    value_end = stream.tell()
    stream.seek(value_start)
    raw = _read_exact(stream, value_end - value_start, what=f"metadata value {key}")
    return Metadata(key, value_type, raw, array_type, array_count)


def _decode_scalar(value_type: int, raw: bytes):
    if value_type == 8:
        if len(raw) < 8:
            raise GGUFError("short GGUF string value")
        length = struct.unpack_from("<Q", raw)[0]
        if len(raw) != 8 + length:
            raise GGUFError("malformed GGUF string value")
        try:
            return raw[8:].decode("utf-8")
        except UnicodeDecodeError as error:
            raise GGUFError("GGUF string value is not UTF-8") from error
    width = META_WIDTHS.get(value_type)
    if width is None or len(raw) != width:
        raise GGUFError(f"invalid scalar metadata type/size {value_type}/{len(raw)}")
    if value_type == 6:
        return struct.unpack("<f", raw)[0]
    if value_type == 12:
        return struct.unpack("<d", raw)[0]
    if value_type == 7:
        if raw[0] not in (0, 1):
            raise GGUFError(f"invalid GGUF bool value {raw[0]}")
        return raw[0] == 1
    signed = value_type in (1, 3, 5, 11)
    return int.from_bytes(raw, "little", signed=signed)


def _decode_array_values(entry: Metadata, *, limit: Optional[int] = None) -> list:
    if entry.value_type != 9 or entry.array_type is None or entry.array_count is None:
        raise GGUFError(f"metadata {entry.key} is not an array")
    item_type = entry.array_type
    count = entry.array_count
    wanted = count if limit is None else min(count, limit)
    cursor = 12  # element type (u32) followed by count (u64)
    result = []
    for index in range(wanted):
        start = cursor
        if item_type == 8:
            if cursor + 8 > len(entry.raw):
                raise GGUFError(f"truncated string array {entry.key}")
            length = struct.unpack_from("<Q", entry.raw, cursor)[0]
            cursor += 8
            end = cursor + length
            if end > len(entry.raw):
                raise GGUFError(f"truncated string array item in {entry.key}")
            raw = entry.raw[start:end]
            cursor = end
        else:
            width = META_WIDTHS.get(item_type)
            if width is None:
                raise GGUFError(f"unsupported array element type {item_type} in {entry.key}")
            end = cursor + width
            if end > len(entry.raw):
                raise GGUFError(f"truncated metadata array {entry.key}")
            raw = entry.raw[cursor:end]
            cursor = end
        result.append(_decode_scalar(item_type, raw))
    if wanted == count and cursor != len(entry.raw):
        raise GGUFError(f"metadata array {entry.key} has trailing bytes")
    return result


def _scalar_metadata(entry: Metadata, *, key: str):
    if entry.value_type == 9:
        raise GGUFError(f"metadata {key} must be a scalar")
    return _decode_scalar(entry.value_type, entry.raw)


def _tensor_payload_bytes(dimensions: list[int], tensor_type: int, name: str) -> int:
    layout = GGML_BLOCKS.get(tensor_type)
    if layout is None:
        raise GGUFError(
            f"tensor {name} uses unsupported GGML type {tensor_type}; refusing an inexact byte count"
        )
    block_elements, block_bytes = layout
    elements = 1
    for dimension in dimensions:
        elements *= dimension
    if elements % block_elements:
        raise GGUFError(
            f"tensor {name} has {elements} elements, not divisible by GGML block size {block_elements}"
        )
    return (elements // block_elements) * block_bytes


def _parse(path: Path) -> GGUFFile:
    try:
        file_bytes = path.stat().st_size
    except OSError as error:
        raise GGUFError(f"cannot stat GGUF input {path.name}: {error}") from error
    with path.open("rb") as stream:
        if _read_exact(stream, 4, what="magic") != b"GGUF":
            raise GGUFError("input does not have GGUF magic")
        version = _u32(stream, what="version")
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}; expected v2 or v3")
        tensor_count = _u64(stream, what="tensor count")
        metadata_count = _u64(stream, what="metadata count")
        if tensor_count > MAX_HEADER_ENTRIES or metadata_count > MAX_HEADER_ENTRIES:
            raise GGUFError("GGUF header entry count is unreasonably large")

        metadata = []
        metadata_by_key = {}
        for _ in range(metadata_count):
            entry = _parse_metadata(stream, file_bytes)
            if entry.key in metadata_by_key:
                raise GGUFError(f"duplicate metadata key {entry.key}")
            metadata.append(entry)
            metadata_by_key[entry.key] = entry

        tensor_records = []
        for _ in range(tensor_count):
            name = _read_string(stream, file_bytes, what="tensor name")
            dimensions_count = _u32(stream, what=f"dimension count for {name}")
            if dimensions_count < 1 or dimensions_count > 4:
                raise GGUFError(f"tensor {name} has unsupported rank {dimensions_count}")
            dimensions = [_u64(stream, what=f"dimension for {name}") for _ in range(dimensions_count)]
            if any(dimension == 0 for dimension in dimensions):
                raise GGUFError(f"tensor {name} has a zero-sized dimension")
            tensor_type = _u32(stream, what=f"GGML type for {name}")
            offset = _u64(stream, what=f"offset for {name}")
            payload_bytes = _tensor_payload_bytes(dimensions, tensor_type, name)
            elements = 1
            for dimension in dimensions:
                elements *= dimension
            match = BLOCK_NAME.match(name)
            if name.startswith("blk.") and match is None:
                raise GGUFError(f"unrecognized decoder tensor name {name}")
            block_index = int(match.group(1)) if match else None
            tensor_records.append(
                (name, dimensions, tensor_type, offset, elements, payload_bytes, block_index)
            )

        alignment_entry = metadata_by_key.get("general.alignment")
        alignment = _scalar_metadata(alignment_entry, key="general.alignment") if alignment_entry else 32
        if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
            raise GGUFError(f"invalid GGUF alignment {alignment!r}; expected a positive power of two")
        data_start = (stream.tell() + alignment - 1) & ~(alignment - 1)
        if data_start > file_bytes:
            raise GGUFError("GGUF header extends beyond file")

        architecture_entry = metadata_by_key.get("general.architecture")
        if architecture_entry is None:
            raise GGUFError("GGUF is missing general.architecture")
        architecture = _scalar_metadata(architecture_entry, key="general.architecture")
        if not isinstance(architecture, str) or not architecture:
            raise GGUFError("general.architecture must be a nonempty string")
        block_entry = metadata_by_key.get(f"{architecture}.block_count")
        if block_entry is None:
            raise GGUFError(f"GGUF is missing {architecture}.block_count")
        block_count = _scalar_metadata(block_entry, key=f"{architecture}.block_count")
        if not isinstance(block_count, int) or isinstance(block_count, bool) or block_count <= 0:
            raise GGUFError(f"invalid {architecture}.block_count value {block_count!r}")

        tensors = [Tensor(*record) for record in tensor_records]
        names = set()
        for tensor in tensors:
            if tensor.name in names:
                raise GGUFError(f"duplicate tensor name {tensor.name}")
            names.add(tensor.name)
            if tensor.offset % alignment:
                raise GGUFError(f"tensor {tensor.name} offset is not {alignment}-byte aligned")
            end = data_start + tensor.offset + tensor.payload_bytes
            if end > file_bytes:
                raise GGUFError(
                    f"tensor {tensor.name} extends past end of GGUF ({end} > {file_bytes})"
                )
            if tensor.block_index is not None and tensor.block_index >= block_count:
                raise GGUFError(
                    f"tensor {tensor.name} refers to block {tensor.block_index}, "
                    f"outside {architecture}.block_count={block_count}"
                )
        ordered = sorted(tensors, key=lambda tensor: tensor.offset)
        for previous, current in zip(ordered, ordered[1:]):
            previous_end = previous.offset + previous.payload_bytes
            if previous_end > current.offset:
                raise GGUFError(
                    f"tensor payloads overlap: {previous.name} and {current.name}"
                )

    return GGUFFile(
        path,
        version,
        metadata,
        metadata_by_key,
        tensors,
        architecture,
        block_count,
        alignment,
        data_start,
        file_bytes,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _layer_metadata(file: GGUFFile) -> dict[str, list]:
    result = {}
    prefix = f"{file.architecture}."
    for entry in file.metadata:
        if entry.value_type != 9:
            continue
        is_known_layer_array = any(entry.key.endswith(suffix) for suffix in LAYER_ARRAY_SUFFIXES)
        if is_known_layer_array:
            if entry.array_count != file.block_count:
                raise GGUFError(
                    f"layer metadata array {entry.key} has {entry.array_count} values; "
                    f"expected {file.block_count}"
                )
            result[entry.key] = _decode_array_values(entry)
        elif entry.key.startswith(prefix) and entry.array_count == file.block_count:
            raise GGUFError(
                f"unrecognized architecture array {entry.key} has one value per decoder layer; "
                "refusing to leave potentially stale layer metadata"
            )
    return result


def _ple_tensors(file: GGUFFile) -> list[Tensor]:
    return [
        tensor
        for tensor in file.tensors
        if any(marker in tensor.name.lower() for marker in PER_LAYER_TENSOR_MARKERS)
    ]


def _validate_prefix_compatible(file: GGUFFile) -> dict[str, list]:
    if file.architecture != "gemma4":
        raise GGUFError(f"prefix pruning is limited to gemma4; found {file.architecture}")
    ple = file.metadata_by_key.get("gemma4.embedding_length_per_layer_input")
    if ple is not None:
        value = _scalar_metadata(ple, key="gemma4.embedding_length_per_layer_input")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise GGUFError(f"invalid Gemma 4 per-layer embedding length {value!r}")
        if value != 0:
            raise GGUFError(
                "nonzero PLE/per-layer embedding length requires tensor-aware cropping; "
                "raw block-prefix retention is unsupported"
            )
    ple_tensors = _ple_tensors(file)
    if ple_tensors:
        names = ", ".join(tensor.name for tensor in ple_tensors[:5])
        raise GGUFError(
            f"PLE/per-layer tensors require layout-aware cropping; refusing: {names}"
        )
    shared_kv_entry = file.metadata_by_key.get("gemma4.attention.shared_kv_layers")
    if shared_kv_entry is not None:
        shared_kv_layers = _scalar_metadata(
            shared_kv_entry, key="gemma4.attention.shared_kv_layers"
        )
        if (
            not isinstance(shared_kv_layers, int)
            or isinstance(shared_kv_layers, bool)
            or shared_kv_layers < 0
        ):
            raise GGUFError(f"invalid Gemma 4 shared KV layer count {shared_kv_layers!r}")
        if shared_kv_layers != 0:
            raise GGUFError(
                "nonzero shared KV layers require shared-suffix-aware pruning; "
                "raw decoder-prefix retention is unsupported"
            )
    return _layer_metadata(file)


def _sum_metrics(file: GGUFFile) -> tuple[int, int]:
    return sum(tensor.elements for tensor in file.tensors), sum(
        tensor.payload_bytes for tensor in file.tensors
    )


def _reported_layer_metadata(file: GGUFFile) -> dict[str, list]:
    result = {}
    for entry in file.metadata:
        if entry.value_type == 9 and any(
            entry.key.endswith(suffix) for suffix in LAYER_ARRAY_SUFFIXES
        ):
            result[entry.key] = _decode_array_values(entry)
    return result


def _output_head_storage(file: GGUFFile) -> Optional[str]:
    names = {tensor.name for tensor in file.tensors}
    if "output.weight" in names:
        return "output.weight"
    if "token_embd.weight" in names:
        # Gemma 4 GGUF stores its tied input/output embeddings once.
        return "token_embd.weight"
    return None


def inspect_gguf(path: os.PathLike | str) -> dict:
    source = Path(path)
    file = _parse(source)
    arrays = _reported_layer_metadata(file)
    parameter_count, tensor_payload_bytes = _sum_metrics(file)
    name_entry = file.metadata_by_key.get("general.name")
    model_name = _scalar_metadata(name_entry, key="general.name") if name_entry else None
    ple_entry = file.metadata_by_key.get("gemma4.embedding_length_per_layer_input")
    ple_length = (
        _scalar_metadata(ple_entry, key="gemma4.embedding_length_per_layer_input")
        if ple_entry
        else None
    )
    shared_kv_entry = file.metadata_by_key.get("gemma4.attention.shared_kv_layers")
    shared_kv_layers = (
        _scalar_metadata(shared_kv_entry, key="gemma4.attention.shared_kv_layers")
        if shared_kv_entry
        else None
    )
    tensor_types = {}
    for tensor in file.tensors:
        key = str(tensor.tensor_type)
        tensor_types[key] = tensor_types.get(key, 0) + 1
    return {
        "schema": SCHEMA,
        "operation": "inspect",
        "source_basename": source.name,
        "source_sha256": _sha256_file(source),
        "file_bytes": file.file_bytes,
        "architecture": file.architecture,
        "model_name": model_name,
        "decoder_layers": file.block_count,
        "original_decoder_layers": file.block_count,
        "pruned_decoder_layers": 0,
        "per_layer_embedding_input_length": ple_length,
        "shared_kv_layers": shared_kv_layers,
        "tensor_count": len(file.tensors),
        "parameter_count": parameter_count,
        "tensor_payload_bytes": tensor_payload_bytes,
        "tensor_type_counts": dict(sorted(tensor_types.items(), key=lambda item: int(item[0]))),
        "layer_metadata_arrays": arrays,
        "per_layer_tensors": [
            {"name": tensor.name, "dimensions": tensor.dimensions, "type": tensor.tensor_type}
            for tensor in _ple_tensors(file)
        ],
        "output_head_storage": _output_head_storage(file),
        "output_head_tied_to_token_embd": (
            _output_head_storage(file) == "token_embd.weight"
        ),
    }


def _encode_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _encode_scalar(value_type: int, value) -> bytes:
    if value_type in (4, 5):
        return struct.pack("<I" if value_type == 4 else "<i", value)
    if value_type in (10, 11):
        return struct.pack("<Q" if value_type == 10 else "<q", value)
    raise GGUFError(
        f"cannot safely update block_count metadata type {META_NAMES.get(value_type, value_type)}"
    )


def _encode_array_prefix(entry: Metadata, count: int) -> bytes:
    if entry.value_type != 9 or entry.array_type is None or entry.array_count is None:
        raise GGUFError(f"cannot trim non-array layer metadata {entry.key}")
    width = META_WIDTHS.get(entry.array_type)
    if width is None:
        raise GGUFError(
            f"layer array {entry.key} uses non-fixed element type {entry.array_type}"
        )
    if count < 0 or count > entry.array_count:
        raise GGUFError(f"invalid prefix count for layer array {entry.key}")
    prefix_end = 12 + count * width
    if prefix_end > len(entry.raw):
        raise GGUFError(f"truncated layer array {entry.key}")
    return struct.pack("<IQ", entry.array_type, count) + entry.raw[12:prefix_end]


def _build_header(
    file: GGUFFile,
    metadata_raw: dict[str, bytes],
    tensors: list[Tensor],
    tensor_offsets: dict[str, int],
) -> bytes:
    header = bytearray(b"GGUF")
    header.extend(struct.pack("<IQQ", file.version, len(tensors), len(file.metadata)))
    for entry in file.metadata:
        header.extend(_encode_string(entry.key))
        header.extend(struct.pack("<I", entry.value_type))
        header.extend(metadata_raw.get(entry.key, entry.raw))
    for tensor in tensors:
        header.extend(_encode_string(tensor.name))
        header.extend(struct.pack("<I", len(tensor.dimensions)))
        for dimension in tensor.dimensions:
            header.extend(struct.pack("<Q", dimension))
        header.extend(struct.pack("<IQ", tensor.tensor_type, tensor_offsets[tensor.name]))
    return bytes(header)


def _write_zeros(stream: BinaryIO, count: int, digest=None) -> None:
    zeros = b"\x00" * min(COPY_CHUNK_BYTES, max(count, 1))
    while count:
        chunk = zeros[: min(len(zeros), count)]
        stream.write(chunk)
        if digest is not None:
            digest.update(chunk)
        count -= len(chunk)


def _publish_no_clobber(temp_path: Path, destination: Path) -> None:
    try:
        os.link(temp_path, destination)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite existing file: {destination.name}")
    except OSError as error:
        raise OSError(f"cannot atomically publish {destination.name}: {error}") from error
    temp_path.unlink()
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _temp_file(directory: Path, prefix: str) -> tuple[int, Path]:
    fd, raw_path = tempfile.mkstemp(prefix=prefix, dir=directory)
    return fd, Path(raw_path)


def _write_temp_model(
    source: GGUFFile,
    destination: Path,
    metadata_raw: dict[str, bytes],
    kept: list[Tensor],
) -> tuple[Path, str, int]:
    output_offsets = {}
    cursor = 0
    for tensor in kept:
        cursor = (cursor + source.alignment - 1) & ~(source.alignment - 1)
        output_offsets[tensor.name] = cursor
        cursor += tensor.payload_bytes
    header = _build_header(source, metadata_raw, kept, output_offsets)
    data_start = (len(header) + source.alignment - 1) & ~(source.alignment - 1)
    fd, temp_path = _temp_file(destination.parent, f".{destination.name}.tmp-")
    digest = hashlib.sha256()
    file_bytes = 0
    try:
        with os.fdopen(fd, "wb") as output, source.path.open("rb") as original:
            output.write(header)
            digest.update(header)
            file_bytes += len(header)
            padding = data_start - len(header)
            _write_zeros(output, padding, digest)
            file_bytes += padding
            data_cursor = 0
            for tensor in kept:
                output_offset = output_offsets[tensor.name]
                padding = output_offset - data_cursor
                if padding < 0:
                    raise GGUFError(f"internal output overlap at tensor {tensor.name}")
                _write_zeros(output, padding, digest)
                file_bytes += padding
                original.seek(source.data_start + tensor.offset)
                remaining = tensor.payload_bytes
                while remaining:
                    chunk = original.read(min(COPY_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise GGUFError(f"source ended inside tensor {tensor.name}")
                    output.write(chunk)
                    digest.update(chunk)
                    file_bytes += len(chunk)
                    remaining -= len(chunk)
                data_cursor = output_offset + tensor.payload_bytes
            output.flush()
            os.fsync(output.fileno())
        return temp_path, digest.hexdigest(), file_bytes
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, report: dict) -> Path:
    fd, temp_path = _temp_file(path.parent, f".{path.name}.tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return temp_path
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _path_is_source(path: Path, source: Path) -> bool:
    try:
        if path.exists() and os.path.samefile(path, source):
            return True
    except OSError:
        pass
    return path.resolve() == source.resolve()


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def prune_gguf(
    input_path: os.PathLike | str,
    output_path: os.PathLike | str,
    keep_blocks: int,
    manifest_path: os.PathLike | str | None = None,
) -> dict:
    source_path = Path(input_path)
    output = Path(output_path)
    manifest = Path(manifest_path) if manifest_path else Path(str(output) + ".manifest.json")
    if not isinstance(keep_blocks, int) or isinstance(keep_blocks, bool) or keep_blocks <= 0:
        raise ValueError("keep_blocks must be a positive integer")
    if _path_is_source(output, source_path) or _path_is_source(manifest, source_path):
        raise ValueError("output and manifest paths must not refer to the input GGUF")
    if output.resolve() == manifest.resolve():
        raise ValueError("output GGUF and manifest paths must be different")
    if not output.parent.is_dir() or not manifest.parent.is_dir():
        raise FileNotFoundError("output and manifest directories must already exist")
    # Catch collisions before the expensive source hash or any temporary output.
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {output.name}")
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {manifest.name}")

    source_stat = _stat_signature(source_path.stat())
    parsed = _parse(source_path)
    layer_arrays = _validate_prefix_compatible(parsed)
    if keep_blocks > parsed.block_count:
        raise ValueError(
            f"keep_blocks={keep_blocks} exceeds source block_count={parsed.block_count}"
        )
    seen_blocks = {tensor.block_index for tensor in parsed.tensors if tensor.block_index is not None}
    expected_blocks = set(range(parsed.block_count))
    if seen_blocks != expected_blocks:
        missing = sorted(expected_blocks - seen_blocks)
        raise GGUFError(f"source decoder blocks are incomplete; missing indices {missing[:12]}")

    known_layer_arrays = {
        key for key in layer_arrays if any(key.endswith(suffix) for suffix in LAYER_ARRAY_SUFFIXES)
    }
    metadata_raw = {}
    block_key = f"{parsed.architecture}.block_count"
    block_entry = parsed.metadata_by_key[block_key]
    metadata_raw[block_key] = _encode_scalar(block_entry.value_type, keep_blocks)
    name_entry = parsed.metadata_by_key.get("general.name")
    if name_entry is None or name_entry.value_type != 8:
        raise GGUFError("GGUF must provide string metadata general.name to mark experimental output")
    source_name = _scalar_metadata(name_entry, key="general.name")
    output_name = f"{source_name} [experimental decoder prefix {keep_blocks}/{parsed.block_count}]"
    metadata_raw["general.name"] = _encode_string(output_name)
    for key in known_layer_arrays:
        entry = parsed.metadata_by_key[key]
        metadata_raw[key] = _encode_array_prefix(entry, keep_blocks)

    kept = [
        tensor
        for tensor in parsed.tensors
        if tensor.block_index is None or tensor.block_index < keep_blocks
    ]
    source_parameter_count, source_payload_bytes = _sum_metrics(parsed)
    parameter_count = sum(tensor.elements for tensor in kept)
    payload_bytes = sum(tensor.payload_bytes for tensor in kept)
    source_sha256 = _sha256_file(source_path)

    output_temp = None
    manifest_temp = None
    output_published = False
    try:
        output_temp, output_sha256, output_file_bytes = _write_temp_model(
            parsed, output, metadata_raw, kept
        )
        if _stat_signature(source_path.stat()) != source_stat:
            raise GGUFError("source file changed while the pruned GGUF was being prepared")

        report = {
            "schema": SCHEMA,
            "operation": "prune",
            "source_basename": source_path.name,
            "source_sha256": source_sha256,
            "source_file_bytes": parsed.file_bytes,
            "source": {
                "basename": source_path.name,
                "sha256": source_sha256,
                "file_bytes": parsed.file_bytes,
            },
            "output_basename": output.name,
            "output_sha256": output_sha256,
            "output_name": output_name,
            "output_file_bytes": output_file_bytes,
            "output": {
                "basename": output.name,
                "sha256": output_sha256,
                "file_bytes": output_file_bytes,
            },
            "architecture": parsed.architecture,
            "original_decoder_layers": parsed.block_count,
            "decoder_layers": keep_blocks,
            "pruned_decoder_layers": parsed.block_count - keep_blocks,
            "shared_kv_layers": 0,
            "source_tensor_count": len(parsed.tensors),
            "tensor_count": len(kept),
            "source_parameter_count": source_parameter_count,
            "parameter_count": parameter_count,
            "source_tensor_payload_bytes": source_payload_bytes,
            "tensor_payload_bytes": payload_bytes,
            "layer_metadata_arrays": {
                key: values[:keep_blocks] for key, values in layer_arrays.items()
            },
            "output_head_storage": _output_head_storage(parsed),
            "output_head_tied_to_token_embd": (
                _output_head_storage(parsed) == "token_embd.weight"
            ),
            "tensor_bytes_are_raw_copies": True,
            "quality_validated": False,
        }
        manifest_temp = _atomic_write_json(manifest, report)
        _publish_no_clobber(output_temp, output)
        output_temp = None
        output_published = True
        _publish_no_clobber(manifest_temp, manifest)
        manifest_temp = None
        return report
    except Exception:
        if output_temp is not None:
            try:
                output_temp.unlink()
            except FileNotFoundError:
                pass
        if manifest_temp is not None:
            try:
                manifest_temp.unlink()
            except FileNotFoundError:
                pass
        if output_published:
            try:
                output.unlink()
            except FileNotFoundError:
                pass
        raise


def _print_json(value: dict) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="inspect GGUF counts and Gemma 4 metadata")
    inspect_parser.add_argument("model", type=Path)
    prune_parser = subparsers.add_parser("prune", help="write a first-N-layer experimental GGUF")
    prune_parser.add_argument("model", type=Path)
    prune_parser.add_argument("--keep-blocks", required=True, type=int)
    prune_parser.add_argument("--output", required=True, type=Path)
    prune_parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            _print_json(inspect_gguf(args.model))
        else:
            _print_json(prune_gguf(args.model, args.output, args.keep_blocks, args.manifest))
        return 0
    except (GGUFError, ValueError, OSError) as error:
        print(f"gemma4_prune_layers: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
