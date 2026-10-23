#!/usr/bin/env python3
"""Build independent CPU-packed, same-format MPS references for Wan affine candidates.

The script consumes the pinned single-projection fixture, writes one immutable
candidate payload per format, decodes those payloads independently on CPU, and
uses MPS only for F.linear projection references. It never loads a Wan model.
The payload budget is shared with canonical Q4_K, including a common external
16-byte descriptor (reported separately from the payload bytes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

SOURCE_ROWS = 4
INPUT_DIM = 5120
OUTPUT_DIM = 13824
BENCH_ROWS = 4480
BLOCK_VALUES = 256
AFFINE_RECORD_BYTES = 136
AFFINE_GROUPS = OUTPUT_DIM * INPUT_DIM // BLOCK_VALUES
SPARSE_RECORD_VALUES = 1024
SPARSE_RECORDS = OUTPUT_DIM * INPUT_DIM // SPARSE_RECORD_VALUES
PATCH_COUNT = 414_719
Q4_K_BYTES = 39_813_120
FORMAT_DESCRIPTOR_BYTES = 16
SHARED_BIAS_BYTES = OUTPUT_DIM * 4
AFFINE_BASE_BYTES = AFFINE_GROUPS * AFFINE_RECORD_BYTES
SPARSE_COUNTS_BYTES = SPARSE_RECORDS * 4
SPARSE_OFFSETS_BYTES = (SPARSE_RECORDS + 1) * 4
SPARSE_PATCH_BYTES = PATCH_COUNT * 4

FIXTURE_MANIFEST_SHA256 = "0dba2df508f72764837c23d8c5743a6d687e2d1b6104406e38325031084c3703"
INPUT_SHA256 = "2d1462e25bfc05523da2ac8a7487c21b08986ef3eb9fe2f0a2f9527c9b65618c"
WEIGHT_BF16_SHA256 = "cd1037a3eb9c98cf5baed570e6825e7bb2463165563707cf7f1c3b367afca557"
BIAS_F32_SHA256 = "6d264e0db1ffc55271a30cc6c8e60501effb4385b4200c007dc7a9c54d549b2f"
Q4_K_SHA256 = "418e4e461b5cc931b9af01e4f3c216f0da61e4fc02087c7deab425fe2e15d737"
FORMAT_IDS = {"q4_k": 0, "sparse": 1, "kchannel_weight": 2, "nrow_weight": 3}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, dtype: str, shape: list[int], root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "dtype": dtype,
        "shape": shape,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_descriptor(path: Path, format_id: int) -> None:
    """Write the 16-byte wire descriptor: magic, format ID, N, K."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4I", 0x31425157, format_id, OUTPUT_DIM, INPUT_DIM))


def copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    words = np.asarray(bits, dtype="<u2").astype(np.uint32) << 16
    return words.view(np.float32)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round F32 values to BF16, ties-to-even; used by small CPU spec tests only."""
    words = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = words + np.uint32(0x7FFF) + ((words >> 16) & 1)
    return (rounded >> 16).astype("<u2")


def _pack_affine_group(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.float32).reshape(BLOCK_VALUES)
    if not np.isfinite(values).all():
        raise ValueError("affine source contains a non-finite value")
    lower = np.float32(np.min(values))
    upper = np.float32(np.max(values))
    span = np.float32(upper - lower)
    scale = np.float32(span / np.float32(15.0)) if span != 0 else np.float32(0.0)
    if not np.isfinite(scale):
        raise ValueError("affine scale is non-finite")
    if scale == 0:
        codes = np.zeros(BLOCK_VALUES, dtype=np.uint8)
    else:
        delta = np.subtract(values, lower, dtype=np.float32)
        normalized = np.divide(delta, scale, dtype=np.float32)
        rounded = np.floor(np.add(normalized, np.float32(0.5), dtype=np.float32))
        codes = np.clip(rounded, 0, 15).astype(np.uint8)
    packed = (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8)
    return packed.tobytes() + struct.pack("<ff", float(scale), float(lower))


def _pack_affine_groups(values: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Pack a bounded batch of [groups,256] values using independent F32 ops."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != BLOCK_VALUES:
        raise ValueError("affine batch must have shape [groups,256]")
    if not np.isfinite(values).all():
        raise ValueError("affine source contains a non-finite value")
    lower = np.min(values, axis=1).astype(np.float32, copy=False)
    upper = np.max(values, axis=1).astype(np.float32, copy=False)
    span = np.subtract(upper, lower, dtype=np.float32)
    scale = np.divide(span, np.float32(15.0), dtype=np.float32)
    scale[span == 0] = np.float32(0.0)
    if not np.isfinite(scale).all():
        raise ValueError("affine scale is non-finite")
    delta = np.subtract(values, lower[:, None], dtype=np.float32)
    normalized = np.zeros_like(delta)
    np.divide(delta, scale[:, None], out=normalized, where=scale[:, None] != 0)
    rounded = np.floor(np.add(normalized, np.float32(0.5), dtype=np.float32))
    codes = np.clip(rounded, 0, 15).astype(np.uint8)
    packed = np.empty((values.shape[0], AFFINE_RECORD_BYTES), dtype=np.uint8)
    packed[:, :128] = codes[:, 0::2] | (codes[:, 1::2] << 4)
    packed[:, 128:132] = scale.astype("<f4", copy=False).view(np.uint8).reshape(-1, 4)
    packed[:, 132:136] = lower.astype("<f4", copy=False).view(np.uint8).reshape(-1, 4)
    scaled = np.multiply(codes.astype(np.float32), scale[:, None], dtype=np.float32)
    decoded = np.add(lower[:, None], scaled, dtype=np.float32)
    return packed.tobytes(), decoded


def _decode_affine_group(record: bytes | memoryview) -> np.ndarray:
    if len(record) != AFFINE_RECORD_BYTES:
        raise ValueError("affine record must be exactly 136 bytes")
    packed = np.frombuffer(record[:128], dtype=np.uint8)
    codes = np.empty(BLOCK_VALUES, dtype=np.float32)
    codes[0::2] = (packed & 0x0F).astype(np.float32)
    codes[1::2] = (packed >> 4).astype(np.float32)
    scale, offset = struct.unpack_from("<ff", record, 128)
    if not math.isfinite(scale) or not math.isfinite(offset) or scale < 0:
        raise ValueError("invalid affine scale/offset")
    scaled = np.multiply(codes, np.float32(scale), dtype=np.float32)
    return np.add(np.float32(offset), scaled, dtype=np.float32)


def encode_affine_base(
    source_bf16: np.memmap,
    base_path: Path,
    decoded_path: Path,
    element_count: int,
) -> None:
    if element_count != source_bf16.size or element_count % BLOCK_VALUES:
        raise ValueError("affine source must contain complete 256-value groups")
    decoded = np.memmap(decoded_path, dtype="<f4", mode="w+", shape=(element_count,))
    with base_path.open("wb") as stream:
        groups_per_chunk = 8192
        total_groups = element_count // BLOCK_VALUES
        for group_start in range(0, total_groups, groups_per_chunk):
            group_end = min(total_groups, group_start + groups_per_chunk)
            start = group_start * BLOCK_VALUES
            end = group_end * BLOCK_VALUES
            source = bf16_bits_to_f32(source_bf16[start:end]).reshape(-1, BLOCK_VALUES)
            records, decoded_chunk = _pack_affine_groups(source)
            stream.write(records)
            decoded[start:end] = decoded_chunk.reshape(-1)
    decoded.flush()
    del decoded
    if base_path.stat().st_size != AFFINE_BASE_BYTES:
        raise RuntimeError(f"unexpected affine base bytes: {base_path.stat().st_size}")


def decode_affine_region(
    payload: BinaryIO,
    base_bytes: int,
    destination: np.memmap,
    start_element: int = 0,
) -> None:
    if base_bytes < 0:
        raise ValueError("affine region byte count cannot be negative")
    if base_bytes % AFFINE_RECORD_BYTES:
        raise ValueError("affine region is not a whole number of groups")
    group_count = base_bytes // AFFINE_RECORD_BYTES
    raw = payload.read(base_bytes)
    if len(raw) != base_bytes:
        raise ValueError("short read from affine region")
    records = np.frombuffer(raw, dtype=np.uint8).reshape(group_count, AFFINE_RECORD_BYTES)
    packed = records[:, :128]
    codes = np.empty((group_count, BLOCK_VALUES), dtype=np.float32)
    codes[:, 0::2] = (packed & 0x0F).astype(np.float32)
    codes[:, 1::2] = (packed >> 4).astype(np.float32)
    scales = records[:, 128:132].copy().view("<f4").reshape(group_count)
    offsets = records[:, 132:136].copy().view("<f4").reshape(group_count)
    if not np.isfinite(scales).all() or not np.isfinite(offsets).all() or np.any(scales < 0):
        raise ValueError("invalid affine scale/offset")
    scaled = np.multiply(codes, scales[:, None], dtype=np.float32)
    decoded = np.add(offsets[:, None], scaled, dtype=np.float32)
    start = start_element
    end = start + group_count * BLOCK_VALUES
    if end > destination.size:
        raise ValueError("affine region exceeds destination")
    destination[start:end] = decoded.reshape(-1)


def _write_u32_array(stream: BinaryIO, values: np.ndarray) -> None:
    stream.write(np.asarray(values, dtype="<u4").tobytes())


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    """Top score indices, stable by descending F32 score then ascending index."""
    size = int(scores.size)
    if count < 0 or count > size:
        raise ValueError("selection count is outside the score array")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if not np.isfinite(scores).all():
        raise ValueError("selection scores contain non-finite values")
    threshold_pos = size - count
    threshold = np.partition(np.asarray(scores), threshold_pos)[threshold_pos]
    greater = np.flatnonzero(scores > threshold)
    equal = np.flatnonzero(scores == threshold)
    chosen = np.concatenate((greater, equal[: count - greater.size]))
    if chosen.size != count:
        raise RuntimeError("stable top-k selection returned the wrong count")
    return np.sort(chosen.astype(np.int64, copy=False))


def write_score_file(
    source_bf16: np.memmap,
    decoded_base: np.memmap,
    score_path: Path,
    element_count: int,
    activation_weights: np.ndarray | None = None,
) -> np.memmap:
    scores = np.memmap(score_path, dtype="<f4", mode="w+", shape=(element_count,))
    chunk_size = 1 << 20
    for start in range(0, element_count, chunk_size):
        end = min(element_count, start + chunk_size)
        original = bf16_bits_to_f32(source_bf16[start:end])
        residual = np.subtract(original, decoded_base[start:end], dtype=np.float32)
        score = np.multiply(residual, residual, dtype=np.float32)
        if activation_weights is not None:
            k_indices = np.arange(start, end, dtype=np.int64) % INPUT_DIM
            score = np.multiply(score, activation_weights[k_indices], dtype=np.float32)
        if not np.isfinite(score).all():
            raise ValueError("non-finite sparse score")
        scores[start:end] = score
    scores.flush()
    return scores


def calibration_weights(input_rows: np.ndarray, row_ids: tuple[int, int]) -> np.ndarray:
    values = np.asarray(input_rows[list(row_ids)], dtype=np.float32)
    squares = np.multiply(values, values, dtype=np.float32)
    total = np.add(squares[0], squares[1], dtype=np.float32)
    return np.multiply(total, np.float32(0.5), dtype=np.float32)


def write_selection_file(
    path: Path, indices: np.ndarray, dtype: str, root: Path
) -> dict[str, Any]:
    values = np.asarray(indices, dtype=dtype)
    values.tofile(path)
    return file_record(path, dtype, [values.size], root)


def build_sparse_payload(
    base_path: Path,
    payload_path: Path,
    selected: np.ndarray,
    source_bf16: np.memmap,
) -> dict[str, int]:
    if selected.size != PATCH_COUNT or np.any(selected[1:] <= selected[:-1]):
        raise ValueError("sparse patch indices must be the exact sorted budget")
    if selected.size and int(selected[-1]) >= source_bf16.size:
        raise ValueError("sparse patch index exceeds the weight tensor")
    groups = source_bf16.size // SPARSE_RECORD_VALUES
    group_ids = selected // SPARSE_RECORD_VALUES
    local_ids = selected % SPARSE_RECORD_VALUES
    counts = np.bincount(group_ids, minlength=groups).astype("<u4")
    offsets = np.empty(groups + 1, dtype="<u4")
    offsets[0] = 0
    np.cumsum(counts, dtype=np.uint64, out=offsets[1:])
    if int(offsets[-1]) != PATCH_COUNT:
        raise RuntimeError("sparse patch offsets do not end at the exact budget")
    patches = np.empty((selected.size, 2), dtype="<u2")
    patches[:, 0] = local_ids.astype("<u2")
    patches[:, 1] = source_bf16[selected].astype("<u2")
    with payload_path.open("wb") as stream:
        with base_path.open("rb") as base:
            shutil.copyfileobj(base, stream, 8 * 1024 * 1024)
        _write_u32_array(stream, counts)
        _write_u32_array(stream, offsets)
        stream.write(patches.tobytes())
    if payload_path.stat().st_size != Q4_K_BYTES:
        raise RuntimeError("sparse payload does not exactly match the Q4_K byte budget")
    return {
        "base_affine_bytes": AFFINE_BASE_BYTES,
        "record_count_u32_bytes": SPARSE_COUNTS_BYTES,
        "offset_directory_bytes": SPARSE_OFFSETS_BYTES,
        "absolute_bf16_patch_bytes": SPARSE_PATCH_BYTES,
        "patch_count": int(selected.size),
    }


def decode_sparse_payload(
    payload_path: Path,
    decoded_path: Path,
    source_bf16: np.memmap,
) -> None:
    if payload_path.stat().st_size != Q4_K_BYTES:
        raise ValueError("sparse payload has the wrong exact byte count")
    element_count = source_bf16.size
    decoded = np.memmap(decoded_path, dtype="<f4", mode="w+", shape=(element_count,))
    with payload_path.open("rb") as stream:
        decode_affine_region(stream, AFFINE_BASE_BYTES, decoded)
        counts = np.fromfile(stream, dtype="<u4", count=SPARSE_RECORDS)
        offsets = np.fromfile(stream, dtype="<u4", count=SPARSE_RECORDS + 1)
        if counts.size != SPARSE_RECORDS or offsets.size != SPARSE_RECORDS + 1:
            raise ValueError("short sparse count or offset directory")
        if offsets[0] != 0 or offsets[-1] != PATCH_COUNT or not np.array_equal(
            np.diff(offsets.astype(np.int64)), counts.astype(np.int64)
        ):
            raise ValueError("invalid sparse offsets/counts")
        patches = np.fromfile(stream, dtype="<u2", count=PATCH_COUNT * 2).reshape(-1, 2)
        if patches.shape[0] != PATCH_COUNT or stream.read(1):
            raise ValueError("sparse patch section has an invalid length")
        for group in range(SPARSE_RECORDS):
            first, last = int(offsets[group]), int(offsets[group + 1])
            if last <= first:
                continue
            local = patches[first:last, 0]
            if np.any(local >= SPARSE_RECORD_VALUES) or np.any(local[1:] <= local[:-1]):
                raise ValueError("sparse local positions must be unique and sorted")
            absolute = group * SPARSE_RECORD_VALUES + local.astype(np.int64)
            decoded[absolute] = bf16_bits_to_f32(patches[first:last, 1])
    decoded.flush()
    del decoded


def channel_scores(
    source_bf16: np.memmap,
    decoded_base: np.memmap,
    score_path: Path,
    axis: str,
) -> np.memmap:
    if axis not in ("k", "n"):
        raise ValueError("channel score axis must be k or n")
    extent = INPUT_DIM if axis == "k" else OUTPUT_DIM
    scores = np.memmap(score_path, dtype="<f4", mode="w+", shape=(extent,))
    scores[:] = 0
    original = np.asarray(source_bf16).reshape(OUTPUT_DIM, INPUT_DIM)
    decoded = np.asarray(decoded_base).reshape(OUTPUT_DIM, INPUT_DIM)
    if axis == "k":
        for n in range(OUTPUT_DIM):
            residual = np.subtract(bf16_bits_to_f32(original[n]), decoded[n], dtype=np.float32)
            squares = np.multiply(residual, residual, dtype=np.float32)
            np.add(scores, squares, out=scores, casting="unsafe")
    else:
        for n in range(OUTPUT_DIM):
            residual = np.subtract(bf16_bits_to_f32(original[n]), decoded[n], dtype=np.float32)
            squares = np.multiply(residual, residual, dtype=np.float32)
            scores[n] = np.cumsum(squares, dtype=np.float32)[-1]
    scores.flush()
    if not np.isfinite(scores).all():
        raise ValueError("non-finite channel score")
    return scores


def channel_capacity(kind: str) -> tuple[int, int, int]:
    if kind == "k":
        map_bytes = INPUT_DIM * 2
        candidates = range(1, INPUT_DIM + 1)
        best = 0
        for count in candidates:
            ids_bytes = ((count + 1) // 2) * 4
            total = AFFINE_BASE_BYTES + map_bytes + ids_bytes + OUTPUT_DIM * count * 2
            if total > Q4_K_BYTES:
                break
            best = count
        ids_bytes = ((best + 1) // 2) * 4
        used = AFFINE_BASE_BYTES + map_bytes + ids_bytes + OUTPUT_DIM * best * 2
        return best, used, Q4_K_BYTES - used
    if kind == "n":
        directory = (OUTPUT_DIM + 1) * 4
        base_row = (INPUT_DIM // BLOCK_VALUES) * AFFINE_RECORD_BYTES
        replacement_row = INPUT_DIM * 2
        per_row_increment = replacement_row - base_row
        count = (Q4_K_BYTES - directory - AFFINE_BASE_BYTES) // per_row_increment
        used = directory + AFFINE_BASE_BYTES + count * per_row_increment
        return count, used, Q4_K_BYTES - used
    raise ValueError("channel capacity kind must be k or n")


def build_kchannel_payload(
    base_path: Path,
    payload_path: Path,
    selected_k: np.ndarray,
    source_bf16: np.memmap,
) -> dict[str, int]:
    count, used, padding = channel_capacity("k")
    if selected_k.size != count or np.any(selected_k[1:] <= selected_k[:-1]) or np.any(selected_k >= INPUT_DIM):
        raise RuntimeError("wrong K-channel count for exact byte fit")
    mapping = np.full(INPUT_DIM, 0xFFFF, dtype="<u2")
    mapping[selected_k] = np.arange(count, dtype="<u2")
    padded_ids = np.zeros((count + 1) // 2 * 2, dtype="<u2")
    padded_ids[:count] = selected_k.astype("<u2")
    weight_bits = np.asarray(source_bf16).reshape(OUTPUT_DIM, INPUT_DIM)[:, selected_k]
    with payload_path.open("wb") as stream:
        with base_path.open("rb") as base:
            shutil.copyfileobj(base, stream, 8 * 1024 * 1024)
        stream.write(mapping.tobytes())
        stream.write(padded_ids.tobytes())
        stream.write(np.asarray(weight_bits, dtype="<u2").tobytes(order="C"))
        if padding:
            stream.write(bytes(padding))
    if payload_path.stat().st_size != Q4_K_BYTES:
        raise RuntimeError("K-channel payload does not match the Q4_K byte budget")
    return {
        "base_affine_bytes": AFFINE_BASE_BYTES,
        "channel_to_ordinal_u16_bytes": INPUT_DIM * 2,
        "ascending_channel_ids_u16_padded_bytes": padded_ids.nbytes,
        "absolute_bf16_sidecar_bytes": weight_bits.nbytes,
        "zero_padding_bytes": padding,
        "selected_channels": count,
        "payload_used_before_padding_bytes": used,
    }


def decode_kchannel_payload(
    payload_path: Path,
    decoded_path: Path,
    source_bf16: np.memmap,
) -> None:
    if payload_path.stat().st_size != Q4_K_BYTES:
        raise ValueError("K-channel payload has the wrong exact byte count")
    element_count = source_bf16.size
    count, _, _ = channel_capacity("k")
    decoded = np.memmap(decoded_path, dtype="<f4", mode="w+", shape=(element_count,))
    with payload_path.open("rb") as stream:
        decode_affine_region(stream, AFFINE_BASE_BYTES, decoded)
        mapping = np.fromfile(stream, dtype="<u2", count=INPUT_DIM)
        ids_padded = np.fromfile(stream, dtype="<u2", count=((count + 1) // 2) * 2)
        ids = ids_padded[:count].astype(np.int64)
        sidecar = np.fromfile(stream, dtype="<u2", count=OUTPUT_DIM * count).reshape(OUTPUT_DIM, count)
        if mapping.size != INPUT_DIM or sidecar.shape != (OUTPUT_DIM, count):
            raise ValueError("short K-channel map or sidecar")
        if not np.array_equal(ids, np.sort(ids)) or np.any(ids >= INPUT_DIM):
            raise ValueError("K-channel IDs are not ascending valid channels")
        if np.any(ids_padded[count:] != 0):
            raise ValueError("K-channel ID padding must be zero")
        expected_map = np.full(INPUT_DIM, 0xFFFF, dtype="<u2")
        expected_map[ids] = np.arange(count, dtype="<u2")
        if not np.array_equal(mapping, expected_map):
            raise ValueError("K-channel ordinal map does not match IDs")
        if stream.read() != bytes(channel_capacity("k")[2]):
            raise ValueError("K-channel payload padding is not all zero")
        decoded_matrix = decoded.reshape(OUTPUT_DIM, INPUT_DIM)
        decoded_matrix[:, ids] = bf16_bits_to_f32(sidecar)
    decoded.flush()
    del decoded


def build_nrow_payload(
    base_path: Path,
    payload_path: Path,
    selected_n: np.ndarray,
    source_bf16: np.memmap,
) -> dict[str, int]:
    count, used, padding = channel_capacity("n")
    if selected_n.size != count or np.any(selected_n[1:] <= selected_n[:-1]) or np.any(selected_n >= OUTPUT_DIM):
        raise RuntimeError("wrong N-row count for exact byte fit")
    selected_mask = np.zeros(OUTPUT_DIM, dtype=np.bool_)
    selected_mask[selected_n] = True
    base_row_bytes = (INPUT_DIM // BLOCK_VALUES) * AFFINE_RECORD_BYTES
    replacement_row_bytes = INPUT_DIM * 2
    directory_bytes = (OUTPUT_DIM + 1) * 4
    offsets = np.empty(OUTPUT_DIM + 1, dtype="<u4")
    offsets[0] = directory_bytes
    for row in range(OUTPUT_DIM):
        row_bytes = replacement_row_bytes if selected_mask[row] else base_row_bytes
        offsets[row + 1] = offsets[row] + row_bytes
    if int(offsets[-1]) != used:
        raise RuntimeError("N-row directory does not end at the exact used payload size")
    with payload_path.open("wb") as stream:
        _write_u32_array(stream, offsets)
        with base_path.open("rb") as base:
            for row in range(OUTPUT_DIM):
                if selected_mask[row]:
                    first = row * INPUT_DIM
                    stream.write(np.asarray(source_bf16[first : first + INPUT_DIM], dtype="<u2").tobytes())
                else:
                    base.seek(row * base_row_bytes)
                    row_payload = base.read(base_row_bytes)
                    if len(row_payload) != base_row_bytes:
                        raise ValueError("short affine row while building N-row payload")
                    stream.write(row_payload)
        if padding:
            stream.write(bytes(padding))
    if payload_path.stat().st_size != Q4_K_BYTES:
        raise RuntimeError("N-row payload does not match the Q4_K byte budget")
    return {
        "absolute_payload_offsets_u32_bytes": directory_bytes,
        "affine_base_bytes_before_row_replacements": AFFINE_BASE_BYTES,
        "affine_row_bytes": base_row_bytes,
        "absolute_bf16_row_bytes": replacement_row_bytes,
        "selected_rows": count,
        "net_replacement_increment_bytes": count * (replacement_row_bytes - base_row_bytes),
        "zero_padding_bytes": padding,
        "payload_used_before_padding_bytes": used,
    }


def decode_nrow_payload(
    payload_path: Path,
    decoded_path: Path,
    source_bf16: np.memmap,
) -> np.ndarray:
    if payload_path.stat().st_size != Q4_K_BYTES:
        raise ValueError("N-row payload has the wrong exact byte count")
    element_count = source_bf16.size
    decoded = np.memmap(decoded_path, dtype="<f4", mode="w+", shape=(element_count,))
    base_row_bytes = (INPUT_DIM // BLOCK_VALUES) * AFFINE_RECORD_BYTES
    replacement_row_bytes = INPUT_DIM * 2
    directory_bytes = (OUTPUT_DIM + 1) * 4
    selected: list[int] = []
    with payload_path.open("rb") as stream:
        offsets = np.fromfile(stream, dtype="<u4", count=OUTPUT_DIM + 1)
        if (
            offsets.size != OUTPUT_DIM + 1
            or offsets[0] != directory_bytes
            or int(offsets[-1]) != channel_capacity("n")[1]
            or np.any(offsets[1:] <= offsets[:-1])
        ):
            raise ValueError("invalid N-row absolute offset directory")
        for row in range(OUTPUT_DIM):
            first, last = int(offsets[row]), int(offsets[row + 1])
            byte_count = last - first
            if byte_count not in (base_row_bytes, replacement_row_bytes):
                raise ValueError("invalid N-row record length")
            stream.seek(first)
            element_start = row * INPUT_DIM
            if byte_count == replacement_row_bytes:
                raw = np.fromfile(stream, dtype="<u2", count=INPUT_DIM)
                if raw.size != INPUT_DIM:
                    raise ValueError("short BF16 replacement row")
                decoded[element_start : element_start + INPUT_DIM] = bf16_bits_to_f32(raw)
                selected.append(row)
            else:
                decode_affine_region(stream, base_row_bytes, decoded, element_start)
        payload_end = int(offsets[-1])
        stream.seek(payload_end)
        if stream.read() != bytes(channel_capacity("n")[2]):
            raise ValueError("N-row payload padding is not all zero")
    decoded.flush()
    del decoded
    return np.asarray(selected, dtype=np.uint16)


def validate_fixture(fixture_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest_path = fixture_dir / "manifest.json"
    if sha256_file(manifest_path) != FIXTURE_MANIFEST_SHA256:
        raise RuntimeError("pinned fixture manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "wan_quantized_projection_reference_v1":
        raise RuntimeError("unsupported source fixture schema")
    geometry = manifest["geometry"]
    if (geometry.get("source_rows"), geometry.get("input_dim"), geometry.get("output_dim")) != (
        SOURCE_ROWS,
        INPUT_DIM,
        OUTPUT_DIM,
    ):
        raise RuntimeError("source fixture geometry does not match the pinned Wan projection")
    if manifest["comparison"].get("atol") != 2e-5 or manifest["comparison"].get("rtol") != 2e-5:
        raise RuntimeError("source fixture uses a different numerical gate")
    if manifest["oracle"].get("device") != "MPS" or not manifest["oracle"].get("fallback_disabled"):
        raise RuntimeError("source fixture is not an MPS reference with fallback disabled")
    paths = {
        "input": fixture_dir / manifest["input"]["path"],
        "bias": fixture_dir / manifest["source_tensor"]["bias_f32_file"]["path"],
        "original_weight": fixture_dir / manifest["source_tensor"]["weight_file"]["path"],
        "q4_payload": fixture_dir / manifest["variants"]["q4_k"]["packed_weight"]["path"],
        "bf16_decoded": fixture_dir / manifest["variants"]["bf16"]["decoded_weight"]["path"],
        "bf16_projection": fixture_dir / manifest["variants"]["bf16"]["expected_projection"]["path"],
        "q4_decoded": fixture_dir / manifest["variants"]["q4_k"]["decoded_weight"]["path"],
        "q4_projection": fixture_dir / manifest["variants"]["q4_k"]["expected_projection"]["path"],
    }
    expected = {
        "input": INPUT_SHA256,
        "bias": BIAS_F32_SHA256,
        "original_weight": WEIGHT_BF16_SHA256,
        "q4_payload": Q4_K_SHA256,
    }
    for key, digest in expected.items():
        if sha256_file(paths[key]) != digest:
            raise RuntimeError(f"pinned source {key} SHA-256 mismatch")
    if paths["input"].stat().st_size != SOURCE_ROWS * INPUT_DIM * 4:
        raise RuntimeError("pinned F32 input has an unexpected size")
    if paths["bias"].stat().st_size != SHARED_BIAS_BYTES:
        raise RuntimeError("pinned F32 bias has an unexpected size")
    if paths["original_weight"].stat().st_size != OUTPUT_DIM * INPUT_DIM * 2:
        raise RuntimeError("pinned BF16 weight has an unexpected size")
    if paths["q4_payload"].stat().st_size != Q4_K_BYTES:
        raise RuntimeError("pinned Q4_K payload has an unexpected size")
    linked_records = {
        "input": manifest["input"],
        "bias": manifest["source_tensor"]["bias_f32_file"],
        "original_weight": manifest["source_tensor"]["weight_file"],
        "q4_payload": manifest["variants"]["q4_k"]["packed_weight"],
        "bf16_decoded": manifest["variants"]["bf16"]["decoded_weight"],
        "bf16_projection": manifest["variants"]["bf16"]["expected_projection"],
        "q4_decoded": manifest["variants"]["q4_k"]["decoded_weight"],
        "q4_projection": manifest["variants"]["q4_k"]["expected_projection"],
    }
    for key, record in linked_records.items():
        if paths[key].stat().st_size != record["bytes"] or sha256_file(paths[key]) != record["sha256"]:
            raise RuntimeError(f"pinned source fixture record mismatch: {key}")
    return manifest, paths


def source_records(
    source_paths: dict[str, Path], stage: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    names = {"input": "input.f32", "bias": "bias.f32", "original_weight": "original_weight.bf16"}
    records: dict[str, Any] = {}
    copied: dict[str, Path] = {}
    for key, name in names.items():
        destination = stage / "source" / name
        copy_or_link(source_paths[key], destination)
        copied[key] = destination
    records["input"] = file_record(copied["input"], "F32", [SOURCE_ROWS, INPUT_DIM], stage)
    records["bias"] = file_record(copied["bias"], "F32", [OUTPUT_DIM], stage)
    records["original_weight"] = file_record(
        copied["original_weight"], "BF16", [OUTPUT_DIM, INPUT_DIM], stage
    )
    return records, copied


def reference_record(
    name: str,
    source_path: Path,
    decoded_source: Path,
    projection_source: Path,
    rows: int,
    stage: Path,
    input_path: Path,
    bias_path: Path,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    out_dir = stage / name
    out_dir.mkdir(parents=True, exist_ok=True)
    decoded_path = out_dir / "decoded_weight.f32"
    projection_path = out_dir / "expected_projection.f32"
    copy_or_link(decoded_source, decoded_path)
    versions: dict[str, str] | None = None
    if rows == BENCH_ROWS:
        copy_or_link(projection_source, projection_path)
    else:
        versions = write_candidate_projection(decoded_path, input_path, bias_path, projection_path, rows)
    return (
        {
            "decoded_weight": file_record(decoded_path, "F32", [OUTPUT_DIM, INPUT_DIM], stage),
            "expected_projection": file_record(projection_path, "F32", [rows, OUTPUT_DIM], stage),
            "source_payload_sha256": sha256_file(source_path) if source_path.is_file() else None,
            "projection_reference": "pinned same-geometry MPS fixture" if rows == BENCH_ROWS else "fresh same-geometry MPS F.linear",
        },
        versions,
    )


def write_candidate_projection(
    decoded_path: Path,
    input_path: Path,
    bias_path: Path,
    projection_path: Path,
    rows: int,
) -> dict[str, str]:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch
    import torch.nn.functional as F

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for same-format projection references")
    source = np.memmap(input_path, dtype="<f4", mode="r", shape=(SOURCE_ROWS, INPUT_DIM))
    input_cpu = np.ascontiguousarray(source[np.arange(rows) % SOURCE_ROWS])
    input_mps = torch.from_numpy(input_cpu).to(device="mps", dtype=torch.float32)
    bias_cpu = np.fromfile(bias_path, dtype="<f4")
    bias_mps = torch.from_numpy(bias_cpu).to(device="mps", dtype=torch.float32)
    decoded_cpu = np.array(
        np.memmap(decoded_path, dtype="<f4", mode="r", shape=(OUTPUT_DIM, INPUT_DIM)),
        copy=True,
        order="C",
    )
    decoded_mps = torch.from_numpy(decoded_cpu).to(device="mps", dtype=torch.float32)
    result = F.linear(input_mps, decoded_mps, bias_mps)
    if result.device.type != "mps" or result.dtype != torch.float32 or tuple(result.shape) != (rows, OUTPUT_DIM):
        raise RuntimeError(f"unexpected MPS F.linear output: {result.device} {result.dtype} {tuple(result.shape)}")
    output_cpu = result.to("cpu").contiguous().numpy()
    with projection_path.open("wb") as stream:
        output_cpu.astype("<f4", copy=False).tofile(stream)
    torch.mps.synchronize()
    versions = {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version}
    del result, output_cpu, decoded_mps, bias_mps, input_mps
    try:
        torch.mps.empty_cache()
    except (AttributeError, RuntimeError):
        pass
    return versions


def distortion_metrics(
    source_bf16: np.memmap,
    decoded_path: Path,
    input_rows: np.ndarray,
) -> dict[str, Any]:
    decoded = np.memmap(decoded_path, dtype="<f4", mode="r", shape=(OUTPUT_DIM, INPUT_DIM))
    total_sq = 0.0
    total_ref_sq = 0.0
    max_abs = 0.0
    weighted = [0.0, 0.0]
    calibration = [calibration_weights(input_rows, (0, 1)), calibration_weights(input_rows, (2, 3))]
    for first in range(0, OUTPUT_DIM, 32):
        last = min(OUTPUT_DIM, first + 32)
        source = bf16_bits_to_f32(np.asarray(source_bf16[first * INPUT_DIM : last * INPUT_DIM]).reshape(last - first, INPUT_DIM))
        estimate = decoded[first:last]
        error = np.subtract(source, estimate, dtype=np.float32)
        squares = np.multiply(error, error, dtype=np.float32)
        total_sq += float(np.sum(squares, dtype=np.float64))
        total_ref_sq += float(np.sum(np.multiply(source, source, dtype=np.float32), dtype=np.float64))
        max_abs = max(max_abs, float(np.max(np.abs(error))))
        for fold in range(2):
            weighted_sq = np.multiply(squares, calibration[fold][None, :], dtype=np.float32)
            weighted[fold] += float(np.sum(weighted_sq, dtype=np.float64))
    del decoded
    values = OUTPUT_DIM * INPUT_DIM
    return {
        "weight_error": {
            "elements": values,
            "max_abs": max_abs,
            "rmse": math.sqrt(total_sq / values),
            "relative_l2": math.sqrt(total_sq / max(total_ref_sq, 1e-300)),
            "squared_l2": total_sq,
        },
        "activation_weighted_error": {
            "fold0_train_rows": [0, 1],
            "fold0_weighted_squared_error": weighted[0],
            "fold1_train_rows": [2, 3],
            "fold1_weighted_squared_error": weighted[1],
        },
    }


def projection_quality(candidate_path: Path, baseline_path: Path, rows: int) -> dict[str, Any]:
    count = rows * OUTPUT_DIM
    candidate = np.memmap(candidate_path, dtype="<f4", mode="r", shape=(count,))
    baseline = np.memmap(baseline_path, dtype="<f4", mode="r", shape=(count,))
    total_sq = 0.0
    total_ref_sq = 0.0
    max_abs = 0.0
    violations = 0
    for start in range(0, count, 1 << 20):
        end = min(count, start + (1 << 20))
        a = np.asarray(candidate[start:end])
        b = np.asarray(baseline[start:end])
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("non-finite projection in quality comparison")
        error = np.subtract(a, b, dtype=np.float32)
        abs_error = np.abs(error)
        total_sq += float(np.sum(np.multiply(error, error, dtype=np.float32), dtype=np.float64))
        total_ref_sq += float(np.sum(np.multiply(b, b, dtype=np.float32), dtype=np.float64))
        max_abs = max(max_abs, float(np.max(abs_error)))
        violations += int(np.count_nonzero(abs_error > (2e-5 + 2e-5 * np.abs(b))))
    del candidate, baseline
    return {
        "compared_values": count,
        "max_abs_vs_bf16": max_abs,
        "rmse_vs_bf16": math.sqrt(total_sq / count),
        "relative_l2_vs_bf16": math.sqrt(total_sq / max(total_ref_sq, 1e-300)),
        "values_exceeding_reference_scale_tolerance_vs_bf16": violations,
        "note": "This is a descriptive lossy-quality comparison; the same-format parity gate applies against this arm's own MPS reference.",
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_fixture(args: argparse.Namespace) -> Path:
    if args.rows not in (SOURCE_ROWS, BENCH_ROWS):
        raise ValueError("--rows must be 4 or 4480")
    fixture_dir = args.fixture_dir.expanduser().resolve()
    output_path = args.output_dir.expanduser().resolve()
    if os.path.lexists(output_path):
        raise FileExistsError(f"immutable output already exists: {output_path}")
    source_manifest, source_paths = validate_fixture(fixture_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=output_path.parent))
    try:
        sources, copied_source_paths = source_records(source_paths, stage)
        input_rows = np.memmap(copied_source_paths["input"], dtype="<f4", mode="r", shape=(SOURCE_ROWS, INPUT_DIM))
        source_bf16 = np.memmap(
            copied_source_paths["original_weight"], dtype="<u2", mode="r", shape=(OUTPUT_DIM * INPUT_DIM,)
        )

        baseline, baseline_torch_versions = reference_record(
            "baseline",
            source_paths["original_weight"],
            source_paths["bf16_decoded"],
            source_paths["bf16_projection"],
            args.rows,
            stage,
            copied_source_paths["input"],
            copied_source_paths["bias"],
        )
        q4, q4_torch_versions = reference_record(
            "variants/q4_k",
            source_paths["q4_payload"],
            source_paths["q4_decoded"],
            source_paths["q4_projection"],
            args.rows,
            stage,
            copied_source_paths["input"],
            copied_source_paths["bias"],
        )
        q4_payload_copy = stage / "variants" / "q4_k" / "payload.bin"
        copy_or_link(source_paths["q4_payload"], q4_payload_copy)
        q4["payload"] = file_record(q4_payload_copy, "Q4_K", [OUTPUT_DIM, INPUT_DIM // BLOCK_VALUES, 144], stage)
        q4["mode"] = -1
        q4["fold"] = 0
        q4_descriptor = stage / "variants" / "q4_k" / "descriptor.bin"
        write_descriptor(q4_descriptor, FORMAT_IDS["q4_k"])
        q4["descriptor"] = file_record(q4_descriptor, "U32LE", [4], stage)
        q4["format_id"] = FORMAT_IDS["q4_k"]
        q4["components"] = {"canonical_q4_k_payload_bytes": Q4_K_BYTES}
        q4["selection"] = {"producer": "pinned ggml quantize_row_q4_K_ref fixture", "selection": "not applicable"}
        q4["distortion"] = distortion_metrics(
            source_bf16, stage / q4["decoded_weight"]["path"], input_rows
        )
        q4["quality_vs_original_bf16"] = projection_quality(
            stage / q4["expected_projection"]["path"],
            stage / "baseline" / "expected_projection.f32",
            args.rows,
        )

        work = Path(tempfile.mkdtemp(prefix=".affine-work-", dir=stage))
        base_path = work / "affine-base.bin"
        base_decoded_path = work / "affine-base.f32"
        encode_affine_base(source_bf16, base_path, base_decoded_path, OUTPUT_DIM * INPUT_DIM)
        decoded_base = np.memmap(base_decoded_path, dtype="<f4", mode="r", shape=(OUTPUT_DIM * INPUT_DIM,))

        variants: dict[str, dict[str, Any]] = {"q4_k": q4}
        torch_versions: dict[str, str] | None = baseline_torch_versions or q4_torch_versions

        def finish_variant(
            name: str,
            mode: int,
            fold: int,
            payload_path: Path,
            decoded_path: Path,
            selection: dict[str, Any],
            components: dict[str, Any],
        ) -> None:
            nonlocal torch_versions
            output_dir = stage / "variants" / name
            output_dir.mkdir(parents=True, exist_ok=True)
            expected_path = output_dir / "expected_projection.f32"
            format_key = "sparse" if mode in (0, 1) else name.rsplit("/", 1)[-1]
            format_id = FORMAT_IDS[format_key]
            descriptor_path = output_dir / "descriptor.bin"
            write_descriptor(descriptor_path, format_id)
            version_record = write_candidate_projection(
                decoded_path,
                copied_source_paths["input"],
                copied_source_paths["bias"],
                expected_path,
                args.rows,
            )
            torch_versions = torch_versions or version_record
            output = {
                "mode": mode,
                "fold": fold,
                "format_id": format_id,
                "descriptor": file_record(descriptor_path, "U32LE", [4], stage),
                "payload": file_record(payload_path, "AffineBudgetPayload", [payload_path.stat().st_size], stage),
                "decoded_weight": file_record(decoded_path, "F32", [OUTPUT_DIM, INPUT_DIM], stage),
                "expected_projection": file_record(expected_path, "F32", [args.rows, OUTPUT_DIM], stage),
                "components": components,
                "selection": selection,
                "distortion": distortion_metrics(source_bf16, decoded_path, input_rows),
                "quality_vs_original_bf16": projection_quality(
                    expected_path, stage / "baseline" / "expected_projection.f32", args.rows
                ),
                "decode_contract": "F32 scale*code followed by F32 offset+scaled; absolute BF16 replacements",
            }
            variants[name.rsplit("/", 1)[-1]] = output

        for name, mode, fold, train_rows, activation_weighted in (
            ("sparse_weight", 0, 0, None, False),
            ("sparse_activation_f0", 1, 0, (0, 1), True),
            ("sparse_activation_f1", 1, 1, (2, 3), True),
        ):
            output_dir = stage / "variants" / name
            output_dir.mkdir(parents=True, exist_ok=True)
            payload_path = output_dir / "payload.bin"
            decoded_path = output_dir / "decoded_weight.f32"
            score_path = work / f"score-{name}.f32"
            weights = None if train_rows is None else calibration_weights(input_rows, train_rows)
            scores = write_score_file(source_bf16, decoded_base, score_path, source_bf16.size, weights)
            selected = top_indices(scores, PATCH_COUNT)
            selected_path = output_dir / "selected-flat-indices.u32"
            selected_record = write_selection_file(selected_path, selected, "<u4", stage)
            components = build_sparse_payload(base_path, payload_path, selected, source_bf16)
            decode_sparse_payload(payload_path, decoded_path, source_bf16)
            threshold = float(np.partition(np.asarray(scores), scores.size - PATCH_COUNT)[scores.size - PATCH_COUNT])
            finish_variant(
                f"variants/{name}",
                mode,
                fold,
                payload_path,
                decoded_path,
                {
                    "selector": "descending F32 squared affine residual; ascending flat-index tie-break"
                    if not activation_weighted
                    else "descending F32 residual_squared * mean(train_input_squared); ascending flat-index tie-break",
                    "train_input_rows": list(train_rows) if train_rows else None,
                    "selected_count": PATCH_COUNT,
                    "score_threshold": threshold,
                    "selected_indices": selected_record,
                    "selected_indices_sha256": _array_hash(selected.astype("<u4")),
                    "tie_rule": "equal score selects lower flat index",
                },
                components,
            )
            del scores
            score_path.unlink()

        k_count, _, _ = channel_capacity("k")
        k_scores_path = work / "score-k.f32"
        k_scores = channel_scores(source_bf16, decoded_base, k_scores_path, "k")
        selected_k = top_indices(k_scores, k_count)
        k_dir = stage / "variants" / "kchannel_weight"
        k_dir.mkdir(parents=True, exist_ok=True)
        k_payload = k_dir / "payload.bin"
        k_decoded = k_dir / "decoded_weight.f32"
        k_selected_record = write_selection_file(k_dir / "selected-k-indices.u16", selected_k, "<u2", stage)
        k_components = build_kchannel_payload(base_path, k_payload, selected_k, source_bf16)
        decode_kchannel_payload(k_payload, k_decoded, source_bf16)
        k_threshold = float(np.partition(np.asarray(k_scores), k_scores.size - k_count)[k_scores.size - k_count])
        finish_variant(
            "variants/kchannel_weight",
            2,
            0,
            k_payload,
            k_decoded,
            {
                "selector": "descending F32 sum of squared residual over N in ascending order; lower K tie-break",
                "selected_count": k_count,
                "score_threshold": k_threshold,
                "selected_indices": k_selected_record,
                "tie_rule": "equal score selects lower K index",
            },
            k_components,
        )
        del k_scores
        k_scores_path.unlink()

        n_count, _, _ = channel_capacity("n")
        n_scores_path = work / "score-n.f32"
        n_scores = channel_scores(source_bf16, decoded_base, n_scores_path, "n")
        selected_n = top_indices(n_scores, n_count)
        n_dir = stage / "variants" / "nrow_weight"
        n_dir.mkdir(parents=True, exist_ok=True)
        n_payload = n_dir / "payload.bin"
        n_decoded = n_dir / "decoded_weight.f32"
        n_selected_record = write_selection_file(n_dir / "selected-n-indices.u16", selected_n, "<u2", stage)
        n_components = build_nrow_payload(base_path, n_payload, selected_n, source_bf16)
        decoded_selected = decode_nrow_payload(n_payload, n_decoded, source_bf16)
        if not np.array_equal(decoded_selected, selected_n.astype(np.uint16)):
            raise RuntimeError("N-row directory selected rows differ from independent selection")
        n_threshold = float(np.partition(np.asarray(n_scores), n_scores.size - n_count)[n_scores.size - n_count])
        finish_variant(
            "variants/nrow_weight",
            3,
            0,
            n_payload,
            n_decoded,
            {
                "selector": "descending F32 sum of squared residual over K in ascending order; lower row tie-break",
                "selected_count": n_count,
                "score_threshold": n_threshold,
                "selected_indices": n_selected_record,
                "tie_rule": "equal score selects lower output row index",
            },
            n_components,
        )
        del n_scores
        n_scores_path.unlink()
        del decoded_base, source_bf16, input_rows
        shutil.rmtree(work)

        manifest = {
            "schema": "wan_affine_budget_reference_v1",
            "created_by": Path(__file__).name,
            "geometry": {
                "rows": args.rows,
                "source_rows": SOURCE_ROWS,
                "input_dim": INPUT_DIM,
                "output_dim": OUTPUT_DIM,
                "input_layout": "row_major periodic source-row tile",
            },
            "comparison": {
                "dtype": "F32",
                "atol": 2e-5,
                "rtol": 2e-5,
                "finite": True,
                "predicate": "finite(actual) && abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)",
                "scope": "native decode/project parity against each payload's independently decoded MPS F.linear reference",
                "lossy_quality_scope": "candidate quality relative to original BF16 is separate from same-format parity",
            },
            "oracle": {
                "device": "MPS",
                "fallback_disabled": True,
                "projection": "torch.nn.functional.linear(input, decoded_weight, bias)",
                "torch_version": torch_versions["torch"] if torch_versions else None,
                "numpy_version": np.__version__,
                "python_version": sys.version,
            },
            "budget": {
                "payload_bytes_per_arm": Q4_K_BYTES,
                "external_descriptor_bytes_per_arm": FORMAT_DESCRIPTOR_BYTES,
                "total_candidate_bytes_per_arm": Q4_K_BYTES + FORMAT_DESCRIPTOR_BYTES,
                "shared_bias_bytes": SHARED_BIAS_BYTES,
                "all_arms_have_same_descriptor_size": True,
                "descriptor_words_u32le": ["magic_WQB1", "format_id", "output_dim", "input_dim"],
                "descriptor_format_ids": FORMAT_IDS,
                "q4_k_payload_bytes": Q4_K_BYTES,
            },
            "source_fixture": {
                "schema": source_manifest["schema"],
                "manifest_sha256": FIXTURE_MANIFEST_SHA256,
                "real_video_derived": source_manifest.get("fixture_policy", {}).get("input_is_real_video_derived"),
                "checkpoint_weight_sha256": WEIGHT_BF16_SHA256,
                "input_sha256": INPUT_SHA256,
                "bias_f32_sha256": BIAS_F32_SHA256,
                "q4_k_sha256": Q4_K_SHA256,
            },
            "input": sources["input"],
            "bias": sources["bias"],
            "original_weight": sources["original_weight"],
            "baseline": {
                "decoded_weight": baseline["decoded_weight"],
                "expected_projection": baseline["expected_projection"],
                "projection_reference": baseline["projection_reference"],
                "control": f"original BF16 source tensor converted to F32; {baseline['projection_reference']}",
            },
            "variants": variants,
            "selection_metrics": {
                "weight_error": "squared residual from the independently decoded affine base",
                "activation_weighted_error": "squared residual multiplied by mean x^2 over the selected two training rows",
                "activation_folds": {"f0": {"train_rows": [0, 1], "heldout_rows": [2, 3]}, "f1": {"train_rows": [2, 3], "heldout_rows": [0, 1]}},
                "channel_error": "weight-only squared residual, F32 sequential accumulation in ascending reduction index",
            },
            "fixture_policy": {
                "immutable_output": True,
                "full_wan_model_loaded": False,
                "real_video_quality_claim": False,
                "source_rows_are_periodically_tiled_when_rows_4480": True,
            },
        }
        write_json(stage / "manifest.json", manifest)
        if os.path.lexists(output_path):
            raise FileExistsError(f"immutable output appeared while preparing: {output_path}")
        os.replace(stage, output_path)
        return output_path
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True, help="pinned wan_quantized_projection_reference_v1 fixture")
    parser.add_argument("--output-dir", type=Path, required=True, help="new immutable output fixture directory")
    parser.add_argument("--rows", type=int, choices=(SOURCE_ROWS, BENCH_ROWS), required=True)
    return parser


def main() -> None:
    output = build_fixture(make_parser().parse_args())
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
