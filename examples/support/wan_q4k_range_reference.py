#!/usr/bin/env python3
"""Build independent CPU Q4_K range ablation references for one Wan projection.

The pinned ggml Q4_K payload is retained as the canonical arm. The two
min/max arms use this file's CPU serializer, and all three payloads are decoded
again by an independent CPU wire reader before MPS F.linear projection
references are produced. The script never loads a Wan model.
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
from typing import Any

import numpy as np

import wan_affine_budget_reference as affine_reference

SOURCE_ROWS = 4
INPUT_DIM = 5120
OUTPUT_DIM = 13824
BENCH_ROWS = 4480
BLOCK_VALUES = 256
GROUP_VALUES = 32
GROUPS_PER_BLOCK = BLOCK_VALUES // GROUP_VALUES
Q4_K_BLOCK_BYTES = 144
Q4_K_BYTES = 39_813_120
Q4_K_BLOCKS = OUTPUT_DIM * INPUT_DIM // BLOCK_VALUES
FORMAT_DESCRIPTOR_BYTES = 16
SHARED_BIAS_BYTES = OUTPUT_DIM * 4
RANGE_ARMS = {"q4_k": 8, "q4_k_minmax8": 8, "q4_k_minmax1": 1}
FORMAT_IDS = {name: 0 for name in RANGE_ARMS}


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


def _bounds_for_groups(values: np.ndarray, ranges: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit zero-inclusive endpoints for 1x256 or 8x32 independent ranges."""
    if ranges not in (1, GROUPS_PER_BLOCK):
        raise ValueError("ranges must be 1 or 8")
    grouped = values.reshape(values.shape[0], GROUPS_PER_BLOCK, GROUP_VALUES)
    if ranges == 1:
        grouped = values.reshape(values.shape[0], 1, BLOCK_VALUES)

    # argmin/argmax make equal-valued signed zeros deterministic: the first
    # source element wins. A strict comparison when adding zero preserves its
    # sign on ties; both zero-valued superscales are canonicalized below.
    lo_index = np.argmin(grouped, axis=-1)
    hi_index = np.argmax(grouped, axis=-1)
    lo = np.take_along_axis(grouped, lo_index[..., None], axis=-1)[..., 0]
    hi = np.take_along_axis(grouped, hi_index[..., None], axis=-1)[..., 0]
    zero = np.float32(0.0)
    lo = np.where(lo > zero, zero, lo).astype(np.float32, copy=False)
    hi = np.where(hi < zero, zero, hi).astype(np.float32, copy=False)
    if ranges == 1:
        lo = np.broadcast_to(lo, (values.shape[0], GROUPS_PER_BLOCK))
        hi = np.broadcast_to(hi, (values.shape[0], GROUPS_PER_BLOCK))
    return lo, hi


def _scalar_superscale(values: np.ndarray, label: str) -> np.float16:
    """Scalar control conversion, kept separate from the vectorized writer."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not values.size or not np.isfinite(values).all() or np.any(values < np.float32(0.0)):
        raise ValueError(f"{label} values are empty, non-finite, or negative")
    maximum = np.float32(np.max(values))
    raw = np.divide(maximum, np.float32(63.0), dtype=np.float32)
    if not np.isfinite(raw):
        raise ValueError(f"{label} superscale is non-finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        half = np.asarray(raw, dtype=np.float32).astype("<f2")[()]
    if not np.isfinite(half):
        raise ValueError(f"{label} superscale overflows F16")
    if maximum > np.float32(0.0) and half == np.float16(0.0):
        raise ValueError(f"{label} superscale underflows positive F32 to F16 zero")
    return np.float16(0.0) if half == np.float16(0.0) else half


def _scalar_quantize(raw: np.ndarray, superscale: np.float16) -> np.ndarray:
    """Scalar half-up six-bit quantizer independent from the vector path."""
    denominator = np.float32(superscale)
    result = np.empty(np.asarray(raw).size, dtype=np.uint8)
    for index, value in enumerate(np.asarray(raw, dtype=np.float32).reshape(-1)):
        if denominator == np.float32(0.0):
            result[index] = 0
            continue
        ratio = np.divide(value, denominator, dtype=np.float32)
        rounded = np.floor(np.add(ratio, np.float32(0.5), dtype=np.float32))
        result[index] = np.uint8(np.clip(rounded, np.float32(0.0), np.float32(63.0)))
    return result


def _superscales(values: np.ndarray, label: str) -> np.ndarray:
    """Vectorized block superscale conversion with strict F16 boundary checks."""
    if not np.isfinite(values).all() or np.any(values < np.float32(0.0)):
        raise ValueError(f"{label} values are non-finite or negative")
    maximum = np.max(values, axis=-1).astype(np.float32, copy=False)
    raw = np.divide(maximum, np.float32(63.0), dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError(f"{label} superscale is non-finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        half = raw.astype("<f2")
    if not np.isfinite(half).all():
        raise ValueError(f"{label} superscale overflows F16")
    if np.any((maximum > np.float32(0.0)) & (half == np.float16(0.0))):
        raise ValueError(f"{label} superscale underflows positive F32 to F16 zero")
    # Explicitly canonicalize signed zero in the wire representation.
    return np.where(half == np.float16(0.0), np.float16(0.0), half).astype("<f2")


def _quantize_scales(raw: np.ndarray, superscale: np.ndarray | np.float16) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    denominator = np.asarray(superscale, dtype=np.float32)
    zero = denominator == np.float32(0.0)
    safe_denominator = np.where(zero, np.float32(1.0), denominator).astype(np.float32, copy=False)
    ratio = np.divide(raw, safe_denominator, dtype=np.float32)
    rounded = np.floor(np.add(ratio, np.float32(0.5), dtype=np.float32))
    quantized = np.clip(rounded, np.float32(0.0), np.float32(63.0)).astype(np.uint8)
    return np.where(zero, np.uint8(0), quantized).astype(np.uint8)


def _pack_scale_min_fields(scales: np.ndarray, minima: np.ndarray) -> np.ndarray:
    """Pack the canonical Q4_K interleaved six-bit scale/minimum fields."""
    scales = np.asarray(scales, dtype=np.uint8).reshape(-1, GROUPS_PER_BLOCK)
    minima = np.asarray(minima, dtype=np.uint8).reshape(-1, GROUPS_PER_BLOCK)
    wire = np.zeros((scales.shape[0], 12), dtype=np.uint8)
    wire[:, :4] = scales[:, :4]
    wire[:, 4:8] = minima[:, :4]
    for group in range(4, GROUPS_PER_BLOCK):
        wire[:, group + 4] = (scales[:, group] & 0x0F) | ((minima[:, group] & 0x0F) << 4)
        wire[:, group - 4] |= (scales[:, group] >> 4) << 6
        wire[:, group] |= (minima[:, group] >> 4) << 6
    return wire


def _pack_code_nibbles(codes: np.ndarray) -> np.ndarray:
    grouped = np.asarray(codes, dtype=np.uint8).reshape(-1, GROUPS_PER_BLOCK, GROUP_VALUES)
    packed = np.empty((grouped.shape[0], 128), dtype=np.uint8)
    for pair in range(4):
        begin = pair * GROUP_VALUES
        packed[:, begin : begin + GROUP_VALUES] = grouped[:, pair * 2] | (
            grouped[:, pair * 2 + 1] << 4
        )
    return packed


def pack_q4k_groups(values: np.ndarray, ranges: int) -> bytes:
    """Serialize an F32 array shaped [blocks,256] into canonical Q4_K bytes."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != BLOCK_VALUES:
        raise ValueError("Q4_K source batch must have shape [blocks,256]")
    if not np.isfinite(values).all():
        raise ValueError("Q4_K source contains a non-finite value")
    if ranges not in (1, GROUPS_PER_BLOCK):
        raise ValueError("ranges must be 1 or 8")

    lower, upper = _bounds_for_groups(values, ranges)
    span = np.subtract(upper, lower, dtype=np.float32)
    raw_scales = np.divide(span, np.float32(15.0), dtype=np.float32)
    offsets = np.negative(lower, dtype=np.float32)
    if not np.isfinite(raw_scales).all() or not np.isfinite(offsets).all():
        raise ValueError("Q4_K range fit produced a non-finite scale or offset")

    d = _superscales(raw_scales, "Q4_K d")
    dmin = _superscales(offsets, "Q4_K dmin")

    scales = _quantize_scales(raw_scales, d[:, None])
    minima = _quantize_scales(offsets, dmin[:, None])
    decoded_scales = np.multiply(d.astype(np.float32)[:, None], scales.astype(np.float32), dtype=np.float32)
    decoded_minima = np.multiply(dmin.astype(np.float32)[:, None], minima.astype(np.float32), dtype=np.float32)

    grouped_values = values.reshape(values.shape[0], GROUPS_PER_BLOCK, GROUP_VALUES)
    numerator = np.add(grouped_values, decoded_minima[:, :, None], dtype=np.float32)
    denominator = decoded_scales[:, :, None]
    normalized = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denominator, out=normalized, where=denominator != np.float32(0.0))
    rounded = np.floor(np.add(normalized, np.float32(0.5), dtype=np.float32))
    codes = np.clip(rounded, np.float32(0.0), np.float32(15.0)).astype(np.uint8)
    codes = np.where(denominator == np.float32(0.0), np.uint8(0), codes).astype(np.uint8)

    wire = np.empty((values.shape[0], Q4_K_BLOCK_BYTES), dtype=np.uint8)
    half_pair = np.stack((d, dmin), axis=1).astype("<f2", copy=False)
    wire[:, :4] = half_pair.view(np.uint8).reshape(values.shape[0], 4)
    wire[:, 4:16] = _pack_scale_min_fields(scales, minima)
    wire[:, 16:] = _pack_code_nibbles(codes)
    return wire.tobytes()


def pack_q4k_record(values: np.ndarray, ranges: int) -> bytes:
    """Scalar block control for the production vectorized record writer."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (BLOCK_VALUES,):
        raise ValueError("Q4_K scalar source must contain exactly 256 values")
    if not np.isfinite(values).all():
        raise ValueError("Q4_K source contains a non-finite value")
    if ranges not in (1, GROUPS_PER_BLOCK):
        raise ValueError("ranges must be 1 or 8")

    group_width = BLOCK_VALUES if ranges == 1 else GROUP_VALUES
    lowers: list[np.float32] = []
    uppers: list[np.float32] = []
    for first in range(0, BLOCK_VALUES, group_width):
        part = values[first : first + group_width]
        low = np.float32(part[0])
        high = np.float32(part[0])
        for item in part[1:]:
            if item < low:
                low = np.float32(item)
            if item > high:
                high = np.float32(item)
        if low > np.float32(0.0):
            low = np.float32(0.0)
        if high < np.float32(0.0):
            high = np.float32(0.0)
        lowers.append(low)
        uppers.append(high)
    if ranges == 1:
        lowers *= GROUPS_PER_BLOCK
        uppers *= GROUPS_PER_BLOCK

    raw_scales: list[np.float32] = []
    offsets: list[np.float32] = []
    for low, high in zip(lowers, uppers, strict=True):
        span = np.subtract(high, low, dtype=np.float32)
        raw_scales.append(np.divide(span, np.float32(15.0), dtype=np.float32))
        offsets.append(np.negative(low, dtype=np.float32))
    raw_scale_array = np.asarray(raw_scales, dtype=np.float32)
    offset_array = np.asarray(offsets, dtype=np.float32)
    d = _scalar_superscale(raw_scale_array, "Q4_K d")
    dmin = _scalar_superscale(offset_array, "Q4_K dmin")
    scales = _scalar_quantize(raw_scale_array, d).reshape(GROUPS_PER_BLOCK)
    minima = _scalar_quantize(offset_array, dmin).reshape(GROUPS_PER_BLOCK)

    group_codes = np.zeros((GROUPS_PER_BLOCK, GROUP_VALUES), dtype=np.uint8)
    for group in range(GROUPS_PER_BLOCK):
        denominator = np.multiply(np.float32(d), np.float32(scales[group]), dtype=np.float32)
        minimum = np.multiply(np.float32(dmin), np.float32(minima[group]), dtype=np.float32)
        if denominator == np.float32(0.0):
            continue
        for local in range(GROUP_VALUES):
            value = np.add(values[group * GROUP_VALUES + local], minimum, dtype=np.float32)
            normalized = np.divide(value, denominator, dtype=np.float32)
            rounded = np.floor(np.add(normalized, np.float32(0.5), dtype=np.float32))
            group_codes[group, local] = np.uint8(np.clip(rounded, 0.0, 15.0))

    scale_wire = np.zeros(12, dtype=np.uint8)
    for group in range(4):
        scale_wire[group] = scales[group]
        scale_wire[group + 4] = minima[group]
    for group in range(4, GROUPS_PER_BLOCK):
        scale_wire[group + 4] = (scales[group] & 0x0F) | ((minima[group] & 0x0F) << 4)
        scale_wire[group - 4] |= (scales[group] >> 4) << 6
        scale_wire[group] |= (minima[group] >> 4) << 6

    code_wire = bytearray(128)
    for pair in range(4):
        first_group = pair * 2
        start = pair * GROUP_VALUES
        for local in range(GROUP_VALUES):
            code_wire[start + local] = group_codes[first_group, local] | (
                group_codes[first_group + 1, local] << 4
            )
    return (
        struct.pack("<ee", float(d), float(dmin))
        + scale_wire.tobytes()
        + bytes(code_wire)
    )


def _decode_q4k_fields(record: bytes | bytearray | memoryview) -> tuple[
    np.float32, np.float32, np.ndarray, np.ndarray, np.ndarray
]:
    if len(record) != Q4_K_BLOCK_BYTES:
        raise ValueError("Q4_K record must be exactly 144 bytes")
    d_half, dmin_half = struct.unpack_from("<ee", record, 0)
    if not math.isfinite(d_half) or not math.isfinite(dmin_half) or d_half < 0.0 or dmin_half < 0.0:
        raise ValueError("Q4_K record has a non-finite or negative F16 superscale")
    d = np.float32(d_half)
    dmin = np.float32(dmin_half)
    scale_bytes = np.frombuffer(record, dtype=np.uint8, count=12, offset=4)
    scales = np.zeros(GROUPS_PER_BLOCK, dtype=np.uint8)
    minima = np.zeros(GROUPS_PER_BLOCK, dtype=np.uint8)
    scales[:4] = scale_bytes[:4] & 0x3F
    minima[:4] = scale_bytes[4:8] & 0x3F
    for group in range(4, GROUPS_PER_BLOCK):
        scales[group] = (scale_bytes[group + 4] & 0x0F) | ((scale_bytes[group - 4] >> 6) << 4)
        minima[group] = ((scale_bytes[group + 4] >> 4) & 0x0F) | ((scale_bytes[group] >> 6) << 4)

    packed_codes = np.frombuffer(record, dtype=np.uint8, count=128, offset=16)
    codes = np.empty((GROUPS_PER_BLOCK, GROUP_VALUES), dtype=np.uint8)
    for pair in range(4):
        begin = pair * GROUP_VALUES
        packed = packed_codes[begin : begin + GROUP_VALUES]
        codes[pair * 2] = packed & 0x0F
        codes[pair * 2 + 1] = packed >> 4
    return d, dmin, scales, minima, codes


def decode_q4k_record(record: bytes | bytearray | memoryview) -> np.ndarray:
    """Decode one strict 144-byte Q4_K record without using the serializer."""
    d, dmin, scales, minima, codes = _decode_q4k_fields(record)
    decoded = np.empty((GROUPS_PER_BLOCK, GROUP_VALUES), dtype=np.float32)
    for group in range(GROUPS_PER_BLOCK):
        decoded_scale = np.multiply(d, np.float32(scales[group]), dtype=np.float32)
        decoded_minimum = np.multiply(dmin, np.float32(minima[group]), dtype=np.float32)
        scaled_codes = np.multiply(decoded_scale, codes[group].astype(np.float32), dtype=np.float32)
        decoded[group] = np.subtract(scaled_codes, decoded_minimum, dtype=np.float32)
    result = decoded.reshape(BLOCK_VALUES)
    if not np.isfinite(result).all():
        raise ValueError("Q4_K record decodes to a non-finite value")
    return result


def decode_q4k_records(payload: bytes | bytearray | memoryview) -> np.ndarray:
    """Decode a non-truncated byte sequence of complete Q4_K records."""
    if len(payload) % Q4_K_BLOCK_BYTES:
        raise ValueError("Q4_K payload length must be a multiple of 144 bytes")
    block_count = len(payload) // Q4_K_BLOCK_BYTES
    records = np.frombuffer(payload, dtype=np.uint8).reshape(block_count, Q4_K_BLOCK_BYTES)
    superscales = records[:, :4].copy().view("<f2").reshape(block_count, 2).astype(np.float32)
    d = superscales[:, 0]
    dmin = superscales[:, 1]
    if not np.isfinite(superscales).all() or np.any(superscales < np.float32(0.0)):
        raise ValueError("Q4_K payload has a non-finite or negative F16 superscale")

    scale_bytes = records[:, 4:16]
    scales = np.zeros((block_count, GROUPS_PER_BLOCK), dtype=np.uint8)
    minima = np.zeros((block_count, GROUPS_PER_BLOCK), dtype=np.uint8)
    scales[:, :4] = scale_bytes[:, :4] & 0x3F
    minima[:, :4] = scale_bytes[:, 4:8] & 0x3F
    for group in range(4, GROUPS_PER_BLOCK):
        scales[:, group] = (scale_bytes[:, group + 4] & 0x0F) | ((scale_bytes[:, group - 4] >> 6) << 4)
        minima[:, group] = ((scale_bytes[:, group + 4] >> 4) & 0x0F) | ((scale_bytes[:, group] >> 6) << 4)

    packed_codes = records[:, 16:]
    codes = np.empty((block_count, GROUPS_PER_BLOCK, GROUP_VALUES), dtype=np.uint8)
    for pair in range(4):
        begin = pair * GROUP_VALUES
        packed = packed_codes[:, begin : begin + GROUP_VALUES]
        codes[:, pair * 2] = packed & 0x0F
        codes[:, pair * 2 + 1] = packed >> 4
    decoded_scales = np.multiply(d[:, None], scales.astype(np.float32), dtype=np.float32)
    decoded_minima = np.multiply(dmin[:, None], minima.astype(np.float32), dtype=np.float32)
    scaled_codes = np.multiply(decoded_scales[:, :, None], codes.astype(np.float32), dtype=np.float32)
    decoded = np.subtract(scaled_codes, decoded_minima[:, :, None], dtype=np.float32)
    decoded = decoded.reshape(block_count, BLOCK_VALUES)
    if not np.isfinite(decoded).all():
        raise ValueError("Q4_K payload decodes to a non-finite value")
    return decoded


def decode_q4k_file(
    payload_path: Path,
    decoded_path: Path,
    block_count: int = Q4_K_BLOCKS,
    blocks_per_chunk: int = 8192,
) -> None:
    expected_bytes = block_count * Q4_K_BLOCK_BYTES
    if payload_path.stat().st_size != expected_bytes:
        raise ValueError(f"Q4_K payload must be exactly {expected_bytes} bytes")
    decoded = np.memmap(decoded_path, dtype="<f4", mode="w+", shape=(block_count, BLOCK_VALUES))
    with payload_path.open("rb") as stream:
        for start in range(0, block_count, blocks_per_chunk):
            count = min(blocks_per_chunk, block_count - start)
            raw = stream.read(count * Q4_K_BLOCK_BYTES)
            if len(raw) != count * Q4_K_BLOCK_BYTES:
                raise ValueError("Q4_K payload ended before the declared record count")
            decoded[start : start + count] = decode_q4k_records(raw)
        if stream.read(1):
            raise ValueError("Q4_K payload has bytes after the declared record count")
    decoded.flush()
    del decoded


def compare_decoded_files(
    candidate_path: Path,
    reference_path: Path,
    element_count: int,
    chunk_elements: int = 1 << 20,
) -> dict[str, Any]:
    """Summarize an independent decoded tensor against a pinned CPU artifact."""
    expected_bytes = element_count * 4
    if candidate_path.stat().st_size != expected_bytes or reference_path.stat().st_size != expected_bytes:
        raise ValueError("decoded Q4_K comparison files have the wrong size")
    candidate = np.memmap(candidate_path, dtype="<f4", mode="r", shape=(element_count,))
    reference = np.memmap(reference_path, dtype="<f4", mode="r", shape=(element_count,))
    squared_error = 0.0
    max_abs = 0.0
    tolerance_violations = 0
    for first in range(0, element_count, chunk_elements):
        last = min(element_count, first + chunk_elements)
        actual = np.asarray(candidate[first:last])
        expected = np.asarray(reference[first:last])
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError("non-finite value in pinned Q4_K decode comparison")
        error = np.subtract(actual, expected, dtype=np.float32)
        absolute = np.abs(error)
        max_abs = max(max_abs, float(np.max(absolute)))
        squared_error += float(np.sum(np.multiply(error, error, dtype=np.float32), dtype=np.float64))
        tolerance_violations += int(np.count_nonzero(absolute > (2e-5 + 2e-5 * np.abs(expected))))
    del candidate, reference
    return {
        "reference": "pinned upstream Q4_K decoded_weight",
        "compared_values": element_count,
        "max_abs": max_abs,
        "rmse": math.sqrt(squared_error / element_count),
        "values_exceeding_2e-5_atol_rtol": tolerance_violations,
    }


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return affine_reference.bf16_bits_to_f32(bits)


def _write_minmax_payload(source_bf16: np.memmap, payload_path: Path, ranges: int) -> None:
    if source_bf16.size != OUTPUT_DIM * INPUT_DIM:
        raise ValueError("pinned BF16 source has an unexpected element count")
    with payload_path.open("wb") as stream:
        groups_per_chunk = 8192
        for first in range(0, Q4_K_BLOCKS, groups_per_chunk):
            last = min(Q4_K_BLOCKS, first + groups_per_chunk)
            start = first * BLOCK_VALUES
            end = last * BLOCK_VALUES
            source = _bf16_to_f32(source_bf16[start:end]).reshape(-1, BLOCK_VALUES)
            stream.write(pack_q4k_groups(source, ranges))
    if payload_path.stat().st_size != Q4_K_BYTES:
        raise RuntimeError(f"unexpected Q4_K payload size: {payload_path.stat().st_size}")


def _write_descriptor(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4I", 0x31425157, 0, OUTPUT_DIM, INPUT_DIM))


def _write_baseline(
    source_paths: dict[str, Path],
    stage: Path,
    rows: int,
    input_path: Path,
    bias_path: Path,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    baseline_dir = stage / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    decoded_path = baseline_dir / "decoded_weight.f32"
    projection_path = baseline_dir / "expected_projection.f32"
    affine_reference.copy_or_link(source_paths["bf16_decoded"], decoded_path)
    versions: dict[str, str] | None = None
    if rows == BENCH_ROWS:
        affine_reference.copy_or_link(source_paths["bf16_projection"], projection_path)
        projection_reference = "pinned same-geometry MPS fixture"
    else:
        versions = affine_reference.write_candidate_projection(
            decoded_path, input_path, bias_path, projection_path, rows
        )
        projection_reference = "fresh same-geometry MPS F.linear"
    return (
        {
            "decoded_weight": file_record(decoded_path, "F32", [OUTPUT_DIM, INPUT_DIM], stage),
            "expected_projection": file_record(projection_path, "F32", [rows, OUTPUT_DIM], stage),
            "projection_reference": projection_reference,
            "control": f"original BF16 source tensor converted to F32; {projection_reference}",
        },
        versions,
    )


def _write_variant(
    name: str,
    ranges: int,
    source_payload: Path,
    source_q4_decoded: Path,
    source_bf16: np.memmap,
    source_input: np.memmap,
    input_path: Path,
    bias_path: Path,
    baseline_projection: Path,
    stage: Path,
    rows: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    variant_dir = stage / "variants" / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    payload_path = variant_dir / "payload.bin"
    descriptor_path = variant_dir / "descriptor.bin"
    decoded_path = variant_dir / "decoded_weight.f32"
    projection_path = variant_dir / "expected_projection.f32"

    if name == "q4_k":
        affine_reference.copy_or_link(source_payload, payload_path)
    else:
        _write_minmax_payload(source_bf16, payload_path, ranges)
    _write_descriptor(descriptor_path)
    decode_q4k_file(payload_path, decoded_path)

    torch_versions = affine_reference.write_candidate_projection(
        decoded_path, input_path, bias_path, projection_path, rows
    )
    input_rows = np.asarray(source_input[:SOURCE_ROWS], dtype=np.float32)
    cpu_quality = affine_reference.distortion_metrics(source_bf16, decoded_path, input_rows)
    if name == "q4_k":
        decode_control = compare_decoded_files(
            decoded_path, source_q4_decoded, OUTPUT_DIM * INPUT_DIM
        )
        cpu_quality["pinned_q4_decode_control"] = decode_control
        if decode_control["values_exceeding_2e-5_atol_rtol"]:
            raise RuntimeError(
                "independent CPU Q4_K decoder disagrees with pinned decoded_weight "
                f"at {decode_control['values_exceeding_2e-5_atol_rtol']} values"
            )
    projection_quality = affine_reference.projection_quality(projection_path, baseline_projection, rows)
    block_count = Q4_K_BLOCKS
    variant = {
        "ranges": ranges,
        "payload": file_record(payload_path, "Q4_K", [OUTPUT_DIM, INPUT_DIM // BLOCK_VALUES, Q4_K_BLOCK_BYTES], stage),
        "descriptor": file_record(descriptor_path, "Q4KDescriptor", [FORMAT_DESCRIPTOR_BYTES], stage),
        "decoded_weight": file_record(decoded_path, "F32", [OUTPUT_DIM, INPUT_DIM], stage),
        "expected_projection": file_record(projection_path, "F32", [rows, OUTPUT_DIM], stage),
        "projection_reference": "fresh same-geometry MPS F.linear with fallback disabled",
        "storage": {
            "payload_bytes": Q4_K_BYTES,
            "descriptor_bytes": FORMAT_DESCRIPTOR_BYTES,
            "total_candidate_bytes": Q4_K_BYTES + FORMAT_DESCRIPTOR_BYTES,
            "record_bytes": Q4_K_BLOCK_BYTES,
            "records": block_count,
            "descriptor_format_id": 0,
        },
        "cpu_quality": {
            "decode_device": "CPU",
            "weight_distortion": cpu_quality,
        },
        "quality_vs_original_bf16": projection_quality,
    }
    return variant, torch_versions


def create_reference_fixture(fixture_dir: Path, output_dir: Path, rows: int) -> Path:
    if rows not in (SOURCE_ROWS, BENCH_ROWS):
        raise ValueError("rows must be 4 or 4480")
    fixture_dir = fixture_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"immutable output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    source_manifest, source_paths = affine_reference.validate_fixture(fixture_dir)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        source_records, copied_sources = affine_reference.source_records(source_paths, stage)
        source_input = np.memmap(source_paths["input"], dtype="<f4", mode="r", shape=(SOURCE_ROWS, INPUT_DIM))
        source_bf16 = np.memmap(source_paths["original_weight"], dtype="<u2", mode="r", shape=(OUTPUT_DIM * INPUT_DIM,))
        baseline, torch_versions = _write_baseline(
            source_paths,
            stage,
            rows,
            copied_sources["input"],
            copied_sources["bias"],
        )
        baseline_projection_path = stage / baseline["expected_projection"]["path"]

        variants: dict[str, dict[str, Any]] = {}
        for name, ranges in RANGE_ARMS.items():
            variant, versions = _write_variant(
                name,
                ranges,
                source_paths["q4_payload"],
                source_paths["q4_decoded"],
                source_bf16,
                source_input,
                copied_sources["input"],
                copied_sources["bias"],
                baseline_projection_path,
                stage,
                rows,
            )
            variants[name] = variant
            if torch_versions is None:
                torch_versions = versions

        del source_bf16, source_input
        support_path = Path(affine_reference.__file__).resolve()
        own_path = Path(__file__).resolve()
        manifest = {
            "schema": "wan_q4k_range_reference_v1",
            "created_by": own_path.name,
            "geometry": {
                "rows": rows,
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
                "scope": "native decode/project parity against each payload's independently CPU-decoded MPS F.linear reference",
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
                "all_arms_have_identical_descriptor_contents": True,
                "descriptor_words_u32le": ["magic_WQB1", "format_id", "output_dim", "input_dim"],
                "descriptor_format_ids": FORMAT_IDS,
                "q4_k_payload_bytes": Q4_K_BYTES,
            },
            "source_fixture": {
                "schema": source_manifest["schema"],
                "manifest_sha256": affine_reference.FIXTURE_MANIFEST_SHA256,
                "real_video_derived": source_manifest.get("fixture_policy", {}).get("input_is_real_video_derived"),
                "checkpoint_weight_sha256": affine_reference.WEIGHT_BF16_SHA256,
                "input_sha256": affine_reference.INPUT_SHA256,
                "bias_f32_sha256": affine_reference.BIAS_F32_SHA256,
                "q4_k_sha256": affine_reference.Q4_K_SHA256,
                "q4_decoded_sha256": source_manifest["variants"]["q4_k"]["decoded_weight"]["sha256"],
            },
            "input": source_records["input"],
            "bias": source_records["bias"],
            "original_weight": source_records["original_weight"],
            "baseline": baseline,
            "variants": variants,
            "producers": {
                "pinned_q4_k": source_manifest.get("producers", {}).get("q4_k"),
                "cpu_reference": {
                    "script": own_path.name,
                    "script_sha256": sha256_file(own_path),
                    "serializer": "zero-inclusive minmax endpoints; F32 arithmetic without fused multiply-add; F16 RNE superscales; canonical Q4_K six-bit fields and nibble pairs",
                    "fixture_validation_and_metrics": {
                        "script": support_path.name,
                        "script_sha256": sha256_file(support_path),
                    },
                },
            },
            "fixture_policy": {
                "immutable_output": True,
                "full_wan_model_loaded": False,
                "real_video_quality_claim": False,
                "source_rows_are_periodically_tiled_when_rows_4480": True,
            },
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if os.path.lexists(output_dir):
            raise FileExistsError(f"immutable output appeared while preparing: {output_dir}")
        os.replace(stage, output_dir)
        return output_dir
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, choices=(SOURCE_ROWS, BENCH_ROWS), default=BENCH_ROWS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = create_reference_fixture(args.fixture_dir, args.output_dir, args.rows)
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
