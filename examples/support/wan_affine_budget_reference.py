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
JOINT_UPGRADE_COUNT = PATCH_COUNT - AFFINE_GROUPS
JOINT_RANGE_CANDIDATES = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))

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


def _pack_affine_groups_with_bounds(
    values: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[bytes, np.ndarray]:
    """Pack affine groups using independently selected F32 endpoints."""
    values = np.asarray(values, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32).reshape(-1)
    upper = np.asarray(upper, dtype=np.float32).reshape(-1)
    if values.ndim != 2 or values.shape[1] != BLOCK_VALUES or values.shape[0] != lower.size or lower.size != upper.size:
        raise ValueError("bounded affine batch must have matching [groups,256] values and endpoints")
    if not np.isfinite(values).all() or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("bounded affine source/endpoints contain a non-finite value")
    if np.any(upper < lower):
        raise ValueError("bounded affine upper endpoint is below lower endpoint")
    span = np.subtract(upper, lower, dtype=np.float32)
    scale = np.divide(span, np.float32(15.0), dtype=np.float32)
    if not np.isfinite(scale).all():
        raise ValueError("bounded affine scale is non-finite")
    scale[span == 0] = np.float32(0.0)
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


def encode_affine_base_with_ranges(
    source_bf16: np.memmap,
    base_path: Path,
    decoded_path: Path,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    """Write an affine base with one preselected range per flattened group."""
    element_count = OUTPUT_DIM * INPUT_DIM
    if source_bf16.size != element_count or element_count % BLOCK_VALUES:
        raise ValueError("affine source must contain complete 256-value groups")
    lower = np.asarray(lower, dtype=np.float32).reshape(-1)
    upper = np.asarray(upper, dtype=np.float32).reshape(-1)
    total_groups = element_count // BLOCK_VALUES
    if lower.size != total_groups or upper.size != total_groups:
        raise ValueError("selected affine ranges have the wrong group count")
    decoded = np.memmap(decoded_path, dtype="<f4", mode="w+", shape=(element_count,))
    with base_path.open("wb") as stream:
        groups_per_chunk = 8192
        for group_start in range(0, total_groups, groups_per_chunk):
            group_end = min(total_groups, group_start + groups_per_chunk)
            start = group_start * BLOCK_VALUES
            end = group_end * BLOCK_VALUES
            source = bf16_bits_to_f32(source_bf16[start:end]).reshape(-1, BLOCK_VALUES)
            records, decoded_chunk = _pack_affine_groups_with_bounds(
                source, lower[group_start:group_end], upper[group_start:group_end]
            )
            stream.write(records)
            decoded[start:end] = decoded_chunk.reshape(-1)
    decoded.flush()
    del decoded
    if base_path.stat().st_size != AFFINE_BASE_BYTES:
        raise RuntimeError(f"unexpected fitted affine base bytes: {base_path.stat().st_size}")


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


def _tree_sum_f32(values: np.ndarray) -> np.ndarray:
    """Sum 256 lanes with the native reducer's fixed 128..1 tree order."""
    reduced = np.array(values, dtype=np.float32, copy=True)
    if reduced.ndim != 2 or reduced.shape[1] != BLOCK_VALUES:
        raise ValueError("joint fit reduction expects [groups,256] values")
    stride = BLOCK_VALUES // 2
    while stride:
        np.add(reduced[:, :stride], reduced[:, stride : stride * 2], out=reduced[:, :stride])
        stride //= 2
    return reduced[:, 0].copy()


def _masked_tree_loss(scores: np.ndarray, positions: np.ndarray) -> np.ndarray:
    masked = np.array(scores, dtype=np.float32, copy=True)
    rows = np.arange(masked.shape[0], dtype=np.int64)
    if positions.ndim == 1:
        masked[rows, positions.astype(np.int64)] = np.float32(0.0)
    else:
        for column in range(positions.shape[1]):
            masked[rows, positions[:, column].astype(np.int64)] = np.float32(0.0)
    return _tree_sum_f32(masked)


def _joint_fit_groups_multi(
    values: np.ndarray,
    metric_weights: dict[str, np.ndarray | None],
    group_start: int = 0,
) -> dict[str, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != BLOCK_VALUES:
        raise ValueError("joint fit values must have shape [groups,256]")
    if not np.isfinite(values).all():
        raise ValueError("joint fit source contains a non-finite value")
    group_count = values.shape[0]
    if group_start < 0:
        raise ValueError("joint fit group offset cannot be negative")

    metric_matrices: dict[str, np.ndarray | None] = {}
    lane_offsets = np.arange(group_count * BLOCK_VALUES, dtype=np.int64)
    for name, weights in metric_weights.items():
        if weights is None:
            metric_matrices[name] = None
            continue
        weights = np.asarray(weights, dtype=np.float32)
        if weights.ndim == 1 and weights.size:
            indices = (group_start * BLOCK_VALUES + lane_offsets) % weights.size
            metric_matrices[name] = weights[indices].reshape(group_count, BLOCK_VALUES)
        elif weights.shape == values.shape:
            metric_matrices[name] = weights
        else:
            raise ValueError("joint fit metric weights must be a nonempty vector or [groups,256]")
        if not np.isfinite(metric_matrices[name]).all():
            raise ValueError("joint fit metric weights contain a non-finite value")

    order_indices = np.argsort(values, axis=1, kind="stable")
    order = np.take_along_axis(values, order_indices, axis=1)
    lows = np.stack(
        (order[:, 0], order[:, 1], order[:, 0], order[:, 2], order[:, 1], order[:, 0]), axis=1
    ).astype(np.float32, copy=False)
    highs = np.stack(
        (order[:, 255], order[:, 255], order[:, 254], order[:, 255], order[:, 254], order[:, 253]), axis=1
    ).astype(np.float32, copy=False)
    outputs: dict[str, dict[str, Any]] = {}
    for name in metric_weights:
        outputs[name] = {
            "q1": {
                "candidate_losses": np.empty((group_count, len(JOINT_RANGE_CANDIDATES)), dtype=np.float32),
                "candidate_positions": np.empty((group_count, len(JOINT_RANGE_CANDIDATES)), dtype=np.uint16),
            },
            "q2": {
                "candidate_losses": np.empty((group_count, len(JOINT_RANGE_CANDIDATES)), dtype=np.float32),
                "candidate_positions": np.empty((group_count, len(JOINT_RANGE_CANDIDATES), 2), dtype=np.uint16),
            },
        }

    for candidate_id in range(len(JOINT_RANGE_CANDIDATES)):
        lower = lows[:, candidate_id]
        upper = highs[:, candidate_id]
        span = np.subtract(upper, lower, dtype=np.float32)
        scale = np.divide(span, np.float32(15.0), dtype=np.float32)
        if not np.isfinite(scale).all():
            raise ValueError("joint fit scale is non-finite")
        scale[span == 0] = np.float32(0.0)
        delta = np.subtract(values, lower[:, None], dtype=np.float32)
        normalized = np.zeros_like(delta)
        np.divide(delta, scale[:, None], out=normalized, where=scale[:, None] != 0)
        rounded = np.floor(np.add(normalized, np.float32(0.5), dtype=np.float32))
        codes = np.clip(rounded, 0, 15).astype(np.float32)
        scaled = np.multiply(scale[:, None], codes, dtype=np.float32)
        decoded = np.add(lower[:, None], scaled, dtype=np.float32)
        residual = np.subtract(values, decoded, dtype=np.float32)
        base_scores = np.multiply(residual, residual, dtype=np.float32)
        if not np.isfinite(base_scores).all():
            raise ValueError("joint fit produced a non-finite residual score")

        for name, weights in metric_matrices.items():
            scores = base_scores if weights is None else np.multiply(base_scores, weights, dtype=np.float32)
            if not np.isfinite(scores).all():
                raise ValueError("joint fit produced a non-finite weighted residual score")
            top1 = np.argmax(scores, axis=1).astype(np.uint16)
            second_scores = np.array(scores, dtype=np.float32, copy=True)
            second_scores[np.arange(group_count), top1.astype(np.int64)] = np.float32(-np.inf)
            top2 = np.argmax(second_scores, axis=1).astype(np.uint16)
            outputs[name]["q1"]["candidate_losses"][:, candidate_id] = _masked_tree_loss(scores, top1)
            outputs[name]["q1"]["candidate_positions"][:, candidate_id] = top1
            outputs[name]["q2"]["candidate_losses"][:, candidate_id] = _masked_tree_loss(
                scores, np.stack((top1, top2), axis=1)
            )
            outputs[name]["q2"]["candidate_positions"][:, candidate_id, 0] = top1
            outputs[name]["q2"]["candidate_positions"][:, candidate_id, 1] = top2

    row_ids = np.arange(group_count, dtype=np.int64)
    for name in outputs:
        for quantity in ("q1", "q2"):
            fit = outputs[name][quantity]
            losses = fit.pop("candidate_losses")
            positions = fit.pop("candidate_positions")
            best_id = np.argmin(losses, axis=1).astype(np.uint8)
            fit["candidate_id"] = best_id
            fit["candidate_losses"] = losses
            fit["loss"] = losses[row_ids, best_id.astype(np.int64)]
            if not np.isfinite(fit["loss"]).all():
                raise ValueError("joint fit produced a non-finite reduced loss")
            if quantity == "q1":
                fit["patch_positions"] = positions[row_ids, best_id.astype(np.int64)][:, None]
            else:
                fit["patch_positions"] = positions[row_ids, best_id.astype(np.int64), :]
            fit["lower"] = lows[row_ids, best_id.astype(np.int64)].copy()
            fit["upper"] = highs[row_ids, best_id.astype(np.int64)].copy()
    return outputs


def joint_fit_groups(
    values: np.ndarray, metric_weights: np.ndarray | None = None
) -> dict[str, dict[str, Any]]:
    """Fit one or more groups with all six range candidates and q1/q2 patches."""
    return _joint_fit_groups_multi(values, {"metric": metric_weights})["metric"]


def joint_fit_source_groups(
    source_bf16: np.memmap,
    metric_weights: dict[str, np.ndarray | None],
    chunk_groups: int = 512,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Chunk the pinned [N,K] BF16 source while retaining only compact fit records."""
    if source_bf16.size != OUTPUT_DIM * INPUT_DIM or source_bf16.size % BLOCK_VALUES:
        raise ValueError("joint fit source must match the pinned complete-group geometry")
    if chunk_groups <= 0:
        raise ValueError("joint fit chunk size must be positive")
    group_count = source_bf16.size // BLOCK_VALUES
    chunks: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        name: {quantity: {} for quantity in ("q1", "q2")} for name in metric_weights
    }
    for first_group in range(0, group_count, chunk_groups):
        last_group = min(group_count, first_group + chunk_groups)
        first_value = first_group * BLOCK_VALUES
        last_value = last_group * BLOCK_VALUES
        values = bf16_bits_to_f32(source_bf16[first_value:last_value]).reshape(-1, BLOCK_VALUES)
        partial = _joint_fit_groups_multi(values, metric_weights, group_start=first_group)
        for name in metric_weights:
            for quantity in ("q1", "q2"):
                for field, value in partial[name][quantity].items():
                    chunks[name][quantity].setdefault(field, []).append(np.asarray(value))
        del values, partial
    return {
        name: {
            quantity: {field: np.concatenate(parts, axis=0) for field, parts in fields.items()}
            for quantity, fields in quantities.items()
        }
        for name, quantities in chunks.items()
    }


def select_joint_quota_patches(
    q1_positions: np.ndarray,
    q2_positions: np.ndarray,
    gains: np.ndarray,
    patch_count: int = PATCH_COUNT,
) -> dict[str, Any]:
    """Give every group one patch, then spend remaining budget on best upgrades."""
    q1_positions = np.asarray(q1_positions).reshape(-1)
    q2_positions = np.asarray(q2_positions)
    gains = np.asarray(gains, dtype=np.float32).reshape(-1)
    group_count = q1_positions.size
    if q2_positions.shape != (group_count, 2) or gains.size != group_count:
        raise ValueError("joint quota arrays must contain matching q1, q2, and gain groups")
    if not np.issubdtype(q1_positions.dtype, np.integer) or not np.issubdtype(
        q2_positions.dtype, np.integer
    ):
        raise ValueError("joint quota positions must be integers")
    if np.any(q1_positions < 0) or np.any(q1_positions >= BLOCK_VALUES) or np.any(
        q2_positions < 0
    ) or np.any(q2_positions >= BLOCK_VALUES):
        raise ValueError("joint quota positions must be within each 256-value group")
    if np.any(q2_positions[:, 0] == q2_positions[:, 1]):
        raise ValueError("joint quota q2 positions must be distinct within each group")
    if not np.isfinite(gains).all():
        raise ValueError("joint quota gains contain a non-finite value")
    if np.any(gains < 0):
        raise ValueError("joint quota gains must be nonnegative")
    q1_positions = q1_positions.astype(np.uint16, copy=False)
    q2_positions = q2_positions.astype(np.uint16, copy=False)
    gains[gains == 0] = np.float32(0.0)
    upgrade_count = int(patch_count) - group_count
    if group_count == 0 or upgrade_count < 0 or upgrade_count > group_count:
        raise ValueError("joint quota must allocate one or two patches to every group")
    upgrade_groups = top_indices(gains, upgrade_count)
    upgraded = np.zeros(group_count, dtype=np.bool_)
    upgraded[upgrade_groups] = True
    regular_groups = np.flatnonzero(~upgraded)
    selected = np.empty(group_count + upgrade_count, dtype=np.int64)
    selected[: regular_groups.size] = (
        regular_groups.astype(np.int64) * BLOCK_VALUES + q1_positions[regular_groups].astype(np.int64)
    )
    if upgrade_count:
        selected[regular_groups.size :] = (
            upgrade_groups[:, None].astype(np.int64) * BLOCK_VALUES
            + q2_positions[upgrade_groups].astype(np.int64)
        ).reshape(-1)
    selected.sort()
    if selected.size != patch_count or (selected.size > 1 and np.any(selected[1:] <= selected[:-1])):
        raise RuntimeError("joint quota did not produce a unique exact patch count")
    return {
        "selected_flat_indices": selected,
        "upgrade_groups": upgrade_groups,
        "upgrade_count": upgrade_count,
        "upgrade_gain_threshold": float(np.partition(gains, gains.size - upgrade_count)[gains.size - upgrade_count])
        if upgrade_count
        else None,
    }


def joint_candidate_histogram(candidate_ids: np.ndarray) -> dict[str, int]:
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    if np.any(candidate_ids < 0) or np.any(candidate_ids >= len(JOINT_RANGE_CANDIDATES)):
        raise ValueError("joint candidate ID is outside the fixed candidate set")
    counts = np.bincount(candidate_ids, minlength=len(JOINT_RANGE_CANDIDATES))
    return {str(index): int(count) for index, count in enumerate(counts)}


def sparse_record_patch_histogram(
    selected: np.ndarray, values_per_group: int = SPARSE_RECORD_VALUES
) -> dict[str, int]:
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    if values_per_group <= 0 or (OUTPUT_DIM * INPUT_DIM) % values_per_group:
        raise ValueError("patch histogram group size must divide the pinned projection")
    if selected.size and (np.any(selected < 0) or np.any(selected >= OUTPUT_DIM * INPUT_DIM)):
        raise ValueError("sparse patch selection exceeds the pinned projection")
    if np.unique(selected).size != selected.size:
        raise ValueError("sparse patch selection contains duplicate indices")
    group_count = OUTPUT_DIM * INPUT_DIM // values_per_group
    _, populated_counts = np.unique(selected // values_per_group, return_counts=True)
    populated_group_count = populated_counts.size
    return {
        "values_per_group": int(values_per_group),
        "group_count": int(group_count),
        "zero_patch_groups": int(group_count - populated_group_count),
        "one_patch_groups": int(np.count_nonzero(populated_counts == 1)),
        "two_patch_groups": int(np.count_nonzero(populated_counts == 2)),
        "more_than_two_patch_groups": int(np.count_nonzero(populated_counts > 2)),
        "minimum_patches_per_group": 0 if populated_group_count < group_count else int(populated_counts.min()),
        "maximum_patches_per_group": int(populated_counts.max()) if populated_group_count else 0,
    }


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
            format_key = "sparse" if mode in (0, 1, 4, 5, 6) else name.rsplit("/", 1)[-1]
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

        joint_fit_metadata: dict[str, Any] | None = None
        if getattr(args, "joint", False):
            fold0_weights = calibration_weights(input_rows, (0, 1))
            fold1_weights = calibration_weights(input_rows, (2, 3))
            joint_fits = joint_fit_source_groups(
                source_bf16,
                {
                    "weight": None,
                    "activation_f0": fold0_weights,
                    "activation_f1": fold1_weights,
                },
            )

            def fit_loss_summary(
                q1_fit: dict[str, np.ndarray],
                q2_fit: dict[str, np.ndarray],
                selected_candidate_ids: np.ndarray,
                selected_policy_loss: np.ndarray,
                selected_policy_baseline_loss: np.ndarray,
                loss_scope: str,
            ) -> dict[str, Any]:
                selected_loss_sum = float(np.sum(selected_policy_loss, dtype=np.float64))
                minmax_loss_sum = float(np.sum(selected_policy_baseline_loss, dtype=np.float64))
                return {
                    "chosen_q1_candidate_local_residual_sum_after_one_patch_per_group": float(
                        np.sum(q1_fit["loss"], dtype=np.float64)
                    ),
                    "chosen_q2_candidate_local_residual_sum_after_two_patches_per_group": float(
                        np.sum(q2_fit["loss"], dtype=np.float64)
                    ),
                    "candidate0_minmax_q1_local_residual_sum_after_one_patch_per_group": float(
                        np.sum(q1_fit["candidate_losses"][:, 0], dtype=np.float64)
                    ),
                    "candidate0_minmax_q2_local_residual_sum_after_two_patches_per_group": float(
                        np.sum(q2_fit["candidate_losses"][:, 0], dtype=np.float64)
                    ),
                    "selected_local_policy_residual_sum": selected_loss_sum,
                    "selected_local_policy_candidate0_minmax_residual_sum": minmax_loss_sum,
                    "selected_local_policy_loss_reduction_vs_candidate0_minmax": minmax_loss_sum - selected_loss_sum,
                    "best_q1_range_candidate_histogram": joint_candidate_histogram(q1_fit["candidate_id"]),
                    "best_q2_range_candidate_histogram": joint_candidate_histogram(q2_fit["candidate_id"]),
                    "selected_range_candidate_histogram": joint_candidate_histogram(selected_candidate_ids),
                    "scope": loss_scope,
                }

            def emit_joint_sparse_variant(
                name: str,
                mode: int,
                fold: int,
                lower: np.ndarray,
                upper: np.ndarray,
                selected: np.ndarray,
                selection: dict[str, Any],
                candidate_ids: np.ndarray,
            ) -> None:
                output_dir = stage / "variants" / name
                output_dir.mkdir(parents=True, exist_ok=True)
                base_path = work / f"joint-base-{name}.bin"
                base_decoded_path = work / f"joint-base-{name}.f32"
                payload_path = output_dir / "payload.bin"
                decoded_path = output_dir / "decoded_weight.f32"
                encode_affine_base_with_ranges(
                    source_bf16, base_path, base_decoded_path, lower, upper
                )
                candidate_ids = np.asarray(candidate_ids, dtype=np.uint8).reshape(-1)
                if candidate_ids.size != AFFINE_GROUPS:
                    raise ValueError("joint selected range IDs do not cover every affine group")
                candidate_path = output_dir / "selected-range-candidate.u8"
                candidate_record = write_selection_file(candidate_path, candidate_ids, "<u1", stage)
                selected = np.asarray(selected, dtype=np.int64)
                selected_path = output_dir / "selected-flat-indices.u32"
                selected_record = write_selection_file(selected_path, selected, "<u4", stage)
                components = build_sparse_payload(base_path, payload_path, selected, source_bf16)
                components["affine_groups"] = AFFINE_GROUPS
                components["affine_group_count"] = AFFINE_GROUPS
                components["group_patch_quota_policy"] = selection.get("group_patch_quota_policy")
                decode_sparse_payload(payload_path, decoded_path, source_bf16)
                selection["selected_count"] = int(selected.size)
                selection["selected_indices"] = selected_record
                selection["selected_indices_sha256"] = _array_hash(selected.astype("<u4"))
                selection["selected_range_candidate_ids"] = candidate_record
                selection["selected_range_candidate_ids_sha256"] = _array_hash(candidate_ids)
                selection["selected_range_candidate_histogram"] = joint_candidate_histogram(candidate_ids)
                selection["patches_per_256_value_group_histogram"] = sparse_record_patch_histogram(
                    selected, BLOCK_VALUES
                )
                selection["patches_per_1024_value_record_histogram"] = sparse_record_patch_histogram(selected)
                selection["tie_rule"] = "equal F32 score selects lower group/flat index; range candidate ties select lower candidate ID"
                finish_variant(
                    f"variants/{name}",
                    mode,
                    fold,
                    payload_path,
                    decoded_path,
                    selection,
                    components,
                )
                base_decoded_path.unlink(missing_ok=True)
                base_path.unlink(missing_ok=True)

            weight_fit = joint_fits["weight"]
            q1_weight = weight_fit["q1"]
            q2_weight = weight_fit["q2"]
            quota = select_joint_quota_patches(
                q1_weight["patch_positions"][:, 0],
                q2_weight["patch_positions"],
                np.subtract(q1_weight["loss"], q2_weight["loss"], dtype=np.float32),
            )
            upgraded = np.zeros(AFFINE_GROUPS, dtype=np.bool_)
            upgraded[quota["upgrade_groups"]] = True
            quota_candidate_ids = np.where(
                upgraded, q2_weight["candidate_id"], q1_weight["candidate_id"]
            ).astype(np.uint8)
            quota_lower = np.where(upgraded, q2_weight["lower"], q1_weight["lower"]).astype(np.float32)
            quota_upper = np.where(upgraded, q2_weight["upper"], q1_weight["upper"]).astype(np.float32)
            quota_policy_loss = np.where(upgraded, q2_weight["loss"], q1_weight["loss"])
            quota_policy_baseline = np.where(
                upgraded,
                q2_weight["candidate_losses"][:, 0],
                q1_weight["candidate_losses"][:, 0],
            )
            quota_summary = fit_loss_summary(
                q1_weight,
                q2_weight,
                quota_candidate_ids,
                quota_policy_loss,
                quota_policy_baseline,
                "Sum of per-256-group F32 tree losses for the emitted quota's one/two local exceptions; candidate-0 minmax uses the same per-group exception count.",
            )
            upgrade_path = stage / "variants" / "joint_quota_weight" / "upgrade-groups.u32"
            upgrade_path.parent.mkdir(parents=True, exist_ok=True)
            upgrade_record = write_selection_file(
                upgrade_path, quota["upgrade_groups"], "<u4", stage
            )
            emit_joint_sparse_variant(
                "joint_quota_weight",
                4,
                0,
                quota_lower,
                quota_upper,
                quota["selected_flat_indices"],
                {
                    "selector": "best q1 range for each group, then best q2 range for exactly the highest F32 L1-L2 group gains",
                    "range_fit": "all six fixed endpoint candidates for q1 and q2; group ties use lower candidate ID",
                    "upgrade_count": quota["upgrade_count"],
                    "upgrade_gain_threshold": quota["upgrade_gain_threshold"],
                    "upgrade_groups": upgrade_record,
                    "upgrade_groups_sha256": _array_hash(quota["upgrade_groups"].astype("<u4")),
                    "group_patch_quota_policy": "one patch per 256-value group, with one additional patch for selected upgrade groups",
                    "fit_loss": quota_summary,
                },
                quota_candidate_ids,
            )

            del upgraded, quota_lower, quota_upper, quota_candidate_ids
            del quota_policy_loss, quota_policy_baseline, quota_summary
            del q1_weight, q2_weight, weight_fit

            for name, mode, fold, train_rows, metric_name in (
                ("joint_global_weight", 5, 0, None, "weight"),
                ("joint_global_activation_f0", 6, 0, (0, 1), "activation_f0"),
                ("joint_global_activation_f1", 6, 1, (2, 3), "activation_f1"),
            ):
                fit = joint_fits[metric_name]
                q1_fit = fit["q1"]
                q2_fit = fit["q2"]
                lower = q2_fit["lower"]
                upper = q2_fit["upper"]
                selected_candidate_ids = q2_fit["candidate_id"]
                fit_summary = fit_loss_summary(
                    q1_fit,
                    q2_fit,
                    selected_candidate_ids,
                    q2_fit["loss"],
                    q2_fit["candidate_losses"][:, 0],
                    "Local q2 proxy: two exceptions per 256-value group before final global top-R competition; this is not the realized global payload loss and cannot represent groups receiving zero or more than two final patches.",
                )
                output_dir = stage / "variants" / name
                output_dir.mkdir(parents=True, exist_ok=True)
                base_path = work / f"joint-base-{name}.bin"
                base_decoded_path = work / f"joint-base-{name}.f32"
                encode_affine_base_with_ranges(
                    source_bf16, base_path, base_decoded_path, lower, upper
                )
                decoded_base_joint = np.memmap(
                    base_decoded_path, dtype="<f4", mode="r", shape=(source_bf16.size,)
                )
                weights = None if train_rows is None else calibration_weights(input_rows, train_rows)
                score_path = work / f"score-{name}.f32"
                scores = write_score_file(
                    source_bf16, decoded_base_joint, score_path, source_bf16.size, weights
                )
                selected = top_indices(scores, PATCH_COUNT)
                threshold = float(
                    np.partition(np.asarray(scores), scores.size - PATCH_COUNT)[scores.size - PATCH_COUNT]
                )
                selection = {
                    "selector": "best q2 range per group, then global top-R residual patches without a per-group floor/cap",
                    "range_fit": "all six fixed endpoint candidates for q1 and q2; q2 candidate ties use lower candidate ID",
                    "train_input_rows": list(train_rows) if train_rows else None,
                    "heldout_input_rows": ([2, 3] if fold == 0 else [0, 1]) if train_rows else None,
                    "selected_count": PATCH_COUNT,
                    "score_threshold": threshold,
                    "group_patch_quota_policy": None,
                    "fit_loss": fit_summary,
                }
                emit_joint_sparse_variant(
                    name,
                    mode,
                    fold,
                    lower,
                    upper,
                    selected,
                    selection,
                    selected_candidate_ids,
                )
                del scores, decoded_base_joint, lower, upper, selected_candidate_ids
                score_path.unlink(missing_ok=True)
                base_decoded_path.unlink(missing_ok=True)
                base_path.unlink(missing_ok=True)
            del joint_fits, fold0_weights, fold1_weights
            joint_fit_metadata = {
                "enabled_by_opt_in_flag": "--joint",
                "candidate_order": [list(pair) for pair in JOINT_RANGE_CANDIDATES],
                "q1_and_q2_consider_all_six_candidates": True,
                "endpoint_rule": "stable numeric sort of source BF16 promoted to F32; equal values preserve ascending local lane order",
                "quantization_rule": "F32 scale=(upper-lower)/15; round floor((value-lower)/scale+0.5), clamp 0..15; decode scale*code then lower+scaled with contraction off",
                "correction_rule": "one or two highest F32 squared residual scores; lower local index breaks ties; corrections are absolute BF16 replacements",
                "loss_reduction": "mask selected exceptions then F32 tree reduction at strides 128,64,...,1",
                "quota_upgrade_count": JOINT_UPGRADE_COUNT,
                "quota_policy": "exactly one patch per 256-value group, then one upgrade patch for the highest F32 L1-L2 group gains",
                "global_policy": "best q2 base per 256-value group followed by global top-R residual selection; groups may receive zero or more than two patches",
                "activation_folds": {
                    "f0": {"train_rows": [0, 1], "heldout_rows": [2, 3]},
                    "f1": {"train_rows": [2, 3], "heldout_rows": [0, 1]},
                },
                "claim_scope": "bounded endpoint-candidate heuristic under the exact payload budget; not a global optimum or trajectory-quality claim",
            }
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
        if joint_fit_metadata is not None:
            manifest["joint_fit"] = joint_fit_metadata
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
    parser.add_argument(
        "--joint",
        action="store_true",
        help="also build four joint affine-range and sparse-exception candidates",
    )
    return parser


def main() -> None:
    output = build_fixture(make_parser().parse_args())
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
