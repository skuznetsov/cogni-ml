#!/usr/bin/env python3
"""Prepare MPS oracle fixtures for positional QBit correction packing.

This is an offline oracle for the opt-in Wan positional-weight probe. It reads
the pinned, bounded Wan FFN fixture and never loads a model. QBit decoding,
per-value error masks, corrected-weight reconstruction, distortion reductions,
and the 4480-row F.linear references run on MPS with PyTorch fallback disabled.
CPU work is limited to selecting the cheapest wire representation from MPS
candidate counts and serializing the independently specified byte layout.

The output directory is immutable and includes exact plan, offset, packed-blob,
full decoded-weight, and four-row projection records for all 18 arms.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
import torch.nn.functional as F

import wan_quantized_projection_diagnostics as diagnostics


SOURCE_ROWS = diagnostics.SOURCE_ROWS
ROWS = diagnostics.ROWS
INPUT_DIM = diagnostics.INPUT_DIM
OUTPUT_DIM = diagnostics.OUTPUT_DIM
BLOCK_SIZE = diagnostics.BLOCK_SIZE
BLOCKS_PER_ROW = INPUT_DIM // BLOCK_SIZE
WEIGHT_COUNT = OUTPUT_DIM * INPUT_DIM
TILE_SIZES = (256, 512, 1024)
THRESHOLDS = (np.float32(0.10), np.float32(0.20), np.float32(0.50))
VALUE_MODES = ((0, "bf16_absolute"), (1, "f16_delta"))
P4_PAYLOAD_BYTES = 136
P5_PAYLOAD_BYTES = 168
RAW_VALUE_BYTES = 2
HEADER_BYTES = 4
OFFSET_WORD_BYTES = 4
POSITION_VALUE_BYTES = 2


def require_mps(tensor: torch.Tensor, label: str) -> torch.Tensor:
    if tensor.device.type != "mps":
        raise RuntimeError(f"{label} must be on MPS, found {tensor.device}")
    return tensor


def align4(size: int) -> int:
    return (size + 3) & ~3


def tile_record_size(tile_size: int, precision: int, correction_count: int) -> int:
    if tile_size not in TILE_SIZES:
        raise ValueError(f"unsupported tile size: {tile_size}")
    if correction_count < 0 or correction_count > tile_size:
        raise ValueError(f"invalid correction count {correction_count} for tile {tile_size}")
    if precision == 16:
        if correction_count != 0:
            raise ValueError("raw BF16 records cannot carry corrections")
        return align4(HEADER_BYTES + tile_size * RAW_VALUE_BYTES)
    if precision not in (4, 5):
        raise ValueError(f"unsupported precision tier: {precision}")
    subblocks = tile_size // BLOCK_SIZE
    base_bytes = subblocks * (P4_PAYLOAD_BYTES if precision == 4 else P5_PAYLOAD_BYTES)
    position_width = 1 if tile_size == 256 else 2
    return align4(HEADER_BYTES + base_bytes + correction_count * (position_width + POSITION_VALUE_BYTES))


def select_precision(tile_size: int, count_p4: int, count_p5: int) -> tuple[int, int, int]:
    """Apply strict-cheaper selection; ties prefer raw, then p5, then p4."""
    raw_cost = tile_record_size(tile_size, 16, 0)
    p5_cost = tile_record_size(tile_size, 5, count_p5)
    p4_cost = tile_record_size(tile_size, 4, count_p4)
    selected_precision = 16
    selected_cost = raw_cost
    selected_count = 0
    if p5_cost < selected_cost:
        selected_precision, selected_cost, selected_count = 5, p5_cost, count_p5
    if p4_cost < selected_cost:
        selected_precision, selected_cost, selected_count = 4, p4_cost, count_p4
    return selected_precision, selected_count, selected_cost


def run_cpu_wire_controls() -> dict[str, Any]:
    """Exercise deterministic size/tie and wire-header boundaries on CPU."""
    if align4(4) != 4 or align4(5) != 8 or align4(8) != 8:
        raise RuntimeError("four-byte record alignment controls failed")
    if tile_record_size(256, 16, 0) != 516:
        raise RuntimeError("raw 256-value tile size changed")
    # These equal-cost controls are intentionally direct: they test the policy
    # gate independently from the formulas that produce real candidate sizes.
    def choose_from_costs(cost4: int, cost5: int, raw: int) -> int:
        precision, best = 16, raw
        if cost5 < best:
            precision, best = 5, cost5
        if cost4 < best:
            precision, best = 4, cost4
        return precision

    if choose_from_costs(80, 80, 80) != 16:
        raise RuntimeError("exact tie must retain raw BF16")
    if choose_from_costs(80, 80, 81) != 5:
        raise RuntimeError("p4/p5 tie must retain p5")
    if choose_from_costs(79, 80, 80) != 4:
        raise RuntimeError("strictly cheaper p4 must be admitted")
    if choose_from_costs(80, 79, 80) != 5:
        raise RuntimeError("strictly cheaper p5 must be admitted")
    if select_precision(256, 125, 114)[0] != 16:
        raise RuntimeError("real p4/p5 costs tied with raw must retain raw")
    if select_precision(256, 115, 105)[0] != 5:
        raise RuntimeError("real p4/p5 equal cost must retain p5")
    if select_precision(256, 124, 114)[0] != 4:
        raise RuntimeError("real p4 cost strictly below p5/raw must select p4")
    plan, offsets, _ = make_plan(
        256,
        np.asarray([124], dtype="<u2"),
        np.asarray([114], dtype="<u2"),
    )
    if int(plan[0, 0]) != int(offsets[1] - offsets[0]):
        raise RuntimeError("plan record-size word differs from the offset delta")
    threshold = np.float32(0.10)
    threshold_cases = (
        (np.float32(0.099), np.float32(1.0), False),
        (np.float32(0.10), np.float32(1.0), False),
        (np.float32(0.101), np.float32(1.0), True),
        (np.float32(0.0), np.float32(0.0), False),
        (np.float32(1.0), np.float32(0.0), True),
    )
    for error, sigma, expected in threshold_cases:
        actual = (
            bool(error != 0)
            if sigma == 0
            else bool(np.float32(error / sigma) > threshold)
        )
        if actual != expected:
            raise RuntimeError("strict normalized-error / zero-sigma CPU control failed")
    if position_value_bytes(
        256,
        np.asarray([0, 255], dtype=np.int32),
        np.asarray([0x3C00, 0x4000], dtype="<u2"),
    ) != bytes([0, 0, 0x3C, 255, 0, 0x40]):
        raise RuntimeError("UInt8 correction position/value interleave control failed")
    if position_value_bytes(
        512,
        np.asarray([0, 511], dtype=np.int32),
        np.asarray([0x3C00, 0x4000], dtype="<u2"),
    ) != bytes([0, 0, 0, 0x3C, 0xFF, 1, 0, 0x40]):
        raise RuntimeError("UInt16 correction position/value interleave control failed")
    for tile_size in TILE_SIZES:
        if tile_size % BLOCK_SIZE or tile_record_size(tile_size, 16, 0) % 4:
            raise RuntimeError("tile geometry/aligned raw record control failed")
        position_width = 1 if tile_size == 256 else 2
        if position_width == 1 and tile_size - 1 > 0xFF:
            raise RuntimeError("UInt8 position boundary control failed")
        if position_width == 2 and tile_size - 1 > 0xFFFF:
            raise RuntimeError("UInt16 position boundary control failed")
    return {
        "passed": True,
        "controls": [
            "four-byte alignment",
            "raw 256-value record length",
            "exact raw-vs-p5 tie retains raw",
            "exact p4-vs-p5 tie retains p5",
            "strictly cheaper p4 is selected",
            "strictly cheaper p5 is selected",
            "actual aligned record-cost tie boundaries retain raw then p5",
            "plan record-size word matches the offset delta",
            "strict normalized-error boundary and sigma-zero semantics",
            "UInt8 and UInt16 interleaved correction bytes",
            "all three tile geometries use representable positions",
        ],
    }


def file_record(path: Path, dtype: str, shape: list[int]) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "dtype": dtype,
        "shape": shape,
        "sha256": diagnostics.sha256_file(path),
    }


def load_qbit_lut(path: Path, precision: int) -> np.ndarray:
    lut = np.fromfile(path, dtype="<f4")
    if lut.shape != (256,) or not np.all(np.isfinite(lut)):
        raise RuntimeError(f"Invalid p{precision} reconstruction LUT")
    if not np.array_equal(lut[:128], -lut[255:127:-1]):
        raise RuntimeError(f"p{precision} reconstruction LUT is not reflection-symmetric")
    return lut


def decode_qbit_weight_on_mps(
    payload_path: Path,
    lut: np.ndarray,
    precision: int,
    row_chunk: int,
) -> torch.Tensor:
    """Decode the existing transposed QBit planes with their exact LUT order."""
    stride = 8 + precision * 32
    lut_mps = require_mps(
        torch.from_numpy(np.ascontiguousarray(lut)).to(device="mps", dtype=torch.float32),
        f"p{precision} QBit LUT",
    )
    decoded = require_mps(
        torch.empty((OUTPUT_DIM, INPUT_DIM), device="mps", dtype=torch.float32),
        f"p{precision} decoded weight",
    )
    with payload_path.open("rb") as stream:
        for first in range(0, OUTPUT_DIM, row_chunk):
            last = min(OUTPUT_DIM, first + row_chunk)
            count = last - first
            moments, codes = diagnostics.read_qbit_fields(stream, first, count, precision)
            moments_mps = require_mps(
                torch.from_numpy(moments).to(device="mps", dtype=torch.float32),
                f"p{precision} block moments",
            )
            code_indices = require_mps(
                torch.from_numpy(codes.astype(np.int64)).to(device="mps"),
                f"p{precision} transposed raw codes",
            )
            lut_values = torch.index_select(lut_mps, 0, code_indices.reshape(-1)).reshape(
                -1, BLOCK_SIZE
            )
            block_values = moments_mps[:, 0:1] + moments_mps[:, 1:2] * lut_values
            decoded[first:last] = block_values.reshape(count, INPUT_DIM)
            del moments_mps, code_indices, lut_values, block_values
    return decoded


def load_sigma_headers(p4_path: Path, p5_path: Path) -> np.ndarray:
    """Read and cross-check the common per-256 QBit sigma headers."""
    p4 = np.memmap(
        p4_path,
        dtype=np.uint8,
        mode="r",
        shape=(OUTPUT_DIM, BLOCKS_PER_ROW, P4_PAYLOAD_BYTES),
    )
    p5 = np.memmap(
        p5_path,
        dtype=np.uint8,
        mode="r",
        shape=(OUTPUT_DIM, BLOCKS_PER_ROW, P5_PAYLOAD_BYTES),
    )
    if not np.array_equal(p4[:, :, :8], p5[:, :, :8]):
        raise RuntimeError("p4/p5 QBit block mean/sigma headers differ")
    moments = np.ascontiguousarray(p4[:, :, :8]).view("<f4").reshape(
        OUTPUT_DIM, BLOCKS_PER_ROW, 2
    )
    sigma = np.asarray(moments[:, :, 1], dtype="<f4")
    if not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
        raise RuntimeError("QBit sigma headers must be finite and non-negative")
    return sigma


def normalized_error_mask(
    source_blocks: torch.Tensor,
    base_blocks: torch.Tensor,
    sigma_blocks: torch.Tensor,
    threshold: np.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return strict threshold mask and absolute residual, with zero-sigma rule."""
    require_mps(source_blocks, "source values for correction mask")
    require_mps(base_blocks, "base values for correction mask")
    require_mps(sigma_blocks, "sigma for correction mask")
    error = torch.abs(base_blocks - source_blocks)
    zero_sigma = sigma_blocks == 0
    safe_sigma = torch.where(zero_sigma, torch.ones_like(sigma_blocks), sigma_blocks)
    ratio = error / safe_sigma
    mask = torch.where(zero_sigma, error != 0, ratio > float(threshold))
    return require_mps(mask, "MPS correction mask"), error


def count_candidate_values(
    source_weight: np.memmap,
    decoded_p4: torch.Tensor,
    decoded_p5: torch.Tensor,
    sigma_cpu: np.ndarray,
    row_chunk: int,
) -> dict[tuple[int, float], tuple[np.ndarray, np.ndarray]]:
    """Count p4/p5 candidate corrections on MPS for every geometry/policy."""
    result: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
    for tile_size in TILE_SIZES:
        tiles_per_row = INPUT_DIM // tile_size
        for threshold in THRESHOLDS:
            result[(tile_size, float(threshold))] = (
                np.zeros((OUTPUT_DIM, tiles_per_row), dtype="<u2"),
                np.zeros((OUTPUT_DIM, tiles_per_row), dtype="<u2"),
            )

    for first in range(0, OUTPUT_DIM, row_chunk):
        last = min(OUTPUT_DIM, first + row_chunk)
        count = last - first
        source_cpu = np.array(source_weight[first:last], dtype="<f4", copy=True)
        source_mps = require_mps(
            torch.from_numpy(source_cpu).to(device="mps", dtype=torch.float32),
            "BF16-derived source weight chunk",
        )
        source_blocks = source_mps.reshape(count, BLOCKS_PER_ROW, BLOCK_SIZE)
        p4_blocks = decoded_p4[first:last].reshape(count, BLOCKS_PER_ROW, BLOCK_SIZE)
        p5_blocks = decoded_p5[first:last].reshape(count, BLOCKS_PER_ROW, BLOCK_SIZE)
        sigma_mps = require_mps(
            torch.from_numpy(np.array(sigma_cpu[first:last], dtype="<f4", copy=True))
            .to(device="mps", dtype=torch.float32)
            .unsqueeze(-1),
            "per-256 sigma chunk",
        )
        mask4_by_threshold = []
        mask5_by_threshold = []
        for threshold in THRESHOLDS:
            mask4, _ = normalized_error_mask(
                source_blocks, p4_blocks, sigma_mps, threshold
            )
            mask5, _ = normalized_error_mask(
                source_blocks, p5_blocks, sigma_mps, threshold
            )
            mask4_by_threshold.append(mask4)
            mask5_by_threshold.append(mask5)

        for tile_size in TILE_SIZES:
            tiles_per_row = INPUT_DIM // tile_size
            subblocks = tile_size // BLOCK_SIZE
            for threshold_index, threshold in enumerate(THRESHOLDS):
                mask4_tiles = mask4_by_threshold[threshold_index].reshape(
                    count, tiles_per_row, subblocks, BLOCK_SIZE
                )
                mask5_tiles = mask5_by_threshold[threshold_index].reshape(
                    count, tiles_per_row, subblocks, BLOCK_SIZE
                )
                count4 = torch.sum(mask4_tiles, dim=(2, 3), dtype=torch.int32)
                count5 = torch.sum(mask5_tiles, dim=(2, 3), dtype=torch.int32)
                target4, target5 = result[(tile_size, float(threshold))]
                target4[first:last] = np.asarray(
                    count4.to("cpu").numpy(), dtype="<u2"
                )
                target5[first:last] = np.asarray(
                    count5.to("cpu").numpy(), dtype="<u2"
                )
        del (
            source_cpu,
            source_mps,
            source_blocks,
            p4_blocks,
            p5_blocks,
            sigma_mps,
            mask4_by_threshold,
            mask5_by_threshold,
        )
        if first == 0 or last == OUTPUT_DIM or last % (row_chunk * 16) == 0:
            print(f"counted correction candidates through weight row {last}/{OUTPUT_DIM}", flush=True)
    return result


def make_plan(
    tile_size: int,
    count_p4: np.ndarray,
    count_p5: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if count_p4.shape != count_p5.shape:
        raise ValueError("p4/p5 candidate count arrays have different shapes")
    tile_count = int(count_p4.size)
    plan = np.zeros((tile_count, 4), dtype="<u4")
    offsets = np.zeros(tile_count + 1, dtype="<u4")
    tiers = np.empty(tile_count, dtype=np.uint8)
    for index, (p4_count, p5_count) in enumerate(
        zip(count_p4.reshape(-1), count_p5.reshape(-1), strict=True)
    ):
        precision, correction_count, record_size = select_precision(
            tile_size, int(p4_count), int(p5_count)
        )
        if correction_count > 0xFFFF:
            raise RuntimeError(f"Correction count exceeds the UInt16 wire field: {correction_count}")
        plan[index] = (record_size, precision, correction_count, 0)
        tiers[index] = precision
        offsets[index + 1] = int(offsets[index]) + record_size
        if offsets[index + 1] > 0xFFFFFFFF:
            raise RuntimeError("Packed positional blob exceeds UInt32 offset capacity")
    unique, counts = np.unique(tiers, return_counts=True)
    histogram = {"p4": 0, "p5": 0, "raw": 0}
    for tier, count in zip(unique, counts, strict=True):
        histogram[{4: "p4", 5: "p5", 16: "raw"}[int(tier)]] = int(count)
    return plan, offsets, histogram


def position_value_bytes(
    tile_size: int,
    positions: np.ndarray,
    value_bits: np.ndarray,
) -> bytes:
    if positions.ndim != 1 or value_bits.ndim != 1 or positions.size != value_bits.size:
        raise ValueError("positions and correction values must be equal-length vectors")
    if positions.size and (
        positions[0] < 0
        or positions[-1] >= tile_size
        or np.any(positions[1:] <= positions[:-1])
    ):
        raise ValueError("correction positions must be in-range and strictly increasing")
    out = bytearray()
    position_width = 1 if tile_size == 256 else 2
    for position, value in zip(positions, value_bits, strict=True):
        if position_width == 1:
            out.append(int(position))
        else:
            out.extend(struct.pack("<H", int(position)))
        out.extend(struct.pack("<H", int(value)))
    return bytes(out)


def encode_record(
    tile_size: int,
    precision: int,
    value_mode: int,
    positions: np.ndarray,
    values: np.ndarray,
    raw_bf16: np.ndarray,
    qbit_streams: dict[int, Any],
    qbit_row: int,
    first_block: int,
) -> bytes:
    if value_mode not in (0, 1):
        raise ValueError(f"unsupported correction value mode: {value_mode}")
    correction_count = int(positions.size)
    if precision == 16:
        if correction_count or values.size:
            raise RuntimeError("raw BF16 tile unexpectedly carries correction values")
        record = bytearray(struct.pack("<BBH", 16, value_mode, 0))
        if raw_bf16.shape != (tile_size,):
            raise ValueError("raw BF16 tile has the wrong number of values")
        record.extend(np.asarray(raw_bf16, dtype="<u2").tobytes())
    elif precision in (4, 5):
        if values.size != correction_count:
            raise ValueError("correction value count does not match plan")
        stride = 8 + precision * 32
        blocks = tile_size // BLOCK_SIZE
        stream = qbit_streams[precision]
        row_byte_offset = (qbit_row * BLOCKS_PER_ROW + first_block) * stride
        stream.seek(row_byte_offset)
        base_payload = stream.read(blocks * stride)
        if len(base_payload) != blocks * stride:
            raise RuntimeError(f"short read from p{precision} QBit tile payload")
        record = bytearray(struct.pack("<BBH", precision, value_mode, correction_count))
        record.extend(base_payload)
        record.extend(position_value_bytes(tile_size, positions, values))
    else:
        raise ValueError(f"unsupported precision tier in plan: {precision}")
    record.extend(b"\x00" * (align4(len(record)) - len(record)))
    return bytes(record)


def build_physical_files(
    stage: Path,
    tile_size: int,
    threshold: np.float32,
    value_mode: int,
    plan: np.ndarray,
    offsets: np.ndarray,
    source_weight: np.memmap,
    source_bf16: np.memmap,
    decoded_p4: torch.Tensor,
    decoded_p5: torch.Tensor,
    sigma_cpu: np.ndarray,
    input_mps: torch.Tensor,
    bias_mps: torch.Tensor,
    weight_row_chunk: int,
    qbit_streams: dict[int, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    mode_name = dict(VALUE_MODES)[value_mode]
    threshold_token = f"{int(round(float(threshold) * 100)):03d}"
    stem = f"tile{tile_size}_t{threshold_token}_mode{value_mode}"
    plan_path = stage / f"plan_tile{tile_size}_t{threshold_token}.u32"
    offsets_path = stage / f"offsets_tile{tile_size}_t{threshold_token}.u32"
    blob_path = stage / f"blob_{stem}.bin"
    decoded_path = stage / f"decoded_weight_{stem}.f32"
    projection_path = stage / f"projection_{stem}.f32"

    if not plan_path.exists():
        plan.tofile(plan_path)
    if not offsets_path.exists():
        offsets.tofile(offsets_path)

    tile_count = plan.shape[0]
    tiles_per_row = INPUT_DIM // tile_size
    if tile_count != OUTPUT_DIM * tiles_per_row:
        raise RuntimeError("plan tile count does not match the row-bounded geometry")

    decoded_weight = require_mps(
        torch.empty((OUTPUT_DIM, INPUT_DIM), device="mps", dtype=torch.float32),
        f"{stem} corrected decoded weight",
    )
    error_sum = 0.0
    reference_energy_sum = 0.0
    max_abs_error = 0.0
    nonfinite_count = 0
    correction_count = 0
    f16_policy_misses = 0
    tier_counts = {"p4": 0, "p5": 0, "raw": 0}
    for precision in (4, 5, 16):
        tier_counts[{4: "p4", 5: "p5", 16: "raw"}[precision]] = int(
            np.count_nonzero(plan[:, 1] == precision)
        )
    correction_count = int(np.sum(plan[:, 2], dtype=np.uint64))

    blob_offset = 0
    with blob_path.open("wb") as blob_stream, decoded_path.open("wb") as decoded_stream:
        for first_row in range(0, OUTPUT_DIM, weight_row_chunk):
            last_row = min(OUTPUT_DIM, first_row + weight_row_chunk)
            count = last_row - first_row
            source_cpu = np.array(source_weight[first_row:last_row], dtype="<f4", copy=True)
            raw_bf16_rows = np.asarray(
                source_bf16[first_row:last_row], dtype="<u2"
            )
            source_mps = require_mps(
                torch.from_numpy(source_cpu).to(device="mps", dtype=torch.float32),
                "source BF16-derived F32 values",
            )
            source_tiles = source_mps.reshape(count, tiles_per_row, tile_size)
            p4_tiles = decoded_p4[first_row:last_row].reshape(count, tiles_per_row, tile_size)
            p5_tiles = decoded_p5[first_row:last_row].reshape(count, tiles_per_row, tile_size)
            sigma_cpu_chunk = np.array(sigma_cpu[first_row:last_row], dtype="<f4", copy=True)
            sigma_mps = require_mps(
                torch.from_numpy(sigma_cpu_chunk).to(device="mps", dtype=torch.float32),
                "per-256 sigma values for tile chunk",
            )
            sigma_tiles = sigma_mps.reshape(count, tiles_per_row, tile_size // BLOCK_SIZE, 1)

            first_tile = first_row * tiles_per_row
            last_tile = last_row * tiles_per_row
            plan_chunk = plan[first_tile:last_tile]
            tier_grid_np = plan_chunk[:, 1].reshape(count, tiles_per_row).astype(
                np.int32, copy=True
            )
            tier_grid = require_mps(
                torch.from_numpy(tier_grid_np).to(device="mps", dtype=torch.int32),
                "selected precision tiers",
            )
            base_tiles = torch.where(
                (tier_grid == 4).unsqueeze(-1),
                p4_tiles,
                torch.where((tier_grid == 5).unsqueeze(-1), p5_tiles, source_tiles),
            )

            source_blocks = source_tiles.reshape(
                count, tiles_per_row, tile_size // BLOCK_SIZE, BLOCK_SIZE
            )
            base_blocks = base_tiles.reshape_as(source_blocks)
            sigma_blocks = sigma_tiles.expand_as(source_blocks)
            correction_masks = []
            for precision, precision_blocks in ((4, p4_tiles), (5, p5_tiles)):
                candidate_blocks = precision_blocks.reshape_as(source_blocks)
                candidate_mask, _ = normalized_error_mask(
                    source_blocks,
                    candidate_blocks,
                    sigma_blocks,
                    threshold,
                )
                correction_masks.append(candidate_mask.reshape(count, tiles_per_row, tile_size))
            selected_mask = (
                ((tier_grid == 4).unsqueeze(-1) & correction_masks[0])
                | ((tier_grid == 5).unsqueeze(-1) & correction_masks[1])
            )

            if value_mode == 0:
                decoded_tiles = torch.where(selected_mask, source_tiles, base_tiles)
                delta_half = None
            else:
                delta_half = (source_tiles - base_tiles).to(torch.float16)
                if not bool(torch.all(torch.isfinite(delta_half) | ~selected_mask).item()):
                    raise RuntimeError("F16 correction delta is non-finite at a selected position")
                decoded_delta = base_tiles + delta_half.to(torch.float32)
                decoded_tiles = torch.where(selected_mask, decoded_delta, base_tiles)
                decoded_blocks_final = decoded_tiles.reshape_as(source_blocks)
                final_error = torch.abs(decoded_blocks_final - source_blocks)
                safe_sigma = torch.where(sigma_blocks == 0, torch.ones_like(sigma_blocks), sigma_blocks)
                normalized = final_error / safe_sigma
                violates = torch.where(
                    sigma_blocks == 0,
                    final_error != 0,
                    normalized > float(threshold),
                )
                f16_policy_misses += int(
                    torch.sum(selected_mask.reshape_as(violates) & violates, dtype=torch.int64)
                    .to("cpu")
                    .item()
                )

            if not bool(torch.all(torch.isfinite(decoded_tiles)).item()):
                raise RuntimeError(f"{stem} decoded weight contains non-finite values")
            require_mps(decoded_tiles, f"{stem} corrected values")
            decoded_chunk = decoded_tiles.reshape(count, INPUT_DIM)
            decoded_weight[first_row:last_row] = decoded_chunk

            difference = decoded_chunk - source_mps
            chunk_error = torch.sum(difference * difference, dtype=torch.float32)
            chunk_energy = torch.sum(source_mps * source_mps, dtype=torch.float32)
            chunk_max = torch.amax(torch.abs(difference))
            chunk_nonfinite = torch.sum(
                (~torch.isfinite(decoded_chunk) | ~torch.isfinite(source_mps)).to(torch.int64),
                dtype=torch.int64,
            )
            error_sum += float(chunk_error.to("cpu").item())
            reference_energy_sum += float(chunk_energy.to("cpu").item())
            max_abs_error = max(max_abs_error, float(chunk_max.to("cpu").item()))
            nonfinite_count += int(chunk_nonfinite.to("cpu").item())

            mask_cpu = selected_mask.to("cpu").numpy().reshape(count, tiles_per_row, tile_size)
            if value_mode == 1:
                delta_cpu_bits = np.asarray(
                    delta_half.to("cpu").numpy(), dtype="<f2"
                ).view("<u2").reshape(count, tiles_per_row, tile_size)
            else:
                delta_cpu_bits = None
            decoded_cpu = decoded_chunk.to("cpu").contiguous().numpy()
            np.asarray(decoded_cpu, dtype="<f4").tofile(decoded_stream)

            for local_row in range(count):
                output_row = first_row + local_row
                for tile_index in range(tiles_per_row):
                    global_tile = output_row * tiles_per_row + tile_index
                    precision = int(plan[global_tile, 1])
                    expected_count = int(plan[global_tile, 2])
                    positions = np.flatnonzero(mask_cpu[local_row, tile_index]).astype(
                        np.int32, copy=False
                    )
                    if positions.size != expected_count:
                        raise RuntimeError(
                            f"MPS correction-mask count differs from plan at tile {global_tile}: "
                            f"{positions.size} != {expected_count}"
                        )
                    if precision == 16:
                        if expected_count:
                            raise RuntimeError("raw BF16 plan contains corrections")
                        values = np.empty(0, dtype="<u2")
                    elif value_mode == 0:
                        values = np.asarray(
                            raw_bf16_rows[local_row,
                                          tile_index * tile_size + positions],
                            dtype="<u2",
                        )
                    else:
                        assert delta_cpu_bits is not None
                        values = np.asarray(
                            delta_cpu_bits[local_row, tile_index, positions], dtype="<u2"
                        )
                    record = encode_record(
                        tile_size,
                        precision,
                        value_mode,
                        positions,
                        values,
                        raw_bf16_rows[
                            local_row,
                            tile_index * tile_size : (tile_index + 1) * tile_size,
                        ],
                        qbit_streams,
                        output_row,
                        tile_index * (tile_size // BLOCK_SIZE),
                    )
                    expected_size = int(offsets[global_tile + 1] - offsets[global_tile])
                    if len(record) != expected_size:
                        raise RuntimeError(
                            f"encoded record length differs from offset plan at tile {global_tile}: "
                            f"{len(record)} != {expected_size}"
                        )
                    if blob_offset != int(offsets[global_tile]):
                        raise RuntimeError("packed blob write cursor differs from offset table")
                    blob_stream.write(record)
                    blob_offset += len(record)

            del (
                source_cpu,
                raw_bf16_rows,
                source_mps,
                source_tiles,
                p4_tiles,
                p5_tiles,
                sigma_cpu_chunk,
                sigma_mps,
                sigma_tiles,
                tier_grid_np,
                tier_grid,
                base_tiles,
                source_blocks,
                base_blocks,
                sigma_blocks,
                correction_masks,
                selected_mask,
                decoded_tiles,
                decoded_chunk,
                difference,
                chunk_error,
                chunk_energy,
                chunk_max,
                chunk_nonfinite,
                mask_cpu,
                decoded_cpu,
            )
            if value_mode == 1:
                del delta_half, delta_cpu_bits, decoded_delta, decoded_blocks_final
            if first_row == 0 or last_row == OUTPUT_DIM or last_row % (weight_row_chunk * 16) == 0:
                print(f"{stem}: wrote through weight row {last_row}/{OUTPUT_DIM}", flush=True)

    if blob_offset != int(offsets[-1]):
        raise RuntimeError("packed blob byte length differs from final offset")
    raw_blob_cap = WEIGHT_COUNT * RAW_VALUE_BYTES + tile_count * HEADER_BYTES
    if blob_offset > raw_blob_cap:
        raise RuntimeError(f"packed blob exceeds raw BF16 record-blob cap: {blob_offset} > {raw_blob_cap}")
    decoded_weight_record = file_record(
        decoded_path, "F32", [OUTPUT_DIM, INPUT_DIM]
    )

    projected = require_mps(
        F.linear(input_mps, decoded_weight, bias_mps),
        f"{stem} F.linear projection",
    )
    if projected.dtype != torch.float32 or tuple(projected.shape) != (ROWS, OUTPUT_DIM):
        raise RuntimeError(f"Unexpected projection shape/dtype: {projected.dtype} {projected.shape}")
    repeated = projected.reshape(ROWS // SOURCE_ROWS, SOURCE_ROWS, OUTPUT_DIM)
    if not bool(torch.all(repeated == projected[:SOURCE_ROWS].unsqueeze(0)).item()):
        raise RuntimeError(f"{stem} projection does not repeat bit-exactly across all 1120 tiles")
    np.asarray(projected[:SOURCE_ROWS].to("cpu").contiguous().numpy(), dtype="<f4").tofile(
        projection_path
    )

    projection_record = file_record(
        projection_path, "F32", [SOURCE_ROWS, OUTPUT_DIM]
    )
    plan_record = file_record(
        plan_path, "U32LE", [tile_count, 4]
    )
    offsets_record = file_record(
        offsets_path, "U32LE", [tile_count + 1]
    )
    blob_record = file_record(blob_path, "U8", [blob_offset])
    relative_l2 = (error_sum / reference_energy_sum) ** 0.5 if reference_energy_sum else 0.0
    stats = {
        "squared_error_sum_f32": error_sum,
        "reference_energy_sum_f32": reference_energy_sum,
        "relative_l2": relative_l2,
        "max_abs_error": max_abs_error,
        "nonfinite_count": nonfinite_count,
        "reduction": "row-chunk MPS F32 sums accumulated as host Float64 scalars",
    }
    plan_summary = {
        "selected_tile_counts": tier_counts,
        "correction_count": correction_count,
        "f16_policy_misses": f16_policy_misses,
    }
    physical_bytes = {
        "tile_count": tile_count,
        "blob_bytes": blob_offset,
        "offset_table_bytes": int(offsets.nbytes),
        "raw_record_blob_cap_bytes": raw_blob_cap,
        "blob_and_offsets_bytes": blob_offset + int(offsets.nbytes),
        "raw_blob_and_offsets_bytes": raw_blob_cap + int(offsets.nbytes),
        "blob_and_offsets_bits_per_weight": (
            (blob_offset + int(offsets.nbytes)) * 8 / WEIGHT_COUNT
        ),
        "plan_reference_bytes_excluded_from_accounting": int(plan.nbytes),
        "accounting_scope": (
            "record blob including per-tile headers and padding plus UInt32LE offsets only; "
            "excludes plan, global/container headers, geometry/tile/mode descriptors, "
            "QBit LUTs, and filesystem or directory overhead"
        ),
    }
    arm = {
        "name": f"positional_{stem}",
        "tile_size": tile_size,
        "threshold": float(threshold),
        "threshold_f32_hex": f"0x{int(np.asarray(threshold).view('<u4')):08x}",
        "value_mode": value_mode,
        "value_mode_name": mode_name,
        "precision_tie_order": [16, 5, 4],
        "plan": plan_record,
        "offsets": offsets_record,
        "blob": blob_record,
        "decoded_weight": decoded_weight_record,
        "projection": projection_record,
        "decoded_weight_stats": stats,
        "plan_summary": plan_summary,
        "storage_accounting": physical_bytes,
        "projection_provenance": {
            "operation": "torch.nn.functional.linear(input, corrected_decoded_weight, bias)",
            "rows_computed": ROWS,
            "rows_saved": SOURCE_ROWS,
            "all_periodic_repeats_bitexact": True,
        },
    }
    del decoded_weight, projected, repeated
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    return arm, {"plan": plan, "offsets": offsets, "histogram": tier_counts}


def build(args: argparse.Namespace) -> Path:
    cpu_controls = run_cpu_wire_controls()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for QBit decode, mask selection and projection")
    if args.weight_row_chunk < 1:
        raise ValueError("--weight-row-chunk must be positive")

    fixture_root = args.fixture.expanduser().resolve()
    fixture_manifest, fixture_paths = diagnostics.load_fixture(fixture_root)
    source_weight = np.memmap(
        fixture_paths["source_weight"],
        dtype="<f4",
        mode="r",
        shape=(OUTPUT_DIM, INPUT_DIM),
    )
    source_weight_bf16_record = fixture_manifest["source_tensor"]["weight_file"]
    source_bf16_path = diagnostics.verify_record(
        fixture_root, source_weight_bf16_record, "original BF16 source weight"
    )
    source_bf16 = np.memmap(
        source_bf16_path,
        dtype="<u2",
        mode="r",
        shape=(OUTPUT_DIM, INPUT_DIM),
    )
    sigma_cpu = load_sigma_headers(fixture_paths["qbit_p4"], fixture_paths["qbit_p5"])
    luts = {
        precision: load_qbit_lut(fixture_paths[f"lut_p{precision}"], precision)
        for precision in (4, 5)
    }

    source_input = np.fromfile(fixture_paths["input"], dtype="<f4").reshape(
        SOURCE_ROWS, INPUT_DIM
    )
    source_input_mps = require_mps(
        torch.from_numpy(source_input.copy()).to(device="mps", dtype=torch.float32),
        "four seeded Wan block-probe projection input rows",
    )
    input_rows = torch.arange(ROWS, device="mps", dtype=torch.int64) % SOURCE_ROWS
    input_mps = require_mps(
        torch.index_select(source_input_mps, 0, input_rows),
        "periodically tiled 4480-row projection input",
    )
    bias = np.fromfile(fixture_paths["bias"], dtype="<f4")
    if bias.shape != (OUTPUT_DIM,) or not np.all(np.isfinite(bias)):
        raise RuntimeError("Wan projection bias must be finite F32 with expected length")
    bias_mps = require_mps(
        torch.from_numpy(bias.copy()).to(device="mps", dtype=torch.float32),
        "Wan projection bias",
    )

    output_path = args.output.expanduser().resolve()
    if os.path.lexists(output_path):
        raise FileExistsError(f"Immutable positional reference output exists: {output_path}")
    if (
        output_path == fixture_root
        or output_path in fixture_root.parents
        or fixture_root in output_path.parents
    ):
        raise ValueError("Output directory must not overlap the immutable source fixture")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=output_path.parent)
    )
    qbit_streams = {
        4: fixture_paths["qbit_p4"].open("rb"),
        5: fixture_paths["qbit_p5"].open("rb"),
    }
    try:
        print("Decoding the pinned p4 QBit weight on MPS...", flush=True)
        decoded_p4 = decode_qbit_weight_on_mps(
            fixture_paths["qbit_p4"], luts[4], 4, args.weight_row_chunk
        )
        print("Decoding the pinned p5 QBit weight on MPS...", flush=True)
        decoded_p5 = decode_qbit_weight_on_mps(
            fixture_paths["qbit_p5"], luts[5], 5, args.weight_row_chunk
        )
        for precision, decoded in ((4, decoded_p4), (5, decoded_p5)):
            if not bool(torch.all(torch.isfinite(decoded)).item()):
                raise RuntimeError(f"p{precision} QBit decoded weight contains non-finite values")
        for first in range(0, OUTPUT_DIM, args.weight_row_chunk):
            last = min(OUTPUT_DIM, first + args.weight_row_chunk)
            source_chunk = np.array(source_weight[first:last], dtype="<f4", copy=True)
            source_mps = require_mps(
                torch.from_numpy(source_chunk).to(device="mps", dtype=torch.float32),
                "finite BF16-derived source chunk",
            )
            if not bool(torch.all(torch.isfinite(source_mps)).item()):
                raise RuntimeError("Original BF16-derived source weight contains non-finite values")
            del source_chunk, source_mps

        print("Counting per-value threshold candidates on MPS...", flush=True)
        candidates = count_candidate_values(
            source_weight,
            decoded_p4,
            decoded_p5,
            sigma_cpu,
            args.weight_row_chunk,
        )

        arms: list[dict[str, Any]] = []
        for tile_size in TILE_SIZES:
            for threshold in THRESHOLDS:
                key = (tile_size, float(threshold))
                count_p4, count_p5 = candidates[key]
                plan, offsets, histogram = make_plan(tile_size, count_p4, count_p5)
                correction_total = int(np.sum(plan[:, 2], dtype=np.uint64))
                print(
                    f"prepared tile{tile_size} threshold={float(threshold):.2f}: "
                    f"p4={histogram['p4']} p5={histogram['p5']} raw={histogram['raw']} "
                    f"corrections={correction_total}",
                    flush=True,
                )
                for value_mode, mode_name in VALUE_MODES:
                    print(
                        f"building tile{tile_size} threshold={float(threshold):.2f} "
                        f"value_mode={mode_name}",
                        flush=True,
                    )
                    arm, _ = build_physical_files(
                        stage,
                        tile_size,
                        threshold,
                        value_mode,
                        plan,
                        offsets,
                        source_weight,
                        source_bf16,
                        decoded_p4,
                        decoded_p5,
                        sigma_cpu,
                        input_mps,
                        bias_mps,
                        args.weight_row_chunk,
                        qbit_streams,
                    )
                    arm["plan_summary"]["selected_tile_counts"] = histogram
                    arms.append(arm)
                    if hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()

        source_hashes = {
            "source_fixture_manifest_sha256": diagnostics.SOURCE_MANIFEST_SHA256,
            "capture_manifest_sha256": fixture_manifest["input_provenance"]["manifest_sha256"],
            "checkpoint_weight_bf16_sha256": source_weight_bf16_record["sha256"],
            "decoded_bf16_source_f32_sha256": fixture_manifest["variants"]["bf16"][
                "decoded_weight"
            ]["sha256"],
            "input_f32_sha256": fixture_manifest["input"]["sha256"],
            "bias_bf16_sha256": fixture_manifest["source_tensor"]["bias_bf16_file"][
                "sha256"
            ],
            "bias_f32_sha256": fixture_manifest["source_tensor"]["bias_f32_file"][
                "sha256"
            ],
            "qbit_p4_payload_sha256": fixture_manifest["variants"]["qbit_p4"][
                "packed_weight"
            ]["sha256"],
            "qbit_p5_payload_sha256": fixture_manifest["variants"]["qbit_p5"][
                "packed_weight"
            ]["sha256"],
            "qbit_lut_p4_sha256": fixture_manifest["variants"]["qbit_p4"]["lut"][
                "sha256"
            ],
            "qbit_lut_p5_sha256": fixture_manifest["variants"]["qbit_p5"]["lut"][
                "sha256"
            ],
            "qbit_codec_source_sha256": diagnostics.CODEC_SHA256,
            "projection_reference_script_sha256": fixture_manifest["source_hashes"][
                "script_sha256"
            ],
            "diagnostic_helper_script_sha256": diagnostics.sha256_file(
                Path(diagnostics.__file__).resolve()
            ),
            "positional_reference_script_sha256": diagnostics.sha256_file(
                Path(__file__).resolve()
            ),
        }
        manifest = {
            "schema": "wan_qbit_positional_weight_reference_v1",
            "created_by": Path(__file__).name,
            "source_manifest_sha256": diagnostics.SOURCE_MANIFEST_SHA256,
            "source_fixture_manifest_path": str(fixture_root / "manifest.json"),
            "source_hashes": source_hashes,
            "geometry": {
                "rows": ROWS,
                "source_rows": SOURCE_ROWS,
                "input_dim": INPUT_DIM,
                "output_dim": OUTPUT_DIM,
                "normalization_block_size": BLOCK_SIZE,
                "tile_sizes": list(TILE_SIZES),
                "input_scope": fixture_manifest["input_provenance"],
                "input_expansion": "MPS index_select(arange(4480) % 4)",
                "projection_rows_saved": SOURCE_ROWS,
                "all_periodic_repeats_bitexact": True,
            },
            "oracle": {
                "device": "MPS",
                "fallback_disabled": True,
                "no_full_wan_model_loaded": True,
                "weight_decode": "raw transposed QBit codes -> MPS index_select LUT -> mean + sigma * LUT",
                "candidate_mask": "MPS abs(base - BF16_source) / per-256 sigma > Float32 threshold; sigma==0 selects iff error!=0",
                "projection": "torch.nn.functional.linear(input, corrected_decoded_weight, bias)",
                "offline_host_waits_and_exports": True,
                "native_measured_path_claim": False,
            },
            "selection_policy": {
                "thresholds_f32": [float(value) for value in THRESHOLDS],
                "candidate_comparison": "strict greater-than after Float32 division",
                "tile_cost": "align4(4-byte header + base payload + K * (position_width + 2-byte value))",
                "raw_cost": "align4(4-byte header + 2 * tile_size)",
                "position_width_bytes": {"256": 1, "512": 2, "1024": 2},
                "tie_order": [16, 5, 4],
                "selection": "start raw; select p5 iff strictly cheaper; then p4 iff strictly cheaper than current best; thresholds are independent fixed policies",
                "f16_mode": "reuse BF16-policy masks and plan; F16(original_F32 - base_F32), then separate F32(base_F32 + delta_F16); report threshold misses without changing plan",
            },
            "wire_format": {
                "offsets": "UInt32LE byte offsets relative to blob; tiles+1 entries; first offset zero",
                "record_header": "precision U8, value_mode U8 (0 BF16 absolute, 1 F16 delta), correction_count UInt16LE",
                "quantized_base": "preserve original canonical p4/p5 QBit blocks, including per-256 mean/sigma and transposed bit planes",
                "corrections": "strictly increasing local position then UInt16LE value; position UInt8 for tile256, UInt16LE for tile512/1024",
                "raw_record": "precision16, correction_count0, original BF16 tile payload",
                "padding": "zero bytes to four-byte record alignment",
                "plan": "four UInt32LE values per tile: [record_size_bytes, precision, correction_count, invalid=0]",
            },
            "cpu_controls": cpu_controls,
            "arms": arms,
        }
        manifest_path = stage / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, output_path)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        for stream in qbit_streams.values():
            stream.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=diagnostics.DEFAULT_FIXTURE,
        help="immutable pinned 4480-row projection fixture",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new immutable output directory; existing paths are rejected",
    )
    parser.add_argument(
        "--weight-row-chunk",
        type=int,
        default=16,
        help="rows per MPS weight decode/selection chunk (default: 16)",
    )
    return parser.parse_args()


def main() -> None:
    output = build(parse_args())
    print(f"wrote immutable positional reference fixture: {output}", flush=True)


if __name__ == "__main__":
    main()
