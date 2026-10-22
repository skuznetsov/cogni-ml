#!/usr/bin/env python3
"""Prepare bounded QBit Gaussian projection ablations from an immutable Wan fixture.

This is an offline MPS oracle for diagnosing projection error. It consumes the
four captured input rows and one already-decoded FFN weight matrix; it never
loads Wan or creates a new activation capture. Every dequantized weight and
projection is computed on MPS with PyTorch fallback disabled. Each output
stores only the first four projected rows after verifying exact periodic
repetition over the full 4480-row geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
import torch.nn.functional as F


SOURCE_MANIFEST_SHA256 = "0dba2df508f72764837c23d8c5743a6d687e2d1b6104406e38325031084c3703"
CODEC_SHA256 = "6397a42e05a44bf6c1a5b588aa02e197de617252ecb6bab0044f8146efd7eab5"
SOURCE_ROWS = 4
ROWS = 4480
INPUT_DIM = 5120
OUTPUT_DIM = 13824
BLOCK_SIZE = 256
DEFAULT_FIXTURE = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/"
    "wan-quantized-projection-20261022/fixture-4480-v3"
)
DEFAULT_CODEC = Path(__file__).resolve().parents[2] / "src/ml/gguf/qwen_qbit_gaussian_codec.cr"
GAUSSIAN_MSE_ANCHORS = {4: 0.017712426355872203, 5: 0.005485447686857607}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_mps(tensor: torch.Tensor, label: str) -> torch.Tensor:
    if tensor.device.type != "mps":
        raise RuntimeError(f"{label} must be on MPS, found {tensor.device}")
    return tensor


def file_record(path: Path, dtype: str, shape: list[int]) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "dtype": dtype,
        "shape": shape,
        "sha256": sha256_file(path),
    }


def verify_record(root: Path, record: dict[str, Any], label: str) -> Path:
    name = record.get("path")
    if not isinstance(name, str) or Path(name).name != name:
        raise RuntimeError(f"{label} has an unsafe fixture path: {name!r}")
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual_bytes = path.stat().st_size
    actual_hash = sha256_file(path)
    if actual_bytes != record.get("bytes") or actual_hash != record.get("sha256"):
        raise RuntimeError(f"{label} does not match its pinned fixture manifest")
    return path


def load_fixture(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest_path = root / "manifest.json"
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != SOURCE_MANIFEST_SHA256:
        raise RuntimeError(
            "Fixture manifest SHA-256 mismatch: expected "
            f"{SOURCE_MANIFEST_SHA256}, found {actual_manifest_hash}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "wan_quantized_projection_reference_v1":
        raise RuntimeError(f"Unexpected source fixture schema: {manifest.get('schema')}")
    geometry = manifest.get("geometry", {})
    expected_geometry = {
        "rows": ROWS,
        "source_rows": SOURCE_ROWS,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
    }
    for key, expected in expected_geometry.items():
        if geometry.get(key) != expected:
            raise RuntimeError(f"Unexpected source geometry {key}: {geometry.get(key)!r}")
    if geometry.get("expected_projection_rows") != ROWS:
        raise RuntimeError("Source fixture does not contain the full 4480-row projection geometry")

    variants = manifest.get("variants", {})
    source_tensor = manifest.get("source_tensor", {})
    records = {
        "input": manifest.get("input"),
        "source_weight": variants.get("bf16", {}).get("decoded_weight"),
        "bias": source_tensor.get("bias_f32_file"),
        "qbit_p4": variants.get("qbit_p4", {}).get("packed_weight"),
        "qbit_p5": variants.get("qbit_p5", {}).get("packed_weight"),
        "lut_p4": variants.get("qbit_p4", {}).get("lut"),
        "lut_p5": variants.get("qbit_p5", {}).get("lut"),
    }
    if any(not isinstance(record, dict) for record in records.values()):
        raise RuntimeError("Source fixture is missing a required input, weight, bias, payload, or LUT")
    expected_shapes = {
        "input": [SOURCE_ROWS, INPUT_DIM],
        "source_weight": [OUTPUT_DIM, INPUT_DIM],
        "bias": [OUTPUT_DIM],
        "qbit_p4": [OUTPUT_DIM, INPUT_DIM // BLOCK_SIZE, 136],
        "qbit_p5": [OUTPUT_DIM, INPUT_DIM // BLOCK_SIZE, 168],
        "lut_p4": [256],
        "lut_p5": [256],
    }
    paths: dict[str, Path] = {}
    for key, record in records.items():
        if record.get("shape") != expected_shapes[key]:
            raise RuntimeError(f"Unexpected {key} shape in source fixture: {record.get('shape')}")
        paths[key] = verify_record(root, record, key)
    return manifest, paths


def parse_codec_levels(codec_path: Path) -> np.ndarray:
    if sha256_file(codec_path) != CODEC_SHA256:
        raise RuntimeError("Repository QBit codec source does not match the pinned source hash")
    source = codec_path.read_text(encoding="utf-8")
    match = re.search(
        r"private POSITIVE_LEVELS\s*=\s*\[(.*?)\]\s+of Float32", source, re.S
    )
    if match is None:
        raise RuntimeError("Could not find pinned POSITIVE_LEVELS in the QBit codec")
    tokens = re.findall(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)_f32",
        match.group(1),
    )
    levels = np.asarray([np.float32(float(token)) for token in tokens], dtype="<f4")
    if levels.shape != (128,) or not np.all(np.isfinite(levels)):
        raise RuntimeError(f"Expected 128 finite positive Lloyd-Max levels, got {levels.shape}")
    if np.any(levels[1:] <= levels[:-1]):
        raise RuntimeError("Pinned positive Lloyd-Max levels are not strictly increasing")
    return levels


def make_arithmetic_lut(levels: np.ndarray, precision: int) -> np.ndarray:
    group_width = 1 << (8 - precision)
    centers = np.asarray(
        [
            np.mean(levels[first : first + group_width], dtype=np.float64)
            for first in range(0, 128, group_width)
        ],
        dtype="<f4",
    )
    lut = np.empty(256, dtype="<f4")
    for positive_code in range(128):
        center = centers[positive_code // group_width]
        lut[positive_code] = center
        lut[255 - positive_code] = np.float32(-center)
    return lut


def normal_pdf(value: float) -> float:
    if math.isinf(value):
        return 0.0
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def normal_cdf(value: float) -> float:
    if math.isinf(value):
        return 0.0 if value < 0 else 1.0
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def interval_mse(lower: float, upper: float, center: float) -> float:
    probability = normal_cdf(upper) - normal_cdf(lower)
    first_moment = normal_pdf(lower) - normal_pdf(upper)
    second_moment = (
        probability
        + (0.0 if math.isinf(lower) else lower * normal_pdf(lower))
        - (0.0 if math.isinf(upper) else upper * normal_pdf(upper))
    )
    return second_moment - 2.0 * center * first_moment + center * center * probability


def normal_theory(
    levels: np.ndarray, gaussian_lut: np.ndarray, precision: int
) -> dict[str, float]:
    group_width = 1 << (8 - precision)
    positive_count = 1 << (precision - 1)
    gaussian_centers = np.asarray(gaussian_lut[0:128:group_width], dtype="<f4")
    arithmetic_lut = make_arithmetic_lut(levels, precision)
    arithmetic_centers = np.asarray(arithmetic_lut[0:128:group_width], dtype="<f4")
    max_relative_conditional_error = 0.0
    gaussian_mse = 0.0
    arithmetic_mse = 0.0

    for index in range(positive_count):
        first = index * group_width
        end = first + group_width
        lower = (
            0.0
            if first == 0
            else float(np.float32((levels[first - 1] + levels[first]) * np.float32(0.5)))
        )
        upper = (
            math.inf
            if end == 128
            else float(np.float32((levels[end - 1] + levels[end]) * np.float32(0.5)))
        )
        probability = normal_cdf(upper) - normal_cdf(lower)
        conditional_mean = (normal_pdf(lower) - normal_pdf(upper)) / probability
        center = float(gaussian_centers[index])
        relative_error = abs(center - conditional_mean) / max(abs(conditional_mean), 1e-30)
        max_relative_conditional_error = max(max_relative_conditional_error, relative_error)
        gaussian_mse += 2.0 * interval_mse(lower, upper, center)
        arithmetic_mse += 2.0 * interval_mse(
            lower, upper, float(arithmetic_centers[index])
        )

    nearest_mse = 0.0
    midpoints = np.asarray(
        (gaussian_centers[:-1] + gaussian_centers[1:]) * np.float32(0.5),
        dtype="<f4",
    )
    for index, center in enumerate(gaussian_centers):
        lower = 0.0 if index == 0 else float(midpoints[index - 1])
        upper = math.inf if index + 1 == positive_count else float(midpoints[index])
        nearest_mse += 2.0 * interval_mse(lower, upper, float(center))

    if max_relative_conditional_error > 1e-6:
        raise RuntimeError(
            "Pinned Gaussian LUT failed its scalar conditional-mean check: "
            f"{max_relative_conditional_error:.9g}"
        )
    anchor_error = abs(gaussian_mse - GAUSSIAN_MSE_ANCHORS[precision])
    if anchor_error > 1e-10:
        raise RuntimeError(
            f"p{precision} theoretical Gaussian MSE differs from the recorded anchor: "
            f"{gaussian_mse:.17g}"
        )
    return {
        "fixed_prefix_gaussian_normal_mse": gaussian_mse,
        "fixed_prefix_gaussian_normal_relative_l2": math.sqrt(gaussian_mse),
        "fixed_prefix_arithmetic_normal_mse": arithmetic_mse,
        "nearest_centroid_gaussian_book_normal_mse": nearest_mse,
        "max_relative_conditional_mean_error": max_relative_conditional_error,
        "recorded_gaussian_mse_anchor": GAUSSIAN_MSE_ANCHORS[precision],
        "gaussian_mse_anchor_absolute_error": anchor_error,
    }


def read_qbit_fields(stream, first_row: int, row_count: int, precision: int):
    stride = 8 + precision * 32
    row_bytes = (INPUT_DIM // BLOCK_SIZE) * stride
    stream.seek(first_row * row_bytes)
    payload = stream.read(row_count * row_bytes)
    if len(payload) != row_count * row_bytes:
        raise RuntimeError(f"Short read from QBit p{precision} payload")
    blocks = np.frombuffer(payload, dtype=np.uint8).reshape(-1, stride)
    moments = np.ascontiguousarray(blocks[:, :8]).view("<f4").reshape(-1, 2).copy()
    codes = np.zeros((blocks.shape[0], BLOCK_SIZE), dtype=np.uint8)
    for plane in range(precision):
        plane_data = blocks[:, 8 + plane * 32 : 8 + (plane + 1) * 32]
        natural_bits = np.unpackbits(plane_data[:, ::-1], axis=1, bitorder="little")
        codes |= natural_bits.astype(np.uint8) << (7 - plane)
    return moments, codes


def nearest_center_indices(values: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """Lower-bound search on MPS; ties at a Float32 midpoint select the lower center."""
    boundary_count = boundaries.numel()
    low = torch.zeros_like(values, dtype=torch.int64)
    high = torch.full_like(low, boundary_count)
    for _ in range((boundary_count + 1).bit_length()):
        active = low < high
        middle = (low + high) // 2
        safe_middle = torch.clamp(middle, max=max(0, boundary_count - 1))
        threshold = torch.index_select(
            boundaries, 0, safe_middle.reshape(-1)
        ).reshape_as(values)
        below = active & (threshold < values)
        above = active & ~below
        low = torch.where(below, middle + 1, low)
        high = torch.where(above, middle, high)
    return low


def decode_weight_on_mps(
    fixture_paths: dict[str, Path],
    source_weight: np.memmap,
    precision: int,
    mode: str,
    threshold: float | None,
    lut: np.ndarray,
    row_chunk: int,
) -> torch.Tensor:
    group_width = 1 << (8 - precision)
    gaussian_centers = np.asarray(lut[0:128:group_width], dtype="<f4")
    boundaries = np.asarray(
        (gaussian_centers[:-1] + gaussian_centers[1:]) * np.float32(0.5),
        dtype="<f4",
    )
    lut_mps = require_mps(
        torch.from_numpy(np.ascontiguousarray(lut)).to(device="mps", dtype=torch.float32),
        f"QBit p{precision} reconstruction LUT",
    )
    centers_mps = require_mps(
        torch.from_numpy(gaussian_centers.copy()).to(device="mps", dtype=torch.float32),
        f"QBit p{precision} positive centroid book",
    )
    boundaries_mps = require_mps(
        torch.from_numpy(boundaries.copy()).to(device="mps", dtype=torch.float32),
        f"QBit p{precision} nearest-centroid boundaries",
    )
    decoded = require_mps(
        torch.empty((OUTPUT_DIM, INPUT_DIM), device="mps", dtype=torch.float32),
        f"QBit p{precision} {mode} full decoded weight",
    )
    payload_path = fixture_paths[f"qbit_p{precision}"]
    with payload_path.open("rb") as stream:
        for first in range(0, OUTPUT_DIM, row_chunk):
            last = min(OUTPUT_DIM, first + row_chunk)
            count = last - first
            moments, codes = read_qbit_fields(stream, first, count, precision)
            moments_mps = require_mps(
                torch.from_numpy(moments).to(device="mps", dtype=torch.float32),
                "QBit block mean and sigma",
            )
            code_indices = require_mps(
                torch.from_numpy(codes.astype(np.int64)).to(device="mps"),
                "QBit raw code indices",
            )
            if mode == "arithmetic" or mode == "tail_restore":
                selected = torch.index_select(
                    lut_mps, 0, code_indices.reshape(-1)
                ).reshape(-1, BLOCK_SIZE)
                decoded_blocks = moments_mps[:, 0:1] + moments_mps[:, 1:2] * selected
                if mode == "tail_restore":
                    source_cpu = np.array(source_weight[first:last], dtype="<f4", copy=True)
                    source_mps = require_mps(
                        torch.from_numpy(source_cpu).to(device="mps", dtype=torch.float32),
                        "original BF16-decoded source weight chunk",
                    )
                    source_blocks = source_mps.reshape(-1, BLOCK_SIZE)
                    sigma = moments_mps[:, 1:2]
                    safe_sigma = torch.where(sigma == 0, torch.ones_like(sigma), sigma)
                    normalized = (source_blocks - moments_mps[:, 0:1]) / safe_sigma
                    normalized = torch.where(sigma == 0, torch.zeros_like(normalized), normalized)
                    restore = torch.abs(normalized) >= float(threshold)
                    decoded_blocks = torch.where(restore, source_blocks, decoded_blocks)
            elif mode == "nearest":
                source_cpu = np.array(source_weight[first:last], dtype="<f4", copy=True)
                source_mps = require_mps(
                    torch.from_numpy(source_cpu).to(device="mps", dtype=torch.float32),
                    "original BF16-decoded source weight chunk",
                )
                source_blocks = source_mps.reshape(-1, BLOCK_SIZE)
                sigma = moments_mps[:, 1:2]
                safe_sigma = torch.where(sigma == 0, torch.ones_like(sigma), sigma)
                normalized = (source_blocks - moments_mps[:, 0:1]) / safe_sigma
                normalized = torch.where(sigma == 0, torch.zeros_like(normalized), normalized)
                magnitudes = torch.abs(normalized)
                center_indices = nearest_center_indices(magnitudes, boundaries_mps)
                selected = torch.index_select(
                    centers_mps, 0, center_indices.reshape(-1)
                ).reshape(-1, BLOCK_SIZE)
                selected = torch.where(normalized < 0, -selected, selected)
                decoded_blocks = moments_mps[:, 0:1] + moments_mps[:, 1:2] * selected
            else:
                raise ValueError(f"Unknown diagnostic mode: {mode}")
            decoded[first:last] = decoded_blocks.reshape(count, INPUT_DIM)
            del moments_mps, code_indices, decoded_blocks
    return decoded


def write_projection(
    output_dir: Path,
    name: str,
    precision: int,
    mode: str,
    threshold: float | None,
    fixture_paths: dict[str, Path],
    source_weight: np.memmap,
    input_mps: torch.Tensor,
    bias_mps: torch.Tensor,
    gaussian_lut: np.ndarray,
    arithmetic_lut: np.ndarray,
    row_chunk: int,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    table = arithmetic_lut if mode == "arithmetic" else gaussian_lut
    decoded_weight = decode_weight_on_mps(
        fixture_paths, source_weight, precision, mode, threshold, table, row_chunk
    )
    result = None
    try:
        result = require_mps(
            F.linear(input_mps, decoded_weight, bias_mps),
            f"{name} MPS F.linear projection",
        )
        if result.dtype != torch.float32 or tuple(result.shape) != (ROWS, OUTPUT_DIM):
            raise RuntimeError(f"Unexpected projection shape or dtype for {name}: {result.shape}")
        repeated = result.reshape(ROWS // SOURCE_ROWS, SOURCE_ROWS, OUTPUT_DIM)
        if not bool(torch.all(repeated == result[:SOURCE_ROWS].unsqueeze(0)).item()):
            raise RuntimeError(f"{name} F.linear outputs are not bit-identical across 1120 tiles")
        first_four = result[:SOURCE_ROWS].to("cpu").contiguous().numpy()
        first_four = np.asarray(first_four, dtype="<f4")
        path = output_dir / f"projection_{name}.f32"
        first_four.tofile(path)
        return file_record(path, "F32", [SOURCE_ROWS, OUTPUT_DIM]), result
    finally:
        del decoded_weight


def build(args: argparse.Namespace) -> Path:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for diagnostic dequantization and F.linear")
    if args.weight_row_chunk < 1:
        raise ValueError("--weight-row-chunk must be positive")

    fixture_root = args.fixture.expanduser().resolve()
    fixture_manifest, fixture_paths = load_fixture(fixture_root)
    codec_path = args.codec.expanduser().resolve()
    levels = parse_codec_levels(codec_path)
    source_weight = np.memmap(
        fixture_paths["source_weight"], dtype="<f4", mode="r", shape=(OUTPUT_DIM, INPUT_DIM)
    )
    source_input = np.fromfile(fixture_paths["input"], dtype="<f4").reshape(SOURCE_ROWS, INPUT_DIM)
    input_source_mps = require_mps(
        torch.from_numpy(source_input.copy()).to(device="mps", dtype=torch.float32),
        "four captured projection input rows",
    )
    row_indices = torch.arange(ROWS, device="mps", dtype=torch.int64) % SOURCE_ROWS
    input_mps = require_mps(
        torch.index_select(input_source_mps, 0, row_indices),
        "periodically tiled MPS projection input",
    )
    bias_record = fixture_manifest["source_tensor"]["bias_f32_file"]
    bias = np.fromfile(fixture_paths["bias"], dtype="<f4")
    if bias.shape != (OUTPUT_DIM,):
        raise RuntimeError(f"Unexpected source bias shape: {bias.shape}")
    bias_mps = require_mps(
        torch.from_numpy(bias).to(device="mps", dtype=torch.float32),
        "source projection bias",
    )

    output_path = args.output.expanduser().resolve()
    if os.path.lexists(output_path):
        raise FileExistsError(f"Immutable diagnostics output already exists: {output_path}")
    if output_path == fixture_root or fixture_root in output_path.parents or output_path in fixture_root.parents:
        raise ValueError("Diagnostics output must not overlap the immutable source fixture")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=output_path.parent))
    try:
        gaussian_luts: dict[int, np.ndarray] = {}
        arithmetic_luts: dict[int, np.ndarray] = {}
        gaussian_lut_records: dict[int, dict[str, Any]] = {}
        arithmetic_lut_records: dict[int, dict[str, Any]] = {}
        scalar_theory: dict[str, Any] = {}
        for precision in (4, 5):
            source_record = fixture_manifest["variants"][f"qbit_p{precision}"]["lut"]
            gaussian_path = fixture_paths[f"lut_p{precision}"]
            gaussian_lut = np.fromfile(gaussian_path, dtype="<f4")
            if gaussian_lut.shape != (256,) or not np.all(np.isfinite(gaussian_lut)):
                raise RuntimeError(f"Invalid pinned p{precision} Gaussian LUT")
            if not np.array_equal(gaussian_lut[:128], -gaussian_lut[255:127:-1]):
                raise RuntimeError(f"p{precision} Gaussian LUT is not reflection-symmetric")
            gaussian_luts[precision] = gaussian_lut
            arithmetic_lut = make_arithmetic_lut(levels, precision)
            arithmetic_luts[precision] = arithmetic_lut
            gaussian_copy = stage / f"lut_p{precision}_gaussian.f32"
            arithmetic_path = stage / f"lut_p{precision}_arithmetic.f32"
            gaussian_lut.astype("<f4", copy=False).tofile(gaussian_copy)
            arithmetic_lut.tofile(arithmetic_path)
            gaussian_lut_records[precision] = file_record(gaussian_copy, "F32", [256])
            arithmetic_lut_records[precision] = file_record(arithmetic_path, "F32", [256])
            scalar_theory[f"p{precision}"] = normal_theory(
                levels, gaussian_lut, precision
            )

        arms: list[dict[str, Any]] = []
        arm_specs = []
        for precision in (4, 5):
            arm_specs.extend(
                [
                    (f"qbit_p{precision}_arithmetic", precision, "arithmetic", None),
                    (f"qbit_p{precision}_tail_z3", precision, "tail_restore", 3.0),
                    (f"qbit_p{precision}_tail_z4", precision, "tail_restore", 4.0),
                    (f"qbit_p{precision}_nearest", precision, "nearest", None),
                ]
            )
        for name, precision, mode, threshold in arm_specs:
            print(f"producing {name}", flush=True)
            projection_record, result = write_projection(
                stage,
                name,
                precision,
                mode,
                threshold,
                fixture_paths,
                source_weight,
                input_mps,
                bias_mps,
                gaussian_luts[precision],
                arithmetic_luts[precision],
                args.weight_row_chunk,
            )
            del result
            lut_record = (
                arithmetic_lut_records[precision]
                if mode == "arithmetic"
                else gaussian_lut_records[precision]
            )
            arms.append(
                {
                    "name": name,
                    "precision": precision,
                    "mode": mode,
                    "threshold": threshold,
                    "projection": projection_record,
                    "lut": lut_record,
                }
            )
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

        source_hashes = {
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "capture_manifest_sha256": fixture_manifest["input_provenance"]["manifest_sha256"],
            "checkpoint_weight_bf16_sha256": fixture_manifest["source_tensor"][
                "weight_file"
            ]["sha256"],
            "checkpoint_bias_bf16_sha256": fixture_manifest["source_tensor"][
                "bias_bf16_file"
            ]["sha256"],
            "input_f32_sha256": fixture_manifest["input"]["sha256"],
            "decoded_bf16_source_f32_sha256": fixture_manifest["variants"]["bf16"][
                "decoded_weight"
            ]["sha256"],
            "bias_f32_sha256": bias_record["sha256"],
            "qbit_p4_sha256": fixture_manifest["variants"]["qbit_p4"]["packed_weight"][
                "sha256"
            ],
            "qbit_p5_sha256": fixture_manifest["variants"]["qbit_p5"]["packed_weight"][
                "sha256"
            ],
            "gaussian_lut_p4_sha256": fixture_manifest["variants"]["qbit_p4"]["lut"][
                "sha256"
            ],
            "gaussian_lut_p5_sha256": fixture_manifest["variants"]["qbit_p5"]["lut"][
                "sha256"
            ],
            "codec_source_sha256": CODEC_SHA256,
            "reference_oracle_script_sha256": fixture_manifest["source_hashes"][
                "script_sha256"
            ],
            "diagnostics_script_sha256": sha256_file(Path(__file__).resolve()),
        }
        manifest = {
            "schema": "wan_quantized_projection_diagnostics_v1",
            "created_by": Path(__file__).name,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_fixture_manifest_path": str(fixture_root / "manifest.json"),
            "source_hashes": source_hashes,
            "geometry": {
                "rows": ROWS,
                "source_rows": SOURCE_ROWS,
                "input_dim": INPUT_DIM,
                "output_dim": OUTPUT_DIM,
                "input_scope": fixture_manifest["input_provenance"],
                "input_operation": "MPS index_select with row_indices = arange(4480) % 4",
                "projection_rows_saved": SOURCE_ROWS,
                "all_rows_repeat_bitexact": True,
            },
            "oracle": {
                "device": "MPS",
                "fallback_disabled": True,
                "projection": "torch.nn.functional.linear(input, decoded_weight, bias)",
                "decode_weight_row_chunk": args.weight_row_chunk,
                "numpy_version": np.__version__,
                "torch_version": torch.__version__,
                "python_version": sys.version,
                "no_full_wan_model_loaded": True,
                "no_new_activation_capture": True,
            },
            "operation_order": {
                "qbit_wire_decode": "repository transposed MSB planes -> raw code indices",
                "gaussian_or_arithmetic_weight": "mean + sigma * LUT[raw_code] on MPS",
                "tail_restore": "z=(source-mean)/sigma on MPS; replace where abs(z)>=threshold",
                "nearest": "same block mean/sigma; normalize source on MPS; lower-bound search over Float32 midpoint boundaries; mean+sigma*selected_signed_centroid",
                "projection": "F.linear over all 4480 tiled input rows on MPS",
                "export": "verify all 1120 four-row groups are bit-identical, then save first four rows only",
            },
            "diagnostic_interpretation": {
                "arithmetic": "uses arithmetic mean of retained positive Lloyd-Max levels per prefix, reflected exactly in Float32; this is a control, not a legacy-codec reconstruction claim",
                "tail_restore": "restores original decoded BF16-derived F32 weights for standardized magnitude at or above the named threshold",
                "nearest": "assigns standardized source weights to the nearest centroid from the existing Gaussian prefix book; nested prefix coding is intentionally discarded",
                "theory_scope": "scalar standard-normal distortion only; it does not assert that Wan weights or activations are Gaussian",
            },
            "scalar_standard_normal_theory": scalar_theory,
            "arms": arms,
        }
        manifest_path = stage / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.lexists(output_path):
            raise FileExistsError(f"Immutable diagnostics output appeared during run: {output_path}")
        stage.rename(output_path)
        return output_path
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--codec", type=Path, default=DEFAULT_CODEC)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weight-row-chunk",
        type=int,
        default=16,
        help="decoded-weight rows staged per MPS upload (the full F.linear geometry is unchanged)",
    )
    args = parser.parse_args()
    print(build(args), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
