#!/usr/bin/env python3
"""Prepare MPS oracle fixtures for bounded adaptive Wan FFN weight selection.

The script consumes the pinned four-row Wan projection fixture and emits one
block-tier vector plus one projection reference per policy. It does not load a
Wan model or produce a compact mixed-weight container. QBit reconstruction,
per-block error reduction, tier selection, and F.linear projection run on MPS
with PyTorch fallback disabled. Projections use all 4480 periodically tiled
rows; only the first four output rows are saved after an exact repetition
check.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
import torch.nn.functional as F

import wan_quantized_projection_diagnostics as diagnostics


SOURCE_MANIFEST_SHA256 = diagnostics.SOURCE_MANIFEST_SHA256
SOURCE_ROWS = diagnostics.SOURCE_ROWS
ROWS = diagnostics.ROWS
INPUT_DIM = diagnostics.INPUT_DIM
OUTPUT_DIM = diagnostics.OUTPUT_DIM
BLOCK_SIZE = diagnostics.BLOCK_SIZE
BLOCKS_PER_ROW = INPUT_DIM // BLOCK_SIZE
BLOCK_COUNT = OUTPUT_DIM * BLOCKS_PER_ROW

P4_PAYLOAD_BYTES = 136
P5_PAYLOAD_BYTES = 168
FALLBACK_BF16_PAYLOAD_BYTES = BLOCK_SIZE * 2
FORECAST_METADATA_BYTES = 4

POLICIES = (
    ("adaptive_p4_020_p5_010", np.float32(0.20), np.float32(0.10)),
    ("adaptive_p4_050_p5_050", np.float32(0.50), np.float32(0.50)),
    ("adaptive_p4_100_p5_050", np.float32(1.00), np.float32(0.50)),
)


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
        "sha256": diagnostics.sha256_file(path),
    }


def block_sigma_on_mps(payload_path: Path) -> torch.Tensor:
    """Read the p4 F32 sigma headers; use them for both precision thresholds."""
    payload = np.memmap(
        payload_path,
        dtype=np.uint8,
        mode="r",
        shape=(OUTPUT_DIM, BLOCKS_PER_ROW, P4_PAYLOAD_BYTES),
    )
    sigma_cpu = np.ascontiguousarray(payload[:, :, 4:8]).view("<f4").reshape(
        OUTPUT_DIM, BLOCKS_PER_ROW
    )
    sigma = require_mps(
        torch.from_numpy(sigma_cpu.copy()).to(device="mps", dtype=torch.float32),
        "original QBit p4 block sigma",
    )
    if not bool(torch.all(torch.isfinite(sigma) & (sigma >= 0)).item()):
        raise RuntimeError("QBit p4 sigma headers must be finite and non-negative")
    return sigma


def max_block_residuals_on_mps(
    source_weight: np.memmap,
    decoded_p4: torch.Tensor,
    decoded_p5: torch.Tensor,
    row_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute max absolute residual per 256-value block on MPS."""
    max_p4 = require_mps(
        torch.empty((OUTPUT_DIM, BLOCKS_PER_ROW), device="mps", dtype=torch.float32),
        "p4 maximum block residuals",
    )
    max_p5 = require_mps(
        torch.empty((OUTPUT_DIM, BLOCKS_PER_ROW), device="mps", dtype=torch.float32),
        "p5 maximum block residuals",
    )
    for first in range(0, OUTPUT_DIM, row_chunk):
        last = min(OUTPUT_DIM, first + row_chunk)
        count = last - first
        source_cpu = np.array(source_weight[first:last], dtype="<f4", copy=True)
        source_mps = require_mps(
            torch.from_numpy(source_cpu).to(device="mps", dtype=torch.float32),
            "BF16-derived F32 source weight chunk",
        )
        source_blocks = source_mps.reshape(count, BLOCKS_PER_ROW, BLOCK_SIZE)
        p4_blocks = decoded_p4[first:last].reshape(count, BLOCKS_PER_ROW, BLOCK_SIZE)
        p5_blocks = decoded_p5[first:last].reshape(count, BLOCKS_PER_ROW, BLOCK_SIZE)
        max_p4[first:last] = torch.amax(torch.abs(source_blocks - p4_blocks), dim=2)
        max_p5[first:last] = torch.amax(torch.abs(source_blocks - p5_blocks), dim=2)
        del source_cpu, source_mps, source_blocks, p4_blocks, p5_blocks
    return max_p4, max_p5


def normalized_residual_on_mps(
    max_error: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Return max_error/sigma with the adaptive probe's zero-sigma rule."""
    require_mps(max_error, "maximum residual")
    require_mps(sigma, "block sigma")
    safe_sigma = torch.where(sigma == 0, torch.ones_like(sigma), sigma)
    ratio = max_error / safe_sigma
    zero_sigma_ratio = torch.where(
        max_error == 0,
        torch.zeros_like(max_error),
        torch.full_like(max_error, float("inf")),
    )
    return require_mps(
        torch.where(sigma == 0, zero_sigma_ratio, ratio),
        "normalized maximum residual",
    )


def select_tiers_on_mps(
    ratio_p4: torch.Tensor,
    ratio_p5: torch.Tensor,
    p4_threshold: torch.Tensor,
    p5_threshold: torch.Tensor,
) -> torch.Tensor:
    """Prefer accepted p4, then p5, and otherwise the BF16 fallback tier."""
    require_mps(ratio_p4, "p4 normalized residual")
    require_mps(ratio_p5, "p5 normalized residual")
    require_mps(p4_threshold, "p4 threshold")
    require_mps(p5_threshold, "p5 threshold")
    selected = torch.where(
        ratio_p4 <= p4_threshold,
        torch.full_like(ratio_p4, 4, dtype=torch.int32),
        torch.where(
            ratio_p5 <= p5_threshold,
            torch.full_like(ratio_p5, 5, dtype=torch.int32),
            torch.full_like(ratio_p5, 16, dtype=torch.int32),
        ),
    )
    return require_mps(selected, "selected adaptive tier")


def selector_boundary_controls(
    p4_value: np.float32, p5_value: np.float32
) -> dict[str, Any]:
    """Exercise below/at/above, p4 precedence, and zero-sigma selection on MPS."""
    p4_threshold = require_mps(
        torch.tensor(float(p4_value), device="mps", dtype=torch.float32),
        "synthetic p4 threshold",
    )
    p5_threshold = require_mps(
        torch.tensor(float(p5_value), device="mps", dtype=torch.float32),
        "synthetic p5 threshold",
    )
    # The first five rows use sigma=1, so their F32 residual equals the test
    # ratio. The last two exercise exact and non-exact zero-sigma blocks.
    p4_errors = np.asarray(
        [p4_value * np.float32(0.5), p4_value, p4_value * np.float32(1.1),
         p4_value * np.float32(1.1), p4_value * np.float32(1.1), 0.0, 1.0],
        dtype="<f4",
    )
    p5_errors = np.asarray(
        [p5_value * np.float32(0.5), 0.0, p5_value * np.float32(0.5),
         p5_value, p5_value * np.float32(1.1), 0.0, 1.0],
        dtype="<f4",
    )
    sigma_values = np.asarray([1, 1, 1, 1, 1, 0, 0], dtype="<f4")
    expected = torch.tensor([4, 4, 5, 5, 16, 4, 16], device="mps", dtype=torch.int32)
    p4_error_mps = require_mps(
        torch.from_numpy(p4_errors).to(device="mps", dtype=torch.float32),
        "synthetic p4 residuals",
    )
    p5_error_mps = require_mps(
        torch.from_numpy(p5_errors).to(device="mps", dtype=torch.float32),
        "synthetic p5 residuals",
    )
    sigma_mps = require_mps(
        torch.from_numpy(sigma_values).to(device="mps", dtype=torch.float32),
        "synthetic block sigma",
    )
    ratio_p4 = normalized_residual_on_mps(p4_error_mps, sigma_mps)
    ratio_p5 = normalized_residual_on_mps(p5_error_mps, sigma_mps)
    actual = select_tiers_on_mps(ratio_p4, ratio_p5, p4_threshold, p5_threshold)
    if not torch.equal(actual, expected):
        raise RuntimeError(
            "MPS adaptive tier selector failed below/at/above threshold or zero-sigma controls"
        )
    return {
        "passed": True,
        "cases": [
            "p4 below and p5 below selects p4",
            "p4 exactly at threshold selects p4",
            "p4 above and p5 below selects p5",
            "p4 above and p5 exactly at threshold selects p5",
            "both above threshold selects BF16 fallback",
            "zero sigma with zero residual selects p4",
            "zero sigma with nonzero residual rejects both quantized tiers",
        ],
        "expected_tiers": [4, 4, 5, 5, 16, 4, 16],
    }


def threshold_tensor(value: np.float32, label: str) -> torch.Tensor:
    return require_mps(
        torch.tensor(float(value), device="mps", dtype=torch.float32), label
    )


def f32_bits(value: np.float32) -> str:
    bits = int(np.asarray(value, dtype="<f4").view("<u4").item())
    return f"0x{bits:08x}"


def build_selected_weight_on_mps(
    source_weight: np.memmap,
    decoded_p4: torch.Tensor,
    decoded_p5: torch.Tensor,
    tiers: torch.Tensor,
    row_chunk: int,
) -> torch.Tensor:
    selected_weight = require_mps(
        torch.empty((OUTPUT_DIM, INPUT_DIM), device="mps", dtype=torch.float32),
        "selected adaptive F32 weight matrix",
    )
    tier_rows = tiers.reshape(OUTPUT_DIM, BLOCKS_PER_ROW)
    for first in range(0, OUTPUT_DIM, row_chunk):
        last = min(OUTPUT_DIM, first + row_chunk)
        count = last - first
        tier_chunk = tier_rows[first:last]
        p4_mask = tier_chunk == 4
        p5_mask = tier_chunk == 5
        p4_mask = p4_mask.unsqueeze(-1).expand(-1, -1, BLOCK_SIZE).reshape(
            count, INPUT_DIM
        )
        p5_mask = p5_mask.unsqueeze(-1).expand(-1, -1, BLOCK_SIZE).reshape(
            count, INPUT_DIM
        )
        p4_chunk = decoded_p4[first:last]
        p5_chunk = decoded_p5[first:last]
        source_cpu = np.array(source_weight[first:last], dtype="<f4", copy=True)
        source_mps = require_mps(
            torch.from_numpy(source_cpu).to(device="mps", dtype=torch.float32),
            "BF16-derived F32 fallback weight chunk",
        )
        selected_weight[first:last] = torch.where(
            p4_mask,
            p4_chunk,
            torch.where(p5_mask, p5_chunk, source_mps),
        )
        del (
            tier_chunk,
            p4_mask,
            p5_mask,
            p4_chunk,
            p5_chunk,
            source_cpu,
            source_mps,
        )
    return selected_weight


def project_and_export(
    stage: Path,
    policy_name: str,
    selected_weight: torch.Tensor,
    input_mps: torch.Tensor,
    bias_mps: torch.Tensor,
) -> dict[str, Any]:
    result = require_mps(
        F.linear(input_mps, selected_weight, bias_mps),
        f"{policy_name} MPS F.linear projection",
    )
    if result.dtype != torch.float32 or tuple(result.shape) != (ROWS, OUTPUT_DIM):
        raise RuntimeError(f"Unexpected projection shape or dtype for {policy_name}: {result.shape}")
    repeated = result.reshape(ROWS // SOURCE_ROWS, SOURCE_ROWS, OUTPUT_DIM)
    if not bool(torch.all(repeated == result[:SOURCE_ROWS].unsqueeze(0)).item()):
        raise RuntimeError(f"{policy_name} projection is not bit-identical across all row tiles")
    first_four = np.asarray(
        result[:SOURCE_ROWS].to("cpu").contiguous().numpy(), dtype="<f4"
    )
    path = stage / f"projection_{policy_name}.f32"
    first_four.tofile(path)
    return file_record(path, "F32", [SOURCE_ROWS, OUTPUT_DIM])


def write_tier_file(stage: Path, policy_name: str, tiers: torch.Tensor) -> dict[str, Any]:
    tiers_cpu = tiers.to("cpu").contiguous().numpy().astype(np.uint8, copy=False)
    path = stage / f"tiers_{policy_name}.u8"
    tiers_cpu.tofile(path)
    return file_record(path, "U8", [BLOCK_COUNT])


def source_hashes(fixture_manifest: dict[str, Any]) -> dict[str, str]:
    variants = fixture_manifest["variants"]
    tensor = fixture_manifest["source_tensor"]
    return {
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "capture_manifest_sha256": fixture_manifest["input_provenance"][
            "manifest_sha256"
        ],
        "checkpoint_weight_bf16_sha256": tensor["weight_file"]["sha256"],
        "checkpoint_bias_bf16_sha256": tensor["bias_bf16_file"]["sha256"],
        "input_f32_sha256": fixture_manifest["input"]["sha256"],
        "decoded_bf16_source_f32_sha256": variants["bf16"]["decoded_weight"][
            "sha256"
        ],
        "bias_f32_sha256": tensor["bias_f32_file"]["sha256"],
        "qbit_p4_sha256": variants["qbit_p4"]["packed_weight"]["sha256"],
        "qbit_p5_sha256": variants["qbit_p5"]["packed_weight"]["sha256"],
        "gaussian_lut_p4_sha256": variants["qbit_p4"]["lut"]["sha256"],
        "gaussian_lut_p5_sha256": variants["qbit_p5"]["lut"]["sha256"],
        "codec_source_sha256": diagnostics.CODEC_SHA256,
        "reference_script_sha256": fixture_manifest["source_hashes"][
            "script_sha256"
        ],
        "diagnostic_helper_script_sha256": diagnostics.sha256_file(
            Path(diagnostics.__file__).resolve()
        ),
        "adaptive_script_sha256": diagnostics.sha256_file(
            Path(__file__).resolve()
        ),
    }


def build(args: argparse.Namespace) -> Path:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for adaptive decode, selection, and projection")
    if args.weight_row_chunk < 1:
        raise ValueError("--weight-row-chunk must be positive")

    fixture_root = args.fixture.expanduser().resolve()
    fixture_manifest, fixture_paths = diagnostics.load_fixture(fixture_root)
    diagnostics.parse_codec_levels(diagnostics.DEFAULT_CODEC)
    source_weight = np.memmap(
        fixture_paths["source_weight"],
        dtype="<f4",
        mode="r",
        shape=(OUTPUT_DIM, INPUT_DIM),
    )

    lut_p4 = np.fromfile(fixture_paths["lut_p4"], dtype="<f4")
    lut_p5 = np.fromfile(fixture_paths["lut_p5"], dtype="<f4")
    if lut_p4.shape != (256,) or lut_p5.shape != (256,):
        raise RuntimeError("Pinned Gaussian LUTs must each contain 256 F32 entries")
    if not np.all(np.isfinite(lut_p4)) or not np.all(np.isfinite(lut_p5)):
        raise RuntimeError("Pinned Gaussian LUTs must contain finite values")

    print("decoding qbit_p4 on MPS", flush=True)
    decoded_p4 = diagnostics.decode_weight_on_mps(
        fixture_paths, source_weight, 4, "arithmetic", None, lut_p4, args.weight_row_chunk
    )
    print("decoding qbit_p5 on MPS", flush=True)
    decoded_p5 = diagnostics.decode_weight_on_mps(
        fixture_paths, source_weight, 5, "arithmetic", None, lut_p5, args.weight_row_chunk
    )
    del lut_p4, lut_p5

    sigma = block_sigma_on_mps(fixture_paths["qbit_p4"])
    max_p4, max_p5 = max_block_residuals_on_mps(
        source_weight, decoded_p4, decoded_p5, args.weight_row_chunk
    )
    ratio_p4 = normalized_residual_on_mps(max_p4, sigma)
    ratio_p5 = normalized_residual_on_mps(max_p5, sigma)
    del max_p4, max_p5

    source_input = np.fromfile(fixture_paths["input"], dtype="<f4").reshape(
        SOURCE_ROWS, INPUT_DIM
    )
    input_source_mps = require_mps(
        torch.from_numpy(source_input.copy()).to(device="mps", dtype=torch.float32),
        "four captured projection input rows",
    )
    row_indices = torch.arange(ROWS, device="mps", dtype=torch.int64) % SOURCE_ROWS
    input_mps = require_mps(
        torch.index_select(input_source_mps, 0, row_indices),
        "periodically tiled MPS projection input",
    )
    bias = np.fromfile(fixture_paths["bias"], dtype="<f4")
    if bias.shape != (OUTPUT_DIM,):
        raise RuntimeError(f"Unexpected bias shape: {bias.shape}")
    bias_mps = require_mps(
        torch.from_numpy(bias).to(device="mps", dtype=torch.float32),
        "source projection bias",
    )

    output_path = args.output.expanduser().resolve()
    if os.path.lexists(output_path):
        raise FileExistsError(f"Immutable adaptive reference output already exists: {output_path}")
    if (
        output_path == fixture_root
        or fixture_root in output_path.parents
        or output_path in fixture_root.parents
    ):
        raise ValueError("Adaptive reference output must not overlap the pinned source fixture")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=output_path.parent)
    )
    try:
        policy_records: list[dict[str, Any]] = []
        selector_controls: dict[str, Any] = {}
        for name, p4_threshold_value, p5_threshold_value in POLICIES:
            print(f"selecting {name} on MPS", flush=True)
            controls = selector_boundary_controls(p4_threshold_value, p5_threshold_value)
            selector_controls[name] = controls
            p4_threshold = threshold_tensor(p4_threshold_value, f"{name} p4 threshold")
            p5_threshold = threshold_tensor(p5_threshold_value, f"{name} p5 threshold")
            tiers = select_tiers_on_mps(
                ratio_p4, ratio_p5, p4_threshold, p5_threshold
            )

            counts = {
                str(tier): int(torch.sum(tiers == tier).item()) for tier in (4, 5, 16)
            }
            if sum(counts.values()) != BLOCK_COUNT:
                raise RuntimeError(f"{name} tier counts do not cover all packed blocks")
            tier_record = write_tier_file(stage, name, tiers)
            selected_weight = build_selected_weight_on_mps(
                source_weight,
                decoded_p4,
                decoded_p5,
                tiers,
                args.weight_row_chunk,
            )
            projection_record = project_and_export(
                stage, name, selected_weight, input_mps, bias_mps
            )
            del selected_weight, tiers
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

            payload_bytes = (
                counts["4"] * (P4_PAYLOAD_BYTES + FORECAST_METADATA_BYTES)
                + counts["5"] * (P5_PAYLOAD_BYTES + FORECAST_METADATA_BYTES)
                + counts["16"]
                * (FALLBACK_BF16_PAYLOAD_BYTES + FORECAST_METADATA_BYTES)
            )
            policy_records.append(
                {
                    "name": name,
                    "p4_max_error": float(p4_threshold_value),
                    "p5_max_error": float(p5_threshold_value),
                    "threshold_f32_bits": {
                        "p4": f32_bits(p4_threshold_value),
                        "p5": f32_bits(p5_threshold_value),
                    },
                    "projection": projection_record,
                    "tiers": tier_record,
                    "selected_block_counts": counts,
                    "selector_boundary_controls": controls,
                    "hypothetical_storage_forecast": {
                        "bytes": payload_bytes,
                        "metadata_bytes_per_block": FORECAST_METADATA_BYTES,
                        "payload_bytes_per_block": {
                            "p4": P4_PAYLOAD_BYTES,
                            "p5": P5_PAYLOAD_BYTES,
                            "bf16_fallback": FALLBACK_BF16_PAYLOAD_BYTES,
                        },
                        "tier_validation_file_bytes_excluded": tier_record["bytes"],
                        "is_actual_mixed_container": False,
                        "does_not_claim_packed_layout_or_quality": True,
                    },
                }
            )
        source_records = {
            key: {
                "path": value.name,
                "bytes": value.stat().st_size,
                "sha256": diagnostics.sha256_file(value),
            }
            for key, value in fixture_paths.items()
        }
        manifest = {
            "schema": "wan_qbit_adaptive_weight_reference_v1",
            "created_by": Path(__file__).name,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_fixture_manifest_path": str(fixture_root / "manifest.json"),
            "source_hashes": source_hashes(fixture_manifest),
            "source_files": source_records,
            "geometry": {
                "rows": ROWS,
                "source_rows": SOURCE_ROWS,
                "input_dim": INPUT_DIM,
                "output_dim": OUTPUT_DIM,
                "block_size": BLOCK_SIZE,
                "blocks_per_row": BLOCKS_PER_ROW,
                "block_count": BLOCK_COUNT,
                "input_scope": fixture_manifest["input_provenance"],
                "input_operation": "MPS index_select with row_indices = arange(4480) % 4",
                "projection_rows_saved": SOURCE_ROWS,
                "all_rows_repeat_bitexact": True,
            },
            "oracle": {
                "device": "MPS",
                "fallback_disabled": True,
                "decode": "repository QBit transposed bitplanes and pinned Gaussian LUT on MPS",
                "selection": "max(abs(decoded-original))/p4_sigma per 256-value block on MPS; prefer p4, then p5, else BF16-derived F32 source",
                "projection": "torch.nn.functional.linear(input, decoded_weight, bias)",
                "projection_scope": "all 4480 tiled input rows on MPS; first four outputs saved after exact repetition check",
                "weight_row_chunk": args.weight_row_chunk,
                "numpy_version": np.__version__,
                "torch_version": torch.__version__,
                "python_version": sys.version,
                "no_full_wan_model_loaded": True,
                "no_new_activation_capture": True,
                "no_mixed_packed_container_emitted": True,
            },
            "selection_semantics": {
                "p4_threshold": "max absolute residual divided by p4 wire sigma, in Float32",
                "p5_threshold": "max absolute residual divided by the same p4 wire sigma, in Float32",
                "zero_sigma": "ratio is 0 for exact zero residual and positive infinity otherwise",
                "tier_precedence": [4, 5, 16],
                "tier_16_source": "original decoded BF16-derived F32 weights",
                "tier_file_purpose": "U8 validation oracle only; not an emitted mixed-weight layout",
            },
            "hypothetical_storage_forecast": {
                "metadata_bytes_per_block": FORECAST_METADATA_BYTES,
                "payload_bytes_per_block": {
                    "p4": P4_PAYLOAD_BYTES,
                    "p5": P5_PAYLOAD_BYTES,
                    "bf16_fallback": FALLBACK_BF16_PAYLOAD_BYTES,
                },
                "scope": "payload-plus-four-byte-metadata estimate only; excludes the U8 validation tier file and makes no claim of a concrete container layout",
            },
            "selector_boundary_controls": selector_controls,
            "policies": policy_records,
        }
        manifest_path = stage / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.lexists(output_path):
            raise FileExistsError(f"Immutable adaptive reference output appeared during run: {output_path}")
        stage.rename(output_path)
        return output_path
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=diagnostics.DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weight-row-chunk",
        type=int,
        default=16,
        help="decoded-weight rows staged per MPS upload; the full F.linear geometry is unchanged",
    )
    args = parser.parse_args()
    print(build(args), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
