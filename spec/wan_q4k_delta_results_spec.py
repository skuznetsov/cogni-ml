#!/usr/bin/env python3
"""Adversarial CPU-only tests for the Q4_K delta receipt analyzer."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "examples/support/wan_q4k_delta_results.py"
SPEC = importlib.util.spec_from_file_location("wan_q4k_delta_results", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(results)

ROWS, DIN, DOUT = 4, results.INPUT_DIM, results.OUTPUT_DIM
BLOCK_ARMS = tuple(f"{arm}_block" for arm in results.ARMS)
EXECUTION_ARMS = results.ARMS + BLOCK_ARMS
MANIFEST_SCHEMA = "wan_q4k_delta_reference_v1"
ZERO_METRIC = lambda count, **changes: {
    "compared_values": count,
    "bit_differences": 0,
    "tolerance_violations": 0,
    "nonfinite": 0,
    "max_abs": 0.0,
    "max_ratio": 0.0,
    "rmse": 0.0,
    "relative_l2": 0.0,
    "cosine": 1.0,
    **changes,
}


def timing_records(rounds: int = 16, reverse_first: bool = False,
                   manifest_sha256: str = "a" * 64,
                   block_validation_comparison: bool = False) -> list[dict]:
    records = []
    arms = EXECUTION_ARMS if block_validation_comparison else results.ARMS
    for mode in results.MODES:
        for round_id in (-2, -1, *range(rounds)):
            if block_validation_comparison:
                rotation = max(round_id, 0) % len(results.ARMS)
                base_order = list(results.ARMS[rotation:]) + list(results.ARMS[:rotation])
                order = [item for arm in base_order for item in (arm, f"{arm}_block")]
                if round_id >= 0 and (round_id // 4) % 2 == 1:
                    order = [item for index in range(0, len(order), 2)
                             for item in (order[index + 1], order[index])]
            else:
                rotation = max(round_id, 0) % len(arms)
                order = list(arms[rotation:]) + list(arms[:rotation])
            reverse = ((round_id == -1) if round_id < 0 else round_id >= rounds // 2)
            if reverse != reverse_first:
                order.reverse()
            for position, arm in enumerate(order):
                records.append({
                    "schema": "wan_q4k_delta_timing_v1",
                    "mode": mode,
                    "manifest_sha256": manifest_sha256,
                    "rows": ROWS,
                    "terminal_waits": 1,
                    "inter_operator_host_waits": 0,
                    "activation_readbacks": 0,
                    "encode_metal_buffer_allocations": 0,
                    "round": round_id,
                    "position": position,
                    "warmup": round_id < 0,
                    "variant": arm,
                    "gpu_seconds": 0.001 + position * 0.00001,
                    "wall_seconds": 0.002 + position * 0.00001,
                    "encode_seconds": 0.0001 + position * 0.000001,
                })
    return records


def fused_timing_records(rounds: int = 16, reverse_first: bool = False,
                         manifest_sha256: str = "a" * 64) -> list[dict]:
    """Create the separately scheduled dense and fused timing populations."""
    records = []
    for mode in results.FUSED_MODES:
        for round_id in (-2, -1, *range(rounds)):
            if mode == "dense_mode":
                if round_id < 0:
                    order = (("dense_native", "dense_mps") if round_id == -2 else
                             ("dense_mps", "dense_native"))
                    if reverse_first:
                        order = tuple(reversed(order))
                else:
                    abba = (round_id < rounds // 2) != reverse_first
                    order = (("dense_native", "dense_mps", "dense_mps", "dense_native")
                             if abba else
                             ("dense_mps", "dense_native", "dense_native", "dense_mps"))
            else:
                rotation = max(round_id, 0) % len(results.ARMS)
                base_order = results.ARMS[rotation:] + results.ARMS[:rotation]
                order = [item for arm in base_order for item in (arm, f"{arm}_fused")]
                if round_id >= 0 and (round_id // len(results.ARMS)) % 2 == 1:
                    order = [item for index in range(0, len(order), 2)
                             for item in (order[index + 1], order[index])]
                reverse = ((round_id == -1) if round_id < 0 else round_id >= rounds // 2)
                if reverse != reverse_first:
                    order.reverse()
            for position, arm in enumerate(order):
                records.append({
                    "schema": "wan_q4k_delta_timing_v1",
                    "mode": mode,
                    "manifest_sha256": manifest_sha256,
                    "rows": ROWS,
                    "terminal_waits": 1,
                    "inter_operator_host_waits": 0,
                    "activation_readbacks": 0,
                    "encode_metal_buffer_allocations": 0,
                    "round": round_id,
                    "position": position,
                    "warmup": round_id < 0,
                    "variant": arm,
                    "gpu_seconds": 0.001 + position * 0.00001,
                    "wall_seconds": 0.002 + position * 0.00001,
                    "encode_seconds": 0.0001 + position * 0.000001,
                })
    return records


def gemm_timing_records(rounds: int = 16, reverse_first: bool = False,
                        manifest_sha256: str = "a" * 64) -> list[dict]:
    records = []
    for round_id in (-2, -1, *range(rounds)):
        order = results._gemm_order(round_id, rounds, reverse_first)
        for position, arm in enumerate(order):
            records.append({
                "schema": "wan_q4k_delta_timing_v1",
                "mode": "dense_gemm",
                "manifest_sha256": manifest_sha256,
                "rows": ROWS,
                "terminal_waits": 1,
                "inter_operator_host_waits": 0,
                "activation_readbacks": 0,
                "encode_metal_buffer_allocations": 0,
                "round": round_id,
                "position": position,
                "warmup": round_id < 0,
                "variant": arm,
                "gpu_seconds": 0.001 + position * 0.00001,
                "wall_seconds": 0.002 + position * 0.00001,
                "encode_seconds": 0.0001 + position * 0.000001,
            })
    return records


def q4k_kernel_timing_records(rounds: int = results.Q4K_KERNEL_ROUNDS,
                              reverse_first: bool = False,
                              manifest_sha256: str = "a" * 64) -> list[dict]:
    records = []
    for round_id in (-1, *range(rounds)):
        for format_name, format_position, arm, position in \
                results.q4k_kernel_round_order(round_id, reverse_first):
            records.append({
                "schema": "wan_q4k_delta_timing_v1",
                "mode": "q4k_kernel_ablation",
                "manifest_sha256": manifest_sha256,
                "rows": ROWS,
                "terminal_waits": 1,
                "inter_operator_host_waits": 0,
                "activation_readbacks": 0,
                "encode_metal_buffer_allocations": 0,
                "round": round_id,
                "position": position,
                "format_position": format_position,
                "warmup": round_id < 0,
                "format": format_name,
                "variant": arm,
                "gpu_seconds": 0.001 + format_position * 0.00001 + position * 0.000001,
                "wall_seconds": 0.002 + format_position * 0.00001 + position * 0.000001,
                "encode_seconds": 0.0001 + format_position * 0.000001 + position * 0.0000001,
            })
    return records


def valid_guard_events() -> list[dict]:
    physical = 64 * 1024**3
    sample = {
        "event": "sample",
        "pressure_free_pct": 80,
        "physical_bytes": physical,
        "headroom_bytes": 20 * 1024**3,
        "group_footprint_bytes": 1024**3,
        "swap_used_bytes": 1_000_000_000,
    }
    limits = {
        "event": "limits",
        "runtime_floor_pct": 15,
        "max_footprint_bytes": 4096 * 1024**2,
        "max_swap_growth_bytes": 64 * 1024**2,
        "timeout_sec": 300,
        "memory_accounting": "file-cache",
        "headroom_formula": "printed free + file-backed (includes speculative)",
        "startup_rule": "memory_pressure -Q >= 15% and selected headroom >= 15% physical + footprint budget",
    }
    return [limits, dict(sample), {"event": "started"}, dict(sample),
            {"event": "finished", "returncode": 0}]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def sparse_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.truncate(size)


def link_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)


def file_record(path: Path, root: Path, dtype: str, shape: list[int]) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "dtype": dtype,
        "shape": shape,
        "bytes": path.stat().st_size,
        "sha256": results.sha(path),
    }


def make_reference_fixture(base: Path, rows: int = ROWS) -> tuple[Path, dict[str, str]]:
    """Write a complete, sparse full-size fixture with internally consistent pins."""
    fixture = base / f"fixture-r{rows}"
    fixture.mkdir(parents=True)
    source_fixture = base.parent / "wan-quantized-projection-20261022" / "fixture-4480-v3"
    source_fixture.mkdir(parents=True)
    source_manifest = source_fixture / "manifest.json"
    write_json(source_manifest, {"schema": "wan_quantized_projection_reference_v1"})

    input_path = fixture / "input.f32"
    bias_path = fixture / "bias.f32"
    original_path = fixture / "original_weight.bf16"
    sparse_file(input_path, 4 * DIN * 4)
    sparse_file(bias_path, DOUT * 4)
    sparse_file(original_path, results.WEIGHT_COUNT * 2)

    baseline_decoded = fixture / "baseline/decoded_weight.f32"
    sparse_file(baseline_decoded, results.WEIGHT_COUNT * 4)
    baseline_projection = fixture / "baseline/expected_projection.f32"
    sparse_file(baseline_projection, rows * DOUT * 4)

    shared_payload = fixture / "variants/q4_k/payload.bin"
    sparse_file(shared_payload, results.BASE_PAYLOAD_BYTES)
    decoded_template = fixture / "variants/q4_k/decoded_weight.f32"
    sparse_file(decoded_template, results.WEIGHT_COUNT * 4)
    # Every weight output is zero in this synthetic fixture; hardlinks keep the
    # positive receipt small while still exercising real hashes and file sizes.
    if os.stat(decoded_template).st_ino != os.stat(baseline_decoded).st_ino:
        baseline_decoded.unlink()
        link_file(decoded_template, baseline_decoded)
    input_rec = file_record(input_path, fixture, "F32", [4, DIN])
    bias_rec = file_record(bias_path, fixture, "F32", [DOUT])
    original_rec = file_record(original_path, fixture, "BF16", [DOUT, DIN])
    baseline = {
        "decoded_weight": file_record(baseline_decoded, fixture, "F32", [DOUT, DIN]),
        "expected_projection": file_record(baseline_projection, fixture, "F32", [rows, DOUT]),
    }

    variants = {}
    all_records = [input_rec, bias_rec, original_rec,
                   baseline["decoded_weight"], baseline["expected_projection"]]
    payload_hash = results.sha(shared_payload)
    decoded_hash = results.sha(decoded_template)
    for arm in results.ARMS:
        patches = results.PATCHES_PER_BLOCK[arm]
        arm_dir = fixture / "variants" / arm
        payload_path = arm_dir / "payload.bin"
        decoded_path = arm_dir / "decoded_weight.f32"
        if arm != "q4_k":
            link_file(shared_payload, payload_path)
            link_file(decoded_template, decoded_path)
        sidecar_path = arm_dir / "sidecar.bin"
        sparse_file(sidecar_path, results.BLOCK_COUNT * patches * 3)
        descriptor_path = arm_dir / "descriptor.bin"
        descriptor_path.write_bytes(struct.pack(
            "<4I", results.DESCRIPTOR_MAGIC,
            0 if patches == 0 else 0x100 + patches, DOUT, DIN))
        projection_path = arm_dir / "expected_projection.f32"
        sparse_file(projection_path, rows * DOUT * 4)

        record_map = {
            "payload": file_record(payload_path, fixture, "Q4_K",
                                   [DOUT, DIN // results.BLOCK_VALUES, results.Q4_K_BLOCK_BYTES]),
            "sidecar": file_record(sidecar_path, fixture, "U8F16LE_3B",
                                   [results.BLOCK_COUNT, patches, 3]),
            "descriptor": file_record(descriptor_path, fixture, "WQB1Descriptor", [16]),
            "decoded_weight": file_record(decoded_path, fixture, "F32", [DOUT, DIN]),
            "expected_projection": file_record(projection_path, fixture, "F32", [rows, DOUT]),
        }
        all_records.extend(record_map.values())
        storage = results.expected_storage(arm)
        package = storage["package_bytes"]
        storage.update({
            "blocks": results.BLOCK_COUNT,
            "delta_record_bytes": 3,
            "logical_storage_bytes": package,
            "resident_storage_bytes": package,
            "logical_weight_bytes_bf16": results.WEIGHT_COUNT * 2,
            "decoded_f32_diagnostic_bytes": results.WEIGHT_COUNT * 4,
            "diagnostic_exports_counted_in_resident_storage": False,
            "resident_storage_basis": "serialized canonical base, fixed-stride sidecar, and 16-byte descriptor; no padding or index arrays",
        })
        variants[arm] = {
            "patches": patches,
            **record_map,
            "storage": storage,
            "cpu_quality": {
                "decode_device": "pinned source fixture" if patches == 0 else "MPS",
                "evaluation_device": "CPU",
                "weight_distortion": {
                    "weight_error": {
                        "elements": results.WEIGHT_COUNT,
                        "max_abs": 0.0,
                        "rmse": 0.0,
                        "relative_l2": 0.0,
                        "squared_l2": 0.0,
                    },
                    "activation_weighted_error": {
                        "fold0_train_rows": [0, 1],
                        "fold1_train_rows": [2, 3],
                        "fold0_weighted_squared_error": 0.0,
                        "fold1_weighted_squared_error": 0.0,
                    },
                },
            },
            "quality_vs_original_bf16": {
                "compared_values": rows * DOUT,
                "max_abs_vs_bf16": 0.0,
                "rmse_vs_bf16": 0.0,
                "relative_l2_vs_bf16": 0.0,
                "values_exceeding_reference_scale_tolerance_vs_bf16": 0,
                "note": "Distinct from the native same-format parity gate.",
            },
        }

    source_hash = results.sha(source_manifest)
    source_pins = {
        "manifest_sha256": source_hash,
        "checkpoint_weight_sha256": original_rec["sha256"],
        "input_sha256": input_rec["sha256"],
        "bias_f32_sha256": bias_rec["sha256"],
        "q4_k_sha256": payload_hash,
        "q4_decoded_sha256": decoded_hash,
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "geometry": {
            "rows": rows,
            "source_rows": 4,
            "input_dim": DIN,
            "output_dim": DOUT,
            "block_values": results.BLOCK_VALUES,
            "blocks_per_output_row": DIN // results.BLOCK_VALUES,
            "block_count": results.BLOCK_COUNT,
            "input_layout": "row_major periodic source-row tile",
        },
        "comparison": {
            "dtype": "F32",
            "atol": 2e-5,
            "rtol": 2e-5,
            "finite": True,
            "predicate": "finite(actual) && abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)",
        },
        "selection": {
            "score": "abs(F32(original BF16 weight) - F32(canonical Q4_K decoded base))",
            "scope": "independent contiguous 256-value Q4_K block",
            "quota_per_block": True,
            "rank_order": "descending absolute residual; equal residuals use ascending local position",
            "wire_order": "selected positions remain in descending residual rank order",
            "delta": "F16 round-to-nearest-ties-to-even of F32(original minus base)",
            "reconstruction": "separate F32(base + F32(F16 delta)) operation",
            "global_optimality_claim": False,
            "quality_scope": "the 4 unique seeded projection rows only; rows=4480 periodically tiles those same 4 rows",
        },
        "oracle": {
            "device": "MPS",
            "fallback_disabled": True,
            "selection_packing_reconstruction_device": "MPS",
        },
        "source_fixture": {
            "schema": "wan_quantized_projection_reference_v1",
            "real_video_derived": False,
            **source_pins,
        },
        "input": input_rec,
        "bias": bias_rec,
        "original_weight": original_rec,
        "baseline": baseline,
        "variants": variants,
    }
    manifest_path = fixture / "manifest.json"
    write_json(manifest_path, manifest)
    all_files = {str(fixture / record["path"]): record["sha256"] for record in all_records}
    all_files[str(manifest_path)] = results.sha(manifest_path)
    all_files[str(source_manifest)] = source_hash
    return fixture, {**all_files, "__source_manifest__": str(source_manifest),
                     "__source_dir__": str(source_fixture),
                     "__source_pins__": source_pins}


def synthetic_device_dense_fields(variant_config: dict, rows: int = ROWS,
                                 in_dim: int = DIN, out_dim: int = DOUT) -> dict:
    """Return source-bound metadata for the selected direct/tail dense paths."""
    return {
        "dense_static_threadgroup_memory_bytes":
            variant_config["dense_static_threadgroup_memory_bytes"],
        "gemm_candidate_dense_pipeline": variant_config["dense_pipeline"],
        "gemm_candidate_dense_static_threadgroup_memory_bytes":
            variant_config["dense_static_threadgroup_memory_bytes"],
        "gemm_candidate_dense_direct_pipeline": variant_config["dense_pipeline"],
        "gemm_candidate_dense_direct_tile_shape": variant_config["tile_shape"],
        "gemm_candidate_dense_direct_threadgroup_threads": variant_config["threads"],
        "gemm_candidate_dense_direct_static_threadgroup_memory_bytes":
            variant_config["dense_static_threadgroup_memory_bytes"],
        "gemm_candidate_dense_staged_tail_pipeline":
            variant_config["staged_tail_pipeline"],
        "gemm_candidate_dense_staged_tail_tile_shape": variant_config["tile_shape"],
        "gemm_candidate_dense_staged_tail_threadgroup_threads":
            variant_config["threads"],
        "gemm_candidate_dense_staged_tail_static_threadgroup_memory_bytes":
            variant_config["staged_tail_static_threadgroup_memory_bytes"],
        "gemm_dense_dispatch_partition": results.expected_dense_dispatch_partition(
            rows, in_dim, out_dim, variant_config["tile_shape"]),
    }


def synthetic_device_q4k_fields(variant_config: dict, rows: int = ROWS,
                               in_dim: int = DIN, out_dim: int = DOUT) -> dict:
    """Return source-bound metadata for packed direct and staged-tail paths."""
    return {
        "gemm_candidate_linear_direct_pipeline": variant_config["linear_pipeline"],
        "gemm_candidate_linear_direct_tile_shape": variant_config["tile_shape"],
        "gemm_candidate_linear_direct_threadgroup_threads": variant_config["threads"],
        "gemm_candidate_linear_direct_static_threadgroup_memory_bytes":
            variant_config["linear_static_threadgroup_memory_bytes"],
        "gemm_candidate_linear_staged_tail_pipeline":
            variant_config["linear_staged_tail_pipeline"],
        "gemm_candidate_linear_staged_tail_tile_shape": variant_config["tile_shape"],
        "gemm_candidate_linear_staged_tail_threadgroup_threads":
            variant_config["threads"],
        "gemm_candidate_linear_staged_tail_static_threadgroup_memory_bytes":
            variant_config["linear_staged_tail_static_threadgroup_memory_bytes"],
        "gemm_linear_dispatch_partition": results.expected_dense_dispatch_partition(
            rows, in_dim, out_dim, variant_config["tile_shape"]),
    }


def synthetic_q4k_kernel_fields(rows: int = ROWS, in_dim: int = DIN,
                                out_dim: int = DOUT,
                                reverse_first: bool = False) -> dict:
    """Return exact metadata for the fixed baseline/map/index ablation."""
    baseline = results.Q4K_KERNEL_VARIANTS["device_q4k_baseline"]
    staged_bytes = results.WEIGHT_COUNT * 4
    return {
        "q4k_kernel_comparison": True,
        "q4k_kernel_variants": results.q4k_kernel_variant_metadata(),
        "q4k_kernel_dispatch_partition": results.expected_dense_dispatch_partition(
            rows, in_dim, out_dim, baseline["tile_shape"]),
        "q4k_kernel_schedule": results.Q4K_KERNEL_SCHEDULE,
        "q4k_kernel_schedule_directions": {
            "first_half": "reverse" if reverse_first else "forward",
            "second_half": "forward" if reverse_first else "reverse",
        },
        "q4k_kernel_warmup_cycles": 1,
        "q4k_kernel_formats": list(results.ARMS),
        "q4k_kernel_candidate_global_f32_weight_stage_bytes": 0,
        "q4k_kernel_control_f32_weight_stage_bytes": staged_bytes,
        "q4k_kernel_retained_harness_bytes_excluding_stage": 1024**3 - staged_bytes,
        "q4k_kernel_dummy_sidecar_bytes": 4,
        "explicit_buffer_bytes_during_pack": 1024**3,
    }


def complete_synthetic_probe(base: Path, rounds: int = 16, *,
                              block_validation_comparison: bool = False,
                              fused_comparison: bool = False,
                              gemm_comparison: bool = False,
                              gemm_variant: str | None = None,
                              q4k_kernel_comparison: bool = False,
                              reverse_first: bool = False) -> tuple[str, dict[str, str]]:
    fixture, fixture_files = make_reference_fixture(base, ROWS)
    source_manifest = Path(fixture_files["__source_manifest__"])
    source_fixture = Path(fixture_files["__source_dir__"])
    source_pins = fixture_files["__source_pins__"]
    binary = base / "probe-v1"
    binary.write_bytes(b"synthetic native executable identity")
    output_manifest = fixture / "manifest.json"
    manifest_sha = results.sha(output_manifest)

    metric = ZERO_METRIC
    if sum((block_validation_comparison, fused_comparison, gemm_comparison,
            q4k_kernel_comparison)) > 1:
        raise ValueError("Comparison modes are mutually exclusive")
    if gemm_variant is None:
        gemm_variant = (results.GEMM_CANDIDATE_VARIANT if gemm_comparison else
                        results.GEMM_DEVICE_Q4K_VARIANT if q4k_kernel_comparison else "legacy")
    variant_config = results.GEMM_VARIANTS[gemm_variant]
    setup = {
        "schema": "wan_q4k_delta_setup_v1",
        "manifest_sha256": manifest_sha,
        "rows": ROWS,
        "din": DIN,
        "dout": DOUT,
        "unique_source_rows": 4,
        "real_video_derived": False,
        "variants": list(results.ARMS),
        "per_arm_storage": {arm: results.expected_storage(arm) for arm in results.ARMS},
        "shared_bias_bytes": DOUT * 4,
        "f32_stage_bytes": results.WEIGHT_COUNT * 4,
        "packing_workspace_bytes": 0,
        "explicit_buffer_bytes_during_pack": 1024**3,
        "reverse_first": False,
        "gemm_comparison": gemm_comparison,
        "gemm_variant": gemm_variant,
        "gemm_variant_name": gemm_variant,
        "gemm_tile_shape": variant_config["tile_shape"],
        "gemm_threadgroup_threads": variant_config["threads"],
        "dense_static_threadgroup_memory_bytes": None,
        "gemm_legacy_dense_static_threadgroup_memory_bytes": None,
        "gemm_candidate_dense_static_threadgroup_memory_bytes": None,
        "gemm_legacy_dense_pipeline": None,
        "gemm_candidate_dense_pipeline": None,
        "candidate_pipeline": None,
        "candidate_static_threadgroup_memory_bytes": None,
        "dense_mode_execution_arms": None,
        "dense_mode_weight_format": None,
        "dense_mode_internal_tile_shape": None,
        "gemm_schedule": None,
        "gemm_schedule_directions": None,
        "mps_storage": "opaque and included in the process guard, excluded from explicit byte sum",
        "tolerance": "finite && abs(actual-ref)<=2e-5+2e-5*abs(ref)",
    }
    if block_validation_comparison:
        setup["block_validation_comparison"] = True
        setup["execution_arms"] = list(EXECUTION_ARMS)
        setup["format_by_arm"] = {
            arm: arm.removesuffix("_block") for arm in EXECUTION_ARMS
        }
    if fused_comparison:
        setup.update({
            "block_validation_comparison": False,
            "fused_comparison": True,
            "execution_arms": list(results.FUSED_ARMS),
            "format_by_arm": {
                **{arm: arm for arm in results.ARMS},
                **{arm: arm.removesuffix("_fused") for arm in results.FUSED_ARMS},
            },
            "fused_staged_control_decoder": "block_validated",
            "candidate_global_f32_weight_stage_bytes": 0,
            "control_f32_weight_stage_bytes": results.WEIGHT_COUNT * 4,
            "retained_harness_bytes_excluding_stage": 1024**3 - results.WEIGHT_COUNT * 4,
            "q4_k_fused_dummy_sidecar_bytes": 4,
            "candidate_pipeline": variant_config["linear_pipeline"],
            "candidate_static_threadgroup_memory_bytes": variant_config.get(
                "linear_static_threadgroup_memory_bytes", 0),
            "dense_static_threadgroup_memory_bytes": variant_config[
                "dense_static_threadgroup_memory_bytes"],
            "dense_mode_execution_arms": ["dense_native", "dense_mps"],
            "dense_mode_weight_format": "q4_k",
            "dense_mode_internal_tile_shape": variant_config["tile_shape"],
        })
        if gemm_variant == results.GEMM_DEVICE_Q4K_VARIANT:
            setup.update(synthetic_device_q4k_fields(variant_config))
            setup.update(synthetic_device_dense_fields(variant_config))
    if gemm_comparison:
        setup.update({
            "execution_arms": list(results.GEMM_ARMS),
            "format_by_arm": {
                **{arm: arm for arm in results.ARMS},
                **{arm: "q4_k" for arm in results.GEMM_ARMS},
            },
            "gemm_tile_shape": variant_config["tile_shape"],
            "gemm_threadgroup_threads": variant_config["threads"],
            "dense_static_threadgroup_memory_bytes": variant_config[
                "dense_static_threadgroup_memory_bytes"],
            "gemm_legacy_dense_static_threadgroup_memory_bytes":
                results.GEMM_VARIANTS["legacy"]["dense_static_threadgroup_memory_bytes"],
            "gemm_candidate_dense_static_threadgroup_memory_bytes": variant_config[
                "dense_static_threadgroup_memory_bytes"],
            "gemm_legacy_dense_pipeline": results.GEMM_VARIANTS["legacy"]["dense_pipeline"],
            "gemm_candidate_dense_pipeline": variant_config["dense_pipeline"],
            "dense_mode_execution_arms": list(results.GEMM_ARMS),
            "dense_mode_weight_format": "q4_k",
            "dense_mode_internal_tile_shape": variant_config["tile_shape"],
            "gemm_schedule": "six_sample_palindrome_opposite_base_direction_by_half_v1",
            "gemm_schedule_directions": {
                "first_half": "reverse" if reverse_first else "forward",
                "second_half": "forward" if reverse_first else "reverse",
            },
        })
        if gemm_variant in (results.GEMM_DEVICE_F32_VARIANT,
                            results.GEMM_DEVICE_Q4K_VARIANT):
            setup.update({
                "gemm_candidate_dense_direct_pipeline": variant_config["dense_pipeline"],
                "gemm_candidate_dense_direct_tile_shape": variant_config["tile_shape"],
                "gemm_candidate_dense_direct_threadgroup_threads": variant_config["threads"],
                "gemm_candidate_dense_direct_static_threadgroup_memory_bytes":
                    variant_config["dense_static_threadgroup_memory_bytes"],
                "gemm_candidate_dense_staged_tail_pipeline": variant_config[
                    "staged_tail_pipeline"],
                "gemm_candidate_dense_staged_tail_tile_shape": variant_config["tile_shape"],
                "gemm_candidate_dense_staged_tail_threadgroup_threads":
                    variant_config["threads"],
                "gemm_candidate_dense_staged_tail_static_threadgroup_memory_bytes":
                    variant_config["staged_tail_static_threadgroup_memory_bytes"],
                "gemm_dense_dispatch_partition": results.expected_dense_dispatch_partition(
                    ROWS, DIN, DOUT, variant_config["tile_shape"]),
            })
    if q4k_kernel_comparison:
        setup.update({
            "block_validation_comparison": False,
            "fused_comparison": False,
            "gemm_comparison": False,
            "execution_arms": list(results.Q4K_KERNEL_ARMS),
            "format_by_arm": {
                **{arm: arm for arm in results.ARMS},
                **results.Q4K_KERNEL_FORMAT_BY_ARM,
            },
            **synthetic_q4k_kernel_fields(reverse_first=reverse_first),
        })
    setup["reverse_first"] = reverse_first
    records = [setup]
    if not gemm_comparison:
        for arm in results.ARMS[1:]:
            records.append({
                "schema": "wan_q4k_delta_packing_v1",
                "variant": arm,
                "gpu_seconds": 0.01,
                "wall_seconds": 0.02,
                "encode_seconds": 0.001,
                "sidecar_word_differences": 0,
                "encode_metal_buffer_allocations": 0,
                "activation_readbacks": 0,
                "terminal_waits": 1,
                "byte_validation_outside_packing_interval": True,
            })
    if not gemm_comparison:
        records.append({
            "schema": "wan_q4k_delta_bf16_control_v1",
            "decoder": metric(DIN * DOUT),
            "projection": metric(ROWS * DOUT),
        })
    execution_arms = (results.GEMM_ARMS if gemm_comparison else
                      results.Q4K_KERNEL_ARMS if q4k_kernel_comparison else
                      results.FUSED_ARMS if fused_comparison else
                      EXECUTION_ARMS if block_validation_comparison else results.ARMS)
    if q4k_kernel_comparison:
        for phase in ("before_timing", "after_timing"):
            for format_name in results.ARMS:
                records.append({
                    "schema": "wan_q4k_delta_kernel_parity_v1",
                    "phase": phase,
                    "format": format_name,
                    "reference_arm": format_name,
                    "decoder": metric(DIN * DOUT),
                    "projections": {
                        arm: metric(ROWS * DOUT) for arm in results.Q4K_KERNEL_ARMS
                    },
                    "pairs": {
                        pair: metric(ROWS * DOUT)
                        for pair in results.Q4K_KERNEL_PAIR_KEYS
                    },
                })
    elif not gemm_comparison:
        for phase in ("before_timing", "after_timing"):
            for arm in execution_arms:
                records.append({
                "schema": "wan_q4k_delta_parity_v1",
                "phase": phase,
                "variant": arm,
                "decoder": metric(DIN * DOUT),
                "projection": metric(ROWS * DOUT),
                **({"fused_vs_staged": metric(ROWS * DOUT)}
                   if fused_comparison and arm.endswith("_fused") else {}),
                "lossy_weight_vs_bf16": metric(DIN * DOUT, bit_differences=DIN * DOUT,
                                               tolerance_violations=DIN * DOUT,
                                               rmse=0.1, relative_l2=0.1, cosine=0.99),
                "lossy_vs_bf16_per_source_row": [
                    metric(DOUT, bit_differences=DOUT, tolerance_violations=DOUT,
                           rmse=0.1, relative_l2=0.1, cosine=0.99)
                    for _ in range(4)
                ],
                "calibration_rows": [],
                    "unique_evaluation_rows": [0, 1, 2, 3],
                })
    else:
        for phase in ("before_timing", "after_timing"):
            records.append({
                "schema": "wan_q4k_delta_gemm_parity_v1",
                "phase": phase,
                "format": "q4_k",
                "reference_arm": "q4_k",
                "decoder": metric(DIN * DOUT),
                "dense_native_legacy": metric(ROWS * DOUT),
                "dense_native_candidate": metric(ROWS * DOUT),
                "dense_mps": metric(ROWS * DOUT),
                "candidate_vs_legacy": metric(ROWS * DOUT),
                "candidate_vs_mps": metric(ROWS * DOUT),
            })
    if fused_comparison:
        for phase in ("before_timing", "after_timing"):
            for arm in ("dense_native", "dense_mps"):
                records.append({
                    "schema": "wan_q4k_delta_dense_parity_v1",
                    "phase": phase,
                    "variant": arm,
                    "format": "q4_k",
                    "reference_arm": "q4_k",
                    "decoder": metric(DIN * DOUT),
                    "dense_native": metric(ROWS * DOUT),
                    "dense_mps": metric(ROWS * DOUT),
                    "dense_native_vs_dense_mps": metric(ROWS * DOUT),
                })
        records.extend(fused_timing_records(
            rounds, reverse_first=reverse_first, manifest_sha256=manifest_sha))
    elif gemm_comparison:
        records.extend(gemm_timing_records(
            rounds, reverse_first=reverse_first, manifest_sha256=manifest_sha))
    elif q4k_kernel_comparison:
        records.extend(q4k_kernel_timing_records(
            rounds, reverse_first=reverse_first, manifest_sha256=manifest_sha))
    else:
        records.extend(timing_records(
            rounds, reverse_first=reverse_first, manifest_sha256=manifest_sha,
            block_validation_comparison=block_validation_comparison))
    control = {
        "schema": "wan_q4k_delta_controls_v1",
        "corruption_rejected": True,
        "corruption": metric(ROWS * DOUT, bit_differences=(4 if fused_comparison else
                                                             53_899 if q4k_kernel_comparison else 2),
                              tolerance_violations=2, nonfinite=1),
        "final_status_zero": True,
    }
    if fused_comparison:
        control.update({
            "fused_vs_staged_corruption_rejected": True,
            "fused_vs_staged_corruption": metric(
                ROWS * DOUT, bit_differences=9, tolerance_violations=2, nonfinite=1),
        })
    if q4k_kernel_comparison:
        control.update({
            "q4k_kernel_vs_staged_corruption_rejected": True,
            "q4k_kernel_vs_staged_corruption": metric(
                ROWS * DOUT, bit_differences=9, tolerance_violations=2, nonfinite=1),
        })
    records.append(control)

    stdout = base / "probe-r4.stdout.log"
    stdout.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records))
    stderr = base / "probe-r4.stderr.log"
    stderr.write_text("")
    guard = base / "probe-r4.guard.jsonl"
    guard.write_text("".join(json.dumps(event, sort_keys=True) + "\n"
                             for event in valid_guard_events()))
    build_intent_path = base / "build-v1.intent.json"
    build_receipt_path = base / "build-v1.completion.json"
    build_intent = {
        "schema": "wan_q4k_delta_intent_v1",
        "mode": "build",
        "tag": "v1",
        "command": ["crystal", "build", "-o", str(binary)],
        "files": {str(binary): results.sha(binary)},
    }
    write_json(build_intent_path, build_intent)
    build_receipt = {
        "schema": "wan_q4k_delta_completion_v1",
        "returncode": 0,
        "changed_files": [],
        "intent_sha256": results.sha(build_intent_path),
        "sources": {},
        "binary_sha256": results.sha(binary),
    }
    write_json(build_receipt_path, build_receipt)

    oracle_intent_path = base / "oracle-r4.intent.json"
    oracle_receipt_path = base / "oracle-r4.completion.json"
    oracle_intent = {
        "schema": "wan_q4k_delta_intent_v1",
        "mode": "oracle",
        "tag": "r4",
        "command": ["python", "reference.py", "--fixture-dir", str(source_fixture),
                    "--output-dir", str(fixture), "--rows", str(ROWS)],
        "files": {str(source_manifest): results.sha(source_manifest)},
    }
    write_json(oracle_intent_path, oracle_intent)
    oracle_receipt = {
        "schema": "wan_q4k_delta_completion_v1",
        "returncode": 0,
        "changed_files": [],
        "intent_sha256": results.sha(oracle_intent_path),
        "sources": {},
        "produced_manifest_sha256": manifest_sha,
    }
    write_json(oracle_receipt_path, oracle_receipt)

    probe_intent_path = base / "probe-r4.intent.json"
    probe_receipt_path = base / "probe-r4.completion.json"
    files = {
        **{name: digest for name, digest in fixture_files.items() if not name.startswith("__")},
        str(binary): results.sha(binary),
        str(build_receipt_path): results.sha(build_receipt_path),
        str(oracle_receipt_path): results.sha(oracle_receipt_path),
    }
    probe_intent = {
        "schema": "wan_q4k_delta_intent_v1",
        "mode": "probe",
        "tag": "r4",
        "command": [str(binary), f"--fixture={fixture}", f"--rounds={rounds}"] +
                   (["--compare-block-validation"] if block_validation_comparison else []) +
                   (["--compare-gemm", f"--gemm-variant={gemm_variant}"]
                    if gemm_comparison else []) +
                   (["--compare-fused", f"--gemm-variant={gemm_variant}"]
                    if fused_comparison else []) +
                   (["--compare-q4k-kernels"] if q4k_kernel_comparison else []) +
                   (["--reverse-first"] if reverse_first else []),
        "reverse_first": reverse_first,
        "block_validation_comparison": block_validation_comparison,
        "gemm_comparison": gemm_comparison,
        "gemm_variant": gemm_variant,
        "fused_comparison": fused_comparison,
        "q4k_kernel_comparison": q4k_kernel_comparison,
        "files": files,
        "memory_accounting": "file-cache",
        "reserve_pct": 15,
        "max_footprint_mib": 4096,
        "max_swap_growth_mib": 64,
        "timeout_sec": 300,
    }
    write_json(probe_intent_path, probe_intent)
    probe_receipt = {
        "schema": "wan_q4k_delta_completion_v1",
        "returncode": 0,
        "changed_files": [],
        "intent_sha256": results.sha(probe_intent_path),
        "stdout_sha256": results.sha(stdout),
        "stderr_sha256": results.sha(stderr),
        "guard_sha256": results.sha(guard),
        "sources": {},
        "binary_sha256": results.sha(binary),
    }
    write_json(probe_receipt_path, probe_receipt)
    return "r4", source_pins


def rewrite_stdout(evidence: Path, mutate) -> None:
    stdout = evidence / "probe-r4.stdout.log"
    lines = [json.loads(line) for line in stdout.read_text().splitlines()]
    mutate(lines)
    stdout.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in lines))
    receipt_path = evidence / "probe-r4.completion.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["stdout_sha256"] = results.sha(stdout)
    write_json(receipt_path, receipt)


class ResultAnalyzerSpec(unittest.TestCase):
    def test_inventory_count_matches_actual_16_round_sequence(self):
        timing_count = len(timing_records())
        self.assertEqual(timing_count, (16 + 2) * len(results.ARMS) * len(results.MODES))
        self.assertEqual(results.expected_probe_record_count(timing_count), 158)
        comparison_timing_count = len(timing_records(block_validation_comparison=True))
        self.assertEqual(comparison_timing_count,
                         (16 + 2) * len(EXECUTION_ARMS) * len(results.MODES))
        self.assertEqual(results.expected_probe_record_count(comparison_timing_count), 310)
        fused_timing_count = len(fused_timing_records())
        self.assertEqual(fused_timing_count, (16 + 2) * len(results.FUSED_ARMS) + 4 + 16 * 4)
        self.assertEqual(results.expected_probe_record_count(
            fused_timing_count, False, True), 238)

        q4k_timing_count = len(q4k_kernel_timing_records())
        self.assertEqual(q4k_timing_count,
                         (results.Q4K_KERNEL_ROUNDS + 1) * len(results.ARMS) *
                         len(results.Q4K_KERNEL_ARMS))
        self.assertEqual(q4k_timing_count, 272)
        self.assertEqual(results.expected_probe_record_count(
            q4k_timing_count, False, False, False, True), 286)

    def test_q4k_kernel_ablation_receipts_bind_parity_and_balanced_schedule(self):
        for reverse_first in (False, True):
            with self.subTest(reverse_first=reverse_first), tempfile.TemporaryDirectory() as temporary:
                evidence = Path(temporary) / "evidence"
                evidence.mkdir()
                tag, source_pins = complete_synthetic_probe(
                    evidence, q4k_kernel_comparison=True, reverse_first=reverse_first)
                with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                    accepted = results.read_probe(evidence, tag)

                control = next(record for record in
                               (json.loads(line) for line in
                                (evidence / "probe-r4.stdout.log").read_text().splitlines())
                               if record.get("schema") == "wan_q4k_delta_controls_v1")
                self.assertEqual(control["corruption"]["bit_differences"], 53_899)

                setup = accepted["setup"]
                self.assertTrue(accepted["lineage"]["q4k_kernel_comparison"])
                self.assertEqual(setup["gemm_variant"], results.GEMM_DEVICE_Q4K_VARIANT)
                self.assertEqual(setup["q4k_kernel_variants"],
                                 results.q4k_kernel_variant_metadata())
                self.assertEqual(setup["q4k_kernel_dispatch_partition"],
                                 results.expected_dense_dispatch_partition(
                                     ROWS, DIN, DOUT,
                                     results.Q4K_KERNEL_VARIANTS[
                                         "device_q4k_baseline"]["tile_shape"]))
                self.assertEqual(setup["q4k_kernel_candidate_global_f32_weight_stage_bytes"], 0)
                self.assertEqual(setup["q4k_kernel_control_f32_weight_stage_bytes"],
                                 results.WEIGHT_COUNT * 4)

                parity = accepted["q4k_kernel_parity"]
                self.assertEqual(len(parity), len(results.ARMS) * 2)
                for summary in parity.values():
                    for metric_record in (summary["decoder"], *summary["projections"].values(),
                                          *summary["pairs"].values()):
                        self.assertEqual(metric_record["tolerance_violations"], 0)
                        self.assertEqual(metric_record["nonfinite"], 0)

                timings = accepted["timings_ms"]["q4k_kernel_ablation"]
                self.assertFalse(timings["speed_winner_selected"])
                for format_name in results.ARMS:
                    arms = timings["arms_by_format"][format_name]
                    self.assertEqual(set(arms), set(results.Q4K_KERNEL_ARMS))
                    for arm_summary in arms.values():
                        self.assertEqual(arm_summary["gpu_seconds"]["count"], 16)
                    positions = timings["measured_position_counts_by_format_and_arm"][format_name]
                    for arm_positions in positions.values():
                        self.assertEqual(set(arm_positions.values()), {4})
                    pairs = timings["paired_gpu_ratios_by_format"][format_name]
                    self.assertEqual(set(pairs), set(results.Q4K_KERNEL_PAIR_KEYS))
                    for pair in pairs.values():
                        ratios = pair["left_over_right_gpu_ratio"]
                        self.assertEqual(ratios["all_rounds"]["count"], 16)
                        self.assertEqual(ratios["by_half"]["first"]["count"], 8)
                        self.assertEqual(ratios["by_half"]["second"]["count"], 8)
                        self.assertEqual(len(ratios["by_order"]), 8)

                rows = [json.loads(line) for line in
                        (evidence / "probe-r4.stdout.log").read_text().splitlines()]
                timed = [record for record in rows
                         if record.get("mode") == "q4k_kernel_ablation"]
                self.assertEqual(len(timed), 272)
                actual_warmup = [
                    (record["format"], record["format_position"],
                     record["variant"], record["position"])
                    for record in timed if record["round"] == -1
                ]
                self.assertEqual(actual_warmup,
                                 results.q4k_kernel_round_order(-1, reverse_first))

    def test_q4k_kernel_setup_and_parity_reject_tampered_metadata(self):
        baseline = results.Q4K_KERNEL_VARIANTS["device_q4k_baseline"]
        setup = {
            "gemm_comparison": False,
            "fused_comparison": False,
            "gemm_variant": results.GEMM_DEVICE_Q4K_VARIANT,
            "gemm_variant_name": baseline["variant_name"],
            "gemm_tile_shape": baseline["tile_shape"],
            "gemm_threadgroup_threads": baseline["threads"],
            "reverse_first": False,
            **synthetic_q4k_kernel_fields(),
        }
        results.validate_q4k_kernel_setup(setup, ROWS, DIN, DOUT)
        partition = setup["q4k_kernel_dispatch_partition"]
        mutations = (
            ("selector", "q4k_kernel_variants", [
                {**item, **({"selector": "unbound-selector"}
                            if item["arm"] == "device_q4k_map" else {})}
                for item in setup["q4k_kernel_variants"]
            ]),
            ("core pipeline", "q4k_kernel_variants", [
                {**item, **({"direct_pipeline": "unbound-pipeline"}
                            if item["arm"] == "device_q4k_index" else {})}
                for item in setup["q4k_kernel_variants"]
            ]),
            ("tail resource", "q4k_kernel_variants", [
                {**item, **({"staged_tail_static_threadgroup_memory_bytes": 14_460}
                            if item["arm"] == "device_q4k_map" else {})}
                for item in setup["q4k_kernel_variants"]
            ]),
            ("partition", "q4k_kernel_dispatch_partition",
             {**partition, "bottom_tail_groups": partition["bottom_tail_groups"] - 1}),
            ("global F32 stage", "q4k_kernel_candidate_global_f32_weight_stage_bytes", 4),
            ("schedule", "q4k_kernel_schedule", "unbound_schedule"),
            ("warmup count", "q4k_kernel_warmup_cycles", 2),
        )
        for name, field, value in mutations:
            with self.subTest(mutation=name):
                corrupt = json.loads(json.dumps(setup))
                corrupt[field] = value
                with self.assertRaises(ValueError):
                    results.validate_q4k_kernel_setup(corrupt, ROWS, DIN, DOUT)

        valid = []
        for phase in ("before_timing", "after_timing"):
            for format_name in results.ARMS:
                valid.append({
                    "schema": "wan_q4k_delta_kernel_parity_v1",
                    "phase": phase,
                    "format": format_name,
                    "reference_arm": format_name,
                    "decoder": ZERO_METRIC(DIN * DOUT),
                    "projections": {
                        arm: ZERO_METRIC(ROWS * DOUT) for arm in results.Q4K_KERNEL_ARMS
                    },
                    "pairs": {
                        pair: ZERO_METRIC(ROWS * DOUT)
                        for pair in results.Q4K_KERNEL_PAIR_KEYS
                    },
                })
        results.validate_q4k_kernel_parity(valid, ROWS, DIN, DOUT)
        corrupt_parity = json.loads(json.dumps(valid))
        corrupt_parity[0]["pairs"][results.Q4K_KERNEL_PAIR_KEYS[0]][
            "tolerance_violations"] = 1
        with self.assertRaisesRegex(ValueError, "Strict finite parity failed"):
            results.validate_q4k_kernel_parity(corrupt_parity, ROWS, DIN, DOUT)

    def test_dense_gemm_candidate_receipts_bind_order_parity_and_pipeline_metadata(self):
        candidate = results.GEMM_CANDIDATE_VARIANT
        for reverse_first in (False, True):
            with self.subTest(reverse_first=reverse_first), tempfile.TemporaryDirectory() as temporary:
                evidence = Path(temporary) / "evidence"
                evidence.mkdir()
                tag, source_pins = complete_synthetic_probe(
                    evidence, gemm_comparison=True, gemm_variant=candidate,
                    reverse_first=reverse_first)
                with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                    accepted = results.read_probe(evidence, tag)
                self.assertEqual(accepted["setup"]["gemm_variant"], candidate)
                self.assertEqual(accepted["setup"]["gemm_tile_shape"],
                                 results.GEMM_VARIANTS[candidate]["tile_shape"])
                self.assertEqual(accepted["setup"]["gemm_legacy_dense_pipeline"],
                                 results.GEMM_VARIANTS["legacy"]["dense_pipeline"])
                self.assertEqual(accepted["setup"]["gemm_candidate_dense_pipeline"],
                                 results.GEMM_VARIANTS[candidate]["dense_pipeline"])
                self.assertEqual(
                    results._gemm_order(-2, 16, reverse_first),
                    tuple(record["variant"] for record in gemm_timing_records(
                        reverse_first=reverse_first)[:6]))
                timing = accepted["timings_ms"]["dense_gemm"]
                self.assertEqual(
                    timing["arms"]["dense_native_candidate"]["gpu_seconds"]["count"], 32)
                paired = timing["paired_candidate_over_legacy_and_mps_gpu"]
                for comparison in paired.values():
                    ratios = comparison["candidate_over_control_gpu_ratio"]
                    self.assertEqual(ratios["all_rounds"]["count"], 16)
                    self.assertEqual(set(ratios["by_half"]), {"first", "second"})
                    self.assertEqual(set(ratios["by_order"]), {
                        ">".join(results._gemm_order(round_id, 16, reverse_first))
                        for round_id in range(16)
                    })

    def test_dense_gemm_b128_v1_receipt_without_pipeline_names_remains_accepted(self):
        legacy_candidate = "candidate-b128n32-sg4-tk32"
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag, source_pins = complete_synthetic_probe(
                evidence, gemm_comparison=True, gemm_variant=legacy_candidate)

            def remove_v1_unavailable_pipeline_names(records):
                setup = records[0]
                setup.pop("gemm_legacy_dense_pipeline")
                setup.pop("gemm_candidate_dense_pipeline")

            rewrite_stdout(evidence, remove_v1_unavailable_pipeline_names)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                accepted = results.read_probe(evidence, tag)
            self.assertEqual(accepted["setup"]["gemm_variant"], legacy_candidate)

    def test_dense_gemm_b64_candidates_require_pipeline_identity(self):
        for candidate in results.GEMM_PIPELINE_METADATA_REQUIRED_VARIANTS:
            for mutation in ("missing", "wrong"):
                with self.subTest(candidate=candidate, mutation=mutation), \
                        tempfile.TemporaryDirectory() as temporary:
                    evidence = Path(temporary) / "evidence"
                    evidence.mkdir()
                    tag, source_pins = complete_synthetic_probe(
                        evidence, gemm_comparison=True, gemm_variant=candidate)

                    def corrupt_pipeline_name(records):
                        if mutation == "missing":
                            records[0].pop("gemm_legacy_dense_pipeline")
                            records[0].pop("gemm_candidate_dense_pipeline")
                        else:
                            records[0]["gemm_candidate_dense_pipeline"] = "unbound_pipeline"

                    rewrite_stdout(evidence, corrupt_pipeline_name)
                    with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                        with self.assertRaisesRegex(
                                ValueError,
                                "Dense-GEMM legacy/candidate pipeline identity changed"):
                            results.read_probe(evidence, tag)

    def test_dense_gemm_tk64_binds_exact_geometry_tgm_and_pipeline_metadata(self):
        candidate = results.GEMM_TK64_VARIANT
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag, source_pins = complete_synthetic_probe(
                evidence, gemm_comparison=True, gemm_variant=candidate)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                accepted = results.read_probe(evidence, tag)
            setup = accepted["setup"]
            self.assertEqual(setup["gemm_tile_shape"], {"tm": 64, "tn": 32, "tk": 64})
            self.assertEqual(setup["dense_static_threadgroup_memory_bytes"], 24_576)
            self.assertEqual(setup["gemm_candidate_dense_static_threadgroup_memory_bytes"], 24_576)
            self.assertEqual(setup["gemm_legacy_dense_static_threadgroup_memory_bytes"], 12_288)
            self.assertEqual(setup["gemm_candidate_dense_pipeline"],
                             results.GEMM_VARIANTS[candidate]["dense_pipeline"])
            self.assertEqual(accepted["timings_ms"]["dense_gemm"]["arms"]
                             ["dense_native_candidate"]["gpu_seconds"]["count"], 32)

    def test_dense_gemm_tk64_rejects_wrong_actual_threadgroup_memory(self):
        candidate = results.GEMM_TK64_VARIANT
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag, source_pins = complete_synthetic_probe(
                evidence, gemm_comparison=True, gemm_variant=candidate)

            def corrupt_tgm(records):
                setup = records[0]
                setup["dense_static_threadgroup_memory_bytes"] = 24_572
                setup["gemm_candidate_dense_static_threadgroup_memory_bytes"] = 24_572

            rewrite_stdout(evidence, corrupt_tgm)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                with self.assertRaisesRegex(
                        ValueError, "Dense-GEMM actual threadgroup memory differs"):
                    results.read_probe(evidence, tag)

    def test_dense_gemm_tk64_rejects_wrong_tile_shape(self):
        candidate = results.GEMM_TK64_VARIANT
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag, source_pins = complete_synthetic_probe(
                evidence, gemm_comparison=True, gemm_variant=candidate)

            def corrupt_tile_shape(records):
                records[0]["gemm_tile_shape"]["tk"] = 32
                records[0]["dense_mode_internal_tile_shape"]["tk"] = 32

            rewrite_stdout(evidence, corrupt_tile_shape)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                with self.assertRaisesRegex(
                        ValueError, "Selected GEMM variant geometry differs"):
                    results.read_probe(evidence, tag)

    def test_dense_device_f32_binds_direct_tail_pipelines_resources_and_m4_fallback(self):
        candidate = results.GEMM_DEVICE_F32_VARIANT
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag, source_pins = complete_synthetic_probe(
                evidence, gemm_comparison=True, gemm_variant=candidate)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                accepted = results.read_probe(evidence, tag)
            setup = accepted["setup"]
            config = results.GEMM_VARIANTS[candidate]
            self.assertEqual(setup["gemm_candidate_dense_direct_pipeline"],
                             config["dense_pipeline"])
            self.assertEqual(setup["gemm_candidate_dense_direct_tile_shape"],
                             {"tm": 64, "tn": 32, "tk": 32})
            self.assertEqual(setup["gemm_candidate_dense_direct_static_threadgroup_memory_bytes"],
                             8_192)
            self.assertEqual(setup["gemm_candidate_dense_staged_tail_pipeline"],
                             config["staged_tail_pipeline"])
            self.assertEqual(setup[
                "gemm_candidate_dense_staged_tail_static_threadgroup_memory_bytes"],
                12_288)
            self.assertEqual(setup["gemm_dense_dispatch_partition"], {
                "direct_rows": 0, "direct_columns": 13824,
                "direct_groups_x": 432, "direct_groups_y": 0,
                "right_tail_groups": 0, "bottom_tail_groups": 432,
                "direct_dispatches": 0, "staged_tail_dispatches": 1,
            })

    def test_dense_device_f32_full_and_mixed_dispatch_partitions_are_exact(self):
        candidate = results.GEMM_DEVICE_F32_VARIANT
        config = results.GEMM_VARIANTS[candidate]
        expected = (
            ((4480, DIN, DOUT), {
                "direct_rows": 4480, "direct_columns": 13824,
                "direct_groups_x": 432, "direct_groups_y": 70,
                "right_tail_groups": 0, "bottom_tail_groups": 0,
                "direct_dispatches": 1, "staged_tail_dispatches": 0,
            }),
            ((65, DIN, 35), {
                "direct_rows": 64, "direct_columns": 32,
                "direct_groups_x": 1, "direct_groups_y": 1,
                "right_tail_groups": 1, "bottom_tail_groups": 2,
                "direct_dispatches": 1, "staged_tail_dispatches": 2,
            }),
        )
        for (rows, in_dim, out_dim), partition in expected:
            with self.subTest(rows=rows, out_dim=out_dim):
                actual = results.expected_dense_dispatch_partition(
                    rows, in_dim, out_dim, config["tile_shape"])
                self.assertEqual(actual, partition)
                setup = {
                    "dense_static_threadgroup_memory_bytes":
                        config["dense_static_threadgroup_memory_bytes"],
                    "gemm_candidate_dense_pipeline": config["dense_pipeline"],
                    "gemm_candidate_dense_direct_pipeline": config["dense_pipeline"],
                    "gemm_candidate_dense_direct_tile_shape": config["tile_shape"],
                    "gemm_candidate_dense_direct_threadgroup_threads": config["threads"],
                    "gemm_candidate_dense_static_threadgroup_memory_bytes":
                        config["dense_static_threadgroup_memory_bytes"],
                    "gemm_candidate_dense_direct_static_threadgroup_memory_bytes":
                        config["dense_static_threadgroup_memory_bytes"],
                    "gemm_candidate_dense_staged_tail_pipeline":
                        config["staged_tail_pipeline"],
                    "gemm_candidate_dense_staged_tail_tile_shape": config["tile_shape"],
                    "gemm_candidate_dense_staged_tail_threadgroup_threads":
                        config["threads"],
                    "gemm_candidate_dense_staged_tail_static_threadgroup_memory_bytes":
                        config["staged_tail_static_threadgroup_memory_bytes"],
                    "gemm_dense_dispatch_partition": actual,
                }
                results.validate_device_f32_dense_metadata(
                    setup, rows, in_dim, out_dim, config)

    def test_dense_device_f32_rejects_tampered_full_and_tail_metadata(self):
        candidate = results.GEMM_DEVICE_F32_VARIANT
        config = results.GEMM_VARIANTS[candidate]
        full_partition = results.expected_dense_dispatch_partition(
            4480, DIN, DOUT, config["tile_shape"])
        full_setup = {
            "dense_static_threadgroup_memory_bytes":
                config["dense_static_threadgroup_memory_bytes"],
            "gemm_candidate_dense_pipeline": config["dense_pipeline"],
            "gemm_candidate_dense_direct_pipeline": config["dense_pipeline"],
            "gemm_candidate_dense_direct_tile_shape": config["tile_shape"],
            "gemm_candidate_dense_direct_threadgroup_threads": config["threads"],
            "gemm_candidate_dense_static_threadgroup_memory_bytes":
                config["dense_static_threadgroup_memory_bytes"],
            "gemm_candidate_dense_direct_static_threadgroup_memory_bytes":
                config["dense_static_threadgroup_memory_bytes"],
            "gemm_candidate_dense_staged_tail_pipeline": config["staged_tail_pipeline"],
            "gemm_candidate_dense_staged_tail_tile_shape": config["tile_shape"],
            "gemm_candidate_dense_staged_tail_threadgroup_threads": config["threads"],
            "gemm_candidate_dense_staged_tail_static_threadgroup_memory_bytes":
                config["staged_tail_static_threadgroup_memory_bytes"],
            "gemm_dense_dispatch_partition": full_partition,
        }
        mutations = (
            ("direct pipeline", "gemm_candidate_dense_direct_pipeline", "wrong_pipeline"),
            ("tail pipeline", "gemm_candidate_dense_staged_tail_pipeline", "wrong_tail"),
            ("direct geometry", "gemm_candidate_dense_direct_tile_shape",
             {"tm": 128, "tn": 32, "tk": 32}),
            ("tail geometry", "gemm_candidate_dense_staged_tail_tile_shape",
             {"tm": 32, "tn": 32, "tk": 32}),
            ("direct resources", "gemm_candidate_dense_direct_static_threadgroup_memory_bytes", 8196),
            ("tail resources", "gemm_candidate_dense_staged_tail_static_threadgroup_memory_bytes", 12284),
            ("full partition", "gemm_dense_dispatch_partition", {
                **full_partition, "direct_rows": 4416}),
        )
        for name, field, value in mutations:
            with self.subTest(mutation=name):
                setup = json.loads(json.dumps(full_setup))
                setup[field] = value
                with self.assertRaises(ValueError):
                    results.validate_device_f32_dense_metadata(
                        setup, 4480, DIN, DOUT, config)

        tail_setup = json.loads(json.dumps(full_setup))
        tail_setup["gemm_dense_dispatch_partition"] = \
            results.expected_dense_dispatch_partition(4, DIN, DOUT, config["tile_shape"])
        tail_setup["gemm_dense_dispatch_partition"]["bottom_tail_groups"] -= 1
        with self.assertRaisesRegex(ValueError, "dispatch partition"):
            results.validate_device_f32_dense_metadata(
                tail_setup, 4, DIN, DOUT, config)

    def test_device_q4k_fused_receipt_binds_packed_and_delegated_dense_paths(self):
        candidate = results.GEMM_DEVICE_Q4K_VARIANT
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag, source_pins = complete_synthetic_probe(
                evidence, fused_comparison=True, gemm_variant=candidate)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                accepted = results.read_probe(evidence, tag)
            setup = accepted["setup"]
            config = results.GEMM_VARIANTS[candidate]
            self.assertEqual(setup["candidate_pipeline"], config["linear_pipeline"])
            self.assertEqual(setup["candidate_static_threadgroup_memory_bytes"], 10_368)
            self.assertEqual(setup["gemm_candidate_linear_staged_tail_pipeline"],
                             config["linear_staged_tail_pipeline"])
            self.assertEqual(
                setup["gemm_candidate_linear_staged_tail_static_threadgroup_memory_bytes"],
                14_464)
            self.assertEqual(setup["gemm_linear_dispatch_partition"], {
                "direct_rows": 0, "direct_columns": 13_824,
                "direct_groups_x": 432, "direct_groups_y": 0,
                "right_tail_groups": 0, "bottom_tail_groups": 432,
                "direct_dispatches": 0, "staged_tail_dispatches": 1,
            })
            self.assertEqual(setup["gemm_candidate_dense_pipeline"],
                             config["dense_pipeline"])
            self.assertEqual(
                setup["gemm_candidate_dense_staged_tail_static_threadgroup_memory_bytes"],
                12_288)

    def test_device_q4k_packed_and_dense_dispatch_partitions_bind_full_and_mixed_shapes(self):
        config = results.GEMM_VARIANTS[results.GEMM_DEVICE_Q4K_VARIANT]
        expected = (
            ((4480, DIN, DOUT), {
                "direct_rows": 4480, "direct_columns": 13_824,
                "direct_groups_x": 432, "direct_groups_y": 70,
                "right_tail_groups": 0, "bottom_tail_groups": 0,
                "direct_dispatches": 1, "staged_tail_dispatches": 0,
            }),
            ((65, DIN, 35), {
                "direct_rows": 64, "direct_columns": 32,
                "direct_groups_x": 1, "direct_groups_y": 1,
                "right_tail_groups": 1, "bottom_tail_groups": 2,
                "direct_dispatches": 1, "staged_tail_dispatches": 2,
            }),
        )
        for (rows, in_dim, out_dim), partition in expected:
            with self.subTest(rows=rows, out_dim=out_dim):
                packed_partition = results.expected_dense_dispatch_partition(
                    rows, in_dim, out_dim, config["tile_shape"])
                self.assertEqual(packed_partition, partition)
                setup = {
                    "candidate_pipeline": config["linear_pipeline"],
                    "candidate_static_threadgroup_memory_bytes":
                        config["linear_static_threadgroup_memory_bytes"],
                    **synthetic_device_q4k_fields(config, rows, in_dim, out_dim),
                    **synthetic_device_dense_fields(config, rows, in_dim, out_dim),
                    "dense_static_threadgroup_memory_bytes":
                        config["dense_static_threadgroup_memory_bytes"],
                }
                self.assertEqual(setup["gemm_linear_dispatch_partition"], partition)
                self.assertEqual(setup["gemm_dense_dispatch_partition"], partition)
                results.validate_device_q4k_fused_metadata(
                    setup, rows, in_dim, out_dim, config)
                results.validate_device_f32_dense_metadata(
                    setup, rows, in_dim, out_dim, config)

    def test_device_q4k_fused_rejects_tampered_packed_or_dense_metadata(self):
        config = results.GEMM_VARIANTS[results.GEMM_DEVICE_Q4K_VARIANT]
        base_setup = {
            "candidate_pipeline": config["linear_pipeline"],
            "candidate_static_threadgroup_memory_bytes":
                config["linear_static_threadgroup_memory_bytes"],
            "dense_static_threadgroup_memory_bytes":
                config["dense_static_threadgroup_memory_bytes"],
            **synthetic_device_q4k_fields(config),
            **synthetic_device_dense_fields(config),
        }
        mutations = (
            ("packed core pipeline", "candidate_pipeline", "wrong"),
            ("packed direct pipeline", "gemm_candidate_linear_direct_pipeline", "wrong"),
            ("packed tail pipeline", "gemm_candidate_linear_staged_tail_pipeline", "wrong"),
            ("packed direct geometry", "gemm_candidate_linear_direct_tile_shape",
             {"tm": 128, "tn": 32, "tk": 32}),
            ("packed tail geometry", "gemm_candidate_linear_staged_tail_tile_shape",
             {"tm": 64, "tn": 16, "tk": 32}),
            ("packed core resources", "gemm_candidate_linear_direct_static_threadgroup_memory_bytes",
             10_364),
            ("packed tail resources",
             "gemm_candidate_linear_staged_tail_static_threadgroup_memory_bytes", 14_460),
            ("packed partition", "gemm_linear_dispatch_partition", {
                **results.expected_dense_dispatch_partition(
                    ROWS, DIN, DOUT, config["tile_shape"]),
                "bottom_tail_groups": 431,
            }),
            ("dense core geometry", "gemm_candidate_dense_direct_tile_shape",
             {"tm": 128, "tn": 32, "tk": 32}),
            ("dense tail geometry", "gemm_candidate_dense_staged_tail_tile_shape",
             {"tm": 64, "tn": 16, "tk": 32}),
            ("dense core pipeline", "gemm_candidate_dense_direct_pipeline", "wrong"),
            ("dense tail pipeline", "gemm_candidate_dense_staged_tail_pipeline", "wrong"),
            ("dense core actual resources", "dense_static_threadgroup_memory_bytes", 8_196),
            ("dense core resources", "gemm_candidate_dense_direct_static_threadgroup_memory_bytes",
             8_196),
            ("dense tail resources",
             "gemm_candidate_dense_staged_tail_static_threadgroup_memory_bytes", 12_284),
            ("dense partition", "gemm_dense_dispatch_partition", {
                **results.expected_dense_dispatch_partition(
                    ROWS, DIN, DOUT, config["tile_shape"]),
                "bottom_tail_groups": 431,
            }),
        )
        for name, field, value in mutations:
            with self.subTest(mutation=name):
                setup = json.loads(json.dumps(base_setup))
                setup[field] = value
                if name.startswith("packed"):
                    with self.assertRaises(ValueError):
                        results.validate_device_q4k_fused_metadata(
                            setup, ROWS, DIN, DOUT, config)
                else:
                    with self.assertRaises(ValueError):
                        results.validate_device_f32_dense_metadata(
                            setup, ROWS, DIN, DOUT, config)

    def test_direct_device_f32_selector_is_rejected_for_fused_comparison(self):
        candidate = results.GEMM_DEVICE_F32_VARIANT
        command = ["/tmp/probe-v1", "--fixture=/tmp/fixture-r4", "--rounds=16",
                   "--compare-fused", f"--gemm-variant={candidate}"]
        setup = {"fused_comparison": True, "gemm_variant": candidate}
        with self.assertRaisesRegex(ValueError, "dense-only|cannot be used for a fused"):
            results.validate_probe_command(command, False, setup,
                                           intent_gemm_variant=candidate)

    def test_fused_tk64_binds_exact_fused_and_dense_pipeline_memory(self):
        candidate = results.GEMM_TK64_VARIANT
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag, source_pins = complete_synthetic_probe(
                evidence, fused_comparison=True, gemm_variant=candidate)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                accepted = results.read_probe(evidence, tag)
            setup = accepted["setup"]
            self.assertEqual(setup["candidate_pipeline"],
                             results.GEMM_VARIANTS[candidate]["linear_pipeline"])
            self.assertEqual(setup["candidate_static_threadgroup_memory_bytes"], 26_752)
            self.assertEqual(setup["dense_static_threadgroup_memory_bytes"], 24_576)

    def test_fused_tk64_rejects_wrong_fused_pipeline_memory(self):
        candidate = results.GEMM_TK64_VARIANT
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag, source_pins = complete_synthetic_probe(
                evidence, fused_comparison=True, gemm_variant=candidate)

            def corrupt_fused_tgm(records):
                records[0]["candidate_static_threadgroup_memory_bytes"] = 26_748

            rewrite_stdout(evidence, corrupt_fused_tgm)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                with self.assertRaisesRegex(
                        ValueError, "TK64 fused/dense actual threadgroup memory differs"):
                    results.read_probe(evidence, tag)

    def test_dense_gemm_rejects_schedule_and_strict_cross_parity_mutations(self):
        candidate = results.GEMM_CANDIDATE_VARIANT
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag, source_pins = complete_synthetic_probe(
                evidence, gemm_comparison=True, gemm_variant=candidate)
            original_stdout = (evidence / f"probe-{tag}.stdout.log").read_text()

            def break_order(records):
                record = next(record for record in records
                              if record["schema"] == "wan_q4k_delta_timing_v1" and
                              record["round"] == 0 and record["position"] == 0)
                record["variant"] = "dense_native_candidate"

            rewrite_stdout(evidence, break_order)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                with self.assertRaisesRegex(ValueError, "Unexpected dense-GEMM arm order"):
                    results.read_probe(evidence, tag)

            stdout = evidence / f"probe-{tag}.stdout.log"
            stdout.write_text(original_stdout)
            completion_path = evidence / f"probe-{tag}.completion.json"
            completion = json.loads(completion_path.read_text())
            completion["stdout_sha256"] = results.sha(stdout)
            write_json(completion_path, completion)

            def break_cross_parity(records):
                record = next(record for record in records
                              if record["schema"] == "wan_q4k_delta_gemm_parity_v1" and
                              record["phase"] == "after_timing")
                record["candidate_vs_mps"]["tolerance_violations"] = 1

            rewrite_stdout(evidence, break_cross_parity)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                with self.assertRaisesRegex(ValueError, "Strict finite parity failed"):
                    results.read_probe(evidence, tag)

    def test_complete_positive_receipt_and_all_arm_records_are_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag, source_pins = complete_synthetic_probe(evidence)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                accepted = results.read_probe(evidence, tag)
            self.assertEqual(accepted["setup"]["rows"], ROWS)
            self.assertTrue(accepted["all_decoder_projection_checks_bit_exact"])
            self.assertFalse(accepted["timings_ms"]["projection"]["speed_winner_selected"])
            self.assertEqual(set(accepted["lossy_quality"]), set(results.ARMS))
            self.assertEqual(accepted["lineage"]["storage"]["q4_k_delta8"]["sidecar_bytes"],
                             results.BLOCK_COUNT * 8 * 3)

    def test_complete_block_validation_comparison_receipt_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag, source_pins = complete_synthetic_probe(
                evidence, block_validation_comparison=True, reverse_first=True)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                accepted = results.read_probe(evidence, tag)
            setup = accepted["setup"]
            self.assertIs(setup["block_validation_comparison"], True)
            self.assertEqual(setup["execution_arms"], list(EXECUTION_ARMS))
            self.assertEqual(setup["format_by_arm"], {
                arm: arm.removesuffix("_block") for arm in EXECUTION_ARMS
            })
            self.assertEqual(set(setup["variants"]), set(results.ARMS))
            self.assertEqual(set(setup["per_arm_storage"]), set(results.ARMS))
            self.assertEqual(set(accepted["lossy_quality"]), set(EXECUTION_ARMS))
            for mode in results.MODES:
                timing = accepted["timings_ms"][mode]
                self.assertEqual(
                    timing["arms"]["q4_k_block"]["gpu_seconds"]["count"], 16)
                paired = timing["paired_block_validation"]
                self.assertEqual(set(paired), set(results.ARMS))
                self.assertIs(paired["q4_k"]["aa_drift_control"], True)
                for format_arm in results.ARMS:
                    comparison = paired[format_arm]
                    self.assertEqual(
                        set(comparison["paired_percent_change"]),
                        {"gpu_seconds", "wall_seconds", "encode_seconds"})
                    self.assertEqual(
                        set(comparison["paired_percent_change"]["gpu_seconds"]["by_old_position"]),
                        {str(position) for position in range(len(EXECUTION_ARMS))})
                    self.assertEqual(
                        set(comparison["paired_percent_change"]["gpu_seconds"]["by_rotation"]),
                        {str(rotation) for rotation in range(len(results.ARMS))})
                    self.assertEqual(
                        set(comparison["paired_percent_change"]["gpu_seconds"]["by_half"]),
                        {"first", "second"})
                    self.assertIn("rotation_median_range",
                                  comparison["old_over_block_gpu_ratio"])

    def test_complete_fused_comparison_receipt_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag, source_pins = complete_synthetic_probe(
                evidence, fused_comparison=True, reverse_first=True)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                accepted = results.read_probe(evidence, tag)
            self.assertIs(accepted["setup"]["fused_comparison"], True)
            self.assertEqual(accepted["setup"]["execution_arms"], list(results.FUSED_ARMS))
            self.assertEqual(accepted["lineage"]["fused_comparison"], True)
            self.assertEqual(set(accepted["lossy_quality"]), set(results.FUSED_ARMS))
            self.assertEqual(set(accepted["dense_parity"]), {
                f"{phase}:{arm}"
                for phase in ("before_timing", "after_timing")
                for arm in ("dense_native", "dense_mps")
            })
            self.assertEqual(set(accepted["timings_ms"]), set(results.FUSED_MODES))
            timing = accepted["timings_ms"]["fused_projection"]
            self.assertEqual(set(timing["arms"]), set(results.FUSED_ARMS))
            self.assertEqual(set(timing["paired_staged_control_over_fused_gpu"]), set(results.ARMS))
            self.assertEqual(
                timing["arms"]["q4_k_fused"]["gpu_seconds"]["count"], 16)
            dense = accepted["timings_ms"]["dense_mode"]
            self.assertEqual(set(dense["arms"]), {"dense_native", "dense_mps"})
            self.assertEqual(
                dense["paired_dense_native_over_mps_gpu"]["all_rounds"]["count"], 16)

    def test_fused_comparison_rejects_schedule_coverage_and_comparator_mutations(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag, source_pins = complete_synthetic_probe(evidence, fused_comparison=True)
            stdout = evidence / f"probe-{tag}.stdout.log"
            original_records = [json.loads(line) for line in stdout.read_text().splitlines()]

            def check(mutator, expected_message):
                records = json.loads(json.dumps(original_records))
                mutator(records)
                stdout.write_text("".join(json.dumps(record, allow_nan=False) + "\n"
                                                 for record in records))
                receipt_path = evidence / f"probe-{tag}.completion.json"
                receipt = json.loads(receipt_path.read_text())
                receipt["stdout_sha256"] = results.sha(stdout)
                write_json(receipt_path, receipt)
                with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                    with self.assertRaisesRegex(ValueError, expected_message):
                        results.read_probe(evidence, tag)

            check(lambda records: records.pop(next(index for index, record in enumerate(records)
                  if record["schema"] == "wan_q4k_delta_timing_v1" and
                  record["mode"] == "fused_projection" and record["round"] == 0)),
                  "fused/dense timing records")

            def break_fused_order(records):
                record = next(record for record in records
                              if record["schema"] == "wan_q4k_delta_timing_v1" and
                              record["mode"] == "fused_projection" and
                              record["round"] == 0 and record["position"] == 0)
                record["variant"] = "q4_k_fused"

            check(break_fused_order, "arm order")

            def break_dense_order(records):
                record = next(record for record in records
                              if record["schema"] == "wan_q4k_delta_timing_v1" and
                              record["mode"] == "dense_mode" and
                              record["round"] == 0 and record["position"] == 0)
                record["variant"] = "dense_mps"

            check(break_dense_order, "Unexpected arm order: dense_mode")

            def make_timing_nonfinite(records):
                record = next(record for record in records
                              if record["schema"] == "wan_q4k_delta_timing_v1" and
                              record["mode"] == "dense_mode" and record["round"] == 0)
                record["gpu_seconds"] = "nan"

            check(make_timing_nonfinite, "Invalid numeric field")

            check(lambda records: records.pop(next(index for index, record in enumerate(records)
                  if record["schema"] == "wan_q4k_delta_dense_parity_v1" and
                  record["phase"] == "after_timing" and record["variant"] == "dense_mps")),
                  "dense-mode parity records")

            def corrupt_fused_comparator(records):
                control = next(record for record in records
                               if record["schema"] == "wan_q4k_delta_controls_v1")
                control["fused_vs_staged_corruption"]["tolerance_violations"] = 0

            check(corrupt_fused_comparator, "Positive fused/staged corruption")

            def remove_fused_parity(records):
                records[:] = [record for record in records if not (
                    record["schema"] == "wan_q4k_delta_parity_v1" and
                    record["variant"] == "q4_k_fused" and
                    record["phase"] == "before_timing")]

            check(remove_fused_parity, "parity records")

    def test_fused_comparison_binds_intent_mapping_and_buffer_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag, source_pins = complete_synthetic_probe(evidence, fused_comparison=True)
            intent_path = evidence / f"probe-{tag}.intent.json"
            receipt_path = evidence / f"probe-{tag}.completion.json"
            original_intent = json.loads(intent_path.read_text())

            intent = json.loads(json.dumps(original_intent))
            intent["fused_comparison"] = False
            write_json(intent_path, intent)
            receipt = json.loads(receipt_path.read_text())
            receipt["intent_sha256"] = results.sha(intent_path)
            write_json(receipt_path, receipt)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                with self.assertRaisesRegex(ValueError, "fused comparison flag"):
                    results.read_probe(evidence, tag)

            write_json(intent_path, original_intent)
            receipt = json.loads(receipt_path.read_text())
            receipt["intent_sha256"] = results.sha(intent_path)
            write_json(receipt_path, receipt)

            def check_setup(mutator, expected_message):
                stdout = evidence / f"probe-{tag}.stdout.log"
                original = [json.loads(line) for line in stdout.read_text().splitlines()]
                records = json.loads(json.dumps(original))
                setup = next(record for record in records
                             if record["schema"] == "wan_q4k_delta_setup_v1")
                mutator(setup)
                stdout.write_text("".join(json.dumps(record, allow_nan=False) + "\n"
                                                 for record in records))
                receipt = json.loads(receipt_path.read_text())
                receipt["stdout_sha256"] = results.sha(stdout)
                write_json(receipt_path, receipt)
                with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                    with self.assertRaisesRegex(ValueError, expected_message):
                        results.read_probe(evidence, tag)

            check_setup(lambda setup: setup["format_by_arm"].update({"q4_k_fused": "q4_k_delta1"}),
                        "format mapping")
            check_setup(lambda setup: setup.update(
                retained_harness_bytes_excluding_stage=
                    setup["retained_harness_bytes_excluding_stage"] - 4),
                "stage, control stage, dummy sidecar or pipeline accounting")

    def test_block_comparison_rejects_missing_arm_bad_order_and_non_bit_exact_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag, source_pins = complete_synthetic_probe(
                evidence, block_validation_comparison=True)
            stdout = evidence / f"probe-{tag}.stdout.log"
            original_records = [json.loads(line) for line in stdout.read_text().splitlines()]

            def check(mutator, expected_message):
                records = json.loads(json.dumps(original_records))
                mutator(records)
                stdout.write_text("".join(json.dumps(record, allow_nan=False) + "\n"
                                                 for record in records))
                receipt_path = evidence / f"probe-{tag}.completion.json"
                receipt = json.loads(receipt_path.read_text())
                receipt["stdout_sha256"] = results.sha(stdout)
                write_json(receipt_path, receipt)
                with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                    with self.assertRaisesRegex(ValueError, expected_message):
                        results.read_probe(evidence, tag)

            check(lambda records: records.__setitem__(
                slice(None), [record for record in records if not (
                    record["schema"] == "wan_q4k_delta_parity_v1" and
                    record["variant"] == "q4_k_block" and
                    record["phase"] == "before_timing")]),
                "parity records")

            def break_order(records):
                record = next(record for record in records
                              if record["schema"] == "wan_q4k_delta_timing_v1" and
                              record["mode"] == "projection" and record["round"] == 0 and
                              record["position"] == 0)
                record["variant"] = "q4_k_delta1_block"

            check(break_order, "arm order")

            def break_bit_exact(records):
                record = next(record for record in records
                              if record["schema"] == "wan_q4k_delta_parity_v1" and
                              record["variant"] == "q4_k_block" and
                              record["phase"] == "before_timing")
                record["decoder"]["bit_differences"] = 1

            check(break_bit_exact, "bit-exact")

    def test_block_comparison_rejects_flag_mapping_and_guard_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag, source_pins = complete_synthetic_probe(
                evidence, block_validation_comparison=True)
            intent_path = evidence / f"probe-{tag}.intent.json"
            receipt_path = evidence / f"probe-{tag}.completion.json"
            stdout = evidence / f"probe-{tag}.stdout.log"
            original_records = [json.loads(line) for line in stdout.read_text().splitlines()]

            original_intent = json.loads(intent_path.read_text())
            intent = json.loads(json.dumps(original_intent))
            intent["command"].remove("--compare-block-validation")
            write_json(intent_path, intent)
            receipt = json.loads(receipt_path.read_text())
            receipt["intent_sha256"] = results.sha(intent_path)
            write_json(receipt_path, receipt)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                with self.assertRaisesRegex(ValueError, "block-validation comparison"):
                    results.read_probe(evidence, tag)

            write_json(intent_path, original_intent)
            receipt = json.loads(receipt_path.read_text())
            receipt["intent_sha256"] = results.sha(intent_path)
            write_json(receipt_path, receipt)
            def break_mapping(records):
                setup = next(record for record in records
                             if record["schema"] == "wan_q4k_delta_setup_v1")
                setup["format_by_arm"]["q4_k_delta4_block"] = "q4_k_delta8"
            rewrite_stdout(evidence, break_mapping)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                with self.assertRaisesRegex(ValueError, "format|mapping|execution"):
                    results.read_probe(evidence, tag)

            # Restore the original stdout, then make a linked low-pressure sample.
            stdout.write_text("".join(json.dumps(record, allow_nan=False) + "\n"
                                             for record in original_records))
            receipt = json.loads(receipt_path.read_text())
            receipt["stdout_sha256"] = results.sha(stdout)
            write_json(receipt_path, receipt)
            events = valid_guard_events()
            events[3]["pressure_free_pct"] = 14
            guard = evidence / f"probe-{tag}.guard.jsonl"
            guard.write_text("".join(json.dumps(event, sort_keys=True) + "\n"
                                              for event in events))
            receipt = json.loads(receipt_path.read_text())
            receipt["guard_sha256"] = results.sha(guard)
            write_json(receipt_path, receipt)
            with patch.object(results, "SOURCE_FIXTURE_PINS", source_pins):
                with self.assertRaisesRegex(ValueError, "pressure"):
                    results.read_probe(evidence, tag)

    def test_storage_accounting_rejects_wrong_sidecar_and_candidate_quota(self):
        expected = results.expected_storage("q4_k_delta4")
        results.validate_storage_accounting(expected, "q4_k_delta4")
        changed = dict(expected, sidecar_bytes=expected["sidecar_bytes"] - 4)
        with self.assertRaises(ValueError):
            results.validate_storage_accounting(changed, "q4_k_delta4")

    def test_metrics_require_full_count_bounded_counters_and_finite_values(self):
        metric = ZERO_METRIC(17)
        results.validate_metric(metric, 17, "complete")
        for mutation in (
            {"compared_values": 16},
            {"nonfinite": 18},
            {"relative_l2": float("nan")},
        ):
            changed = {**metric, **mutation}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                results.validate_metric(changed, 17, "mutated")

    def test_timing_accepts_balanced_rotations_and_rejects_order_wait_nan(self):
        records = timing_records()
        summary = results.validate_timing_records(
            records, {"rows": ROWS, "manifest_sha256": "a" * 64}, 16, False)
        self.assertEqual(summary["projection"]["arms"]["q4_k"]["gpu_seconds"]["count"], 16)
        self.assertEqual(len(summary["decode_only"]["paired_q4_k_over_candidate_gpu"]), 3)
        mutations = [
            lambda items: items[0].update(variant="invalid_arm"),
            lambda items: items[0].update(terminal_waits=2),
            lambda items: items[0].update(gpu_seconds=float("nan")),
            lambda items: items[0].update(encode_metal_buffer_allocations=1),
        ]
        for mutate in mutations:
            changed = timing_records()
            mutate(changed)
            with self.assertRaises(ValueError):
                results.validate_timing_records(
                    changed, {"rows": ROWS, "manifest_sha256": "a" * 64}, 16, False)

    def test_command_rejects_unrecognized_fourth_argument(self):
        command = ["/tmp/probe-v1", "--fixture=/tmp/fixture-r4", "--rounds=16",
                   "--unrecognized"]
        setup = {"reverse_first": False}
        with self.assertRaises(ValueError):
            results.validate_probe_command(command, False, setup)

    def test_source_binding_rejects_missing_or_changed_repo_pin(self):
        source = str(ROOT / "src/fake.cr")
        results.validate_source_binding({source: "b" * 64}, {source: "b" * 64})
        with self.assertRaises(ValueError):
            results.validate_source_binding({source: "b" * 64}, {})
        with self.assertRaises(ValueError):
            results.validate_source_binding({source: "b" * 64}, {source: "c" * 64})

    def test_linked_receipt_uses_archived_repo_source_but_keeps_external_pins_live(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "evidence"
            repo = Path(directory) / "checkout"
            source = repo / "src/kernel.cr"
            source_extra = repo / "src/other.cr"
            source_archive = base / "sources-build-v1/src/kernel.cr"
            source_extra_archive = base / "sources-build-v1/src/other.cr"
            partial_archive = base / "sources-build-v0/src/kernel.cr"
            native_object = Path(directory) / "build/bridge.o"
            fixture_manifest = base / "fixture-old/manifest.json"
            source.parent.mkdir(parents=True)
            source_archive.parent.mkdir(parents=True)
            source.write_text("current checkout source")
            source_extra.write_text("current checkout secondary source")
            source_archive.write_text("source bytes used by historical build")
            source_extra_archive.write_text("secondary source bytes used by historical build")
            partial_archive.parent.mkdir(parents=True)
            partial_archive.write_text("source bytes used by historical build")
            native_object.parent.mkdir(parents=True)
            native_object.write_bytes(b"historical native object")
            fixture_manifest.parent.mkdir(parents=True)
            fixture_manifest.write_bytes(b"historical fixture manifest")

            files = {
                str(source): results.sha(source_archive),
                str(source_extra): results.sha(source_extra_archive),
                str(native_object): results.sha(native_object),
                str(fixture_manifest): results.sha(fixture_manifest),
            }
            intent_path = base / "oracle-old.intent.json"
            receipt_path = base / "oracle-old.completion.json"
            write_json(intent_path, {
                "schema": "wan_q4k_delta_intent_v1",
                "mode": "oracle",
                "tag": "oracle-old",
                "files": files,
            })
            write_json(receipt_path, {
                "schema": "wan_q4k_delta_completion_v1",
                "returncode": 0,
                "changed_files": [],
                "intent_sha256": results.sha(intent_path),
                "sources": {str(source): files[str(source)],
                            str(source_extra): files[str(source_extra)]},
            })

            # A changed live source symlink must not redirect historical source
            # validation, and a partial snapshot must not satisfy the full pin set.
            live_source_target = Path(directory) / "changed-live-source.cr"
            live_source_target.write_text("mutated live source target")
            source.unlink()
            source.symlink_to(live_source_target)
            self.assertEqual(results._matching_source_archive(base, files, repo),
                             source_archive.parents[1])

            validated, _ = results._linked_receipt(
                base, receipt_path, intent_path, "oracle", "oracle-old",
                source_root=repo)
            self.assertEqual(validated["sources"], {
                str(source): files[str(source)],
                str(source_extra): files[str(source_extra)],
            })

            source_archive.write_text("mutated archived source")
            with self.assertRaisesRegex(ValueError, "No archived build source snapshot"):
                results._linked_receipt(
                    base, receipt_path, intent_path, "oracle", "oracle-old",
                    source_root=repo)
            source_archive.write_text("source bytes used by historical build")
            source_archive.unlink()
            source_extra_archive.unlink()
            with self.assertRaisesRegex(ValueError, "No archived build source snapshot"):
                results._linked_receipt(
                    base, receipt_path, intent_path, "oracle", "oracle-old",
                    source_root=repo)

            source_archive.write_text("source bytes used by historical build")
            source_extra_archive.write_text("secondary source bytes used by historical build")
            fixture_manifest.write_bytes(b"mutated fixture manifest")
            with self.assertRaisesRegex(ValueError, "Pinned file changed or missing"):
                results._linked_receipt(
                    base, receipt_path, intent_path, "oracle", "oracle-old",
                    source_root=repo)
            fixture_manifest.write_bytes(b"historical fixture manifest")
            native_object.write_bytes(b"mutated native object")
            with self.assertRaisesRegex(ValueError, "Pinned file changed or missing"):
                results._linked_receipt(
                    base, receipt_path, intent_path, "oracle", "oracle-old",
                    source_root=repo)

    def test_full_weight_quality_metric_is_required_on_pre_and_post_parity(self):
        valid = {
            "variant": "q4_k",
            "phase": "before_timing",
            "decoder": ZERO_METRIC(DIN * DOUT),
            "projection": ZERO_METRIC(ROWS * DOUT),
            "lossy_weight_vs_bf16": ZERO_METRIC(DIN * DOUT, nonfinite=0),
            "lossy_vs_bf16_per_source_row": [ZERO_METRIC(DOUT, nonfinite=0) for _ in range(4)],
            "calibration_rows": [],
            "unique_evaluation_rows": [0, 1, 2, 3],
        }
        records = [
            {**valid, "variant": arm, "phase": phase,
             "decoder": ZERO_METRIC(DIN * DOUT),
             "projection": ZERO_METRIC(ROWS * DOUT),
             "lossy_weight_vs_bf16": ZERO_METRIC(DIN * DOUT),
             "lossy_vs_bf16_per_source_row": [ZERO_METRIC(DOUT) for _ in range(4)]}
            for arm in results.ARMS for phase in ("before_timing", "after_timing")
        ]
        results.validate_parity(records, ROWS, DIN, DOUT)
        records[0]["lossy_weight_vs_bf16"]["nonfinite"] = 1
        with self.assertRaises(ValueError):
            results.validate_parity(records, ROWS, DIN, DOUT)
        records[0]["lossy_weight_vs_bf16"]["nonfinite"] = 0
        del records[0]["lossy_weight_vs_bf16"]
        with self.assertRaises(ValueError):
            results.validate_parity(records, ROWS, DIN, DOUT)

    def test_production_source_manifest_pin_is_retained(self):
        self.assertEqual(results.SOURCE_FIXTURE_PINS["manifest_sha256"],
                         "0dba2df508f72764837c23d8c5743a6d687e2d1b6104406e38325031084c3703")

    def test_pack_mutations_fail_closed(self):
        pack = {
            "variant": "q4_k_delta1",
            "sidecar_word_differences": 0,
            "encode_metal_buffer_allocations": 0,
            "activation_readbacks": 0,
            "terminal_waits": 1,
            "byte_validation_outside_packing_interval": True,
            "gpu_seconds": 0.1,
            "wall_seconds": 0.2,
            "encode_seconds": 0.01,
        }
        results.validate_pack_records([pack, {**pack, "variant": "q4_k_delta4"},
                                       {**pack, "variant": "q4_k_delta8"}])
        changed_pack = [{**pack, "sidecar_word_differences": 1},
                        {**pack, "variant": "q4_k_delta4"},
                        {**pack, "variant": "q4_k_delta8"}]
        with self.assertRaises(ValueError):
            results.validate_pack_records(changed_pack)

    def test_corruption_control_requires_exact_known_positive_signature(self):
        record = {
            "corruption": ZERO_METRIC(ROWS * DOUT, bit_differences=2,
                                      tolerance_violations=2, nonfinite=1),
            "corruption_rejected": True,
            "final_status_zero": True,
        }
        results.validate_corruption_control(record, ROWS * DOUT)
        record["corruption"]["bit_differences"] = 3
        with self.assertRaises(ValueError):
            results.validate_corruption_control(record, ROWS * DOUT)
        record["corruption"]["bit_differences"] = 2
        record["corruption"]["nonfinite"] = 0
        with self.assertRaises(ValueError):
            results.validate_corruption_control(record, ROWS * DOUT)

    def test_q4k_kernel_corruption_control_allows_rounding_differences_but_requires_injection(self):
        record = {
            "corruption": ZERO_METRIC(ROWS * DOUT, bit_differences=53_899,
                                      tolerance_violations=2, nonfinite=1),
            "corruption_rejected": True,
            "final_status_zero": True,
            "q4k_kernel_vs_staged_corruption_rejected": True,
            "q4k_kernel_vs_staged_corruption": ZERO_METRIC(
                ROWS * DOUT, bit_differences=53_899,
                tolerance_violations=2, nonfinite=1),
        }
        results.validate_corruption_control(
            record, ROWS * DOUT, q4k_kernel_comparison=True)

        for field, counter, value in (
                ("corruption", "bit_differences", 1),
                ("corruption", "tolerance_violations", 1),
                ("corruption", "nonfinite", 0),
                ("q4k_kernel_vs_staged_corruption", "bit_differences", 1),
                ("q4k_kernel_vs_staged_corruption", "tolerance_violations", 1),
                ("q4k_kernel_vs_staged_corruption", "nonfinite", 0)):
            with self.subTest(missing_perturbation=(field, counter)):
                corrupt = json.loads(json.dumps(record))
                corrupt[field][counter] = value
                with self.assertRaises(ValueError):
                    results.validate_corruption_control(
                        corrupt, ROWS * DOUT, q4k_kernel_comparison=True)

        corrupt_flag = {**record, "q4k_kernel_vs_staged_corruption_rejected": False}
        with self.assertRaises(ValueError):
            results.validate_corruption_control(
                corrupt_flag, ROWS * DOUT, q4k_kernel_comparison=True)

    def test_guard_accepts_policy_and_rejects_runtime_or_startup_breach(self):
        intent = {
            "memory_accounting": "file-cache",
            "reserve_pct": 15,
            "max_footprint_mib": 4096,
            "max_swap_growth_mib": 64,
            "timeout_sec": 300,
        }
        memory = results.validate_guard(valid_guard_events(), intent)
        self.assertEqual(memory["sampled_min_pressure_free_pct"], 80)
        events = valid_guard_events()
        events[3]["pressure_free_pct"] = 14
        with self.assertRaises(ValueError):
            results.validate_guard(events, intent)
        events = valid_guard_events()
        events[1]["headroom_bytes"] = 13 * 1024**3
        with self.assertRaises(ValueError):
            results.validate_guard(events, intent)


if __name__ == "__main__":
    unittest.main()
