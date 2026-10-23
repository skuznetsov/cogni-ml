#!/usr/bin/env python3
"""Verify and summarize immutable receipts for the Q4_K sparse-delta probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


ARMS = ("q4_k", "q4_k_delta1", "q4_k_delta4", "q4_k_delta8")
BLOCK_ARMS = tuple(f"{arm}_block" for arm in ARMS)
COMPARISON_ARMS = ARMS + BLOCK_ARMS
FUSED_ARMS = tuple(item for arm in ARMS for item in (arm, f"{arm}_fused"))
FUSED_CANDIDATE_ARMS = tuple(f"{arm}_fused" for arm in ARMS)
PATCHES_PER_BLOCK = {"q4_k": 0, "q4_k_delta1": 1, "q4_k_delta4": 4, "q4_k_delta8": 8}
MODES = ("projection", "decode_only")
FUSED_MODES = ("dense_mode", "fused_projection")
ROOT = Path(__file__).resolve().parents[2]
MIB = 1024 * 1024
PHYSICAL_RESERVE_PCT = 15
MAX_FOOTPRINT_BYTES = 4096 * MIB
MAX_SWAP_GROWTH_BYTES = 64 * MIB
BLOCK_VALUES = 256
BLOCK_COUNT = 276_480
WEIGHT_COUNT = BLOCK_VALUES * BLOCK_COUNT
BASE_PAYLOAD_BYTES = 39_813_120
DESCRIPTOR_BYTES = 16
DESCRIPTOR_MAGIC = 0x31425157
Q4_K_BLOCK_BYTES = 144
INPUT_DIM = 5120
OUTPUT_DIM = 13824
SOURCE_FIXTURE_PINS = {
    "manifest_sha256": "0dba2df508f72764837c23d8c5743a6d687e2d1b6104406e38325031084c3703",
    "checkpoint_weight_sha256": "cd1037a3eb9c98cf5baed570e6825e7bb2463165563707cf7f1c3b367afca557",
    "input_sha256": "2d1462e25bfc05523da2ac8a7487c21b08986ef3eb9fe2f0a2f9527c9b65618c",
    "bias_f32_sha256": "6d264e0db1ffc55271a30cc6c8e60501effb4385b4200c007dc7a9c54d549b2f",
    "q4_k_sha256": "418e4e461b5cc931b9af01e4f3c216f0da61e4fc02087c7deab425fe2e15d737",
}


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(2**20), b""):
            digest.update(data)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_number(value, label: str, *, positive: bool = False) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"Invalid numeric field: {label}")
    result = float(value)
    require(math.isfinite(result), f"Non-finite numeric field: {label}")
    require(result > 0 if positive else result >= 0, f"Out-of-range numeric field: {label}")
    return result


def summary(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Missing, non-finite or non-positive timing")
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values)}


def _finite_summary(values):
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Missing or non-finite paired timing summary")
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values)}


def expected_storage(arm: str) -> dict:
    patches = PATCHES_PER_BLOCK[arm]
    sidecar = BLOCK_COUNT * patches * 3
    package = BASE_PAYLOAD_BYTES + sidecar + DESCRIPTOR_BYTES
    return {
        "base_payload_bytes": BASE_PAYLOAD_BYTES,
        "sidecar_bytes": sidecar,
        "descriptor_bytes": DESCRIPTOR_BYTES,
        "package_bytes": package,
        "patches_per_block": patches,
        "correction_count": BLOCK_COUNT * patches,
        "bits_per_weight": package * 8 / WEIGHT_COUNT,
    }


def validate_storage_accounting(storage, label: str) -> dict:
    require(isinstance(storage, dict), f"Missing per-arm storage accounting: {label}")
    expected = expected_storage(label)
    for key in ("base_payload_bytes", "sidecar_bytes", "descriptor_bytes", "package_bytes",
                "patches_per_block", "correction_count"):
        require(type(storage.get(key)) is int and storage[key] == expected[key],
                f"Unexpected {key} storage accounting: {label}")
    bits = storage.get("bits_per_weight")
    require(isinstance(bits, (int, float)) and not isinstance(bits, bool) and
            math.isfinite(bits) and math.isclose(bits, expected["bits_per_weight"], rel_tol=0, abs_tol=1e-12),
            f"Unexpected bits-per-weight accounting: {label}")
    return expected


def validate_metric(metric, expected_count: int, label: str, *, strict: bool = True) -> None:
    require(isinstance(metric, dict), f"Missing metric object: {label}")
    require(type(metric.get("compared_values")) is int and
            metric["compared_values"] == expected_count,
            f"Incomplete metric coverage: {label}")
    for key in ("bit_differences", "tolerance_violations", "nonfinite"):
        require(type(metric.get(key)) is int and metric[key] >= 0,
                f"Invalid metric counter {key}: {label}")
        require(metric[key] <= expected_count,
                f"Metric counter exceeds compared element count: {label}")
    for key in ("max_abs", "max_ratio", "rmse", "relative_l2", "cosine"):
        value = metric.get(key)
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and
                math.isfinite(value), f"Invalid metric value {key}: {label}")
    for key in ("max_abs", "max_ratio", "rmse", "relative_l2"):
        require(metric[key] >= 0, f"Negative metric value {key}: {label}")
    require(-1.000001 <= metric["cosine"] <= 1.000001,
            f"Invalid cosine metric: {label}")
    if strict:
        require(metric["tolerance_violations"] == 0 and metric["nonfinite"] == 0,
                f"Strict finite parity failed: {label}")


def validate_parity(records, rows: int, din: int, dout: int,
                    execution_arms=ARMS, *, require_bit_exact: bool = False,
                    require_fused_staged: bool = False) -> None:
    expected = {(arm, phase) for arm in execution_arms
                for phase in ("before_timing", "after_timing")}
    actual = [(record.get("variant"), record.get("phase")) for record in records]
    require(len(actual) == len(expected) and set(actual) == expected,
            "Missing, duplicate or unexpected pre/post parity records")
    for record in records:
        arm = record["variant"]
        decoder = record.get("decoder")
        projection = record.get("projection")
        validate_metric(decoder, din * dout, f"{arm} decoder")
        validate_metric(projection, rows * dout, f"{arm} projection")
        if require_fused_staged and arm.endswith("_fused"):
            validate_metric(record.get("fused_vs_staged"), rows * dout,
                            f"{arm} fused/staged projection")
        elif require_fused_staged:
            require("fused_vs_staged" not in record,
                    f"Staged control unexpectedly reports fused/staged parity: {arm}")
        if require_bit_exact:
            require(decoder["bit_differences"] == 0 and
                    projection["bit_differences"] == 0,
                    f"Block-validation comparison requires bit-exact decoder and projection parity: {arm}")
        weight_loss = record.get("lossy_weight_vs_bf16")
        validate_metric(weight_loss, din * dout, f"{arm} full-weight BF16 comparison", strict=False)
        require(weight_loss["nonfinite"] == 0,
                f"Non-finite full-weight BF16 loss metric: {arm}")
        row_metrics = record.get("lossy_vs_bf16_per_source_row")
        require(isinstance(row_metrics, list) and len(row_metrics) == 4,
                f"Incomplete per-source-row BF16 comparison: {arm}")
        for index, metric in enumerate(row_metrics):
            validate_metric(metric, dout, f"{arm} BF16 source row {index}", strict=False)
            require(metric["nonfinite"] == 0, f"Non-finite BF16 loss metric: {arm} row {index}")
        require(record.get("calibration_rows") == [] and
                record.get("unique_evaluation_rows") == [0, 1, 2, 3],
                f"Unexpected calibration/evaluation scope: {arm}")


def validate_dense_parity(records, rows: int, din: int, dout: int) -> dict:
    expected_phases = {"before_timing", "after_timing"}
    require(len(records) == 4 and
            {(record.get("phase"), record.get("variant")) for record in records} ==
            {(phase, arm) for phase in expected_phases
             for arm in ("dense_native", "dense_mps")},
            "Missing, duplicate or unexpected dense-mode parity records")
    summaries = {}
    for record in records:
        phase, arm = record["phase"], record["variant"]
        decoder = record.get("decoder")
        validate_metric(decoder, din * dout, f"dense_mode {phase} Q4_K decoder")
        require(record.get("format") == "q4_k",
                "Dense mode must use the Q4_K-decoded coefficient matrix")
        native = record.get("dense_native")
        mps = record.get("dense_mps")
        cross = record.get("dense_native_vs_dense_mps")
        validate_metric(native, rows * dout, f"dense_mode {phase} native projection")
        validate_metric(mps, rows * dout, f"dense_mode {phase} MPS projection")
        validate_metric(cross, rows * dout, f"dense_mode {phase} native/MPS parity")
        require(record.get("reference_arm") == "q4_k" and
                record.get("variant") in ("dense_native", "dense_mps"),
                "Unexpected dense-mode reference mapping")
        summaries[f"{phase}:{arm}"] = {
            "decoder": decoder,
            "dense_native": native,
            "dense_mps": mps,
            "dense_native_vs_dense_mps": cross,
        }
    return summaries


def _rotated_order(rotation: int, reverse: bool) -> tuple[str, ...]:
    order = ARMS[rotation:] + ARMS[:rotation]
    return tuple(reversed(order)) if reverse else order


def _comparison_order(round_id: int, rounds: int, reverse_first: bool) -> tuple[str, ...]:
    rotation = max(round_id, 0) % len(ARMS)
    base_order = ARMS[rotation:] + ARMS[:rotation]
    order = [item for arm in base_order for item in (arm, f"{arm}_block")]
    if round_id >= 0 and (round_id // 4) % 2 == 1:
        order = [item for index in range(0, len(order), 2)
                 for item in (order[index + 1], order[index])]
    reverse = ((round_id == -1) if round_id < 0 else round_id >= rounds // 2)
    if reverse != reverse_first:
        order.reverse()
    return tuple(order)


def _fused_order(round_id: int, rounds: int, reverse_first: bool) -> tuple[str, ...]:
    rotation = max(round_id, 0) % len(ARMS)
    base_order = ARMS[rotation:] + ARMS[:rotation]
    order = [item for arm in base_order for item in (arm, f"{arm}_fused")]
    if round_id >= 0 and (round_id // len(ARMS)) % 2 == 1:
        order = [item for index in range(0, len(order), 2)
                 for item in (order[index + 1], order[index])]
    reverse = ((round_id == -1) if round_id < 0 else round_id >= rounds // 2)
    if reverse != reverse_first:
        order.reverse()
    return tuple(order)


def _dense_order(round_id: int, rounds: int, reverse_first: bool) -> tuple[str, ...]:
    if round_id < 0:
        order = ("dense_native", "dense_mps") if round_id == -2 else ("dense_mps", "dense_native")
        return tuple(reversed(order)) if reverse_first else order
    abba = (round_id < rounds // 2) != reverse_first
    if abba:
        return ("dense_native", "dense_mps", "dense_mps", "dense_native")
    return ("dense_mps", "dense_native", "dense_native", "dense_mps")


def _validate_fused_timing_records(records, setup: dict, rounds: int,
                                   reverse_first: bool) -> dict:
    require(rounds in (16, 32), "Unsupported measured-round count")
    require(type(reverse_first) is bool, "Invalid reverse-first flag")
    require(setup.get("execution_arms") == list(FUSED_ARMS) and
            setup.get("dense_mode_execution_arms") == ["dense_native", "dense_mps"],
            "Unexpected fused or dense execution arms")
    expected_count = (rounds + 2) * len(FUSED_ARMS) + 4 + rounds * 4
    require(len(records) == expected_count,
            "Missing or extra fused/dense timing records, including warmups")
    by_mode_round = {}
    for record in records:
        mode = record.get("mode")
        require(mode in FUSED_MODES, f"Unexpected fused timing mode: {mode}")
        require(record.get("manifest_sha256") == setup["manifest_sha256"],
                "Timing provenance changed")
        require(record.get("rows") == setup["rows"], "Timing geometry changed")
        require(type(record.get("terminal_waits")) is int and record["terminal_waits"] == 1 and
                type(record.get("inter_operator_host_waits")) is int and
                record["inter_operator_host_waits"] == 0 and
                type(record.get("activation_readbacks")) is int and
                record["activation_readbacks"] == 0 and
                type(record.get("encode_metal_buffer_allocations")) is int and
                record["encode_metal_buffer_allocations"] == 0,
                "Command-chain constraints failed")
        round_id = record.get("round")
        position = record.get("position")
        require(type(round_id) is int and -2 <= round_id < rounds,
                "Invalid timing round")
        expected_size = (2 if round_id < 0 else 4) if mode == "dense_mode" else len(FUSED_ARMS)
        require(type(position) is int and 0 <= position < expected_size,
                "Invalid timing position")
        require(type(record.get("warmup")) is bool and record["warmup"] == (round_id < 0),
                "Warmup flag/round mismatch")
        for key in ("gpu_seconds", "wall_seconds", "encode_seconds"):
            _finite_number(record.get(key), f"{mode} {key}", positive=round_id >= 0)
        bucket = by_mode_round.setdefault((mode, round_id), {})
        require(position not in bucket, "Duplicate timing position")
        bucket[position] = record

    for mode in FUSED_MODES:
        for round_id in (-2, -1, *range(rounds)):
            bucket = by_mode_round.get((mode, round_id))
            expected_size = ((2 if round_id < 0 else 4) if mode == "dense_mode"
                             else len(FUSED_ARMS))
            require(bucket is not None and set(bucket) == set(range(expected_size)),
                    f"Incomplete timing round: {mode} {round_id}")
            expected_order = (_dense_order(round_id, rounds, reverse_first)
                              if mode == "dense_mode" else
                              _fused_order(round_id, rounds, reverse_first))
            actual_order = tuple(bucket[position]["variant"] for position in range(expected_size))
            require(actual_order == expected_order,
                    f"Unexpected arm order: {mode} {round_id}: {actual_order}")

    summaries = {}
    dense_by_round = {round_id: by_mode_round[("dense_mode", round_id)]
                      for round_id in range(rounds)}
    dense_ratios = {}
    dense_pairs = []
    dense_samples = [record for (mode, round_id), bucket in by_mode_round.items()
                     if mode == "dense_mode" and round_id >= 0
                     for record in bucket.values()]
    for round_id, bucket in dense_by_round.items():
        native_values = [bucket[position]["gpu_seconds"] for position in bucket
                         if bucket[position]["variant"] == "dense_native"]
        mps_values = [bucket[position]["gpu_seconds"] for position in bucket
                      if bucket[position]["variant"] == "dense_mps"]
        require(len(native_values) == len(mps_values) == 2,
                "Dense paired round must contain two native and two MPS samples")
        native_mean = statistics.mean(native_values)
        mps_mean = statistics.mean(mps_values)
        ratio = native_mean / mps_mean
        dense_ratios[round_id] = ratio
        dense_pairs.append({"round": round_id,
                            "order": [bucket[position]["variant"] for position in range(4)],
                            "native_mean_gpu_seconds": native_mean,
                            "mps_mean_gpu_seconds": mps_mean,
                            "native_over_mps_gpu_ratio": ratio})
    dense_summaries = {
        "arms": {arm: {metric: summary([record[metric] * 1000 for record in dense_samples
                                        if record["variant"] == arm])
                        for metric in ("gpu_seconds", "wall_seconds", "encode_seconds")}
                  for arm in ("dense_native", "dense_mps")},
        "paired_dense_native_over_mps_gpu": {
            "all_rounds": summary(list(dense_ratios.values())),
            "by_half": {
                half: summary([ratio for round_id, ratio in dense_ratios.items()
                               if (round_id < rounds // 2) == (half == "first")])
                for half in ("first", "second")
            },
            "by_order": {
                order: summary([pair["native_over_mps_gpu_ratio"] for pair in dense_pairs
                               if ("ABBA" if pair["order"] == ["dense_native", "dense_mps", "dense_mps", "dense_native"]
                                   else "BAAB") == order])
                for order in ("ABBA", "BAAB")
            },
            "rounds": dense_pairs,
        },
        "speed_winner_selected": False,
    }

    fused_by_round = {round_id: by_mode_round[("fused_projection", round_id)]
                      for round_id in range(rounds)}
    fused_samples = [record for (mode, round_id), bucket in by_mode_round.items()
                     if mode == "fused_projection" and round_id >= 0
                     for record in bucket.values()]
    paired_fused = {}
    for arm in ARMS:
        candidate = f"{arm}_fused"
        round_ratios = {}
        pair_details = []
        for round_id, bucket in fused_by_round.items():
            control = next(record for record in bucket.values() if record["variant"] == arm)
            fused = next(record for record in bucket.values() if record["variant"] == candidate)
            require(abs(control["position"] - fused["position"]) == 1,
                    f"Fused/staged pair is not adjacent: {arm} round {round_id}")
            ratio = control["gpu_seconds"] / fused["gpu_seconds"]
            round_ratios[round_id] = ratio
            pair_details.append({"round": round_id, "control": control,
                                 "fused": fused, "control_over_fused_gpu_ratio": ratio})
        by_rotation = {
            str(rotation): summary([ratio for round_id, ratio in round_ratios.items()
                                    if round_id % len(ARMS) == rotation])
            for rotation in range(len(ARMS))
        }
        paired_fused[arm] = {
            "candidate_arm": candidate,
            "staged_control_over_fused_gpu": {
                "all_rounds": summary(list(round_ratios.values())),
                "by_rotation": by_rotation,
                "rounds": pair_details,
            },
        }
    fused_summaries = {
        "arms": {arm: {metric: summary([record[metric] * 1000 for record in fused_samples
                                        if record["variant"] == arm])
                        for metric in ("gpu_seconds", "wall_seconds", "encode_seconds")}
                  for arm in FUSED_ARMS},
        "paired_staged_control_over_fused_gpu": paired_fused,
        "speed_winner_selected": False,
    }
    return {"dense_mode": dense_summaries, "fused_projection": fused_summaries}


def expected_probe_record_count(timing_count: int,
                                block_validation_comparison: bool | None = None,
                                fused_comparison: bool | None = None) -> int:
    """Count setup, three pack records, BF16 control, parity, timing and control."""
    if fused_comparison is None:
        fused_counts = {
            (rounds + 2) * len(FUSED_ARMS) + 4 + rounds * 4: rounds
            for rounds in (16, 32)
        }
        if timing_count in fused_counts:
            fused_comparison = True
        else:
            fused_comparison = False
    require(type(fused_comparison) is bool, "Invalid fused comparison flag")
    require(not (fused_comparison and block_validation_comparison is True),
            "Fused and block-validation comparisons are mutually exclusive")
    if fused_comparison:
        require(timing_count in {
            (rounds + 2) * len(FUSED_ARMS) + 4 + rounds * 4
            for rounds in (16, 32)
        }, "Unexpected fused timing record count")
        arm_count = len(FUSED_ARMS)
        dense_parity_count = 4
    elif block_validation_comparison is None:
        counts = {
            (rounds + 2) * arm_count * len(MODES): arm_count
            for rounds in (16, 32) for arm_count in (len(ARMS), len(COMPARISON_ARMS))
        }
        matching_counts = [count for observed, count in counts.items()
                           if observed == timing_count]
        require(len(matching_counts) == 1,
                "Cannot infer execution-arm count from timing record count")
        arm_count = matching_counts[0]
        dense_parity_count = 0
    else:
        require(type(block_validation_comparison) is bool,
                "Invalid block-validation comparison flag")
        arm_count = len(COMPARISON_ARMS) if block_validation_comparison else len(ARMS)
        dense_parity_count = 0
    return 1 + len(ARMS[1:]) + 1 + arm_count * 2 + dense_parity_count + timing_count + 1


def validate_corruption_control(record: dict, expected_count: int, *, fused_comparison: bool = False) -> None:
    metric = record.get("corruption")
    validate_metric(metric, expected_count, "positive corruption control", strict=False)
    valid_primary_bits = (metric["bit_differences"] >= 2 if fused_comparison else
                          metric["bit_differences"] == 2)
    require(record.get("corruption_rejected") is True and
            record.get("final_status_zero") is True and
            valid_primary_bits and
            metric["tolerance_violations"] == 2 and
            metric["nonfinite"] == 1,
            "Positive corruption/NaN control did not detect the expected failures")
    if fused_comparison:
        staged = record.get("fused_vs_staged_corruption")
        validate_metric(staged, expected_count, "fused/staged corruption control", strict=False)
        require(record.get("fused_vs_staged_corruption_rejected") is True and
                staged["bit_differences"] >= 2 and
                staged["tolerance_violations"] == 2 and
                staged["nonfinite"] == 1,
                "Positive fused/staged corruption control did not detect the expected failures")


def validate_pack_records(records) -> list[dict]:
    require(isinstance(records, list) and
            sorted(record.get("variant") for record in records if isinstance(record, dict)) ==
            sorted(ARMS[1:]) and len(records) == len(ARMS) - 1,
            "Missing, duplicate or unexpected independent pack comparisons")
    for record in records:
        require(type(record.get("sidecar_word_differences")) is int and
                record["sidecar_word_differences"] == 0 and
                type(record.get("encode_metal_buffer_allocations")) is int and
                record["encode_metal_buffer_allocations"] == 0 and
                type(record.get("activation_readbacks")) is int and
                record["activation_readbacks"] == 0 and
                type(record.get("terminal_waits")) is int and record["terminal_waits"] == 1 and
                record.get("byte_validation_outside_packing_interval") is True,
                "Independent GPU pack validation failed")
        for metric in ("gpu_seconds", "wall_seconds", "encode_seconds"):
            _finite_number(record.get(metric), f"pack {metric}")
    return records


def validate_probe_command(command, reverse_first, setup,
                           intent_block_validation_comparison=None,
                           intent_fused_comparison=None) -> tuple[Path, Path, int, bool]:
    require(isinstance(command, list) and 3 <= len(command) <= 5,
            "Unexpected probe command")
    binary = Path(command[0])
    require(binary.is_absolute() and binary.name.startswith("probe-"),
            "Unexpected probe binary path")
    fixture_arg, rounds_arg = command[1], command[2]
    require(isinstance(fixture_arg, str) and fixture_arg.startswith("--fixture="),
            "Probe command lacks fixture argument")
    fixture = Path(fixture_arg.removeprefix("--fixture="))
    require(fixture.is_absolute() and fixture.name.startswith("fixture-"),
            "Unexpected probe fixture path")
    require(isinstance(rounds_arg, str) and rounds_arg.startswith("--rounds="),
            "Probe command lacks rounds argument")
    rounds_text = rounds_arg.removeprefix("--rounds=")
    require(rounds_text in ("16", "32"), "Unexpected probe round count")
    rounds = int(rounds_text)
    setup_comparison = setup.get("block_validation_comparison", False)
    setup_fused = setup.get("fused_comparison", False)
    require(type(setup_comparison) is bool,
            "Invalid block-validation comparison setup flag")
    require(type(setup_fused) is bool,
            "Invalid fused comparison setup flag")
    require(not (setup_comparison and setup_fused),
            "Fused and block-validation comparisons are mutually exclusive")
    if intent_block_validation_comparison is None:
        intent_comparison = setup_comparison
    else:
        require(type(intent_block_validation_comparison) is bool,
                "Invalid block-validation comparison intent flag")
        intent_comparison = intent_block_validation_comparison
    require(intent_comparison is setup_comparison,
            "block-validation comparison flag differs between intent and setup")
    if intent_fused_comparison is None:
        intent_fused = setup_fused
    else:
        require(type(intent_fused_comparison) is bool,
                "Invalid fused comparison intent flag")
        intent_fused = intent_fused_comparison
    require(intent_fused is setup_fused,
            "fused comparison flag differs between intent and setup")

    if setup_fused:
        require(len(command) in (4, 5) and command[3] == "--compare-fused",
                "Probe command fused comparison flag disagrees with setup/intent")
        expected_reverse = len(command) == 5
        if expected_reverse:
            require(command[4] == "--reverse-first", "Unexpected probe command flag order")
    elif setup_comparison:
        require(len(command) in (4, 5) and command[3] == "--compare-block-validation",
                "Probe command block-validation comparison flag disagrees with setup/intent")
        expected_reverse = len(command) == 5
        if expected_reverse:
            require(command[4] == "--reverse-first", "Unexpected probe command flag order")
    else:
        require(len(command) in (3, 4) and
                (len(command) == 3 or command[3] == "--reverse-first"),
                "Unexpected probe command")
        expected_reverse = len(command) == 4
    require(type(reverse_first) is bool and reverse_first == expected_reverse,
            "Probe reverse-order intent differs from command")
    require(setup.get("reverse_first") is reverse_first,
            "Setup arm-order declaration differs from the pinned probe intent")
    return binary, fixture, rounds, reverse_first


def validate_timing_records(records, setup: dict, rounds: int, reverse_first: bool):
    require(rounds in (16, 32), "Unsupported measured-round count")
    require(type(reverse_first) is bool, "Invalid reverse-first flag")
    fused = setup.get("fused_comparison", False)
    require(type(fused) is bool, "Invalid fused comparison timing flag")
    if fused:
        require(setup.get("block_validation_comparison", False) is False,
                "Fused and block-validation comparisons are mutually exclusive")
        return _validate_fused_timing_records(records, setup, rounds, reverse_first)
    rows = setup["rows"]
    comparison = setup.get("block_validation_comparison", False)
    require(type(comparison) is bool, "Invalid block-validation comparison setup flag")
    execution_arms = COMPARISON_ARMS if comparison else ARMS
    if "execution_arms" in setup:
        require(setup["execution_arms"] == list(execution_arms),
                "Unexpected execution arm declaration")
    expected_records_per_mode = (rounds + 2) * len(execution_arms)
    require(len(records) == expected_records_per_mode * len(MODES),
            "Missing or extra timing records, including warmups")
    by_mode_round = {}
    for record in records:
        mode = record.get("mode")
        require(mode in MODES, f"Unexpected timing mode: {mode}")
        require(record.get("manifest_sha256") == setup["manifest_sha256"],
                "Timing provenance changed")
        require(record.get("rows") == rows, "Timing geometry changed")
        require(type(record.get("terminal_waits")) is int and record["terminal_waits"] == 1 and
                type(record.get("inter_operator_host_waits")) is int and
                record["inter_operator_host_waits"] == 0 and
                type(record.get("activation_readbacks")) is int and
                record["activation_readbacks"] == 0 and
                type(record.get("encode_metal_buffer_allocations")) is int and
                record["encode_metal_buffer_allocations"] == 0,
                "Command-chain constraints failed")
        round_id = record.get("round")
        position = record.get("position")
        require(type(round_id) is int and -2 <= round_id < rounds,
                "Invalid timing round")
        require(type(position) is int and 0 <= position < len(execution_arms),
                "Invalid timing position")
        require(type(record.get("warmup")) is bool and record["warmup"] == (round_id < 0),
                "Warmup flag/round mismatch")
        for key in ("gpu_seconds", "wall_seconds", "encode_seconds"):
            _finite_number(record.get(key), f"{mode} {key}", positive=round_id >= 0)
        key = (mode, round_id)
        bucket = by_mode_round.setdefault(key, {})
        require(position not in bucket, "Duplicate timing position")
        bucket[position] = record

    for mode in MODES:
        for round_id in (-2, -1, *range(rounds)):
            bucket = by_mode_round.get((mode, round_id))
            require(bucket is not None and set(bucket) == set(range(len(execution_arms))),
                    f"Incomplete timing round: {mode} {round_id}")
            if comparison:
                expected_order = _comparison_order(round_id, rounds, reverse_first)
            else:
                if round_id < 0:
                    reverse = (round_id == -1) != reverse_first
                else:
                    reverse = (round_id >= rounds // 2) != reverse_first
                expected_order = _rotated_order(max(round_id, 0) % len(ARMS), reverse)
            actual_order = tuple(bucket[position]["variant"]
                                 for position in range(len(execution_arms)))
            require(actual_order == expected_order,
                    f"Unexpected arm order: {mode} {round_id}: {actual_order}")

    timings = {}
    for mode in MODES:
        measured = [record for (candidate_mode, round_id), bucket in by_mode_round.items()
                    if candidate_mode == mode and round_id >= 0
                    for record in bucket.values()]
        by_round = {round_id: {bucket[position]["variant"]: bucket[position]
                              for position in range(len(execution_arms))}
                    for (candidate_mode, round_id), bucket in by_mode_round.items()
                    if candidate_mode == mode and round_id >= 0}
        paired = {}
        rotation_medians = {}
        for arm in ARMS[1:]:
            ratios = {round_id: by_round[round_id][ARMS[0]]["gpu_seconds"] /
                      by_round[round_id][arm]["gpu_seconds"]
                      for round_id in range(rounds)}
            by_rotation = {
                str(rotation): summary([ratios[round_id] for round_id in range(rounds)
                                        if round_id % len(ARMS) == rotation])
                for rotation in range(len(ARMS))
            }
            medians = [item["median"] for item in by_rotation.values()]
            paired[f"q4_k_over_{arm}_gpu"] = {
                "all_rounds": summary(list(ratios.values())),
                "by_rotation": by_rotation,
                "rotation_median_range": {"min": min(medians), "max": max(medians)},
            }
            rotation_medians[arm] = medians
        timings[mode] = {
            "arms": {arm: {metric: summary([record[metric] * 1000 for record in measured
                                           if record["variant"] == arm])
                            for metric in ("gpu_seconds", "wall_seconds", "encode_seconds")}
                     for arm in execution_arms},
            "paired_q4_k_over_candidate_gpu": paired,
            "speed_winner_selected": False,
        }
        if comparison:
            block_paired = {}
            for arm in ARMS:
                paired_records = []
                for round_id in range(rounds):
                    old = by_round[round_id][arm]
                    block = by_round[round_id][f"{arm}_block"]
                    paired_records.append({
                        "round": round_id,
                        "rotation": round_id % len(ARMS),
                        "half": "first" if round_id < rounds // 2 else "second",
                        "old_position": old["position"],
                        "old": old,
                        "block": block,
                        "gpu_ratio": old["gpu_seconds"] / block["gpu_seconds"],
                    })
                ratios = [item["gpu_ratio"] for item in paired_records]
                by_rotation = {
                    str(rotation): summary([item["gpu_ratio"] for item in paired_records
                                            if item["rotation"] == rotation])
                    for rotation in range(len(ARMS))
                }
                rotation_medians = [item["median"] for item in by_rotation.values()]
                by_old_position = {
                    str(position): summary([item["gpu_ratio"] for item in paired_records
                                            if item["old_position"] == position])
                    for position in sorted({item["old_position"] for item in paired_records})
                }
                by_half = {
                    half: summary([item["gpu_ratio"] for item in paired_records
                                   if item["half"] == half])
                    for half in ("first", "second")
                }
                metric_changes = {}
                for metric in ("gpu_seconds", "wall_seconds", "encode_seconds"):
                    changes = [100.0 * (item["block"][metric] - item["old"][metric]) /
                               item["old"][metric] for item in paired_records]
                    metric_changes[metric] = {
                        "definition": "100 * (block_validation - per_value) / per_value",
                        "all_rounds": _finite_summary(changes),
                        "by_old_position": {
                            str(position): _finite_summary([
                                changes[index] for index, item in enumerate(paired_records)
                                if item["old_position"] == position])
                            for position in sorted({item["old_position"]
                                                    for item in paired_records})
                        },
                        "by_rotation": {
                            str(rotation): _finite_summary([
                                changes[index] for index, item in enumerate(paired_records)
                                if item["rotation"] == rotation])
                            for rotation in range(len(ARMS))
                        },
                        "by_half": {
                            half: _finite_summary([
                                changes[index] for index, item in enumerate(paired_records)
                                if item["half"] == half])
                            for half in ("first", "second")
                        },
                    }
                block_paired[arm] = {
                    "aa_drift_control": arm == "q4_k",
                    "old_over_block_gpu_ratio": {
                        "all_rounds": summary(ratios),
                        "by_rotation": by_rotation,
                        "by_old_position": by_old_position,
                        "by_half": by_half,
                        "rotation_median_range": {
                            "min": min(rotation_medians), "max": max(rotation_medians),
                        },
                    },
                    "paired_percent_change": metric_changes,
                }
            timings[mode]["paired_block_validation"] = block_paired
    return timings


def validate_guard(events, intent: dict) -> dict:
    limits = [event for event in events if event.get("event") == "limits"]
    require(len(limits) == 1 and events[0] is limits[0], "Missing or misplaced guard limits record")
    limits = limits[0]
    expected_limits = {
        "runtime_floor_pct": PHYSICAL_RESERVE_PCT,
        "max_footprint_bytes": MAX_FOOTPRINT_BYTES,
        "max_swap_growth_bytes": MAX_SWAP_GROWTH_BYTES,
        "timeout_sec": 300,
        "memory_accounting": "file-cache",
        "headroom_formula": "printed free + file-backed (includes speculative)",
        "startup_rule": "memory_pressure -Q >= 15% and selected headroom >= 15% physical + footprint budget",
    }
    for key, value in expected_limits.items():
        require(limits.get(key) == value, f"Unexpected guard limit: {key}")
    require(intent.get("memory_accounting") == "file-cache" and
            intent.get("reserve_pct") == PHYSICAL_RESERVE_PCT and
            intent.get("max_footprint_mib") == 4096 and
            intent.get("max_swap_growth_mib") == 64 and
            intent.get("timeout_sec") == 300,
            "Probe intent does not record the required guard policy")
    forbidden = {"blocked", "stopping", "cleanup_failed", "cleanup_incomplete", "launch_failed"}
    require(not any(event.get("event") in forbidden for event in events),
            "Memory guard rejected, stopped or failed to clean up the probe")
    finished = [event for event in events if event.get("event") == "finished"]
    started = [event for event in events if event.get("event") == "started"]
    require(len(finished) == 1 and type(finished[0].get("returncode")) is int and
            finished[0]["returncode"] == 0,
            "Missing successful guard completion")
    samples = [event for event in events if event.get("event") == "sample"]
    require(len(samples) >= 2, "Missing startup/runtime memory samples")
    require(len(started) == 1 and events.index(samples[0]) < events.index(started[0]) <
            events.index(finished[0]) and
            max(events.index(sample) for sample in samples) < events.index(finished[0]),
            "Guard startup/runtime sample lifecycle is incomplete")
    startup_swap = samples[0].get("swap_used_bytes")
    require(type(startup_swap) is int and startup_swap >= 0, "Invalid startup swap sample")
    for index, sample in enumerate(samples):
        pressure = sample.get("pressure_free_pct")
        physical = sample.get("physical_bytes")
        headroom = sample.get("headroom_bytes")
        footprint = sample.get("group_footprint_bytes")
        swap = sample.get("swap_used_bytes")
        require(type(pressure) is int and pressure >= PHYSICAL_RESERVE_PCT,
                "Observed pressure crossed the 15% floor")
        require(type(physical) is int and physical > 0 and type(headroom) is int and
                headroom >= (physical * PHYSICAL_RESERVE_PCT + 99) // 100,
                "Observed cache-aware headroom crossed the 15% floor")
        require(type(footprint) is int and 0 <= footprint <= MAX_FOOTPRINT_BYTES,
                "Observed process-group footprint exceeded 4 GiB")
        require(type(swap) is int and swap >= 0 and
                max(0, swap - startup_swap) <= MAX_SWAP_GROWTH_BYTES,
                "Observed swap growth exceeded 64 MiB")
        if index == 0:
            required_startup = ((physical * PHYSICAL_RESERVE_PCT + 99) // 100) + MAX_FOOTPRINT_BYTES
            require(headroom >= required_startup,
                    "Startup headroom was below 15% physical memory plus the footprint budget")
    return {"sampled_peak_footprint_bytes": max(event["group_footprint_bytes"] for event in samples),
            "sampled_min_headroom_bytes": min(event["headroom_bytes"] for event in samples),
            "sampled_min_pressure_free_pct": min(event["pressure_free_pct"] for event in samples),
            "sampled_max_swap_growth_bytes": max(0, max(event["swap_used_bytes"] for event in samples)
                                                    - startup_swap)}


def _pin_for(files: dict, target: Path) -> tuple[str, str]:
    resolved = target.resolve()
    matches = [(name, digest) for name, digest in files.items()
               if Path(name).resolve() == resolved]
    require(len(matches) == 1, f"Missing or ambiguous file pin: {target}")
    return matches[0]


def _verify_manifest_file(fixture: Path, files: dict, record, label: str,
                          expected_bytes: int | None = None,
                          expected_dtype: str | None = None,
                          expected_shape: list[int] | None = None) -> tuple[Path, str]:
    require(isinstance(record, dict), f"Missing manifest file record: {label}")
    rel = record.get("path")
    digest = record.get("sha256")
    size = record.get("bytes")
    dtype = record.get("dtype")
    shape = record.get("shape")
    require(isinstance(rel, str) and rel and not Path(rel).is_absolute(),
            f"Invalid manifest file path: {label}")
    target = (fixture / rel).resolve()
    require(target.is_relative_to(fixture.resolve()), f"Manifest file escapes fixture: {label}")
    require(type(size) is int and size >= 0 and
            (expected_bytes is None or size == expected_bytes),
            f"Unexpected manifest file size: {label}")
    require(isinstance(dtype, str) and dtype and
            (expected_dtype is None or dtype == expected_dtype),
            f"Unexpected manifest file dtype: {label}")
    require(isinstance(shape, list) and all(type(dimension) is int and dimension >= 0
                                            for dimension in shape) and
            (expected_shape is None or shape == expected_shape),
            f"Unexpected manifest file shape: {label}")
    require(isinstance(digest, str) and len(digest) == 64 and
            all(character in "0123456789abcdef" for character in digest),
            f"Invalid manifest file digest: {label}")
    require(target.is_file() and target.stat().st_size == size and sha(target) == digest,
            f"Manifest file bytes differ from their record: {label}")
    _, pinned_digest = _pin_for(files, target)
    require(pinned_digest == digest, f"Manifest file differs from probe intent pin: {label}")
    return target, digest


def _validate_reference_manifest(fixture: Path, files: dict, setup: dict) -> dict:
    manifest_path = fixture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema") == "wan_q4k_delta_reference_v1",
            "Unsupported reference manifest schema")
    geometry = manifest.get("geometry")
    require(isinstance(geometry, dict) and geometry.get("rows") == setup.get("rows") and
            geometry.get("source_rows") == 4 and geometry.get("input_dim") == INPUT_DIM and
            geometry.get("output_dim") == OUTPUT_DIM and
            geometry.get("block_values") == BLOCK_VALUES and
            geometry.get("blocks_per_output_row") == INPUT_DIM // BLOCK_VALUES and
            geometry.get("block_count") == BLOCK_COUNT and
            geometry.get("input_layout") == "row_major periodic source-row tile",
            "Reference manifest geometry differs from the selected projection")
    require(setup.get("rows") in (4, 4480) and setup.get("din") == 5120 and
            setup.get("dout") == 13824 and setup.get("unique_source_rows") == 4 and
            setup.get("real_video_derived") is False,
            "Unexpected projection geometry or source-row scope")
    comparison = manifest.get("comparison")
    require(isinstance(comparison, dict) and comparison.get("dtype") == "F32" and
            comparison.get("atol") == 2e-5 and comparison.get("rtol") == 2e-5 and
            comparison.get("finite") is True and
            comparison.get("predicate") ==
            "finite(actual) && abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)",
            "Reference manifest numerical gate changed")
    selection = manifest.get("selection")
    require(isinstance(selection, dict) and
            selection.get("score") ==
            "abs(F32(original BF16 weight) - F32(canonical Q4_K decoded base))" and
            selection.get("scope") == "independent contiguous 256-value Q4_K block" and
            selection.get("quota_per_block") is True and
            selection.get("rank_order") ==
            "descending absolute residual; equal residuals use ascending local position" and
            selection.get("wire_order") ==
            "selected positions remain in descending residual rank order" and
            selection.get("delta") == "F16 round-to-nearest-ties-to-even of F32(original minus base)" and
            selection.get("reconstruction") ==
            "separate F32(base + F32(F16 delta)) operation" and
            selection.get("global_optimality_claim") is False and
            selection.get("quality_scope") ==
            "the 4 unique seeded projection rows only; rows=4480 periodically tiles those same 4 rows",
            "Reference manifest delta selection contract changed")
    oracle = manifest.get("oracle")
    require(isinstance(oracle, dict) and oracle.get("device") == "MPS" and
            oracle.get("fallback_disabled") is True and
            oracle.get("selection_packing_reconstruction_device") == "MPS",
            "Reference manifest oracle provenance changed")

    variants = manifest.get("variants")
    require(isinstance(variants, dict) and set(variants) == set(ARMS),
            "Reference manifest arm set mismatch")
    setup_storage = setup.get("per_arm_storage")
    require(isinstance(setup_storage, dict) and set(setup_storage) == set(ARMS),
            "Setup is missing per-arm storage accounting")
    require(setup.get("shared_bias_bytes") == OUTPUT_DIM * 4,
            "Shared bias storage accounting changed")
    input_record = manifest.get("input")
    bias_record = manifest.get("bias")
    original_record = manifest.get("original_weight")
    _verify_manifest_file(fixture, files, input_record, "input",
                          4 * INPUT_DIM * 4, "F32", [4, INPUT_DIM])
    _verify_manifest_file(fixture, files, bias_record, "bias", OUTPUT_DIM * 4,
                          "F32", [OUTPUT_DIM])
    _verify_manifest_file(fixture, files, original_record, "original BF16 weight",
                          WEIGHT_COUNT * 2, "BF16", [OUTPUT_DIM, INPUT_DIM])
    baseline = manifest.get("baseline")
    require(isinstance(baseline, dict), "Missing BF16 baseline records")
    _verify_manifest_file(fixture, files, baseline.get("decoded_weight"),
                          "BF16 control decoded weight", WEIGHT_COUNT * 4,
                          "F32", [OUTPUT_DIM, INPUT_DIM])
    _verify_manifest_file(fixture, files, baseline.get("expected_projection"),
                          "BF16 control projection", setup["rows"] * OUTPUT_DIM * 4,
                          "F32", [setup["rows"], OUTPUT_DIM])

    source_fixture = manifest.get("source_fixture")
    require(isinstance(source_fixture, dict) and
            source_fixture.get("schema") == "wan_quantized_projection_reference_v1" and
            source_fixture.get("real_video_derived") is False,
            "Reference manifest source fixture identity changed")
    for key, expected_hash in SOURCE_FIXTURE_PINS.items():
        require(source_fixture.get(key) == expected_hash,
                f"Reference manifest source fixture pin changed: {key}")

    source_payload_hash = None
    storage_summary = {}
    quality_summary = {}
    for arm in ARMS:
        variant = variants[arm]
        require(isinstance(variant, dict) and variant.get("patches") == PATCHES_PER_BLOCK[arm],
                f"Reference manifest patch quota mismatch: {arm}")
        for field, filename in (("payload", "payload.bin"), ("sidecar", "sidecar.bin"),
                                ("descriptor", "descriptor.bin"),
                                ("decoded_weight", "decoded_weight.f32"),
                                ("expected_projection", "expected_projection.f32")):
            file_record = variant.get(field)
            require(isinstance(file_record, dict) and
                    file_record.get("path") == f"variants/{arm}/{filename}",
                    f"Unexpected fixture path for {arm} {field}")
        expected = expected_storage(arm)
        actual_setup = validate_storage_accounting(setup_storage[arm], arm)
        require(actual_setup == expected, f"Setup storage accounting changed: {arm}")
        storage = variant.get("storage")
        require(isinstance(storage, dict), f"Missing manifest storage accounting: {arm}")
        logical = expected["base_payload_bytes"] + expected["sidecar_bytes"] + expected["descriptor_bytes"]
        manifest_expected = {
            **expected,
            "blocks": BLOCK_COUNT,
            "delta_record_bytes": 3,
            "logical_storage_bytes": logical,
            "resident_storage_bytes": logical,
            "logical_weight_bytes_bf16": WEIGHT_COUNT * 2,
            "decoded_f32_diagnostic_bytes": WEIGHT_COUNT * 4,
        }
        for key, value in manifest_expected.items():
            require(type(storage.get(key)) is type(value) and storage.get(key) == value,
                    f"Unexpected manifest {key}: {arm}")
        require(storage.get("diagnostic_exports_counted_in_resident_storage") is False and
                storage.get("resident_storage_basis") ==
                "serialized canonical base, fixed-stride sidecar, and 16-byte descriptor; no padding or index arrays",
                f"Unexpected resident-storage basis: {arm}")

        payload_path, payload_hash = _verify_manifest_file(
            fixture, files, variant.get("payload"), f"{arm} canonical payload", BASE_PAYLOAD_BYTES,
            "Q4_K", [OUTPUT_DIM, INPUT_DIM // BLOCK_VALUES, Q4_K_BLOCK_BYTES])
        if source_payload_hash is None:
            source_payload_hash = payload_hash
        else:
            require(payload_hash == source_payload_hash,
                    f"Canonical Q4_K payload identity changed across arms: {arm}")
        sidecar_path, sidecar_hash = _verify_manifest_file(
            fixture, files, variant.get("sidecar"), f"{arm} sidecar", expected["sidecar_bytes"],
            "U8F16LE_3B", [BLOCK_COUNT, PATCHES_PER_BLOCK[arm], 3])
        require(expected["sidecar_bytes"] % 4 == 0,
                f"Sidecar cannot be compared as complete U32 words: {arm}")
        descriptor_path, descriptor_hash = _verify_manifest_file(
            fixture, files, variant.get("descriptor"), f"{arm} descriptor", DESCRIPTOR_BYTES,
            "WQB1Descriptor", [DESCRIPTOR_BYTES])
        descriptor_words = struct.unpack("<4I", descriptor_path.read_bytes())
        expected_format_id = 0 if arm == "q4_k" else 0x100 + PATCHES_PER_BLOCK[arm]
        require(descriptor_words == (DESCRIPTOR_MAGIC, expected_format_id, 13824, 5120),
                f"Descriptor contents mismatch: {arm}")

        _verify_manifest_file(fixture, files, variant.get("decoded_weight"),
                              f"{arm} decoded weight", WEIGHT_COUNT * 4,
                              "F32", [OUTPUT_DIM, INPUT_DIM])
        _verify_manifest_file(fixture, files, variant.get("expected_projection"),
                              f"{arm} expected projection", setup["rows"] * OUTPUT_DIM * 4,
                              "F32", [setup["rows"], OUTPUT_DIM])
        quality_summary[arm] = _validate_reference_quality(variant, arm, setup["rows"])
        storage_summary[arm] = {
            **expected,
            "canonical_payload_sha256": payload_hash,
            "sidecar_sha256": sidecar_hash,
            "descriptor_sha256": descriptor_hash,
        }

    return {"manifest": manifest, "storage": storage_summary,
            "quality": quality_summary}


def _validate_reference_quality(variant: dict, arm: str, rows: int) -> dict:
    quality = variant.get("cpu_quality")
    require(isinstance(quality, dict) and quality.get("evaluation_device") == "CPU" and
            quality.get("decode_device") in ("MPS", "pinned source fixture"),
            f"Missing or unexpected independent quality provenance: {arm}")
    distortion = quality.get("weight_distortion")
    require(isinstance(distortion, dict), f"Missing full-weight reference quality: {arm}")
    weight_error = distortion.get("weight_error")
    require(isinstance(weight_error, dict) and
            weight_error.get("elements") == WEIGHT_COUNT,
            f"Incomplete independent full-weight quality: {arm}")
    for key in ("max_abs", "rmse", "relative_l2", "squared_l2"):
        _finite_number(weight_error.get(key), f"{arm} reference weight {key}")
    weighted = distortion.get("activation_weighted_error")
    require(isinstance(weighted, dict) and
            weighted.get("fold0_train_rows") == [0, 1] and
            weighted.get("fold1_train_rows") == [2, 3],
            f"Unexpected independent weight-quality folds: {arm}")
    for key in ("fold0_weighted_squared_error", "fold1_weighted_squared_error"):
        _finite_number(weighted.get(key), f"{arm} {key}")

    projection = variant.get("quality_vs_original_bf16")
    require(isinstance(projection, dict) and
            projection.get("compared_values") == rows * OUTPUT_DIM,
            f"Incomplete independent projection quality: {arm}")
    for key in ("max_abs_vs_bf16", "rmse_vs_bf16", "relative_l2_vs_bf16"):
        _finite_number(projection.get(key), f"{arm} reference projection {key}")
    violations = projection.get("values_exceeding_reference_scale_tolerance_vs_bf16")
    require(type(violations) is int and 0 <= violations <= rows * OUTPUT_DIM,
            f"Invalid independent projection quality counter: {arm}")
    require(isinstance(projection.get("note"), str) and
            "same-format parity gate" in projection["note"],
            f"Projection quality/parity distinction missing: {arm}")
    return {"full_weight_vs_bf16": weight_error,
            "activation_weighted_error": weighted,
            "projection_vs_bf16": projection}


def _validate_file_map(files, *, check_bytes: bool = True) -> dict:
    require(isinstance(files, dict) and files, "Missing receipt file map")
    normalized = {}
    for name, digest in files.items():
        require(isinstance(name, str) and Path(name).is_absolute(), "Receipt contains a non-absolute file path")
        require(isinstance(digest, str) and len(digest) == 64 and
                all(character in "0123456789abcdef" for character in digest),
                f"Invalid SHA-256 pin: {name}")
        path = Path(name)
        if check_bytes:
            require(path.is_file() and sha(path) == digest, f"Pinned file changed or missing: {name}")
        normalized[name] = digest
    return normalized


def validate_source_binding(files: dict, completion_sources: dict, root: Path = ROOT) -> None:
    """Require receipt source hashes to equal the repository subset of intent pins."""
    expected = {name: digest for name, digest in files.items()
                if Path(name).is_relative_to(root)}
    require(isinstance(completion_sources, dict) and completion_sources == expected,
            "Completion source map differs from the intent file pins")


def _linked_receipt(base: Path, receipt_path: Path, intent_path: Path,
                    mode: str, tag: str) -> tuple[dict, dict]:
    require(receipt_path.is_file() and intent_path.is_file(),
            f"Missing linked {mode} receipt or intent")
    completion = json.loads(receipt_path.read_text())
    intent = json.loads(intent_path.read_text())
    require(completion.get("schema") == "wan_q4k_delta_completion_v1" and
            type(completion.get("returncode")) is int and completion["returncode"] == 0 and
            completion.get("changed_files") == [],
            f"Unsuccessful linked {mode} operation")
    require(intent.get("schema") == "wan_q4k_delta_intent_v1" and
            intent.get("mode") == mode and intent.get("tag") == tag,
            f"Unexpected linked {mode} intent")
    require(completion.get("intent_sha256") == sha(intent_path),
            f"Linked {mode} intent digest mismatch")
    files = _validate_file_map(intent.get("files"))
    validate_source_binding(files, completion.get("sources"))
    return completion, intent


def validate_lineage(base: Path, tag: str, receipt: dict, intent: dict,
                     setup: dict) -> dict:
    """Verify the probe -> build/oracle -> source/fixture receipt chain."""
    require(receipt.get("schema") == "wan_q4k_delta_completion_v1" and
            type(receipt.get("returncode")) is int and receipt["returncode"] == 0 and
            receipt.get("changed_files") == [],
            f"Unsuccessful probe: {tag}")
    require(intent.get("schema") == "wan_q4k_delta_intent_v1" and
            intent.get("mode") == "probe" and intent.get("tag") == tag,
            "Probe intent mode or tag mismatch")
    files = _validate_file_map(intent.get("files"))
    validate_source_binding(files, receipt.get("sources"))
    command = intent.get("command")
    binary, fixture, rounds, reverse_first = validate_probe_command(
        command, intent.get("reverse_first"), setup,
        intent.get("block_validation_comparison", False),
        intent.get("fused_comparison", False))
    comparison = setup.get("block_validation_comparison", False)
    fused = setup.get("fused_comparison", False)
    require(type(comparison) is bool and type(fused) is bool and not (comparison and fused),
            "Invalid or conflicting comparison mode setup")
    execution_arms = FUSED_ARMS if fused else COMPARISON_ARMS if comparison else ARMS
    expected_formats = {arm: arm for arm in ARMS}
    if comparison:
        expected_formats.update({f"{arm}_block": arm for arm in ARMS})
    if fused:
        expected_formats = {arm: arm.removesuffix("_fused") for arm in FUSED_ARMS}
        require("execution_arms" in setup and "format_by_arm" in setup,
                "Fused comparison setup is missing execution-arm mapping")
        stage_bytes = WEIGHT_COUNT * 4
        require(setup.get("fused_staged_control_decoder") == "block_validated" and
                setup.get("candidate_global_f32_weight_stage_bytes") == 0 and
                setup.get("control_f32_weight_stage_bytes") == stage_bytes and
                type(setup.get("retained_harness_bytes_excluding_stage")) is int and
                setup["retained_harness_bytes_excluding_stage"] >= 0 and
                type(setup.get("explicit_buffer_bytes_during_pack")) is int and
                setup["explicit_buffer_bytes_during_pack"] ==
                setup["retained_harness_bytes_excluding_stage"] + stage_bytes and
                setup.get("q4_k_fused_dummy_sidecar_bytes") == 4 and
                setup.get("candidate_pipeline") == "video_q4k_delta_linear_f32" and
                type(setup.get("candidate_static_threadgroup_memory_bytes")) is int and
                0 <= setup["candidate_static_threadgroup_memory_bytes"] <= 32 * 1024,
                "Fused candidate stage, control stage, dummy sidecar or pipeline accounting changed")
        require(setup.get("dense_mode_execution_arms") == ["dense_native", "dense_mps"] and
                setup.get("dense_mode_weight_format") == "q4_k" and
                setup.get("dense_mode_internal_tile_shape") == {"tm": 64, "tn": 32, "tk": 32},
                "Dense same-format control mapping or TM64/TN32/TK32 tile shape changed")
    if comparison or fused:
        require("execution_arms" in setup and "format_by_arm" in setup,
                "Comparison setup is missing execution-arm mapping")
    if "execution_arms" in setup:
        require(setup.get("execution_arms") == list(execution_arms),
                "Unexpected execution arm declaration")
    if "format_by_arm" in setup:
        require(setup.get("format_by_arm") == expected_formats,
                "Execution arm format mapping differs from the frozen oracle formats")
    require(fixture.is_absolute() and fixture.resolve().parent == base.resolve() and
            fixture.name.startswith("fixture-"), "Unexpected probe fixture path")
    require(receipt.get("binary_sha256") == sha(binary), "Probe binary hash mismatch")
    _pin_for(files, binary)

    manifest_path = fixture / "manifest.json"
    manifest_sha = sha(manifest_path)
    _pin_for(files, manifest_path)
    require(setup.get("manifest_sha256") == manifest_sha,
            "Setup manifest digest differs from the selected fixture")
    fixture_tag = fixture.name.removeprefix("fixture-")

    build_tag = binary.name.removeprefix("probe-")
    build_receipt_path = base / f"build-{build_tag}.completion.json"
    build_intent_path = base / f"build-{build_tag}.intent.json"
    _pin_for(files, build_receipt_path)
    build_receipt, build_intent = _linked_receipt(
        base, build_receipt_path, build_intent_path, "build", build_tag)
    build_command = build_intent.get("command")
    require(isinstance(build_command, list) and "-o" in build_command,
            "Malformed build command")
    build_binary = Path(build_command[build_command.index("-o") + 1])
    require(build_binary.resolve() == binary.resolve(), "Probe binary differs from the pinned build output")
    require(build_receipt.get("binary_sha256") == sha(binary) and
            build_receipt.get("sources") == receipt.get("sources"),
            "Probe source/binary identity differs from the successful build")

    oracle_receipt_path = base / f"oracle-{fixture_tag}.completion.json"
    oracle_intent_path = base / f"oracle-{fixture_tag}.intent.json"
    _pin_for(files, oracle_receipt_path)
    oracle_receipt, oracle_intent = _linked_receipt(
        base, oracle_receipt_path, oracle_intent_path, "oracle", fixture_tag)
    oracle_command = oracle_intent.get("command")
    require(isinstance(oracle_command, list) and "--output-dir" in oracle_command and
            "--rows" in oracle_command and "--fixture-dir" in oracle_command,
            "Malformed oracle command")
    oracle_fixture = Path(oracle_command[oracle_command.index("--output-dir") + 1])
    oracle_rows = oracle_command[oracle_command.index("--rows") + 1]
    oracle_source = Path(oracle_command[oracle_command.index("--fixture-dir") + 1])
    expected_source = base.parent / "wan-quantized-projection-20261022" / "fixture-4480-v3"
    require(oracle_fixture.resolve() == fixture.resolve() and oracle_rows == str(setup.get("rows")),
            "Fixture output path/geometry differs from its oracle command")
    require(oracle_source.resolve() == expected_source.resolve(),
            "Oracle source fixture differs from the pinned projection fixture")
    _pin_for(oracle_intent["files"], oracle_source / "manifest.json")
    require(oracle_receipt.get("produced_manifest_sha256") == manifest_sha,
            "Oracle receipt does not bind the selected fixture manifest")
    output_manifest = json.loads(manifest_path.read_text())
    require(output_manifest.get("source_fixture", {}).get("manifest_sha256") ==
            sha(oracle_source / "manifest.json"),
            "Output fixture does not bind the pinned source fixture manifest")
    require(setup.get("rows") in (4, 4480) and setup.get("din") == 5120 and
            setup.get("dout") == 13824 and setup.get("unique_source_rows") == 4 and
            setup.get("real_video_derived") is False,
            "Unexpected projection geometry or source-row scope")
    require(setup.get("variants") == list(ARMS), "Unexpected arm set")
    reference = _validate_reference_manifest(fixture, files, setup)
    source = reference["manifest"].get("source_fixture")
    require(isinstance(source, dict) and source.get("manifest_sha256") ==
            sha(oracle_source / "manifest.json") and source.get("real_video_derived") is False,
            "Reference manifest does not bind the expected non-video source fixture")
    variants = reference["manifest"]["variants"]
    input_hash = reference["manifest"]["input"]["sha256"]
    bias_hash = reference["manifest"]["bias"]["sha256"]
    original_hash = reference["manifest"]["original_weight"]["sha256"]
    require(source.get("input_sha256") == input_hash and
            source.get("bias_f32_sha256") == bias_hash and
            source.get("checkpoint_weight_sha256") == original_hash and
            source.get("q4_k_sha256") == reference["storage"]["q4_k"]["canonical_payload_sha256"] and
            source.get("q4_decoded_sha256") == variants["q4_k"]["decoded_weight"]["sha256"],
            "Reference manifest source hashes disagree with the serialized files")
    weight_count = setup["din"] * setup["dout"]
    explicit_bytes = setup.get("explicit_buffer_bytes_during_pack")
    require(setup.get("f32_stage_bytes") == weight_count * 4 and
            setup.get("packing_workspace_bytes") == 0 and
            type(explicit_bytes) is int and 0 <= explicit_bytes <= 3 * 1024**3,
            "Unexpected F32 stage, packing workspace or explicit-buffer accounting")
    return {"build_tag": build_tag, "fixture_tag": fixture_tag, "rounds": rounds,
            "reverse_first": reverse_first, "manifest_sha256": manifest_sha,
            "block_validation_comparison": comparison, "fused_comparison": fused,
            "storage": reference["storage"],
            "reference_quality": reference["quality"],
            "binary_sha256": receipt["binary_sha256"],
            "build_receipt_sha256": sha(build_receipt_path),
            "oracle_receipt_sha256": sha(oracle_receipt_path)}


def read_probe(base: Path, tag: str):
    key = f"probe-{tag}"
    paths = {name: base / f"{key}.{name}" for name in
             ("completion.json", "intent.json", "stdout.log", "stderr.log", "guard.jsonl")}
    receipt = json.loads(paths["completion.json"].read_text())
    intent = json.loads(paths["intent.json"].read_text())
    for field, name in (("intent_sha256", "intent.json"), ("stdout_sha256", "stdout.log"),
                        ("stderr_sha256", "stderr.log"), ("guard_sha256", "guard.jsonl")):
        require(sha(paths[name]) == receipt.get(field), f"Receipt artifact changed: {paths[name]}")
    records = [json.loads(line) for line in paths["stdout.log"].read_text().splitlines()]
    allowed_schemas = {
        "wan_q4k_delta_setup_v1", "wan_q4k_delta_packing_v1", "wan_q4k_delta_bf16_control_v1",
        "wan_q4k_delta_parity_v1", "wan_q4k_delta_dense_parity_v1",
        "wan_q4k_delta_timing_v1", "wan_q4k_delta_controls_v1",
    }
    require(all(isinstance(record, dict) and record.get("schema") in allowed_schemas for record in records),
            "Unknown or malformed probe output record")
    setup_records = [record for record in records if record["schema"] == "wan_q4k_delta_setup_v1"]
    require(len(setup_records) == 1, "Missing or duplicated setup record")
    setup = setup_records[0]
    lineage = validate_lineage(base, tag, receipt, intent, setup)
    rows, din, dout = setup["rows"], setup["din"], setup["dout"]
    comparison = setup.get("block_validation_comparison", False)
    fused = setup.get("fused_comparison", False)
    require(type(comparison) is bool and type(fused) is bool and not (comparison and fused),
            "Invalid or conflicting probe comparison modes")
    execution_arms = FUSED_ARMS if fused else COMPARISON_ARMS if comparison else ARMS
    require(setup.get("tolerance") == "finite && abs(actual-ref)<=2e-5+2e-5*abs(ref)",
            "Unexpected numerical acceptance gate")

    parity = [record for record in records if record["schema"] == "wan_q4k_delta_parity_v1"]
    validate_parity(parity, rows, din, dout, execution_arms,
                    require_bit_exact=comparison, require_fused_staged=fused)
    dense_parity_records = [record for record in records
                            if record["schema"] == "wan_q4k_delta_dense_parity_v1"]
    if fused:
        dense_parity = validate_dense_parity(dense_parity_records, rows, din, dout)
    else:
        require(not dense_parity_records,
                "Unexpected dense-mode parity records outside fused comparison")
        dense_parity = None
    bf16 = [record for record in records if record["schema"] == "wan_q4k_delta_bf16_control_v1"]
    require(len(bf16) == 1, "Missing or duplicated BF16 control")
    validate_metric(bf16[0].get("decoder"), din * dout, "BF16 decoder control")
    validate_metric(bf16[0].get("projection"), rows * dout, "BF16 projection control")

    packs = validate_pack_records(
        [record for record in records if record["schema"] == "wan_q4k_delta_packing_v1"])

    timings = [record for record in records if record["schema"] == "wan_q4k_delta_timing_v1"]
    timings_ms = validate_timing_records(timings, setup, lineage["rounds"], lineage["reverse_first"])

    controls = [record for record in records if record["schema"] == "wan_q4k_delta_controls_v1"]
    require(len(controls) == 1, "Missing or duplicated corruption control")
    validate_corruption_control(controls[0], rows * dout, fused_comparison=fused)

    require(len(records) == expected_probe_record_count(len(timings), comparison, fused),
            "Missing or extra setup, pack, parity, timing or control records")
    guards = [json.loads(line) for line in paths["guard.jsonl"].read_text().splitlines()]
    memory = validate_guard(guards, intent)
    loss = {}
    before = {record["variant"]: record for record in parity if record["phase"] == "before_timing"}
    after = {record["variant"]: record for record in parity if record["phase"] == "after_timing"}
    formats = setup.get("format_by_arm", {arm: arm for arm in ARMS})
    for arm in execution_arms:
        format_arm = formats[arm]
        native_quality = {}
        for phase, record in (("before_timing", before[arm]), ("after_timing", after[arm])):
            row_metrics = record["lossy_vs_bf16_per_source_row"]
            full_weight = record["lossy_weight_vs_bf16"]
            native_quality[phase] = {
                "full_weight_vs_bf16": {
                    key: full_weight[key] for key in (
                        "compared_values", "bit_differences", "tolerance_violations", "nonfinite",
                        "max_abs", "rmse", "relative_l2", "cosine",
                    )
                },
                "per_source_row_relative_l2_vs_bf16": [metric["relative_l2"]
                                                        for metric in row_metrics],
            }
        loss[arm] = {
            "native_gpu": native_quality,
            "reference_cpu": lineage["reference_quality"][format_arm],
        }
    return {"tag": tag, "completion_sha256": sha(paths["completion.json"]),
            "lineage": lineage, "setup": setup, "timings_ms": timings_ms,
            "dense_parity": dense_parity,
            "lossy_quality": loss, "packing": packs,
            "all_decoder_projection_checks_bit_exact": all(
                record[metric]["bit_differences"] == 0
                for record in parity for metric in ("decoder", "projection")),
            "memory": memory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("tags", nargs="+")
    args = parser.parse_args()
    result = {"schema": "wan_q4k_delta_results_v1", "time_unit": "milliseconds",
              "scope": "One real projection; four unique seeded inputs; no video quality claim",
              "probes": [read_probe(args.evidence, tag) for tag in args.tags]}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
