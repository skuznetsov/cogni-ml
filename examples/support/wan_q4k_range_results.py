#!/usr/bin/env python3
"""Verify and summarize immutable receipts for the Q4_K range-count probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


ARMS = ("q4_k", "q4_k_minmax8", "q4_k_minmax1")
MODES = ("projection", "decode_only")
ROOT = Path(__file__).resolve().parents[2]
MIB = 1024 * 1024
PHYSICAL_RESERVE_PCT = 15
MAX_FOOTPRINT_BYTES = 4096 * MIB
MAX_SWAP_GROWTH_BYTES = 64 * MIB


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


def validate_parity(records, rows: int, din: int, dout: int) -> None:
    expected = {(arm, phase) for arm in ARMS for phase in ("before_timing", "after_timing")}
    actual = [(record.get("variant"), record.get("phase")) for record in records]
    require(len(actual) == len(expected) and set(actual) == expected,
            "Missing, duplicate or unexpected pre/post parity records")
    for record in records:
        arm = record["variant"]
        validate_metric(record.get("decoder"), din * dout, f"{arm} decoder")
        validate_metric(record.get("projection"), rows * dout, f"{arm} projection")
        row_metrics = record.get("lossy_vs_bf16_per_source_row")
        require(isinstance(row_metrics, list) and len(row_metrics) == 4,
                f"Incomplete per-source-row BF16 comparison: {arm}")
        for index, metric in enumerate(row_metrics):
            validate_metric(metric, dout, f"{arm} BF16 source row {index}", strict=False)
            require(metric["nonfinite"] == 0, f"Non-finite BF16 loss metric: {arm} row {index}")
        require(record.get("calibration_rows") == [] and
                record.get("unique_evaluation_rows") == [0, 1, 2, 3],
                f"Unexpected calibration/evaluation scope: {arm}")


def _rotated_order(rotation: int, reverse: bool) -> tuple[str, ...]:
    order = ARMS[rotation:] + ARMS[:rotation]
    return tuple(reversed(order)) if reverse else order


def expected_probe_record_count(timing_count: int) -> int:
    """Count setup, two pack records, BF16 control, parity, timing and control."""
    return 1 + len(ARMS[1:]) + 1 + len(ARMS) * 2 + timing_count + 1


def validate_corruption_control(record: dict, expected_count: int) -> None:
    metric = record.get("corruption")
    validate_metric(metric, expected_count, "positive corruption control", strict=False)
    require(record.get("corruption_rejected") is True and
            record.get("final_status_zero") is True and
            metric["bit_differences"] == 2 and
            metric["tolerance_violations"] == 2 and
            metric["nonfinite"] == 1,
            "Positive corruption/NaN control did not detect the expected failures")


def validate_timing_records(records, setup: dict, rounds: int, reverse_first: bool):
    require(rounds in (12, 24), "Unsupported measured-round count")
    require(type(reverse_first) is bool, "Invalid reverse-first flag")
    rows = setup["rows"]
    expected_records_per_mode = (rounds + 2) * len(ARMS)
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
        require(type(position) is int and 0 <= position < len(ARMS),
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
            require(bucket is not None and set(bucket) == {0, 1, 2},
                    f"Incomplete timing round: {mode} {round_id}")
            if round_id < 0:
                reverse = (round_id == -1) != reverse_first
            else:
                reverse = (round_id >= rounds // 2) != reverse_first
            expected_order = _rotated_order(max(round_id, 0) % len(ARMS), reverse)
            actual_order = tuple(bucket[position]["variant"] for position in range(len(ARMS)))
            require(actual_order == expected_order,
                    f"Unexpected arm order: {mode} {round_id}: {actual_order}")

    timings = {}
    for mode in MODES:
        measured = [record for (candidate_mode, round_id), bucket in by_mode_round.items()
                    if candidate_mode == mode and round_id >= 0
                    for record in bucket.values()]
        by_round = {round_id: {bucket[position]["variant"]: bucket[position]
                              for position in range(len(ARMS))}
                    for (candidate_mode, round_id), bucket in by_mode_round.items()
                    if candidate_mode == mode and round_id >= 0}
        timings[mode] = {
            "arms": {arm: {metric: summary([record[metric] * 1000 for record in measured
                                           if record["variant"] == arm])
                            for metric in ("gpu_seconds", "wall_seconds", "encode_seconds")}
                     for arm in ARMS},
            "paired_shared1_over_local8_gpu": summary([
                by_round[r][ARMS[2]]["gpu_seconds"] / by_round[r][ARMS[1]]["gpu_seconds"]
                for r in range(rounds)]),
            "paired_local8_over_ggml_gpu": summary([
                by_round[r][ARMS[1]]["gpu_seconds"] / by_round[r][ARMS[0]]["gpu_seconds"]
                for r in range(rounds)]),
        }
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
    require(completion.get("schema") == "wan_q4k_range_completion_v1" and
            type(completion.get("returncode")) is int and completion["returncode"] == 0 and
            completion.get("changed_files") == [],
            f"Unsuccessful linked {mode} operation")
    require(intent.get("schema") == "wan_q4k_range_intent_v1" and
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
    require(receipt.get("schema") == "wan_q4k_range_completion_v1" and
            type(receipt.get("returncode")) is int and receipt["returncode"] == 0 and
            receipt.get("changed_files") == [],
            f"Unsuccessful probe: {tag}")
    require(intent.get("schema") == "wan_q4k_range_intent_v1" and
            intent.get("mode") == "probe" and intent.get("tag") == tag,
            "Probe intent mode or tag mismatch")
    files = _validate_file_map(intent.get("files"))
    validate_source_binding(files, receipt.get("sources"))
    command = intent.get("command")
    require(isinstance(command, list) and len(command) in (3, 4), "Unexpected probe command")
    binary = Path(command[0])
    fixture_arg = command[1]
    rounds_arg = command[2]
    require(binary.is_absolute() and binary.name.startswith("probe-"), "Unexpected probe binary path")
    require(isinstance(fixture_arg, str) and fixture_arg.startswith("--fixture="),
            "Probe command lacks fixture argument")
    fixture = Path(fixture_arg.removeprefix("--fixture="))
    require(fixture.is_absolute() and fixture.resolve().parent == base.resolve() and
            fixture.name.startswith("fixture-"), "Unexpected probe fixture path")
    require(isinstance(rounds_arg, str) and rounds_arg.startswith("--rounds="),
            "Probe command lacks rounds argument")
    rounds_text = rounds_arg.removeprefix("--rounds=")
    require(rounds_text in ("12", "24"), "Unexpected probe round count")
    rounds = int(rounds_text)
    reverse_first = intent.get("reverse_first")
    expected_reverse = len(command) == 4 and command[3] == "--reverse-first"
    require(type(reverse_first) is bool and reverse_first == expected_reverse,
            "Probe reverse-order intent differs from command")
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
    weight_count = setup["din"] * setup["dout"]
    require(setup.get("payload_bytes_per_arm") == 39_813_120 and
            setup.get("common_descriptor_bytes_per_arm") == 16 and
            setup.get("package_bytes_per_arm") == 39_813_136 and
            setup.get("f32_stage_bytes") == weight_count * 4 and
            setup.get("packing_workspace_bytes") == 0 and
            setup.get("explicit_buffer_bytes_during_pack", MAX_FOOTPRINT_BYTES + 1) <= 3 * 1024**3,
            "Unexpected equal-package or explicit-buffer accounting")
    return {"build_tag": build_tag, "fixture_tag": fixture_tag, "rounds": rounds,
            "reverse_first": reverse_first, "manifest_sha256": manifest_sha,
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
        "wan_q4k_range_setup_v1", "wan_q4k_range_packing_v1", "wan_q4k_range_bf16_control_v1",
        "wan_q4k_range_parity_v1", "wan_q4k_range_timing_v1", "wan_q4k_range_controls_v1",
    }
    require(all(isinstance(record, dict) and record.get("schema") in allowed_schemas for record in records),
            "Unknown or malformed probe output record")
    setup_records = [record for record in records if record["schema"] == "wan_q4k_range_setup_v1"]
    require(len(setup_records) == 1, "Missing or duplicated setup record")
    setup = setup_records[0]
    lineage = validate_lineage(base, tag, receipt, intent, setup)
    rows, din, dout = setup["rows"], setup["din"], setup["dout"]
    require(setup.get("tolerance") == "finite && abs(actual-ref)<=2e-5+2e-5*abs(ref)",
            "Unexpected numerical acceptance gate")

    parity = [record for record in records if record["schema"] == "wan_q4k_range_parity_v1"]
    validate_parity(parity, rows, din, dout)
    bf16 = [record for record in records if record["schema"] == "wan_q4k_range_bf16_control_v1"]
    require(len(bf16) == 1, "Missing or duplicated BF16 control")
    validate_metric(bf16[0].get("decoder"), din * dout, "BF16 decoder control")
    validate_metric(bf16[0].get("projection"), rows * dout, "BF16 projection control")

    packs = [record for record in records if record["schema"] == "wan_q4k_range_packing_v1"]
    require(sorted(record.get("variant") for record in packs) == sorted(ARMS[1:]),
            "Missing, duplicate or unexpected independent pack comparisons")
    for record in packs:
        require(type(record.get("payload_word_differences")) is int and
                record["payload_word_differences"] == 0 and
                type(record.get("encode_metal_buffer_allocations")) is int and
                record["encode_metal_buffer_allocations"] == 0 and
                type(record.get("activation_readbacks")) is int and
                record["activation_readbacks"] == 0 and
                type(record.get("terminal_waits")) is int and record["terminal_waits"] == 1 and
                record.get("byte_validation_outside_packing_interval") is True,
                "Independent GPU pack validation failed")
        for metric in ("gpu_seconds", "wall_seconds", "encode_seconds"):
            _finite_number(record.get(metric), f"pack {metric}")

    timings = [record for record in records if record["schema"] == "wan_q4k_range_timing_v1"]
    timings_ms = validate_timing_records(timings, setup, lineage["rounds"], lineage["reverse_first"])

    controls = [record for record in records if record["schema"] == "wan_q4k_range_controls_v1"]
    require(len(controls) == 1, "Missing or duplicated corruption control")
    validate_corruption_control(controls[0], rows * dout)

    require(len(records) == expected_probe_record_count(len(timings)),
            "Missing or extra setup, pack, parity, timing or control records")
    guards = [json.loads(line) for line in paths["guard.jsonl"].read_text().splitlines()]
    memory = validate_guard(guards, intent)
    loss = {}
    before = {record["variant"]: record for record in parity if record["phase"] == "before_timing"}
    for arm in ARMS:
        row_metrics = before[arm]["lossy_vs_bf16_per_source_row"]
        error_sq = [record["rmse"]**2 * record["compared_values"] for record in row_metrics]
        reference_sq = [error / record["relative_l2"]**2
                        for error, record in zip(error_sq, row_metrics)
                        if record["relative_l2"] > 0]
        require(len(reference_sq) == len(row_metrics),
                f"Cannot pool zero-distortion reference rows for {arm}")
        loss[arm] = {"pooled_relative_l2_vs_bf16": math.sqrt(sum(error_sq) / sum(reference_sq)),
                     "per_source_row_relative_l2_vs_bf16": [record["relative_l2"]
                                                            for record in row_metrics]}
    return {"tag": tag, "completion_sha256": sha(paths["completion.json"]),
            "lineage": lineage, "setup": setup, "timings_ms": timings_ms,
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
    result = {"schema": "wan_q4k_range_results_v1", "time_unit": "milliseconds",
              "scope": "One real projection; four unique seeded inputs; no video quality claim",
              "probes": [read_probe(args.evidence, tag) for tag in args.tags]}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
