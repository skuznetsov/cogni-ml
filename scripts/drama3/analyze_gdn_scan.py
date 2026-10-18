#!/usr/bin/env python3
"""Validate and summarize an offline DRAMA-3 GDN scan probe capture.

This reads only the JSON report and preserved stderr log. GPU command times are
descriptive measurements of this diagnostic corridor; graph-depth-2 GPU
intervals may overlap and are not additive. Results make no full-prefill or
WAV-level performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import sys
from typing import Any


SCHEMA = "drama3-gdn-scan-probe-v1"
LAYERS = (0, 38)
ROWS = 29
GEOMETRY = (32, 128, 16, 128)
WARMUPS = 2
MEASURED_PAIRS = 8
PAIR_COUNT = WARMUPS + MEASURED_PAIRS
METRIC_COLUMNS = (
    "bit_mismatches",
    "tolerance_failures",
    "max_abs",
    "max_tolerance_ratio",
    "nonfinite",
)
MODES = ("sequential", "serial_tile", "affine_scan")
EXPECTED_CONTROLS = {
    "DRAMA3_BACKEND": "metal",
    "DRAMA3_DELTA_TRANSPOSE": "1",
    "DRAMA3_MOE_BATCH": "1",
    "DRAMA3_EXPERT_THREADS": "64",
    "DRAMA3_DENSE_REDUCTION": "tree",
    "DRAMA3_BACKBONE_DENSE_REDUCTION": "tree",
    "DRAMA3_EXPERT_DOWN_REDUCTION": "native",
    "DRAMA3_MMAP_SHARED": "0",
    "DRAMA3_GDN_POST_FUSION": "0",
}
REQUIRED_SOURCE_KEYS = {
    "examples/drama3_gdn_scan_probe.cr",
    "examples/support/drama3_gdn_scan.cr",
    "examples/support/drama3_gdn_scan.metal",
    "examples/support/drama3_prefill_compare.metal",
    "src/ml/audio/drama3/backbone.cr",
    "src/ml/audio/drama3/backbone_prefill.cr",
    "src/ml/audio/drama3/backbone_resident.cr",
    "src/ml/audio/drama3/generator_resident.cr",
    "src/ml/audio/drama3/kernels/backbone_resident.metal",
    "src/ml/audio/drama3/kernels/resident.metal",
    "src/ml/audio/drama3/kernels/resident_batch.metal",
    "src/ml/audio/drama3/kernels/tower_resident.metal",
    "src/ml/audio/drama3/native.cr",
    "src/ml/audio/drama3/prompt.cr",
    "src/ml/audio/drama3/prompt_resident.cr",
    "src/ml/audio/drama3/resident.cr",
    "src/ml/audio/drama3/resident_batch.cr",
    "src/ml/audio/drama3/tower.cr",
    "src/ml/audio/drama3/tower_resident.cr",
}
NATIVE_GATE_TEXT = "zero output/state/prefix finite bit differences; nonfinite rejects"
TOLERANCE_GATE_TEXT = (
    "6e-5 + 3e-4 * abs(reference), unchanged; does not override native bit gate"
)


class ValidationError(ValueError):
    """Input evidence does not satisfy the probe's capture contract."""


def _fail(message: str) -> None:
    raise ValidationError(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    _fail(f"non-finite JSON constant: {value}")


def _json_loads(value: str, label: str) -> Any:
    try:
        parsed = json.loads(
            value, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError(f"invalid {label} JSON: {exc}") from exc
    _check_finite_tree(parsed, label)
    return parsed


def _check_finite_tree(value: Any, label: str, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"{label} has non-finite number at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _check_finite_tree(child, label, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_finite_tree(child, label, f"{path}[{index}]")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        constraint = "positive finite" if positive else "finite nonnegative"
        _fail(f"{label} must be {constraint}")
    return result


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{label} must be a boolean")
    return value


def _hex_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _metric(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 5:
        _fail(f"{label} must contain the five comparison metrics")
    result = [_number(item, f"{label}[{index}]") for index, item in enumerate(value)]
    for index, name in ((0, "bit_mismatches"), (1, "tolerance_failures"), (4, "nonfinite")):
        if not result[index].is_integer():
            _fail(f"{label} {name} must be an integer-valued count")
    if result[1] > result[0] or (result[0] == 0 and (result[2] != 0 or result[3] != 0)):
        _fail(f"{label} comparison metrics contradict the bit mismatch count")
    return result


def _native_pass(metrics: list[float]) -> bool:
    return metrics[0] == 0 and metrics[4] == 0


def _tolerance_pass(metrics: list[float]) -> bool:
    return metrics[1] == 0 and metrics[4] == 0


def _validate_source_manifest(report: dict[str, Any]) -> dict[str, str]:
    sources = _object(report.get("compiled_source_sha256"), "compiled_source_sha256")
    if not sources:
        _fail("compiled_source_sha256 must not be empty")
    normalized: dict[str, str] = {}
    for path, digest in sources.items():
        if not isinstance(path, str) or not path or path.startswith("/") or ".." in pathlib.PurePosixPath(path).parts:
            _fail(f"compiled source path is not repository-relative: {path!r}")
        normalized[path] = _hex_digest(digest, f"compiled_source_sha256[{path!r}]")
    missing = REQUIRED_SOURCE_KEYS - normalized.keys()
    if missing:
        _fail(f"compiled source manifest lacks required keys: {sorted(missing)}")
    for field in ("executable_sha256", "embedded_shader_sha256"):
        _hex_digest(report.get(field), field)
    if report["embedded_shader_sha256"] != normalized["examples/support/drama3_gdn_scan.metal"]:
        _fail("embedded_shader_sha256 differs from the compiled scan shader entry")
    model_hashes = _object(report.get("model_files_sha256"), "model_files_sha256")
    if not model_hashes:
        _fail("model_files_sha256 must not be empty")
    for path, digest in model_hashes.items():
        if not isinstance(path, str) or not path:
            _fail("model_files_sha256 has an empty path")
        _hex_digest(digest, f"model_files_sha256[{path!r}]")
    return normalized


def _validate_top_level(report: dict[str, Any]) -> tuple[int, dict[str, str]]:
    if report.get("schema") != SCHEMA:
        _fail(f"schema must be {SCHEMA!r}")
    source_head = report.get("source_head")
    if not isinstance(source_head, str) or len(source_head) not in (40, 64) or any(
        char not in "0123456789abcdef" for char in source_head
    ):
        _fail("source_head must be a lowercase Git object id")
    source_hashes = _validate_source_manifest(report)
    if _bool(report.get("source_tree_checked_before_and_after"), "source_tree_checked_before_and_after") is not True:
        _fail("source tree was not checked before and after the probe")
    if _integer(report.get("rows"), "rows", minimum=1) != ROWS:
        _fail(f"rows must equal the captured prefill row count {ROWS}")
    if report.get("geometry") != list(GEOMETRY):
        _fail(f"geometry must equal {list(GEOMETRY)}")
    if report.get("layers") != list(LAYERS):
        _fail(f"layers must equal {list(LAYERS)}")
    prompt_ids = report.get("prompt_ids")
    if not isinstance(prompt_ids, list) or len(prompt_ids) != ROWS:
        _fail("prompt_ids must cover every captured row")
    for index, token_id in enumerate(prompt_ids):
        _integer(token_id, f"prompt_ids[{index}]")
    if not isinstance(report.get("text"), str) or not report["text"]:
        _fail("text must identify the fixed prompt")
    depth = _integer(report.get("graph_depth"), "graph_depth")
    if depth not in (0, 2):
        _fail("graph_depth must be 0 or 2")
    controls = _object(report.get("controls"), "controls")
    expected_controls = dict(EXPECTED_CONTROLS)
    expected_controls["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = str(depth)
    if controls != expected_controls:
        _fail("probe controls differ from the pinned capture command")
    if _bool(report.get("capture_encoding_added_reads_submits_waits"), "capture_encoding_added_reads_submits_waits"):
        _fail("capture encoding claims added reads, submits, or waits")
    if _integer(report.get("activation_readbacks"), "activation_readbacks") != 0:
        _fail("activation_readbacks must be zero")
    if _bool(report.get("production_routing_changed"), "production_routing_changed"):
        _fail("production routing must remain unchanged")
    if not isinstance(report.get("scope"), str) or "no full-prefill/WAV" not in report["scope"]:
        _fail("scope must limit interpretation to recurrence, not full-prefill/WAV")
    if report.get("native_gate") != NATIVE_GATE_TEXT:
        _fail("native gate wording differs from the pinned bit-exact contract")
    if report.get("tolerance_metric_only") != TOLERANCE_GATE_TEXT:
        _fail("tolerance gate wording differs from the pinned diagnostic-only contract")

    capture_counts = _object(report.get("capture_counts"), "capture_counts")
    if capture_counts != {str(layer): ROWS for layer in LAYERS}:
        _fail("capture_counts must contain exactly 29 captured rows for layers 0 and 38")
    _number(report.get("capture_seconds"), "capture_seconds", positive=True)
    graph = _object(report.get("graph"), "graph")
    if _integer(graph.get("depth"), "graph.depth") != depth:
        _fail("graph depth differs from graph_depth")
    submitted = _integer(graph.get("submitted"), "graph.submitted", minimum=1)
    if _integer(graph.get("completed"), "graph.completed") != submitted:
        _fail("final graph must have completed every submitted command")
    if _integer(graph.get("pending"), "graph.pending") != 0:
        _fail("final graph must have no pending command")
    readbacks = _object(report.get("readbacks"), "readbacks")
    if not readbacks or any(_integer(value, f"readbacks.{key}") < 0 for key, value in readbacks.items()):
        _fail("readbacks must contain nonnegative byte counts")

    qualification = _object(report.get("comparator_qualification"), "comparator_qualification")
    if set(qualification) != {"clean", "seeded"}:
        _fail("comparator qualification must contain clean and seeded controls")
    clean = _metric(qualification["clean"], "comparator_qualification.clean")
    seeded = _metric(qualification["seeded"], "comparator_qualification.seeded")
    if not _native_pass(clean) or not _tolerance_pass(clean):
        _fail("comparator clean control does not pass both gates")
    if _native_pass(seeded) or _tolerance_pass(seeded) or seeded[4] <= 0:
        _fail("comparator seeded defect did not trigger native, tolerance, and nonfinite controls")
    return depth, source_hashes


def _read_timeline(log_payload: bytes, depth: int) -> dict[int, float]:
    try:
        text = log_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"stderr log is not UTF-8: {exc}") from exc
    timeline: dict[int, float] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        candidate = line.strip()
        if not candidate.startswith("{"):
            # The preserved process log also carries preflight, Cocoa and shell
            # diagnostics. Only JSON records participate in the timeline.
            continue
        event = _json_loads(candidate, f"stderr line {line_number}")
        if not isinstance(event, dict):
            _fail(f"stderr line {line_number} JSON value must be an object")
        if event.get("event") != "drama3_gpu_command":
            continue
        index = _integer(event.get("index"), f"stderr line {line_number} command index")
        if index in timeline:
            _fail(f"duplicate drama3_gpu_command index {index}")
        duration = _number(event.get("gpu_seconds"), f"GPU command {index} gpu_seconds", positive=True)
        if "completion_status" in event and _integer(event["completion_status"], f"GPU command {index} completion_status") != 0:
            _fail(f"GPU command {index} did not complete successfully")
        if "gpu_timing_valid" in event and _bool(event["gpu_timing_valid"], f"GPU command {index} gpu_timing_valid") is not True:
            _fail(f"GPU command {index} has invalid GPU timing")
        if depth == 2 and ("completion_status" not in event or "gpu_timing_valid" not in event):
            _fail(f"graph-depth-2 GPU command {index} lacks completion/timing validity fields")
        timeline[index] = duration
    if not timeline:
        _fail("stderr log contains no drama3_gpu_command records")
    return timeline


def _expected_groups() -> set[tuple[int, int, int, int]]:
    expected: set[tuple[int, int, int, int]] = set()
    for layer in LAYERS:
        for base, count in ((0, 29), (7, 19)):
            for block_size in dict.fromkeys((8, 16, count)):
                expected.add((layer, base, count, block_size))
    return expected


def _expected_checks(rows: int, blocks: int) -> set[str]:
    names = {"replay_vs_capture.output", "replay_vs_capture.state", "q_varies", "initial_nonzero"}
    for mode in ("serial_tile", "affine_scan"):
        names.update(f"{mode}.output.{row}" for row in range(rows))
        names.add(f"{mode}.state")
    names.update(f"affine_scan.prefix.{block}" for block in range(blocks))
    return names


def _aggregate_metrics(values: list[list[float]]) -> dict[str, Any]:
    if not values:
        return {
            "bit_mismatches": 0,
            "tolerance_failures": 0,
            "max_abs": 0.0,
            "max_tolerance_ratio": 0.0,
            "nonfinite": 0,
            "native_gate_pass": True,
            "tolerance_gate_pass": True,
        }
    bit_mismatches = sum(value[0] for value in values)
    tolerance_failures = sum(value[1] for value in values)
    nonfinite = sum(value[4] for value in values)
    return {
        "bit_mismatches": int(bit_mismatches),
        "tolerance_failures": int(tolerance_failures),
        "max_abs": max(value[2] for value in values),
        "max_tolerance_ratio": max(value[3] for value in values),
        "nonfinite": int(nonfinite),
        "native_gate_pass": bit_mismatches == 0 and nonfinite == 0,
        "tolerance_gate_pass": tolerance_failures == 0 and nonfinite == 0,
    }


def _receipt_summary(checks: dict[str, Any], mode: str, rows: int, blocks: int) -> dict[str, Any]:
    output = [_metric(checks[f"{mode}.output.{row}"], f"{mode}.output.{row}") for row in range(rows)]
    state = _metric(checks[f"{mode}.state"], f"{mode}.state")
    result: dict[str, Any] = {
        "every_row_output": _aggregate_metrics(output),
        "final_state": _aggregate_metrics([state]),
    }
    if mode == "affine_scan":
        prefixes = [_metric(checks[f"affine_scan.prefix.{block}"], f"affine_scan.prefix.{block}") for block in range(blocks)]
        result["composed_prefix_state"] = _aggregate_metrics(prefixes)
    else:
        result["composed_prefix_state"] = None
    return result


def _validate_group(
    group: Any,
    timeline: dict[int, float],
    used_indices: set[int],
) -> dict[str, Any]:
    group = _object(group, "group")
    layer = _integer(group.get("layer"), "group.layer")
    base = _integer(group.get("base"), "group.base")
    rows = _integer(group.get("rows"), "group.rows", minimum=1)
    block_size = _integer(group.get("block_size"), "group.block_size", minimum=1)
    if layer not in LAYERS or (base, rows) not in ((0, 29), (7, 19)) or base + rows > ROWS:
        _fail(f"unsupported capture slice layer={layer}, base={base}, rows={rows}")
    if not 1 <= block_size <= rows:
        _fail("block_size must be within the group row count")
    blocks = (rows + block_size - 1) // block_size
    if _integer(group.get("blocks"), "group.blocks", minimum=1) != blocks:
        _fail("group.blocks does not match the block geometry")
    if _integer(group.get("ragged_tail"), "group.ragged_tail") != rows % block_size:
        _fail("group.ragged_tail does not match the block geometry")
    state_count = GEOMETRY[0] * GEOMETRY[1] * GEOMETRY[3]
    summary_count = blocks * GEOMETRY[0] * GEOMETRY[3] * (GEOMETRY[3] + GEOMETRY[1])
    expected_scratch_bytes = (2 * summary_count + blocks * state_count) * 4
    if _integer(group.get("scratch_bytes"), "group.scratch_bytes") != expected_scratch_bytes:
        _fail("group.scratch_bytes does not match the declared plan geometry")

    validation = _object(group.get("validation"), "group.validation")
    if set(validation) != {
        "checks",
        "comparison_columns",
        "serial_tile_native_gate_pass",
        "affine_scan_native_gate_pass",
        "affine_scan_tolerance_gate_pass",
    }:
        _fail("validation has missing or unknown fields")
    if validation.get("comparison_columns") != list(METRIC_COLUMNS):
        _fail("validation comparison columns have changed")
    checks = _object(validation.get("checks"), "validation.checks")
    expected_check_names = _expected_checks(rows, blocks)
    if set(checks) != expected_check_names:
        missing = sorted(expected_check_names - set(checks))
        extra = sorted(set(checks) - expected_check_names)
        _fail(f"validation receipts have missing/extra coverage (missing={missing[:3]}, extra={extra[:3]})")
    metrics = {name: _metric(value, f"validation.checks.{name}") for name, value in checks.items()}
    if not _native_pass(metrics["replay_vs_capture.output"]) or not _native_pass(metrics["replay_vs_capture.state"]):
        _fail("sequential replay does not reproduce captured output/state")
    if rows > 1 and metrics["q_varies"][0] <= 0:
        _fail("q_varies receipt does not prove row variation")
    expected_nonzero_initial = base > 0
    if (metrics["initial_nonzero"][0] > 0) != expected_nonzero_initial:
        _fail("initial_nonzero receipt disagrees with the slice's recurrence state")

    serial_checks = [metrics[f"serial_tile.output.{row}"] for row in range(rows)] + [metrics["serial_tile.state"]]
    affine_checks = (
        [metrics[f"affine_scan.output.{row}"] for row in range(rows)]
        + [metrics["affine_scan.state"]]
        + [metrics[f"affine_scan.prefix.{block}"] for block in range(blocks)]
    )
    recomputed_gates = {
        "serial_tile_native_gate_pass": all(_native_pass(value) for value in serial_checks),
        "affine_scan_native_gate_pass": all(_native_pass(value) for value in affine_checks),
        "affine_scan_tolerance_gate_pass": all(_tolerance_pass(value) for value in affine_checks),
    }
    for name, expected in recomputed_gates.items():
        if _bool(validation.get(name), f"validation.{name}") != expected:
            _fail(f"forged validation gate {name}: receipt-derived value is {expected}")

    measurements = _object(group.get("measurements"), "group.measurements")
    expected_modes = {"affine_scan"} | ({"serial_tile"} if block_size == 8 else set())
    if set(measurements) != expected_modes:
        _fail("measurements have missing or unexpected modes for this block size")
    mode_summaries: dict[str, Any] = {}
    for mode in sorted(expected_modes):
        pairs = measurements[mode]
        if not isinstance(pairs, list) or len(pairs) != PAIR_COUNT:
            _fail(f"{mode} must contain exactly {PAIR_COUNT} pairs including warmups")
        measured_gpu_ratios: list[float] = []
        measured_encode_ratios: list[float] = []
        per_arm_gpu: dict[str, list[float]] = {"sequential": [], mode: []}
        per_arm_encode: dict[str, list[float]] = {"sequential": [], mode: []}
        runtime_output: list[list[float]] = []
        runtime_state: list[list[float]] = []
        for pair_index, pair_value in enumerate(pairs):
            pair = _object(pair_value, f"{mode}.pair[{pair_index}]")
            if _integer(pair.get("pair"), f"{mode}.pair[{pair_index}].pair") != pair_index:
                _fail(f"{mode} pair order is incomplete or duplicated")
            if _bool(pair.get("warmup"), f"{mode}.pair[{pair_index}].warmup") != (pair_index < WARMUPS):
                _fail(f"{mode} pair {pair_index} warmup marker is inconsistent")
            if pair.get("mode") != mode:
                _fail(f"{mode} pair {pair_index} mode field is inconsistent")
            expected_order = ["sequential", mode] if pair_index % 4 in (0, 3) else [mode, "sequential"]
            if pair.get("order") != expected_order:
                _fail(f"{mode} pair {pair_index} violates the pinned ABBA command order")
            indices = _object(pair.get("command_indices"), f"{mode}.pair[{pair_index}].command_indices")
            if set(indices) != {"sequential", mode}:
                _fail(f"{mode} pair {pair_index} command_indices must identify both arms")
            normalized_indices: dict[str, int] = {}
            for arm, value in indices.items():
                index = _integer(value, f"{mode} pair {pair_index} {arm} command index")
                if index not in timeline:
                    _fail(f"missing timing for referenced command index {index}")
                if index in used_indices:
                    _fail(f"timed command index {index} is reused")
                used_indices.add(index)
                normalized_indices[arm] = index
                per_arm_gpu[arm].append(timeline[index])
            if len(set(normalized_indices.values())) != 2:
                _fail(f"{mode} pair {pair_index} arms reuse a command index")
            if normalized_indices[expected_order[0]] >= normalized_indices[expected_order[1]]:
                _fail(f"{mode} pair {pair_index} command indices contradict the declared arm order")
            encode = _object(pair.get("host_encode_seconds"), f"{mode}.pair[{pair_index}].host_encode_seconds")
            if set(encode) != {"sequential", mode}:
                _fail(f"{mode} pair {pair_index} encode times must identify both arms")
            for arm, value in encode.items():
                per_arm_encode[arm].append(_number(value, f"{mode} pair {pair_index} {arm} encode time", positive=True))
            checks_pair = _object(pair.get("checks"), f"{mode}.pair[{pair_index}].checks")
            if set(checks_pair) != {"output", "state"}:
                _fail(f"{mode} pair {pair_index} lacks full output/state receipts")
            output_metric = _metric(checks_pair["output"], f"{mode} pair {pair_index} output")
            state_metric = _metric(checks_pair["state"], f"{mode} pair {pair_index} state")
            runtime_output.append(output_metric)
            runtime_state.append(state_metric)
            expected_native = _native_pass(output_metric) and _native_pass(state_metric)
            if _bool(pair.get("native_gate_pass"), f"{mode} pair {pair_index} native_gate_pass") != expected_native:
                _fail(f"forged pair native_gate_pass for {mode} pair {pair_index}")
            if pair_index >= WARMUPS:
                measured_gpu_ratios.append(
                    timeline[normalized_indices["sequential"]] / timeline[normalized_indices[mode]]
                )
                measured_encode_ratios.append(
                    per_arm_encode["sequential"][-1] / per_arm_encode[mode][-1]
                )
        if len(measured_gpu_ratios) != MEASURED_PAIRS or len(measured_encode_ratios) != MEASURED_PAIRS:
            _fail(f"{mode} must provide exactly {MEASURED_PAIRS} measured pairs")
        validation_drift = _receipt_summary(checks, mode, rows, blocks)
        validation_native_pass = recomputed_gates[f"{mode}_native_gate_pass"]
        validation_tolerance_pass = (
            recomputed_gates["affine_scan_tolerance_gate_pass"]
            if mode == "affine_scan"
            else all(_tolerance_pass(value) for value in serial_checks)
        )
        pair_metrics = runtime_output + runtime_state
        pair_native_pass = all(_native_pass(value) for value in pair_metrics)
        pair_tolerance_pass = all(_tolerance_pass(value) for value in pair_metrics)
        mode_summaries[mode] = {
            "measured_pairs": MEASURED_PAIRS,
            "warmups_excluded": WARMUPS,
            "paired_gpu_speedup_ratio_sequential_over_candidate": _range_summary(measured_gpu_ratios),
            "paired_host_encode_speedup_ratio_sequential_over_candidate": _range_summary(measured_encode_ratios),
            "host_encode_seconds": {
                arm: _range_summary(per_arm_encode[arm][WARMUPS:]) for arm in ("sequential", mode)
            },
            "gpu_seconds": {
                arm: _range_summary(per_arm_gpu[arm][WARMUPS:]) for arm in ("sequential", mode)
            },
            "validation_drift": validation_drift,
            "all_pair_output_drift_including_warmups": _aggregate_metrics(runtime_output),
            "all_pair_final_state_drift_including_warmups": _aggregate_metrics(runtime_state),
            "measured_pair_output_drift": _aggregate_metrics(runtime_output[WARMUPS:]),
            "measured_pair_final_state_drift": _aggregate_metrics(runtime_state[WARMUPS:]),
            "validation_native_bit_gate_pass": validation_native_pass,
            "timed_pair_receipts_native_gate_pass": pair_native_pass,
            "validation_tolerance_gate_pass": validation_tolerance_pass,
            "timed_pair_receipts_tolerance_gate_pass": pair_tolerance_pass,
            "native_bit_gate_pass": validation_native_pass and pair_native_pass,
            "tolerance_gate_pass": validation_tolerance_pass and pair_tolerance_pass,
        }
    return {
        "key": {"layer": layer, "base": base, "rows": rows, "block_size": block_size},
        "blocks": blocks,
        "ragged_tail": rows % block_size,
        "scratch_bytes": expected_scratch_bytes,
        "modes": mode_summaries,
    }


def _range_summary(values: list[float]) -> dict[str, float]:
    if not values:
        _fail("cannot summarize an empty sample")
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def analyze(report_payload: bytes, log_payload: bytes, *, report_path: str = "report.json", log_path: str = "stderr.log") -> dict[str, Any]:
    report_value = _json_loads(report_payload.decode("utf-8"), "probe report")
    report = _object(report_value, "probe report")
    depth, source_hashes = _validate_top_level(report)
    timeline = _read_timeline(log_payload, depth)
    submitted = report["graph"]["submitted"]
    if any(index >= submitted for index in timeline):
        _fail("stderr GPU command index is outside the report's submitted command count")
    groups_value = report.get("groups")
    if not isinstance(groups_value, list) or len(groups_value) != len(_expected_groups()):
        _fail("groups must contain the complete pinned 12-group grid")
    expected_groups = _expected_groups()
    seen_groups: set[tuple[int, int, int, int]] = set()
    used_indices: set[int] = set()
    summaries = []
    for group in groups_value:
        identity = (
            _integer(group.get("layer"), "group.layer"),
            _integer(group.get("base"), "group.base"),
            _integer(group.get("rows"), "group.rows", minimum=1),
            _integer(group.get("block_size"), "group.block_size", minimum=1),
        ) if isinstance(group, dict) else _fail("group must be an object")
        if identity in seen_groups:
            _fail(f"duplicate group identity {identity}")
        if identity not in expected_groups:
            _fail(f"unexpected group identity {identity}")
        seen_groups.add(identity)
        summaries.append(_validate_group(group, timeline, used_indices))
    if seen_groups != expected_groups:
        _fail(f"group grid is incomplete: missing {sorted(expected_groups - seen_groups)}")
    if not used_indices:
        _fail("no timed pair commands were joined")

    source_manifest_json = json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    model_hashes = report["model_files_sha256"]
    model_manifest_json = json.dumps(model_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "drama3-gdn-scan-analysis-v1",
        "input": {"report": report_path, "stderr_log": log_path},
        "digests": {
            "report_sha256": _sha256_bytes(report_payload),
            "stderr_log_sha256": _sha256_bytes(log_payload),
            "analyzer_source_sha256": sha256_file(pathlib.Path(__file__)),
            "compiled_source_manifest_sha256": _sha256_bytes(source_manifest_json),
            "compiled_source_sha256": source_hashes,
            "model_file_manifest_sha256": _sha256_bytes(model_manifest_json),
            "model_files_sha256": model_hashes,
            "executable_sha256": report["executable_sha256"],
            "source_head": report["source_head"],
        },
        "interpretation": {
            "graph_depth": depth,
            "timed_command_events": len(timeline),
            "joined_timed_pair_commands": len(used_indices),
            "paired_measurements_per_group_mode": MEASURED_PAIRS,
            "warmups_excluded_per_group_mode": WARMUPS,
            "gpu_intervals": (
                "descriptive and non-additive; depth-2 GPU intervals may overlap"
                if depth == 2 else "descriptive GPU command intervals; no whole-request claim"
            ),
            "claim_scope": "actual-input GDN recurrence diagnostics only; no full-prefill or WAV gain claim",
            "native_bit_gate_is_independent_of_tolerance_gate": True,
        },
        "groups": summaries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=pathlib.Path, help="probe JSON report")
    parser.add_argument("log", type=pathlib.Path, help="preserved stderr log")
    parser.add_argument("--output", type=pathlib.Path, help="write analysis JSON here (default: stdout)")
    args = parser.parse_args(argv)
    try:
        report_payload = args.report.read_bytes()
        log_payload = args.log.read_bytes()
        result = analyze(report_payload, log_payload, report_path=str(args.report), log_path=str(args.log))
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        print(f"gdn scan analysis failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
