#!/usr/bin/env python3
"""Fail-closed, ID-aware comparison of the pinned DRAMA-3 layer-39 MoE trace.

The upstream bundle captures the actual MLX sparse sum.  The native wrapper
contains a full-run trace and a replay on the exact upstream normalized input;
only the replay is comparable when the full-run inputs differ.  Auxiliary
weighted terms and prefix sums are reported separately and never stand in for
either implementation's official routed sum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any


UPSTREAM_FORMAT = "drama3-moe-layer-diagnostic-v1"
NATIVE_FORMAT = "drama3-moe-trace-native-v1"
INPUT_FORMAT = "drama3-moe-input-v1"
REPORT_FORMAT = "drama3-moe-comparison-v1"
MODEL_REPOSITORY = "cguldogan/drama-3-mlx-q4"
MODEL_REVISION = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
UPSTREAM_REPOSITORY = "https://github.com/cguldogan/mlx-audio"
UPSTREAM_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
COORDINATE = {"row": 1, "layer": 39}
INPUT_NAME = "post_attention_layernorm"
INPUT_WIDTH = 2048
INTERMEDIATE_WIDTH = 512
EXPERT_COUNT = 256
TOP_K = 8
DEFAULT_ATOL = 6e-5
DEFAULT_RTOL = 3e-4
REQUIRED_SOURCE_HASHES = {
    "config.json",
    "model.safetensors.index.json",
    "mlx_audio/lm/models/qwen3_5.py",
    "mlx_audio/lm/models/switch_layers.py",
    "mlx_audio/utils.py",
    "input_bundle",
    "target_moe_weights",
}
REQUIRED_STAGES = (
    "normalized_input",
    "router_logits",
    "expert_ids_argpartition_order",
    "selected_weights",
    "expert_gate_outputs",
    "expert_up_outputs",
    "expert_activation_outputs",
    "expert_down_outputs",
    "expert_weighted_outputs",
    "routed_sum_official",
    "diagnostic_prefix_sums_argpartition_order",
    "shared_gate_logits",
    "shared_expert_output",
    "shared_weighted_output",
    "moe_output_official",
)
OPTIONAL_STAGES = ("attention_residual", "hidden_output")
UPSTREAM_EXTRA_STAGES = ("router_softmax", "selected_probabilities", "shared_gate_sigmoid")
EXPERT_MATRIX_STAGES = (
    "expert_gate_outputs",
    "expert_up_outputs",
    "expert_activation_outputs",
    "expert_down_outputs",
    "expert_weighted_outputs",
)
AUTHORITATIVE_STAGES = (
    "normalized_input",
    "router_logits",
    "selected_weights",
    "expert_gate_outputs",
    "expert_up_outputs",
    "expert_activation_outputs",
    "expert_down_outputs",
    "routed_sum_official",
    "shared_gate_logits",
    "shared_expert_output",
    "moe_output_official",
    "attention_residual",
    "hidden_output",
)
UPSTREAM_CAPTURE_KINDS = {
    "normalized_input": "upstream_sparse_argument",
    "router_logits": "upstream_linear_return",
    "router_softmax": "upstream_local",
    "selected_probabilities": "upstream_local",
    "selected_weights": "upstream_local",
    "expert_gate_outputs": "upstream_switch_mlp_gate_projection_return",
    "expert_up_outputs": "upstream_switch_mlp_up_projection_return",
    "expert_activation_outputs": "upstream_swiglu_return",
    "expert_down_outputs": "upstream_switch_mlp_return",
    "expert_weighted_outputs": "upstream_sum_operand",
    "routed_sum_official": "upstream_sum_return",
    "diagnostic_prefix_sums_argpartition_order": "diagnostic_recompute_sequential_add",
    "shared_gate_logits": "upstream_linear_return",
    "shared_gate_sigmoid": "upstream_sigmoid_return",
    "shared_expert_output": "upstream_local",
    "shared_weighted_output": "upstream_return_local",
    "moe_output_official": "upstream_return",
    "attention_residual": "upstream_decoder_local",
    "hidden_output": "upstream_target_decoder_output",
}
NATIVE_CAPTURE_KINDS = {
    **{name: "official_in_path" for name in AUTHORITATIVE_STAGES},
    "expert_weighted_outputs": "diagnostic_auxiliary",
    "diagnostic_prefix_sums_argpartition_order": None,
    "shared_weighted_output": "diagnostic_auxiliary",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{where}: cannot read JSON: {exc}") from exc
    require(isinstance(value, dict), f"{where}: expected a JSON object")
    return value


def require_sha256(value: Any, where: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{where}: expected lowercase SHA-256 hex")
    return value


def validate_hashes(value: Any, where: str, *, require_sources: bool = False) -> dict[str, str]:
    require(isinstance(value, dict), f"{where}: expected source_hashes object")
    if require_sources:
        require(REQUIRED_SOURCE_HASHES <= set(value),
                f"{where}: missing required source hashes")
    return {str(name): require_sha256(digest, f"{where}.{name}")
            for name, digest in value.items()}


def f32(value: Any, where: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)),
            f"{where}: expected numeric Float32 value")
    number = float(value)
    require(math.isfinite(number), f"{where}: non-finite value")
    try:
        result = struct.unpack("<f", struct.pack("<f", number))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{where}: value is outside Float32 range") from exc
    require(math.isfinite(result), f"{where}: non-finite Float32 value")
    return result


def f32_values(value: Any, shape: Any, where: str) -> tuple[list[int], list[float]]:
    require(isinstance(shape, list) and bool(shape) and
            all(type(dimension) is int and dimension > 0 for dimension in shape),
            f"{where}: invalid shape")
    require(isinstance(value, list), f"{where}: expected values array")
    count = math.prod(shape)
    require(len(value) == count, f"{where}: expected {count} values, found {len(value)}")
    return shape, [f32(item, f"{where}[{index}]") for index, item in enumerate(value)]


def canonical_f32_sha256(values: list[float]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<f", value))
    return digest.hexdigest()


def safe_sibling(bundle_path: Path, name: Any, where: str) -> Path:
    require(isinstance(name, str) and name not in ("", ".", "..") and
            Path(name).name == name and "/" not in name and "\\" not in name,
            f"{where}: fixture path must be a sibling filename")
    result = bundle_path.parent / name
    require(result.resolve().parent == bundle_path.parent.resolve(),
            f"{where}: fixture must resolve to a sibling file")
    return result


def validate_pinned_identity(value: dict[str, Any], where: str) -> None:
    require(value.get("coordinate") == COORDINATE,
            f"{where}: expected row 1 layer 39")
    model = value.get("model")
    require(isinstance(model, dict) and
            model.get("repository") == MODEL_REPOSITORY and
            model.get("revision") == MODEL_REVISION,
            f"{where}: wrong model metadata")
    upstream = value.get("upstream")
    require(isinstance(upstream, dict) and
            upstream.get("repository") == UPSTREAM_REPOSITORY and
            upstream.get("revision") == UPSTREAM_REVISION,
            f"{where}: wrong upstream metadata")


def validate_model_config(value: Any, where: str) -> None:
    require(isinstance(value, dict), f"{where}: missing pinned model_config")
    expected = {
        "hidden_size": INPUT_WIDTH,
        "moe_intermediate_size": INTERMEDIATE_WIDTH,
        "num_experts": EXPERT_COUNT,
        "num_experts_per_tok": TOP_K,
    }
    for name, expected_value in expected.items():
        actual = value.get(name)
        require(type(actual) is int and actual == expected_value,
                f"{where}.{name}: expected pinned value {expected_value}")


def validate_fixture(
    bundle_path: Path, input_record: dict[str, Any], where: str,
) -> tuple[dict[str, Any], list[float], str, dict[str, str]]:
    require(input_record.get("format") == INPUT_FORMAT,
            f"{where}: wrong input fixture format")
    path = safe_sibling(bundle_path, input_record.get("path"), where)
    claimed_sha = require_sha256(input_record.get("sha256"), f"{where}.sha256")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{where}: cannot read fixture {path}: {exc}") from exc
    actual_sha = hashlib.sha256(raw).hexdigest()
    require(actual_sha == claimed_sha, f"{where}: stale fixture bytes")
    fixture = read_json(path, where)
    require(fixture.get("format") == INPUT_FORMAT, f"{where}: wrong fixture bytes format")
    validate_pinned_identity(fixture, where)
    require(fixture.get("activation_name") == INPUT_NAME,
            f"{where}: wrong activation name")
    require(fixture.get("activation_shape") == [INPUT_WIDTH] and
            fixture.get("activation_dtype") == "float32",
            f"{where}: wrong activation metadata")
    values = fixture.get("activation")
    require(isinstance(values, list) and len(values) == INPUT_WIDTH,
            f"{where}: wrong activation value count")
    normalized = [f32(item, f"{where}.activation[{i}]") for i, item in enumerate(values)]
    activation_sha = canonical_f32_sha256(normalized)
    require(require_sha256(fixture.get("activation_sha256"),
                           f"{where}.activation_sha256") == activation_sha,
            f"{where}: stale activation hash")
    if "activation_sha256" in input_record:
        require(require_sha256(input_record["activation_sha256"],
                               f"{where}.record.activation_sha256") == activation_sha,
                f"{where}: bundle activation hash differs from fixture")
    require(input_record.get("shape") == [INPUT_WIDTH] and
            input_record.get("dtype") == "float32" and
            input_record.get("activation_name") == INPUT_NAME,
            f"{where}: bundle input metadata differs from fixture")
    validate_pinned_identity(fixture, where)
    hashes = validate_hashes(fixture.get("source_hashes"), f"{where}.source_hashes",
                             require_sources=True)
    return fixture, normalized, actual_sha, hashes


def validate_native_weight_provenance(
    native: dict[str, Any], fixture_hashes: dict[str, str],
) -> dict[str, Any]:
    """Require native weights to be attested from loaded raw tensor bytes pre-GPU."""
    where = "native.model_weight_provenance"
    record = native.get("model_weight_provenance")
    require(isinstance(record, dict), f"{where}: missing actual loaded-weight evidence")
    required_fields = {
        "target_layer", "target_moe_weights_sha256", "tensor_count", "model_path",
        "config_sha256", "index_sha256", "verified_before_gpu",
    }
    require(set(record) == required_fields, f"{where}: missing or unexpected evidence fields")
    require(type(record.get("target_layer")) is int and
            record["target_layer"] == COORDINATE["layer"],
            f"{where}.target_layer: expected layer {COORDINATE['layer']}")
    require(type(record.get("tensor_count")) is int and record["tensor_count"] > 0,
            f"{where}.tensor_count: expected a positive tensor count")
    require(isinstance(record.get("model_path"), str) and
            Path(record["model_path"]).is_absolute(),
            f"{where}.model_path: expected expanded absolute model path")
    require(record.get("verified_before_gpu") is True,
            f"{where}.verified_before_gpu: must be true")

    expected = {
        "target_moe_weights_sha256": "target_moe_weights",
        "config_sha256": "config.json",
        "index_sha256": "model.safetensors.index.json",
    }
    for field, source_name in expected.items():
        actual = require_sha256(record.get(field), f"{where}.{field}")
        require(actual == fixture_hashes[source_name],
                f"{where}.{field}: does not match fixture source hash {source_name}")
    return {
        "verified_before_gpu": True,
        "target_layer": record["target_layer"],
        "tensor_count": record["tensor_count"],
        "source_hashes_match_fixture": True,
    }


def parse_tensor(record: Any, where: str, *, dtype: str = "float32") -> dict[str, Any]:
    require(isinstance(record, dict), f"{where}: expected tensor record")
    actual_dtype = record.get("dtype")
    require(actual_dtype == dtype, f"{where}: expected {dtype}, found {actual_dtype!r}")
    shape, values = f32_values(record.get("values"), record.get("shape"), where)
    result = {"shape": shape, "values": values, "capture_kind": record.get("capture_kind")}
    if "expert_ids" in record:
        result["expert_ids"] = parse_ids(record["expert_ids"], f"{where}.expert_ids")
    return result


def parse_ids(record: Any, where: str) -> list[int]:
    if isinstance(record, dict):
        require(record.get("dtype") in ("int32", "int64", "float32"),
                f"{where}: expected integer ID dtype")
        raw = record.get("values")
        shape = record.get("shape")
        require(shape == [TOP_K],
                f"{where}: expected {TOP_K} IDs")
    else:
        raw = record
    require(isinstance(raw, list) and len(raw) == TOP_K,
            f"{where}: expected {TOP_K} IDs")
    ids: list[int] = []
    for index, value in enumerate(raw):
        require(not isinstance(value, bool) and isinstance(value, (int, float)) and
                math.isfinite(float(value)) and float(value).is_integer(),
                f"{where}[{index}]: expected an integer expert ID")
        ids.append(int(value))
    require(len(set(ids)) == TOP_K, f"{where}: duplicate or missing expert IDs")
    return ids


def parse_stages(raw: Any, where: str, *, require_expert_ids: bool) -> dict[str, dict[str, Any]]:
    require(isinstance(raw, dict), f"{where}: expected stages object")
    missing = [name for name in REQUIRED_STAGES if name not in raw]
    require(not missing, f"{where}: missing stages {missing}")
    parsed: dict[str, dict[str, Any]] = {}
    for name, record in raw.items():
        if name == "expert_ids_argpartition_order":
            parsed[name] = {"shape": [TOP_K],
                            "values": parse_ids(record, f"{where}.{name}")}
        else:
            parsed[name] = parse_tensor(record, f"{where}.{name}")
    intermediate_width: int | None = None
    for name in EXPERT_MATRIX_STAGES:
        if name not in parsed:
            continue
        tensor = parsed[name]
        shape = tensor["shape"]
        require(len(shape) == 2 and shape[0] == TOP_K,
                f"{where}.{name}: expected [top_k, width] shape")
        if name in ("expert_down_outputs", "expert_weighted_outputs"):
            require(shape == [TOP_K, INPUT_WIDTH],
                    f"{where}.{name}: expected shape [{TOP_K},{INPUT_WIDTH}]")
        else:
            require(shape == [TOP_K, INTERMEDIATE_WIDTH],
                    f"{where}.{name}: expected shape [{TOP_K},{INTERMEDIATE_WIDTH}]")
            if intermediate_width is None:
                intermediate_width = shape[1]
            else:
                require(shape[1] == intermediate_width,
                        f"{where}.{name}: intermediate width differs from sibling expert stages")
        ids = tensor.get("expert_ids")
        if require_expert_ids:
            require(ids is not None, f"{where}.{name}: explicit expert_ids required")
        if ids is not None:
            require(set(ids) == set(parsed["expert_ids_argpartition_order"]["values"]),
                    f"{where}.{name}: expert IDs do not match selected route IDs")
    require(parsed["selected_weights"]["shape"] == [TOP_K],
            f"{where}.selected_weights: expected shape [{TOP_K}]")
    require(parsed["router_logits"]["shape"] == [EXPERT_COUNT],
            f"{where}.router_logits: expected shape [{EXPERT_COUNT}]")
    expert_count = EXPERT_COUNT
    route_ids = parsed["expert_ids_argpartition_order"]["values"]
    require(all(0 <= expert_id < expert_count for expert_id in route_ids),
            f"{where}: expert ID outside router logits range")
    require(parsed["routed_sum_official"]["shape"] == [INPUT_WIDTH],
            f"{where}.routed_sum_official: expected shape [{INPUT_WIDTH}]")
    require(parsed["diagnostic_prefix_sums_argpartition_order"]["shape"] ==
            [TOP_K, INPUT_WIDTH],
            f"{where}.diagnostic_prefix_sums_argpartition_order: wrong shape")
    for name in ("shared_expert_output", "shared_weighted_output",
                 "moe_output_official"):
        require(parsed[name]["shape"] == [INPUT_WIDTH],
                f"{where}.{name}: expected shape [{INPUT_WIDTH}]")
    for name in OPTIONAL_STAGES:
        if name in raw:
            parsed[name] = parse_tensor(raw[name], f"{where}.{name}")
    return parsed


def validate_stage_capture_kinds(
    stages: dict[str, dict[str, Any]], where: str, *, native_candidate: bool,
) -> None:
    expected = NATIVE_CAPTURE_KINDS if native_candidate else UPSTREAM_CAPTURE_KINDS
    for name, expected_kind in expected.items():
        if name not in stages or expected_kind is None:
            continue
        kind = stages[name].get("capture_kind")
        require(kind == expected_kind,
                f"{where}.{name}: expected capture_kind {expected_kind!r}, found {kind!r}")
    if native_candidate:
        prefix_kind = stages["diagnostic_prefix_sums_argpartition_order"].get("capture_kind")
        require(prefix_kind in ("official_in_path", "diagnostic_auxiliary"),
                f"{where}.diagnostic_prefix_sums_argpartition_order: invalid diagnostic capture kind")


def rank_map(value: Any, route_ids: list[int], where: str) -> list[int]:
    require(isinstance(value, list), f"{where}: expert_rank_to_id is required")
    pairs: dict[int, int] = {}
    for item in value:
        require(isinstance(item, dict) and type(item.get("rank")) is int and
                type(item.get("expert_id")) is int,
                f"{where}: malformed rank-to-expert mapping")
        rank, expert_id = item["rank"], item["expert_id"]
        require(rank not in pairs, f"{where}: duplicate expert rank")
        pairs[rank] = expert_id
    require(set(pairs) == set(range(TOP_K)), f"{where}: missing or extra expert ranks")
    mapped = [pairs[rank] for rank in range(TOP_K)]
    require(len(set(mapped)) == TOP_K and set(mapped) == set(route_ids),
            f"{where}: rank map has duplicate, missing, or unexpected expert IDs")
    require(mapped == route_ids, f"{where}: rank map order differs from route ID order")
    return mapped


def load_inputs(
    upstream_path: Path, upstream: dict[str, Any], native_path: Path, native: dict[str, Any],
) -> tuple[dict[str, Any], list[float], str, dict[str, str], dict[str, Any]]:
    upstream_input = upstream.get("input")
    native_input = native.get("input_fixture")
    require(isinstance(upstream_input, dict), "upstream: missing input fixture metadata")
    require(isinstance(native_input, dict), "native: missing input fixture metadata")
    up_fixture, up_values, up_sha, up_hashes = validate_fixture(
        upstream_path, upstream_input, "upstream input fixture"
    )
    native_record = {
        "format": INPUT_FORMAT,
        "path": native_input.get("path"),
        "sha256": native_input.get("sha256"),
        "shape": [INPUT_WIDTH],
        "dtype": "float32",
        "activation_name": INPUT_NAME,
        "activation_sha256": native_input.get("activation_sha256",
                                               up_fixture["activation_sha256"]),
    }
    nat_fixture, nat_values, nat_sha, nat_hashes = validate_fixture(
        native_path, native_record, "native input fixture"
    )
    require(up_sha == nat_sha, "upstream and native refer to different input fixture bytes")
    require(up_values == nat_values, "upstream and native fixture activations differ")
    require(up_hashes == nat_hashes, "fixture source hashes differ")
    require(up_hashes == validate_hashes(upstream.get("source_hashes"),
                                         "upstream.source_hashes", require_sources=True),
            "upstream bundle and fixture source hashes differ")
    native_hashes = native.get("source_hashes")
    require(isinstance(native_hashes, dict), "native.source_hashes: expected provenance object")
    require(set(native_hashes) == {"native", "fixture"},
            "native.source_hashes: expected native and fixture maps")
    native_code_hashes = validate_hashes(native_hashes["native"],
                                         "native.source_hashes.native")
    native_fixture_hashes = validate_hashes(native_hashes["fixture"],
                                            "native.source_hashes.fixture",
                                            require_sources=True)
    require(native_fixture_hashes == up_hashes,
            "native trace fixture source hashes differ from shared fixture")
    require({"backbone_resident.cr", "backbone_resident.metal", "full_inputs",
             "upstream_input_fixture", "drama3_moe_provenance.cr"} <=
            set(native_code_hashes),
            "native.source_hashes.native: missing required native provenance")
    require(bool(native_code_hashes), "native.source_hashes.native: empty provenance")
    native_weight_provenance = validate_native_weight_provenance(native, up_hashes)
    require(upstream_input.get("activation_sha256") == up_fixture["activation_sha256"],
            "upstream bundle activation hash differs from fixture")
    return up_fixture, up_values, up_sha, up_hashes, native_weight_provenance


def validate_native_capture_evidence(
    native: dict[str, Any], runs: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Require the trace's single-drain and final-capture controls to hold.

    The trace format records every terminal tensor read. Aggregate expert
    records are assembled by reading each selected rank separately, so their
    read-call count is TOP_K even though each aggregate is one JSON record.
    The capture control compares four 2048-wide outputs on both contexts.
    These relations make added diagnostic reads or graph waits invalidate a
    qualified report instead of treating control fields as decoration.
    """
    capture = native.get("capture_control")
    require(isinstance(capture, dict), "native.capture_control: missing evidence")
    require(set(capture) == {"passed", "comparison", "count", "compared_values", "mismatches"},
            "native.capture_control: unexpected evidence fields")
    require(capture.get("passed") is True and
            capture.get("comparison") == "Float32 bitwise equality" and
            type(capture.get("count")) is int and capture["count"] == 4 and
            type(capture.get("compared_values")) is int and
            capture["compared_values"] == 4 * INPUT_WIDTH and
            capture.get("mismatches") == [],
            "native.capture_control: expected four bitwise-equal 2048-value outputs")

    graph_keys = {"depth", "submitted", "completed", "pending", "max_pending",
                  "completion_waits", "retained_bytes", "max_retained_bytes"}
    graph_values: dict[str, dict[str, int]] = {}
    for name in ("graph_before_drain", "graph_after_drain", "graph_after_serialization"):
        graph = native.get(name)
        require(isinstance(graph, dict) and set(graph) == graph_keys,
                f"native.{name}: missing or unexpected graph counters")
        require(all(type(value) is int and value >= 0 for value in graph.values()),
                f"native.{name}: graph counters must be non-negative integers")
        graph_values[name] = graph

    before = graph_values["graph_before_drain"]
    require(before == {"depth": 2, "submitted": 0, "completed": 0, "pending": 0,
                       "max_pending": 0, "completion_waits": 0, "retained_bytes": 0,
                       "max_retained_bytes": 0},
            "native.graph_before_drain: expected idle depth-2 graph before final drain")
    after = graph_values["graph_after_drain"]
    require(after["depth"] == 2 and after["submitted"] == 1 and
            after["completed"] == 1 and after["pending"] == 0 and
            after["max_pending"] == 1 and after["completion_waits"] == 1 and
            after["retained_bytes"] == 0 and
            after["max_retained_bytes"] >= after["retained_bytes"],
            "native.graph_after_drain: expected exactly one final submit and completion wait")
    require(graph_values["graph_after_serialization"] == after,
            "native.graph_after_serialization: graph changed during serialization")

    readbacks = native.get("readbacks")
    require(isinstance(readbacks, dict) and set(readbacks) == {"control", "terminal"} and
            all(type(value) is int and value >= 0 for value in readbacks.values()),
            "native.readbacks: expected non-negative control and terminal counters")
    stage_values = 0
    stage_read_calls = 0
    aggregate_names = set(EXPERT_MATRIX_STAGES) | {
        "diagnostic_prefix_sums_argpartition_order"
    }
    for run_name, stages in runs.items():
        for stage_name, stage in stages.items():
            stage_values += len(stage["values"])
            shape = stage["shape"]
            stage_read_calls += (
                TOP_K if stage_name in aggregate_names and len(shape) == 2 and
                shape[0] == TOP_K else 1
            )
    capture_reads = 2 * capture["count"]
    expected_control = 4 * (stage_read_calls + capture_reads)
    expected_terminal = 4 * (stage_values + 2 * capture["compared_values"])
    require(readbacks["control"] == expected_control,
            f"native.readbacks.control: expected {expected_control}, found {readbacks['control']}")
    require(readbacks["terminal"] == expected_terminal,
            f"native.readbacks.terminal: expected {expected_terminal}, found {readbacks['terminal']}")


def input_stage_matches(stages: dict[str, dict[str, Any]], values: list[float], where: str) -> bool:
    stage = stages["normalized_input"]
    require(stage["shape"] == [INPUT_WIDTH], f"{where}: wrong normalized input shape")
    return (stage["values"] == values and
            canonical_f32_sha256(stage["values"]) == canonical_f32_sha256(values))


def expert_rows(tensor: dict[str, Any], where: str) -> dict[int, list[float]]:
    ids = tensor.get("expert_ids")
    require(isinstance(ids, list) and len(ids) == TOP_K,
            f"{where}: missing explicit expert IDs")
    width = tensor["shape"][1]
    return {expert_id: tensor["values"][rank * width:(rank + 1) * width]
            for rank, expert_id in enumerate(ids)}


def aligned_vector(stage: dict[str, Any], ids: list[int], where: str) -> dict[int, float]:
    require(stage["shape"] == [TOP_K], f"{where}: wrong selected-weight shape")
    require(len(ids) == TOP_K and len(set(ids)) == TOP_K,
            f"{where}: invalid route ID mapping")
    return {expert_id: stage["values"][rank] for rank, expert_id in enumerate(ids)}


def compare_values(reference: list[float], actual: list[float], atol: float,
                   rtol: float) -> dict[str, Any]:
    require(len(reference) == len(actual), "internal comparison length mismatch")
    worst_abs = -1.0
    worst: dict[str, float | int] = {}
    failures = 0
    for index, (expected, observed) in enumerate(zip(reference, actual)):
        absolute = abs(observed - expected)
        allowed = atol + rtol * abs(expected)
        if absolute > allowed:
            failures += 1
        if absolute > worst_abs:
            worst_abs = absolute
            worst = {"index": index, "reference": expected, "actual": observed,
                     "absolute_error": absolute, "allowed_error": allowed}
    return {"status": "match" if failures == 0 else "mismatch",
            "count": len(reference), "failures": failures,
            "max_absolute_error": max(0.0, worst_abs), "worst": worst}


def compare_stage(name: str, reference: dict[str, Any], actual: dict[str, Any],
                  atol: float, rtol: float) -> dict[str, Any]:
    require(reference["shape"] == actual["shape"],
            f"stage {name}: shape mismatch {reference['shape']} vs {actual['shape']}")
    if name in EXPERT_MATRIX_STAGES:
        ref_rows, actual_rows = expert_rows(reference, f"upstream.{name}"), expert_rows(
            actual, f"native.{name}"
        )
        require(set(ref_rows) == set(actual_rows), f"stage {name}: expert IDs differ")
        per_expert = {
            str(expert_id): compare_values(ref_rows[expert_id], actual_rows[expert_id], atol, rtol)
            for expert_id in sorted(ref_rows)
        }
        failures = sum(row["failures"] for row in per_expert.values())
        worst = max(per_expert.items(), key=lambda item: item[1]["max_absolute_error"])
        return {"status": "match" if failures == 0 else "mismatch",
                "authority": ("auxiliary" if "auxiliary" in str(actual.get("capture_kind", ""))
                              else "captured_stage"),
                "count": sum(row["count"] for row in per_expert.values()),
                "failures": failures, "per_expert": per_expert,
                "max_absolute_error": worst[1]["max_absolute_error"],
                "worst_expert_id": int(worst[0]),
                "native_capture_kind": actual.get("capture_kind")}
    return compare_values(reference["values"], actual["values"], atol, rtol) | {
        "authority": "official" if name in ("routed_sum_official", "moe_output_official")
        else ("auxiliary" if "auxiliary" in str(actual.get("capture_kind", ""))
              else "captured_stage"),
        "native_capture_kind": actual.get("capture_kind"),
    }


def cancellation_diagnostic(
    stages: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Describe cancellation without changing the fixed pass/fail tolerance."""
    actual_terms = expert_rows(stages["expert_weighted_outputs"], "native weighted outputs")
    ref_terms = expert_rows(reference["expert_weighted_outputs"], "upstream weighted outputs")
    actual_sum = stages["routed_sum_official"]["values"]
    ref_sum = reference["routed_sum_official"]["values"]
    route = stages["expert_ids_argpartition_order"]["values"]
    max_condition = 0.0
    max_delta_over_bound = 0.0
    worst_index = 0
    unit_roundoff = 2.0 ** -24
    gamma = ((TOP_K - 1) * unit_roundoff) / (1.0 - (TOP_K - 1) * unit_roundoff)
    for index in range(INPUT_WIDTH):
        total_abs = sum(abs(actual_terms[expert_id][index]) for expert_id in route)
        denominator = abs(actual_sum[index])
        condition = total_abs / max(denominator, 1e-38)
        bound = 2.0 * gamma * total_abs
        delta = abs(actual_sum[index] - ref_sum[index])
        ratio = delta / max(bound, 1e-38)
        if condition > max_condition:
            max_condition, worst_index = condition, index
        max_delta_over_bound = max(max_delta_over_bound, ratio)
    return {
        "max_condition_ratio": max_condition,
        "worst_index": worst_index,
        "max_sum_delta_over_two_gamma_bound": max_delta_over_bound,
        "two_gamma_bound_uses": "2*gamma_7*sum(abs(native weighted terms)), Float32 u=2^-24",
        "interpretation": "diagnostic_only; does not widen tolerance or establish root cause",
        "weighted_terms_by_id_compared": True,
        "native_weighted_terms_capture_kind":
            stages["expert_weighted_outputs"].get("capture_kind"),
        "native_terms_authoritative":
            "auxiliary" not in str(stages["expert_weighted_outputs"].get("capture_kind", "")),
    }


def compare_run(
    upstream_stages: dict[str, dict[str, Any]],
    native_stages: dict[str, dict[str, Any]],
    upstream_ids: list[int],
    native_ids: list[int],
    upstream_input_match: bool,
    atol: float,
    rtol: float,
    where: str,
) -> dict[str, Any]:
    if not upstream_input_match:
        return {"status": "different_input", "input_bitwise_equal": False,
                "stages_compared": {},
                "reason": "normalized Float32 activations differ; numeric stages not compared"}
    input_record = compare_values(upstream_stages["normalized_input"]["values"],
                                  native_stages["normalized_input"]["values"], 0.0, 0.0)
    stages: dict[str, Any] = {}
    pipeline = ("router_logits", "expert_gate_outputs",
                "expert_up_outputs", "expert_activation_outputs", "expert_down_outputs",
                "expert_weighted_outputs", "routed_sum_official", "shared_gate_logits",
                "shared_expert_output", "shared_weighted_output", "moe_output_official",
                "attention_residual", "hidden_output")
    route_set_equal = set(upstream_ids) == set(native_ids)
    stages["expert_ids_argpartition_order"] = {
        "status": "match" if route_set_equal else "mismatch",
        "authority": "routing",
        "upstream_order": upstream_ids,
        "native_order": native_ids,
        "same_expert_set": route_set_equal,
        "same_order": upstream_ids == native_ids,
        "order_difference_is_reported_but_not_failed_when_set_matches": True,
    }
    if route_set_equal:
        ref_weights = aligned_vector(upstream_stages["selected_weights"], upstream_ids,
                                     "upstream.selected_weights")
        nat_weights = aligned_vector(native_stages["selected_weights"], native_ids,
                                     "native.selected_weights")
        ids = sorted(ref_weights)
        stages["selected_weights_by_expert_id"] = compare_values(
            [ref_weights[expert_id] for expert_id in ids],
            [nat_weights[expert_id] for expert_id in ids], atol, rtol,
        ) | {"expert_ids": ids, "authority": "captured_stage"}
    else:
        stages["selected_weights_by_expert_id"] = {
            "status": "not_compared", "reason": "selected expert sets differ"}
    for name in pipeline:
        if name not in upstream_stages or name not in native_stages:
            if name in OPTIONAL_STAGES and name not in upstream_stages and name not in native_stages:
                continue
            stages[name] = {"status": "unavailable", "reason": "stage absent on one side"}
            continue
        if name in EXPERT_MATRIX_STAGES and not route_set_equal:
            stages[name] = {"status": "not_compared", "reason": "selected expert sets differ"}
            continue
        stages[name] = compare_stage(name, upstream_stages[name], native_stages[name], atol, rtol)

    prefix_order_equal = upstream_ids == native_ids
    up_prefix = upstream_stages["diagnostic_prefix_sums_argpartition_order"]
    nat_prefix = native_stages["diagnostic_prefix_sums_argpartition_order"]
    prefix_authority = {
        "upstream_capture_kind": up_prefix.get("capture_kind"),
        "native_capture_kind": nat_prefix.get("capture_kind"),
        "upstream_is_diagnostic_only": True,
        "native_is_diagnostic_only": "auxiliary" in str(nat_prefix.get("capture_kind", "")),
        "official_sum_stage": "routed_sum_official",
    }
    if prefix_order_equal and up_prefix["shape"] == nat_prefix["shape"]:
        prefix_authority["same_order_diagnostic_comparison"] = compare_values(
            up_prefix["values"], nat_prefix["values"], atol, rtol
        )
    else:
        prefix_authority["same_order_diagnostic_comparison"] = {
            "status": "not_compared",
            "reason": "rank order differs; prefix coordinates are order-dependent",
        }

    failures = [name for name, result in stages.items()
                if result.get("status") == "mismatch" and
                result.get("authority") != "auxiliary"]
    auxiliary_failures = [name for name, result in stages.items()
                          if result.get("status") == "mismatch" and
                          result.get("authority") == "auxiliary"]
    if stages["expert_ids_argpartition_order"]["status"] == "mismatch":
        first = "expert_ids_argpartition_order"
    elif stages.get("selected_weights_by_expert_id", {}).get("status") == "mismatch":
        first = "selected_weights_by_expert_id"
    else:
        first = next((name for name in pipeline
                      if stages.get(name, {}).get("status") == "mismatch" and
                      stages.get(name, {}).get("authority") != "auxiliary"), None)
    return {
        "status": "match" if not failures else "mismatch",
        "input_bitwise_equal": True,
        "input_comparison": input_record,
        "expert_order": stages["expert_ids_argpartition_order"],
        "stages": stages,
        "first_failing_stage": first,
        "auxiliary_diagnostic_mismatches": auxiliary_failures,
        "diagnostic_prefix_reductions": prefix_authority,
        "cancellation": cancellation_diagnostic(native_stages, upstream_stages)
        if route_set_equal else {"status": "not_computed", "reason": "expert sets differ"},
        "numeric_failure_stages": failures,
    }


def bitwise_values(left: list[float], right: list[float]) -> dict[str, Any]:
    require(len(left) == len(right), "internal bitwise comparison length mismatch")
    changed = [index for index, (a, b) in enumerate(zip(left, right))
               if struct.pack("<f", a) != struct.pack("<f", b)]
    max_abs = max((abs(a - b) for a, b in zip(left, right)), default=0.0)
    return {"status": "match" if not changed else "mismatch",
            "count": len(left), "bitwise_mismatches": len(changed),
            "first_mismatch_index": changed[0] if changed else None,
            "max_absolute_difference": max_abs}


def compare_upstream_control(
    full: dict[str, dict[str, Any]], isolated: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    full_ids = full["expert_ids_argpartition_order"]["values"]
    isolated_ids = isolated["expert_ids_argpartition_order"]["values"]
    same_set = set(full_ids) == set(isolated_ids)
    stages: dict[str, Any] = {
        "expert_ids_argpartition_order": {
            "status": "match" if same_set else "mismatch",
            "full_order": full_ids,
            "isolated_order": isolated_ids,
            "same_expert_set": same_set,
            "same_order": full_ids == isolated_ids,
        }
    }
    if same_set:
        full_weights = aligned_vector(full["selected_weights"], full_ids,
                                      "full reference selected weights")
        isolated_weights = aligned_vector(isolated["selected_weights"], isolated_ids,
                                          "isolated reference selected weights")
        expert_ids = sorted(full_weights)
        stages["selected_weights_by_expert_id"] = bitwise_values(
            [full_weights[i] for i in expert_ids],
            [isolated_weights[i] for i in expert_ids],
        ) | {"expert_ids": expert_ids}
    else:
        stages["selected_weights_by_expert_id"] = {
            "status": "not_compared", "reason": "selected expert sets differ"}
    names = ("router_logits", "router_softmax", "selected_probabilities",
             "expert_gate_outputs", "expert_up_outputs", "expert_activation_outputs",
             "expert_down_outputs", "expert_weighted_outputs", "routed_sum_official",
             "shared_gate_logits", "shared_gate_sigmoid", "shared_expert_output",
             "shared_weighted_output", "moe_output_official", "attention_residual",
             "hidden_output")
    for name in names:
        if name not in full or name not in isolated:
            if name in OPTIONAL_STAGES and name not in full and name not in isolated:
                continue
            stages[name] = {"status": "unavailable", "reason": "stage absent on one side"}
            continue
        if name in EXPERT_MATRIX_STAGES:
            if not same_set:
                stages[name] = {"status": "not_compared",
                                "reason": "selected expert sets differ"}
                continue
            full_rows = expert_rows(full[name], f"full reference {name}")
            isolated_rows = expert_rows(isolated[name], f"isolated reference {name}")
            per_id = {str(i): bitwise_values(full_rows[i], isolated_rows[i])
                      for i in sorted(full_rows)}
            changed = sum(value["bitwise_mismatches"] for value in per_id.values())
            max_diff = max((value["max_absolute_difference"] for value in per_id.values()),
                           default=0.0)
            stages[name] = {"status": "match" if changed == 0 else "mismatch",
                            "bitwise_mismatches": changed,
                            "max_absolute_difference": max_diff,
                            "per_expert_id": per_id}
            continue
        require(full[name]["shape"] == isolated[name]["shape"],
                f"upstream control {name}: shape mismatch")
        stages[name] = bitwise_values(full[name]["values"], isolated[name]["values"])
    prefix_note: dict[str, Any] = {
        "authority": "diagnostic_only",
        "official_sum_stage": "routed_sum_official",
        "same_route_order": full_ids == isolated_ids,
    }
    if full_ids == isolated_ids:
        prefix_note["bitwise_comparison"] = bitwise_values(
            full["diagnostic_prefix_sums_argpartition_order"]["values"],
            isolated["diagnostic_prefix_sums_argpartition_order"]["values"],
        )
    else:
        prefix_note["bitwise_comparison"] = {
            "status": "not_compared", "reason": "prefixes depend on route order"}
    failures = [name for name, item in stages.items() if item.get("status") == "mismatch"]
    first = next((name for name in names if stages.get(name, {}).get("status") == "mismatch"), None)
    if stages["expert_ids_argpartition_order"]["status"] == "mismatch":
        first = "expert_ids_argpartition_order"
    elif stages.get("selected_weights_by_expert_id", {}).get("status") == "mismatch":
        first = "selected_weights_by_expert_id"
    return {"status": "match" if not failures else "mismatch",
            "bitwise_equivalent": not failures,
            "same_expert_set": same_set,
            "same_route_order": full_ids == isolated_ids,
            "stages": stages,
            "first_failing_stage": first,
            "diagnostic_prefix_reductions": prefix_note,
            "failure_stages": failures}


def compare(upstream_path: Path, native_path: Path, atol: float = DEFAULT_ATOL,
            rtol: float = DEFAULT_RTOL,
            upstream_isolated_path: Path | None = None) -> dict[str, Any]:
    require(math.isfinite(atol) and atol >= 0 and math.isfinite(rtol) and rtol >= 0,
            "tolerances must be finite and non-negative")
    require(atol <= DEFAULT_ATOL and rtol <= DEFAULT_RTOL,
            f"tolerances may only be tightened from atol={DEFAULT_ATOL} and rtol={DEFAULT_RTOL}")
    upstream = read_json(upstream_path, "upstream")
    native = read_json(native_path, "native")
    require(upstream.get("format") == UPSTREAM_FORMAT, "upstream: unexpected format")
    require(native.get("format") == NATIVE_FORMAT, "native: unexpected format")
    validate_pinned_identity(upstream, "upstream")
    validate_pinned_identity(native, "native")
    require(upstream.get("mode") in ("full_run", "isolated_replay"),
            "upstream: unexpected capture mode")
    validate_model_config(upstream.get("model_config"), "upstream.model_config")
    require(native.get("coordinate") == COORDINATE, "native: expected row 1 layer 39")
    execution = native.get("execution")
    require(isinstance(execution, dict) and
            type(execution.get("batch_experts_requested")) is bool and
            type(execution.get("full_native_batch_path")) is bool and
            type(execution.get("isolated_batch_path")) is bool and
            execution.get("dense_reduction") in ("tree", "simd"),
            "native: invalid execution metadata")
    fixture, fixture_values, fixture_sha, source_hashes, weight_provenance = load_inputs(
        upstream_path, upstream, native_path, native
    )
    up_stages = parse_stages(upstream.get("stages"), "upstream.stages",
                             require_expert_ids=True)
    for name in UPSTREAM_EXTRA_STAGES:
        require(name in up_stages, f"upstream.stages: missing stage {name}")
    require(up_stages["router_softmax"]["shape"] == [EXPERT_COUNT],
            "upstream.stages.router_softmax: wrong shape")
    require(up_stages["selected_probabilities"]["shape"] == [TOP_K],
            "upstream.stages.selected_probabilities: wrong shape")
    require(up_stages["shared_gate_sigmoid"]["shape"] == [1],
            "upstream.stages.shared_gate_sigmoid: wrong shape")
    require(input_stage_matches(up_stages, fixture_values, "upstream normalized input"),
            "upstream normalized input differs from input fixture")
    validate_stage_capture_kinds(up_stages, "upstream.stages", native_candidate=False)
    up_ids = up_stages["expert_ids_argpartition_order"]["values"]
    up_rank = upstream.get("expert_rank_to_id")
    if up_rank is not None:
        rank_map(up_rank, up_ids, "upstream.expert_rank_to_id")

    raw_runs = native.get("runs")
    require(isinstance(raw_runs, list), "native: runs must be an array")
    runs: dict[str, dict[str, Any]] = {}
    for item in raw_runs:
        require(isinstance(item, dict) and item.get("name") in
                ("full_native", "isolated_upstream_input"),
                "native: unexpected run record")
        name = item["name"]
        require(name not in runs, f"native: duplicate run {name}")
        stages = parse_stages(item.get("stages"), f"native.{name}.stages",
                              require_expert_ids=False)
        ids = stages["expert_ids_argpartition_order"]["values"]
        rank_map(item.get("expert_rank_to_id"), ids, f"native.{name}.expert_rank_to_id")
        for tensor_name in EXPERT_MATRIX_STAGES:
            if tensor_name not in stages:
                continue
            if stages[tensor_name].get("expert_ids") is None:
                stages[tensor_name]["expert_ids"] = list(ids)
            else:
                require(set(stages[tensor_name]["expert_ids"]) == set(ids),
                        f"native.{name}.{tensor_name}: tensor IDs differ from rank map")
        validate_stage_capture_kinds(stages, f"native.{name}.stages", native_candidate=True)
        runs[name] = stages
    require(set(runs) == {"full_native", "isolated_upstream_input"},
            "native: expected full_native and isolated_upstream_input runs")
    validate_native_capture_evidence(native, runs)
    isolated_input_matches = input_stage_matches(
        runs["isolated_upstream_input"], fixture_values, "native isolated input"
    )
    require(isolated_input_matches,
            "native isolated replay input differs from shared upstream fixture")
    full_input_matches = input_stage_matches(runs["full_native"], fixture_values,
                                             "native full input")
    full_native = compare_run(
        up_stages, runs["full_native"], up_ids,
        runs["full_native"]["expert_ids_argpartition_order"]["values"],
        full_input_matches, atol, rtol, "full_native",
    )
    isolated = compare_run(
        up_stages, runs["isolated_upstream_input"], up_ids,
        runs["isolated_upstream_input"]["expert_ids_argpartition_order"]["values"],
        isolated_input_matches, atol, rtol, "isolated_upstream_input",
    )
    # The native full-run activation can legitimately differ from the shared
    # upstream fixture. Keep its input mismatch visible, but qualify the
    # exact-input replay; compare full_native numerically only when its input
    # is bitwise identical to the fixture.
    statuses = [isolated["status"]]
    if full_input_matches:
        statuses.append(full_native["status"])
    control: dict[str, Any] | None = None
    if upstream_isolated_path is not None:
        isolated_upstream = read_json(upstream_isolated_path, "isolated upstream")
        require(isolated_upstream.get("format") == UPSTREAM_FORMAT,
                "isolated upstream: unexpected format")
        validate_pinned_identity(isolated_upstream, "isolated upstream")
        require(isolated_upstream.get("mode") == "isolated_replay",
                "isolated upstream: expected isolated_replay mode")
        validate_model_config(isolated_upstream.get("model_config"),
                              "isolated upstream.model_config")
        second_fixture_record = isolated_upstream.get("input")
        require(isinstance(second_fixture_record, dict),
                "isolated upstream: missing input fixture metadata")
        second_fixture, second_input, second_sha, second_hashes = validate_fixture(
            upstream_isolated_path, second_fixture_record, "isolated upstream input fixture"
        )
        require(second_sha == fixture_sha and second_input == fixture_values,
                "full and isolated upstream input fixtures differ")
        require(second_hashes == source_hashes and
                validate_hashes(isolated_upstream.get("source_hashes"),
                                "isolated upstream.source_hashes", require_sources=True)
                == source_hashes,
                "full and isolated upstream source hashes differ")
        full_input = parse_stages(upstream.get("stages"), "upstream.stages",
                                  require_expert_ids=True)
        isolated_stages = parse_stages(isolated_upstream.get("stages"),
                                       "isolated_upstream.stages",
                                       require_expert_ids=True)
        for name in UPSTREAM_EXTRA_STAGES:
            require(name in isolated_stages,
                    f"isolated_upstream.stages: missing stage {name}")
        require(isolated_stages["router_softmax"]["shape"] == [EXPERT_COUNT] and
                isolated_stages["selected_probabilities"]["shape"] == [TOP_K] and
                isolated_stages["shared_gate_sigmoid"]["shape"] == [1],
                "isolated_upstream.stages: routing/shared sigmoid stage has wrong shape")
        require(input_stage_matches(full_input, fixture_values,
                                    "upstream full normalized input") and
                input_stage_matches(isolated_stages, fixture_values,
                                    "upstream isolated normalized input"),
                "upstream full/isolated normalized input differs from fixture")
        validate_stage_capture_kinds(isolated_stages, "isolated_upstream.stages",
                                     native_candidate=False)
        control = compare_upstream_control(full_input, isolated_stages)
        control["input_activation_sha256"] = second_fixture["activation_sha256"]
        statuses.append(control["status"])
    overall = "match" if all(status == "match" for status in statuses) else "mismatch"
    report = {
        "format": REPORT_FORMAT,
        "status": overall,
        "qualified": overall == "match",
        "coordinate": COORDINATE,
        "model": upstream["model"],
        "upstream": upstream["upstream"],
        "input_fixture": {"sha256": fixture_sha,
                           "activation_sha256": fixture["activation_sha256"],
                           "source_hashes_match": True},
        "native_weight_provenance": weight_provenance,
        "qualification_scope": (
            "full_native_and_isolated_exact_input_replay"
            if full_input_matches else
            "isolated_exact_input_replay_only; full_native_original_input_differs"
        ),
        "full_native_matches_input_fixture": full_input_matches,
        "full_native_original_discrepancy_qualified": False if not full_input_matches else None,
        "tolerances": {"atol": atol, "rtol": rtol,
                       "policy": "fixed project gate; widening rejected"},
        "execution": native["execution"],
        "full_native": full_native,
        "isolated_upstream_input": isolated,
        "interpretation": (
            "diagnostic comparison only; mismatch does not establish root cause; "
            "an isolated exact-input pass does not qualify a different-input full run"
        ),
    }
    if control is not None:
        report["upstream_full_vs_isolated_control"] = control
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True,
                        help="pinned MLX layer-39 diagnostic bundle")
    parser.add_argument("--native", type=Path, required=True,
                        help="native full-run and isolated-input trace bundle")
    parser.add_argument("--upstream-isolated", type=Path,
                        help="optional pinned MLX replay of the same fixture activation")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    args = parser.parse_args()
    try:
        report = compare(args.upstream, args.native, args.atol, args.rtol,
                         args.upstream_isolated)
    except (AssertionError, ValueError) as exc:
        print(json.dumps({"format": REPORT_FORMAT, "status": "invalid",
                          "qualified": False, "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
