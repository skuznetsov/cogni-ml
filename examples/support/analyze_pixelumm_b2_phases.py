#!/usr/bin/env python3
"""Fail-closed CPU-only phase analysis for paired-window B2 full36 reports.

This validates report-local schedules, saved capture comparisons, command-root
intervals, all 36 block marker sequences, MPS API counts, and continuation
traces. Host markers, API spans, and GPU root timestamps are separate clocks:
root duration sums are not active shader time or additive phase work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_SUPPORT = Path(__file__).resolve().parent
if str(_SUPPORT) not in sys.path:
    sys.path.insert(0, str(_SUPPORT))
import analyze_pixelumm_order_profile as common  # noqa: E402


REPORT = "cogni_ml.pixelumm_window_batch_full_probe/v2"
PROFILE = "cogni_ml.pixelumm_decoder_phase_profile/v1"
SUMMARY = "cogni_ml.pixelumm_b2_full36_phase_analysis/v1"
BLOCKS = 36
TILES_PER_BLOCK = 60
EXPECTED_ROOTS = {False: 4579, True: 2779}
PHASES = common.PHASES
MARKERS = common.MARKERS
API_PHASES = common.API_PHASES
PIXELUMM_API_PHASES = common.PIXELUMM_API_PHASES
OPS = common.OPS
API_FIELDS = common.API_FIELDS
ROOT_SEMANTICS = (
    "Candidate root-duration sums are non-additive: shared roots may appear in multiple phase candidates "
    "and root GPU intervals may overlap; they are not per-phase GPU work."
)
FIXTURE_PIN_FIELDS = (
    "reference_sha256", "alternate_reference_sha256", "head_reference_sha256",
    "head_alternate_reference_sha256", "bf16_cache_sha256",
)
CAPTURE_REQUESTS = {0, 1, 2, 3, 13, 14}
PROFILED_TIMED_OFFSETS = {2, 3, 4, 5}


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def obj(value: object, where: str) -> dict[str, Any]:
    need(isinstance(value, dict), f"{where}: expected object")
    return value  # type: ignore[return-value]


def arr(value: object, where: str) -> list[Any]:
    need(isinstance(value, list), f"{where}: expected array")
    return value  # type: ignore[return-value]


def exact_keys(value: dict[str, Any], keys: tuple[str, ...] | set[str], where: str) -> None:
    need(set(value) == set(keys), f"{where}: missing or unexpected fields")


def uint(value: object, where: str, minimum: int = 0) -> int:
    need(isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= (1 << 64) - 1,
         f"{where}: expected unsigned integer")
    return int(value)


def finite(value: object, where: str, minimum: float = 0.0) -> float:
    need(isinstance(value, (int, float)) and not isinstance(value, bool), f"{where}: expected number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{where}: invalid number") from exc
    need(math.isfinite(result) and result >= minimum, f"{where}: non-finite or out of range")
    return result


def boolean(value: object, where: str) -> bool:
    need(isinstance(value, bool), f"{where}: expected boolean")
    return value


def _root_profile(raw: object, index: int, expected_count: int) -> tuple[list[tuple[float, float]], dict[str, float]]:
    profile = obj(raw, f"request {index} root_profile")
    exact_keys(profile, ("root_count", "root_duration_seconds", "envelope_seconds", "union_seconds",
                         "overlap_seconds", "roots"), f"request {index} root_profile")
    count = uint(profile["root_count"], "root_profile.root_count", minimum=1)
    need(count == expected_count, f"request {index}: command root count differs from selected topology")
    intervals, _ = common._root_intervals(arr(profile["roots"], "root_profile.roots"),
                                           f"request {index} root_profile", profile=False)
    need(len(intervals) == count, f"request {index}: root inventory/count mismatch")
    metrics = common._root_metrics(intervals)
    for key in ("root_duration_seconds", "envelope_seconds", "union_seconds", "overlap_seconds"):
        need(common.near(finite(profile[key], f"root_profile.{key}"), metrics[key], 2e-8, 1e-8),
             f"request {index}: root_profile.{key} disagrees with root intervals")
    return intervals, metrics


def _validate_api(profile: dict[str, Any], trace: dict[str, Any], request_index: int,
                  candidate: bool, root_count: int) -> dict[int, dict[str, dict[str, int]]]:
    api = obj(profile.get("api_profile"), f"request {request_index}: missing API profile")
    exact_keys(api, ("source_schema", "phases"), "API profile")
    need(api["source_schema"] == "MPSAPIProfile WAN phase IDs", "API profile source schema changed")
    rows = arr(api["phases"], "API phases")
    need(len(rows) == 6, f"request {request_index}: expected six API phase records")
    indexed: dict[int, dict[str, dict[str, int]]] = {}
    for raw_phase in rows:
        phase = obj(raw_phase, "API phase")
        exact_keys(phase, ("pixelumm_phase", "source_phase_id", "source_phase_name", "operations"), "API phase")
        phase_id = uint(phase["source_phase_id"], "API phase id")
        need(phase_id in API_PHASES and phase_id not in indexed and
             phase["source_phase_name"] == API_PHASES[phase_id] and
             phase["pixelumm_phase"] == PIXELUMM_API_PHASES[phase_id],
             "API phase identity/name mismatch")
        operations = obj(phase["operations"], "API operations")
        exact_keys(operations, OPS, "API operations")
        parsed: dict[str, dict[str, int]] = {}
        for operation in OPS:
            raw_stats = obj(operations[operation], f"API {operation}")
            exact_keys(raw_stats, API_FIELDS, f"API {operation}")
            stats = {key: uint(raw_stats[key], f"API {operation}.{key}") for key in API_FIELDS}
            common._validate_api_operation_stats(stats, operation, request_index)
            parsed[operation] = stats
        graph, continuation = parsed["graph_encode"], parsed["commit_and_continue"]
        need(tuple(graph[key] for key in ("nested_count", "nested_wall_total_ns", "nested_thread_cpu_total_ns")) ==
             tuple(continuation[key] for key in ("nested_count", "nested_wall_total_ns", "nested_thread_cpu_total_ns")),
             f"request {request_index}: Graph/continuation nested totals disagree")
        indexed[phase_id] = parsed

    need(set(indexed) == set(API_PHASES), f"request {request_index}: API phase matrix is incomplete")
    attention_graphs = 3 * (TILES_PER_BLOCK // 2 if candidate else TILES_PER_BLOCK) * BLOCKS
    expected_graphs = {0: 5, 1: 9 * BLOCKS, 2: attention_graphs,
                       3: 3 * BLOCKS, 4: 0, 5: 3 * BLOCKS}
    need({phase_id: indexed[phase_id]["graph_encode"]["count"] for phase_id in API_PHASES} == expected_graphs,
         f"request {request_index}: Graph phase counts do not match independent/B2 full36 geometry")
    expected_total = 5 + 15 * BLOCKS + attention_graphs
    need(sum(expected_graphs.values()) == expected_total == (3785 if candidate else 7025) and
         all(indexed[phase_id]["t5_linear_encode"]["count"] == 0 for phase_id in API_PHASES),
         f"request {request_index}: Graph total or T5 linear count differs from full36 topology")

    for phase_id in API_PHASES:
        events = [event for event in trace["events"] if event["phase"] == phase_id]
        continuation = indexed[phase_id]["commit_and_continue"]
        need(continuation["count"] == len(events) and
             continuation["nested_count"] == sum(event["graph_depth"] > 0 for event in events),
             f"request {request_index}: continuation count/nesting differs from trace phase {phase_id}")
        wall_ns = sum(event["wall_seconds"] * 1e9 for event in events)
        cpu_ns = sum(event["end_thread_cpu_ns"] - event["begin_thread_cpu_ns"] for event in events)
        need(wall_ns <= continuation["wall_total_ns"] + len(events) * common.AGGREGATE_WALL_ROUNDING_NS_PER_CALL and
             cpu_ns <= continuation["thread_cpu_total_ns"],
             f"request {request_index}: raw continuation spans exceed API totals in phase {phase_id}")
    need(sum(indexed[phase_id]["commit_and_continue"]["count"] for phase_id in API_PHASES) == root_count - 1,
         f"request {request_index}: continuation total differs from root_count - 1")
    return indexed


def _validate_phase_profile(raw: object, request_index: int, candidate: bool,
                            root_intervals: list[tuple[float, float]], trace: dict[str, Any]) -> dict[str, Any]:
    profile = obj(raw, f"request {request_index} phase_profile")
    keys = {"schema", "block_count", "ffn_graph_topology", "fused_kv_gather", "scale_softmax",
            "blocks", "aggregate_phase_wall_seconds", "aggregate_phase_thread_cpu_seconds",
            "head_root_range", "root_candidates", "phase_candidate_root_duration_sum_seconds",
            "phase_definite_root_duration_sum_seconds", "phase_root_duration_sum_semantics",
            "root_profile", "roots", "api_profile"}
    if candidate:
        keys.add("attention_graph_topology")
    exact_keys(profile, keys, f"request {request_index} phase_profile")
    need(profile["schema"] == PROFILE and uint(profile["block_count"], "profile.block_count") == BLOCKS and
         profile["ffn_graph_topology"] == "separate_gate_up" and
         boolean(profile["fused_kv_gather"], "profile.fused_kv_gather") is False and
         boolean(profile["scale_softmax"], "profile.scale_softmax") is False,
         f"request {request_index}: profile schema or non-B2 selectors changed")
    if candidate:
        need(profile["attention_graph_topology"] == "paired_windows_b2",
             f"request {request_index}: candidate profile omitted or changed B2 topology tag")
    need(profile["phase_root_duration_sum_semantics"] == ROOT_SEMANTICS,
         f"request {request_index}: root-duration non-additive semantics missing")

    blocks = arr(profile["blocks"], "profile blocks")
    need(len(blocks) == BLOCKS, f"request {request_index}: profile must contain 36 blocks")
    expected_wall = dict.fromkeys(PHASES, 0.0)
    expected_cpu = dict.fromkeys(PHASES, 0.0)
    expected_candidates: list[list[str]] = [[] for _ in root_intervals]
    per_block: list[dict[str, Any]] = []
    previous_end = 1
    for slot, raw_block in enumerate(blocks):
        block = obj(raw_block, f"profile block {slot}")
        exact_keys(block, ("slot", "attention_tile_count", "markers", "phase_wall_seconds",
                           "phase_thread_cpu_seconds"), f"profile block {slot}")
        need(uint(block["slot"], "block.slot") == slot and
             uint(block["attention_tile_count"], "block.attention_tile_count") == TILES_PER_BLOCK,
             f"request {request_index}: block/tile topology changed at {slot}")
        markers = arr(block["markers"], f"block {slot} markers")
        need(len(markers) == len(MARKERS), f"request {request_index}: block {slot} marker count changed")
        parsed_markers: list[dict[str, Any]] = []
        prior_wall = prior_cpu = 0.0
        prior_root = previous_end
        for marker_index, raw_marker in enumerate(markers):
            marker = obj(raw_marker, f"block {slot} marker {marker_index}")
            exact_keys(marker, ("name", "elapsed_seconds", "thread_cpu_seconds", "root_count"), "phase marker")
            wall = finite(marker["elapsed_seconds"], "marker wall")
            cpu = finite(marker["thread_cpu_seconds"], "marker thread CPU")
            root_cursor = uint(marker["root_count"], "marker root cursor", minimum=1)
            need(marker["name"] == MARKERS[marker_index] and wall >= prior_wall and cpu >= prior_cpu and
                 prior_root <= root_cursor <= len(root_intervals),
                 f"request {request_index}: marker order/clock/root cursor invalid at block {slot} marker {marker_index}")
            if marker_index == 0:
                need(wall == 0.0 and cpu == 0.0 and root_cursor == previous_end,
                     f"request {request_index}: block {slot} did not restart at contiguous zero marker")
            parsed_markers.append({"name": marker["name"], "elapsed_seconds": wall,
                                   "thread_cpu_seconds": cpu, "root_count": root_cursor})
            prior_wall, prior_cpu, prior_root = wall, cpu, root_cursor
        block_wall = obj(block["phase_wall_seconds"], "block phase wall")
        block_cpu = obj(block["phase_thread_cpu_seconds"], "block phase CPU")
        exact_keys(block_wall, PHASES, f"block {slot} phase wall")
        exact_keys(block_cpu, PHASES, f"block {slot} phase CPU")
        expected_block_wall: dict[str, float] = {}
        expected_block_cpu: dict[str, float] = {}
        for phase_index, phase in enumerate(PHASES):
            left, right = parsed_markers[phase_index], parsed_markers[phase_index + 1]
            wall = right["elapsed_seconds"] - left["elapsed_seconds"]
            cpu = right["thread_cpu_seconds"] - left["thread_cpu_seconds"]
            need(common.near(finite(block_wall[phase], "block phase wall"), wall) and
                 common.near(finite(block_cpu[phase], "block phase CPU"), cpu),
                 f"request {request_index}: block {slot} {phase} disagrees with marker deltas")
            expected_block_wall[phase] = wall
            expected_block_cpu[phase] = cpu
            expected_wall[phase] += wall
            expected_cpu[phase] += cpu
            label = f"block_{slot}/{phase}"
            for root_index in range(left["root_count"] - 1, right["root_count"]):
                expected_candidates[root_index].append(label)
        previous_end = parsed_markers[-1]["root_count"]
        per_block.append({"slot": slot, "attention_tile_count": TILES_PER_BLOCK,
                          "phase_wall_seconds": expected_block_wall,
                          "phase_thread_cpu_seconds": expected_block_cpu})

    head = obj(profile["head_root_range"], "profile head root range")
    exact_keys(head, ("start_root_count", "end_root_count", "candidate_root_indices_inclusive"), "head root range")
    head_start = uint(head["start_root_count"], "head start root count", minimum=1)
    head_end = uint(head["end_root_count"], "head end root count", minimum=1)
    need(head_start == previous_end and head_end == len(root_intervals) and head_end >= head_start and
         head["candidate_root_indices_inclusive"] == [head_start - 1, head_end - 1],
         f"request {request_index}: head roots do not continue the 36-block chain")
    for root_index in range(head_start - 1, head_end):
        expected_candidates[root_index].append("head")
    need(all(expected_candidates), f"request {request_index}: phase profile leaves a root unattributed")

    for field, expected in (("aggregate_phase_wall_seconds", expected_wall),
                            ("aggregate_phase_thread_cpu_seconds", expected_cpu)):
        values = obj(profile[field], f"profile {field}")
        exact_keys(values, PHASES, field)
        for phase, want in expected.items():
            need(common.near(finite(values[phase], f"{field}.{phase}"), want),
                 f"request {request_index}: aggregate {field}.{phase} differs from markers")

    labels = [f"block_{slot}/{phase}" for slot in range(BLOCKS) for phase in PHASES] + ["head"]
    candidate_sums = dict.fromkeys(labels, 0.0)
    definite_sums = dict.fromkeys(labels, 0.0)
    phase_roots = arr(profile["roots"], "profile roots")
    need(len(phase_roots) == len(root_intervals), f"request {request_index}: profile/root count mismatch")
    parsed_roots = []
    for index, raw_root in enumerate(phase_roots):
        root = obj(raw_root, f"profile root {index}")
        exact_keys(root, ("sequence", "completed", "has_error", "gpu_start_seconds", "gpu_end_seconds",
                          "duration_seconds", "candidate_phases"), f"profile root {index}")
        need(uint(root["sequence"], "profile root sequence") == index and root["completed"] is True and
             root["has_error"] is False, f"request {request_index}: incomplete/reordered profile root {index}")
        start = finite(root["gpu_start_seconds"], "profile root start")
        end = finite(root["gpu_end_seconds"], "profile root end")
        duration = end - start
        candidates = arr(root["candidate_phases"], "profile root candidates")
        need(start > 0 and duration > 0 and
             common.near(start, root_intervals[index][0], common.ROOT_CLOCK_BOUND_SECONDS, 0.0) and
             common.near(end, root_intervals[index][1], common.ROOT_CLOCK_BOUND_SECONDS, 0.0) and
             common.near(finite(root["duration_seconds"], "profile root duration"), duration, 2e-9, 1e-9) and
             candidates == expected_candidates[index],
             f"request {request_index}: root interval/candidate mapping differs at {index}")
        for label in candidates:
            need(label in candidate_sums, f"request {request_index}: unknown candidate phase {label}")
            candidate_sums[label] += duration
        if len(candidates) == 1:
            definite_sums[candidates[0]] += duration
        parsed_roots.append({"sequence": index, "start_seconds": start, "end_seconds": end,
                             "duration_seconds": duration, "candidate_phases": candidates})
    need(arr(profile["root_candidates"], "profile root candidate table") == expected_candidates,
         f"request {request_index}: root candidate table disagrees with marker cursors")
    for field, expected in (("phase_candidate_root_duration_sum_seconds", candidate_sums),
                            ("phase_definite_root_duration_sum_seconds", definite_sums)):
        values = obj(profile[field], field)
        exact_keys(values, set(labels), field)
        for label, want in expected.items():
            need(common.near(finite(values[label], f"{field}.{label}"), want, 2e-8, 1e-8),
                 f"request {request_index}: {field} differs at {label}")

    nested = obj(profile["root_profile"], "nested profile root metrics")
    exact_keys(nested, ("root_count", "root_duration_seconds", "envelope_seconds", "union_seconds",
                        "overlap_seconds", "phase_union_seconds"), "nested profile root metrics")
    root_metrics = common._root_metrics(root_intervals)
    need(uint(nested["root_count"], "nested root count") == len(root_intervals),
         f"request {request_index}: nested root count mismatch")
    for key in ("root_duration_seconds", "envelope_seconds", "union_seconds", "overlap_seconds"):
        need(common.near(finite(nested[key], f"nested root {key}"), root_metrics[key], 2e-8, 1e-8),
             f"request {request_index}: nested root {key} disagrees with intervals")
    phase_union = obj(nested["phase_union_seconds"], "nested phase unions")
    exact_keys(phase_union, ("whole_chain",), "nested phase unions")
    need(common.near(finite(phase_union["whole_chain"], "whole chain union"), root_metrics["union_seconds"],
                     2e-8, 1e-8), f"request {request_index}: whole-chain union mismatch")
    api = _validate_api(profile, trace, request_index, candidate, len(root_intervals))
    return {"phase_wall_seconds": expected_wall, "phase_thread_cpu_seconds": expected_cpu,
            "per_block": per_block, "roots": parsed_roots, "root_metrics": root_metrics,
            "candidate_root_duration_sums_nonadditive": candidate_sums,
            "definite_root_duration_sums_nonadditive": definite_sums, "api": api}


def _expected_rows(reverse: bool) -> list[dict[str, Any]]:
    gate_modes = [True, False, False, True]
    if reverse:
        gate_modes = [not mode for mode in gate_modes]
    timed_modes = [False, True, False, True, True, False, True, False]
    if reverse:
        timed_modes = [not mode for mode in timed_modes]
    warm_modes = [False, True]
    if reverse:
        warm_modes = [not mode for mode in warm_modes]
    rows: list[dict[str, Any]] = []
    gate_inputs = [False, False, True, True]
    for i in range(4):
        rows.append({"cohort": "poisoned_gate", "candidate": gate_modes[i], "alternate": gate_inputs[i],
                     "profiled": True, "measured": False, "timing_eligible": False,
                     "poisoned": True, "compare": True})
    for mode in warm_modes:
        rows.append({"cohort": "warmup", "candidate": mode, "alternate": False,
                     "profiled": False, "measured": False, "timing_eligible": False,
                     "poisoned": False, "compare": False})
    for offset, mode in enumerate(timed_modes):
        profiled = offset in PROFILED_TIMED_OFFSETS
        rows.append({"cohort": "profiled_measured" if profiled else "observer_off_measured",
                     "candidate": mode, "alternate": False, "profiled": profiled, "measured": True,
                     "timing_eligible": not profiled, "poisoned": False,
                     "compare": offset == 7})
    rows.append({"cohort": "candidate_tail", "candidate": True, "alternate": False,
                 "profiled": True, "measured": False, "timing_eligible": False,
                 "poisoned": False, "compare": True})
    return rows


def validate_contract_metadata(raw: object, where: str = "report") -> dict[str, Any]:
    data = obj(raw, where)
    need(data.get("schema") == REPORT and data.get("profile_phases") is True,
         f"{where}: expected opt-in full36 B2 phase schema")
    need(data.get("generation") is False and data.get("production_promoted") is False and
         data.get("full_model_qualified") is False and data.get("passed") is True and
         data.get("cleanup_quarantined") is False,
         f"{where}: report does not represent a successful bounded diagnostic")
    need(finite(data.get("atol"), f"{where}.atol") == 2e-5 and
         finite(data.get("rtol"), f"{where}.rtol") == 2e-5,
         f"{where}: numerical tolerance changed from the saved-capture contract")
    need(uint(data.get("block_count"), f"{where}.block_count") == BLOCKS and
         data.get("includes_head") is True and data.get("grid") == [6, 44, 80] and
         uint(data.get("expected_request_count"), f"{where}.expected_request_count") == 15,
         f"{where}: full36+head geometry or request count changed")
    need(data.get("timed") is True and isinstance(data.get("reverse"), bool),
         f"{where}: timed normal/reverse cohort metadata missing")
    contract = obj(data.get("profiling_contract"), f"{where}.profiling_contract")
    contract_keys = (
        "phase_profile_extraction", "root_profile_extraction", "continuation_trace_extraction",
        "early_phase_read_rejected_on_first_profiled_gate",
        "cpu_tensor_diagnostics_between_timed_requests", "only_last_measured_request_has_capture_readback",
        "one_existing_terminal_wait_per_request", "named_buffer_allocations_during_encode",
        "activation_readbacks_before_terminal_wait", "stage_metric_readbacks", "full36_tiles_per_block",
        "baseline_attention_graph_calls", "candidate_attention_graph_calls", "baseline_total_graph_calls",
        "candidate_total_graph_calls", "baseline_command_segments", "candidate_command_segments",
    )
    exact_keys(contract, contract_keys, f"{where}.profiling_contract")
    need(contract["phase_profile_extraction"] == "post_terminal_wait" and
         contract["root_profile_extraction"] == "post_terminal_wait" and
         contract["continuation_trace_extraction"] == "post_terminal_wait" and
         contract["early_phase_read_rejected_on_first_profiled_gate"] is True and
         contract["cpu_tensor_diagnostics_between_timed_requests"] is False and
         contract["only_last_measured_request_has_capture_readback"] is True and
         contract["one_existing_terminal_wait_per_request"] is True,
         f"{where}: profile/readback timing contract changed")
    contract_counts = {
        "named_buffer_allocations_during_encode": 0,
        "activation_readbacks_before_terminal_wait": 0,
        "stage_metric_readbacks": 0,
        "full36_tiles_per_block": 60,
        "baseline_attention_graph_calls": 6480,
        "candidate_attention_graph_calls": 3240,
        "baseline_total_graph_calls": 7025,
        "candidate_total_graph_calls": 3785,
        "baseline_command_segments": EXPECTED_ROOTS[False],
        "candidate_command_segments": EXPECTED_ROOTS[True],
    }
    for field, expected in contract_counts.items():
        need(uint(contract[field], f"{where}.profiling_contract.{field}") == expected,
             f"{where}: profiling contract count {field} changed")
    reverse = data["reverse"]
    rows = _expected_rows(reverse)
    measured_modes = [row["candidate"] for row in rows[6:14]]
    schedule = obj(data.get("observer_schedule"), f"{where}.observer_schedule")
    exact_keys(schedule, ("poisoned_gate_requests", "poisoned_gates_profiled", "warmup_requests",
                          "warmups_profiled", "measured_requests", "measured_candidate_modes",
                          "measured_profiled", "measured_timing_eligible", "tail_request",
                          "tail_candidate", "tail_profiled", "reverse_flips_modes_only"),
               f"{where}.observer_schedule")
    need(schedule == {
        "poisoned_gate_requests": [0, 1, 2, 3], "poisoned_gates_profiled": True,
        "warmup_requests": [4, 5], "warmups_profiled": False,
        "measured_requests": list(range(6, 14)), "measured_candidate_modes": measured_modes,
        "measured_profiled": [row["profiled"] for row in rows[6:14]],
        "measured_timing_eligible": [row["timing_eligible"] for row in rows[6:14]],
        "tail_request": 14, "tail_candidate": True, "tail_profiled": True,
        "reverse_flips_modes_only": True,
    }, f"{where}: observer schedule metadata differs from the admitted cohort")
    pins = {field: common.digest(data.get(field), f"{where}.{field}") for field in FIXTURE_PIN_FIELDS}
    need(common.digest(data.get("input_sha256"), f"{where}.input_sha256") and
         common.digest(data.get("alternate_input_sha256"), f"{where}.alternate_input_sha256"),
         f"{where}: fixture input digest missing")
    need(data["input_sha256"] != data["alternate_input_sha256"],
         f"{where}: A/B input fixtures must be distinct")
    capture_order = data.get("capture_order")
    need(capture_order == common._expected_capture_order(), f"{where}: ordered 43-capture inventory changed")
    inventory = obj(data.get("capture_inventory"), f"{where}.capture_inventory")
    need(inventory == {"block_attention": 36, "retained_block_outputs": 2, "head_stages": 5, "total": 43},
         f"{where}: capture inventory changed")
    return {"reverse": data["reverse"], "fixture_pins": pins,
            "input_sha256": data["input_sha256"], "alternate_input_sha256": data["alternate_input_sha256"],
            "capture_order": capture_order}


def validate_report(raw: object, report_sha256: str = "", path: str = "<memory>") -> dict[str, Any]:
    data = obj(raw, "report")
    metadata = validate_contract_metadata(data, path)
    schedule = _expected_rows(metadata["reverse"])
    requests = arr(data.get("requests"), "requests")
    need(len(requests) == 15, "expected four gates, two warmups, eight measured arms, and one candidate tail")
    parsed_requests: list[dict[str, Any]] = []
    for index, (raw_request, expected) in enumerate(zip(requests, schedule)):
        request = obj(raw_request, f"request {index}")
        need(uint(request.get("request"), f"request {index}.request") == index and
             request.get("cohort") == expected["cohort"] and
             boolean(request.get("candidate"), f"request {index}.candidate") is expected["candidate"] and
             boolean(request.get("alternate_input"), f"request {index}.alternate_input") is expected["alternate"] and
             boolean(request.get("profiled"), f"request {index}.profiled") is expected["profiled"] and
             boolean(request.get("measured"), f"request {index}.measured") is expected["measured"] and
             boolean(request.get("timing_eligible"), f"request {index}.timing_eligible") is expected["timing_eligible"] and
             boolean(request.get("poisoned"), f"request {index}.poisoned") is expected["poisoned"],
             f"request {index}: observer/mode/input schedule differs from admitted schedule")
        expected_hash = metadata["alternate_input_sha256"] if expected["alternate"] else metadata["input_sha256"]
        need(common.digest(request.get("input_sha256"), f"request {index}.input_sha256") == expected_hash,
             f"request {index}: saved input identity disagrees with schedule")
        need(request.get("passed") is True and request.get("head_input_is_block35_output") is True,
             f"request {index}: full chain edge or request comparison failed")
        for field, value in (("terminal_wait_count", 1), ("named_allocations_during_encode", 0),
                             ("activation_readbacks_before_terminal_wait", 0),
                             ("metrics_readbacks_after_terminal_wait", 0),
                             ("capture_readbacks_after_terminal_wait", 43 if expected["compare"] else 0)):
            need(uint(request.get(field), f"request {index}.{field}") == value,
                 f"request {index}: {field} must equal {value}")
        if index == 0:
            need(request.get("early_profile_read_rejected") is True,
                 "request 0: pre-wait profile-read guard did not reject")
            lifecycle = obj(request.get("early_lifecycle_rejections"), "request 0 lifecycle guards")
            expected_lifecycle = {"selection_change_rejected": True, "selection_release_rejected": True,
                                  "selection_poison_rejected": True, "session_release_rejected": True,
                                  "session_poison_rejected": True, "read_rejected": True}
            need(lifecycle == expected_lifecycle, "request 0: in-flight lifecycle guards changed")
        else:
            need(request.get("early_profile_read_rejected") in (None, False),
                 f"request {index}: one-shot pre-wait profile guard witness appeared on another request")
        expected_roots = EXPECTED_ROOTS[expected["candidate"]]
        root_count = uint(request.get("root_count"), f"request {index}.root_count", minimum=1)
        segment_count = uint(request.get("command_segments"), f"request {index}.command_segments", minimum=1)
        need(root_count == segment_count == expected_roots,
             f"request {index}: root/command topology differs from baseline/B2 full36 counts")
        intervals, roots = common._root_intervals(arr(request.get("roots"), f"request {index}.roots"),
                                                  f"request {index}", profile=False)
        need(len(intervals) == expected_roots, f"request {index}: request root inventory differs from topology")
        root_metrics = common._root_metrics(intervals)
        if expected["profiled"]:
            profiled_intervals, profiled_metrics = _root_profile(request.get("root_profile"), index, expected_roots)
            need(profiled_intervals == intervals and profiled_metrics == root_metrics,
                 f"request {index}: detailed root profile differs from request root inventory")
        else:
            need(request.get("root_profile") is None,
                 f"request {index}: observer-off request contains a detailed root profile")
        gpu_seconds = finite(request.get("gpu_elapsed_seconds"), f"request {index}.gpu_elapsed_seconds")
        need(common.near(gpu_seconds, root_metrics["root_duration_seconds"], 2e-6, 1e-8),
             f"request {index}: gpu_elapsed_seconds differs from root duration sum")
        for field in ("encode_seconds", "terminal_wait_seconds", "total_seconds"):
            finite(request.get(field), f"request {index}.{field}")
        need(finite(request["terminal_wait_seconds"], "terminal wait") <=
             finite(request["total_seconds"], "total time") + 1e-6,
             f"request {index}: terminal wait exceeds total request envelope")
        comparisons_raw = arr(request.get("comparisons"), f"request {index}.comparisons")
        comparisons = common.validate_comparisons(comparisons_raw, index) if expected["compare"] else []
        if not expected["compare"]:
            need(not comparisons_raw, f"request {index}: unexpected CPU tensor comparison/readback")
        profile_result = None
        trace_result = None
        if expected["profiled"]:
            need(request.get("profiled") is True and request.get("phase_profile") is not None and
                 request.get("continuation_trace") is not None,
                 f"request {index}: profiled request is missing phase or continuation data")
            trace_result = common.validate_trace(request["continuation_trace"], index, intervals)
            profile_result = _validate_phase_profile(request["phase_profile"], index,
                                                     expected["candidate"], intervals, trace_result)
            need(trace_result["info"]["root_count"] == expected_roots,
                 f"request {index}: trace root count differs from command roots")
        else:
            need(request.get("phase_profile") is None and request.get("continuation_trace") is None,
                 f"request {index}: profile-off request contains detailed profiling data")
        parsed_requests.append({"index": index, "cohort": expected["cohort"],
            "candidate": expected["candidate"], "alternate_input": expected["alternate"],
            "profiled": expected["profiled"], "measured": expected["measured"],
            "timing_eligible": expected["timing_eligible"], "input_sha256": expected_hash,
            "timing": {field: float(request[field]) for field in
                       ("encode_seconds", "terminal_wait_seconds", "total_seconds", "gpu_elapsed_seconds")},
            "command_segments": segment_count, "root_metrics": root_metrics,
            "comparisons": comparisons, "profile": profile_result, "trace": trace_result})
    profiled_count = sum(request["profiled"] for request in parsed_requests)
    compared_count = sum(bool(request["comparisons"]) for request in parsed_requests)
    need(profiled_count == 9 and compared_count == 6,
         "report must contain four profile gates, four profile-on measured arms, and a profile-on tail; six capture requests")
    need(sum(len(request["comparisons"]) for request in parsed_requests) == 258,
         "report does not contain exactly 6 x 43 saved-capture comparisons")
    return {"path": path, "sha256": report_sha256, "reverse": metadata["reverse"],
            "fixture_pins": metadata["fixture_pins"], "input_sha256": metadata["input_sha256"],
            "alternate_input_sha256": metadata["alternate_input_sha256"],
            "capture_order": metadata["capture_order"], "requests": parsed_requests}


def _aggregate_api(requests: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[int, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for request in requests:
        for phase_id, operations in request["profile"]["api"].items():
            for operation, stats in operations.items():
                for field, value in stats.items():
                    if field in ("wall_max_ns", "thread_cpu_max_ns"):
                        totals[phase_id][operation][field] = max(totals[phase_id][operation][field], value)
                    else:
                        totals[phase_id][operation][field] += value
    result: dict[str, Any] = {}
    for phase_id, phase_name in API_PHASES.items():
        result[phase_name] = {operation: dict(totals[phase_id][operation]) for operation in OPS}
    return result


def _quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)

    def percentile(p: float) -> float:
        # Nearest-rank empirical quantile, kept explicit for reproducibility.
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]

    return {"count": len(ordered), "min": ordered[0], "p50": percentile(0.50),
            "p90": percentile(0.90), "p95": percentile(0.95), "p99": percentile(0.99),
            "max": ordered[-1]}


def _continuation_prior_root_detail(requests: list[dict[str, Any]]) -> dict[str, Any]:
    per_call: list[dict[str, Any]] = []
    grouped: dict[int, list[dict[str, Any]]] = {phase: [] for phase in API_PHASES}
    clock_bound = common.ROOT_CLOCK_BOUND_SECONDS
    for request in requests:
        trace = request["trace"]
        host_seconds = trace["host_seconds"]
        for event in trace["events"]:
            sequence = event["before_root_sequence"]
            begin = host_seconds(event["begin_mach_ticks"])
            end = host_seconds(event["end_mach_ticks"])
            prior = trace["roots"][:sequence]  # root i is the current submission and is excluded
            intersections = [(max(begin, root["start_seconds"]), min(end, root["end_seconds"]))
                             for root in prior
                             if root["start_seconds"] < end and root["end_seconds"] > begin]
            intersections = [(left, right) for left, right in intersections if right > left]
            overlap = common.interval_union(intersections)
            pending = [root for root in prior if root["end_seconds"] > begin]
            definitely_pending = [root for root in prior if root["end_seconds"] > begin + clock_bound]
            boundary = [root for root in prior if abs(root["end_seconds"] - begin) <= clock_bound]
            latest_end = max((root["end_seconds"] for root in prior), default=None)
            row = {
                "request": request["index"], "cohort": request["cohort"],
                "candidate": request["candidate"], "continuation_event": event["serial"],
                "before_root_sequence": sequence, "api_phase_id": event["phase"],
                "api_phase": API_PHASES[event["phase"]], "graph_depth": event["graph_depth"],
                "wall_seconds": end - begin, "thread_cpu_seconds": event["thread_cpu_seconds"],
                "prior_submitted_root_count": len(prior),
                "prior_roots_with_gpu_end_after_call_entry": len(pending),
                "prior_roots_definitely_ending_after_call_entry_under_clock_bound": len(definitely_pending),
                "prior_roots_with_end_inside_cross_clock_bound": len(boundary),
                "prior_root_interval_overlap_union_seconds": overlap,
                "prior_roots_interval_overlapping_call": len(intersections),
                "latest_prior_gpu_root_end_seconds": latest_end,
                "call_return_minus_latest_prior_gpu_end_seconds": (end - latest_end if latest_end is not None else None),
            }
            per_call.append(row)
            grouped[event["phase"]].append(row)

    by_phase: dict[str, Any] = {}
    for phase_id, phase_name in API_PHASES.items():
        rows = grouped[phase_id]
        by_phase[phase_name] = {
            "continuation_count": len(rows),
            "wall_total_seconds": sum(row["wall_seconds"] for row in rows),
            "thread_cpu_total_seconds": sum(row["thread_cpu_seconds"] for row in rows),
            "per_call_wall_seconds": _quantiles([row["wall_seconds"] for row in rows]),
            "per_call_thread_cpu_seconds": _quantiles([row["thread_cpu_seconds"] for row in rows]),
            "prior_root_overlap_union_seconds_per_call": _quantiles(
                [row["prior_root_interval_overlap_union_seconds"] for row in rows]),
            "prior_pending_root_count_per_call": _quantiles(
                [float(row["prior_roots_with_gpu_end_after_call_entry"]) for row in rows]),
            "prior_definitely_pending_root_count_per_call": _quantiles(
                [float(row["prior_roots_definitely_ending_after_call_entry_under_clock_bound"]) for row in rows]),
            "call_return_minus_latest_prior_root_end_seconds": _quantiles(
                [row["call_return_minus_latest_prior_gpu_end_seconds"] for row in rows
                 if row["call_return_minus_latest_prior_gpu_end_seconds"] is not None]),
        }
    return {
        "clock_bound_seconds": clock_bound,
        "prior_root_rule": "For continuation event before_root_sequence=i, only root indices < i count as already submitted; root i is excluded.",
        "per_phase": by_phase,
        "per_call": per_call,
        "interpretation": (
            "GPU root end timestamps and host continuation intervals are compared with the existing clock bound. "
            "Pending counts and interval intersections are descriptive cross-clock bounds, not exact completion, "
            "queue-stall, semaphore-wait, or active-shader measurements. No wall-minus-CPU causal estimate is made."
        ),
    }


def _block_early_late_contrasts(per_block: list[dict[str, Any]]) -> dict[str, Any]:
    need(len(per_block) == BLOCKS, "early/late contrast requires all 36 blocks")
    result: dict[str, Any] = {"early_blocks_inclusive": [0, 5], "late_blocks_inclusive": [30, 35],
                              "wall_seconds": {}, "thread_cpu_seconds": {}}
    for metric, phase_key in (("wall_seconds", "phase_wall_seconds"),
                              ("thread_cpu_seconds", "phase_thread_cpu_seconds")):
        for phase in PHASES:
            early = statistics.mean(row[phase_key][phase] for row in per_block[:6])
            late = statistics.mean(row[phase_key][phase] for row in per_block[-6:])
            result[metric][phase] = {"early_mean_per_block": early, "late_mean_per_block": late,
                                     "late_minus_early": late - early,
                                     "late_over_early": late / early if early else None}
    result["interpretation"] = "Descriptive host marker drift across block slots; these intervals are not GPU shader phase times."
    return result


def _profile_summary(report: dict[str, Any]) -> dict[str, Any]:
    profiled = [request for request in report["requests"] if request["profiled"]]
    per_request = []
    groups: dict[str, Any] = {}
    continuation_rows = []
    for request in profiled:
        profile = request["profile"]
        per_request.append({"request": request["index"], "cohort": request["cohort"],
            "candidate": request["candidate"], "timing": request["timing"],
            "recomputed_phase_wall_seconds": profile["phase_wall_seconds"],
            "recomputed_phase_thread_cpu_seconds": profile["phase_thread_cpu_seconds"],
            "per_block_phase_wall_and_thread_cpu": profile["per_block"],
            "root_interval_metrics": profile["root_metrics"],
            "candidate_root_duration_sums_nonadditive": profile["candidate_root_duration_sums_nonadditive"],
            "definite_root_duration_sums_nonadditive": profile["definite_root_duration_sums_nonadditive"],
            "exclusive_graph_and_continuation_api_by_phase": _serialize_api(profile["api"]),
            "early_late_block_contrasts": _block_early_late_contrasts(profile["per_block"]),
        })
        key = f"{request['cohort']}/{'B2' if request['candidate'] else 'baseline'}"
        groups.setdefault(key, []).append(request)
        continuation_rows.append({"index": request["index"], "cohort": request["cohort"],
            "candidate": request["candidate"], "combined_attention": request["candidate"], "phase": profile,
            "trace": request["trace"]})

    by_group = {}
    for key, requests in sorted(groups.items()):
        by_group[key] = {
            "request_count": len(requests),
            "phase_wall_seconds_by_phase": {phase: common.distribution(
                [request["profile"]["phase_wall_seconds"][phase] for request in requests]) for phase in PHASES},
            "phase_thread_cpu_seconds_by_phase": {phase: common.distribution(
                [request["profile"]["phase_thread_cpu_seconds"][phase] for request in requests]) for phase in PHASES},
            "per_block_phase_wall_seconds": [request["profile"]["per_block"] for request in requests],
            "exclusive_graph_and_continuation_api_by_phase": _aggregate_api(requests),
        }
    return {"profiled_requests": per_request, "by_cohort_and_topology": by_group,
            "continuation_trace": common._continuation_summary(continuation_rows),
            "continuation_per_call_quantiles_and_prior_root_overlap": _continuation_prior_root_detail(continuation_rows),
            "interpretation": (
                "Block wall/thread-CPU marker deltas and MPS API spans are host-side localization. "
                "Continuation calls may overlap timestamps of earlier submitted GPU roots; overlap is not active GPU work. "
                "Root GPU timestamp duration sums and phase candidate sums are non-additive and are not phase shader time. "
                "Observer-on profiles are not performance samples; profile-off arms remain the timing cohort."
            )}


def _serialize_api(api: dict[int, dict[str, dict[str, int]]]) -> dict[str, Any]:
    return {API_PHASES[phase_id]: operations for phase_id, operations in api.items()}


def analyze_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    need(len(reports) == 2, "supply exactly two reports: normal and reverse")
    by_direction = {report["reverse"]: report for report in reports}
    need(set(by_direction) == {False, True}, "reports must contain one normal and one reverse cohort")
    normal, reverse = by_direction[False], by_direction[True]
    need(normal["fixture_pins"] == reverse["fixture_pins"] and
         normal["input_sha256"] == reverse["input_sha256"] and
         normal["alternate_input_sha256"] == reverse["alternate_input_sha256"],
         "normal/reverse reports do not use identical fixtures and A/B inputs")
    for index, (left, right) in enumerate(zip(normal["requests"], reverse["requests"])):
        need(left["profiled"] == right["profiled"] and left["cohort"] == right["cohort"] and
             left["alternate_input"] == right["alternate_input"],
             f"normal/reverse observer/input schedule differs at request {index}")
        if index < 14:
            need(left["candidate"] is (not right["candidate"]),
                 f"normal/reverse mode assignment was not flipped at request {index}")
        else:
            need(left["candidate"] is right["candidate"] is True,
                 "candidate tail must remain B2 in both cohorts")
    run_summaries = []
    for run in (normal, reverse):
        run_summaries.append({"path": run["path"], "sha256": run["sha256"],
            "order": "reverse" if run["reverse"] else "normal",
            "fixture_pins": run["fixture_pins"], "input_sha256": run["input_sha256"],
            "requests": [{"request": request["index"], "cohort": request["cohort"],
                "candidate": request["candidate"], "profiled": request["profiled"],
                "measured": request["measured"], "timing_eligible": request["timing_eligible"],
                "timing": request["timing"], "command_segments": request["command_segments"],
                "root_interval_metrics": request["root_metrics"],
                "different_words": [capture["different_words"] for capture in request["comparisons"]]}
                for request in run["requests"]],
            "phase_localization": _profile_summary(run)})
    return {"schema": SUMMARY, "input_report_schema": REPORT,
            "scope": "Two 15-request same-fixture normal/reverse B2 full36 plus global-head diagnostic cohorts; no generation.",
            "numerical_scope": {"atol": 2e-5, "rtol": 2e-5, "capture_requests_per_cohort": 6,
                "captures_per_request": 43, "total_capture_comparisons": 516,
                "all_saved_comparisons_finite_and_within_tolerance": True,
                "different_bf16_words_preserved": True,
                "not_claimed": ["bitwise parity", "internal F32 softmax/probability parity",
                                "full-model qualification", "generation speed"]},
            "lifecycle_scope": {"requests_per_cohort": 15, "terminal_waits_per_request": 1,
                "named_encode_allocations_per_request": 0, "preterminal_activation_reads_per_request": 0,
                "metrics_reads_per_request": 0, "postwait_captures_per_cohort": 258,
                "profile_and_trace_extraction": "after the existing terminal wait",
                "cpu_tensor_diagnostics_between_timed_requests": False},
            "runs": run_summaries,
            "analysis_limit": (
                "Phase wall/thread-CPU markers, Graph/continuation host spans, callback observations, and "
                "GPU root timestamps are different observables. Root duration sums and candidate-phase sums "
                "are non-additive; continuation/root overlap does not establish active shader time. The trace "
                "parent/depth check is syntactic because exported data omits some graph scopes. This analyzer "
                "does not validate external receipts, source manifests, process guards, or memory attribution."
            )}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, type=Path,
                        help="opt-in B2 phase report; repeat for normal and reverse")
    parser.add_argument("--output", type=Path, help="fresh summary file (exclusive create); defaults to stdout")
    args = parser.parse_args(argv)
    try:
        reports = []
        for path in args.report:
            raw_bytes = path.read_bytes()
            reports.append(validate_report(json.loads(raw_bytes), hashlib.sha256(raw_bytes).hexdigest(), str(path)))
        summary = analyze_reports(reports)
        encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                output.write(encoded)
        else:
            sys.stdout.write(encoded)
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"pixelumm B2 phase analysis rejected report: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
