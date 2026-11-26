#!/usr/bin/env python3
"""Fail-closed offline analyzer for the DRAMA-3 dense decode replay.

This report describes only the captured dense projection commands. It does not
estimate a complete decode request or infer costs for uncaptured work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


SCOPE = "dense_expert_projections_only"
FRAMES = (0, 29, 59)
OMITTED_CELLS = ((0, "main.head"),)
INVOCATION_EVENT = "decode_replay_invocation"
PAIR_EVENT = "decode_replay_pair"
SUMMARY_EVENT = "decode_replay_summary"
CAPTURE_EVENT = "capture"
GPU_EVENT = "drama3_gpu_command"
FORMAT_NAME = "drama3-decode-replay-analysis-v1"


class ReplayError(ValueError):
    """Input logs do not satisfy the decode-replay measurement contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def integer(value: object, message: str, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum, message)
    return int(value)


def finite_number(value: object, message: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), message)
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ReplayError(message) from exc
    require(math.isfinite(result), message)
    return result


def finite_positive(value: object, message: str) -> float:
    result = finite_number(value, message)
    require(result > 0.0, message)
    return result


def _events_from_jsonl(text: str, recognized: set[str], label: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        start = line.find("{")
        if start < 0:
            if any(name in line for name in recognized):
                raise ReplayError(f"{label} line {line_number}: malformed recognized event")
            continue
        try:
            value, _ = decoder.raw_decode(line[start:])
        except json.JSONDecodeError as exc:
            if any(name in line for name in recognized):
                raise ReplayError(f"{label} line {line_number}: malformed recognized event") from exc
            continue
        if isinstance(value, dict) and value.get("event") in recognized:
            events.append(value)
    return events


def _one_event(events: list[dict[str, Any]], kind: str, label: str) -> dict[str, Any]:
    selected = [event for event in events if event.get("event") == kind]
    require(len(selected) == 1, f"expected exactly one {label} event")
    return selected[0]


def _valid_graph(value: object, label: str) -> dict[str, int]:
    require(isinstance(value, dict), f"{label}: missing graph snapshot")
    graph = value
    required = ("depth", "submitted", "completed", "completion_waits", "pending", "retained_bytes")
    normalized = {key: integer(graph.get(key), f"{label}: invalid graph {key}") for key in required}
    require(normalized["depth"] == 2, f"{label}: graph depth is not two")
    if "max_pending" in graph:
        max_pending = integer(graph["max_pending"], f"{label}: invalid graph max_pending")
        require(max_pending <= 2, f"{label}: graph max_pending exceeds two")
    return normalized


def _graph_pair(before_value: object, after_value: object, pair_number: int) -> tuple[dict[str, int], dict[str, int]]:
    before = _valid_graph(before_value, f"pair {pair_number} before")
    after = _valid_graph(after_value, f"pair {pair_number} after")
    require(before["submitted"] == before["completed"] == before["completion_waits"],
            f"pair {pair_number}: graph was not drained before replay")
    require(before["pending"] == before["retained_bytes"] == 0,
            f"pair {pair_number}: pending or retained graph state before replay")
    for key in ("submitted", "completed", "completion_waits"):
        require(after[key] == before[key] + 3,
                f"pair {pair_number}: graph {key} did not advance by three")
    require(after["pending"] == 0 and after["retained_bytes"] == 0,
            f"pair {pair_number}: pending or retained graph state after replay")
    return before, after


def _readbacks(value: object, label: str) -> dict[str, int]:
    require(isinstance(value, dict), f"{label}: missing readback snapshot")
    return {str(key): integer(count, f"{label}: invalid readback counter {key}")
            for key, count in value.items()}


def _readback_pair(before_value: object, after_value: object, receipt_bytes: int,
                   pair_number: int) -> tuple[dict[str, int], dict[str, int]]:
    before = _readbacks(before_value, f"pair {pair_number} before")
    after = _readbacks(after_value, f"pair {pair_number} after")
    names = set(before) | set(after) | {"control", "terminal", "diagnostic"}
    for name in names:
        old, new = before.get(name, 0), after.get(name, 0)
        expected_delta = receipt_bytes if name == "terminal" else 0
        require(new - old == expected_delta,
                f"pair {pair_number}: unexpected {name} readback delta")
    require(after.get("diagnostic", 0) == 0,
            f"pair {pair_number}: diagnostic activation readback is nonzero")
    return before, after


def _summary_stats(samples: list[float]) -> dict[str, float | int]:
    require(samples, "cannot summarize empty timing sample")
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "p10": percentile(0.10),
        "p90": percentile(0.90),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _timeline_rows(text: str) -> dict[int, dict[str, Any]]:
    rows = _events_from_jsonl(text, {GPU_EVENT}, "timeline")
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = integer(row.get("index"), "invalid GPU command index")
        require(index not in by_index, f"duplicate GPU command index {index}")
        by_index[index] = row
    return by_index


def _command_seconds(row: dict[str, Any], index: int) -> float:
    status = integer(row.get("completion_status"), f"command {index}: invalid completion status")
    require(status == 0, f"command {index}: completion status is not successful (0)")
    require(row.get("gpu_timing_valid") is True, f"command {index}: GPU timing is not valid")
    return finite_positive(row.get("gpu_seconds"), f"command {index}: GPU duration must be finite and positive")


def _capture_contract(capture: dict[str, Any], invocation: dict[str, Any]) -> None:
    require(capture.get("event") == CAPTURE_EVENT, "invalid capture event")
    summary = capture.get("capture_summary")
    require(isinstance(summary, dict), "capture event has no capture summary")
    require(summary.get("scope") == SCOPE, "capture scope does not match replay scope")
    require(integer(summary.get("records"), "invalid capture record count") == invocation["records"],
            "capture record count does not match replay invocation")
    require(summary.get("frames") == list(FRAMES), "capture frame coverage is incomplete")
    require(summary.get("families") == invocation["families"],
            "capture family list does not match replay invocation")
    require(summary.get("activation_readbacks") == 0,
            "capture reports activation readbacks")
    require(summary.get("full_decode_coverage_claimed") is False,
            "capture claims full decode coverage")
    require(integer(capture.get("capture_records"), "invalid capture event record count") == invocation["records"],
            "capture event record count does not match replay invocation")
    wav = capture.get("wav")
    require(isinstance(wav, dict), "capture event has no WAV provenance")
    digest = wav.get("sha256")
    expected_digest = wav.get("expected_sha256")
    require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            "capture WAV SHA-256 is malformed")
    require(expected_digest == digest, "capture WAV SHA-256 differs from expected baseline")
    capture_graph = _valid_graph(capture.get("graph"), "capture")
    require(capture_graph["pending"] == 0 and capture_graph["retained_bytes"] == 0,
            "capture graph has pending or retained state")


def analyze(events_text: str, timeline_text: str) -> dict[str, Any]:
    """Validate replay receipts and return local, paired timing summaries."""
    event_rows = _events_from_jsonl(
        events_text,
        {CAPTURE_EVENT, INVOCATION_EVENT, PAIR_EVENT, SUMMARY_EVENT},
        "events",
    )
    capture = _one_event(event_rows, CAPTURE_EVENT, "capture")
    invocation = _one_event(event_rows, INVOCATION_EVENT, "invocation")
    final = _one_event(event_rows, SUMMARY_EVENT, "summary")
    pair_rows = [event for event in event_rows if event.get("event") == PAIR_EVENT]

    require(integer(invocation.get("format"), "invalid decode replay format", 1) == 1,
            "unsupported decode replay format")
    require(invocation.get("scope") == SCOPE, "decode replay scope mismatch")
    pairs = integer(invocation.get("pairs"), "invalid pair count", 3)
    warmups = integer(invocation.get("warmups"), "invalid warmup count", 1)
    repeats = integer(invocation.get("repeats"), "invalid repeat count", 1)
    require(pairs <= 21 and warmups <= 4 and repeats <= 8, "decode replay bounds exceeded")
    require(invocation.get("frames") == list(FRAMES), "decode replay frames are incomplete or reordered")
    raw_families = invocation.get("families")
    require(isinstance(raw_families, list) and raw_families, "decode replay families are missing")
    require(all(isinstance(family, str) and family for family in raw_families),
            "decode replay family names are invalid")
    families = list(raw_families)
    require(families == sorted(set(families)), "decode replay family list is not unique and sorted")
    require("main.head" in families, "decode replay omitted the main head family")
    omitted = invocation.get("omitted_cells")
    omitted_valid = (
        isinstance(omitted, list) and len(omitted) == 1 and
        isinstance(omitted[0], list) and len(omitted[0]) == 2 and
        integer(omitted[0][0], "invalid omitted frame") == 0 and
        omitted[0][1] == "main.head"
    )
    require(omitted_valid,
            "decode replay omission must be exactly frame-zero main.head")
    records = integer(invocation.get("records"), "invalid capture record count", 1)
    covered_values = integer(invocation.get("covered_values"), "invalid covered value count", 1)
    _capture_contract(capture, {"records": records, "families": families})

    keys = [(frame, family) for frame in FRAMES for family in families
            if (frame, family) not in OMITTED_CELLS]
    require(any(frame == 0 for frame, _ in keys), "frame zero has no replayable projection family")
    require((29, "main.head") in keys and (59, "main.head") in keys,
            "main head capture is missing from a later frame")
    cells = len(keys)
    total_pairs = pairs + warmups
    expected_pair_count = cells * total_pairs
    require(len(pair_rows) == expected_pair_count,
            f"expected {expected_pair_count} pair events, found {len(pair_rows)}")
    require(integer(final.get("pairs_completed"), "invalid completed pair count") == expected_pair_count,
            "summary completed pair count does not match schedule")
    require(final.get("scope") == SCOPE, "summary scope mismatch")
    require(integer(final.get("activation_readbacks"), "invalid activation readback count") == 0,
            "summary reports activation readbacks")
    require(final.get("full_decode_coverage_claimed") is False,
            "summary claims full decode coverage")

    timeline = _timeline_rows(timeline_text)
    normalized_pairs: list[dict[str, Any]] = []
    cell_values: dict[tuple[int, str], int] = {}
    cell_records: dict[tuple[int, str], int] = {}
    seen_cells: set[tuple[int, str, int]] = set()
    pair_gpu: dict[tuple[int, str], dict[str, list[float]]] = {
        key: {"a": [], "b": [], "host_a": [], "host_b": [], "noise_abs": [], "noise_relative": []}
        for key in keys
    }
    first_graph: dict[str, int] | None = None
    first_reads: dict[str, int] | None = None
    previous_reads_after: dict[str, int] | None = None
    final_graph: dict[str, int] | None = None
    final_reads: dict[str, int] | None = None
    expected_value_sum = 0

    for ordinal, row in enumerate(pair_rows):
        pair_index, cell_ordinal = divmod(ordinal, cells)
        scheduled_keys = keys if pair_index % 2 == 0 else list(reversed(keys))
        wanted = scheduled_keys[cell_ordinal]
        frame = integer(row.get("frame"), f"pair event {ordinal}: invalid frame")
        family = row.get("family")
        cell = (frame, family)
        require(cell == wanted, f"pair event {ordinal}: frame/family does not match scheduled coverage")
        repetition = integer(row.get("pair"), f"pair event {ordinal}: invalid repetition")
        require(repetition == pair_index, f"pair event {ordinal}: unexpected repetition index")
        require(row.get("warmup") is (pair_index < warmups),
                f"pair event {ordinal}: warmup flag does not match schedule")
        require(integer(row.get("repeats"), f"pair event {ordinal}: invalid repeat count", 1) == repeats,
                f"pair event {ordinal}: repeat count changed")
        require((frame, family, repetition) not in seen_cells,
                f"duplicate replay cell {frame}/{family} pair {repetition}")
        seen_cells.add((frame, family, repetition))

        order = row.get("order")
        expected_order = ["a", "b"] if pair_index % 2 == 0 else ["b", "a"]
        require(order == expected_order, f"pair event {ordinal}: arm order does not match alternating schedule")
        command_indices = row.get("command_indices")
        require(isinstance(command_indices, dict) and set(command_indices) == {"a", "b"},
                f"pair event {ordinal}: missing arm command indices")
        command_a = integer(command_indices.get("a"), f"pair event {ordinal}: invalid arm-a command index")
        command_b = integer(command_indices.get("b"), f"pair event {ordinal}: invalid arm-b command index")
        require(command_a != command_b, f"pair event {ordinal}: arms reuse a command index")

        before, after = _graph_pair(row.get("graph_before"), row.get("graph_after"), ordinal)
        if first_graph is None:
            first_graph = before
            require(before["submitted"] >= 0, "invalid initial graph submission count")
        expected_first = first_graph["submitted"] + ordinal * 3
        arm_slots = {expected_order[0]: expected_first, expected_order[1]: expected_first + 1}
        require(command_indices == arm_slots,
                f"pair event {ordinal}: arm indices do not match the three-command ledger")
        require(before["submitted"] == expected_first,
                f"pair event {ordinal}: graph command ledger is not consecutive")
        if ordinal == 0:
            capture_graph = _valid_graph(capture.get("graph"), "capture")
            require(before == capture_graph,
                    "graph commands changed between capture and replay setup")
        final_graph = after

        negative = row.get("negative_control")
        is_negative_cell = pair_index == 0 and cell_ordinal == 0
        if is_negative_cell:
            require(isinstance(negative, list) and len(negative) == 2 and
                    integer(negative[0], "invalid negative-control mismatch count") == 2 and
                    integer(negative[1], "invalid negative-control nonfinite count") == 1,
                    "seeded negative control did not produce exactly [2,1]")
        else:
            require(negative is None, "negative control appeared outside the seeded first pair")
        receipt_bytes = integer(row.get("terminal_receipt_bytes"),
                                f"pair event {ordinal}: invalid terminal receipt size", 1)
        expected_receipt_bytes = 16 if is_negative_cell else 8
        require(receipt_bytes == expected_receipt_bytes,
                f"pair event {ordinal}: terminal receipt size does not match control")
        reads_before, reads_after = _readback_pair(
            row.get("readbacks_before"), row.get("readbacks_after"), receipt_bytes, ordinal)
        if first_reads is None:
            first_reads = reads_before
            capture_reads = _readbacks(capture.get("readbacks"), "capture")
            require(first_reads == capture_reads, "readbacks changed between capture and replay setup")
        else:
            require(reads_before == previous_reads_after,
                    f"pair {ordinal}: readback snapshots are discontinuous")
        previous_reads_after = reads_after
        final_reads = reads_after

        values_per_arm = integer(row.get("values_per_arm"),
                                 f"pair event {ordinal}: invalid values-per-arm", 1)
        old_value = cell_values.setdefault(cell, values_per_arm)
        require(values_per_arm == old_value,
                f"pair event {ordinal}: values per arm changed across repetitions")
        if pair_index == 0:
            expected_value_sum += values_per_arm
        record_count = integer(row.get("records"), f"pair event {ordinal}: invalid record count", 1)
        old_records = cell_records.setdefault(cell, record_count)
        require(record_count == old_records,
                f"pair event {ordinal}: record count changed across repetitions")
        if pair_index == 0:
            require(sum(cell_records.values()) <= records,
                    "replay cell record counts exceed capture record count")
        host = row.get("host_encode_seconds")
        require(isinstance(host, dict) and set(host) == {"a", "b"},
                f"pair event {ordinal}: missing per-arm host encoding times")
        host_a = finite_number(host.get("a"), f"pair event {ordinal}: invalid arm-a host time")
        host_b = finite_number(host.get("b"), f"pair event {ordinal}: invalid arm-b host time")
        require(host_a >= 0 and host_b >= 0, f"pair event {ordinal}: negative host encoding time")
        require(integer(row.get("bit_mismatches"), f"pair event {ordinal}: invalid bit mismatch count") == 0,
                f"pair event {ordinal}: bitwise mismatch detected")
        require(integer(row.get("nonfinite_pairs"), f"pair event {ordinal}: invalid nonfinite count") == 0,
                f"pair event {ordinal}: nonfinite output pair detected")

        pair_command_seconds: dict[str, float] = {}
        ledger_start = expected_first
        for index in range(ledger_start, ledger_start + 3):
            require(index in timeline, f"pair event {ordinal}: missing GPU command {index}")
            pair_command_seconds[str(index)] = _command_seconds(timeline[index], index)
        for name, index in (("a", command_a), ("b", command_b)):
            normalized = pair_command_seconds[str(index)] / repeats
            if pair_index >= warmups:
                pair_gpu[cell][name].append(normalized)
        if pair_index >= warmups:
            pair_gpu[cell]["host_a"].append(host_a)
            pair_gpu[cell]["host_b"].append(host_b)
            a_time, b_time = pair_command_seconds[str(command_a)] / repeats, pair_command_seconds[str(command_b)] / repeats
            pair_gpu[cell]["noise_abs"].append(abs(a_time - b_time))
            pair_gpu[cell]["noise_relative"].append(abs(a_time - b_time) / ((a_time + b_time) / 2.0))

        normalized_pairs.append({
            "frame": frame,
            "family": family,
            "pair": pair_index,
            "warmup": pair_index < warmups,
            "order": order,
            "command_indices": {"a": command_a, "b": command_b},
            "gpu_seconds": {"a": pair_command_seconds[str(command_a)],
                            "b": pair_command_seconds[str(command_b)],
                            "comparison_receipt": pair_command_seconds[str(ledger_start + 2)]},
            "values_per_arm": values_per_arm,
            "terminal_receipt_bytes": receipt_bytes,
        })

    require(len(seen_cells) == expected_pair_count, "replay schedule contains missing cells")
    require(expected_value_sum == covered_values,
            "captured values do not match the sum of replayable cells")
    require(sum(cell_records.values()) == records,
            "replay cell record counts do not cover capture records")
    require(integer(final.get("verified_values_per_arm"), "invalid verified value total") ==
            sum(row["values_per_arm"] for row in normalized_pairs),
            "summary verified value total does not match pair events")
    require(final_graph is not None and final_reads is not None, "replay has no final state")
    summary_graph = _valid_graph(final.get("graph"), "summary")
    require(summary_graph == final_graph, "summary graph snapshot differs from final replay pair")
    summary_reads = _readbacks(final.get("readbacks"), "summary")
    require(summary_reads == final_reads, "summary readback snapshot differs from final replay pair")

    replay_start = first_graph["submitted"] if first_graph is not None else 0
    replay_end = replay_start + expected_pair_count * 3
    expected_indices = set(range(replay_start, replay_end))
    observed_replay_indices = {index for index in timeline if replay_start <= index < replay_end}
    require(observed_replay_indices == expected_indices,
            "replay timeline contains missing or unassigned command indices")
    setup_rows = []
    for index in sorted(set(timeline) - expected_indices):
        raw = timeline[index]
        time_value: float | None = None
        if raw.get("gpu_timing_valid") is True:
            candidate = raw.get("gpu_seconds")
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool) and math.isfinite(float(candidate)) and candidate > 0:
                time_value = float(candidate)
        setup_rows.append({"index": index, "gpu_seconds": time_value,
                           "classification": "uncosted_setup_or_outside_replay"})

    cell_reports = []
    for frame, family in keys:
        samples = pair_gpu[(frame, family)]
        stats_a = _summary_stats(samples["a"])
        stats_b = _summary_stats(samples["b"])
        noise_abs = _summary_stats(samples["noise_abs"])
        noise_relative = _summary_stats(samples["noise_relative"])
        cell_reports.append({
            "frame": frame,
            "family": family,
            "measured_pairs": len(samples["a"]),
            "values_per_arm": cell_values[(frame, family)],
            "gpu_seconds_per_repeat": {"arm_a": stats_a, "arm_b": stats_b},
            "host_encode_seconds": {"arm_a": _summary_stats(samples["host_a"]),
                                    "arm_b": _summary_stats(samples["host_b"])},
            "identical_arm_noise": {
                "absolute_seconds": noise_abs,
                "relative_fraction": noise_relative,
            },
            "paired_arm_mean_median_seconds": statistics.median(
                (a + b) / 2 for a, b in zip(samples["a"], samples["b"])),
        })

    for frame in FRAMES:
        frame_cells = [row for row in cell_reports if row["frame"] == frame]
        # Rank locally by measured cost: rank 1 is the slowest captured cell.
        frame_cells.sort(key=lambda row: (-row["paired_arm_mean_median_seconds"], row["family"]))
        for rank, row in enumerate(frame_cells, 1):
            row["local_rank_within_frame"] = rank

    return {
        "format": FORMAT_NAME,
        "scope": SCOPE,
        "claim_boundary": {
            "ranking": "captured projection cells within each frame only; rank 1 is highest paired median GPU cost",
            "uncaptured_work": "attention execution, delta state, sampler, and all other uncaptured decode work remain unranked",
            "request_cost_additivity_assumed": False,
            "legacy_profile_subtraction_performed": False,
            "full_decode_coverage_claimed": False,
        },
        "provenance": {
            "events_jsonl_sha256": hashlib.sha256(events_text.encode("utf-8")).hexdigest(),
            "timeline_jsonl_sha256": hashlib.sha256(timeline_text.encode("utf-8")).hexdigest(),
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "capture_event": capture,
        "invocation": invocation,
        "summary_event": final,
        "pair_count": expected_pair_count,
        "verified_values_per_arm_total": sum(row["values_per_arm"] for row in normalized_pairs),
        "uncosted_setup_commands": setup_rows,
        "cells": cell_reports,
        "pairs": normalized_pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events_jsonl", type=Path)
    parser.add_argument("timeline_jsonl", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    events_text = args.events_jsonl.read_text(encoding="utf-8")
    timeline_text = args.timeline_jsonl.read_text(encoding="utf-8")
    report = analyze(events_text, timeline_text)
    with args.output_json.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
