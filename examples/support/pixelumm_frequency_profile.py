#!/usr/bin/env python3
"""CPU-only matched-frequency accounting for PixelUMM profile requests.

GPU root timestamps are grouped by their candidate phase families. Overlaps
are kept as explicit mixed-membership buckets; these timestamps do not measure
per-shader service time and are not an optimization promotion criterion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_SCHEMA = "cogni_ml.pixelumm_native_aa_profile/v1"
LAUNCH_SCHEMA = "cogni_ml.pixelumm_matched_frequency_profile/v1"
OBSERVER_SCHEMA = "cogni_ml.pixelumm_power_observer/v1"
OUTPUT_SCHEMA = "cogni_ml.pixelumm_frequency_profile_analysis/v1"
MEMORY_SCHEMA = "cogni_ml.pixelumm_memory_phase/v1"
MEMORY_BOUNDARIES = ("request_before_encode", "request_after_terminal_wait")
EXPECTED_COMMAND_SEGMENTS = 4939
EXPECTED_TOKEN_COUNT = 21120
MAX_OBSERVER_CLOCK_ENDPOINT_RESIDUAL_MS = 100


def _load_observer() -> Any:
    path = Path(__file__).with_name("pixelumm_power_observer.py")
    spec = importlib.util.spec_from_file_location("pixelumm_power_observer_reused", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load power observer helper at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_aa_validator() -> Any:
    path = Path(__file__).with_name("analyze_pixelumm_aa_profile.py")
    spec = importlib.util.spec_from_file_location("pixelumm_aa_profile_reused", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixed-baseline AA validator at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OBSERVER = _load_observer()
_AA = _load_aa_validator()


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: object, where: str) -> Mapping[str, Any]:
    _need(isinstance(value, dict), f"{where}: expected object")
    return value  # type: ignore[return-value]


def _array(value: object, where: str) -> list[Any]:
    _need(isinstance(value, list), f"{where}: expected array")
    return value  # type: ignore[return-value]


def _integer(value: object, where: str, *, minimum: int = 0) -> int:
    _need(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
          f"{where}: expected integer >= {minimum}")
    return int(value)


def _finite(value: object, where: str, *, minimum: float | None = None) -> float:
    _need(isinstance(value, (int, float)) and not isinstance(value, bool),
          f"{where}: expected finite number")
    result = float(value)
    _need(math.isfinite(result), f"{where}: expected finite number")
    if minimum is not None:
        _need(result >= minimum, f"{where}: expected value >= {minimum}")
    return result


def _json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value} is not accepted")


def _finite_tree(value: object, where: str) -> None:
    if isinstance(value, float):
        _need(math.isfinite(value), f"{where}: non-finite value")
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{where}[{index}]")


def _utc_unix_ms(value: object, where: str) -> int:
    _need(isinstance(value, str) and bool(value), f"{where}: expected UTC timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{where}: invalid ISO UTC timestamp") from error
    _need(parsed.tzinfo is not None and parsed.utcoffset() is not None,
          f"{where}: timestamp must include a UTC offset")
    return round(parsed.astimezone(timezone.utc).timestamp() * 1_000)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_constant=_json_constant)


def _parse_samples(raw: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"samples line {line_number}: invalid JSON: {error}") from error
        rows.append(dict(_object(row, f"samples line {line_number}")))
    return rows


def _validate_launch(launch_raw: object) -> tuple[dict[str, int], dict[str, Any]]:
    launch = _object(launch_raw, "launch")
    _need(launch.get("schema") == LAUNCH_SCHEMA, "launch schema mismatch")
    anchor = _object(launch.get("clock_anchor"), "launch clock_anchor")
    _need(anchor.get("implementation") == "mach_absolute_time()",
          "launch clock anchor must use mach_absolute_time()")
    before = _integer(anchor.get("monotonic_before_ns"), "clock_anchor.monotonic_before_ns", minimum=1)
    unix_ns = _integer(anchor.get("unix_ns"), "clock_anchor.unix_ns", minimum=1)
    after = _integer(anchor.get("monotonic_after_ns"), "clock_anchor.monotonic_after_ns", minimum=1)
    _need(before <= after, "launch clock anchor monotonic bracket is reversed")
    bracket_width = after - before
    _need(bracket_width <= 1_000_000,
          "launch clock anchor bracket exceeds the 1ms alignment bound")
    _need(isinstance(anchor.get("unix_ns"), int) and not isinstance(anchor.get("unix_ns"), bool)
          and anchor["unix_ns"] > 0,
          "clock_anchor.unix_ns: expected positive integer")
    gate_raw = _object(launch.get("frequency_gate"), "launch frequency_gate")
    difference = _finite(gate_raw.get("max_mean_relative_difference"),
                         "frequency_gate.max_mean_relative_difference", minimum=0.0)
    _need(difference == 0.02, "launch frequency match threshold must be exactly 2 percent")
    minimum_samples = _integer(gate_raw.get("minimum_interior_samples"),
                                "frequency_gate.minimum_interior_samples", minimum=1)
    _need(minimum_samples == 5, "launch frequency gate must require at least five interior samples")
    residency = _finite(gate_raw.get("minimum_mean_active_residency_pct"),
                         "frequency_gate.minimum_mean_active_residency_pct", minimum=0.0)
    _need(residency == 99.0, "launch frequency gate must require at least 99 percent mean residency")
    _need(gate_raw.get("all_observed_samples_valid") is True,
          "launch must require all observed power samples to be valid")
    return ({"monotonic_before_ns": before, "unix_ns": int(anchor["unix_ns"]),
             "monotonic_after_ns": after,
             "midpoint_ns": before + bracket_width // 2,
             "bracket_width_ns": bracket_width,
             "half_bracket_uncertainty_ns": (bracket_width + 1) // 2},
            {"max_mean_relative_difference": difference,
             "minimum_interior_samples": minimum_samples,
             "minimum_mean_active_residency_pct": residency})


def _validate_report(report_raw: object, expected_command_segments: int,
                     expected_token_count: int | None = EXPECTED_TOKEN_COUNT
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    report = _object(report_raw, "report")
    _need(report.get("compare_cpu_diagnostics") is False and
          report.get("reverse_cpu_diagnostics") is False and
          report.get("align_cpu_diagnostic_lifecycle") is False and
          report.get("reverse_tiled_gate_up") is False and
          _integer(report.get("tiled_gate_up_rows"), "report.tiled_gate_up_rows") == 0,
          "report has an incompatible CPU-diagnostic or tiled-FFN selector")
    try:
        _AA._validate_report(report_raw, "0" * 64, "<in-memory report>", expected_command_segments)
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(f"report failed the fixed-baseline AA contract: {error}") from error
    _need(report.get("schema") == REPORT_SCHEMA and report.get("generation") is False and
          report.get("scope") == "native_36_blocks_plus_global_head_in_one_gpu_chain" and
          report.get("profile_aa") is True,
          "report is not the fixed-baseline 36-block-plus-head profile-AA report")
    _need(report.get("passed") is True and report.get("failure") is None and
          report.get("cleanup_quarantined") is False and
          report.get("spacing_abort_before_submission", False) is False,
          "report failed or did not complete cleanly")
    requests_raw = _array(report.get("requests"), "report requests")
    _need(bool(requests_raw), "report has no requests")
    requests: list[dict[str, Any]] = []
    ids: list[int] = []
    for position, row_raw in enumerate(requests_raw):
        row = dict(_object(row_raw, f"request row {position}"))
        request_id = _integer(row.get("request"), f"request row {position}.request")
        _need(row.get("passed") is True and isinstance(row.get("profiling"), bool),
              f"request {request_id} failed or lacks a profiling flag")
        ids.append(request_id)
        requests.append(row)
    _need(ids == list(range(len(ids))), "report request indices must be unique and contiguous from zero")
    _need(sum(row["profiling"] is True for row in requests) == 2,
          "report must contain exactly two profiled baseline requests")
    for row in requests:
        if row["profiling"]:
            _need(isinstance(row.get("root_profile"), dict) and
                  isinstance(row.get("phase_profile"), dict),
                  f"profiled request {row['request']} is missing GPU phase boundaries")
            phase = _object(row["phase_profile"], f"request {row['request']}.phase_profile")
            if expected_token_count is not None:
                _need(_integer(phase.get("token_count"),
                               f"request {row['request']}.phase_profile.token_count", minimum=1)
                      == expected_token_count,
                      f"request {row['request']} phase profile is not scoped to J={expected_token_count}")
    memory = [dict(_object(item, f"memory event {index}"))
              for index, item in enumerate(_array(report.get("memory_phases"), "report memory_phases"))]
    return requests, memory


def _validate_fixture_pins(launch_raw: object, report_raw: object) -> dict[str, str]:
    launch = _object(launch_raw, "launch")
    report = _object(report_raw, "report")
    argv = _array(launch.get("native_argv"), "launch native_argv")
    _need(all(isinstance(value, str) for value in argv) and argv.count("--profile-aa") == 1,
          "launch native argv must select exactly one fixed-baseline A/A profile run")
    option_pairs = {
        "--reference": ("--reference-sha256", "reference_sha256"),
        "--alternate-reference": ("--alternate-reference-sha256", "alternate_reference_sha256"),
        "--head-reference": ("--head-reference-sha256", "head_reference_sha256"),
        "--head-alternate-reference": ("--head-alternate-reference-sha256", "head_alternate_reference_sha256"),
        "--bf16-cache": ("--bf16-cache-sha256", "bf16_cache_sha256"),
    }
    expected_pins: dict[str, str] = {}
    report_fields: dict[str, str] = {}
    for path_option, (digest_option, report_field) in option_pairs.items():
        path_positions = [index for index, value in enumerate(argv) if value == path_option]
        digest_positions = [index for index, value in enumerate(argv) if value == digest_option]
        _need(len(path_positions) == len(digest_positions) == 1 and
              path_positions[0] + 1 < len(argv) and digest_positions[0] + 1 < len(argv),
              f"launch argv must specify exactly one {path_option} and {digest_option}")
        path_value = argv[path_positions[0] + 1]
        digest_value = argv[digest_positions[0] + 1]
        _need(isinstance(path_value, str) and bool(path_value) and
              isinstance(digest_value, str) and len(digest_value) == 64 and
              all(character in "0123456789abcdef" for character in digest_value),
              f"launch {path_option} fixture pin is malformed")
        expected_pins[path_value] = digest_value
        report_fields[report_field] = digest_value
    pins = _object(launch.get("fixture_pins"), "launch fixture_pins")
    _need(len(expected_pins) == len(option_pairs),
          "launch fixture paths must be distinct across pinned inputs")
    _need(dict(pins) == expected_pins, "launch fixture_pins disagree with native argv")
    for field, digest in report_fields.items():
        _need(report.get(field) == digest, f"report {field} does not match the launch fixture pin")
    return expected_pins


def _request_ranges(requests: Sequence[Mapping[str, Any]], memory: Sequence[Mapping[str, Any]],
                    anchor: Mapping[str, int]) -> dict[int, tuple[int, int]]:
    wanted = set(MEMORY_BOUNDARIES)
    found: dict[tuple[int, str], int] = {}
    request_ids = {int(row["request"]) for row in requests}
    for index, event in enumerate(memory):
        if event.get("phase") not in wanted:
            continue
        _need(event.get("schema") == MEMORY_SCHEMA,
              f"memory boundary event {index} has wrong schema")
        request_id = _integer(event.get("request"), f"memory event {index}.request")
        if request_id not in request_ids:
            continue
        _need(event.get("guard_timestamp_clock") == "mach_absolute_time",
              f"memory boundary event {index} lacks the Mach guard clock")
        key = (request_id, str(event["phase"]))
        _need(key not in found, f"duplicate memory boundary {key}")
        found[key] = _integer(event.get("guard_monotonic_ns"),
                              f"memory boundary {key}.guard_monotonic_ns", minimum=1)
    ranges: dict[int, tuple[int, int]] = {}
    for request_id in sorted(request_ids):
        bounds = []
        for phase in MEMORY_BOUNDARIES:
            key = (request_id, phase)
            _need(key in found, f"request {request_id} is missing memory boundary {phase}")
            bounds.append(found[key])
        start_ns, end_ns = bounds
        _need(start_ns < end_ns, f"request {request_id} memory boundaries are reversed")
        ranges[request_id] = (start_ns, end_ns)

    ordered = [ranges[index] for index in sorted(ranges)]
    _need(all(ordered[index][1] <= ordered[index + 1][0]
              for index in range(len(ordered) - 1)),
          "request guard-clock ranges overlap or are out of order")
    return ranges


def _validate_observer(summary_raw: object, samples: Sequence[Mapping[str, Any]],
                       guard_ranges: Mapping[int, tuple[int, int]],
                       anchor: Mapping[str, int],
                       samples_bytes: bytes | None) -> dict[str, Any]:
    _finite_tree(summary_raw, "observer summary")
    _finite_tree(list(samples), "samples")
    summary = _object(summary_raw, "observer summary")
    status = summary.get("status")
    _need(summary.get("schema") == OBSERVER_SCHEMA and status in ("complete", "stopped_early"),
          "observer summary is missing, incomplete, or has the wrong schema")
    clock = _object(summary.get("clock"), "observer summary clock")
    started = _integer(clock.get("started_monotonic_ns"), "observer started_monotonic_ns", minimum=1)
    finished = _integer(clock.get("finished_monotonic_ns"), "observer finished_monotonic_ns", minimum=1)
    elapsed = _integer(clock.get("monotonic_elapsed_ns"), "observer monotonic_elapsed_ns", minimum=1)
    _need(started < finished and elapsed == finished - started,
          "observer summary monotonic clock range is invalid")
    started_utc_ms = _utc_unix_ms(clock.get("started_utc"), "observer started_utc")
    finished_utc_ms = _utc_unix_ms(clock.get("finished_utc"), "observer finished_utc")
    _need(started_utc_ms < finished_utc_ms, "observer UTC capture interval is invalid")
    mapped_started_utc_ms = _map_monotonic_to_unix_ms(started, anchor)
    mapped_finished_utc_ms = _map_monotonic_to_unix_ms(finished, anchor)
    started_residual_ms = started_utc_ms - mapped_started_utc_ms
    finished_residual_ms = finished_utc_ms - mapped_finished_utc_ms
    _need(abs(started_residual_ms) <= MAX_OBSERVER_CLOCK_ENDPOINT_RESIDUAL_MS and
          abs(finished_residual_ms) <= MAX_OBSERVER_CLOCK_ENDPOINT_RESIDUAL_MS,
          "observer UTC endpoint residual exceeds the 100ms launch-anchor bound")
    utc_wall_elapsed_residual_ms = (finished_utc_ms - started_utc_ms) - elapsed / 1_000_000
    _need(abs(utc_wall_elapsed_residual_ms) <= 2 * MAX_OBSERVER_CLOCK_ENDPOINT_RESIDUAL_MS,
          "observer UTC wall elapsed differs from its monotonic elapsed by more than 200ms")
    _need(_integer(clock.get("sample_timestamp_resolution_ms"),
                   "observer sample_timestamp_resolution_ms", minimum=1) == 1_000,
          "observer clock must declare one-second sample timestamp precision")
    _need(min(start for start, _ in guard_ranges.values()) >= started and
          max(end for _, end in guard_ranges.values()) <= finished,
          "observer monotonic coverage does not enclose every request")
    parsed = _object(summary.get("parsed_samples"), "observer parsed_samples")
    coverage = _object(parsed.get("coverage"), "observer parsed_samples.coverage")
    expected_samples = _integer(coverage.get("expected_sample_count"),
                                "observer expected_sample_count", minimum=1)
    _need((status == "complete" and expected_samples == len(samples)) or
          (status == "stopped_early" and expected_samples > len(samples)),
          "observer sample count does not match its complete or bounded-stop status")
    _need(_integer(coverage.get("sample_count"), "observer sample_count") == len(samples) and
          bool(samples) and coverage.get("all_observed_samples_valid") is True and
          _integer(coverage.get("valid_sample_count"), "observer valid_sample_count") == len(samples) and
          coverage.get("duration_complete") is (status == "complete") and
          coverage.get("interrupted") is (status == "stopped_early") and
          _finite(coverage.get("valid_sample_fraction"), "observer valid_sample_fraction") == 1.0,
          "observer summary and samples disagree or report invalid sample coverage")
    collector = _object(summary.get("collector"), "observer collector")
    _need(collector.get("sampler_stopped") is True,
          "observer did not confirm sampler shutdown")
    _need(_integer(collector.get("sample_interval_ms"),
                   "observer sample_interval_ms", minimum=1) == 2_000,
          "observer capture interval must be two seconds")
    _need(_integer(collector.get("planned_sample_count"), "observer planned_sample_count", minimum=1) == expected_samples and
          _integer(collector.get("actual_returncode"), "observer actual_returncode", minimum=0) == 0 and
          collector.get("interrupted_by_signal") in (None, 15) and
          collector.get("sampler_terminated_by_signal") is False and
          collector.get("timed_out") is False and collector.get("launch_error") is None,
          "observer collector did not complete the planned capture cleanly")
    _need((status == "complete" and collector.get("interrupted_by_signal") is None) or
          (status == "stopped_early" and collector.get("interrupted_by_signal") == 15),
          "observer completion status does not match the bounded SIGTERM stop")
    stop_attempts = _array(collector.get("stop_attempts"), "observer stop_attempts")
    _need(all(_object(attempt, f"observer stop attempt {index}").get("signal") == "INT"
                for index, attempt in enumerate(stop_attempts)),
          "observer sampler required escalation beyond the bounded SIGINT stop")
    _need(status != "stopped_early" or bool(stop_attempts),
          "stopped-early observer has no recorded sampler SIGINT stop attempt")
    _need(all(attempt.get("sent") is True and
              type(attempt.get("returncode")) is int and attempt["returncode"] == 0 and
              attempt.get("error") is None and attempt.get("stderr_excerpt") is None
              for attempt in stop_attempts),
          "observer sampler SIGINT attempt did not succeed cleanly")
    process_group_id = _integer(collector.get("sampler_process_group_id"),
                                "observer sampler_process_group_id", minimum=1)
    group_checks = _array(collector.get("sampler_group_checks"), "observer sampler_group_checks")
    _need(bool(group_checks), "observer has no owned process-group check")
    last_check = _object(group_checks[-1], "observer final sampler group check")
    _need(_integer(last_check.get("process_group_id"), "observer final group-check PGID", minimum=1) == process_group_id and
          last_check.get("state") == "absent" and
          _integer(last_check.get("returncode"), "observer final group-check returncode", minimum=0) == 1 and
          last_check.get("stderr_excerpt") is None and last_check.get("error") is None,
          "observer final owned process-group check did not prove absence")
    raw_capture = _object(summary.get("raw_capture"), "observer raw_capture")
    _need(raw_capture.get("finalized") is True,
          "observer raw capture was not finalized")
    raw_digest = raw_capture.get("sha256")
    raw_size = _integer(raw_capture.get("size_bytes"), "observer raw_capture.size_bytes", minimum=1)
    _need(raw_capture.get("file") == "powermetrics.txt" and
          isinstance(raw_digest, str) and len(raw_digest) == 64 and
          all(character in "0123456789abcdef" for character in raw_digest),
          "observer raw capture identity or digest is missing")
    _need(parsed.get("file") == "samples.jsonl", "observer parsed sample file identity changed")
    digest = parsed.get("sha256")
    size_bytes = _integer(parsed.get("size_bytes"), "observer parsed_samples.size_bytes", minimum=1)
    _need(isinstance(digest, str) and len(digest) == 64 and
          all(character in "0123456789abcdef" for character in digest),
          "observer parsed sample digest is missing or malformed")
    if samples_bytes is None:
        samples_bytes = b"".join(
            (json.dumps(sample, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
            for sample in samples
        )
    _need(len(samples_bytes) == size_bytes and hashlib.sha256(samples_bytes).hexdigest() == digest,
          "observer sample bytes do not match summary digest/size")

    previous: tuple[int, int] | None = None
    seen: set[tuple[int, int]] = set()
    seen_indices: set[int] = set()
    seen_timestamps: set[int] = set()
    previous_timestamp: int | None = None
    for index, sample_raw in enumerate(samples):
        sample = _object(sample_raw, f"sample {index}")
        _need(sample.get("valid") is True, f"sample {index} is not valid")
        sample_index = _integer(sample.get("sample_index"), f"sample {index}.sample_index")
        _need(sample_index == index and sample_index not in seen_indices,
              f"sample {index} index is missing, duplicated, or out of order")
        seen_indices.add(sample_index)
        start = _integer(sample.get("sample_window_start_unix_ms"),
                         f"sample {index}.sample_window_start_unix_ms")
        end = _integer(sample.get("sample_window_end_unix_ms"),
                       f"sample {index}.sample_window_end_unix_ms")
        elapsed = _finite(sample.get("elapsed_ms"), f"sample {index}.elapsed_ms", minimum=0.000001)
        _need(end > start and math.isclose(end - start, elapsed, rel_tol=0.0, abs_tol=1.0),
              f"sample {index} has invalid window bounds")
        timestamp = _integer(sample.get("timestamp_unix_ms"), f"sample {index}.timestamp_unix_ms")
        resolution = _integer(sample.get("timestamp_resolution_ms"),
                              f"sample {index}.timestamp_resolution_ms", minimum=1)
        _need(resolution == 1_000 and sample.get("timestamp_is_window_end_assumption") is True and
              timestamp == end and round(timestamp - elapsed) == start,
              f"sample {index} timestamp/window alignment differs from observer semantics")
        timestamp_utc_ms = _utc_unix_ms(sample.get("timestamp_utc"), f"sample {index}.timestamp_utc")
        start_utc_ms = _utc_unix_ms(sample.get("sample_window_start_utc"),
                                   f"sample {index}.sample_window_start_utc")
        end_utc_ms = _utc_unix_ms(sample.get("sample_window_end_utc"),
                                 f"sample {index}.sample_window_end_utc")
        _need(abs(timestamp_utc_ms - timestamp) <= resolution and
              abs(end_utc_ms - end) <= resolution and abs(start_utc_ms - start) <= 1,
              f"sample {index} UTC and Unix-millisecond window fields disagree")
        _need(timestamp not in seen_timestamps, f"sample {index} duplicates a whole-second timestamp")
        seen_timestamps.add(timestamp)
        if previous_timestamp is not None:
            _need(0 < timestamp - previous_timestamp <= 2_000 + resolution,
                  "observer sample timestamps have a missing interval beyond one-second quantization")
        previous_timestamp = timestamp
        _finite(sample.get("gpu_active_frequency_mhz"),
                f"sample {index}.gpu_active_frequency_mhz", minimum=0.0)
        residency = _finite(sample.get("gpu_active_residency_pct"),
                            f"sample {index}.gpu_active_residency_pct", minimum=0.0)
        power = _finite(sample.get("gpu_power_mw"), f"sample {index}.gpu_power_mw", minimum=0.0)
        _need(residency <= 100.0, f"sample {index} residency exceeds 100 percent")
        _need(isinstance(sample.get("thermal_pressure"), str) and bool(sample["thermal_pressure"]),
              f"sample {index} lacks thermal-pressure status")
        bounds = (start, end)
        _need(bounds not in seen, f"duplicate observer sample window at sample {index}")
        seen.add(bounds)
        if previous is not None:
            previous_start, previous_end = previous
            _need(start > previous_start and end > previous_end,
                  "observer sample windows are not strictly chronological")
            _need(previous_end - start <= resolution,
                  "observer sample windows overlap beyond whole-second timestamp quantization")
        previous = bounds
        _need(start >= started_utc_ms - resolution and end <= finished_utc_ms + resolution,
              f"sample {index} window lies outside the observer's UTC capture interval")
    return {"status": status,
            "observer_utc_interval": {"started_utc": clock["started_utc"],
                                      "finished_utc": clock["finished_utc"]},
            "utc_endpoint_anchor_residual_ms": {
                "started": started_residual_ms,
                "finished": finished_residual_ms,
                "positive_means_observer_utc_is_ahead": True,
                "maximum_absolute_residual_ms": MAX_OBSERVER_CLOCK_ENDPOINT_RESIDUAL_MS},
            "utc_wall_elapsed_residual_ms": utc_wall_elapsed_residual_ms,
            "observer_monotonic_interval_ns": {"started": started, "finished": finished},
            "sample_sha256": digest, "sample_size_bytes": size_bytes,
            "sample_count": len(samples), "planned_sample_count": expected_samples,
            "sample_interval_ms": 2_000, "sample_timestamp_resolution_ms": 1_000,
            "maximum_inter_sample_timestamp_delta_ms": 3_000,
            "maximum_inter_sample_overlap_ms": 1_000,
            "sampler_process_group_id": process_group_id,
            "final_owned_group_state": "absent", "sampler_stopped": True,
            "raw_capture_finalized": True, "raw_capture_size_bytes": raw_size,
            "raw_capture_sha256": raw_digest,
            "collector_returncode": collector.get("actual_returncode"),
            "collector_interrupted_by_signal": collector.get("interrupted_by_signal"),
            "sampler_stop_signals": [attempt["signal"] for attempt in stop_attempts]}


def _map_monotonic_to_unix_ms(value_ns: int, anchor: Mapping[str, int]) -> int:
    unix_ns = anchor["unix_ns"] + value_ns - anchor["midpoint_ns"]
    _need(unix_ns > 0, "clock anchor maps a request boundary before the Unix epoch")
    return (unix_ns + 500_000) // 1_000_000


def _family(label: str) -> str:
    if label.startswith("block_") and "/" in label:
        return label.split("/", 1)[1]
    return label


def _union_length(intervals: Sequence[tuple[float, float]]) -> float:
    valid = sorted((start, end) for start, end in intervals if end > start)
    if not valid:
        return 0.0
    total = 0.0
    left, right = valid[0]
    for start, end in valid[1:]:
        if start > right:
            total += right - left
            left, right = start, end
        elif end > right:
            right = end
    return total + right - left


def interval_accounting(roots_raw: Sequence[Mapping[str, Any]],
                        candidates_raw: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Partition the GPU interval union by simultaneous candidate-family sets."""
    _need(len(roots_raw) == len(candidates_raw) and bool(roots_raw),
          "root intervals and phase candidates must have equal nonzero lengths")
    intervals: list[tuple[float, float]] = []
    memberships: list[frozenset[str]] = []
    for index, (root, labels_raw) in enumerate(zip(roots_raw, candidates_raw)):
        start = _finite(root.get("gpu_start_seconds"), f"root {index} gpu_start_seconds", minimum=0.0)
        end = _finite(root.get("gpu_end_seconds"), f"root {index} gpu_end_seconds", minimum=0.0)
        _need(end >= start, f"root {index} GPU interval ends before it starts")
        _need(isinstance(labels_raw, (list, tuple)),
              f"root {index} candidate phases: expected array")
        labels = list(labels_raw)
        _need(bool(labels), f"root {index} has no candidate phase")
        normalized: set[str] = set()
        for label in labels:
            _need(isinstance(label, str) and bool(label), f"root {index} has an invalid candidate label")
            normalized.add(_family(label))
        intervals.append((start, end))
        memberships.append(frozenset(normalized))

    families = sorted({family for membership in memberships for family in membership})
    same_family = {family: _union_length([intervals[index] for index, members in enumerate(memberships)
                                          if family in members])
                   for family in families}
    definite_family = {family: _union_length([intervals[index] for index, members in enumerate(memberships)
                                              if members == frozenset((family,))])
                       for family in families}

    endpoints = sorted({point for interval in intervals for point in interval})
    bucket_totals: dict[tuple[tuple[str, ...], ...], float] = {}
    for left, right in zip(endpoints, endpoints[1:]):
        if right <= left:
            continue
        active_memberships: list[tuple[str, ...]] = []
        for (start, end), member_set in zip(intervals, memberships):
            if start < right and end > left:
                active_memberships.append(tuple(sorted(member_set)))
        if active_memberships:
            # Preserve the full multiset: a single root that is ambiguous
            # across two phases is distinct from two overlapping roots with
            # one phase candidate each, and duplicate roots remain visible.
            signature = tuple(sorted(active_memberships))
            bucket_totals[signature] = bucket_totals.get(signature, 0.0) + right - left

    chain_union = _union_length(intervals)
    bucket_total = math.fsum(bucket_totals.values())
    _need(math.isclose(bucket_total, chain_union, rel_tol=1e-10, abs_tol=1e-9),
          "disjoint candidate-family buckets do not conserve the GPU interval union")
    buckets = [{"active_root_candidate_family_sets": [list(members) for members in signature],
                "seconds": seconds}
               for signature, seconds in sorted(bucket_totals.items())]
    return {"root_count": len(intervals),
            "chain_union_seconds": chain_union,
            "same_family_union_seconds": same_family,
            "definite_single_family_union_seconds": definite_family,
            "disjoint_membership_buckets": buckets,
            "disjoint_membership_total_seconds": bucket_total,
            "conservation_error_seconds": bucket_total - chain_union}


def _api_accounting(phase_profile: Mapping[str, Any]) -> dict[str, Any]:
    api = phase_profile.get("api_profile")
    if not isinstance(api, dict) or not isinstance(api.get("phases"), list):
        return {"available": False, "reason": "profile has no API phase summary"}
    exclusive: dict[str, dict[str, int]] = {}
    nested: dict[str, dict[str, int]] = {}
    for phase_index, phase_raw in enumerate(api["phases"]):
        phase = _object(phase_raw, f"API phase {phase_index}")
        phase_name = phase.get("pixelumm_phase")
        _need(isinstance(phase_name, str) and bool(phase_name),
              f"API phase {phase_index} lacks PixelUMM phase name")
        operations = _object(phase.get("operations"), f"API phase {phase_index}.operations")
        for operation, stats_raw in operations.items():
            stats = _object(stats_raw, f"API {phase_name}.{operation}")
            for field, target in (("exclusive_wall_total_ns", exclusive),
                                  ("nested_wall_total_ns", nested)):
                value = _integer(stats.get(field), f"API {phase_name}.{operation}.{field}")
                by_operation = target.setdefault(str(phase_name), {})
                by_operation[str(operation)] = by_operation.get(str(operation), 0) + value
    return {"available": True,
            "exclusive_wall_ns_by_phase_and_operation": exclusive,
            "nested_wall_ns_by_phase_and_operation": nested,
            "caveat": "API wall timings are CPU-side instrumentation; they are separate from GPU root interval accounting."}


def _validate_profile_roots(request_id: int, request: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[list[str]]]:
    root_profile = _object(request.get("root_profile"), f"request {request_id}.root_profile")
    roots = [dict(_object(root, f"request {request_id} root {index}"))
             for index, root in enumerate(_array(root_profile.get("roots"),
                                                  f"request {request_id}.root_profile.roots"))]
    phase = _object(request.get("phase_profile"), f"request {request_id}.phase_profile")
    candidates = _array(phase.get("root_candidates"),
                        f"request {request_id}.phase_profile.root_candidates")
    _need(roots and len(roots) == len(candidates) and
          _integer(root_profile.get("root_count"), f"request {request_id} root_count") == len(roots),
          f"request {request_id} root interval/candidate coverage is incomplete")
    profile_roots = phase.get("roots")
    if isinstance(profile_roots, list):
        _need(len(profile_roots) == len(roots), f"request {request_id} phase profile root count changed")
        for index, (root_raw, phase_root_raw) in enumerate(zip(roots, profile_roots)):
            phase_root = _object(phase_root_raw, f"request {request_id} phase root {index}")
            _need(phase_root.get("sequence") == root_raw.get("sequence"),
                  f"request {request_id} root sequence differs between profiles")
            if "candidate_phases" in phase_root:
                _need(set(phase_root["candidate_phases"]) == set(candidates[index]),
                      f"request {request_id} root {index} candidate phases disagree")
    sequences = [_integer(root.get("sequence"), f"request {request_id} root {index}.sequence")
                 for index, root in enumerate(roots)]
    _need(len(sequences) == len(set(sequences)), f"request {request_id} has duplicate root sequences")
    for index, root in enumerate(roots):
        _need(root.get("completed") is True and root.get("has_error") is False,
              f"request {request_id} root {index} is incomplete or errored")
    return roots, [list(_array(row, f"request {request_id} candidates {i}"))
                   for i, row in enumerate(candidates)]


def analyze_documents(report_raw: object, launch_raw: object, observer_summary_raw: object,
                      samples_raw: Sequence[Mapping[str, Any]], *,
                      samples_bytes: bytes | None = None,
                      expected_command_segments: int = EXPECTED_COMMAND_SEGMENTS,
                      expected_token_count: int | None = EXPECTED_TOKEN_COUNT) -> dict[str, Any]:
    _finite_tree(report_raw, "report")
    _finite_tree(launch_raw, "launch")
    requests, memory = _validate_report(report_raw, expected_command_segments, expected_token_count)
    anchor, gate = _validate_launch(launch_raw)
    fixture_pins = _validate_fixture_pins(launch_raw, report_raw)
    guard_ranges = _request_ranges(requests, memory, anchor)
    observer_provenance = _validate_observer(observer_summary_raw, samples_raw,
                                             guard_ranges, anchor, samples_bytes)

    utc_ranges: dict[str, tuple[int, int]] = {}
    request_labels: dict[int, str] = {}
    boundary_extra_exclusion_ms = (anchor["half_bracket_uncertainty_ns"] + 999_999) // 1_000_000 + 1
    for request_id, (start_ns, end_ns) in guard_ranges.items():
        label = f"request_{request_id}"
        request_labels[request_id] = label
        start_ms = _map_monotonic_to_unix_ms(start_ns, anchor) + boundary_extra_exclusion_ms
        end_ms = _map_monotonic_to_unix_ms(end_ns, anchor) - boundary_extra_exclusion_ms
        _need(start_ms < end_ms, f"request {request_id} is shorter than clock-alignment exclusion")
        utc_ranges[label] = (start_ms, end_ms)
    range_summaries = _OBSERVER.summarize_utc_ranges(samples_raw, utc_ranges)
    request_summaries: dict[int, dict[str, Any]] = {}
    for request_id, label in request_labels.items():
        row = range_summaries[label]
        eligible = _integer(row.get("eligible_samples"), f"{label}.eligible_samples")
        valid = _integer(row.get("valid_samples"), f"{label}.valid_samples")
        fraction = row.get("valid_sample_fraction")
        _need(eligible >= gate["minimum_interior_samples"] and valid == eligible and fraction == 1.0,
              f"{label} lacks the required valid interior power samples")
        frequency = _object(row.get("gpu_active_frequency_mhz"), f"{label}.frequency summary")
        residency = _object(row.get("gpu_active_residency_pct"), f"{label}.residency summary")
        request_summaries[request_id] = {
            "start_unix_ms": utc_ranges[label][0], "end_unix_ms": utc_ranges[label][1],
            "interior_samples": eligible,
            "mean_frequency_mhz": _finite(frequency.get("mean"), f"{label}.mean_frequency_mhz", minimum=0.000001),
            "mean_active_residency_pct": _finite(residency.get("mean"), f"{label}.mean_residency_pct", minimum=0.0),
        }

    request_by_id = {int(row["request"]): row for row in requests}
    profile_rows: list[dict[str, Any]] = []
    frequency_profile_rows: list[dict[str, Any]] = []
    match_passes: list[bool] = []
    for position, request in enumerate(requests):
        if request["profiling"] is not True:
            continue
        request_id = int(request["request"])
        roots, candidates = _validate_profile_roots(request_id, request)
        phase = _object(request["phase_profile"], f"request {request_id}.phase_profile")
        accounting = interval_accounting(roots, candidates)
        profile_rows.append({"request": request_id,
                             "kind": request.get("kind"),
                             "gpu_interval_accounting": accounting,
                             "api_profile_accounting": _api_accounting(phase),
                             "topology": {key: phase.get(key) for key in
                                          ("shared_gate_up", "fused_kv_gather", "scale_softmax",
                                           "paired_window_attention") if key in phase}})

        prior = next((int(row["request"]) for row in reversed(requests[:position])
                      if row["profiling"] is False), None)
        following = next((int(row["request"]) for row in requests[position + 1:]
                          if row["profiling"] is False), None)
        controls = [control for control in (prior, following) if control is not None]
        comparisons: list[dict[str, Any]] = []
        for control in controls:
            profiled = request_summaries[request_id]
            off = request_summaries[control]
            relative = abs(profiled["mean_frequency_mhz"] - off["mean_frequency_mhz"]) / off["mean_frequency_mhz"]
            on_residency_ok = profiled["mean_active_residency_pct"] >= gate["minimum_mean_active_residency_pct"]
            off_residency_ok = off["mean_active_residency_pct"] >= gate["minimum_mean_active_residency_pct"]
            frequency_ok = relative <= gate["max_mean_relative_difference"]
            pair_pass = frequency_ok and on_residency_ok and off_residency_ok
            comparisons.append({"control_request": control,
                                "control_position": "preceding" if control == prior else "following",
                                "profiled_mean_frequency_mhz": profiled["mean_frequency_mhz"],
                                "control_mean_frequency_mhz": off["mean_frequency_mhz"],
                                "relative_frequency_difference": relative,
                                "relative_difference_denominator": "off_control_mean_mhz",
                                "profiled_mean_active_residency_pct": profiled["mean_active_residency_pct"],
                                "control_mean_active_residency_pct": off["mean_active_residency_pct"],
                                "profiled_interior_samples": profiled["interior_samples"],
                                "control_interior_samples": off["interior_samples"],
                                "frequency_match": frequency_ok,
                                "profiled_residency_gate": on_residency_ok,
                                "control_residency_gate": off_residency_ok,
                                "passed": pair_pass})
        bracketing = prior is not None and following is not None
        on_pass = bracketing and len(comparisons) == 2 and all(row["passed"] for row in comparisons)
        match_passes.append(on_pass)
        frequency_profile_rows.append({"request": request_id,
                                       "nearest_preceding_off_request": prior,
                                       "nearest_following_off_request": following,
                                       "bracketed_by_off_requests": bracketing,
                                       "comparisons": comparisons,
                                       "passed": on_pass})

    return {"schema": OUTPUT_SCHEMA,
            "input_report_schema": REPORT_SCHEMA,
            "clock_conversion": {"source_clock": "mach_absolute_time nanoseconds",
                                 "anchor_implementation": "mach_absolute_time()",
                                 "anchor_monotonic_midpoint_ns": anchor["midpoint_ns"],
                                 "anchor_bracket_width_ns": anchor["bracket_width_ns"],
                                 "anchor_half_bracket_uncertainty_ns": anchor["half_bracket_uncertainty_ns"],
                                 "extra_boundary_exclusion_ms": boundary_extra_exclusion_ms,
                                 "utc_ranges_are_unix_ms": True,
                                 "observer_sample_timestamp_resolution_ms": 1_000,
                                 "observer_maximum_inter_sample_overlap_ms": 1_000,
                                 "observer_maximum_inter_sample_timestamp_delta_ms": 3_000,
                                 "observer_boundary_exclusion_uses_one_sample_interval": True,
                                 "transient_utc_step_limit": "endpoint checks cannot detect a UTC clock step that returns to its prior offset between observer capture endpoints"},
            "fixture_pins": fixture_pins,
            "observer_provenance": observer_provenance,
            "frequency_gate": gate,
            "request_power_windows": [dict({"request": request_id}, **summary)
                                      for request_id, summary in sorted(request_summaries.items())],
            "profiled_requests": profile_rows,
            "frequency_match_gate": {"passed": bool(match_passes) and all(match_passes),
                                     "relative_difference_formula": "abs(profile_mean - off_mean) / off_mean",
                                     "profiled_requests": frequency_profile_rows,
                                     "auto_promotion": False},
            "gpu_interval_caveat": "GPU root timestamps are candidate-phase interval accounting, not per-shader service time; overlapping families remain mixed buckets.",
            "claims_not_made": ["causal profiler overhead", "per-shader service time",
                                "optimization correctness", "speedup or promotion"]}


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="integrated PixelUMM JSON report")
    parser.add_argument("--launch", required=True, type=Path, help="external launch manifest with Mach/UTC anchor")
    parser.add_argument("--observer-summary", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path, help="power observer samples.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="new output JSON path; existing paths are refused")
    args = parser.parse_args(argv)
    try:
        report = _load_json(args.report)
        launch = _load_json(args.launch)
        observer_summary = _load_json(args.observer_summary)
        samples_bytes = args.samples.read_bytes()
        samples = _parse_samples(samples_bytes)
        result = analyze_documents(report, launch, observer_summary, samples,
                                   samples_bytes=samples_bytes)
        _write_new(args.output, result)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
