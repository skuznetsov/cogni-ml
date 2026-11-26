#!/usr/bin/env python3
"""Validate and summarize a paired full-forward Hy-Nomas text compaction run.

This is a CPU-only report analyzer. It validates the report's pinned upstream
fixture/cache lineage, unchanged 2e-5 gate, paired request receipts, sample
schedule, and the external memory guard log. It does not execute the model or
recompute tensor comparisons from device buffers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


SCHEMA = "cogni_ml.hy_nomas_text_compaction_benchmark"
SCHEMA_VERSION = 1
ANALYSIS_SCHEMA = "cogni_ml.hy_nomas_text_compaction_analysis"
ANALYSIS_SCHEMA_VERSION = 1
REFERENCE_SHA256 = "fd51333fb7908f64915a06bf5bdc392be88332270dd0c18c969eef26781a5c2e"
CACHE_INDEX_SHA256 = "a235ca592b47161472c7fd94cd3d886fce164480cd0571179c8860002063cbb0"
ATOL = 2e-5
RTOL = 2e-5
TEXT_ROW_ALIGNMENT = 32
RUNTIME_FLOOR_PCT = 10
MAX_SWAP_GROWTH_BYTES = 64 * 1024 * 1024
MAX_GUARD_FOOTPRINT_BYTES = 24 * 1024**3
MAX_GUARD_TIMEOUT_SEC = 900
MODES = ("padded", "compact")
CAPTURE_ELEMENTS = {
    "block0.img.output": 5280 * 2048,
    "block0.txt.output": 1280 * 2048,
    "block7.img.output": 5280 * 2048,
    "block7.txt.output": 1280 * 2048,
    "block53.img.output": 5280 * 2048,
    "block53.txt.output": 1280 * 2048,
    "conditioning.projected_text": 1280 * 2048,
    "conditioning.vec": 2048,
    "final.linear": 5280 * 32,
    "final.modulated": 5280 * 2048,
    "final.modulation": 4096,
    "final.norm": 5280 * 2048,
    "img_in": 5280 * 2048,
    "output.latent": 32 * 6 * 22 * 40,
}
TEXT_CAPTURES = frozenset((
    "block0.txt.output", "block7.txt.output", "block53.txt.output",
))
GEOMETRY = {
    "batches": 1,
    "video_tokens": 5280,
    "text_tokens": 1280,
    "valid_text_tokens": 23,
    "blocks": 54,
}
SAMPLE_METRICS = (
    "encode_seconds", "terminal_wait_seconds", "total_seconds",
    "gpu_elapsed_seconds", "command_segments",
    "named_allocations_during_encode",
    "activation_readbacks_before_terminal_wait", "terminal_wait_count",
    "diagnostic_readbacks",
)
SAMPLE_ORDERS = {
    0: (("padded", "compact"), ("compact", "padded"), ("padded", "compact")),
    1: (("compact", "padded"), ("padded", "compact"), ("compact", "padded")),
}


class ValidationError(ValueError):
    """The benchmark or guard report is incomplete or contradicts its claims."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{label} must be an array")
    return value


def _int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if type(value) not in (int, float):
        raise ValidationError(f"{label} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValidationError(f"{label} must be a finite number >= {minimum}")
    return result


def _bool(value: Any, label: str, *, expected: bool | None = None) -> bool:
    if type(value) is not bool or (expected is not None and value is not expected):
        suffix = "true" if expected is True else "false" if expected is False else "a boolean"
        raise ValidationError(f"{label} must be {suffix}")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_json(data: bytes, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValidationError(f"{label} contains nonfinite JSON number {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"{label} contains duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(data, parse_constant=reject_constant, object_pairs_hook=reject_duplicate_keys)
    except ValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{label} is not valid JSON: {error}") from None


def _expected_elements(mode: str, name: str, valid_text_tokens: int) -> int:
    elements = CAPTURE_ELEMENTS[name]
    if mode == "compact" and name in TEXT_CAPTURES:
        return valid_text_tokens * 2048
    return elements


def _validate_comparisons(value: Any, mode: str, request: int,
                          valid_text_tokens: int) -> dict[str, str]:
    rows = _array(value, f"{mode} request {request} comparisons")
    seen: dict[str, str] = {}
    for index, raw in enumerate(rows):
        label = f"{mode} request {request} comparison {index}"
        row = _mapping(raw, label)
        name = row.get("name")
        if not isinstance(name, str) or name not in CAPTURE_ELEMENTS:
            raise ValidationError(f"{label} has an unexpected capture name")
        if name in seen:
            raise ValidationError(f"{label} duplicates capture {name}")
        expected_elements = _expected_elements(mode, name, valid_text_tokens)
        if _int(row.get("elements"), f"{label}.elements", minimum=1) != expected_elements:
            raise ValidationError(f"{label} element count does not match {mode} representation")
        if _int(row.get("failures"), f"{label}.failures") != 0:
            raise ValidationError(f"{label} exceeds the unchanged finite tolerance")
        raw_differences = _int(row.get("raw_bit_differences"), f"{label}.raw_bit_differences")
        if raw_differences > expected_elements:
            raise ValidationError(f"{label} raw bit-difference count exceeds its element count")
        _bool(row.get("finite"), f"{label}.finite", expected=True)
        _number(row.get("max_abs"), f"{label}.max_abs")
        _sha256(row.get("native_sha256"), f"{label}.native_sha256")
        repeat_match = row.get("first_request_sha256_match")
        if request == 0:
            if repeat_match is not None:
                raise ValidationError(f"{label} first request must have no prior hash comparison")
        elif repeat_match is not True:
            raise ValidationError(f"{label} output drifted after scratch poisoning")
        seen[name] = row["native_sha256"]
    if set(seen) != set(CAPTURE_ELEMENTS):
        missing = sorted(set(CAPTURE_ELEMENTS) - set(seen))
        raise ValidationError(f"{mode} request {request} comparison inventory is incomplete: {missing}")
    return seen


def _validate_validations(report: dict[str, Any], valid_text_tokens: int) -> None:
    rows = _array(report.get("validations"), "validations")
    if len(rows) != 4:
        raise ValidationError("validations must contain both requests for both representations")
    found: dict[tuple[str, int], dict[str, str]] = {}
    for index, raw in enumerate(rows):
        label = f"validation {index}"
        row = _mapping(raw, label)
        mode = row.get("mode")
        request = row.get("request")
        if not isinstance(mode, str) or mode not in MODES or type(request) is not int or request not in (0, 1):
            raise ValidationError(f"{label} has an invalid mode/request identity")
        key = (mode, request)
        if key in found:
            raise ValidationError(f"{label} duplicates {mode} request {request}")
        _bool(row.get("passed"), f"{label}.passed", expected=True)
        _bool(row.get("scratch_poisoned"), f"{label}.scratch_poisoned", expected=request == 1)
        _int(row.get("poison_buffer_count"), f"{label}.poison_buffer_count",
             minimum=1 if request == 1 else 0)
        if request == 0 and row["poison_buffer_count"] != 0:
            raise ValidationError(f"{label} request 0 must not poison scratch")
        for field in ("named_allocations_during_encode", "activation_readbacks_before_terminal_wait"):
            if _int(row.get(field), f"{label}.{field}") != 0:
                raise ValidationError(f"{label} added a buffer allocation or early activation readback")
        if _int(row.get("terminal_wait_count"), f"{label}.terminal_wait_count") != 1:
            raise ValidationError(f"{label} must use exactly one terminal wait")
        _bool(row.get("excluded_suffix_poisoned"), f"{label}.excluded_suffix_poisoned",
              expected=mode == "compact" and request == 1)
        found[key] = _validate_comparisons(
            row.get("comparisons"), mode, request, valid_text_tokens
        )

    expected = {(mode, request) for mode in MODES for request in (0, 1)}
    if set(found) != expected:
        raise ValidationError("validation request set is incomplete")
    for mode in MODES:
        first, repeat = found[(mode, 0)], found[(mode, 1)]
        for name in CAPTURE_ELEMENTS:
            if first[name] != repeat[name]:
                raise ValidationError(f"{mode} output hash drifted between requests: {name}")


def _validate_geometry(value: Any) -> dict[str, int]:
    geometry = _mapping(value, "geometry")
    expected_keys = set(GEOMETRY) | {"compact_text_tokens"}
    if (set(geometry) != expected_keys
            or any(type(geometry.get(key)) is not int or geometry[key] != value
                   for key, value in GEOMETRY.items())):
        raise ValidationError("benchmark geometry is stale or mismatched")

    compact_tokens = _int(
        geometry.get("compact_text_tokens"), "geometry.compact_text_tokens",
        minimum=GEOMETRY["valid_text_tokens"],
    )
    if compact_tokens >= GEOMETRY["text_tokens"]:
        raise ValidationError("compact text-token storage must be below padded geometry")
    if (compact_tokens != GEOMETRY["valid_text_tokens"]
            and compact_tokens % TEXT_ROW_ALIGNMENT != 0):
        raise ValidationError("compact text-token storage must be valid-length or 32-token aligned")
    return {**GEOMETRY, "compact_text_tokens": compact_tokens}


def _validate_representation(report: dict[str, Any],
                             geometry: dict[str, int]) -> dict[str, int]:
    modes = _mapping(report.get("modes"), "modes")
    if set(modes) != set(MODES):
        raise ValidationError("modes must describe exactly padded and compact representations")
    result: dict[str, int] = {}
    expected_tokens = {
        "padded": geometry["text_tokens"],
        "compact": geometry["compact_text_tokens"],
    }
    for mode in MODES:
        item = _mapping(modes[mode], f"modes.{mode}")
        if _int(item.get("dit_text_tokens"), f"modes.{mode}.dit_text_tokens", minimum=1) != expected_tokens[mode]:
            raise ValidationError(f"modes.{mode} DIT text-token representation mismatch")
        result[mode] = _int(item.get("named_bytes"), f"modes.{mode}.named_bytes", minimum=1)
    if result["compact"] >= result["padded"]:
        raise ValidationError("compact representation must use fewer named bytes than padded")
    ceiling = _int(report.get("max_combined_named_bytes"), "max_combined_named_bytes", minimum=1)
    if result["padded"] + result["compact"] > ceiling:
        raise ValidationError("combined named residency exceeds the declared ceiling")
    return result


def _validate_epochs(report: dict[str, Any]) -> None:
    epochs = _array(report.get("epochs"), "epochs")
    if len(epochs) != 2:
        raise ValidationError("epochs must contain the two counterbalanced epochs")
    expected_warmups = {
        0: ["padded", "compact"],
        1: ["compact", "padded"],
    }
    for index, raw in enumerate(epochs):
        row = _mapping(raw, f"epoch {index}")
        epoch = row.get("epoch")
        if type(epoch) is not int or epoch not in (0, 1) or epoch != index:
            raise ValidationError("epochs must be ordered 0 then 1")
        if row.get("warmup_order") != expected_warmups[epoch]:
            raise ValidationError(f"epoch {epoch} warmup order does not counterbalance")
        if _int(row.get("pairs"), f"epoch {epoch}.pairs", minimum=1) != 3:
            raise ValidationError(f"epoch {epoch} must declare exactly 3 measured pairs")

    warmups = _array(report.get("warmups"), "warmups")
    if len(warmups) != 4:
        raise ValidationError("warmups must contain both representations in each epoch")
    expected_rows = [
        (epoch, position, mode)
        for epoch in (0, 1)
        for position, mode in enumerate(expected_warmups[epoch])
    ]
    for index, (raw, expected) in enumerate(zip(warmups, expected_rows, strict=True)):
        row = _mapping(raw, f"warmup {index}")
        if (type(row.get("epoch")) is not int or type(row.get("position")) is not int
                or not isinstance(row.get("mode"), str)
                or (row.get("epoch"), row.get("position"), row.get("mode")) != expected):
            raise ValidationError("warmup samples do not match declared counterbalanced order")
        _validate_sample_telemetry(row, f"warmup {index}")


def _validate_sample_telemetry(row: dict[str, Any], label: str) -> None:
    encode = _number(row.get("encode_seconds"), f"{label}.encode_seconds")
    wait = _number(row.get("terminal_wait_seconds"), f"{label}.terminal_wait_seconds")
    total = _number(row.get("total_seconds"), f"{label}.total_seconds", minimum=0.000000001)
    _number(row.get("gpu_elapsed_seconds"), f"{label}.gpu_elapsed_seconds", minimum=0.000000001)
    if total + 1e-9 < encode + wait:
        raise ValidationError(f"{label}.total_seconds underreports encode plus terminal wait")
    if _int(row.get("command_segments"), f"{label}.command_segments", minimum=1) < 1:
        raise ValidationError(f"{label} must submit at least one command segment")
    for field in (
        "named_allocations_during_encode", "activation_readbacks_before_terminal_wait",
        "diagnostic_readbacks",
    ):
        if _int(row.get(field), f"{label}.{field}") != 0:
            raise ValidationError(f"{label} violates the allocation/readback boundary")
    if _int(row.get("terminal_wait_count"), f"{label}.terminal_wait_count") != 1:
        raise ValidationError(f"{label} must use exactly one terminal wait")


def _validate_samples(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _array(report.get("samples"), "samples")
    if len(rows) != 12:
        raise ValidationError("samples must contain 3 paired measurements in each of 2 epochs")
    normalized: list[dict[str, Any]] = []
    expected_identity = [
        (epoch, pair, position, mode)
        for epoch in (0, 1)
        for pair, pair_order in enumerate(SAMPLE_ORDERS[epoch])
        for position, mode in enumerate(pair_order)
    ]
    for index, (raw, expected) in enumerate(zip(rows, expected_identity, strict=True)):
        label = f"sample {index}"
        row = _mapping(raw, label)
        identity = (row.get("epoch"), row.get("pair"), row.get("position"), row.get("mode"))
        if (any(type(row.get(field)) is not int for field in ("epoch", "pair", "position"))
                or not isinstance(row.get("mode"), str) or identity != expected):
            raise ValidationError("samples do not follow the interleaved AB/BA epoch schedule")
        _validate_sample_telemetry(row, label)
        normalized.append(row)
    return normalized


def _validate_guard(events: list[Any]) -> dict[str, Any]:
    if len(events) < 5:
        raise ValidationError("guard log is missing limits, process start, samples, or successful finish")
    records = [_mapping(event, f"guard event {i}") for i, event in enumerate(events)]
    if records[0].get("event") != "limits" or records[-1].get("event") != "finished":
        raise ValidationError("guard log must begin with limits and end with finished")
    limits = records[0]
    if limits.get("memory_accounting") != "file-cache":
        raise ValidationError("guard must use file-cache memory accounting")
    floor = _number(limits.get("runtime_floor_pct"), "guard runtime_floor_pct")
    if floor != RUNTIME_FLOOR_PCT or limits.get("runtime_floor_pct_exact") != "10":
        raise ValidationError("guard runtime floor must remain 10 percent")
    if _int(limits.get("max_swap_growth_bytes"), "guard max_swap_growth_bytes") != MAX_SWAP_GROWTH_BYTES:
        raise ValidationError("guard swap-growth cap must remain 64 MiB")
    footprint_cap = _int(limits.get("max_footprint_bytes"), "guard max_footprint_bytes", minimum=1)
    if footprint_cap > MAX_GUARD_FOOTPRINT_BYTES:
        raise ValidationError("guard footprint cap exceeds the 24 GiB paired-run limit")
    timeout = _int(limits.get("timeout_sec"), "guard timeout_sec", minimum=1)
    if timeout > MAX_GUARD_TIMEOUT_SEC:
        raise ValidationError("guard timeout exceeds the 900-second run limit")
    intermediate = records[1:-1]
    samples = [record for record in intermediate if record.get("event") == "sample"]
    starts = [record for record in intermediate if record.get("event") == "started"]
    if (not samples or len(starts) != 1
            or any(record.get("event") not in {"sample", "started"} for record in intermediate)
            or intermediate[0].get("event") != "sample"
            or intermediate[1].get("event") != "started"):
        raise ValidationError("guard log is missing samples or contains a failure event")
    start = starts[0]
    process_id = _int(start.get("pid"), "guard started pid", minimum=1)
    if _int(start.get("pgid"), "guard started pgid", minimum=1) != process_id:
        raise ValidationError("guard process group does not match its started process")
    initial_swap = _int(samples[0].get("swap_used_bytes"), "guard initial swap_used_bytes")
    previous_monotonic = _number(limits.get("monotonic"), "guard limits monotonic")
    for index, record in enumerate(records[1:], start=1):
        event = record.get("event")
        label = f"guard {event} {index}"
        monotonic = _number(record.get("monotonic"), f"{label}.monotonic")
        if monotonic <= previous_monotonic:
            raise ValidationError("guard event timestamps are not strictly increasing")
        previous_monotonic = monotonic
    for index, sample in enumerate(samples):
        label = f"guard sample {index}"
        physical = _int(sample.get("physical_bytes"), f"{label}.physical_bytes", minimum=1)
        headroom = _int(sample.get("headroom_bytes"), f"{label}.headroom_bytes")
        pressure = _number(sample.get("pressure_free_pct"), f"{label}.pressure_free_pct")
        if (pressure > 100 or pressure < RUNTIME_FLOOR_PCT
                or headroom < physical * RUNTIME_FLOOR_PCT / 100):
            raise ValidationError("guard sample fell below the 10 percent memory floor")
        footprint = _int(sample.get("group_footprint_bytes"), f"{label}.group_footprint_bytes")
        if footprint > footprint_cap:
            raise ValidationError("guard sample exceeded the process footprint cap")
        swap = _int(sample.get("swap_used_bytes"), f"{label}.swap_used_bytes")
        if swap - initial_swap > MAX_SWAP_GROWTH_BYTES:
            raise ValidationError("guard sample exceeded the 64 MiB swap-growth cap")
        for field in ("file_backed_bytes", "free_spec_bytes", "speculative_bytes", "group_live_processes"):
            _int(sample.get(field), f"{label}.{field}")
    initial = samples[0]
    startup_required = (
        _int(initial.get("physical_bytes"), "guard initial physical_bytes") * RUNTIME_FLOOR_PCT / 100
        + footprint_cap
    )
    if _int(initial.get("headroom_bytes"), "guard initial headroom_bytes") < startup_required:
        raise ValidationError("guard startup sample lacks the required floor plus footprint headroom")
    finished = records[-1]
    if _int(finished.get("returncode"), "guard finished returncode") != 0:
        raise ValidationError("guarded benchmark did not exit successfully")
    return {
        "sample_count": len(samples),
        "runtime_floor_pct": RUNTIME_FLOOR_PCT,
        "memory_accounting": "file-cache",
        "max_swap_growth_bytes": MAX_SWAP_GROWTH_BYTES,
        "max_footprint_bytes": footprint_cap,
        "timeout_sec": timeout,
        "peak_group_footprint_bytes": max(_int(sample["group_footprint_bytes"], "guard footprint") for sample in samples),
        "minimum_sampled_headroom_bytes": min(_int(sample["headroom_bytes"], "guard headroom") for sample in samples),
        "physical_bytes": _int(initial["physical_bytes"], "guard initial physical_bytes"),
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def analyze(report: dict[str, Any], guard_events: list[Any], *,
            report_sha256: str | None = None, guard_sha256: str | None = None) -> dict[str, Any]:
    """Validate a parsed benchmark and guard log, then summarize paired timings."""
    report = _mapping(report, "benchmark report")
    if (report.get("schema") != SCHEMA or type(report.get("schema_version")) is not int
            or report.get("schema_version") != SCHEMA_VERSION):
        raise ValidationError("benchmark schema/version mismatch")
    reference = _sha256(report.get("reference_sha256"), "reference_sha256")
    if reference != REFERENCE_SHA256:
        raise ValidationError("benchmark references a stale or mismatched upstream fixture")
    cache = _sha256(report.get("cache_index_sha256"), "cache_index_sha256")
    if cache != CACHE_INDEX_SHA256:
        raise ValidationError("benchmark references a mismatched Q4 cache index")
    if report.get("precision_mode") != "q4":
        raise ValidationError("benchmark precision mode must be q4")
    geometry = _validate_geometry(report.get("geometry"))
    tolerance = _mapping(report.get("tolerance"), "tolerance")
    if (_number(tolerance.get("atol"), "tolerance.atol") != ATOL
            or _number(tolerance.get("rtol"), "tolerance.rtol") != RTOL):
        raise ValidationError("benchmark must preserve unchanged atol=rtol=2e-5")
    _bool(report.get("diffusion_generation"), "diffusion_generation", expected=False)
    _bool(report.get("passed"), "passed", expected=True)

    named_bytes = _validate_representation(report, geometry)
    _validate_epochs(report)
    _validate_validations(report, geometry["valid_text_tokens"])
    samples = _validate_samples(report)
    guard = _validate_guard(guard_events)
    if named_bytes["padded"] + named_bytes["compact"] > guard["max_footprint_bytes"]:
        raise ValidationError("combined named residency exceeds the guarded process footprint cap")

    by_mode: dict[str, dict[str, float]] = {}
    for mode in MODES:
        selected = [row for row in samples if row["mode"] == mode]
        by_mode[mode] = {
            field: _median([float(row[field]) for row in selected])
            for field in ("encode_seconds", "terminal_wait_seconds", "total_seconds", "gpu_elapsed_seconds")
        }

    paired_ratios: list[float] = []
    epoch_summaries: list[dict[str, Any]] = []
    for epoch in (0, 1):
        ratios: list[float] = []
        for pair in range(3):
            pair_rows = [row for row in samples if row["epoch"] == epoch and row["pair"] == pair]
            durations = {row["mode"]: float(row["total_seconds"]) for row in pair_rows}
            ratio = durations["padded"] / durations["compact"]
            ratios.append(ratio)
            paired_ratios.append(ratio)
        epoch_summaries.append({
            "epoch": epoch,
            "paired_padded_over_compact_total_ratio_median": _median(ratios),
            "pair_count": len(ratios),
        })

    summary = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "passed": True,
        "report_sha256": report_sha256,
        "guard_sha256": guard_sha256,
        "fixture_sha256": reference,
        "cache_index_sha256": cache,
        "precision_mode": "q4",
        "geometry": geometry,
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "oracle_gate": "producer-declared per-element comparisons passed; this CPU analyzer validates receipts, not tensor bytes",
        "representations": {
            "padded": {"dit_text_tokens": 1280, "named_bytes": named_bytes["padded"]},
            "compact": {
                "dit_text_tokens": geometry["compact_text_tokens"],
                "valid_text_tokens": geometry["valid_text_tokens"],
                "named_bytes": named_bytes["compact"],
            },
            "named_bytes_saved": named_bytes["padded"] - named_bytes["compact"],
            "named_bytes_reduction_fraction": (
                named_bytes["padded"] - named_bytes["compact"]
            ) / named_bytes["padded"],
            "combined_named_bytes": named_bytes["padded"] + named_bytes["compact"],
            "declared_combined_ceiling_bytes": _int(
                report["max_combined_named_bytes"], "max_combined_named_bytes", minimum=1
            ),
        },
        "timing": {
            "median_seconds_by_mode": by_mode,
            "paired_padded_over_compact_total_ratio_median": _median(paired_ratios),
            "epoch_summaries": epoch_summaries,
            "measured_pairs": len(paired_ratios),
            "interpretation_scope": "descriptive timing on this fixed saved-input one-forward run",
        },
        "guard": guard,
        "guard_note": "sampled file-cache headroom is a reclaimability estimate, as recorded by the guard",
        "checks": {
            "validations": 4,
            "warmups": 4,
            "measured_samples": len(samples),
            "one_terminal_wait_per_request": True,
            "no_encode_allocations_or_preterminal_readbacks": True,
            "compact_suffix_poison_control_passed": True,
            "repeated_output_hashes_stable": True,
            "diffusion_generation": False,
        },
    }
    return summary


def analyze_files(report_path: Path, guard_path: Path) -> dict[str, Any]:
    report_bytes = report_path.read_bytes()
    guard_bytes = guard_path.read_bytes()
    report = _strict_json(report_bytes, "benchmark report")
    guard_events: list[Any] = []
    for line_number, line in enumerate(guard_bytes.splitlines(), start=1):
        if not line.strip():
            continue
        guard_events.append(_strict_json(line, f"guard log line {line_number}"))
    return analyze(
        report, guard_events,
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        guard_sha256=hashlib.sha256(guard_bytes).hexdigest(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="paired text-compaction benchmark JSON")
    parser.add_argument("--guard", required=True, type=Path, help="external memory-guard JSONL log")
    args = parser.parse_args(argv)
    try:
        summary = analyze_files(args.report, args.guard)
    except (OSError, ValidationError) as error:
        print(f"analysis rejected: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
