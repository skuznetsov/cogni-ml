#!/usr/bin/env python3
"""Build a same-clock Gemma 4 campaign report from raw captures and VM exports.

The input reports provide request windows and token-progress observations. GPU
and UT61B JSONL captures are written to loopback VictoriaMetrics, exported back
as raw samples, and compared exactly before any measurements are published.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import ipaddress
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gemma4_layer_report as layer_report  # noqa: E402
import prometheus_remote_write as remote_write  # noqa: E402


SCHEMA = "cogni.gemma4.layer_depth.v1"
TOKEN_METRIC = "cogni_llm_generated_tokens_total"
GPU_METRICS = {
    "gpu_power": ("cogni_gpu_power_watts", "power_watts"),
    "gpu_utilization": ("cogni_gpu_utilization_percent", "utilization_percent"),
    "gpu_memory": ("cogni_gpu_memory_used_bytes", "memory_used_bytes"),
}
UT_METRICS = {
    "ut61b_valid": "ut61b_measurement_valid",
    "ut61b_current": "ut61b_current_amperes",
    "ut61b_apparent": "ut61b_apparent_power_va",
}
MAX_GAP_MS = 2000
VALUE_TOLERANCE = 1e-12
VM_EXPORT_ATTEMPTS = 6
VM_EXPORT_TIMEOUT_S = 10.0
MAX_EXPORT_BYTES = 64 * 1024 * 1024
MAX_CAPTURE_LINES = 500_000
MAX_CAPTURE_LINE_BYTES = 65_536


class ReportError(ValueError):
    """Raised when campaign evidence is incomplete, ambiguous, or inconsistent."""


@dataclass(frozen=True)
class CaptureRange:
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Integration:
    area_value_seconds: float
    curve: tuple[dict[str, float | int], ...]
    sample_count: int


def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{location} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ReportError(f"{location} must be finite")
    return number


def _timestamp(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 2**63 - 1:
        raise ReportError(f"{location} must be a nonnegative int64 epoch-millisecond timestamp")
    return value


def _capture_rows(source: str | Iterable[str]) -> list[dict[str, Any]]:
    lines = source.splitlines() if isinstance(source, str) else source
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if len(line.encode("utf-8")) > MAX_CAPTURE_LINE_BYTES:
            raise ReportError(f"line {line_number}: JSONL row exceeds {MAX_CAPTURE_LINE_BYTES} bytes")
        if not line.strip():
            continue
        if len(rows) >= MAX_CAPTURE_LINES:
            raise ReportError(f"capture exceeds the {MAX_CAPTURE_LINES} row safety bound")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReportError(f"line {line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ReportError(f"line {line_number}: each JSONL row must be an object")
        rows.append(row)
    return rows


def _canonical_series(values: Mapping[tuple[tuple[str, str], ...], Mapping[int, float]]) -> tuple[remote_write.TimeSeries, ...]:
    result = []
    for labels, timestamp_values in sorted(values.items()):
        samples = tuple(
            remote_write.Sample(timestamp, value)
            for timestamp, value in sorted(timestamp_values.items())
        )
        if samples:
            result.append(remote_write.TimeSeries(labels, samples))
    return tuple(result)


def _add_observation(
    values: dict[tuple[tuple[str, str], ...], dict[int, float]],
    labels: tuple[tuple[str, str], ...],
    timestamp: int,
    value: float,
    *,
    source: str,
) -> None:
    series = values.setdefault(labels, {})
    previous = series.get(timestamp)
    if previous is not None and previous != value:
        raise ReportError(
            f"{source}: conflicting duplicate sample for labels={dict(labels)!r} at {timestamp} ms"
        )
    series[timestamp] = value


def _metric_labels(metric: str, **labels: str) -> tuple[tuple[str, str], ...]:
    return tuple(sorted({"__name__": metric, **labels}.items()))


def parse_ut_capture(source: str | Iterable[str]) -> tuple[tuple[remote_write.TimeSeries, ...], int]:
    """Parse raw UT61B JSONL, rejecting ambiguous same-ms duplicates first."""
    lines = source.splitlines() if isinstance(source, str) else list(source)
    rows = _capture_rows(lines)
    if not rows:
        raise ReportError("UT61B capture contains no rows")
    seen: dict[tuple[tuple[str, str], ...], dict[int, float]] = {}
    for line_number, row in enumerate(rows, 1):
        if row.get("schema_version") != 1 or row.get("source") != "ut61b":
            raise ReportError(f"line {line_number}: expected schema_version=1 and source='ut61b'")
        if row.get("circuit") != "gputer":
            raise ReportError(f"line {line_number}: UT61B circuit must be 'gputer'")
        if type(row.get("valid")) is not bool:
            raise ReportError(f"line {line_number}: valid must be a boolean")
        if row.get("mode") not in ("AC", "DC", "unknown"):
            raise ReportError(f"line {line_number}: mode must be AC, DC, or unknown")
        timestamp = _timestamp(row.get("receipt_epoch_ms"), f"line {line_number}.receipt_epoch_ms")
        voltage = _finite_number(row.get("voltage_assumption_v"), f"line {line_number}.voltage_assumption_v")
        if abs(voltage - 116.0) > VALUE_TOLERANCE:
            raise ReportError("UT61B capture must declare the required 116 V assumption")
        if row.get("voltage_provenance") != "assumed_default":
            raise ReportError("UT61B 116 V assumption must retain voltage_provenance='assumed_default'")
        parsed, _ = remote_write.parse_input_jsonl(json.dumps(row, separators=(",", ":")))
        for series in parsed:
            for sample in series.samples:
                _add_observation(
                    seen,
                    tuple(sorted(series.labels)),
                    sample.timestamp_ms,
                    float(sample.value),
                    source=f"UT61B line {line_number}",
                )
    try:
        parsed, coalesced = remote_write.parse_input_jsonl(lines)
    except remote_write.RemoteWriteError as exc:
        raise ReportError(f"UT61B capture: {exc}") from exc
    # The helper's last-write-wins behavior is safe only after conflicts have
    # already been rejected above. Keep its accounting for identical repeats.
    return parsed, coalesced


def parse_gpu_capture(source: str | Iterable[str], *, campaign_id: str) -> tuple[tuple[remote_write.TimeSeries, ...], CaptureRange]:
    """Map raw nvidia-smi rows to the three campaign GPU metric series."""
    rows = _capture_rows(source)
    values: dict[tuple[tuple[str, str], ...], dict[int, float]] = {}
    timestamps: list[int] = []
    for line_number, row in enumerate(rows, 1):
        if row.get("schema") != "cogni.gemma4.gpu-telemetry.v1":
            raise ReportError(f"line {line_number}: unexpected GPU telemetry schema")
        if row.get("run") != campaign_id or row.get("host") != "gputer" or row.get("gpu") != "0":
            raise ReportError(f"line {line_number}: GPU row campaign/host/device does not match the campaign")
        if row.get("gpu_index") != 0:
            raise ReportError(f"line {line_number}: gpu_index must match gpu='0'")
        record_type = row.get("record_type")
        if record_type == "session":
            if row.get("event") == "error":
                raise ReportError(f"line {line_number}: GPU collector reported an error")
            continue
        if record_type != "sample":
            raise ReportError(f"line {line_number}: unknown GPU telemetry record_type")
        if row.get("sample_status") != "ok":
            raise ReportError(f"line {line_number}: GPU sample is not fully valid")
        timestamp = _timestamp(row.get("timestamp_unix_ms"), f"line {line_number}.timestamp_unix_ms")
        timestamps.append(timestamp)
        for metric, field in GPU_METRICS.values():
            value = _finite_number(row.get(field), f"line {line_number}.{field}")
            if value < 0:
                raise ReportError(f"line {line_number}.{field} must be nonnegative")
            if metric == "cogni_gpu_utilization_percent" and value > 100:
                raise ReportError(f"line {line_number}.utilization_percent exceeds 100")
            labels = _metric_labels(metric, host="gputer", gpu="0", run=campaign_id)
            _add_observation(values, labels, timestamp, value, source=f"GPU line {line_number}")
    if not timestamps:
        raise ReportError("GPU capture contains no usable samples")
    return _canonical_series(values), CaptureRange(min(timestamps), max(timestamps))


def token_series_from_reports(reports: Sequence[dict[str, Any]]) -> tuple[remote_write.TimeSeries, ...]:
    """Validate exact received-token progress and construct Crystal's RW series."""
    if not reports:
        raise ReportError("no batch reports were supplied")
    campaigns = {report.get("campaign_id") for report in reports}
    if len(campaigns) != 1 or not isinstance(next(iter(campaigns)), str) or not next(iter(campaigns)):
        raise ReportError("all input reports must have one nonempty campaign_id")
    campaign_id = next(iter(campaigns))
    result: list[remote_write.TimeSeries] = []
    seen_runs: set[tuple[str, int]] = set()
    for report_index, report in enumerate(reports, 1):
        if report.get("schema") != SCHEMA or report.get("requested_tokens") != 256:
            raise ReportError(f"input report {report_index} must use {SCHEMA} and request 256 tokens")
        runs = report.get("runs")
        if not isinstance(runs, list):
            raise ReportError(f"input report {report_index}.runs must be an array")
        for run_index, run in enumerate(runs, 1):
            where = f"report {report_index} run {run_index}"
            if not isinstance(run, dict):
                raise ReportError(f"{where} must be an object")
            label, layers, trial = run.get("label"), run.get("decoder_layers"), run.get("trial")
            if not isinstance(label, str) or not label:
                raise ReportError(f"{where}.label must be nonempty")
            if type(layers) is not int or layers < 1:
                raise ReportError(f"{where}.decoder_layers must be a positive integer")
            if type(trial) is not int or trial < 1:
                raise ReportError(f"{where}.trial must be a positive integer")
            identity = (label, trial)
            if identity in seen_runs:
                raise ReportError(f"duplicate campaign label/trial identity {identity!r}")
            seen_runs.add(identity)
            if run.get("status") != "completed":
                raise ReportError(f"{where}.status must be completed")
            if run.get("requested_tokens") != 256:
                raise ReportError(f"{where}.requested_tokens must be 256")
            if run.get("actual_tokens") != 256:
                raise ReportError(f"{where}.actual_tokens must be exactly 256")
            start = _timestamp(run.get("started_at_epoch_ms"), f"{where}.started_at_epoch_ms")
            end = _timestamp(run.get("ended_at_epoch_ms"), f"{where}.ended_at_epoch_ms")
            if end <= start:
                raise ReportError(f"{where} has a nonpositive request window")
            progress = run.get("token_progress")
            if not isinstance(progress, dict) or progress.get("status") != "available":
                raise ReportError(f"{where}.token_progress must be available")
            if progress.get("basis") != "received_token_ids":
                raise ReportError(f"{where}.token_progress basis must be received_token_ids")
            if any(progress.get(key) != 256 for key in ("requested_tokens", "final_tokens", "observed_tokens")):
                raise ReportError(f"{where}.token_progress counts must all equal 256")
            points = progress.get("points")
            if not isinstance(points, list) or len(points) < 2:
                raise ReportError(f"{where}.token_progress.points needs at least two samples")
            previous_time = -1
            previous_count = -1
            samples: list[remote_write.Sample] = []
            for point_index, point in enumerate(points, 1):
                if not isinstance(point, dict):
                    raise ReportError(f"{where}.token_progress point {point_index} must be an object")
                elapsed, count = point.get("elapsed_ms"), point.get("generated_tokens")
                if type(elapsed) is not int or elapsed < 0 or elapsed <= previous_time:
                    raise ReportError(f"{where}.token_progress elapsed_ms must increase strictly")
                if type(count) is not int or not previous_count <= count <= 256:
                    raise ReportError(f"{where}.token_progress generated_tokens must be monotone in [0,256]")
                samples.append(remote_write.Sample(start + elapsed, float(count)))
                previous_time, previous_count = elapsed, count
            if points[0].get("elapsed_ms") != 0 or points[0].get("generated_tokens") != 0:
                raise ReportError(f"{where}.token_progress must start at (0 ms, 0 tokens)")
            if previous_time != end - start or previous_count != 256:
                raise ReportError(f"{where}.token_progress must cover the request and finish at 256")
            remote = progress.get("remote_write")
            if not isinstance(remote, dict) or remote.get("status") != "available":
                raise ReportError(f"{where}.token_progress.remote_write must be available")
            if remote.get("metric") != TOKEN_METRIC:
                raise ReportError(f"{where}.token_progress.remote_write metric is not {TOKEN_METRIC}")
            if remote.get("sample_count") != len(samples):
                raise ReportError(f"{where}.token_progress.remote_write sample_count does not match points")
            labels = {
                "job": "cogni-gemma4",
                "host": "gputer",
                "campaign": campaign_id,
                "label": label,
                "layers": str(layers),
                "trial": str(trial),
                "phase": "answer",
            }
            selector = f"{TOKEN_METRIC}{{" + ",".join(
                f"{name}={json.dumps(value, ensure_ascii=False)}"
                for name, value in sorted(labels.items())
            ) + "}"
            if remote.get("selector") != selector:
                raise ReportError(f"{where}.token_progress.remote_write selector labels do not match the run")
            result.append(
                remote_write.TimeSeries(
                    _metric_labels(TOKEN_METRIC, **labels),
                    tuple(samples),
                )
            )
    return tuple(result)


def validate_campaign_matrix(report: dict[str, Any]) -> None:
    """Require the campaign's four depth batches and three trials each."""
    expected = {
        ("depth48-before", 48),
        ("depth32", 32),
        ("depth16", 16),
        ("depth48-after", 48),
    }
    actual: dict[tuple[str, int], int] = {}
    for run in report.get("runs", []):
        key = (run.get("label"), run.get("decoder_layers"))
        if key not in expected:
            raise ReportError(f"unexpected campaign label/depth pair {key!r}")
        identity = (run["label"], run["trial"])
        if identity in actual:
            raise ReportError(f"duplicate campaign label/trial identity {identity!r}")
        actual[identity] = run["decoder_layers"]
    expected_identities = {
        (label, trial)
        for label, _depth in expected
        for trial in (1, 2, 3)
    }
    if set(actual) != expected_identities:
        missing = sorted(expected_identities - set(actual))
        extra = sorted(set(actual) - expected_identities)
        raise ReportError(f"campaign must have exactly 3 trials per expected batch; missing={missing}, extra={extra}")
    if report.get("requested_tokens") != 256:
        raise ReportError("campaign root requested_tokens must equal 256")


def parse_vm_export(raw: str | bytes) -> dict[tuple[tuple[str, str], ...], dict[int, float]]:
    """Parse VM export JSONL, merging identical points and rejecting conflicts."""
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReportError("VictoriaMetrics export is not UTF-8 JSONL") from exc
    values: dict[tuple[tuple[str, str], ...], dict[int, float]] = {}
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReportError(f"VictoriaMetrics export line {line_number}: invalid JSON") from exc
        if not isinstance(row, dict) or not isinstance(row.get("metric"), dict):
            raise ReportError(f"VictoriaMetrics export line {line_number}: metric labels must be an object")
        labels_map = row["metric"]
        if any(not isinstance(name, str) or not isinstance(value, str) for name, value in labels_map.items()):
            raise ReportError(f"VictoriaMetrics export line {line_number}: labels must be strings")
        if "__name__" not in labels_map:
            raise ReportError(f"VictoriaMetrics export line {line_number}: missing __name__ label")
        labels = tuple(sorted(labels_map.items()))
        points, timestamps = row.get("values"), row.get("timestamps")
        if not isinstance(points, list) or not isinstance(timestamps, list) or len(points) != len(timestamps):
            raise ReportError(f"VictoriaMetrics export line {line_number}: values/timestamps mismatch")
        for point_index, (raw_value, raw_timestamp) in enumerate(zip(points, timestamps), 1):
            if isinstance(raw_timestamp, str):
                try:
                    parsed_timestamp = float(raw_timestamp)
                except ValueError as exc:
                    raise ReportError(f"VictoriaMetrics export line {line_number}: invalid timestamp") from exc
                if not math.isfinite(parsed_timestamp) or not parsed_timestamp.is_integer():
                    raise ReportError(f"VictoriaMetrics export line {line_number}: timestamp must be integral")
                timestamp = int(parsed_timestamp)
            elif isinstance(raw_timestamp, (int, float)) and not isinstance(raw_timestamp, bool):
                if not math.isfinite(float(raw_timestamp)) or not float(raw_timestamp).is_integer():
                    raise ReportError(f"VictoriaMetrics export line {line_number}: timestamp must be integral")
                timestamp = int(raw_timestamp)
            else:
                raise ReportError(f"VictoriaMetrics export line {line_number}: invalid timestamp")
            if timestamp < 0 or timestamp > 2**63 - 1:
                raise ReportError(f"VictoriaMetrics export line {line_number}: timestamp outside int64 range")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ReportError(f"VictoriaMetrics export line {line_number}: sample value is not numeric") from exc
            if isinstance(raw_value, bool) or not math.isfinite(value):
                raise ReportError(f"VictoriaMetrics export line {line_number}: sample value must be finite")
            _add_observation(
                values,
                labels,
                timestamp,
                value,
                source=f"VictoriaMetrics export line {line_number} point {point_index}",
            )
    return values


def compare_samples(
    expected_series: Sequence[remote_write.TimeSeries],
    actual_by_labels: Mapping[tuple[tuple[str, str], ...], Mapping[int, float]],
    capture_start_ms: int,
    capture_end_ms: int,
) -> None:
    """Verify all selected labels/timestamps/values and reject extra samples."""
    if capture_end_ms < capture_start_ms:
        raise ReportError("invalid capture range")
    expected: dict[tuple[tuple[str, str], ...], dict[int, float]] = {}
    for series in expected_series:
        labels = tuple(sorted(series.labels))
        values = expected.setdefault(labels, {})
        for sample in series.samples:
            if capture_start_ms <= sample.timestamp_ms <= capture_end_ms:
                old = values.get(sample.timestamp_ms)
                if old is not None and old != float(sample.value):
                    raise ReportError("expected samples contain a conflicting duplicate")
                values[sample.timestamp_ms] = float(sample.value)
    actual: dict[tuple[tuple[str, str], ...], dict[int, float]] = {
        tuple(sorted(labels)): {
            timestamp: float(value)
            for timestamp, value in samples.items()
            if capture_start_ms <= timestamp <= capture_end_ms
        }
        for labels, samples in actual_by_labels.items()
    }
    for labels, values in actual.items():
        if values and labels not in expected:
            raise ReportError(f"extra VM series within capture range: labels={dict(labels)!r}")
    for labels, expected_values in expected.items():
        observed = actual.get(labels, {})
        missing = sorted(set(expected_values) - set(observed))
        if missing:
            raise ReportError(f"missing VM sample(s) for {dict(labels)!r} at {missing[:5]}")
        extra = sorted(set(observed) - set(expected_values))
        if extra:
            raise ReportError(f"extra VM sample(s) for {dict(labels)!r} at {extra[:5]}")
        for timestamp, expected_value in expected_values.items():
            observed_value = observed[timestamp]
            if abs(observed_value - expected_value) > VALUE_TOLERANCE:
                raise ReportError(
                    f"value mismatch for {dict(labels)!r} at {timestamp}: "
                    f"expected {expected_value!r}, received {observed_value!r}"
                )


def integrate_window(
    samples: Sequence[tuple[int, float]],
    start_ms: int,
    end_ms: int,
    max_gap_ms: int = MAX_GAP_MS,
) -> Integration:
    """Integrate a raw signal with linear interpolation and strict bracketing."""
    if type(start_ms) is not int or type(end_ms) is not int or end_ms <= start_ms:
        raise ReportError("coverage: measurement window must have positive duration")
    if type(max_gap_ms) is not int or max_gap_ms <= 0:
        raise ReportError("max_gap_ms must be a positive integer")
    normalized: list[tuple[int, float]] = []
    previous = -1
    for timestamp, value in samples:
        timestamp = _timestamp(timestamp, "sample timestamp")
        number = _finite_number(value, "sample value")
        if timestamp <= previous:
            raise ReportError("sample timestamps must be strictly increasing")
        normalized.append((timestamp, number))
        previous = timestamp
    if len(normalized) < 2 or normalized[0][0] > start_ms or normalized[-1][0] < end_ms:
        raise ReportError("coverage: samples must bracket both request-window boundaries")
    area = 0.0
    raw_points: set[int] = set()
    curve_values: dict[int, float] = {}
    for (left_ms, left_value), (right_ms, right_value) in zip(normalized, normalized[1:]):
        if right_ms <= start_ms or left_ms >= end_ms:
            continue
        gap = right_ms - left_ms
        if gap > max_gap_ms:
            raise ReportError(f"gap: raw sample gap {gap} ms exceeds {max_gap_ms} ms")
        lo, hi = max(start_ms, left_ms), min(end_ms, right_ms)
        if hi <= lo:
            continue
        span = right_ms - left_ms
        low_value = left_value + (right_value - left_value) * (lo - left_ms) / span
        high_value = left_value + (right_value - left_value) * (hi - left_ms) / span
        area += (low_value + high_value) * 0.5 * (hi - lo) / 1000.0
        raw_points.update((left_ms, right_ms))
        curve_values[lo - start_ms] = low_value
        curve_values[hi - start_ms] = high_value
    if 0 not in curve_values or end_ms - start_ms not in curve_values:
        raise ReportError("coverage: no sample segments span the full request window")
    curve = tuple(
        {"elapsed_ms": elapsed, "value": value}
        for elapsed, value in sorted(curve_values.items())
    )
    if not math.isfinite(area):
        raise ReportError("integral overflow")
    return Integration(area, curve, len(raw_points))


def validity_events(series: Sequence[remote_write.TimeSeries]) -> tuple[tuple[int, float], ...]:
    """Merge all mode validity series into one held valid-AC state signal."""
    events: dict[int, float] = {}
    found = False
    for item in series:
        labels = dict(item.labels)
        if labels.get("__name__") != UT_METRICS["ut61b_valid"]:
            continue
        found = True
        mode = labels.get("mode")
        if mode not in ("ac", "dc", "unknown") or labels.get("circuit") != "gputer" or labels.get("meter") != "ut61b":
            raise ReportError("UT61B validity series has unexpected labels")
        for sample in item.samples:
            if sample.value not in (0.0, 1.0):
                raise ReportError("UT61B validity values must be exactly 0 or 1")
            state = 1.0 if mode == "ac" and sample.value == 1.0 else 0.0
            old = events.get(sample.timestamp_ms)
            if old is not None and old != state:
                raise ReportError(f"conflicting held validity states at {sample.timestamp_ms} ms")
            events[sample.timestamp_ms] = state
    if not found:
        raise ReportError("UT61B capture has no validity series")
    return tuple(sorted(events.items()))


def require_valid_held(
    events: Sequence[tuple[int, float]],
    start_ms: int,
    end_ms: int,
    max_gap_ms: int = MAX_GAP_MS,
) -> None:
    """Require a held valid-AC event throughout the window, including boundaries."""
    if end_ms <= start_ms or max_gap_ms <= 0:
        raise ReportError("invalid validity-check window")
    normalized = sorted(events)
    if len(normalized) < 2 or normalized[0][0] > start_ms or normalized[-1][0] < end_ms:
        raise ReportError("coverage: held UT61B validity must bracket the request window")
    prior = [value for timestamp, value in normalized if timestamp <= start_ms]
    if not prior or prior[-1] != 1.0:
        raise ReportError("invalid: UT61B held validity is not valid AC at request start")
    covered = False
    for (left_ms, left_state), (right_ms, _right_state) in zip(normalized, normalized[1:]):
        if right_ms <= start_ms or left_ms >= end_ms:
            continue
        covered = True
        if right_ms - left_ms > max_gap_ms:
            raise ReportError(f"gap: UT61B held validity gap exceeds {max_gap_ms} ms")
        if left_state != 1.0:
            raise ReportError("invalid: a non-AC or rejected frame was held during the request window")
    at_end = [value for timestamp, value in normalized if timestamp <= end_ms]
    if not covered or not at_end or at_end[-1] != 1.0:
        raise ReportError("invalid: UT61B held validity is not valid AC at request end")


def _selector(metric: str, labels: Mapping[str, str]) -> str:
    return metric + "{" + ",".join(
        f"{name}={json.dumps(value, ensure_ascii=False)}" for name, value in sorted(labels.items())
    ) + "}"


def _measurement_identity() -> dict[str, Any]:
    labels = {"circuit": "gputer", "meter": "ut61b", "mode": "ac"}
    return {
        "meter": "ut61b",
        "selector": _selector("ut61b_current_amperes", labels),
        "metric": {"__name__": "ut61b_current_amperes", **labels},
        "kind": "current_ac",
        "power_unit": "VA",
        "unit_contract": "ut61b_current_amperes with mode=ac; RMS voltage assumed 116 V (provenance=assumed_default)",
        "assumed_voltage_volts": 116.0,
        "assumed_power_factor": None,
    }


def _unavailable_measurement(reason: str, *, meter: str | None = None) -> dict[str, Any]:
    identity = _measurement_identity() if meter == "ut61b" else {}
    return {
        "status": "unavailable",
        **identity,
        "reason": reason,
        "mean_active_power_watts": None,
        "active_energy_joules": None,
        "mean_apparent_power_va": None,
        "apparent_energy_va_seconds": None,
        "estimated_active_energy_joules": None,
        "curve": [],
    }


def install_campaign_metadata(
    report: dict[str, Any],
    *,
    source_reports: Sequence[str],
    evidence_exports: Mapping[str, str],
    remote_write_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Set same-clock metadata and replace any older circuit source identity."""
    output = deepcopy(report)
    output["schema"] = SCHEMA
    output["circuit_meter"] = "ut61b"
    output["measurement_clock"] = "gputer"
    output["source_reports"] = list(source_reports)
    output["evidence_exports"] = dict(evidence_exports)
    output["capture_clock"] = "UTC epoch milliseconds from gputer system clock"
    output["notes"] = (
        "GPU board power is measured in W/J and UT61B circuit current is reported as apparent VA/VA·s "
        "using an assumed RMS voltage of 116 V (assumed_default). Both collector timestamps use the "
        "gputer UTC epoch clock. No active circuit power is inferred from apparent power; request "
        "windows are integrated from bracketing raw samples with a 2000 ms maximum gap."
    )
    if remote_write_summary is not None:
        output["remote_write"] = deepcopy(dict(remote_write_summary))
    for run in output.get("runs", []):
        run["circuit_measurement"] = _unavailable_measurement(
            "UT61B measurement not yet computed", meter="ut61b"
        )
    return output


def _available_power_measurement(
    metric: str,
    labels: Mapping[str, str],
    start_ms: int,
    end_ms: int,
    integral: Integration,
    *,
    unit: str,
    kind: str,
    voltage: float | None = None,
    meter: str | None = None,
) -> dict[str, Any]:
    duration = (end_ms - start_ms) / 1000.0
    scaled_curve = [
        {"elapsed_ms": point["elapsed_ms"], "value": float(point["value"]) * (voltage or 1.0)}
        for point in integral.curve
    ]
    integrated = integral.area_value_seconds * (voltage or 1.0)
    result: dict[str, Any] = {
        "status": "available",
        "kind": kind,
        "power_unit": unit,
        "metric": {"__name__": metric, **labels},
        "selector": _selector(metric, labels),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_seconds": duration,
        "sample_count": integral.sample_count,
        "max_gap_ms": MAX_GAP_MS,
        "integration": "linear interpolation of bracketing raw samples",
        "curve": scaled_curve,
    }
    if meter is not None:
        result.update(_measurement_identity())
        # Preserve the exact metric identity/selector selected for this object.
        result["metric"] = {"__name__": metric, **labels}
        result["selector"] = _selector(metric, labels)
        result["meter"] = meter
        result["kind"] = "current_ac"
        result["power_unit"] = "VA"
        result["unit_contract"] = (
            "ut61b_current_amperes with mode=ac; RMS voltage assumed 116 V "
            "(provenance=assumed_default)"
        )
        result["assumed_voltage_volts"] = float(voltage or 116.0)
        result["mean_apparent_power_va"] = integrated / duration
        result["apparent_energy_va_seconds"] = integrated
        result["mean_active_power_watts"] = None
        result["active_energy_joules"] = None
        result["estimated_active_energy_joules"] = None
        result["assumed_power_factor"] = None
    else:
        result["mean_active_power_watts"] = integrated / duration
        result["active_energy_joules"] = integrated
        result["mean_apparent_power_va"] = None
        result["apparent_energy_va_seconds"] = None
        result["estimated_active_energy_joules"] = None
        result["assumed_voltage_volts"] = None
        result["assumed_power_factor"] = None
    return result


def _series_map(series: Sequence[remote_write.TimeSeries]) -> dict[tuple[tuple[str, str], ...], remote_write.TimeSeries]:
    return {tuple(sorted(item.labels)): item for item in series}


def _one_series(
    series: Sequence[remote_write.TimeSeries],
    *,
    metric: str,
    required_labels: Mapping[str, str],
    allow_missing: bool = False,
) -> remote_write.TimeSeries | None:
    matches = []
    for item in series:
        labels = dict(item.labels)
        if labels.get("__name__") == metric and all(labels.get(key) == value for key, value in required_labels.items()):
            matches.append(item)
    if len(matches) > 1:
        raise ReportError(f"expected one {metric} series for {required_labels}, found {len(matches)}")
    if not matches and not allow_missing:
        raise ReportError(f"missing {metric} series for {required_labels}")
    return matches[0] if matches else None


def _samples(item: remote_write.TimeSeries | None) -> list[tuple[int, float]]:
    if item is None:
        return []
    return [(sample.timestamp_ms, float(sample.value)) for sample in item.samples]


def _utilization_measurement(
    series: remote_write.TimeSeries | None,
    start_ms: int,
    end_ms: int,
) -> dict[str, Any]:
    if series is None:
        return {"status": "unavailable", "reason": "GPU utilization series is absent", "curve": []}
    values = [(timestamp, value) for timestamp, value in _samples(series) if start_ms <= timestamp <= end_ms]
    if not values:
        return {"status": "unavailable", "reason": "no GPU utilization samples within request window", "curve": []}
    if any(value < 0 or value > 100 for _, value in values):
        return {"status": "unavailable", "reason": "GPU utilization is outside [0,100] percent", "curve": []}
    return {
        "status": "available",
        "metric": dict(series.labels),
        "selector": _selector("cogni_gpu_utilization_percent", {k: v for k, v in dict(series.labels).items() if k != "__name__"}),
        "method": "arithmetic mean of samples within window",
        "source": "whole-board utilization sample",
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_seconds": (end_ms - start_ms) / 1000.0,
        "sample_count": len(values),
        "mean_percent": sum(value for _, value in values) / len(values),
        "curve": [{"elapsed_ms": timestamp - start_ms, "value": value} for timestamp, value in values],
    }


def _memory_measurement(
    series: remote_write.TimeSeries | None,
    start_ms: int,
    end_ms: int,
) -> dict[str, Any]:
    if series is None:
        return {"status": "unavailable", "reason": "GPU memory series is absent", "curve": []}
    values = [(timestamp, value) for timestamp, value in _samples(series) if start_ms <= timestamp <= end_ms]
    if not values:
        return {"status": "unavailable", "reason": "no GPU memory samples within request window", "curve": []}
    return {
        "status": "available",
        "metric": dict(series.labels),
        "selector": _selector("cogni_gpu_memory_used_bytes", {k: v for k, v in dict(series.labels).items() if k != "__name__"}),
        "method": "sample summary within request window",
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_seconds": (end_ms - start_ms) / 1000.0,
        "sample_count": len(values),
        "mean_bytes": sum(value for _, value in values) / len(values),
        "max_bytes": max(value for _, value in values),
        "curve": [{"elapsed_ms": timestamp - start_ms, "value": value} for timestamp, value in values],
    }


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _vm_base_url(value: str) -> tuple[str, str]:
    try:
        parsed = urllib.parse.urlsplit(value)
        host = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ReportError("--vm-url must be a valid loopback HTTP URL") from exc
    if parsed.scheme != "http" or not host or parsed.username is not None or parsed.password is not None:
        raise ReportError("--vm-url must be plain HTTP on loopback without credentials")
    if parsed.query or parsed.fragment:
        raise ReportError("--vm-url must not include query or fragment")
    try:
        loopback = host.casefold() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback or parsed.path not in ("", "/"):
        raise ReportError("--vm-url must identify a loopback VM base URL with no path")
    base = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    return base, base + "/api/v1/export"


def _fetch_export(
    *,
    opener,
    endpoint: str,
    selector: str,
    capture_range: CaptureRange,
    expected_series: Sequence[remote_write.TimeSeries],
    destination: Path,
    max_gap_ms: int = MAX_GAP_MS,
) -> tuple[dict[tuple[tuple[str, str], ...], dict[int, float]], int]:
    query = urllib.parse.urlencode(
        [
            ("match[]", selector),
            ("start", format((capture_range.start_ms - max_gap_ms) / 1000.0, ".3f")),
            ("end", format((capture_range.end_ms + max_gap_ms) / 1000.0, ".3f")),
        ]
    )
    url = endpoint + "?" + query
    last_body = b""
    last_missing: ReportError | None = None
    opener_instance = opener
    for attempt in range(1, VM_EXPORT_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
            with opener_instance(request, timeout=VM_EXPORT_TIMEOUT_S) as response:
                if response.getcode() != 200:
                    status = response.getcode()
                    if status not in (429,) and not 500 <= status <= 599:
                        raise ReportError(f"VictoriaMetrics export failed with HTTP {status}")
                    raise urllib.error.HTTPError(url, status, "retryable export response", response.headers, None)
                body = response.read(MAX_EXPORT_BYTES + 1)
                if len(body) > MAX_EXPORT_BYTES:
                    raise ReportError("VictoriaMetrics export exceeds the 64 MiB response bound")
                last_body = body
            try:
                actual = parse_vm_export(last_body)
                compare_samples(expected_series, actual, capture_range.start_ms, capture_range.end_ms)
            except ReportError as exc:
                if not str(exc).startswith("missing VM sample"):
                    raise
                last_missing = exc
                if attempt < VM_EXPORT_ATTEMPTS:
                    time.sleep(min(0.5 * (2 ** (attempt - 1)), 8.0))
                    continue
                raise
            _write_new_bytes(destination, last_body)
            return actual, attempt - 1
        except urllib.error.HTTPError as exc:
            exc.close()
            if exc.code not in (429,) and not 500 <= exc.code <= 599:
                raise ReportError(f"VictoriaMetrics export failed with HTTP {exc.code}") from None
            if attempt == VM_EXPORT_ATTEMPTS:
                raise ReportError(f"VictoriaMetrics export failed after {attempt} attempts with HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == VM_EXPORT_ATTEMPTS:
                raise ReportError(f"VictoriaMetrics export network failure after {attempt} attempts: {exc}") from None
        if attempt < VM_EXPORT_ATTEMPTS:
            time.sleep(min(0.5 * (2 ** (attempt - 1)), 8.0))
    if last_missing is not None:
        raise last_missing
    raise ReportError("VictoriaMetrics export did not return usable samples")


def _write_new_bytes(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as output:
            output.write(content)
    except FileExistsError as exc:
        raise ReportError(f"refusing to overwrite evidence file {path}") from exc


def _export_paths(prefix: Path) -> dict[str, Path]:
    names = {
        "gpu_power": "gpu_power",
        "gpu_utilization": "gpu_utilization",
        "gpu_memory": "gpu_memory",
        "ut61b_valid": "ut61b_valid",
        "ut61b_current": "ut61b_current",
        "ut61b_apparent": "ut61b_apparent",
        "tokens": "tokens",
    }
    return {key: Path(f"{prefix}.{suffix}.raw.jsonl") for key, suffix in names.items()}


def _new_output_preflight(output: Path, exports: Mapping[str, Path]) -> None:
    paths = [output, *exports.values()]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ReportError("output and evidence export paths must be distinct")
    for path in paths:
        if path.exists():
            raise ReportError(f"refusing to overwrite existing output {path}")
        if not path.parent.is_dir():
            raise ReportError(f"output directory does not exist: {path.parent}")


def _load_and_merge(inputs: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inputs:
        raise ReportError("--input requires at least one report")
    try:
        reports = [layer_report._load_report(path) for path in inputs]
        merged = layer_report._merge_reports(list(inputs))
    except (layer_report.ReportError, OSError) as exc:
        raise ReportError(str(exc)) from exc
    validate_campaign_matrix(merged)
    # This explicit call forces token checks even if report validation later
    # becomes permissive about optional progress metadata.
    token_series_from_reports(reports)
    return reports, merged


def _range_for_series(series: Sequence[remote_write.TimeSeries]) -> CaptureRange:
    times = [sample.timestamp_ms for item in series for sample in item.samples]
    if not times:
        raise ReportError("cannot export an empty capture range")
    return CaptureRange(min(times), max(times))


def _range_for_tokens(series: Sequence[remote_write.TimeSeries]) -> CaptureRange:
    return _range_for_series(series)


def _fetch_and_verify_exports(
    *,
    vm_url: str,
    expected: Mapping[str, Sequence[remote_write.TimeSeries]],
    ranges: Mapping[str, CaptureRange],
    paths: Mapping[str, Path],
) -> tuple[dict[str, dict[tuple[tuple[str, str], ...], dict[int, float]]], dict[str, int]]:
    _base, endpoint = _vm_base_url(vm_url)
    opener = urllib.request.build_opener(_NoRedirectHandler()).open
    actuals: dict[str, dict[tuple[tuple[str, str], ...], dict[int, float]]] = {}
    retries: dict[str, int] = {}
    campaign = dict(expected["tokens"][0].labels)["campaign"]
    selectors = {
        "gpu_power": _selector("cogni_gpu_power_watts", {"host": "gputer", "gpu": "0", "run": campaign}),
        "gpu_utilization": _selector("cogni_gpu_utilization_percent", {"host": "gputer", "gpu": "0", "run": campaign}),
        "gpu_memory": _selector("cogni_gpu_memory_used_bytes", {"host": "gputer", "gpu": "0", "run": campaign}),
        "ut61b_valid": _selector("ut61b_measurement_valid", {"circuit": "gputer", "meter": "ut61b"}),
        "ut61b_current": _selector("ut61b_current_amperes", {"circuit": "gputer", "meter": "ut61b"}),
        "ut61b_apparent": _selector("ut61b_apparent_power_va", {"circuit": "gputer", "meter": "ut61b"}),
        "tokens": _selector(TOKEN_METRIC, {"campaign": campaign}),
    }
    for key, selector in selectors.items():
        actual, retried = _fetch_export(
            opener=opener,
            endpoint=endpoint,
            selector=selector,
            capture_range=ranges[key],
            expected_series=expected[key],
            destination=paths[key],
        )
        actuals[key] = actual
        retries[key] = retried
    return actuals, retries


def _actual_metric_series(
    actual: Mapping[tuple[tuple[str, str], ...], Mapping[int, float]], metric: str
) -> tuple[remote_write.TimeSeries, ...]:
    result = []
    for labels, samples in sorted(actual.items()):
        if dict(labels).get("__name__") != metric:
            continue
        result.append(
            remote_write.TimeSeries(
                labels,
                tuple(remote_write.Sample(timestamp, value) for timestamp, value in sorted(samples.items())),
            )
        )
    return tuple(result)


def _populate_run_measurements(
    report: dict[str, Any],
    *,
    exported: Mapping[str, Mapping[tuple[tuple[str, str], ...], Mapping[int, float]]],
) -> list[str]:
    unavailable: list[str] = []
    gpu_power_series = _actual_metric_series(exported["gpu_power"], "cogni_gpu_power_watts")
    gpu_util_series = _actual_metric_series(exported["gpu_utilization"], "cogni_gpu_utilization_percent")
    gpu_memory_series = _actual_metric_series(exported["gpu_memory"], "cogni_gpu_memory_used_bytes")
    ut_current_series = _actual_metric_series(exported["ut61b_current"], "ut61b_current_amperes")
    ut_valid_series = _actual_metric_series(exported["ut61b_valid"], "ut61b_measurement_valid")
    validity = validity_events(ut_valid_series)
    for run in report["runs"]:
        label, trial = run["label"], run["trial"]
        start, end = run["started_at_epoch_ms"], run["ended_at_epoch_ms"]
        run_key = f"{label}/trial{trial}"
        gpu_labels = {"host": "gputer", "gpu": "0", "run": report["campaign_id"]}
        power = _one_series(gpu_power_series, metric="cogni_gpu_power_watts", required_labels=gpu_labels)
        utilization = _one_series(gpu_util_series, metric="cogni_gpu_utilization_percent", required_labels=gpu_labels)
        memory = _one_series(gpu_memory_series, metric="cogni_gpu_memory_used_bytes", required_labels=gpu_labels)
        try:
            if power is None:
                raise ReportError("GPU power series unavailable")
            integral = integrate_window(_samples(power), start, end)
            run["gpu_measurement"] = _available_power_measurement(
                "cogni_gpu_power_watts", gpu_labels, start, end, integral,
                unit="W", kind="active_power",
            )
        except ReportError as exc:
            run["gpu_measurement"] = _unavailable_measurement(str(exc))
            unavailable.append(f"{run_key}: GPU power: {exc}")
        run["gpu_utilization_measurement"] = _utilization_measurement(utilization, start, end)
        if run["gpu_utilization_measurement"]["status"] != "available":
            unavailable.append(f"{run_key}: GPU utilization: {run['gpu_utilization_measurement'].get('reason')}")
        run["gpu_memory_measurement"] = _memory_measurement(memory, start, end)
        if run["gpu_memory_measurement"]["status"] != "available":
            unavailable.append(f"{run_key}: GPU memory: {run['gpu_memory_measurement'].get('reason')}")
        try:
            require_valid_held(validity, start, end)
            current = _one_series(
                ut_current_series,
                metric="ut61b_current_amperes",
                required_labels={"circuit": "gputer", "meter": "ut61b", "mode": "ac"},
                allow_missing=True,
            )
            if current is None:
                raise ReportError("UT61B AC current series unavailable")
            samples = _samples(current)
            if any(value < 0 for timestamp, value in samples if start <= timestamp <= end):
                raise ReportError("UT61B AC current contains a negative sample")
            integral = integrate_window(samples, start, end)
            run["circuit_measurement"] = _available_power_measurement(
                "ut61b_current_amperes",
                {"circuit": "gputer", "meter": "ut61b", "mode": "ac"},
                start,
                end,
                integral,
                unit="VA",
                kind="current_ac",
                voltage=116.0,
                meter="ut61b",
            )
        except ReportError as exc:
            run["circuit_measurement"] = _unavailable_measurement(str(exc), meter="ut61b")
            unavailable.append(f"{run_key}: UT61B: {exc}")
    try:
        layer_report.validate_report(report)
    except layer_report.ReportError as exc:
        raise ReportError(f"combined report validation failed: {exc}") from exc
    return unavailable


def _summary_for_write(stats: remote_write.WriteStats, series: Sequence[remote_write.TimeSeries]) -> dict[str, Any]:
    return {
        "status": "available",
        "write_attempted": True,
        "metric_families": sorted({dict(item.labels)["__name__"] for item in series}),
        "sample_count": stats.sample_count,
        "series_count": stats.series_count,
        "batches": stats.batches,
        "attempts": stats.attempts,
        "retry_count": stats.retry_count,
        "failures": stats.failures,
    }


def _summary_for_existing(series_by_family: Mapping[str, Sequence[remote_write.TimeSeries]]) -> dict[str, Any]:
    """Describe verification-only mode without fabricating remote-write stats."""
    gpu = [item for name in ("gpu_power", "gpu_utilization", "gpu_memory") for item in series_by_family[name]]
    ut61b = [item for name in ("ut61b_valid", "ut61b_current", "ut61b_apparent") for item in series_by_family[name]]
    tokens = list(series_by_family["tokens"])
    all_series = [*gpu, *ut61b, *tokens]
    return {
        "status": "verified_existing",
        "write_attempted": False,
        "verification_mode": "existing VM samples exported and compared; no remote-write call made",
        "metric_families": sorted({dict(item.labels)["__name__"] for item in all_series}),
        "expected_series_count": len(all_series),
        "expected_sample_count": sum(len(item.samples) for item in all_series),
        "gpu": {
            "expected_series_count": len(gpu),
            "expected_sample_count": sum(len(item.samples) for item in gpu),
        },
        "ut61b": {
            "expected_series_count": len(ut61b),
            "expected_sample_count": sum(len(item.samples) for item in ut61b),
        },
        "tokens": {
            "expected_series_count": len(tokens),
            "expected_sample_count": sum(len(item.samples) for item in tokens),
            "writer": "Crystal token-progress producer; this command did not write tokens",
        },
    }


def build_report(
    *,
    inputs: Sequence[Path],
    gpu_jsonl: Path,
    ut61b_jsonl: Path,
    vm_url: str,
    output: Path,
    evidence_prefix: Path,
    verify_existing: bool = False,
) -> tuple[dict[str, Any], int]:
    """Execute parse → write → export → compare → summarize for one campaign."""
    exports = _export_paths(evidence_prefix)
    _new_output_preflight(output, exports)
    reports, merged = _load_and_merge(inputs)
    token_series = token_series_from_reports(reports)
    campaign = merged["campaign_id"]
    try:
        gpu_series, gpu_range = parse_gpu_capture(gpu_jsonl.read_text(encoding="utf-8"), campaign_id=campaign)
        ut_text = ut61b_jsonl.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportError(f"could not read capture: {exc}") from exc
    ut_series, ut_coalesced = parse_ut_capture(ut_text)
    expected = {
        "gpu_power": tuple(item for item in gpu_series if dict(item.labels).get("__name__") == GPU_METRICS["gpu_power"][0]),
        "gpu_utilization": tuple(item for item in gpu_series if dict(item.labels).get("__name__") == GPU_METRICS["gpu_utilization"][0]),
        "gpu_memory": tuple(item for item in gpu_series if dict(item.labels).get("__name__") == GPU_METRICS["gpu_memory"][0]),
        "ut61b_valid": tuple(item for item in ut_series if dict(item.labels).get("__name__") == UT_METRICS["ut61b_valid"]),
        "ut61b_current": tuple(item for item in ut_series if dict(item.labels).get("__name__") == UT_METRICS["ut61b_current"]),
        "ut61b_apparent": tuple(item for item in ut_series if dict(item.labels).get("__name__") == UT_METRICS["ut61b_apparent"]),
        "tokens": token_series,
    }
    ut_range = _range_for_series(ut_series)
    token_range = _range_for_tokens(token_series)
    ranges = {
        key: gpu_range if key.startswith("gpu_") else ut_range if key.startswith("ut61b_") else token_range
        for key in expected
    }
    all_new_series = [*gpu_series, *ut_series]
    if verify_existing:
        write_summary = _summary_for_existing(expected)
        write_summary["ut61b"]["identical_duplicate_rows_coalesced"] = ut_coalesced
    else:
        try:
            writer = remote_write.RemoteWriteClient(vm_url)
            stats = writer.write(all_new_series)
        except remote_write.RemoteWriteError as exc:
            raise ReportError(f"remote write failed: {exc}") from exc
        write_summary = _summary_for_write(stats, all_new_series)
        write_summary["gpu"] = {
            "series_count": sum(1 for item in gpu_series),
            "sample_count": sum(len(item.samples) for item in gpu_series),
        }
        write_summary["ut61b"] = {
            "series_count": sum(1 for item in ut_series),
            "sample_count": sum(len(item.samples) for item in ut_series),
            "identical_duplicate_rows_coalesced": ut_coalesced,
        }
        write_summary["tokens"] = {
            "series_count": len(token_series),
            "sample_count": sum(len(item.samples) for item in token_series),
            "writer": "Crystal token-progress producer",
        }
    exported, retries = _fetch_and_verify_exports(
        vm_url=vm_url,
        expected=expected,
        ranges=ranges,
        paths=exports,
    )
    # Exports include a two-second visibility margin for boundary reads. Only
    # points inside the raw capture range may drive measurements.
    exported = {
        key: {
            labels: {
                timestamp: value
                for timestamp, value in samples.items()
                if ranges[key].start_ms <= timestamp <= ranges[key].end_ms
            }
            for labels, samples in by_labels.items()
        }
        for key, by_labels in exported.items()
    }
    export_names = {key: path.name for key, path in exports.items()}
    report = install_campaign_metadata(
        merged,
        source_reports=[path.name for path in inputs],
        evidence_exports=export_names,
        remote_write_summary=write_summary,
    )
    report["export_visibility"] = {
        "status": "verified",
        "attempts_by_export": {key: retries[key] + 1 for key in retries},
        "retry_count_by_export": retries,
        "selected_capture_ranges_ms": {
            key: {"start_ms": value.start_ms, "end_ms": value.end_ms}
            for key, value in ranges.items()
        },
        "sample_value_tolerance": VALUE_TOLERANCE,
    }
    unavailable = _populate_run_measurements(report, exported=exported)
    report["measurement_coverage"] = {
        "status": "unavailable" if unavailable else "available",
        "max_gap_ms": MAX_GAP_MS,
        "unavailable_runs": unavailable,
        "ut61b_validity": "held valid AC state merged across AC, DC, and unknown mode validity samples",
    }
    try:
        layer_report.validate_report(report)
    except layer_report.ReportError as exc:
        raise ReportError(f"final combined report validation failed: {exc}") from exc
    _write_new_bytes(output, (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    # Measurement coverage failures are reported and retained in the JSON, but
    # nonzero status prevents a downstream campaign publisher from accepting it.
    return report, 2 if unavailable else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+", type=Path, help="batch reports in campaign order")
    parser.add_argument("--gpu-jsonl", required=True, type=Path, help="raw-only GPU telemetry JSONL")
    parser.add_argument("--ut61b-jsonl", required=True, type=Path, help="raw-only UT61B telemetry JSONL")
    parser.add_argument("--vm-url", default="http://127.0.0.1:8428", help="loopback VictoriaMetrics base URL")
    parser.add_argument("--output", required=True, type=Path, help="new combined JSON report path")
    parser.add_argument("--evidence-prefix", required=True, type=Path, help="prefix for new raw VM export evidence files")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="skip remote-write entirely and verify already-ingested GPU, UT61B, and token samples",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report, status = build_report(
            inputs=args.input,
            gpu_jsonl=args.gpu_jsonl,
            ut61b_jsonl=args.ut61b_jsonl,
            vm_url=args.vm_url,
            output=args.output,
            evidence_prefix=args.evidence_prefix,
            verify_existing=args.verify_existing,
        )
        print(json.dumps({
            "output": str(args.output),
            "evidence_exports": report["evidence_exports"],
            "remote_write": report["remote_write"],
            "export_visibility": report["export_visibility"],
            "measurement_coverage": report["measurement_coverage"],
        }, indent=2, allow_nan=False))
        return status
    except (ReportError, OSError, ValueError) as exc:
        print(f"gemma4_same_clock_report: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
