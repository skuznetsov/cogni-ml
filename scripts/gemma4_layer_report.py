#!/usr/bin/env python3
"""Render cogni.gemma4.layer_depth.v1 reports as standalone comparison HTML."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


SCHEMA = "cogni.gemma4.layer_depth.v1"
DEPTH_COLORS = ("#247a72", "#c07824", "#6952a3", "#2878b5", "#b34a5a", "#668a35")
CIRCUIT_METERS = {"fluke": "Fluke", "ut61b": "UT61B"}


class ReportError(ValueError):
    """Raised when an input report is malformed or incompatible with a merge."""


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _optional_type(
    mapping: dict[str, Any], key: str, expected: type | tuple[type, ...], location: str
) -> Any:
    value = mapping.get(key)
    if value is not None and not isinstance(value, expected):
        if isinstance(expected, tuple):
            expected_name = " or ".join(item.__name__ for item in expected)
        else:
            expected_name = expected.__name__
        raise ReportError(f"{location}.{key} must be {expected_name} or null")
    return value


def _number(value: Any, location: str, *, integer: bool = False) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{location} must be a finite number or null")
    if not math.isfinite(value) or value < 0:
        raise ReportError(f"{location} must be a finite non-negative number or null")
    if integer and not isinstance(value, int):
        raise ReportError(f"{location} must be an integer or null")
    return value


def _sha256(value: Any, location: str) -> str | None:
    value = _optional_type({"value": value}, "value", str, location)
    if value is not None and (len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value)):
        raise ReportError(f"{location} must be a 64-character SHA-256 hex digest or null")
    return value


def _circuit_meter(report: dict[str, Any]) -> str:
    """Return the selected circuit meter, preserving legacy reports as Fluke."""
    return report.get("circuit_meter", "fluke")


def _circuit_meter_name(report: dict[str, Any]) -> str:
    return CIRCUIT_METERS[_circuit_meter(report)]


def _measurement_clock(report: dict[str, Any]) -> str:
    return report.get("measurement_clock") or "gputer"


def _circuit_chart_title(report: dict[str, Any], *, paired: bool = False) -> str:
    if _circuit_meter(report) == "ut61b":
        return "UT61B circuit"
    return "Fluke" if paired else "Circuit"


def _validate_circuit_measurement_identity(
    measurement: Any,
    location: str,
    report_meter: str,
    measurement_clock: str | None,
) -> None:
    if report_meter == "ut61b" and not isinstance(measurement, dict):
        raise ReportError(f"{location} must retain UT61B source identity, including when unavailable")
    if measurement is None or not isinstance(measurement, dict):
        return
    measurement_meter = _optional_type(measurement, "meter", str, location)
    selector = _optional_type(measurement, "selector", str, location)
    if measurement_meter is not None and measurement_meter not in CIRCUIT_METERS:
        raise ReportError(f"{location}.meter must be 'fluke', 'ut61b', or null")
    if measurement_meter is not None and measurement_meter != report_meter:
        raise ReportError(
            f"{location}.meter {measurement_meter!r} does not match report.circuit_meter {report_meter!r}"
        )
    metric = measurement.get("metric")
    if report_meter == "fluke" and (
        (isinstance(metric, dict) and metric.get("__name__") == "ut61b_current_amperes")
        or (selector is not None and selector.lstrip().startswith("ut61b_current_amperes"))
    ):
        raise ReportError(f"{location} identifies UT61B data in a Fluke report")
    if report_meter == "ut61b":
        if measurement_meter != "ut61b":
            raise ReportError(f"{location}.meter must be 'ut61b' for a UT61B report")
        if measurement_clock is None:
            raise ReportError("report.measurement_clock is required for UT61B source identity")
        selector_match = re.fullmatch(
            r"\s*ut61b_current_amperes\s*\{\s*(.*?)\s*\}\s*", selector or ""
        )
        if selector_match is None:
            raise ReportError(
                f"{location}.selector must identify the UT61B current-amperes metric and labels"
            )
        selector_labels: dict[str, str] = {}
        for item in selector_match.group(1).split(","):
            key, separator, value = item.partition("=")
            key, value = key.strip(), value.strip()
            if not separator or not key or len(value) < 2 or value[0] != '"' or value[-1] != '"':
                raise ReportError(f"{location}.selector has malformed UT61B label matchers")
            if key in selector_labels:
                raise ReportError(f"{location}.selector repeats a UT61B label matcher")
            selector_labels[key] = value[1:-1]

        if not isinstance(metric, dict) or metric.get("__name__") != "ut61b_current_amperes":
            raise ReportError(f"{location}.metric.__name__ must be 'ut61b_current_amperes'")
        expected_labels = {
            "circuit": measurement_clock,
            "meter": "ut61b",
            "mode": "ac",
        }
        for key, expected in expected_labels.items():
            if selector_labels.get(key) != expected or metric.get(key) != expected:
                raise ReportError(
                    f"{location} must identify UT61B metric label {key}={expected!r}"
                )

        active_fields = (
            "mean_active_power_watts",
            "active_energy_joules",
            "estimated_active_energy_joules",
        )
        if any(measurement.get(field) is not None for field in active_fields):
            raise ReportError(f"{location} cannot report UT61B current as active W or J")

        if measurement.get("status") == "available":
            if measurement.get("kind") != "current_ac" or measurement.get("power_unit") != "VA":
                raise ReportError(
                    f"{location} must preserve current_ac and VA for UT61B apparent power"
                )
        elif (
            measurement.get("kind") not in (None, "current_ac")
            or measurement.get("power_unit") not in (None, "VA")
        ):
            raise ReportError(f"{location} has conflicting UT61B current_ac/VA metadata")


def _validate_curve(measurement: dict[str, Any], location: str, *, percent: bool = False) -> None:
    curve = measurement.get("curve")
    if curve is not None and not isinstance(curve, list):
        raise ReportError(f"{location}.curve must be an array or null")
    if not isinstance(curve, list):
        return
    for index, point in enumerate(curve):
        point_location = f"{location}.curve[{index}]"
        if not isinstance(point, dict):
            raise ReportError(f"{point_location} must be an object")
        elapsed = _number(point.get("elapsed_ms"), f"{point_location}.elapsed_ms")
        value = _number(point.get("value"), f"{point_location}.value")
        if elapsed is None or value is None:
            raise ReportError(f"{point_location} requires elapsed_ms and value")
        if percent and value > 100:
            raise ReportError(f"{point_location}.value must be at most 100 for utilization")


def _validate_measurement(value: Any, location: str, *, utilization: bool = False) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ReportError(f"{location} must be an object or null")
    status = _optional_type(value, "status", str, location)
    if status not in (None, "available", "unavailable"):
        raise ReportError(f"{location}.status must be 'available', 'unavailable', or null")
    _optional_type(value, "reason", str, location)
    if utilization:
        mean = _number(value.get("mean_percent"), f"{location}.mean_percent")
        if mean is not None and mean > 100:
            raise ReportError(f"{location}.mean_percent must be at most 100")
        _number(value.get("duration_seconds"), f"{location}.duration_seconds")
        _number(value.get("sample_count"), f"{location}.sample_count", integer=True)
        _validate_curve(value, location, percent=True)
        return

    _optional_type(value, "kind", str, location)
    _optional_type(value, "power_unit", str, location)
    for key in (
        "duration_seconds",
        "mean_active_power_watts",
        "active_energy_joules",
        "mean_apparent_power_va",
        "apparent_energy_va_seconds",
        "estimated_active_energy_joules",
        "assumed_voltage_volts",
        "assumed_power_factor",
    ):
        _number(value.get(key), f"{location}.{key}")
    _number(value.get("sample_count"), f"{location}.sample_count", integer=True)
    _validate_curve(value, location)


def _validate_token_progress(run: dict[str, Any], location: str) -> None:
    progress = run.get("token_progress")
    if progress is None:
        return
    if not isinstance(progress, dict):
        raise ReportError(f"{location}.token_progress must be an object")
    if progress.get("status") != "available":
        # Partial observations remain in the raw report, never a denominator.
        return
    prefix = f"{location}.token_progress"
    if progress.get("basis") != "received_token_ids":
        raise ReportError(f"{prefix} requires received_token_ids")
    total = progress.get("final_tokens")
    if isinstance(total, bool) or not isinstance(total, int) or total < 1 or total != run.get("actual_tokens"):
        raise ReportError(f"{prefix} final count must agree with actual_tokens")
    points = progress.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise ReportError(f"{prefix} needs at least two observations")
    previous_time, previous_count = -1, -1
    for point in points:
        if not isinstance(point, dict):
            raise ReportError(f"{prefix} points must be objects")
        time_ms, count = point.get("elapsed_ms"), point.get("generated_tokens")
        if (isinstance(time_ms, bool) or not isinstance(time_ms, int) or time_ms < 0
                or isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= total
                or time_ms <= previous_time or count < previous_count):
            raise ReportError(f"{prefix} requires ordered integer times and nondecreasing token counts")
        previous_time, previous_count = time_ms, count
    start, end = run.get("started_at_epoch_ms"), run.get("ended_at_epoch_ms")
    if (not isinstance(start, int) or isinstance(start, bool)
            or not isinstance(end, int) or isinstance(end, bool)
            or points[0] != {"elapsed_ms": 0, "generated_tokens": 0}
            or previous_time != end - start or previous_count != total):
        raise ReportError(f"{prefix} must cover the full request window from zero to the final count")


def validate_report(report: Any) -> dict[str, Any]:
    """Validate all fields used by the report while allowing future extra fields."""
    if not isinstance(report, dict):
        raise ReportError("report root must be an object")
    if report.get("schema") != SCHEMA:
        raise ReportError(f"schema must be {SCHEMA!r}")
    circuit_meter = _circuit_meter(report)
    if not isinstance(circuit_meter, str) or circuit_meter not in CIRCUIT_METERS:
        raise ReportError("report.circuit_meter must be 'fluke' or 'ut61b'")
    measurement_clock = _optional_type(report, "measurement_clock", str, "report")
    if measurement_clock is not None and not measurement_clock.strip():
        raise ReportError("report.measurement_clock must be a non-empty string or null")
    if circuit_meter == "ut61b" and measurement_clock is None:
        raise ReportError("report.measurement_clock is required for UT61B reports")
    _optional_type(report, "campaign_id", str, "report")
    _sha256(report.get("prompt_sha256"), "report.prompt_sha256")
    _optional_type(report, "notes", str, "report")
    _optional_type(report, "full_depth", bool, "report")
    _number(report.get("model_file_bytes"), "report.model_file_bytes", integer=True)
    _optional_type(report, "ignore_eos", bool, "report")
    _optional_type(report, "cache_prompt", bool, "report")
    _number(report.get("temperature"), "report.temperature")
    for key in ("seed", "warmup_tokens"):
        value = report.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ReportError(f"report.{key} must be an integer or null")
    _number(report.get("warmup_tokens"), "report.warmup_tokens", integer=True)
    requested = _number(report.get("requested_tokens"), "report.requested_tokens", integer=True)
    if requested is None or requested < 1:
        raise ReportError("report.requested_tokens must be a positive integer")
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise ReportError("report.runs must be an array")
    for index, run in enumerate(runs):
        location = f"runs[{index}]"
        if not isinstance(run, dict):
            raise ReportError(f"{location} must be an object")
        label = _optional_type(run, "label", str, location)
        if not label:
            raise ReportError(f"{location}.label must be a non-empty string")
        depth = _number(run.get("decoder_layers"), f"{location}.decoder_layers", integer=True)
        if depth is None or depth < 1:
            raise ReportError(f"{location}.decoder_layers must be a positive integer")
        trial = _number(run.get("trial"), f"{location}.trial", integer=True)
        if trial is None or trial < 1:
            raise ReportError(f"{location}.trial must be a positive integer")
        status = _optional_type(run, "status", str, location)
        if status not in (None, "completed", "failed"):
            raise ReportError(f"{location}.status must be 'completed', 'failed', or null")
        _number(run.get("requested_tokens"), f"{location}.requested_tokens", integer=True)
        for key in ("started_at_epoch_ms", "ended_at_epoch_ms", "elapsed_ms"):
            _number(run.get(key), f"{location}.{key}")
        _number(run.get("actual_tokens"), f"{location}.actual_tokens", integer=True)
        _validate_token_progress(run, location)
        _optional_type(run, "model_identity", str, location)
        _sha256(run.get("model_sha256"), f"{location}.model_sha256")
        _number(run.get("parameter_count"), f"{location}.parameter_count", integer=True)
        _number(run.get("weight_bytes"), f"{location}.weight_bytes", integer=True)
        _optional_type(run, "answer_preview", str, location)
        _optional_type(run, "error", str, location)
        _optional_type(run, "_source_report", str, location)
        _validate_measurement(run.get("gpu_measurement"), f"{location}.gpu_measurement")
        circuit_measurement = run.get("circuit_measurement")
        _validate_measurement(circuit_measurement, f"{location}.circuit_measurement")
        _validate_circuit_measurement_identity(
            circuit_measurement,
            f"{location}.circuit_measurement",
            circuit_meter,
            measurement_clock,
        )
        _validate_measurement(
            run.get("gpu_utilization_measurement"),
            f"{location}.gpu_utilization_measurement",
            utilization=True,
        )
    sources = report.get("source_reports")
    if sources is not None:
        if not isinstance(sources, list) or any(not isinstance(item, str) for item in sources):
            raise ReportError("report.source_reports must be an array of strings or null")
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return "Unavailable"
    return format(value, ".4g")


def _power_class(measurement: dict[str, Any] | None) -> tuple[str, str] | None:
    if not measurement or measurement.get("status") != "available":
        return None
    kind = measurement.get("kind")
    unit = measurement.get("power_unit")
    if kind == "active_power" and unit == "W":
        return ("W", "active")
    if kind == "current_ac" and unit == "VA":
        return ("VA", "apparent")
    return None


def _measurement_cell(measurement: dict[str, Any] | None, source: str, actual_tokens: Any) -> str:
    power_source = "GPU board" if source == "GPU" else source
    if measurement is None:
        return (
            f"<div>{power_source} power: Unavailable</div>"
            f"<div>{power_source} energy: Unavailable</div>"
            "<div class=\"muted\">sample window: Unavailable; samples: Unavailable</div>"
        )
    status = measurement.get("status")
    if status != "available":
        reason = measurement.get("reason")
        reason_html = f": {_escape(reason)}" if reason else ""
        return (
            f"<div>{power_source} power: Unavailable</div>"
            f"<div>{power_source} energy: Unavailable</div>"
            f"<div class=\"muted\">{source} unavailable{reason_html}</div>"
            "<div class=\"muted\">sample window: Unavailable; samples: Unavailable</div>"
        )
    classification = _power_class(measurement)
    if classification is None:
        return (
            f"<div>{power_source} power: Unavailable</div>"
            f"<div>{power_source} energy: Unavailable</div>"
            f"<div class=\"muted\">power unit unavailable or inconsistent</div>"
            f"<div class=\"muted\">sample window: {_fmt(measurement.get('duration_seconds'))} s; "
            f"samples: {_fmt(measurement.get('sample_count'))}</div>"
        )
    unit, kind = classification
    lines: list[str] = []
    if kind == "active":
        mean = measurement.get("mean_active_power_watts")
        energy = measurement.get("active_energy_joules")
        lines.append(f"<div>{power_source} power: {_fmt(mean)} W</div>")
        lines.append(f"<div>{power_source} energy: {_fmt(energy)} J</div>")
        if energy is not None and actual_tokens is not None and actual_tokens > 0:
            per_token_label = "GPU energy per actual token" if source == "GPU" else f"{source} energy per actual token"
            lines.append(f"<div>{per_token_label}: {_fmt(energy / actual_tokens)} J/token</div>")
        else:
            per_token_label = "GPU energy per actual token" if source == "GPU" else f"{source} energy per actual token"
            lines.append(f"<div>{per_token_label}: Unavailable</div>")
    else:
        mean = measurement.get("mean_apparent_power_va")
        energy = measurement.get("apparent_energy_va_seconds")
        power_label = f"{source} apparent power"
        energy_label = f"{source} apparent energy"
        lines.append(f"<div>{power_label}: {_fmt(mean)} VA</div>")
        lines.append(f"<div>{energy_label}: {_fmt(energy)} VA·s</div>")
        estimate = measurement.get("estimated_active_energy_joules")
        if estimate is not None:
            assumptions = []
            voltage = measurement.get("assumed_voltage_volts")
            factor = measurement.get("assumed_power_factor")
            if voltage is not None:
                assumptions.append(f"{_fmt(voltage)} V assumed voltage")
            if factor is not None:
                assumptions.append(f"power factor {_fmt(factor)}")
            assumption_text = "; ".join(assumptions) or "assumptions unavailable"
            lines.append(
                f"<div>Estimated active energy: {_fmt(estimate)} J ({_escape(assumption_text)})</div>"
            )
    duration = measurement.get("duration_seconds")
    samples = measurement.get("sample_count")
    lines.append(
        f"<div class=\"muted\">sample window: {_fmt(duration)} s; samples: {_fmt(samples)}</div>"
    )
    return "".join(lines)


def _utilization_cell(measurement: dict[str, Any] | None) -> str:
    if measurement is None or measurement.get("status") != "available":
        reason = measurement.get("reason") if measurement else None
        suffix = f": {_escape(reason)}" if reason else ""
        duration = measurement.get("duration_seconds") if measurement else None
        samples = measurement.get("sample_count") if measurement else None
        return (
            "<div>Mean GPU utilization: Unavailable</div>"
            f"<div class=\"muted\">GPU utilization unavailable{suffix}</div>"
            f"<div class=\"muted\">sample window: {_fmt(duration)} s; samples: {_fmt(samples)}</div>"
        )
    mean = measurement.get("mean_percent")
    return (
        f"<div>Mean GPU utilization: {_fmt(mean)}%</div>"
        f"<div class=\"muted\">sample window: {_fmt(measurement.get('duration_seconds'))} s; "
        f"samples: {_fmt(measurement.get('sample_count'))}</div>"
    )


def _gate_reasons(report: dict[str, Any]) -> list[str]:
    runs = report["runs"]
    requested = report["requested_tokens"]
    reasons: list[str] = []
    if len({run["decoder_layers"] for run in runs}) < 2:
        reasons.append("at least two decoder depths are required")
    if not report.get("prompt_sha256"):
        reasons.append("prompt SHA-256 unavailable")
    for run in runs:
        label = run["label"]
        if run.get("status") != "completed":
            reasons.append(f"{label}: status is {run.get('status') or 'unavailable'}")
        run_requested = run.get("requested_tokens")
        if run_requested is None:
            reasons.append(f"{label}: run requested token count unavailable")
        elif run_requested != requested:
            reasons.append(
                f"{label}: run requested {run_requested}, report requests {requested}"
            )
        actual = run.get("actual_tokens")
        if actual is None:
            reasons.append(f"{label}: actual token count unavailable")
        elif actual != requested:
            reasons.append(f"{label}: observed {actual} of {requested} requested")
    return reasons


def _color_by_depth(runs: list[dict[str, Any]]) -> dict[int, str]:
    depths = sorted({run["decoder_layers"] for run in runs})
    return {depth: DEPTH_COLORS[index % len(DEPTH_COLORS)] for index, depth in enumerate(depths)}


def _is_exact_run(run: dict[str, Any], requested: int) -> bool:
    return (
        run.get("status") == "completed"
        and run.get("requested_tokens") == requested
        and run.get("actual_tokens") == requested
    )


def _mean(values: list[float | int | None], eligible: bool) -> str:
    if not eligible or not values or any(value is None for value in values):
        return "Unavailable"
    return _fmt(sum(value for value in values if value is not None) / len(values))


def _parameter_summary(runs: list[dict[str, Any]]) -> str:
    values = [run.get("parameter_count") for run in runs]
    present = {value for value in values if value is not None}
    if not present:
        return "Unavailable"
    if len(present) > 1:
        return "Varies"
    value = next(iter(present))
    formatted = f"{value:,}"
    return formatted if all(item is not None for item in values) else f"{formatted} (partial)"


def _summary_table(report: dict[str, Any]) -> str:
    requested = report["requested_tokens"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in report["runs"]:
        groups.setdefault(run["label"], []).append(run)

    rows: list[str] = []
    for label, runs in groups.items():
        exact = all(_is_exact_run(run, requested) for run in runs)
        depths = {run["decoder_layers"] for run in runs}
        depth_text = str(next(iter(depths))) if len(depths) == 1 else "Mixed"
        gpu_measurements = [run.get("gpu_measurement") for run in runs]

        def gpu_values(key: str) -> list[float | int | None]:
            return [measurement.get(key) if measurement else None for measurement in gpu_measurements]

        active_w_units = all(_power_class(measurement) == ("W", "active") for measurement in gpu_measurements)
        mean_elapsed = _mean(
            [run.get("elapsed_ms") / 1000 if run.get("elapsed_ms") is not None else None for run in runs],
            exact,
        )
        mean_watts = _mean(gpu_values("mean_active_power_watts"), exact and active_w_units)
        mean_joules = _mean(gpu_values("active_energy_joules"), exact and active_w_units)
        mean_joules_per_token = _mean(
            [
                measurement.get("active_energy_joules") / run["actual_tokens"]
                if measurement
                and measurement.get("active_energy_joules") is not None
                and run.get("actual_tokens")
                else None
                for run, measurement in zip(runs, gpu_measurements)
            ],
            exact and active_w_units,
        )
        cells = (
            _escape(label),
            _escape(depth_text),
            f"{sum(_is_exact_run(run, requested) for run in runs)} / {len(runs)}",
            _escape(_parameter_summary(runs)),
            mean_elapsed,
            mean_watts,
            mean_joules,
            mean_joules_per_token,
        )
        rows.append(
            "<tr>"
            + "".join(f"<td>{cell}</td>" for cell in cells)
            + "</tr>"
        )

    return (
        '<div class="table-wrap"><table class="summary-table"><thead><tr>'
        "<th>Run label</th><th>Decoder depth</th><th>Exact trials / total</th>"
        "<th>Reported parameter count</th><th>Mean elapsed / trial (s)</th>"
        "<th>Mean GPU power / trial (W)</th><th>Mean GPU energy / request (J)</th>"
        "<th>Mean GPU energy / token (J/token)</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _chart(
    report: dict[str, Any],
    source_key: str,
    title_prefix: str,
    *,
    utilization: bool = False,
) -> str:
    candidates = []
    units: set[str] = set()
    for run in report["runs"]:
        measurement = run.get(source_key)
        if not measurement or measurement.get("status") != "available":
            continue
        classification = ("%", "utilization") if utilization else _power_class(measurement)
        curve = measurement.get("curve")
        if classification is None or not isinstance(curve, list) or len(curve) < 2:
            continue
        unit, kind = classification
        units.add(unit)
        candidates.append((run, measurement, unit, kind))
    if len(units) > 1:
        source_label = "GPU" if source_key == "gpu_measurement" else (
            f"{_circuit_meter_name(report)} circuit" if _circuit_meter(report) == "ut61b" else "Circuit"
        )
        unit_names = " and ".join(unit for unit in ("W", "VA", "%") if unit in units)
        return f'<p class="chart-note">{source_label} chart omitted because available curves mix {unit_names}</p>'
    if not candidates:
        return f'<p class="chart-note">{_escape(title_prefix)} chart unavailable: no compatible sampled curves</p>'
    unit = next(iter(units))
    kind = candidates[0][3]
    descriptor = "utilization" if utilization else ("active power" if kind == "active" else "apparent power")
    title = f"{title_prefix} {descriptor} ({unit}) over elapsed time (ms)"
    width, height = 720, 280
    left, right, top, bottom = 58, 18, 20, 42
    plot_width, plot_height = width - left - right, height - top - bottom
    all_x = [point["elapsed_ms"] for _, measurement, _, _ in candidates for point in measurement["curve"]]
    all_y = [point["value"] for _, measurement, _, _ in candidates for point in measurement["curve"]]
    min_x, max_x = min(all_x), max(all_x)
    if max_x == min_x:
        max_x = min_x + 1
    min_y = 0 if min(all_y) >= 0 else min(all_y)
    max_y = max(all_y)
    if max_y <= min_y:
        max_y = min_y + 1
    else:
        max_y *= 1.08
    colors = _color_by_depth(report["runs"])
    svg: list[str] = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}" xmlns="http://www.w3.org/2000/svg">',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>',
        f'<text x="{left - 7}" y="{top + 5}" text-anchor="end" class="tick">{_fmt(max_y)}</text>',
        f'<text x="{left - 7}" y="{top + plot_height}" text-anchor="end" class="tick">{_fmt(min_y)}</text>',
        f'<text x="{left}" y="{height - 12}" class="tick">{_fmt(min_x)} ms</text>',
        f'<text x="{left + plot_width}" y="{height - 12}" text-anchor="end" class="tick">{_fmt(max_x)} ms</text>',
    ]
    for run, measurement, _, _ in candidates:
        points = []
        for point in measurement["curve"]:
            x = left + (point["elapsed_ms"] - min_x) / (max_x - min_x) * plot_width
            y = top + (max_y - point["value"]) / (max_y - min_y) * plot_height
            points.append(f"{x:.2f},{y:.2f}")
        color = colors[run["decoder_layers"]]
        svg.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
    svg.append("</svg>")
    legend_depths = sorted({run["decoder_layers"] for run, _, _, _ in candidates})
    legend = "".join(
        f'<span><i style="background:{colors[depth]}"></i>decoder depth {depth}</span>'
        for depth in legend_depths
    )
    return f'<figure><figcaption>{_escape(title)}</figcaption>{"".join(svg)}<div class="legend">{legend}</div></figure>'


def _paired_window_reason(run: dict[str, Any], circuit_meter: str = "fluke") -> str | None:
    """Require recorded common windows; matching clocks does not prove sensor sync."""
    start, end = run.get("started_at_epoch_ms"), run.get("ended_at_epoch_ms")
    if type(start) is not int or type(end) is not int or end <= start:
        return "request window is missing or invalid"
    duration_ms = end - start
    for key, source, expected in (
        ("gpu_measurement", "GPU", ("W", "active")),
        ("circuit_measurement", CIRCUIT_METERS[circuit_meter], ("VA", "apparent")),
    ):
        measurement = run.get(key)
        if _power_class(measurement) != expected:
            return f"{source} measurement is unavailable or has incompatible units"
        assert measurement is not None
        if (
            type(measurement.get("start_ms")) is not int
            or type(measurement.get("end_ms")) is not int
            or measurement["start_ms"] != start
            or measurement["end_ms"] != end
        ):
            return f"{source} measurement does not match the request window"
        duration = measurement.get("duration_seconds")
        if duration is None or not math.isclose(duration * 1000, duration_ms, abs_tol=0.001):
            return f"{source} duration disagrees with the request window"
        curve = measurement.get("curve")
        if not isinstance(curve, list) or len(curve) < 2:
            return f"{source} full-window curve is unavailable"
        times = [point["elapsed_ms"] for point in curve]
        if times[0] != 0 or times[-1] != duration_ms:
            return f"{source} curve does not cover the full request window"
        if any(left >= right for left, right in zip(times, times[1:])):
            return f"{source} curve timestamps are not strictly increasing"
    return None


def _depth_chart_payload(
    report: dict[str, Any], *, presentation_source: str | None = None
) -> dict[str, Any]:
    """Keep per-trial source curves and reject ambiguous chart identities."""
    circuit_meter = _circuit_meter(report)
    if presentation_source is not None:
        if presentation_source not in {"gpu", "fluke"}:
            raise ReportError("presentation_source must be 'gpu' or 'fluke'")
        if circuit_meter != "ut61b":
            raise ReportError("presentation_source is reserved for the UT61B demo")
    identities: dict[tuple[str, int], int] = {}
    for run in report["runs"]:
        identity = (run["label"], run["trial"])
        identities[identity] = identities.get(identity, 0) + 1
    labels = list(dict.fromkeys(run["label"] for run in report["runs"]))
    palette = ("#3167bd", "#d8942b", "#168467", "#8161b4", "#b34a5a", "#668a35")
    colors = {label: palette[index % len(palette)] for index, label in enumerate(labels)}
    runs = []
    for run in report["runs"]:
        if (
            identities[(run["label"], run["trial"])] != 1
            or not _is_exact_run(run, report["requested_tokens"])
            or _paired_window_reason(run, circuit_meter) is not None
        ):
            continue
        item = {
            "label": run["label"],
            "decoder_layers": run["decoder_layers"],
            "trial": run["trial"],
            "duration_ms": run["ended_at_epoch_ms"] - run["started_at_epoch_ms"],
            "color": colors[run["label"]],
        }
        progress = run.get("token_progress")
        if progress and progress.get("status") == "available":
            _validate_token_progress(run, run["label"])
            item["token_progress"] = {
                "basis": progress["basis"],
                "points": [[point["elapsed_ms"], point["generated_tokens"]] for point in progress["points"]],
            }
        for name, key, unit, integral_unit, mean_key, integral_key in (
            ("gpu", "gpu_measurement", "W", "J", "mean_active_power_watts", "active_energy_joules"),
            ("fluke", "circuit_measurement", "VA", "VA·s", "mean_apparent_power_va", "apparent_energy_va_seconds"),
        ):
            measurement = run[key]
            item[name] = {
                "unit": unit,
                "integral_unit": integral_unit,
                "mean": measurement.get(mean_key),
                "integral": measurement.get(integral_key),
                "points": [[point["elapsed_ms"], point["value"]] for point in measurement["curve"]],
            }
        runs.append(item)
    payload = {
        "schema": "cogni.gemma4.depth_chart.v1",
        "campaign_id": report.get("campaign_id") or "Gemma 4 decoder-depth comparison",
        "requested_tokens": report["requested_tokens"],
        "circuit_meter": circuit_meter,
        "measurement_clock": _measurement_clock(report),
        "runs": runs,
        "excluded_count": len(report["runs"]) - len(runs),
    }
    if presentation_source is not None:
        payload["presentation_source"] = presentation_source
    return payload


def _depth_chart_assets(
    report: dict[str, Any], *, presentation_source: str | None = None
) -> tuple[str, str, str]:
    data = json.dumps(
        _depth_chart_payload(report, presentation_source=presentation_source),
        ensure_ascii=False,
        allow_nan=False,
    ).replace("<", "\\u003c")
    css = Path(__file__).with_name("gemma4_depth_chart.css").read_text(encoding="utf-8")
    javascript = Path(__file__).with_name("gemma4_depth_chart.js").read_text(encoding="utf-8")
    return data, css, javascript


def _paired_sensor_section(report: dict[str, Any]) -> str:
    meter_name = _circuit_meter_name(report)
    meter_label = f"{meter_name} circuit"
    circuit_meter = _circuit_meter(report)
    groups: dict[str, list[tuple[dict[str, Any], str | None]]] = {}
    pairs = [(run, _paired_window_reason(run, circuit_meter)) for run in report["runs"]]
    for run, reason in pairs:
        groups.setdefault(run["label"], []).append((run, reason))
    columns = (
        ("GPU W", "GPU board mean (W)", "gpu_measurement", "mean_active_power_watts"),
        (f"{meter_name} VA", f"{meter_label} mean (VA)", "circuit_measurement", "mean_apparent_power_va"),
        ("GPU J", "GPU energy / request (J)", "gpu_measurement", "active_energy_joules"),
        (f"{meter_name} VA·s", f"{meter_name} integral (VA·s)", "circuit_measurement", "apparent_energy_va_seconds"),
    )
    rows = []
    for label, group in groups.items():
        matched = sum(reason is None for _, reason in group)
        eligible = matched == len(group) and all(
            _is_exact_run(run, report["requested_tokens"]) for run, _ in group
        )
        cells = [
            f'<td data-label="Run label">{_escape(label)}</td>',
            f'<td data-label="Matched windows">{matched} / {len(group)}</td>',
        ]
        for short, _, key, field in columns:
            values = [(run.get(key) or {}).get(field) for run, _ in group]
            cells.append(f'<td data-label="{short}">{_mean(values, eligible)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    details = []
    opened = False
    for run, reason in pairs:
        label = f'{run["label"]} / trial {run["trial"]} / {run["decoder_layers"]} layers'
        if reason is not None:
            details.append(f'<p class="muted">{_escape(label)}: pairing unavailable — {_escape(reason)}.</p>')
            continue
        window_ms = run["ended_at_epoch_ms"] - run["started_at_epoch_ms"]
        single = {**report, "runs": [run]}
        charts = _chart(single, "gpu_measurement", "GPU") + _chart(
            single, "circuit_measurement", _circuit_chart_title(report, paired=True)
        )
        open_attribute = " open" if not opened else ""
        opened = True
        details.append(
            f'<details class="sensor-pair"{open_attribute}><summary>{_escape(label)} — shared window {_fmt(window_ms)} ms</summary>'
            '<p class="muted">Aligned request time; independent vertical axes, each starting at zero.</p>'
            f'<div class="charts paired-charts">{charts}</div></details>'
        )
    matched_total = sum(reason is None for _, reason in pairs)
    return (
        f'<section class="panel" id="sensor-comparison"><h2>{_escape(meter_name)} and NVIDIA: same request windows</h2>'
        f'<p><strong>Matched sensor windows: {matched_total} / {len(pairs)}</strong>.</p>'
        f'<p>NVIDIA reports active GPU board power in W; {_escape(meter_name)} reports whole-circuit apparent power in VA '
        'from AC current and the assumed grid voltage. Do not subtract these readings or infer GPU share or PSU efficiency. '
        f'Converting {_escape(meter_name)} to active W requires the circuit power factor.</p>'
        '<p>Compare timing and trends within each source. Matching recorded timestamps does not remove '
        'sensor lag, sensor averaging or host-clock offset. No time shift or idle subtraction is applied.</p>'
        '<p class="muted">Request-boundary values are linearly interpolated from bracketing raw samples; '
        'the plotted source curves receive no additional smoothing.</p>'
        '<p class="muted">All trials must have exact token counts and matched sensor windows for a label mean. '
        'Means are per-trial averages; unavailable pairs are retained.</p>'
        '<div class="table-wrap"><table><thead><tr><th>Run label</th><th>Matched windows</th>'
        + "".join(f"<th>{title}</th>" for _, title, _, _ in columns)
        + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        '<h3>Paired sensor traces</h3>' + "".join(details) + "</section>"
    )


def render_html(report: Any) -> str:
    """Render a validated report to a self-contained HTML document."""
    report = validate_report(report)
    meter_name = _circuit_meter_name(report)
    measurement_clock = _measurement_clock(report)
    reasons = _gate_reasons(report)
    gate = "blocked" if reasons else "eligible"
    requested = report["requested_tokens"]
    colors = _color_by_depth(report["runs"])
    rows: list[str] = []
    for run in report["runs"]:
        depth = run["decoder_layers"]
        label = run["label"]
        status = run.get("status") or "unavailable"
        actual = run.get("actual_tokens")
        elapsed_ms = run.get("elapsed_ms")
        throughput = actual / (elapsed_ms / 1000) if actual is not None and elapsed_ms else None
        run_requested = run.get("requested_tokens")
        token_display = f"{_fmt(actual)} / {_fmt(run_requested)} run-requested"
        duration_display = f"{_fmt(elapsed_ms)} ms"
        throughput_display = f"Observed throughput: {_fmt(throughput)} tokens/s"
        model = _escape(run.get("model_identity") or "Unavailable")
        model_hash = _escape(run.get("model_sha256") or "Unavailable")
        metadata = (
            f"<div>Model: {model}</div><div>Model SHA-256: <code>{model_hash}</code></div>"
            f"<div>Parameters: {_fmt(run.get('parameter_count'))}</div>"
            f"<div>Weight bytes: {_fmt(run.get('weight_bytes'))}</div>"
            "<div>weight size is not a power measurement</div>"
        )
        source = run.get("_source_report")
        source_line = f"<div>Source report: {_escape(source)}</div>" if source else ""
        error = run.get("error")
        error_line = f'<div class="error">Error: {_escape(error)}</div>' if error else ""
        answer = run.get("answer_preview")
        answer_line = (
            f"<details><summary>Answer preview (quality unvalidated)</summary><pre>{_escape(answer)}</pre></details>"
            if answer is not None
            else "<div>Answer preview: Unavailable</div>"
        )
        rows.append(
            "<tr>"
            f'<td data-label="Run"><strong>{_escape(label)}</strong><div class="depth"><i style="background:{colors[depth]}"></i>decoder depth {depth}</div><div>Trial {_fmt(run["trial"])}</div><div class="status {status}">{_escape(status)}</div>{source_line}</td>'
            f'<td data-label="Tokens"><div>Actual tokens: {_fmt(actual)}</div><div>{_escape(token_display)}</div><div>Report request: {_fmt(requested)}</div><div>{_escape(throughput_display)}</div></td>'
            f'<td data-label="Runtime"><div>Elapsed: {_escape(duration_display)}</div><div>Started: {_fmt(run.get("started_at_epoch_ms"))} epoch ms</div><div>Ended: {_fmt(run.get("ended_at_epoch_ms"))} epoch ms</div></td>'
            f'<td data-label="GPU board">{_measurement_cell(run.get("gpu_measurement"), "GPU", actual)}</td>'
            f'<td data-label="Circuit meter">{_measurement_cell(run.get("circuit_measurement"), meter_name, actual)}</td>'
            f'<td data-label="GPU utilization">{_utilization_cell(run.get("gpu_utilization_measurement"))}</td>'
            f'<td data-label="Model and output">{metadata}{error_line}{answer_line}</td>'
            "</tr>"
        )
    depths_present = {run["decoder_layers"] for run in report["runs"]}
    expected_depths = {16, 32, 48}
    missing_expected_depths = sorted(expected_depths - depths_present)
    if reasons:
        scope_message = "The fixed-token comparison is blocked."
    elif missing_expected_depths:
        missing = ", ".join(str(depth) for depth in missing_expected_depths)
        scope_message = (
            "Eligible for a partial depth comparison; expected campaign depths are "
            f"16/32/48, with {missing} not represented."
        )
    else:
        scope_message = "Eligible; the expected 16/32/48 decoder-depth set is represented."
    blocker_html = (
        "<p>Every run must complete with exactly the requested output-token count, and at least two decoder depths must be present.</p>"
        + "<ul>" + "".join(f"<li>{_escape(reason)}</li>" for reason in reasons) + "</ul>"
        if reasons
        else f"<p>{_escape(scope_message)} Every listed run completed with the requested output-token count. This is a fixed-token gate, not an answer-quality result.</p>"
    )
    title_id = _escape(report.get("campaign_id") or "Gemma 4 decoder-depth comparison")
    prompt_hash = _escape(report.get("prompt_sha256") or "Unavailable")
    notes = report.get("notes")
    notes_html = f'<p class="notes">Notes: {_escape(notes)}</p>' if notes else ""
    sources = report.get("source_reports") or []
    sources_html = (
        "<p>Source reports: " + ", ".join(_escape(source) for source in sources) + "</p>"
        if sources
        else ""
    )
    gpu_chart = _chart(report, "gpu_measurement", "GPU")
    circuit_chart = _chart(report, "circuit_measurement", _circuit_chart_title(report))
    utilization_chart = _chart(report, "gpu_utilization_measurement", "GPU", utilization=True)
    chart_data, chart_css, chart_js = _depth_chart_assets(report)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_id} — Gemma 4 depth report</title>
<style>
:root{{color-scheme:light;--ink:#1f2937;--muted:#64748b;--line:#dbe3ea;--paper:#fff;--wash:#f3f6f8;--accent:#246b63}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1480px;margin:0 auto;padding:clamp(14px,3vw,32px)}}h1{{font-size:clamp(1.5rem,3vw,2rem);line-height:1.15;margin:0 0 .4rem}}h2{{font-size:1.15rem;margin:0 0 .7rem}}p{{margin:.4rem 0 .8rem}}.muted,.notes,small{{color:var(--muted)}}.panel{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0;box-shadow:0 2px 8px #2333440a}}
.gate{{border-left:5px solid var(--accent)}}.gate.blocked{{border-left-color:#b45309}}.gate strong{{font-size:1.1rem}}.table-wrap{{width:100%}}table{{border-collapse:collapse;width:100%;table-layout:fixed}}th,td{{border-bottom:1px solid var(--line);padding:11px 10px;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);background:#f8fafb}}td{{font-size:.88rem}}td>div{{margin:0 0 3px}}code{{font-size:.77rem;overflow-wrap:anywhere}}.depth{{margin-top:4px}}.depth i,.legend i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}.status{{display:inline-block;font-weight:650;text-transform:capitalize}}.status.failed,.error{{color:#a32929}}.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:12px}}figure{{margin:0;min-width:0}}figcaption{{font-weight:650;margin-bottom:6px}}.chart{{display:block;width:100%;height:auto;max-height:320px}}.axis{{stroke:#607080;stroke-width:1}}.tick{{font:11px system-ui,sans-serif;fill:#586575}}.legend{{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:.78rem;color:var(--muted)}}.chart-note{{color:var(--muted);font-size:.9rem}}details{{margin-top:8px}}summary{{cursor:pointer;color:#315f93}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:7px 0 0;font:inherit;background:#f6f8fa;padding:8px;border-radius:6px}}
.paired-charts{{grid-template-columns:1fr;margin-top:10px}}.sensor-pair{{border-top:1px solid var(--line);padding:10px 0}}h3{{font-size:1rem;margin:18px 0 8px}}
@media(max-width:1050px){{table,tbody,tr,td{{display:block;width:100%}}thead{{position:absolute;clip:rect(0 0 0 0);clip-path:inset(50%);height:1px;width:1px;overflow:hidden}}tr{{border:1px solid var(--line);border-radius:10px;margin:0 0 12px;padding:4px 12px;background:#fff}}td{{border:0;border-bottom:1px solid #edf1f4;padding:9px 4px 9px 132px;min-height:38px;position:relative}}td:last-child{{border-bottom:0}}td::before{{content:attr(data-label);position:absolute;left:4px;top:9px;width:120px;color:var(--muted);font-size:.75rem;font-weight:700;text-transform:uppercase}}}}
@media(max-width:520px){{main{{padding:12px}}.panel{{padding:12px}}td{{padding-left:100px}}td::before{{width:90px}}}}
{chart_css}
</style></head><body><main>
<header><h1>{title_id}</h1><p>Prefix-pruned Gemma 4 decoder-layer experiment; report-level requested output: <strong>{_fmt(requested)} tokens per run</strong>.</p><p>Measurement clock: {_escape(measurement_clock)}</p><p>Prompt SHA-256: <code>{prompt_hash}</code></p>{notes_html}{sources_html}</header>
<section class="panel gate {gate}"><h2>Fixed-token comparison gate: {gate}</h2>{blocker_html}</section>
<section id="depth-explorer"><script type="application/json" id="depth-chart-data">{chart_data}</script></section>
<noscript><p class="panel">The interactive explorer requires JavaScript. The measured tables and SVG traces below remain available.</p></noscript>
{_paired_sensor_section(report)}
<section class="panel"><h2>Summary by run label</h2><p>Means are per-trial averages. Aggregate metrics are available only when every run in that label completed with the exact requested token count and the relevant measurement is available in its stated unit.</p>{_summary_table(report)}</section>
<section class="panel"><h2>Sampled curves</h2><p>Each chart has its own unit axis. Colors identify decoder depth; trials at the same depth share a color.</p><div class="charts">{gpu_chart}{circuit_chart}{utilization_chart}</div></section>
<section class="panel"><h2>Run comparison (details)</h2><p>Actual token counts are compared with the requested count. Observed throughput uses actual tokens and elapsed wall time.</p><div class="table-wrap"><table><thead><tr><th>Run</th><th>Output tokens</th><th>Runtime</th><th>GPU board measurement</th><th>{_escape(meter_name)} circuit meter</th><th>GPU utilization</th><th>Model and output</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class="panel"><h2>Measurement scope and interpretation</h2><ul><li>GPU readings are board-level; the shared host may contribute concurrent load.</li><li>{_escape(meter_name)} circuit meter covers its measured circuit and may include other loads.</li><li>Each measurement reports its sample window and sample count where supplied; these limits affect comparability.</li><li>Weights are metadata: weight size is not a power measurement and cannot establish watts or energy.</li><li>Answer quality is unvalidated; nonsense accuracy is accepted for this experiment.</li><li>A passing fixed-token gate does not establish full-depth generation or quality equivalence.</li><li>Current-derived values stay in VA and VA·s. They are never relabeled as W or J; any active-energy estimate is shown separately with assumptions.</li></ul></section>
</main><script>{chart_js}</script></body></html>"""


def render_presentation(report: Any) -> str:
    """Render the same measured explorer as one offline, embeddable HTML file."""
    report = validate_report(report)
    meter_name = _circuit_meter_name(report)
    measurement_clock = _measurement_clock(report)
    ut61b_demo = _circuit_meter(report) == "ut61b"
    data, css, javascript = _depth_chart_assets(
        report, presentation_source="fluke" if ut61b_demo else None
    )
    reasons = _gate_reasons(report)
    gate = "blocked" if reasons else "eligible"
    blockers = "".join(f"<li>{_escape(reason)}</li>" for reason in reasons)
    gate_details = f"<ul>{blockers}</ul>" if blockers else ""
    campaign = _escape(report.get("campaign_id") or "Gemma 4 decoder-depth comparison")
    if ut61b_demo:
        measurement_scope = (
            f"Measurement clock: {_escape(measurement_clock)}. UT61B reports whole-circuit "
            "apparent power in VA and apparent-power integral in VA·s from AC current using "
            "the assumed grid voltage. These readings do not establish active watts or joules."
        )
        curve_notes = (
            "Request-window UT61B apparent-power samples; boundary values are linearly interpolated "
            "from bracketing samples. The interactive display uses bounded Catmull–Rom; VA·s "
            "integrates the original piecewise-linear samples. No extrapolation or clock-offset "
            "correction. Matching recorded windows does not remove sensor lag or averaging. "
            "Shared-host activity may contribute."
        )
    else:
        measurement_scope = (
            f"Measurement clock: {_escape(measurement_clock)}. NVIDIA: active GPU board power, W and J. "
            f"{_escape(meter_name)}: whole-circuit apparent power, VA and VA·s from AC current using "
            "the assumed grid voltage. These are different physical scopes and quantities; do not "
            "subtract them or infer GPU share or PSU efficiency."
        )
        curve_notes = (
            "Request-window source curves; boundary values are linearly interpolated from bracketing "
            "raw samples. The interactive display uses bounded Catmull–Rom; energy uses original "
            "piecewise-linear samples. No extrapolation, idle subtraction or clock-offset correction. "
            "Matching recorded windows does not remove sensor lag or averaging. Shared-host activity "
            "may contribute."
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Energy Audit — {campaign}</title><style>
:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#efeee6;color:#202b24;font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1320px;margin:auto;padding:clamp(10px,2vw,24px)}}.audit-context{{padding:12px 4px;font-size:12px;color:#58675d}}.audit-context p{{margin:4px 0}}.audit-context code{{overflow-wrap:anywhere}}.audit-context ul{{margin:6px 0}}
{css}
</style></head><body><main>
<section id="depth-explorer"><script type="application/json" id="depth-chart-data">{data}</script></section>
<noscript><p>The interactive graph requires JavaScript. Use the full report for static measured tables and SVG traces.</p></noscript>
<footer class="audit-context"><p><strong>Campaign: {campaign}. Fixed-token gate: {gate}.</strong> {report['requested_tokens']} generated tokens per request; answer quality is unvalidated.</p>{gate_details}
<p>{measurement_scope}</p>
<p>{curve_notes}</p>
<p>Prompt SHA-256: <code>{_escape(report.get('prompt_sha256') or 'Unavailable')}</code>. Imported report provenance requires the original experiment evidence.</p></footer>
</main><script>{javascript}</script></body></html>"""


def _reject_constant(value: str) -> None:
    raise ReportError(f"JSON contains non-finite value {value}")


def _load_report(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle, parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReportError(f"cannot read {path}: {error}") from error
    return validate_report(report)


def _merge_reports(paths: list[Path]) -> dict[str, Any]:
    reports = [_load_report(path) for path in paths]
    if not reports:
        raise ReportError("merge requires at least one input report")
    base = reports[0]
    generation_settings = (
        "ignore_eos",
        "cache_prompt",
        "temperature",
        "seed",
        "warmup_tokens",
    )
    base_meter = _circuit_meter(base)
    base_clock = _measurement_clock(base)
    for path, report in zip(paths[1:], reports[1:]):
        if _circuit_meter(report) != base_meter:
            raise ReportError(f"cannot merge {path}: circuit_meter differs between reports")
        if _measurement_clock(report) != base_clock:
            raise ReportError(f"cannot merge {path}: measurement_clock differs between reports")
        for key in ("requested_tokens", "prompt_sha256", "campaign_id"):
            left = base.get(key)
            right = report.get(key)
            if left != right:
                raise ReportError(f"cannot merge {path}: {key} differs between reports")
    for key in generation_settings:
        values = [candidate.get(key) for candidate in reports]
        is_present = [key in candidate for candidate in reports]
        if any(is_present) and (
            not all(is_present) or any(value != values[0] for value in values[1:])
        ):
            raise ReportError(
                f"cannot merge reports: generation setting {key} is missing or differs"
            )
    merged = dict(base)
    merged_runs: list[dict[str, Any]] = []
    for path, report in zip(paths, reports):
        for run in report["runs"]:
            copy = dict(run)
            copy["_source_report"] = path.name
            merged_runs.append(copy)
    merged["runs"] = merged_runs
    merged["full_depth"] = bool(merged_runs) and all(
        run["decoder_layers"] == 48 for run in merged_runs
    )
    run_model_hashes = [run.get("model_sha256") for run in merged_runs]
    root_model_sizes = [report.get("model_file_bytes") for report in reports]
    one_known_model = (
        bool(run_model_hashes)
        and all(model_hash is not None for model_hash in run_model_hashes)
        and len(set(run_model_hashes)) == 1
    )
    one_known_size = (
        bool(root_model_sizes)
        and all(size is not None for size in root_model_sizes)
        and len(set(root_model_sizes)) == 1
    )
    if one_known_model and one_known_size:
        merged["model_file_bytes"] = root_model_sizes[0]
    else:
        merged.pop("model_file_bytes", None)
    merged["source_reports"] = [path.name for path in paths]
    note_parts = []
    for report in reports:
        notes = report.get("notes")
        if notes and notes not in note_parts:
            note_parts.append(notes)
    if note_parts:
        merged["notes"] = " | ".join(note_parts)
    return validate_report(merged)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] not in ("render", "merge", "-h", "--help"):
        argv = ["render", *argv]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    single = subparsers.add_parser("render", help="render one JSON report")
    single.add_argument("report", type=Path)
    single.add_argument("--output", required=True, type=Path)
    single.add_argument("--presentation-only", action="store_true", help="write a compact offline Canvas page for a pitch or iframe")
    merge = subparsers.add_parser("merge", help="merge per-depth JSON reports")
    merge.add_argument("--input", required=True, nargs="+", type=Path)
    merge.add_argument("--output", required=True, type=Path)
    merge.add_argument("--presentation-only", action="store_true", help="write a compact offline Canvas page for a pitch or iframe")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        inputs = [args.report] if args.command == "render" else args.input
        output_resolved = args.output.resolve()
        for input_path in inputs:
            if input_path.resolve() == output_resolved:
                raise ReportError("output path must not replace an input report")
        report = _load_report(args.report) if args.command == "render" else _merge_reports(args.input)
        rendered = render_presentation(report) if args.presentation_only else render_html(report)
        args.output.write_text(rendered, encoding="utf-8")
    except (ReportError, OSError, argparse.ArgumentError) as error:
        print(f"gemma4_layer_report: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
