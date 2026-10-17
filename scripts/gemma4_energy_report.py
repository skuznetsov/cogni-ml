#!/usr/bin/env python3
"""Render a cogni.gemma4.energy.v1 JSON report as standalone HTML."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import sys
from typing import Any


SCHEMA = "cogni.gemma4.energy.v1"
PROFILE_COLORS = {
    "eco": "#168477",
    "balanced": "#c47c13",
    "quality": "#6952a3",
}
FALLBACK_COLORS = ("#2878b5", "#bf4b62", "#547a35", "#80533e", "#477b87")


class ReportError(ValueError):
    """Raised when a JSON document does not match the supported report shape."""


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _optional_type(
    mapping: dict[str, Any], key: str, expected: type | tuple[type, ...], location: str
) -> Any:
    value = mapping.get(key)
    if value is not None and not isinstance(value, expected):
        expected_name = (
            " or ".join(item.__name__ for item in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise ReportError(f"{location}.{key} must be {expected_name} or null")
    return value


def _number(value: Any, location: str, *, integer: bool = False) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{location} must be a finite number or null")
    if not math.isfinite(value):
        raise ReportError(f"{location} must be a finite number or null")
    if value < 0:
        raise ReportError(f"{location} must not be negative")
    if integer and not isinstance(value, int):
        raise ReportError(f"{location} must be an integer or null")
    return value


def validate_report(report: Any) -> dict[str, Any]:
    """Validate the report envelope and the nested shapes used by this view."""
    if not isinstance(report, dict):
        raise ReportError("report root must be an object")
    if report.get("schema") != SCHEMA:
        raise ReportError(f"schema must be {SCHEMA!r}")

    for key in ("model_identity", "prompt_sha256", "model_sha256", "notes"):
        _optional_type(report, key, str, "report")
    _optional_type(report, "full_depth", bool, "report")

    runs = report.get("runs")
    if not isinstance(runs, list):
        raise ReportError("report.runs must be an array")
    for index, run in enumerate(runs):
        location = f"runs[{index}]"
        if not isinstance(run, dict):
            raise ReportError(f"{location} must be an object")
        for key in ("answer", "status", "error"):
            _optional_type(run, key, str, location)
        _number(run.get("round"), f"{location}.round", integer=True)
        _number(run.get("elapsed_ms"), f"{location}.elapsed_ms")
        _number(run.get("started_at_epoch_ms"), f"{location}.started_at_epoch_ms")
        _number(run.get("ended_at_epoch_ms"), f"{location}.ended_at_epoch_ms")

        profile = _optional_type(run, "profile", dict, location)
        if profile is not None:
            _optional_type(profile, "name", str, f"{location}.profile")
            _number(
                profile.get("reasoning_tokens"),
                f"{location}.profile.reasoning_tokens",
                integer=True,
            )
            _number(
                profile.get("answer_tokens"),
                f"{location}.profile.answer_tokens",
                integer=True,
            )

        phases = _optional_type(run, "phases", dict, location)
        if phases is not None:
            for phase_name in ("reasoning", "answer"):
                phase = phases.get(phase_name)
                if phase is None:
                    continue
                if not isinstance(phase, dict):
                    raise ReportError(f"{location}.phases.{phase_name} must be an object or null")
                phase_location = f"{location}.phases.{phase_name}"
                _number(phase.get("predicted_tokens"), f"{phase_location}.predicted_tokens", integer=True)
                _optional_type(phase, "outcome", str, phase_location)
                _optional_type(phase, "forced_close", bool, phase_location)
                _optional_type(phase, "truncated", bool, phase_location)

        measurement = _optional_type(run, "measurement", dict, location)
        if measurement is not None:
            _optional_type(measurement, "status", str, f"{location}.measurement")
            _optional_type(measurement, "kind", str, f"{location}.measurement")
            _optional_type(measurement, "power_unit", str, f"{location}.measurement")
            curve = measurement.get("curve")
            if curve is not None and not isinstance(curve, list):
                raise ReportError(f"{location}.measurement.curve must be an array or null")
            if isinstance(curve, list):
                for point_index, point in enumerate(curve):
                    point_location = f"{location}.measurement.curve[{point_index}]"
                    if not isinstance(point, dict):
                        raise ReportError(f"{point_location} must be an object")
                    _number(point.get("elapsed_ms"), f"{point_location}.elapsed_ms")
                    _number(point.get("value"), f"{point_location}.value")
            for field in (
                "duration_seconds",
                "mean_apparent_power_va",
                "apparent_energy_va_seconds",
                "mean_active_power_watts",
                "active_energy_joules",
                "estimated_active_energy_joules",
                "assumed_voltage_volts",
                "assumed_power_factor",
            ):
                _number(measurement.get(field), f"{location}.measurement.{field}")
            _number(measurement.get("sample_count"), f"{location}.measurement.sample_count", integer=True)
    return report


def _available(value: Any, suffix: str = "") -> str:
    if value is None:
        return "Unavailable"
    return _escape(value) + suffix


def _format_number(value: Any) -> str:
    number = _number(value, "display value")
    if number is None:
        return "Unavailable"
    return format(number, ".4g")


def _token_count(value: Any) -> str:
    if value is None:
        return "Unavailable"
    return _escape(value)


def _phase_cell(profile: dict[str, Any], phase: dict[str, Any] | None, phase_name: str) -> str:
    budget_key = "reasoning_tokens" if phase_name == "Reasoning" else "answer_tokens"
    budget = _token_count(profile.get(budget_key))
    if phase is None:
        forced_close = "<div>Forced close: Unavailable</div>" if phase_name == "Reasoning" else ""
        return (
            f"<div><strong>{phase_name}:</strong> unavailable</div>"
            f"<div>Budget / actual: {budget} / Unavailable</div>"
            "<div>Outcome: Unavailable</div>"
            f"{forced_close}<div>Truncated: Unavailable</div>"
        )

    actual = _token_count(phase.get("predicted_tokens"))
    outcome = _available(phase.get("outcome"))
    truncated = phase.get("truncated")
    truncation_text = "truncated" if truncated is True else "not truncated" if truncated is False else "truncation unavailable"
    close_line = ""
    if phase_name == "Reasoning":
        forced_close = phase.get("forced_close")
        close_text = "forced close" if forced_close is True else "not forced closed" if forced_close is False else "forced close unavailable"
        close_line = f"<div>Forced close: {_escape(close_text)}</div>"
    return (
        f"<div><strong>{phase_name}:</strong> budget / actual: {budget} / {actual}</div>"
        f"<div>Outcome: {outcome}</div>"
        f"{close_line}<div>Truncated: {_escape(truncation_text)}</div>"
    )


def _measurement_cell(measurement: dict[str, Any] | None) -> str:
    if measurement is None:
        return '<span class="unavailable">Unavailable</span>'
    status = measurement.get("status")
    if status != "available":
        reason = measurement.get("reason")
        if status == "unavailable":
            suffix = f": {_escape(reason)}" if reason is not None else ""
            return f'<span class="unavailable">Unavailable{suffix}</span>'
        return '<span class="unavailable">Unavailable (measurement status missing)</span>'

    kind = measurement.get("kind")
    unit = measurement.get("power_unit")
    if (kind, unit) not in (("current_ac", "VA"), ("active_power", "W")):
        return '<span class="unavailable">Unavailable (measurement kind and unit are missing or inconsistent)</span>'

    lines: list[str] = []
    if kind == "current_ac":
        apparent_energy = measurement.get("apparent_energy_va_seconds")
        mean_power = measurement.get("mean_apparent_power_va")
        lines.append(
            "Apparent energy: "
            + (f"{_format_number(apparent_energy)} VA·s" if apparent_energy is not None else "Unavailable")
        )
        lines.append(
            "Mean apparent power: "
            + (f"{_format_number(mean_power)} VA" if mean_power is not None else "Unavailable")
        )
        estimated_energy = measurement.get("estimated_active_energy_joules")
        if estimated_energy is not None:
            voltage = measurement.get("assumed_voltage_volts")
            factor = measurement.get("assumed_power_factor")
            voltage_text = f"{_format_number(voltage)} V" if voltage is not None else "voltage Unavailable"
            factor_text = f"PF {_format_number(factor)}" if factor is not None else "PF Unavailable"
            lines.append(
                "Estimated active energy (assumed " + voltage_text + ", " + factor_text + "): "
                + f"{_format_number(estimated_energy)} J"
            )
        else:
            lines.append("Estimated active energy: Unavailable")
    else:
        active_energy = measurement.get("active_energy_joules")
        mean_power = measurement.get("mean_active_power_watts")
        lines.append(
            "Active energy: "
            + (f"{_format_number(active_energy)} J" if active_energy is not None else "Unavailable")
        )
        lines.append(
            "Mean active power: "
            + (f"{_format_number(mean_power)} W" if mean_power is not None else "Unavailable")
        )

    sample_count = measurement.get("sample_count")
    lines.append(f"Samples: {_escape(sample_count) if sample_count is not None else 'Unavailable'}")
    return "<br>".join(_escape(line) for line in lines)


def _usable_series(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    series: list[dict[str, Any]] = []
    units: set[str] = set()
    for index, run in enumerate(runs):
        measurement = run.get("measurement")
        if not isinstance(measurement, dict) or measurement.get("status") != "available":
            continue
        kind = measurement.get("kind")
        unit = measurement.get("power_unit")
        if (kind, unit) not in (("current_ac", "VA"), ("active_power", "W")):
            continue
        curve = measurement.get("curve")
        if not isinstance(curve, list):
            continue
        points = []
        for point in curve:
            if not isinstance(point, dict):
                continue
            x = point.get("elapsed_ms")
            y = point.get("value")
            if (
                isinstance(x, (int, float)) and not isinstance(x, bool)
                and isinstance(y, (int, float)) and not isinstance(y, bool)
                and math.isfinite(x) and math.isfinite(y) and x >= 0 and y >= 0
            ):
                points.append((float(x), float(y)))
        if not points:
            continue
        profile = run.get("profile")
        profile = profile if isinstance(profile, dict) else {}
        name = profile.get("name")
        if not isinstance(name, str) or not name:
            name = "Unknown profile"
        round_number = run.get("round")
        series.append({"index": index, "name": name, "round": round_number, "unit": unit, "points": points})
        units.add(unit)
    return series, units


def _chart(series: list[dict[str, Any]], unit: str) -> str:
    width, height = 840, 330
    left, right, top, bottom = 76, 22, 28, 52
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_x = [x for item in series for x, _ in item["points"]]
    all_y = [y for item in series for _, y in item["points"]]
    x_max = max(all_x) if all_x else 1.0
    x_max = x_max if x_max > 0 else 1.0
    y_max = max(all_y) if all_y else 1.0
    y_max = y_max * 1.08 if y_max > 0 else 1.0

    def px(x: float) -> float:
        return left + (x / x_max) * plot_width

    def py(y: float) -> float:
        return top + plot_height - (y / y_max) * plot_height

    colors: dict[str, str] = {}
    fallback_index = 0
    for item in series:
        name = item["name"]
        if name not in colors:
            known = PROFILE_COLORS.get(name.casefold())
            if known is not None:
                colors[name] = known
            else:
                colors[name] = FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)]
                fallback_index += 1

    label = "Apparent power (VA)" if unit == "VA" else "Active power (W)"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label} over elapsed time (ms)" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:system-ui,-apple-system,sans-serif;font-size:12px;fill:#334155}.grid{stroke:#dbe3eb;stroke-width:1}.axis{stroke:#64748b;stroke-width:1.2}</style>',
    ]

    # Shared elapsed-time axis and three readable power ticks.
    for fraction in (0.0, 0.5, 1.0):
        y = top + plot_height * (1.0 - fraction)
        value = y_max * fraction
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text x="{left-9}" y="{y+4:.2f}" text-anchor="end">{_escape(_format_number(value))} {_escape(unit)}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}"/>')
    parts.append(f'<text x="{left}" y="{height-23}" text-anchor="start">0</text>')
    parts.append(f'<text x="{width-right}" y="{height-23}" text-anchor="end">{_escape(_format_number(x_max))}</text>')
    parts.append(f'<text x="{left + plot_width/2:.2f}" y="{height-5}" text-anchor="middle">Elapsed time (ms)</text>')

    for item in series:
        color = colors[item["name"]]
        points = sorted(item["points"], key=lambda point: point[0])
        coordinates = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in points)
        round_label = f"round {item['round']}" if item["round"] is not None else "round unavailable"
        title = f"{item['name']} — {round_label}"
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{coordinates}"><title>{_escape(title)}</title></polyline>')
        for x, y in points:
            parts.append(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="3" fill="{color}"/>')

    legend_y = 17
    for name, color in colors.items():
        parts.append(f'<line x1="{left}" y1="{legend_y-4}" x2="{left+18}" y2="{legend_y-4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{left+24}" y="{legend_y}">{_escape(name)}</text>')
        left += max(100, len(name) * 8 + 60)
    parts.append("</svg>")
    return "".join(parts)


def render_html(report: Any) -> str:
    """Return a complete HTML document with an inline SVG comparison chart."""
    data = validate_report(report)
    runs = data["runs"]
    rows: list[str] = []
    for index, run in enumerate(runs):
        profile = run.get("profile") if isinstance(run.get("profile"), dict) else {}
        phases = run.get("phases") if isinstance(run.get("phases"), dict) else {}
        profile_name = profile.get("name")
        if not isinstance(profile_name, str) or not profile_name:
            profile_name = "Unavailable"
        round_number = run.get("round")
        round_text = _available(round_number)
        run_label = f"{profile_name} · round {round_text}"
        status = run.get("status")
        error = run.get("error")
        if error is not None:
            state_label = "Failed" if status == "failed" else "Error"
            state_text = state_label + f": {_escape(error)}"
        elif status == "failed":
            state_text = "Failed"
        else:
            state_text = "Status: " + _available(status)
        answer = run.get("answer")
        if answer is None:
            answer_html = '<span class="unavailable">Answer unavailable</span>'
        else:
            answer_html = f"<details><summary>Show answer</summary><pre>{_escape(answer)}</pre></details>"
        rows.append(
            "<tr>"
            f"<th scope=\"row\"><span class=\"profile profile-{_escape(profile_name.casefold())}\">{_escape(run_label)}</span><div class=\"run-state\">{state_text}</div></th>"
            f"<td>{_phase_cell(profile, phases.get('reasoning'), 'Reasoning')}</td>"
            f"<td>{_phase_cell(profile, phases.get('answer'), 'Answer')}</td>"
            f"<td>{_available(run.get('elapsed_ms'), ' ms')}</td>"
            f"<td>{_measurement_cell(run.get('measurement') if isinstance(run.get('measurement'), dict) else None)}</td>"
            f"<td>{answer_html}</td>"
            "</tr>"
        )

    series, units = _usable_series(runs)
    if len(units) > 1:
        chart_html = '<p class="chart-unavailable">Chart omitted because available curves mix VA and W. Use one measurement kind for a shared power axis.</p>'
    elif series:
        unit = next(iter(units))
        chart_html = _chart(series, unit)
    else:
        chart_html = '<p class="chart-unavailable">Chart unavailable: no valid measurement curves.</p>'

    full_depth = data.get("full_depth")
    full_depth_text = "yes" if full_depth is True else "no" if full_depth is False else "Unavailable"
    metadata = [
        ("Model", _available(data.get("model_identity"))),
        ("Model SHA-256", _available(data.get("model_sha256"))),
        ("Prompt SHA-256", _available(data.get("prompt_sha256"))),
        ("Full depth", full_depth_text),
    ]
    metadata_html = "".join(
        f'<div class="meta-item"><dt>{_escape(label)}</dt><dd>{value}</dd></div>'
        for label, value in metadata
    )
    notes = data.get("notes")
    notes_html = (
        f'<p class="notes">{_escape(notes)}</p>'
        if notes is not None
        else '<p class="notes unavailable">Notes: Unavailable</p>'
    )
    row_html = "\n".join(rows) if rows else '<tr><td colspan="6" class="unavailable">No runs</td></tr>'
    unit_notes = []
    if "VA" in units:
        unit_notes.append("VA·s is apparent energy.")
    if "W" in units:
        unit_notes.append("W is active power.")
    unit_notes.extend(("J is active energy.", "Estimates are labelled with their assumptions.", "Missing or null measurements are shown as unavailable."))
    footnote = " ".join(unit_notes)
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemma 4 Energy comparison</title>
<style>
:root{{color-scheme:light;--ink:#17212b;--muted:#5d6b79;--line:#d8e0e8;--panel:#f5f8fb;--eco:#168477;--balanced:#c47c13;--quality:#6952a3}}
*{{box-sizing:border-box}}body{{margin:0;padding:28px;font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink);background:#fff}}
main{{max-width:1500px;margin:0 auto}}h1{{margin:0 0 8px;font-size:1.8rem}}h2{{margin:28px 0 12px;font-size:1.15rem}}.intro{{color:var(--muted);margin:0 0 20px}}
dl.metadata{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:18px 0;padding:0}}.meta-item{{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:var(--panel);min-width:0}}dt{{font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}}dd{{margin:4px 0 0;overflow-wrap:anywhere}}.notes{{padding:12px 14px;border-left:3px solid #7b91a6;background:var(--panel);white-space:pre-wrap;overflow-wrap:anywhere}}
.chart-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:10px;padding:14px;background:white}}svg{{display:block;width:100%;min-width:600px;max-height:420px}}.chart-unavailable{{margin:0;padding:14px;color:var(--muted);background:var(--panel);border-radius:8px}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:10px}}table{{width:100%;border-collapse:collapse;min-width:1120px}}th,td{{padding:12px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}thead th{{background:#eef3f7;font-size:.78rem;text-transform:uppercase;letter-spacing:.035em;color:#455463}}tbody tr:last-child>*{{border-bottom:0}}tbody th{{min-width:150px;font-weight:600}}td{{min-width:145px}}.profile{{display:inline-block;padding:3px 8px;border-radius:999px;background:#e9eef3;color:#263746}}.profile-eco{{background:#dff3ed;color:#0a655b}}.profile-balanced{{background:#fff0d8;color:#855208}}.profile-quality{{background:#eee8fa;color:#58428d}}.run-state{{margin-top:6px;font-size:.82rem;font-weight:400;color:var(--muted);overflow-wrap:anywhere}}.unavailable{{color:#687786}}details summary{{cursor:pointer;color:#245b8b}}pre{{max-width:420px;white-space:pre-wrap;overflow-wrap:anywhere;padding:10px;background:#f2f5f8;border-radius:6px;margin:8px 0 0;font:inherit}}.footnote{{margin-top:18px;color:var(--muted);font-size:.85rem}}
@media(max-width:700px){{body{{padding:16px}}h1{{font-size:1.45rem}}}}
</style>
</head>
<body>
<main>
<h1>Gemma 4 Energy comparison</h1>
<p class="intro">Per-run reasoning and answer budgets, observed token counts, completion outcomes, elapsed time, and VictoriaMetrics measurements.</p>
<dl class="metadata">{metadata_html}</dl>
<h2>Power over elapsed time</h2>
<div class="chart-wrap">{chart_html}</div>
<h2>Run comparison</h2>
<div class="table-wrap"><table>
<thead><tr><th scope="col">Profile / round</th><th scope="col">Reasoning phase</th><th scope="col">Answer phase</th><th scope="col">Elapsed</th><th scope="col">Energy and power</th><th scope="col">Answer detail</th></tr></thead>
<tbody>{row_html}</tbody>
</table></div>
{notes_html}
<p class="footnote">{_escape(footnote)}</p>
</main>
</body>
</html>
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="input cogni.gemma4.energy.v1 JSON report")
    parser.add_argument("--output", "-o", help="write HTML to PATH (default: stdout; '-' also means stdout)")
    args = parser.parse_args(argv)

    input_path = Path(args.report)
    try:
        report = json.loads(input_path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
        rendered = render_html(report)
        if args.output and args.output != "-":
            output_path = Path(args.output)
            if output_path.resolve() == input_path.resolve():
                raise ReportError("output path must differ from the input JSON path")
            output_path.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
            sys.stdout.write("\n")
    except (OSError, json.JSONDecodeError, ReportError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def _reject_json_constant(value: str) -> None:
    raise ReportError(f"invalid JSON numeric constant: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
