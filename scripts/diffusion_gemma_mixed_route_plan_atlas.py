#!/usr/bin/env python3
"""Rank DiffusionGemma mixed route plans into LTP/WBA optimization windows.

This is an offline attribution helper, not benchmark evidence. It consumes a
JSONL route plan emitted by diffusion_gemma_prompt_artifact_suite_gate.sh and,
when child logs are still present, folds in gate_metric phase rows from those
logs. The goal is to identify the recomputed mixed fast/exact bottleneck before
launching another heavy 26B run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


SUMMARY_KIND = "diffusion_gemma_mixed_route_plan_summary_v1"
WINDOW_KIND = "diffusion_gemma_mixed_route_plan_window_v1"


def fmt(value: float) -> str:
    if math.isnan(value):
        return "NA"
    if math.isinf(value):
        return "inf"
    return f"{value:.6f}"


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return float("nan")
    return float("nan")


def as_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool):
        raise ValueError
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError


def parse_kv_line(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in line.split()[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    return fields


def load_plan(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] | None = None
    windows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{lineno}: route-plan row must be an object")
            kind = row.get("kind")
            if kind == SUMMARY_KIND:
                if summary is not None:
                    raise SystemExit(f"{path}:{lineno}: duplicate summary row")
                summary = row
            elif kind == WINDOW_KIND:
                try:
                    key = (as_int(row, "prompt_token"), as_int(row, "canvas_token"))
                except (KeyError, ValueError) as exc:
                    raise SystemExit(f"{path}:{lineno}: window row requires integer prompt_token/canvas_token") from exc
                if key in seen:
                    raise SystemExit(f"{path}:{lineno}: duplicate window {key[0]}:{key[1]}")
                seen.add(key)
                if row.get("selected_route") not in {"variant_fast", "base_exact"}:
                    raise SystemExit(f"{path}:{lineno}: unsupported selected_route={row.get('selected_route')!r}")
                windows.append(row)
            else:
                raise SystemExit(f"{path}:{lineno}: unsupported route-plan row kind={kind!r}")

    if summary is None:
        raise SystemExit(f"{path}: missing {SUMMARY_KIND} row")
    expected = as_int(summary, "windows")
    if expected != len(windows):
        raise SystemExit(f"{path}: summary windows={expected} but found {len(windows)} window rows")
    return summary, windows


def parse_child_metrics(log_path: str) -> dict[tuple[str, str], dict[str, float]]:
    if not log_path or not os.path.isfile(log_path):
        return {}
    metrics: dict[tuple[str, str], dict[str, float]] = {}
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line.startswith("gate_metric "):
                continue
            fields = parse_kv_line(line)
            try:
                kind = fields["kind"]
                metric = fields["metric"]
                metrics[(kind, metric)] = {
                    "base_ms": float(fields["base_ms"]),
                    "variant_ms": float(fields["variant_ms"]),
                    "speedup": float(fields["speedup"]),
                    "delta_ms": float(fields["delta_ms"]),
                    "range_over_delta": float(fields.get("range_over_delta", "nan")),
                }
            except (KeyError, ValueError) as exc:
                raise SystemExit(f"{log_path}: malformed gate_metric row: {line}") from exc
    return metrics


def aggregate_phase_rows(windows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    totals: dict[tuple[str, str], dict[str, float]] = {}
    for window in windows:
        for key, metric in parse_child_metrics(str(window.get("child_log", ""))).items():
            row = totals.setdefault(
                key,
                {"base_ms": 0.0, "variant_ms": 0.0, "delta_ms": 0.0, "range_over_delta": 0.0, "windows": 0.0},
            )
            row["base_ms"] += metric["base_ms"]
            row["variant_ms"] += metric["variant_ms"]
            row["delta_ms"] += metric["delta_ms"]
            row["windows"] += 1.0
            if math.isfinite(metric["range_over_delta"]):
                row["range_over_delta"] = max(row["range_over_delta"], metric["range_over_delta"])
    for row in totals.values():
        row["speedup"] = row["base_ms"] / row["variant_ms"] if row["variant_ms"] > 0 else float("inf")
    return totals


def selected_window_cost(window: dict[str, Any]) -> float:
    return as_float(window, "mixed_variant_ms")


def unsafe_saved_ms(window: dict[str, Any]) -> float:
    return as_float(window, "base_ms") - as_float(window, "observed_variant_ms")


def mixed_saved_ms(window: dict[str, Any]) -> float:
    return as_float(window, "base_ms") - as_float(window, "mixed_variant_ms")


def route_label(window: dict[str, Any]) -> str:
    return f"{window.get('prompt_token')}:{window.get('canvas_token')}"


def print_text(path: Path, summary: dict[str, Any], windows: list[dict[str, Any]], top: int) -> None:
    mixed_total = as_float(summary, "mixed_variant_ms")
    fallback_count = sum(1 for row in windows if row.get("selected_route") == "base_exact")
    fallback_ms = sum(selected_window_cost(row) for row in windows if row.get("selected_route") == "base_exact")
    dominant = max(windows, key=selected_window_cost)
    dominant_ms = selected_window_cost(dominant)
    tied = sum(1 for row in windows if dominant_ms > 0 and selected_window_cost(row) >= dominant_ms * 0.80)
    phases = aggregate_phase_rows(windows)
    noisy_phases = sum(
        1 for row in phases.values()
        if row["range_over_delta"] > 2.0 and abs(row["delta_ms"]) > 0.0
    )
    conflict_count = fallback_count + noisy_phases

    print("DiffusionGemma mixed route plan atlas")
    print(f"  source={path}")
    print(
        "  summary decision=%s windows=%s candidate=%s fallback=%s unsafe_speedup=%s mixed_speedup=%s"
        % (
            summary.get("decision", "NA"),
            summary.get("windows", "NA"),
            summary.get("candidate_windows", "NA"),
            summary.get("fallback_windows", "NA"),
            fmt(as_float(summary, "unsafe_speedup")),
            fmt(as_float(summary, "mixed_speedup")),
        )
    )
    print(
        "  Phi=(dominant_window=%s, tied_dominant_windows=%d, conflict_or_sync_count=%d, remaining_mixed_ms=%s)"
        % (route_label(dominant), tied, conflict_count, fmt(mixed_total))
    )
    fallback_pct = fallback_ms * 100.0 / mixed_total if mixed_total > 0 else float("nan")
    print(
        "  exact_fallback_ms=%s pct_of_mixed=%s%%"
        % (fmt(fallback_ms), fmt(fallback_pct))
    )

    if fallback_count:
        print(
            "  LTP/WBA: Diamond first. The mixed corridor is dominated by exact fallback; fix the certificate/fallback boundary before micro-tuning accepted fast windows."
        )
    else:
        print(
            "  LTP/WBA: Ladder. No exact fallback remains; rank accepted fast windows by mixed wall and phase deltas before the next kernel move."
        )
    print(
        "  Dual frame: keep base_exact for non-certified windows; promote only after quiet ABBA recomputes the same mixed Phi."
    )

    print("  windows:")
    for row in sorted(windows, key=selected_window_cost, reverse=True)[:top]:
        pct = selected_window_cost(row) * 100.0 / mixed_total if mixed_total > 0 else float("nan")
        print(
            "    %s route=%s status=%s mixed_ms=%s pct=%s%% observed_speedup=%s mixed_speedup=%s unsafe_saved_ms=%s mixed_saved_ms=%s"
            % (
                route_label(row),
                row.get("selected_route", "NA"),
                row.get("status", "NA"),
                fmt(selected_window_cost(row)),
                fmt(pct),
                fmt(as_float(row, "observed_speedup")),
                fmt(as_float(row, "mixed_speedup")),
                fmt(unsafe_saved_ms(row)),
                fmt(mixed_saved_ms(row)),
            )
        )
    if phases:
        print("  phase_deltas:")
        for (kind, metric), row in sorted(phases.items(), key=lambda item: (-abs(item[1]["delta_ms"]), item[0]))[:top]:
            print(
                "    kind=%s metric=%s windows=%d base_ms=%s variant_ms=%s speedup=%s delta_ms=%s max_range_over_delta=%s"
                % (
                    kind,
                    metric,
                    int(row["windows"]),
                    fmt(row["base_ms"]),
                    fmt(row["variant_ms"]),
                    fmt(row["speedup"]),
                    fmt(row["delta_ms"]),
                    fmt(row["range_over_delta"]),
                )
            )


def print_tsv(path: Path, summary: dict[str, Any], windows: list[dict[str, Any]]) -> None:
    fields = [
        "kind",
        "source",
        "index",
        "prompt_token",
        "canvas_token",
        "selected_route",
        "status",
        "reason",
        "base_ms",
        "observed_variant_ms",
        "mixed_variant_ms",
        "observed_speedup",
        "mixed_speedup",
        "unsafe_saved_ms",
        "mixed_saved_ms",
        "child_log",
    ]
    print("\t".join(fields))
    for row in sorted(windows, key=selected_window_cost, reverse=True):
        values = {
            "kind": "window",
            "source": str(path),
            "index": row.get("index", ""),
            "prompt_token": row.get("prompt_token", ""),
            "canvas_token": row.get("canvas_token", ""),
            "selected_route": row.get("selected_route", ""),
            "status": row.get("status", ""),
            "reason": row.get("reason", ""),
            "base_ms": fmt(as_float(row, "base_ms")),
            "observed_variant_ms": fmt(as_float(row, "observed_variant_ms")),
            "mixed_variant_ms": fmt(as_float(row, "mixed_variant_ms")),
            "observed_speedup": fmt(as_float(row, "observed_speedup")),
            "mixed_speedup": fmt(as_float(row, "mixed_speedup")),
            "unsafe_saved_ms": fmt(unsafe_saved_ms(row)),
            "mixed_saved_ms": fmt(mixed_saved_ms(row)),
            "child_log": row.get("child_log", ""),
        }
        print("\t".join(str(values[field]) for field in fields))

    phase_fields = ["kind", "source", "phase_kind", "metric", "windows", "base_ms", "variant_ms", "speedup", "delta_ms", "range_over_delta"]
    print("\t".join(phase_fields))
    for (kind, metric), row in sorted(aggregate_phase_rows(windows).items(), key=lambda item: (-abs(item[1]["delta_ms"]), item[0])):
        values = {
            "kind": "phase",
            "source": str(path),
            "phase_kind": kind,
            "metric": metric,
            "windows": int(row["windows"]),
            "base_ms": fmt(row["base_ms"]),
            "variant_ms": fmt(row["variant_ms"]),
            "speedup": fmt(row["speedup"]),
            "delta_ms": fmt(row["delta_ms"]),
            "range_over_delta": fmt(row["range_over_delta"]),
        }
        print("\t".join(str(values[field]) for field in phase_fields))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("route_plan", type=Path, help="mixed route plan JSONL")
    parser.add_argument("--top", type=int, default=8, help="number of windows/phase rows to print")
    parser.add_argument("--tsv", action="store_true", help="emit machine-readable TSV tables")
    args = parser.parse_args()

    summary, windows = load_plan(args.route_plan)
    if args.tsv:
        print_tsv(args.route_plan, summary, windows)
    else:
        print_text(args.route_plan, summary, windows, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
