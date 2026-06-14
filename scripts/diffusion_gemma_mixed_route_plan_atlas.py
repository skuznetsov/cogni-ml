#!/usr/bin/env python3
"""Rank DiffusionGemma mixed route plans into LTP/WBA optimization windows.

This is an offline attribution helper, not benchmark evidence. It consumes a
JSONL route plan emitted by diffusion_gemma_prompt_artifact_suite_gate.sh and,
when child logs are still present, folds in gate_metric phase rows from those
logs. The goal is to identify the recomputed mixed fast/exact bottleneck before
launching another heavy 26B run.

For exact fallback windows, the helper also reads the child output-cert log when
available and prints a cert-derived dual-cache canvas-band fallback estimate.
That estimate is a cross-probe guard: route-plan selected costs come from ABBA
timing, while output-cert rows include certificate decode timing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from diffusion_gemma_output_cert_atlas import (
    dual_cache_band_break_even_uses,
    dual_cache_band_fallback_ms,
    group_rows,
    read_cert,
    timing_by_window,
    truthy,
)


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


def parse_child_paths(log_path: str) -> dict[str, str]:
    if not log_path or not os.path.isfile(log_path):
        return {}
    result: dict[str, str] = {}
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key in {"output_cert_log", "prompt_cache_abba_log"}:
                result[key] = value
    return result


def aggregate_phase_rows(windows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    totals: dict[tuple[str, str], dict[str, float]] = {}
    for window in windows:
        for key, metric in parse_child_metrics(str(window.get("child_log", ""))).items():
            selected_route = str(window.get("selected_route", ""))
            mixed_variant_ms = metric["variant_ms"] if selected_route == "variant_fast" else metric["base_ms"]
            row = totals.setdefault(
                key,
                {
                    "base_ms": 0.0,
                    "unsafe_variant_ms": 0.0,
                    "mixed_variant_ms": 0.0,
                    "unsafe_delta_ms": 0.0,
                    "mixed_delta_ms": 0.0,
                    "range_over_delta": 0.0,
                    "windows": 0.0,
                    "candidate_windows": 0.0,
                    "fallback_windows": 0.0,
                },
            )
            row["base_ms"] += metric["base_ms"]
            row["unsafe_variant_ms"] += metric["variant_ms"]
            row["mixed_variant_ms"] += mixed_variant_ms
            row["unsafe_delta_ms"] += metric["base_ms"] - metric["variant_ms"]
            row["mixed_delta_ms"] += metric["base_ms"] - mixed_variant_ms
            row["windows"] += 1.0
            if selected_route == "variant_fast":
                row["candidate_windows"] += 1.0
            elif selected_route == "base_exact":
                row["fallback_windows"] += 1.0
            if math.isfinite(metric["range_over_delta"]):
                row["range_over_delta"] = max(row["range_over_delta"], metric["range_over_delta"])
    for row in totals.values():
        row["unsafe_speedup"] = row["base_ms"] / row["unsafe_variant_ms"] if row["unsafe_variant_ms"] > 0 else float("inf")
        row["mixed_speedup"] = row["base_ms"] / row["mixed_variant_ms"] if row["mixed_variant_ms"] > 0 else float("inf")
    return totals


def selected_window_cost(window: dict[str, Any]) -> float:
    return as_float(window, "mixed_variant_ms")


def unsafe_saved_ms(window: dict[str, Any]) -> float:
    return as_float(window, "base_ms") - as_float(window, "observed_variant_ms")


def mixed_saved_ms(window: dict[str, Any]) -> float:
    return as_float(window, "base_ms") - as_float(window, "mixed_variant_ms")


def route_label(window: dict[str, Any]) -> str:
    return f"{window.get('prompt_token')}:{window.get('canvas_token')}"


def route_owner(window: dict[str, Any]) -> str:
    selected = str(window.get("selected_route", ""))
    if selected == "base_exact":
        return "exact_base_fallback"
    artifact_arm = str(window.get("selected_route_artifact_arm", "variant") or "variant")
    artifact_env = str(window.get("selected_route_artifact_env_role", "variant") or "variant")
    variant_env = str(window.get("variant_env_role", "variant") or "variant")
    if artifact_arm == "base" or artifact_env == "base":
        return f"foreign_{artifact_arm}_artifact_{variant_env}_env"
    return "certified_variant_replay"


def phase_owner(metric: str, row: dict[str, float]) -> str:
    base_ms = row["base_ms"]
    unsafe_variant_ms = row["unsafe_variant_ms"]
    mixed_variant_ms = row["mixed_variant_ms"]
    if base_ms > 0.0 and mixed_variant_ms == 0.0:
        return "eliminated_or_replayed"
    if base_ms > 0.0 and unsafe_variant_ms == 0.0 and mixed_variant_ms > 0.0:
        return "mixed_fallback_residual"
    if metric == "total_ms":
        return "end_to_end_mixed_route"
    if metric.startswith("materialize_moe_"):
        return "moe_materialize_transport"
    if metric.startswith("materialize"):
        return "residual_materialize_boundary"
    if "context" in metric:
        return "attention_context_boundary"
    if "qkv" in metric or "projection" in metric:
        return "projection_boundary"
    return "unknown_phase_boundary"


def phase_next_move(owner: str) -> str:
    if owner == "eliminated_or_replayed":
        return "do_not_microtune; protect certificate/replay boundary"
    if owner == "mixed_fallback_residual":
        return "fallback/certificate Diamond before subphase tuning"
    if owner == "end_to_end_mixed_route":
        return "reduce dominant fallback or recompute promoted mixed route"
    if owner == "moe_materialize_transport":
        return "fuse_or_keep_resident_across_grouped_moe"
    if owner == "residual_materialize_boundary":
        return "remove_cpu_materialization_or_fuse_tail_corridor"
    if owner == "attention_context_boundary":
        return "keep_q_context_output_resident_or_batch_rows"
    if owner == "projection_boundary":
        return "resident_projection_or_batch_matmul"
    return "measure_before_tuning"


def cert_fallback_alternative(
    window: dict[str, Any],
    reuse_count: int,
    min_speedup: float,
) -> dict[str, Any]:
    paths = parse_child_paths(str(window.get("child_log", "")))
    cert_path_raw = paths.get("output_cert_log", "")
    result: dict[str, Any] = {
        "status": "missing_output_cert",
        "output_cert_log": cert_path_raw,
    }
    if not cert_path_raw or not os.path.isfile(cert_path_raw):
        return result

    cert_path = Path(cert_path_raw)
    rows, timing_rows, _ = read_cert(cert_path)
    key = (str(window.get("prompt_token")), str(window.get("canvas_token")))
    grouped = group_rows(rows)
    window_rows = grouped.get(key)
    if not window_rows:
        result["status"] = "missing_window"
        return result
    timings = timing_by_window(timing_rows)
    timing = timings.get(key, {})
    argmax_failures = sum(1 for row in window_rows if not truthy(row.get("argmax_match", "")))
    sampled_failures = sum(1 for row in window_rows if not truthy(row.get("sampled_match", "")))
    base_total = as_float(timing, "base_cache_ms") + as_float(timing, "base_predict_ms")
    variant_total = as_float(timing, "variant_cache_ms") + as_float(timing, "variant_predict_ms")
    band_ms = dual_cache_band_fallback_ms(timing, len(window_rows), argmax_failures, reuse_count)
    band_speedup = base_total / band_ms if band_ms > 0.0 else float("inf")
    break_even_status, break_even_uses = dual_cache_band_break_even_uses(
        timing,
        len(window_rows),
        argmax_failures,
        min_speedup,
    )
    selected_ms = selected_window_cost(window)
    vs_selected = selected_ms / band_ms if band_ms > 0.0 else float("inf")
    cert_vs_selected = base_total / selected_ms if selected_ms > 0.0 else float("inf")
    return {
        "status": "ok",
        "output_cert_log": cert_path_raw,
        "rows": len(window_rows),
        "argmax_failures": argmax_failures,
        "sampled_failures": sampled_failures,
        "cert_base_total_ms": base_total,
        "cert_variant_total_ms": variant_total,
        "dual_cache_band_ms": band_ms,
        "dual_cache_band_speedup_vs_cert_base": band_speedup,
        "dual_cache_band_vs_selected_route": vs_selected,
        "dual_cache_band_delta_vs_selected_ms": selected_ms - band_ms,
        "dual_cache_band_break_even_status": break_even_status,
        "dual_cache_band_break_even_uses": break_even_uses,
        "cert_base_total_vs_selected_route": cert_vs_selected,
    }


def print_text(
    path: Path,
    summary: dict[str, Any],
    windows: list[dict[str, Any]],
    top: int,
    fallback_reuse_count: int,
    fallback_min_speedup: float,
) -> None:
    mixed_total = as_float(summary, "mixed_variant_ms")
    fallback_count = sum(1 for row in windows if row.get("selected_route") == "base_exact")
    fallback_ms = sum(selected_window_cost(row) for row in windows if row.get("selected_route") == "base_exact")
    dominant = max(windows, key=selected_window_cost)
    dominant_ms = selected_window_cost(dominant)
    tied = sum(1 for row in windows if dominant_ms > 0 and selected_window_cost(row) >= dominant_ms * 0.80)
    phases = aggregate_phase_rows(windows)
    noisy_phases = sum(
        1 for row in phases.values()
        if row["range_over_delta"] > 2.0 and abs(row["mixed_delta_ms"]) > 0.0
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
            "    %s route=%s owner=%s status=%s mixed_ms=%s pct=%s%% observed_speedup=%s mixed_speedup=%s unsafe_saved_ms=%s mixed_saved_ms=%s"
            % (
                route_label(row),
                row.get("selected_route", "NA"),
                route_owner(row),
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
        for (kind, metric), row in sorted(phases.items(), key=lambda item: (-abs(item[1]["mixed_delta_ms"]), item[0]))[:top]:
            owner = phase_owner(metric, row)
            print(
                "    kind=%s metric=%s owner=%s next=%s windows=%d candidate=%d fallback=%d base_ms=%s unsafe_variant_ms=%s mixed_variant_ms=%s unsafe_speedup=%s mixed_speedup=%s unsafe_delta_ms=%s mixed_delta_ms=%s max_range_over_delta=%s"
                % (
                    kind,
                    metric,
                    owner,
                    phase_next_move(owner),
                    int(row["windows"]),
                    int(row["candidate_windows"]),
                    int(row["fallback_windows"]),
                    fmt(row["base_ms"]),
                    fmt(row["unsafe_variant_ms"]),
                    fmt(row["mixed_variant_ms"]),
                    fmt(row["unsafe_speedup"]),
                    fmt(row["mixed_speedup"]),
                    fmt(row["unsafe_delta_ms"]),
                    fmt(row["mixed_delta_ms"]),
                    fmt(row["range_over_delta"]),
                )
            )
    fallback_windows = [row for row in windows if row.get("selected_route") == "base_exact"]
    if fallback_windows:
        print("  fallback_alternatives:")
        for row in sorted(fallback_windows, key=selected_window_cost, reverse=True)[:top]:
            alt = cert_fallback_alternative(row, fallback_reuse_count, fallback_min_speedup)
            if alt["status"] != "ok":
                print(
                    "    %s status=%s output_cert_log=%s"
                    % (route_label(row), alt["status"], alt.get("output_cert_log") or "NA")
                )
                continue
            print(
                "    %s selected_route_ms=%s cert_base_total_ms=%s cert_base_vs_selected=%s dual_cache_band_ms=%s vs_selected_route=%s delta_vs_selected_ms=%s break_even_status=%s break_even_uses=%s rows=%s argmax_failures=%s sampled_failures=%s"
                % (
                    route_label(row),
                    fmt(selected_window_cost(row)),
                    fmt(float(alt["cert_base_total_ms"])),
                    fmt(float(alt["cert_base_total_vs_selected_route"])),
                    fmt(float(alt["dual_cache_band_ms"])),
                    fmt(float(alt["dual_cache_band_vs_selected_route"])),
                    fmt(float(alt["dual_cache_band_delta_vs_selected_ms"])),
                    alt["dual_cache_band_break_even_status"],
                    "NA" if alt["dual_cache_band_break_even_uses"] is None else str(alt["dual_cache_band_break_even_uses"]),
                    alt["rows"],
                    alt["argmax_failures"],
                    alt["sampled_failures"],
                )
            )
        print(
            "  Fallback note: dual-cache band rows use output-cert timing, while selected_route_ms comes from route-plan ABBA timing. Treat vs_selected_route as a cross-probe guard, not promotion evidence."
        )


def print_tsv(
    path: Path,
    summary: dict[str, Any],
    windows: list[dict[str, Any]],
    fallback_reuse_count: int,
    fallback_min_speedup: float,
) -> None:
    fields = [
        "kind",
        "source",
        "index",
        "prompt_token",
        "canvas_token",
        "selected_route",
        "route_owner",
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
            "route_owner": route_owner(row),
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

    phase_fields = [
        "kind",
        "source",
        "phase_kind",
        "metric",
        "phase_owner",
        "next_move",
        "windows",
        "candidate_windows",
        "fallback_windows",
        "base_ms",
        "unsafe_variant_ms",
        "mixed_variant_ms",
        "unsafe_speedup",
        "mixed_speedup",
        "unsafe_delta_ms",
        "mixed_delta_ms",
        "range_over_delta",
    ]
    print("\t".join(phase_fields))
    for (kind, metric), row in sorted(aggregate_phase_rows(windows).items(), key=lambda item: (-abs(item[1]["mixed_delta_ms"]), item[0])):
        owner = phase_owner(metric, row)
        values = {
            "kind": "phase",
            "source": str(path),
            "phase_kind": kind,
            "metric": metric,
            "phase_owner": owner,
            "next_move": phase_next_move(owner),
            "windows": int(row["windows"]),
            "candidate_windows": int(row["candidate_windows"]),
            "fallback_windows": int(row["fallback_windows"]),
            "base_ms": fmt(row["base_ms"]),
            "unsafe_variant_ms": fmt(row["unsafe_variant_ms"]),
            "mixed_variant_ms": fmt(row["mixed_variant_ms"]),
            "unsafe_speedup": fmt(row["unsafe_speedup"]),
            "mixed_speedup": fmt(row["mixed_speedup"]),
            "unsafe_delta_ms": fmt(row["unsafe_delta_ms"]),
            "mixed_delta_ms": fmt(row["mixed_delta_ms"]),
            "range_over_delta": fmt(row["range_over_delta"]),
        }
        print("\t".join(str(values[field]) for field in phase_fields))

    fallback_fields = [
        "kind",
        "source",
        "prompt_token",
        "canvas_token",
        "status",
        "selected_route_ms",
        "output_cert_log",
        "rows",
        "argmax_failures",
        "sampled_failures",
        "cert_base_total_ms",
        "cert_variant_total_ms",
        "cert_base_total_vs_selected_route",
        "dual_cache_reuse_count",
        "dual_cache_min_speedup",
        "dual_cache_band_ms",
        "dual_cache_band_speedup_vs_cert_base",
        "dual_cache_band_vs_selected_route",
        "dual_cache_band_delta_vs_selected_ms",
        "dual_cache_band_break_even_status",
        "dual_cache_band_break_even_uses",
    ]
    print("\t".join(fallback_fields))
    for row in sorted((row for row in windows if row.get("selected_route") == "base_exact"), key=selected_window_cost, reverse=True):
        alt = cert_fallback_alternative(row, fallback_reuse_count, fallback_min_speedup)
        values = {
            "kind": "fallback_alt",
            "source": str(path),
            "prompt_token": row.get("prompt_token", ""),
            "canvas_token": row.get("canvas_token", ""),
            "status": alt["status"],
            "selected_route_ms": fmt(selected_window_cost(row)),
            "output_cert_log": alt.get("output_cert_log", ""),
            "rows": alt.get("rows", ""),
            "argmax_failures": alt.get("argmax_failures", ""),
            "sampled_failures": alt.get("sampled_failures", ""),
            "cert_base_total_ms": fmt(float(alt["cert_base_total_ms"])) if alt["status"] == "ok" else "",
            "cert_variant_total_ms": fmt(float(alt["cert_variant_total_ms"])) if alt["status"] == "ok" else "",
            "cert_base_total_vs_selected_route": fmt(float(alt["cert_base_total_vs_selected_route"])) if alt["status"] == "ok" else "",
            "dual_cache_reuse_count": fallback_reuse_count,
            "dual_cache_min_speedup": fmt(fallback_min_speedup),
            "dual_cache_band_ms": fmt(float(alt["dual_cache_band_ms"])) if alt["status"] == "ok" else "",
            "dual_cache_band_speedup_vs_cert_base": fmt(float(alt["dual_cache_band_speedup_vs_cert_base"])) if alt["status"] == "ok" else "",
            "dual_cache_band_vs_selected_route": fmt(float(alt["dual_cache_band_vs_selected_route"])) if alt["status"] == "ok" else "",
            "dual_cache_band_delta_vs_selected_ms": fmt(float(alt["dual_cache_band_delta_vs_selected_ms"])) if alt["status"] == "ok" else "",
            "dual_cache_band_break_even_status": alt.get("dual_cache_band_break_even_status", ""),
            "dual_cache_band_break_even_uses": "NA" if alt.get("dual_cache_band_break_even_uses") is None else alt.get("dual_cache_band_break_even_uses", ""),
        }
        print("\t".join(str(values[field]) for field in fallback_fields))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("route_plan", type=Path, help="mixed route plan JSONL")
    parser.add_argument("--top", type=int, default=8, help="number of windows/phase rows to print")
    parser.add_argument("--tsv", action="store_true", help="emit machine-readable TSV tables")
    parser.add_argument("--fallback-reuse-count", type=int, default=2, help="base exact prompt-cache reuse count for cert-derived fallback alternatives")
    parser.add_argument("--fallback-min-speedup", type=float, default=1.10, help="target speedup for cert-derived fallback break-even estimates")
    args = parser.parse_args()
    if args.fallback_reuse_count <= 0:
        raise SystemExit("--fallback-reuse-count must be positive")
    if args.fallback_min_speedup <= 0.0:
        raise SystemExit("--fallback-min-speedup must be positive")

    summary, windows = load_plan(args.route_plan)
    if args.tsv:
        print_tsv(args.route_plan, summary, windows, args.fallback_reuse_count, args.fallback_min_speedup)
    else:
        print_text(args.route_plan, summary, windows, args.top, args.fallback_reuse_count, args.fallback_min_speedup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
