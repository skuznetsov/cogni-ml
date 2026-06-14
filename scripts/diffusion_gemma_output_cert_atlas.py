#!/usr/bin/env python3
"""Inspect DiffusionGemma output-certificate row failures.

This helper is offline attribution, not a promotion gate. It reads
diffusion_gemma_prompt_output_cert_probe.cr output and identifies whether a
window failed globally or only on a small row/candidate subset. Any row-local
fallback estimate is explicitly optimistic until runtime can prove that the
exact boundary can be narrowed without rebuilding the full exact prompt cache.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any


def fmt(value: float) -> str:
    if math.isnan(value):
        return "NA"
    if math.isinf(value):
        return "inf"
    return f"{value:.6f}"


def truthy(value: str) -> bool:
    return value.lower() == "true"


def as_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def as_int(value: str | None) -> int:
    if value is None or value == "":
        raise ValueError
    return int(value)


def safe_speedup(numerator: float, denominator: float) -> float:
    if math.isnan(numerator) or math.isnan(denominator):
        return float("nan")
    if denominator == 0.0:
        return float("inf")
    return numerator / denominator


def parse_kv_parts(parts: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    return fields


def read_cert(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    timing: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []
    header: list[str] | None = None

    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        for raw_parts in csv.reader(handle, delimiter="\t"):
            if not raw_parts:
                continue
            first = raw_parts[0]
            if first.startswith("#"):
                continue
            if first == "kind":
                header = raw_parts
                continue
            if first == "timing_summary":
                timing.append(parse_kv_parts(raw_parts[1:]))
                continue
            if first in {"cert_summary", "aggregate_summary", "hidden_summary"}:
                summary = parse_kv_parts(raw_parts[1:])
                summary["kind"] = first
                summaries.append(summary)
                continue
            if first == "row":
                if header is None:
                    raise SystemExit(f"{path}: row appeared before header")
                if len(raw_parts) != len(header):
                    raise SystemExit(f"{path}: row field count {len(raw_parts)} does not match header {len(header)}")
                rows.append(dict(zip(header, raw_parts)))
                continue
            raise SystemExit(f"{path}: unsupported output-cert row kind {first!r}")
    if not rows:
        raise SystemExit(f"{path}: no certificate row records found")
    return rows, timing, summaries


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("prompt_token", ""), row.get("canvas_token", ""))
        grouped.setdefault(key, []).append(row)
    return grouped


def timing_by_window(timing_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in timing_rows:
        key = (row.get("prompt_token", ""), row.get("canvas_token", ""))
        result[key] = row
    return result


def optimistic_row_fallback_ms(timing: dict[str, str], total_rows: int, fail_rows: int) -> float:
    variant_cache = as_float(timing.get("variant_cache_ms"))
    variant_predict = as_float(timing.get("variant_predict_ms"))
    base_predict = as_float(timing.get("base_predict_ms"))
    if total_rows <= 0 or fail_rows <= 0:
        return variant_cache + variant_predict
    return variant_cache + variant_predict + base_predict * fail_rows / total_rows


def dual_cache_row_fallback_ms(
    timing: dict[str, str],
    total_rows: int,
    fail_rows: int,
    reuse_count: int,
) -> float:
    variant_cache = as_float(timing.get("variant_cache_ms"))
    variant_predict = as_float(timing.get("variant_predict_ms"))
    base_cache = as_float(timing.get("base_cache_ms"))
    base_predict = as_float(timing.get("base_predict_ms"))
    if total_rows <= 0 or fail_rows <= 0:
        return variant_cache + variant_predict
    return variant_cache + variant_predict + base_cache / reuse_count + base_predict * fail_rows / total_rows


def dual_cache_break_even_uses(
    timing: dict[str, str],
    total_rows: int,
    fail_rows: int,
    min_speedup: float,
) -> tuple[str, int | None]:
    if total_rows <= 0 or fail_rows <= 0:
        return "not_needed", 0
    if min_speedup <= 0.0:
        raise SystemExit("--min-speedup must be positive")

    base_cache = as_float(timing.get("base_cache_ms"))
    base_predict = as_float(timing.get("base_predict_ms"))
    variant_cache = as_float(timing.get("variant_cache_ms"))
    variant_predict = as_float(timing.get("variant_predict_ms"))
    base_total = base_cache + base_predict
    row_exact_predict = base_predict * fail_rows / total_rows
    fixed_candidate = variant_cache + variant_predict + row_exact_predict
    required_margin = base_total - min_speedup * fixed_candidate
    if any(math.isnan(value) for value in (base_cache, base_predict, variant_cache, variant_predict)):
        return "missing_timing", None
    if required_margin <= 0.0:
        return "impossible", None
    return "finite", max(1, math.ceil(min_speedup * base_cache / required_margin))


def print_text(
    path: Path,
    rows: list[dict[str, str]],
    timing_rows: list[dict[str, str]],
    reuse_count: int,
    min_speedup: float,
) -> None:
    grouped = group_rows(rows)
    timings = timing_by_window(timing_rows)
    print("DiffusionGemma output certificate atlas")
    print(f"  source={path}")
    print(f"  dual_cache_candidate reuse_count={reuse_count} min_speedup={fmt(min_speedup)}")
    for key, window_rows in sorted(grouped.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        prompt_token, canvas_token = key
        argmax_fail = [row for row in window_rows if not truthy(row.get("argmax_match", ""))]
        sampled_fail = [row for row in window_rows if not truthy(row.get("sampled_match", ""))]
        row_count = len(window_rows)
        timing = timings.get(key, {})
        base_total = as_float(timing.get("base_cache_ms")) + as_float(timing.get("base_predict_ms"))
        variant_total = as_float(timing.get("variant_cache_ms")) + as_float(timing.get("variant_predict_ms"))
        row_fallback = optimistic_row_fallback_ms(timing, row_count, len(argmax_fail))
        dual_cache = dual_cache_row_fallback_ms(timing, row_count, len(argmax_fail), reuse_count)
        dual_speedup = safe_speedup(base_total, dual_cache)
        break_even_status, break_even_uses = dual_cache_break_even_uses(
            timing,
            row_count,
            len(argmax_fail),
            min_speedup,
        )
        current_route = "variant_fast" if not argmax_fail else "base_exact"
        print(
            "  window %s:%s rows=%d argmax=%d/%d sampled=%d/%d current_legal_route=%s"
            % (
                prompt_token,
                canvas_token,
                row_count,
                row_count - len(argmax_fail),
                row_count,
                row_count - len(sampled_fail),
                row_count,
                current_route,
            )
        )
        if timing:
            print(
                "    cert_timing base_total_ms=%s variant_total_ms=%s optimistic_row_fallback_ms=%s"
                % (fmt(base_total), fmt(variant_total), fmt(row_fallback))
            )
            print(
                "    dual_cache_candidate_ms=%s speedup_vs_base_exact=%s break_even_status=%s break_even_uses=%s"
                % (
                    fmt(dual_cache),
                    fmt(dual_speedup),
                    break_even_status,
                    "NA" if break_even_uses is None else str(break_even_uses),
                )
            )
        if argmax_fail:
            details = ", ".join(
                "row=%s base_argmax=%s variant_argmax=%s max_logit_delta=%s"
                % (
                    row.get("row", "NA"),
                    row.get("base_argmax", "NA"),
                    row.get("variant_argmax", "NA"),
                    row.get("max_logit_abs_delta", "NA"),
                )
                for row in argmax_fail
            )
            print(f"    argmax_failures: {details}")
        sampled_only = [row for row in sampled_fail if truthy(row.get("argmax_match", ""))]
        if sampled_only:
            details = ", ".join(
                "row=%s base_sampled=%s variant_sampled=%s"
                % (row.get("row", "NA"), row.get("base_sampled", "NA"), row.get("variant_sampled", "NA"))
                for row in sampled_only
            )
            print(f"    sampled_only_failures: {details}")
        if argmax_fail:
            print(
                "    LTP/WBA: Diamond. Candidate window is row-local, but exact fallback is only legal after proving the prompt-cache/hidden boundary can be narrowed or reused; dual-cache numbers are branch-selection estimates, not promotion evidence."
            )
        else:
            print("    LTP/WBA: Collapse. Argmax certificate holds for this bounded candidate row set.")


def print_tsv(
    path: Path,
    rows: list[dict[str, str]],
    timing_rows: list[dict[str, str]],
    reuse_count: int,
    min_speedup: float,
) -> None:
    timings = timing_by_window(timing_rows)
    fields = [
        "kind",
        "source",
        "prompt_token",
        "canvas_token",
        "row",
        "argmax_match",
        "sampled_match",
        "base_argmax",
        "variant_argmax",
        "base_sampled",
        "variant_sampled",
        "base_logit_margin",
        "variant_logit_margin",
        "max_logit_abs_delta",
    ]
    print("\t".join(fields))
    for row in rows:
        values: dict[str, Any] = {
            "kind": "row",
            "source": str(path),
            **row,
        }
        print("\t".join(str(values.get(field, "")) for field in fields))

    summary_fields = [
        "kind",
        "source",
        "prompt_token",
        "canvas_token",
        "rows",
        "argmax_failures",
        "sampled_failures",
        "current_legal_route",
        "base_cache_ms",
        "base_predict_ms",
        "variant_cache_ms",
        "variant_predict_ms",
        "base_total_ms",
        "variant_total_ms",
        "optimistic_row_fallback_ms",
        "dual_cache_reuse_count",
        "dual_cache_min_speedup",
        "dual_cache_candidate_ms",
        "dual_cache_speedup_vs_base_exact",
        "dual_cache_break_even_status",
        "dual_cache_break_even_uses",
    ]
    print("\t".join(summary_fields))
    for key, window_rows in sorted(group_rows(rows).items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        timing = timings.get(key, {})
        argmax_failures = sum(1 for row in window_rows if not truthy(row.get("argmax_match", "")))
        sampled_failures = sum(1 for row in window_rows if not truthy(row.get("sampled_match", "")))
        base_cache = as_float(timing.get("base_cache_ms"))
        base_predict = as_float(timing.get("base_predict_ms"))
        variant_cache = as_float(timing.get("variant_cache_ms"))
        variant_predict = as_float(timing.get("variant_predict_ms"))
        base_total = as_float(timing.get("base_cache_ms")) + as_float(timing.get("base_predict_ms"))
        variant_total = as_float(timing.get("variant_cache_ms")) + as_float(timing.get("variant_predict_ms"))
        dual_cache = dual_cache_row_fallback_ms(timing, len(window_rows), argmax_failures, reuse_count)
        dual_speedup = safe_speedup(base_total, dual_cache)
        break_even_status, break_even_uses = dual_cache_break_even_uses(
            timing,
            len(window_rows),
            argmax_failures,
            min_speedup,
        )
        values = {
            "kind": "summary",
            "source": str(path),
            "prompt_token": key[0],
            "canvas_token": key[1],
            "rows": len(window_rows),
            "argmax_failures": argmax_failures,
            "sampled_failures": sampled_failures,
            "current_legal_route": "variant_fast" if argmax_failures == 0 else "base_exact",
            "base_cache_ms": fmt(base_cache),
            "base_predict_ms": fmt(base_predict),
            "variant_cache_ms": fmt(variant_cache),
            "variant_predict_ms": fmt(variant_predict),
            "base_total_ms": fmt(base_total),
            "variant_total_ms": fmt(variant_total),
            "optimistic_row_fallback_ms": fmt(optimistic_row_fallback_ms(timing, len(window_rows), argmax_failures)),
            "dual_cache_reuse_count": reuse_count,
            "dual_cache_min_speedup": fmt(min_speedup),
            "dual_cache_candidate_ms": fmt(dual_cache),
            "dual_cache_speedup_vs_base_exact": fmt(dual_speedup),
            "dual_cache_break_even_status": break_even_status,
            "dual_cache_break_even_uses": "NA" if break_even_uses is None else str(break_even_uses),
        }
        print("\t".join(str(values[field]) for field in summary_fields))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_cert", type=Path, help="output_cert.tsv from diffusion_gemma_prompt_output_cert_probe")
    parser.add_argument("--tsv", action="store_true", help="emit machine-readable row and summary tables")
    parser.add_argument("--reuse-count", type=int, default=1, help="base exact prompt-cache reuse count for dual-cache estimates")
    parser.add_argument("--min-speedup", type=float, default=1.10, help="target speedup versus current exact fallback for break-even uses")
    args = parser.parse_args()
    if args.reuse_count <= 0:
        raise SystemExit("--reuse-count must be positive")
    if args.min_speedup <= 0.0:
        raise SystemExit("--min-speedup must be positive")

    rows, timing, _ = read_cert(args.output_cert)
    if args.tsv:
        print_tsv(args.output_cert, rows, timing, args.reuse_count, args.min_speedup)
    else:
        print_text(args.output_cert, rows, timing, args.reuse_count, args.min_speedup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
