#!/usr/bin/env python3
"""Aggregate DiffusionGemma ABBA row decisions into a suite decision."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import diffusion_gemma_abba_dir_summary as dir_summary


def as_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def choose_suite_decision(counts: dict[str, int], rows: int) -> str:
    if rows == 0:
        return "blocked_no_rows"
    if counts.get("blocked_by_host_noise", 0) > 0:
        return "blocked_by_host_noise"
    if counts.get("blocked_by_range", 0) > 0:
        return "blocked_by_range"
    if counts.get("blocked_missing_delta", 0) > 0:
        return "blocked_missing_delta"
    if counts.get("reject_regression", 0) > 0:
        return "reject_regression"
    if counts.get("neutral", 0) > 0:
        return "neutral"
    if counts.get("candidate_speedup", 0) == rows:
        return "candidate_speedup"
    return "blocked_unknown"


def format_float(value: float) -> str:
    return f"{value:.4f}" if not math.isnan(value) else "NA"


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    counts: dict[str, int] = {}
    loop_speedups: list[float] = []
    context_speedups: list[float] = []
    context_bounded = 0
    for row in rows:
        decision = str(row.get("promotion_decision", "missing"))
        counts[decision] = counts.get(decision, 0) + 1
        loop_speedup = as_float(str(row.get("loop_ms_median_speedup", "nan")))
        context_speedup = as_float(str(row.get("loop_decode_context_ms_speedup", "nan")))
        if not math.isnan(loop_speedup):
            loop_speedups.append(loop_speedup)
        if not math.isnan(context_speedup):
            context_speedups.append(context_speedup)
        if row.get("loop_decode_context_ms_delta_confidence") == "range_bounded":
            context_bounded += 1

    return {
        "suite_decision": choose_suite_decision(counts, len(rows)),
        "rows": len(rows),
        "counts": counts,
        "min_loop_speedup": min(loop_speedups) if loop_speedups else float("nan"),
        "min_context_speedup": min(context_speedups) if context_speedups else float("nan"),
        "context_range_bounded": context_bounded,
    }


def print_key_values(summary: dict[str, object]) -> None:
    print(f"suite_decision={summary['suite_decision']}")
    print(f"rows={summary['rows']}")
    counts = summary["counts"]
    if isinstance(counts, dict):
        for decision in sorted(counts):
            print(f"{decision}={counts[decision]}")
    print(f"min_loop_speedup={format_float(float(summary['min_loop_speedup']))}")
    print(f"min_context_speedup={format_float(float(summary['min_context_speedup']))}")
    print(f"context_range_bounded={summary['context_range_bounded']}/{summary['rows']}")


def run_self_test() -> None:
    cases = [
        ({}, 0, "blocked_no_rows"),
        ({"blocked_by_host_noise": 1, "candidate_speedup": 2}, 3, "blocked_by_host_noise"),
        ({"blocked_by_range": 1, "candidate_speedup": 2}, 3, "blocked_by_range"),
        ({"blocked_missing_delta": 1, "candidate_speedup": 2}, 3, "blocked_missing_delta"),
        ({"reject_regression": 1, "candidate_speedup": 2}, 3, "reject_regression"),
        ({"neutral": 1, "candidate_speedup": 2}, 3, "neutral"),
        ({"candidate_speedup": 3}, 3, "candidate_speedup"),
        ({"missing": 1, "candidate_speedup": 2}, 3, "blocked_unknown"),
    ]
    for counts, rows, expected in cases:
        actual = choose_suite_decision(counts, rows)
        if actual != expected:
            raise AssertionError(f"expected {expected}, got {actual} for counts={counts}, rows={rows}")
    print("self_test=ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--format", choices=("kv", "json"), default="kv")
    parser.add_argument("--require-candidate", action="store_true", help="Exit nonzero unless the suite is a candidate speedup.")
    parser.add_argument("--roots", nargs="+", type=Path, help="Summarize ABBA run directories directly.")
    parser.add_argument("summary_tsv", nargs="?", default="-")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.roots:
        rows = []
        for root in args.roots:
            rows.extend(dir_summary.summarize(root))
    elif args.summary_tsv == "-":
        rows = list(csv.DictReader(sys.stdin, delimiter="\t"))
    else:
        with open(args.summary_tsv, newline="", encoding="utf-8") as io:
            rows = list(csv.DictReader(io, delimiter="\t"))

    summary = summarize_rows(rows)
    if args.format == "json":
        print(json.dumps(summary, sort_keys=True))
    else:
        print_key_values(summary)
    if args.require_candidate and summary["suite_decision"] != "candidate_speedup":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
