#!/usr/bin/env python3
"""Aggregate DiffusionGemma ABBA row decisions into a suite decision."""

from __future__ import annotations

import argparse
import csv
import math
import sys


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_tsv", nargs="?", default="-")
    args = parser.parse_args()

    if args.summary_tsv == "-":
        rows = list(csv.DictReader(sys.stdin, delimiter="\t"))
    else:
        with open(args.summary_tsv, newline="", encoding="utf-8") as io:
            rows = list(csv.DictReader(io, delimiter="\t"))

    counts: dict[str, int] = {}
    loop_speedups: list[float] = []
    context_speedups: list[float] = []
    context_bounded = 0
    for row in rows:
        decision = row.get("promotion_decision", "missing")
        counts[decision] = counts.get(decision, 0) + 1
        loop_speedup = as_float(row.get("loop_ms_median_speedup", "nan"))
        context_speedup = as_float(row.get("loop_decode_context_ms_speedup", "nan"))
        if not math.isnan(loop_speedup):
            loop_speedups.append(loop_speedup)
        if not math.isnan(context_speedup):
            context_speedups.append(context_speedup)
        if row.get("loop_decode_context_ms_delta_confidence") == "range_bounded":
            context_bounded += 1

    suite_decision = choose_suite_decision(counts, len(rows))
    print(f"suite_decision={suite_decision}")
    print(f"rows={len(rows)}")
    for decision in sorted(counts):
        print(f"{decision}={counts[decision]}")
    print(f"min_loop_speedup={format_float(min(loop_speedups) if loop_speedups else float('nan'))}")
    print(f"min_context_speedup={format_float(min(context_speedups) if context_speedups else float('nan'))}")
    print(f"context_range_bounded={context_bounded}/{len(rows)}")


if __name__ == "__main__":
    main()
