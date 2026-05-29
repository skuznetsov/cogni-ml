#!/usr/bin/env python3
"""Summarize qwen35_serving_route_matrix TSV output.

The matrix intentionally reports raw per-route timings. This helper adds the
route economics view: per-prompt speedup versus greedy and median route cost.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def f64(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    args = parser.parse_args()

    with args.matrix.open(newline="", encoding="utf-8") as io:
        rows = list(csv.DictReader(io, delimiter="\t"))

    by_prompt: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    by_mode: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        prompt = row["prompt_id"]
        mode = row["mode"]
        by_prompt[prompt][mode] = row
        by_mode[mode].append(f64(row["p50_total_ms"]))

    print("prompt_id\tmode\tp50_total_ms\tspeedup_vs_greedy\troutes\tlog")
    for prompt in sorted(by_prompt):
        greedy = by_prompt[prompt].get("greedy")
        greedy_ms = f64(greedy["p50_total_ms"]) if greedy else 0.0
        for mode in sorted(by_prompt[prompt]):
            row = by_prompt[prompt][mode]
            p50 = f64(row["p50_total_ms"])
            speedup = (greedy_ms / p50) if greedy_ms > 0.0 and p50 > 0.0 else 0.0
            print(
                f"{prompt}\t{mode}\t{p50:.6g}\t{speedup:.3f}\t"
                f"{row.get('routes', '')}\t{row.get('log', '')}"
            )

    print("")
    print("mode\tmedian_p50_total_ms\tmin_p50_total_ms\tmax_p50_total_ms\trows")
    for mode in sorted(by_mode):
        values = [v for v in by_mode[mode] if v == v]
        if not values:
            continue
        print(
            f"{mode}\t{statistics.median(values):.6g}\t"
            f"{min(values):.6g}\t{max(values):.6g}\t{len(values)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
