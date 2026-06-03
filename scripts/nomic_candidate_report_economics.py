#!/usr/bin/env python3
"""Score Nomic shallow-candidate reports under a simple hot-index cost model.

The input is produced by:
  bin/nomic_encoder_tricks_probe --candidate-report=report.tsv ...

This intentionally does not claim a product speedup. It asks whether a shallow
candidate band could pay for itself after full-depth query cost cancels out.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Row:
    depth: int
    query: str
    candidate_count: int
    shallow_has_full: bool
    rerank_matches_full: bool


def read_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {
            "depth",
            "query",
            "candidate_count",
            "shallow_has_full",
            "rerank_matches_full",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"missing report columns: {','.join(sorted(missing))}")
        for rec in reader:
            rows.append(
                Row(
                    depth=int(rec["depth"]),
                    query=rec["query"],
                    candidate_count=int(rec["candidate_count"]),
                    shallow_has_full=rec["shallow_has_full"] == "1",
                    rerank_matches_full=rec["rerank_matches_full"] == "1",
                )
            )
    if not rows:
        raise SystemExit("candidate report has no data rows")
    return rows


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", type=Path)
    ap.add_argument("--docs", type=int, required=True, help="Total docs in the scanned corpus")
    ap.add_argument("--full-doc-scan-ms", type=float, default=0.01, help="Baseline full-depth scan/rerank cost per doc per query")
    ap.add_argument("--shallow-doc-scan-ms", type=float, default=0.001, help="Shallow index scan cost per doc per query")
    ap.add_argument("--full-candidate-rerank-ms", type=float, default=None, help="Full-depth candidate rerank cost per candidate; default = full-doc-scan-ms")
    ap.add_argument("--shallow-query-ms", type=float, default=0.0, help="Extra shallow query embedding cost per query")
    ap.add_argument("--fixed-route-ms", type=float, default=0.0, help="Extra fixed route overhead per query")
    ap.add_argument("--min-contained-pct", type=float, default=100.0, help="Fail quality below this containment percent")
    ap.add_argument("--include-full-depth", action="store_true", help="Also score the max-depth rows; off by default because they are the full-depth reference")
    args = ap.parse_args()

    rows = read_rows(args.report)
    by_depth: dict[int, list[Row]] = defaultdict(list)
    for row in rows:
        by_depth[row.depth].append(row)

    docs_total = args.docs
    if docs_total <= 0:
        raise SystemExit("--docs must be positive")
    full_candidate_rerank_ms = args.full_candidate_rerank_ms
    if full_candidate_rerank_ms is None:
        full_candidate_rerank_ms = args.full_doc_scan_ms

    print(
        "depth\tqueries\tcontained\trerank_ok\tavg_candidates\tmax_candidates\t"
        "baseline_scan_ms\troute_scan_ms\tnet_ms\tspeedup\t"
        "break_even_full_doc_ms\tmax_shallow_doc_ms\tverdict"
    )

    full_depth = max(by_depth)
    for depth, depth_rows in sorted(by_depth.items()):
        if depth == full_depth and not args.include_full_depth:
            continue
        queries = len(depth_rows)
        contained = sum(1 for row in depth_rows if row.shallow_has_full)
        rerank_ok = sum(1 for row in depth_rows if row.rerank_matches_full)
        candidate_counts = [row.candidate_count for row in depth_rows]
        avg_candidates = sum(candidate_counts) / queries
        max_candidates = max(candidate_counts)

        baseline_per_query = docs_total * args.full_doc_scan_ms
        route_per_query = (
            args.shallow_query_ms
            + args.fixed_route_ms
            + docs_total * args.shallow_doc_scan_ms
            + avg_candidates * full_candidate_rerank_ms
        )
        net_per_query = baseline_per_query - route_per_query
        speedup = baseline_per_query / route_per_query if route_per_query > 0 else 0.0
        break_even_full_doc_ms = route_per_query / docs_total
        max_shallow_doc_ms = (
            baseline_per_query
            - args.shallow_query_ms
            - args.fixed_route_ms
            - avg_candidates * full_candidate_rerank_ms
        ) / docs_total

        quality_pct = pct(min(contained, rerank_ok), queries)
        if quality_pct < args.min_contained_pct:
            verdict = "quality_fail"
        elif net_per_query > 0.0:
            verdict = "candidate"
        else:
            verdict = "economics_fail"

        print(
            f"{depth}\t{queries}\t{contained}/{queries}\t{rerank_ok}/{queries}\t"
            f"{avg_candidates:.3f}\t{max_candidates}\t"
            f"{baseline_per_query:.6f}\t{route_per_query:.6f}\t"
            f"{net_per_query:.6f}\t{speedup:.3f}\t"
            f"{break_even_full_doc_ms:.8f}\t{max_shallow_doc_ms:.8f}\t{verdict}"
        )


if __name__ == "__main__":
    main()
