#!/usr/bin/env python3
"""Summarize DiffusionGemma sparse-loop TSV sweeps."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


REQUIRED = {
    "prompt_len",
    "candidate_count",
    "prompt_cache_ms",
    "prompt_cache_ms_ratio_vs_first",
    "prompt_cache_tokens_per_ms",
    "loop_ms_median",
    "loop_candidate_tokens_per_ms",
}


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except ValueError as exc:
        raise SystemExit(f"bad numeric field {key}={row[key]!r}") from exc


def as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except ValueError as exc:
        raise SystemExit(f"bad integer field {key}={row[key]!r}") from exc


def read_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        reader = csv.DictReader(sys.stdin, delimiter="\t")
        rows = list(reader)
    else:
        with path.open(newline="", encoding="utf-8") as io:
            reader = csv.DictReader(io, delimiter="\t")
            rows = list(reader)

    if reader.fieldnames is None:
        raise SystemExit("missing TSV header")
    missing = sorted(REQUIRED.difference(reader.fieldnames))
    if missing:
        raise SystemExit(f"missing required columns: {','.join(missing)}")
    if not rows:
        raise SystemExit("no data rows")
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[as_int(row, "prompt_len")].append(row)

    out = []
    for prompt_len in sorted(groups):
        group = groups[prompt_len]
        cache_row = min(group, key=lambda r: as_float(r, "prompt_cache_ms"))
        best_loop = min(group, key=lambda r: as_float(r, "loop_ms_median"))
        best_throughput = max(group, key=lambda r: as_float(r, "loop_candidate_tokens_per_ms"))
        out.append(
            {
                "prompt_len": str(prompt_len),
                "rows": str(len(group)),
                "prompt_cache_ms": f"{as_float(cache_row, 'prompt_cache_ms'):.3f}",
                "prompt_cache_ms_ratio_vs_first": f"{as_float(cache_row, 'prompt_cache_ms_ratio_vs_first'):.6f}",
                "prompt_cache_tokens_per_ms": f"{as_float(cache_row, 'prompt_cache_tokens_per_ms'):.6f}",
                "best_loop_candidate_count": str(as_int(best_loop, "candidate_count")),
                "best_loop_ms": f"{as_float(best_loop, 'loop_ms_median'):.3f}",
                "best_throughput_candidate_count": str(as_int(best_throughput, "candidate_count")),
                "best_candidate_tokens_per_ms": f"{as_float(best_throughput, 'loop_candidate_tokens_per_ms'):.6f}",
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", nargs="?", type=Path, help="TSV file; defaults to stdin")
    args = parser.parse_args()

    summary = summarize(read_rows(args.tsv))
    writer = csv.DictWriter(sys.stdout, fieldnames=list(summary[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
