#!/usr/bin/env python3
"""Compute a per-task no-validator route oracle from quality score files."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

DEFAULT_SUFFIXES = (
    "_strict_code",
    "_raw_fence",
    "_file_prefill",
    "_final_code",
    "_fence_prefill",
    "_fence",
    "_strict",
)


def ffloat(value: str | None) -> float:
    try:
        return float(value or "0")
    except ValueError:
        return 0.0


def normalize_name(name: str, suffixes: tuple[str, ...]) -> str:
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def quality_path(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        group, path = raw.split("=", 1)
    else:
        path = raw
        group = Path(path).parent.name or Path(path).stem
    p = Path(path)
    if p.is_dir():
        p = p / "quality_per_prompt.tsv"
    if not p.exists():
        raise SystemExit(f"missing quality_per_prompt.tsv: {p}")
    return group, p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("quality", nargs="+", help="quality_per_prompt.tsv path, dir containing it, or group=path")
    ap.add_argument("--suffix", action="append", default=[], help="Additional prompt-name suffix to strip for task grouping")
    ap.add_argument("--out", type=Path, default=None, help="Optional TSV path for per-task oracle rows")
    args = ap.parse_args()

    suffixes = tuple(args.suffix) + DEFAULT_SUFFIXES
    rows: list[dict[str, str]] = []
    for group, path in map(quality_path, args.quality):
        with path.open(newline="", encoding="utf-8") as io:
            for row in csv.DictReader(io, delimiter="\t"):
                row = dict(row)
                row["group"] = group
                row["task"] = normalize_name(row.get("name", ""), suffixes)
                rows.append(row)
    if not rows:
        raise SystemExit("no rows loaded")

    by_task: dict[str, list[dict[str, str]]] = {}
    by_route: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
        by_route.setdefault(row.get("route", ""), []).append(row)

    oracle_rows: list[dict[str, str]] = []
    for task, candidates in sorted(by_task.items()):
        best = max(candidates, key=lambda r: ffloat(r.get("score")))
        oracle_rows.append({
            "task": task,
            "score": best.get("score", "0"),
            "group": best.get("group", ""),
            "route": best.get("route", ""),
            "name": best.get("name", ""),
            "status": best.get("status", ""),
            "think_leak": best.get("think_leak", ""),
            "substantive_code": best.get("substantive_code", ""),
            "compile_ok": best.get("compile_ok", ""),
            "repair_ok": best.get("repair_ok", ""),
            "speed_ratio": best.get("speed_ratio", ""),
            "draft_text": best.get("draft_text", ""),
        })

    score_values = [ffloat(row["score"]) for row in oracle_rows]
    print(f"oracle_tasks={len(oracle_rows)} oracle_mean={statistics.mean(score_values):.3f} oracle_median={statistics.median(score_values):.3f}")
    print("task\tscore\tgroup\troute\tthink\tsubstantive\trepair\tspeed\tstatus")
    for row in oracle_rows:
        print(
            f"{row['task']}\t{row['score']}\t{row['group']}\t{row['route']}\t"
            f"{row['think_leak']}\t{row['substantive_code']}\t{row['repair_ok']}\t"
            f"{row['speed_ratio']}\t{row['status']}"
        )

    complete_routes = [(route, route_rows) for route, route_rows in by_route.items() if len(route_rows) == len(oracle_rows)]
    if complete_routes:
        best_route, best_rows = max(complete_routes, key=lambda item: statistics.mean(ffloat(r.get("score")) for r in item[1]))
        best_mean = statistics.mean(ffloat(r.get("score")) for r in best_rows)
        print(f"best_single_route={best_route} best_single_mean={best_mean:.3f} oracle_gap={statistics.mean(score_values) - best_mean:.3f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "task", "score", "group", "route", "name", "status", "think_leak",
            "substantive_code", "compile_ok", "repair_ok", "speed_ratio", "draft_text",
        ]
        with args.out.open("w", newline="", encoding="utf-8") as io:
            writer = csv.DictWriter(io, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(oracle_rows)
        print(f"oracle_tsv={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
