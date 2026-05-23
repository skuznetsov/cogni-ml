#!/usr/bin/env python3
"""Analyze route-labeled self-spec cycle dumps."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cycle:
    prompt: str
    position: int
    route: str
    wall_ms: float
    accepted: int
    proposed: int
    reject_index: int
    expected_gain_ms: float


def route_from_policy(policy: str) -> str:
    marker = "/route="
    if marker not in policy:
        return "pure"
    return policy.split(marker, 1)[1]


def read_cycles(paths: list[Path]) -> list[Cycle]:
    cycles: list[Cycle] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cycles.append(
                Cycle(
                    prompt=row["prompt_name"],
                    position=int(row["position"]),
                    route=route_from_policy(row["policy"]),
                    wall_ms=float(row["wall_ms"]),
                    accepted=int(row["accepted_count"]),
                    proposed=int(row["proposed_count"]),
                    reject_index=int(row["reject_index"]),
                    expected_gain_ms=float(row["expected_gain_ms"]),
                )
            )
    return cycles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    cycles = read_cycles(args.jsonl)
    groups: dict[tuple[str, int], list[Cycle]] = defaultdict(list)
    for cycle in cycles:
        groups[(cycle.prompt, cycle.position)].append(cycle)

    pure_total = 0.0
    oracle_total = 0.0
    route_stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    picks: list[tuple[str, int, str, float, float, float, int, int, int]] = []
    usable = 0
    for (prompt, position), rows in groups.items():
        pure_rows = [row for row in rows if row.route == "pure"]
        if not pure_rows:
            continue
        pure = min(pure_rows, key=lambda row: row.wall_ms)
        candidates = [pure] + [row for row in rows if row.route != "pure" and row.accepted > 0]
        best = min(candidates, key=lambda row: row.wall_ms)
        if pure.wall_ms <= 0.0:
            continue
        delta = (pure.wall_ms - best.wall_ms) * 100.0 / pure.wall_ms
        pure_total += pure.wall_ms
        oracle_total += best.wall_ms
        usable += 1
        stat = route_stats[best.route]
        stat["picks"] += 1
        stat["accepted"] += best.accepted
        stat["proposed"] += best.proposed
        stat["rejects"] += 1 if best.reject_index >= 0 else 0
        stat["delta_sum"] += delta
        picks.append((prompt, position, best.route, pure.wall_ms, best.wall_ms, delta, best.accepted, best.proposed, best.reject_index))

    total_delta = (pure_total - oracle_total) * 100.0 / pure_total if pure_total > 0.0 else 0.0
    print(f"route_cycle_oracle cycles={len(cycles)} groups={len(groups)} usable={usable} pure_total={pure_total:.3f} oracle_total={oracle_total:.3f} oracle_delta%={total_delta:.2f}")
    print("route picks accepted proposed rejects mean_delta%")
    for route, stat in sorted(route_stats.items(), key=lambda item: item[1]["picks"], reverse=True):
        picks_count = int(stat["picks"])
        mean_delta = stat["delta_sum"] / picks_count if picks_count else 0.0
        print(f"{route} {picks_count} {int(stat['accepted'])} {int(stat['proposed'])} {int(stat['rejects'])} {mean_delta:.2f}")
    print("rank prompt position best_route pure_ms best_ms delta% accepted proposed reject_index")
    for i, row in enumerate(sorted(picks, key=lambda item: item[5], reverse=True)[: args.limit], start=1):
        prompt, position, route, pure_ms, best_ms, delta, accepted, proposed, reject_index = row
        print(f"{i} {prompt} {position} {route} {pure_ms:.3f} {best_ms:.3f} {delta:.2f} {accepted} {proposed} {reject_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
