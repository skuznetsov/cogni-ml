#!/usr/bin/env python3
"""Aggregate qwen35 self-spec route selector scoreboards from probe logs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


HEADER_PREFIX = "rank mode split route updown feature op threshold "


@dataclass
class Aggregate:
    key: tuple[str, str, str, str, str, str, str]
    logs: set[str]
    prompts: int = 0
    selected: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    baseline_total: float = 0.0
    policy_total: float = 0.0
    worst_delta: float | None = None
    max_loss: float = 0.0

    def add(self, log_name: str, row: list[str]) -> None:
        self.logs.add(log_name)
        self.prompts += int(row[8])
        self.selected += int(row[9])
        self.wins += int(row[10])
        self.losses += int(row[11])
        self.ties += int(row[12])
        self.baseline_total += float(row[13])
        self.policy_total += float(row[14])
        row_worst = float(row[16])
        self.worst_delta = row_worst if self.worst_delta is None else min(self.worst_delta, row_worst)
        self.max_loss = max(self.max_loss, float(row[17]))

    @property
    def delta(self) -> float:
        if self.baseline_total <= 0.0:
            return 0.0
        return (self.baseline_total - self.policy_total) * 100.0 / self.baseline_total

    @property
    def score(self) -> float:
        worst = self.worst_delta or 0.0
        return self.delta + worst * 0.5 - self.losses * 10.0 - self.max_loss * 0.25


def parse_selector_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    in_table = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("self_spec_route_selector_scoreboard "):
            in_table = True
            continue
        if in_table and line.startswith(HEADER_PREFIX):
            continue
        if in_table:
            if not line.strip() or not line[0].isdigit():
                in_table = False
                continue
            parts = line.split()
            if len(parts) >= 19:
                rows.append(parts)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--min-logs", type=int, default=1)
    args = parser.parse_args()

    aggregates: dict[tuple[str, str, str, str, str, str, str], Aggregate] = {}
    for path in args.logs:
        rows = parse_selector_rows(path)
        if not rows:
            print(f"warn no_selector_rows path={path}")
            continue
        for row in rows:
            key = tuple(row[1:8])
            agg = aggregates.get(key)
            if agg is None:
                agg = Aggregate(key=key, logs=set())
                aggregates[key] = agg
            agg.add(str(path), row)

    ranked = sorted(
        (row for row in aggregates.values() if len(row.logs) >= args.min_logs),
        key=lambda row: row.score,
        reverse=True,
    )
    print(f"route_selector_aggregate policies={len(ranked)} logs={len(args.logs)} limit={args.limit}")
    print("rank mode split route updown feature op threshold logs prompts selected wins losses ties baseline_total policy_total delta% worst_delta% max_loss% score")
    for i, agg in enumerate(ranked[: args.limit], start=1):
        mode, split, route, updown, feature, op, threshold = agg.key
        print(
            f"{i} {mode} {split} {route} {updown} {feature} {op} {threshold} "
            f"{len(agg.logs)} {agg.prompts} {agg.selected} {agg.wins} {agg.losses} {agg.ties} "
            f"{agg.baseline_total:.3f} {agg.policy_total:.3f} {agg.delta:.2f} "
            f"{(agg.worst_delta or 0.0):.2f} {agg.max_loss:.2f} {agg.score:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
