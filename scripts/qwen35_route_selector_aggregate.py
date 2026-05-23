#!/usr/bin/env python3
"""Aggregate qwen35 self-spec route selector scoreboards from probe logs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


HEADER_PREFIX = "rank mode split route updown feature op threshold "
FEATURES = (
    "residual_mean",
    "residual_p90",
    "residual_max",
    "repeat_rate",
    "bigram_repeat_rate",
    "unique_rate",
)


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

    def add_policy(
        self,
        log_name: str,
        prompts: int,
        selected: int,
        wins: int,
        losses: int,
        ties: int,
        baseline_total: float,
        policy_total: float,
        worst_delta: float,
        max_loss: float,
    ) -> None:
        self.logs.add(log_name)
        self.prompts += prompts
        self.selected += selected
        self.wins += wins
        self.losses += losses
        self.ties += ties
        self.baseline_total += baseline_total
        self.policy_total += policy_total
        self.worst_delta = worst_delta if self.worst_delta is None else min(self.worst_delta, worst_delta)
        self.max_loss = max(self.max_loss, max_loss)

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


def parse_key_values(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in line.split()[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def route_mode_and_split(fields: dict[str, str]) -> tuple[str, str]:
    if "schedule" in fields:
        mode = f"schedule={fields['schedule']}"
    else:
        mode = f"gamma={fields.get('gamma', 'unknown')}"
    split = fields.get("draft_split", "nil")
    return mode, split


def parse_raw_route_rows(path: Path):
    features: dict[str, dict[str, float]] = {}
    route_rows: list[tuple[str, str, str, str, str, float]] = []
    in_oracle = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("self_spec_route_oracle "):
            in_oracle = True
            continue
        if in_oracle and line.startswith("rank prompt mode split best_route "):
            continue
        if in_oracle and line and line[0].isdigit():
            parts = line.split()
            if len(parts) >= 20:
                try:
                    features[parts[1]] = {
                        "residual_mean": float(parts[14]),
                        "residual_p90": float(parts[15]),
                        "residual_max": float(parts[16]),
                        "repeat_rate": float(parts[17]),
                        "bigram_repeat_rate": float(parts[18]),
                        "unique_rate": float(parts[19]),
                    }
                except ValueError:
                    pass
            continue
        if in_oracle and (not line.strip() or not line[0].isdigit()):
            in_oracle = False

        if line.startswith("self_spec_gpu_pipeline_suite_hybrid ") or line.startswith("self_spec_gpu_pipeline_hybrid "):
            fields = parse_key_values(line)
            if fields.get("parity") != "true":
                continue
            name = fields.get("name", "main")
            route = fields.get("hybrid_route")
            overlap = fields.get("overlap_ms")
            if not name or not route or not overlap:
                continue
            mode, split = route_mode_and_split(fields)
            updown = fields.get("draft_pca_updown", "-")
            route_rows.append((name, mode, split, route, updown, float(overlap)))
    return features, route_rows


def raw_selector_rows(path: Path) -> list[list[str]]:
    features_by_prompt, route_rows = parse_raw_route_rows(path)
    baselines: dict[tuple[str, str, str], float] = {}
    candidates: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
    for prompt, mode, split, route, updown, overlap in route_rows:
        if route == "pure" and updown == "-":
            baselines[(prompt, mode, split)] = overlap
        else:
            candidates.setdefault((mode, split, route, updown), {})[prompt] = overlap

    rows: list[list[str]] = []
    for (mode, split, route, updown), by_prompt in candidates.items():
        candidate_baselines = {
            key: value
            for key, value in baselines.items()
            if key[1] == mode and key[2] == split and key[0] in features_by_prompt
        }
        if not candidate_baselines:
            continue
        for feature in FEATURES:
            thresholds = sorted({features_by_prompt[prompt][feature] for prompt, _mode, _split in candidate_baselines})
            for threshold in thresholds:
                for op in ("<=", ">="):
                    selected = wins = losses = ties = 0
                    baseline_total = 0.0
                    policy_total = 0.0
                    deltas: list[float] = []
                    for (prompt, _mode, _split), baseline in candidate_baselines.items():
                        baseline_total += baseline
                        feature_value = features_by_prompt[prompt][feature]
                        use_candidate = feature_value <= threshold if op == "<=" else feature_value >= threshold
                        candidate = by_prompt.get(prompt)
                        if use_candidate and candidate is not None:
                            selected += 1
                            policy_total += candidate
                            delta = (baseline - candidate) * 100.0 / baseline if baseline > 0.0 else 0.0
                            deltas.append(delta)
                            if delta > 0.5:
                                wins += 1
                            elif delta < -0.5:
                                losses += 1
                            else:
                                ties += 1
                        else:
                            policy_total += baseline
                    if selected == 0 or baseline_total <= 0.0:
                        continue
                    total_delta = (baseline_total - policy_total) * 100.0 / baseline_total
                    worst_delta = min(deltas) if deltas else 0.0
                    max_loss = max(0.0, -worst_delta)
                    rows.append(
                        [
                            "0",
                            mode,
                            split,
                            route,
                            updown,
                            feature,
                            op,
                            f"{threshold:.4f}",
                            str(len(candidate_baselines)),
                            str(selected),
                            str(wins),
                            str(losses),
                            str(ties),
                            f"{baseline_total:.3f}",
                            f"{policy_total:.3f}",
                            f"{total_delta:.2f}",
                            f"{worst_delta:.2f}",
                            f"{max_loss:.2f}",
                            "0",
                        ]
                    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--min-logs", type=int, default=1)
    parser.add_argument("--raw", action="store_true", help="Recompute selector policies from raw suite_hybrid rows instead of printed top-N selector rows")
    args = parser.parse_args()

    aggregates: dict[tuple[str, str, str, str, str, str, str], Aggregate] = {}
    for path in args.logs:
        rows = raw_selector_rows(path) if args.raw else parse_selector_rows(path)
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
