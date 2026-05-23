#!/usr/bin/env python3
"""Mine simple chunk-local route selector policies from self-spec cycle dumps.

This is intentionally conservative: it only uses features available before a
chunk route decision in a running controller (prompt category/name, gamma,
position, and previous pure-route chunk outcome). It does not use the current
candidate route outcome as a trigger.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median


@dataclass(frozen=True)
class Row:
    prompt: str
    category: str
    gamma_label: str
    position: int
    route: str
    wall_ms: float
    accepted: int
    proposed: int
    reject_index: int

    @property
    def full_accept(self) -> bool:
        return self.proposed > 0 and self.accepted >= self.proposed and self.reject_index < 0


def route_from_policy(policy: str) -> str:
    marker = "/route="
    return policy.split(marker, 1)[1] if marker in policy else "pure"


def read_rows(paths: list[Path]) -> list[Row]:
    rows: list[Row] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("kind") == "self_lowrank_exact_suffix":
                continue
            rows.append(
                Row(
                    prompt=str(obj["prompt_name"]),
                    category=str(obj.get("prompt_category", "unknown")),
                    gamma_label=str(obj.get("gamma_label", obj.get("gamma", ""))),
                    position=int(obj["position"]),
                    route=route_from_policy(str(obj["policy"])),
                    wall_ms=float(obj["wall_ms"]),
                    accepted=int(obj["accepted_count"]),
                    proposed=int(obj["proposed_count"]),
                    reject_index=int(obj["reject_index"]),
                )
            )
    return rows


@dataclass(frozen=True)
class Group:
    key: tuple[str, str, int]
    prompt: str
    category: str
    gamma_label: str
    position: int
    pure: Row
    by_route: dict[str, Row]
    prev_pure: Row | None


def build_groups(rows: list[Row]) -> list[Group]:
    grouped: dict[tuple[str, str, int], list[Row]] = defaultdict(list)
    for row in rows:
        grouped[(row.prompt, row.gamma_label, row.position)].append(row)

    pure_by_prompt_gamma: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for group_rows in grouped.values():
        pure_rows = [row for row in group_rows if row.route == "pure"]
        if pure_rows:
            pure = min(pure_rows, key=lambda row: row.wall_ms)
            pure_by_prompt_gamma[(pure.prompt, pure.gamma_label)].append(pure)
    for values in pure_by_prompt_gamma.values():
        values.sort(key=lambda row: row.position)

    groups: list[Group] = []
    for key, group_rows in grouped.items():
        pure_rows = [row for row in group_rows if row.route == "pure"]
        if not pure_rows:
            continue
        pure = min(pure_rows, key=lambda row: row.wall_ms)
        by_route: dict[str, Row] = {}
        for row in group_rows:
            if row.route == "pure":
                continue
            old = by_route.get(row.route)
            if old is None or row.wall_ms < old.wall_ms:
                by_route[row.route] = row
        prev = None
        ordered = pure_by_prompt_gamma[(pure.prompt, pure.gamma_label)]
        for candidate in ordered:
            if candidate.position < pure.position:
                prev = candidate
            else:
                break
        groups.append(Group(key, pure.prompt, pure.category, pure.gamma_label, pure.position, pure, by_route, prev))
    groups.sort(key=lambda g: (g.prompt, g.gamma_label, g.position))
    return groups


def condition_catalog(groups: list[Group]) -> list[tuple[str, object, callable]]:
    cats = sorted({g.category for g in groups})
    prompts = sorted({g.prompt for g in groups})
    gammas = sorted({g.gamma_label for g in groups})
    positions = sorted({g.position for g in groups})
    thresholds = sorted(set(positions + [4, 8, 12, 16, 24, 32]))
    conds: list[tuple[str, object, callable]] = [("all", True, lambda g: True)]
    conds += [("category", c, lambda g, c=c: g.category == c) for c in cats]
    conds += [("prompt", p, lambda g, p=p: g.prompt == p) for p in prompts]
    conds += [("gamma", ga, lambda g, ga=ga: g.gamma_label == ga) for ga in gammas]
    conds += [("position>=", t, lambda g, t=t: g.position >= t) for t in thresholds]
    conds += [("position<=", t, lambda g, t=t: g.position <= t) for t in thresholds]
    conds += [
        ("prev_full_accept", True, lambda g: g.prev_pure is not None and g.prev_pure.full_accept),
        ("prev_reject", True, lambda g: g.prev_pure is not None and g.prev_pure.reject_index >= 0),
        ("prev_partial", True, lambda g: g.prev_pure is not None and not g.prev_pure.full_accept),
        ("first_chunk", True, lambda g: g.prev_pure is None),
    ]
    return conds


def score_policy(groups: list[Group], route: str, pred) -> dict[str, object] | None:
    selected = []
    deltas = []
    total_pure = 0.0
    total_policy = 0.0
    losses = 0
    rejects = 0
    missing = 0
    for g in groups:
        pure_ms = g.pure.wall_ms
        chosen = g.pure
        if pred(g):
            candidate = g.by_route.get(route)
            if candidate is None:
                missing += 1
            else:
                chosen = candidate
                selected.append((g, candidate))
                delta = (pure_ms - candidate.wall_ms) * 100.0 / pure_ms if pure_ms > 0 else 0.0
                deltas.append(delta)
                if delta < 0.0:
                    losses += 1
                if candidate.reject_index >= 0:
                    rejects += 1
        total_pure += pure_ms
        total_policy += chosen.wall_ms
    if not selected:
        return None
    total_delta = (total_pure - total_policy) * 100.0 / total_pure if total_pure > 0 else 0.0
    return {
        "route": route,
        "selected": len(selected),
        "missing": missing,
        "losses": losses,
        "rejects": rejects,
        "total_delta": total_delta,
        "median_delta": median(deltas),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "selected_groups": selected,
    }


def condition_label(kind: str, value: object) -> str:
    if kind.endswith((">=", "<=")):
        return f"{kind}{value}"
    return f"{kind}={value}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--min-selected", type=int, default=2)
    parser.add_argument("--allow-losses", action="store_true")
    args = parser.parse_args()

    groups = build_groups(read_rows(args.jsonl))
    routes = sorted({route for g in groups for route in g.by_route})
    print(f"route_cycle_selector_mine groups={len(groups)} routes={','.join(routes)}")
    candidates = []
    for route in routes:
        for kind, value, pred in condition_catalog(groups):
            score = score_policy(groups, route, pred)
            if score is None or score["selected"] < args.min_selected:
                continue
            if not args.allow_losses and score["losses"] > 0:
                continue
            score["condition"] = condition_label(kind, value)
            candidates.append(score)
    candidates.sort(key=lambda s: (s["total_delta"], s["median_delta"], -s["losses"]), reverse=True)
    print("rank route condition selected losses rejects missing total_delta% median_delta% min_delta% max_delta%")
    for i, s in enumerate(candidates[: args.limit], 1):
        print(
            f"{i} {s['route']} {s['condition']} {s['selected']} {s['losses']} {s['rejects']} {s['missing']} "
            f"{s['total_delta']:.2f} {s['median_delta']:.2f} {s['min_delta']:.2f} {s['max_delta']:.2f}"
        )
    if candidates:
        best = candidates[0]
        print("best_groups prompt gamma position category pure_ms route_ms delta% accepted proposed reject_index prev_full prev_reject")
        for g, row in best["selected_groups"][: args.limit]:
            delta = (g.pure.wall_ms - row.wall_ms) * 100.0 / g.pure.wall_ms if g.pure.wall_ms > 0 else 0.0
            prev_full = g.prev_pure.full_accept if g.prev_pure else False
            prev_reject = g.prev_pure.reject_index >= 0 if g.prev_pure else False
            print(
                f"{g.prompt} {g.gamma_label} {g.position} {g.category} {g.pure.wall_ms:.3f} {row.wall_ms:.3f} "
                f"{delta:.2f} {row.accepted} {row.proposed} {row.reject_index} {prev_full} {prev_reject}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
