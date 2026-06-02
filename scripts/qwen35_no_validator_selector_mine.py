#!/usr/bin/env python3
"""Mine cheap prompt-feature route selectors for no-validator Qwen code routes.

This intentionally avoids task-name equality as a feature. The goal is to test
whether prompt-visible structural cues can approximate the route oracle before
we spend model time on a real router.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEFAULT_SUFFIXES = (
    "_strict_code",
    "_raw_fence",
    "_file_prefill",
    "_final_code",
    "_fence_prefill",
    "_fence",
    "_strict",
)

ROUTE_SUFFIXES = (
    "_exactfirst16",
    "_exactfirst8",
    "_exactfirst1",
    "_exactfirst0",
)


@dataclass(frozen=True)
class ScoreRow:
    group: str
    route: str
    name: str
    task: str
    score: float
    speed_ratio: float
    think_leak: int
    substantive_code: int
    repair_ok: int
    status: str


@dataclass(frozen=True)
class TaskExample:
    task: str
    prompt: str
    scores: dict[str, ScoreRow]


def ffloat(value: str | None) -> float:
    try:
        return float(value or "0")
    except ValueError:
        return 0.0


def normalize_name(name: str, suffixes: tuple[str, ...]) -> str:
    out = name
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if out.endswith(suffix):
                out = out[: -len(suffix)]
                changed = True
                break
    return out


def route_key(route: str) -> str:
    out = route
    for suffix in ROUTE_SUFFIXES:
        out = out.replace(suffix, suffix)
    return out


def load_prompts(paths: list[Path], suffixes: tuple[str, ...]) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            raise SystemExit(f"missing prompts file: {path}")
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            obj = json.loads(raw)
            name = str(obj["name"])
            task = normalize_name(name, suffixes)
            prompts.setdefault(task, str(obj["prompt"]))
    return prompts


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


def load_scores(inputs: list[str], suffixes: tuple[str, ...]) -> dict[str, dict[str, ScoreRow]]:
    by_task: dict[str, dict[str, ScoreRow]] = {}
    for group, path in map(quality_path, inputs):
        with path.open(newline="", encoding="utf-8") as io:
            for raw in csv.DictReader(io, delimiter="\t"):
                name = raw.get("name", "")
                task = normalize_name(name, suffixes)
                route = raw.get("route", "")
                key = route_key(route)
                row = ScoreRow(
                    group=group,
                    route=route,
                    name=name,
                    task=task,
                    score=ffloat(raw.get("score")),
                    speed_ratio=ffloat(raw.get("speed_ratio")),
                    think_leak=int(raw.get("think_leak", "0") or 0),
                    substantive_code=int(raw.get("substantive_code", "0") or 0),
                    repair_ok=int(raw.get("repair_ok", "0") or 0),
                    status=raw.get("status", ""),
                )
                old = by_task.setdefault(task, {}).get(key)
                if old is None or row.score > old.score:
                    by_task[task][key] = row
    return by_task


def prompt_features(prompt: str) -> dict[str, float | bool]:
    p = prompt.lower()
    bullets = len(re.findall(r"(?m)^- ", prompt))
    features: dict[str, float | bool] = {
        "has_parser": "parser" in p or "parse" in p or "parsing" in p,
        "has_streaming": "stream" in p or "incremental" in p or "feed(" in p,
        "has_time": "time" in p or "clock" in p or "rate" in p or "elapsed" in p,
        "has_range": "range" in p or "interval" in p,
        "has_capacity": "capacity" in p,
        "has_generic": "generic" in p or "(t)" in p or "keys and values" in p,
        "has_errors": "error" in p or "errors" in p,
        "has_line_numbers": "line number" in p or "line numbers" in p,
        "has_filesystem": "file system" in p,
        "has_json": "json" in p,
        "has_cycle": "cycle" in p,
        "has_cache": "cache" in p,
        "has_quotes": "quoted" in p or "quotes" in p,
        "has_comments": "comments" in p,
        "bullet_count": float(bullets),
        "prompt_len": float(len(prompt)),
    }
    return features


def conditions(examples: list[TaskExample]) -> list[tuple[str, Callable[[TaskExample], bool]]]:
    feature_values = {ex.task: prompt_features(ex.prompt) for ex in examples}
    bool_features = sorted(k for k, v in next(iter(feature_values.values())).items() if isinstance(v, bool))
    num_features = sorted(k for k, v in next(iter(feature_values.values())).items() if not isinstance(v, bool))
    conds: list[tuple[str, Callable[[TaskExample], bool]]] = [("all", lambda ex: True)]
    for feat in bool_features:
        conds.append((feat, lambda ex, feat=feat: bool(feature_values[ex.task][feat])))
        conds.append((f"not_{feat}", lambda ex, feat=feat: not bool(feature_values[ex.task][feat])))
    for feat in num_features:
        vals = sorted({float(feature_values[ex.task][feat]) for ex in examples})
        for threshold in vals:
            conds.append((f"{feat}<={threshold:g}", lambda ex, feat=feat, threshold=threshold: float(feature_values[ex.task][feat]) <= threshold))
            conds.append((f"{feat}>={threshold:g}", lambda ex, feat=feat, threshold=threshold: float(feature_values[ex.task][feat]) >= threshold))
    return conds


def baseline_route(examples: list[TaskExample]) -> str:
    routes = sorted({route for ex in examples for route in ex.scores})
    best = max(routes, key=lambda route: statistics.mean(ex.scores.get(route, ScoreRow("", route, "", ex.task, 0.0, 0.0, 0, 0, 0, "")).score for ex in examples))
    return best


def policy_score(examples: list[TaskExample], default_route: str, selected_route: str, pred: Callable[[TaskExample], bool]) -> dict[str, float | int | str]:
    total = 0.0
    default_total = 0.0
    oracle_total = 0.0
    selected = losses = wins = missing = 0
    for ex in examples:
        default_score = ex.scores.get(default_route).score if default_route in ex.scores else 0.0
        oracle_score = max((row.score for row in ex.scores.values()), default=0.0)
        chosen_score = default_score
        if pred(ex):
            selected += 1
            row = ex.scores.get(selected_route)
            if row is None:
                missing += 1
            else:
                chosen_score = row.score
                if chosen_score < default_score:
                    losses += 1
                elif chosen_score > default_score:
                    wins += 1
        total += chosen_score
        default_total += default_score
        oracle_total += oracle_score
    n = max(1, len(examples))
    return {
        "mean": total / n,
        "default_mean": default_total / n,
        "oracle_mean": oracle_total / n,
        "delta": total / n - default_total / n,
        "capture": (total - default_total) / (oracle_total - default_total) if oracle_total > default_total else 0.0,
        "selected": selected,
        "wins": wins,
        "losses": losses,
        "missing": missing,
    }


def mine(examples: list[TaskExample], *, allow_losses: bool) -> list[dict[str, object]]:
    default = baseline_route(examples)
    routes = sorted({route for ex in examples for route in ex.scores if route != default})
    out: list[dict[str, object]] = []
    for route in routes:
        for label, pred in conditions(examples):
            s = policy_score(examples, default, route, pred)
            if s["selected"] == 0:
                continue
            if not allow_losses and s["losses"]:
                continue
            row = dict(s)
            row.update({"default_route": default, "route": route, "condition": label})
            out.append(row)
    out.sort(key=lambda r: (float(r["mean"]), float(r["capture"]), -int(r["losses"])), reverse=True)
    return out


def leave_one_out(examples: list[TaskExample], *, allow_losses: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for held in examples:
        train = [ex for ex in examples if ex.task != held.task]
        policies = mine(train, allow_losses=allow_losses)
        if not policies:
            default = baseline_route(train)
            chosen_route = default
            condition = "default_only"
            fires = False
        else:
            best = policies[0]
            default = str(best["default_route"])
            chosen_route = str(best["route"])
            condition = str(best["condition"])
            pred = dict(conditions([held]))
            # Rebuild the predicate on train+held so threshold features are valid.
            full_preds = dict(conditions(train + [held]))
            fires = full_preds.get(condition, lambda _ex: False)(held)
            if not fires:
                chosen_route = default
        default_score = held.scores.get(default).score if default in held.scores else 0.0
        chosen_score = held.scores.get(chosen_route).score if chosen_route in held.scores else 0.0
        oracle_score = max((r.score for r in held.scores.values()), default=0.0)
        rows.append({
            "task": held.task,
            "default_route": default,
            "condition": condition,
            "fired": int(fires),
            "chosen_route": chosen_route,
            "chosen_score": chosen_score,
            "default_score": default_score,
            "oracle_score": oracle_score,
            "delta": chosen_score - default_score,
            "oracle_gap_left": oracle_score - chosen_score,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("quality", nargs="+", help="quality_per_prompt.tsv path/dir or group=path")
    ap.add_argument("--prompts", action="append", type=Path, required=True, help="Prompt JSONL; may repeat")
    ap.add_argument("--suffix", action="append", default=[], help="Additional prompt-name suffix to strip")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--allow-losses", action="store_true")
    args = ap.parse_args()

    suffixes = tuple(args.suffix) + DEFAULT_SUFFIXES
    prompts = load_prompts(args.prompts, suffixes)
    scores = load_scores(args.quality, suffixes)
    examples = [TaskExample(task=t, prompt=prompts[t], scores=scores[t]) for t in sorted(prompts) if t in scores]
    if not examples:
        raise SystemExit("no examples matched prompts to score rows")

    policies = mine(examples, allow_losses=args.allow_losses)
    default = baseline_route(examples)
    oracle_mean = statistics.mean(max((row.score for row in ex.scores.values()), default=0.0) for ex in examples)
    default_mean = statistics.mean(ex.scores.get(default).score if default in ex.scores else 0.0 for ex in examples)
    print(f"selector_mine tasks={len(examples)} default_route={default} default_mean={default_mean:.3f} oracle_mean={oracle_mean:.3f} oracle_gap={oracle_mean-default_mean:.3f}")
    print("rank route condition mean delta capture selected wins losses missing")
    for i, row in enumerate(policies[: args.limit], 1):
        print(
            f"{i} {row['route']} {row['condition']} {float(row['mean']):.3f} {float(row['delta']):.3f} "
            f"{float(row['capture']):.3f} {row['selected']} {row['wins']} {row['losses']} {row['missing']}"
        )
    loo = leave_one_out(examples, allow_losses=args.allow_losses)
    loo_mean = statistics.mean(float(row["chosen_score"]) for row in loo)
    print(f"loo_mean={loo_mean:.3f} loo_delta_vs_default={loo_mean-default_mean:.3f}")
    print("loo task chosen_route chosen_score default_route default_score oracle_score delta oracle_gap_left condition fired")
    for row in loo:
        print(
            f"{row['task']} {row['chosen_route']} {float(row['chosen_score']):.3f} {row['default_route']} "
            f"{float(row['default_score']):.3f} {float(row['oracle_score']):.3f} {float(row['delta']):.3f} "
            f"{float(row['oracle_gap_left']):.3f} {row['condition']} {row['fired']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
