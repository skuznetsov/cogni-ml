#!/usr/bin/env python3
"""Summarize closed-book vs memory-grounded CogniQwen surrogate probe rows.

Reads stdout/logs from bin/qwen35_deltanet_fixed_basis_probe.cr and compares
prompt names ending in `_closed` with matching `_memory` or `_mem` rows.
This is intentionally format-tolerant: missing metrics stay blank instead of
making the falsifier look stronger than the evidence.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

PAIR_SUFFIXES = ("_closed", "_memory", "_mem")
NAME_RE = re.compile(r"\bname=([^\s]+)")
RANK_RE = re.compile(r"\brank=([^\s]+)")
LAYERS_RE = re.compile(r"\blayers=([^\s]+)")
ROW_TYPES = {
    "lowrank_eval_logit": "logit",
    "lowrank_eval_greedy": "greedy",
    "lowrank_eval_self_spec": "self_spec",
    "self_spec_gpu_pipeline_suite_hybrid": "gpu_hybrid",
    "self_spec_gpu_pipeline_route_selector": "route_selector",
    "self_spec_gpu_pipeline_suite": "suite_features",
    "self_spec_prompt_route_features": "route_features",
    "self_spec_ffn_updown_route_features": "ffn_updown_features",
    "current_hidden_proposal": "current_hidden",
}
METRIC_KEYS = (
    "top1",
    "topk",
    "top1_match",
    "top5_hit",
    "accept_rate",
    "rejections",
    "plain_speedup",
    "overlap_ms",
    "plain_exact_ms",
    "residual_mean",
    "residual_p90",
    "residual_max",
    "rel_rmse_mean",
    "cos_mean",
    "avg_best_cos",
    "proposal_ms_per_eval",
)


def parse_scalar(raw: str):
    value = raw.rstrip("%x")
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return raw


def row_kind(line: str) -> str | None:
    first = line.split(maxsplit=1)[0] if line.strip() else ""
    return ROW_TYPES.get(first)


def pair_key(name: str) -> Tuple[str, str] | None:
    if name.endswith("_closed"):
        return name[: -len("_closed")], "closed"
    if name.endswith("_memory"):
        return name[: -len("_memory")], "memory"
    if name.endswith("_mem"):
        return name[: -len("_mem")], "memory"
    return None


def parse_metrics(line: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, raw = token.split("=", 1)
        if key in METRIC_KEYS:
            out[key] = parse_scalar(raw)
    return out


def load_rows(paths: Iterable[Path]):
    rows = defaultdict(dict)
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                kind = row_kind(line)
                if not kind:
                    continue
                name_m = NAME_RE.search(line)
                if not name_m:
                    continue
                name = name_m.group(1)
                pair = pair_key(name)
                if not pair:
                    continue
                rank_m = RANK_RE.search(line)
                rank = rank_m.group(1) if rank_m else "na"
                layers_m = LAYERS_RE.search(line)
                layers = layers_m.group(1) if layers_m else "na"
                base, side = pair
                metrics = parse_metrics(line)
                if metrics:
                    rows[(base, kind, rank, layers)][side] = metrics
    return rows


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def delta(memory, closed):
    if isinstance(memory, (int, float)) and isinstance(closed, (int, float)):
        return memory - closed
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--metric", action="append", dest="metrics", default=[], help="Metric to print; may repeat")
    args = parser.parse_args()

    metrics = args.metrics or ["top1_match", "accept_rate", "plain_speedup", "residual_mean", "rel_rmse_mean"]
    rows = load_rows(args.logs)
    if not rows:
        print("No paired rows found. Expected names like task_closed and task_memory.")
        return 1

    header = ["pair", "row", "rank", "layers"]
    for metric in metrics:
        header.extend([f"closed_{metric}", f"memory_{metric}", f"delta_{metric}"])
    print("\t".join(header))

    for (base, kind, rank, layers), sides in sorted(rows.items()):
        closed = sides.get("closed", {})
        memory = sides.get("memory", {})
        if not closed or not memory:
            continue
        line = [base, kind, rank, layers]
        for metric in metrics:
            c = closed.get(metric)
            m = memory.get(metric)
            line.extend([fmt(c), fmt(m), fmt(delta(m, c))])
        print("\t".join(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
