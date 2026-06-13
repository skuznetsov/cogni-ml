#!/usr/bin/env python3
"""Emit pp/tg-like rates from DiffusionGemma prompt ABBA TSV artifacts.

These are probe rates, not ordinary autoregressive llama.cpp pp/tg:
- pp_like_tok_s measures prompt-cache construction tokens per second.
- tg_like_rows_s measures sparse canvas rows processed per second.
- tg_like_candidates_s measures sparse candidate-token scores per second.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or "nan")
    except ValueError:
        return float("nan")


def median(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return statistics.median(clean) if clean else float("nan")


def fmt(value: float) -> str:
    return f"{value:.3f}" if not math.isnan(value) and not math.isinf(value) else "NA"


def read_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("run_*.tsv")):
        match = re.match(r"run_(\d+)_(base|variant)\.tsv", path.name)
        if not match:
            continue
        with path.open(newline="", encoding="utf-8") as io:
            for row in csv.DictReader(io, delimiter="\t"):
                row["_arm"] = match.group(2)
                row["_source"] = path.name
                rows.append(row)
    return rows


def summarize_arm(root: Path, case: str, arm: str, rows: list[dict[str, str]]) -> dict[str, object]:
    prompt_len = median([as_float(row, "prompt_len") for row in rows])
    canvas_len = median([as_float(row, "canvas_len") for row in rows])
    candidate_count = median([as_float(row, "candidate_count") for row in rows])
    prompt_cache_ms = median([as_float(row, "prompt_cache_ms") for row in rows])
    prompt_projection_ms = median([as_float(row, "prompt_projection_ms") for row in rows])
    loop_ms = median([as_float(row, "loop_ms_median") for row in rows])
    moe_ms = median([as_float(row, "loop_decode_moe_ffn_ms") for row in rows])
    pp_like_tok_s = prompt_len * 1000.0 / prompt_cache_ms if prompt_cache_ms > 0 else float("nan")
    tg_like_rows_s = canvas_len * 1000.0 / loop_ms if loop_ms > 0 else float("nan")
    tg_like_candidates_s = canvas_len * candidate_count * 1000.0 / loop_ms if loop_ms > 0 else float("nan")
    return {
        "root": str(root),
        "case": case,
        "arm": arm,
        "prompt_len": prompt_len,
        "canvas_len": canvas_len,
        "candidate_count": candidate_count,
        "prompt_cache_ms": prompt_cache_ms,
        "prompt_projection_ms": prompt_projection_ms,
        "loop_ms": loop_ms,
        "moe_ms": moe_ms,
        "pp_like_tok_s": pp_like_tok_s,
        "tg_like_rows_s": tg_like_rows_s,
        "tg_like_candidates_s": tg_like_candidates_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="ABBA log directory with run_*.tsv files")
    args = parser.parse_args()
    rows = read_rows(args.root)
    if not rows:
        raise SystemExit(f"no run_*.tsv rows found in {args.root}")

    summaries: list[dict[str, object]] = []
    for case in sorted({row.get("case", "") for row in rows}):
        for arm in ("base", "variant"):
            arm_rows = [
                row for row in rows
                if row.get("case", "") == case and row.get("_arm") == arm and row.get("status") == "ok"
            ]
            if arm_rows:
                summaries.append(summarize_arm(args.root, case, arm, arm_rows))

    fields = [
        "root", "case", "arm", "prompt_len", "canvas_len", "candidate_count",
        "prompt_cache_ms", "prompt_projection_ms", "pp_like_tok_s",
        "loop_ms", "moe_ms", "tg_like_rows_s", "tg_like_candidates_s",
        "loop_speedup_vs_base", "pp_speedup_vs_base", "tg_rows_speedup_vs_base",
    ]
    print("\t".join(fields))
    by_case_arm = {(str(row["case"]), str(row["arm"])): row for row in summaries}
    for row in summaries:
        base = by_case_arm.get((str(row["case"]), "base"))
        loop_speedup = float("nan")
        pp_speedup = float("nan")
        tg_speedup = float("nan")
        if base and row["arm"] == "variant":
            loop_speedup = float(base["loop_ms"]) / float(row["loop_ms"])
            pp_speedup = float(row["pp_like_tok_s"]) / float(base["pp_like_tok_s"])
            tg_speedup = float(row["tg_like_rows_s"]) / float(base["tg_like_rows_s"])
        values = []
        for field in fields:
            if field == "loop_speedup_vs_base":
                values.append(fmt(loop_speedup))
            elif field == "pp_speedup_vs_base":
                values.append(fmt(pp_speedup))
            elif field == "tg_rows_speedup_vs_base":
                values.append(fmt(tg_speedup))
            elif isinstance(row.get(field), float):
                values.append(fmt(float(row[field])))
            else:
                values.append(str(row.get(field, "")))
        print("\t".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
