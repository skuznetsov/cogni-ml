#!/usr/bin/env python3
"""Emit pp/tg-like rates from DiffusionGemma prompt ABBA artifacts.

These are probe rates, not ordinary autoregressive llama.cpp pp/tg:
- pp_like_tok_s measures prompt-cache construction tokens per second.
- tg_like_rows_s measures sparse canvas rows processed per second.
- tg_like_candidates_s measures sparse candidate-token scores per second.
- suite logs additionally report prompt-layer-row throughput and mixed
  fast/exact fallback accounting from artifact-suite gates.
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


def parse_kv_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in line.split()[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields.setdefault(key, value.strip("'\""))
    return fields


def parse_scalar_line(lines: list[str], key: str) -> float:
    prefix = f"{key}="
    for line in lines:
        if line.startswith(prefix):
            try:
                return float(line[len(prefix):])
            except ValueError:
                return float("nan")
    return float("nan")


def first_existing_suite_log(path: Path) -> Path | None:
    if path.is_file():
        return path
    for name in ("gate.stdout", "promotion.stdout", "stdout"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    matches = sorted(path.glob("*.stdout"))
    return matches[0] if matches else None


def parse_child_shape(lines: list[str]) -> tuple[float, float, float]:
    prompt_len = parse_scalar_line(lines, "prompt_len")
    canvas_len = parse_scalar_line(lines, "canvas_len")
    max_layers = parse_scalar_line(lines, "max_layers")
    if not (math.isnan(prompt_len) or math.isnan(canvas_len) or math.isnan(max_layers)):
        return prompt_len, canvas_len, max_layers

    for line in lines:
        if not line.startswith("suite_window "):
            continue
        fields = parse_kv_fields(line)
        child_log = fields.get("log")
        if not child_log:
            continue
        child_path = Path(child_log)
        if not child_path.is_file():
            continue
        child_lines = child_path.read_text(encoding="utf-8", errors="replace").splitlines()
        prompt_len = parse_scalar_line(child_lines, "prompt_len")
        canvas_len = parse_scalar_line(child_lines, "canvas_len")
        max_layers = parse_scalar_line(child_lines, "max_layers")
        if not (math.isnan(prompt_len) or math.isnan(canvas_len) or math.isnan(max_layers)):
            return prompt_len, canvas_len, max_layers
    return prompt_len, canvas_len, max_layers


def rate(work : float, ms : float) -> float:
    return work * 1000.0 / ms if work > 0 and ms > 0 else float("nan")


def parse_suite_summary(path: Path) -> list[dict[str, object]] | None:
    suite_log = first_existing_suite_log(path)
    if not suite_log:
        return None
    lines = suite_log.read_text(encoding="utf-8", errors="replace").splitlines()
    if not any(line.startswith("suite_summary ") or line.startswith("suite_compat_summary ") for line in lines):
        return None

    compat_line = next((line for line in reversed(lines) if line.startswith("suite_compat_summary ")), "")
    summary_line = next((line for line in reversed(lines) if line.startswith("suite_summary ")), "")
    decision_line = next((line for line in reversed(lines) if line.startswith("artifact_suite_gate decision=")), "")
    promotion_line = next((line for line in reversed(lines) if line.startswith("artifact_suite_promotion decision=")), "")
    fields = parse_kv_fields(compat_line or summary_line)
    summary_fields = parse_kv_fields(summary_line) if summary_line else {}
    decision_fields = parse_kv_fields(decision_line) if decision_line else {}
    promotion_fields = parse_kv_fields(promotion_line) if promotion_line else {}

    prompt_len, canvas_len, max_layers = parse_child_shape(lines)
    windows = float(fields.get("windows", summary_fields.get("windows", "nan")))
    base_ms = float(fields.get("base_ms", summary_fields.get("base_ms", "nan")))
    unsafe_variant_ms = float(fields.get("unsafe_variant_ms", summary_fields.get("variant_ms", "nan")))
    mixed_variant_ms = float(fields.get("mixed_variant_ms", fields.get("variant_ms", "nan")))
    candidate_windows = float(fields.get("candidate_windows", "nan"))
    fallback_windows = float(fields.get("fallback_windows", "nan"))
    aggregate_speedup = float(fields.get("unsafe_speedup", summary_fields.get("aggregate_speedup", "nan")))
    mixed_speedup = float(fields.get("mixed_speedup", fields.get("aggregate_speedup", "nan")))

    prompt_token_work = windows * prompt_len
    canvas_row_work = windows * canvas_len
    layer_work = windows * max_layers
    prompt_layer_row_work = prompt_token_work * max_layers

    return [{
        "source": str(suite_log),
        "kind": "artifact_suite",
        "gate_decision": decision_fields.get("decision", ""),
        "promotion_decision": promotion_fields.get("decision", ""),
        "windows": windows,
        "candidate_windows": candidate_windows,
        "fallback_windows": fallback_windows,
        "prompt_len": prompt_len,
        "canvas_len": canvas_len,
        "max_layers": max_layers,
        "base_ms": base_ms,
        "unsafe_variant_ms": unsafe_variant_ms,
        "mixed_variant_ms": mixed_variant_ms,
        "unsafe_speedup": aggregate_speedup,
        "mixed_speedup": mixed_speedup,
        "base_pp_like_tok_s": rate(prompt_token_work, base_ms),
        "unsafe_pp_like_tok_s": rate(prompt_token_work, unsafe_variant_ms),
        "mixed_pp_like_tok_s": rate(prompt_token_work, mixed_variant_ms),
        "base_canvas_rows_s": rate(canvas_row_work, base_ms),
        "unsafe_canvas_rows_s": rate(canvas_row_work, unsafe_variant_ms),
        "mixed_canvas_rows_s": rate(canvas_row_work, mixed_variant_ms),
        "base_layers_s": rate(layer_work, base_ms),
        "unsafe_layers_s": rate(layer_work, unsafe_variant_ms),
        "mixed_layers_s": rate(layer_work, mixed_variant_ms),
        "base_prompt_layer_rows_s": rate(prompt_layer_row_work, base_ms),
        "unsafe_prompt_layer_rows_s": rate(prompt_layer_row_work, unsafe_variant_ms),
        "mixed_prompt_layer_rows_s": rate(prompt_layer_row_work, mixed_variant_ms),
    }]


def print_suite_summary(rows: list[dict[str, object]]) -> None:
    fields = [
        "source", "kind", "gate_decision", "promotion_decision",
        "windows", "candidate_windows", "fallback_windows",
        "prompt_len", "canvas_len", "max_layers",
        "base_ms", "unsafe_variant_ms", "mixed_variant_ms",
        "unsafe_speedup", "mixed_speedup",
        "base_pp_like_tok_s", "unsafe_pp_like_tok_s", "mixed_pp_like_tok_s",
        "base_canvas_rows_s", "unsafe_canvas_rows_s", "mixed_canvas_rows_s",
        "base_layers_s", "unsafe_layers_s", "mixed_layers_s",
        "base_prompt_layer_rows_s", "unsafe_prompt_layer_rows_s", "mixed_prompt_layer_rows_s",
    ]
    print("\t".join(fields))
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value))
        print("\t".join(values))


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
    parser.add_argument("root", type=Path, help="ABBA log directory with run_*.tsv files, or artifact-suite stdout/log directory")
    args = parser.parse_args()

    suite_rows = parse_suite_summary(args.root)
    if suite_rows:
        print_suite_summary(suite_rows)
        return 0

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
