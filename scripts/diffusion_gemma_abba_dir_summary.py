#!/usr/bin/env python3
"""Summarize DiffusionGemma ABBA run directories.

Input directories are produced by scripts/diffusion_gemma_prompt_variant_abba.sh.
The summary intentionally preserves quiet-gate status so noisy rows are easy to
separate from promotion evidence.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from pathlib import Path


METRICS = (
    "loop_ms_median",
    "loop_decode_context_ms",
    "loop_decode_qkv_ms",
    "loop_decode_attention_out_ms",
)


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or "nan")
    except ValueError:
        return float("nan")


def median_value(rows: list[dict[str, str]], key: str) -> float:
    values = [as_float(row, key) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return statistics.median(values) if values else float("nan")


def unique_value(rows: list[dict[str, str]], key: str) -> str:
    values = sorted({row.get(key, "") for row in rows if row.get(key, "") != ""})
    return ",".join(values) if values else "NA"


def read_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("run_*.tsv")):
        match = re.match(r"run_(\d+)_(base|variant)\.tsv", path.name)
        if not match:
            continue
        with path.open(newline="", encoding="utf-8") as io:
            for row in csv.DictReader(io, delimiter="\t"):
                row["_index"] = match.group(1)
                row["_arm"] = match.group(2)
                row["_source"] = path.name
                rows.append(row)
    return rows


def quiet_status(root: Path) -> tuple[str, str]:
    statuses: list[str] = []
    for path in sorted(root.glob("run_*.log")):
        with path.open(encoding="utf-8", errors="replace") as io:
            for line in io:
                if not line.startswith("quiet_gate_result "):
                    continue
                fields = {}
                for token in line.strip().split()[1:]:
                    key, _, value = token.partition("=")
                    fields[key] = value
                statuses.append(f"{fields.get('label', path.stem)}:{fields.get('status', 'unknown')}")
    if not statuses:
        return "unknown", "0/0"
    non_ok = [status for status in statuses if not status.endswith(":ok")]
    promotion = "quiet_gate_ok" if not non_ok else "blocked_by_host_noise"
    return promotion, f"{len(non_ok)}/{len(statuses)}"


def snapshot_max(root: Path) -> tuple[float, float, str]:
    max_cpu = float("nan")
    max_total = float("nan")
    max_process = "NA"
    for path in sorted(root.glob("host_snapshot_*.txt")):
        fields = {}
        with path.open(encoding="utf-8", errors="replace") as io:
            for line in io:
                if "=" not in line:
                    continue
                key, value = line.rstrip("\n").split("=", 1)
                fields[key] = value
        try:
            cpu = float(fields.get("max_process_cpu", "nan"))
            total = float(fields.get("total_cpu", "nan"))
        except ValueError:
            continue
        if math.isnan(max_cpu) or cpu > max_cpu:
            max_cpu = cpu
            max_process = fields.get("max_process", "NA")
        if math.isnan(max_total) or total > max_total:
            max_total = total
    return max_cpu, max_total, max_process


def summarize(root: Path) -> list[dict[str, object]]:
    rows = read_rows(root)
    promotion, quiet_non_ok = quiet_status(root)
    max_cpu, max_total, max_process = snapshot_max(root)
    out: list[dict[str, object]] = []
    for case in sorted({row.get("case", "") for row in rows}):
        case_rows = [row for row in rows if row.get("case", "") == case]
        base = [row for row in case_rows if row.get("_arm") == "base" and row.get("status") == "ok"]
        variant = [row for row in case_rows if row.get("_arm") == "variant" and row.get("status") == "ok"]
        if not base or not variant:
            out.append(
                {
                    "root": str(root),
                    "case": case,
                    "status": "incomplete",
                    "promotion_status": promotion,
                    "quiet_non_ok": quiet_non_ok,
                }
            )
            continue
        summary: dict[str, object] = {
            "root": str(root),
            "case": case,
            "status": "ok",
            "promotion_status": promotion,
            "quiet_non_ok": quiet_non_ok,
            "base_backend": unique_value(base, "loop_context_backend"),
            "variant_backend": unique_value(variant, "loop_context_backend"),
            "base_batch_rows": unique_value(base, "loop_context_batch_rows"),
            "variant_batch_rows": unique_value(variant, "loop_context_batch_rows"),
            "base_fixed_gqa2": unique_value(base, "loop_context_fixed_gqa2"),
            "variant_fixed_gqa2": unique_value(variant, "loop_context_fixed_gqa2"),
            "max_process_cpu": f"{max_cpu:.1f}" if not math.isnan(max_cpu) else "NA",
            "max_total_cpu": f"{max_total:.1f}" if not math.isnan(max_total) else "NA",
            "max_process": max_process,
        }
        for metric in METRICS:
            base_metric = median_value(base, metric)
            variant_metric = median_value(variant, metric)
            ratio = variant_metric / base_metric if base_metric else float("inf")
            summary[f"{metric}_base_ms"] = f"{base_metric:.3f}" if not math.isnan(base_metric) else "NA"
            summary[f"{metric}_variant_ms"] = f"{variant_metric:.3f}" if not math.isnan(variant_metric) else "NA"
            summary[f"{metric}_speedup"] = f"{1.0 / ratio:.4f}" if ratio and not math.isnan(ratio) else "NA"
        out.append(summary)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for root in args.roots:
        rows.extend(summarize(root))

    fields = [
        "root",
        "case",
        "status",
        "promotion_status",
        "quiet_non_ok",
        "base_backend",
        "variant_backend",
        "base_batch_rows",
        "variant_batch_rows",
        "base_fixed_gqa2",
        "variant_fixed_gqa2",
        "loop_ms_median_base_ms",
        "loop_ms_median_variant_ms",
        "loop_ms_median_speedup",
        "loop_decode_context_ms_base_ms",
        "loop_decode_context_ms_variant_ms",
        "loop_decode_context_ms_speedup",
        "loop_decode_qkv_ms_base_ms",
        "loop_decode_qkv_ms_variant_ms",
        "loop_decode_qkv_ms_speedup",
        "loop_decode_attention_out_ms_base_ms",
        "loop_decode_attention_out_ms_variant_ms",
        "loop_decode_attention_out_ms_speedup",
        "max_process_cpu",
        "max_total_cpu",
        "max_process",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
