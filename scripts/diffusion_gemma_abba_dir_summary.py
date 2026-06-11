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
    "loop_prediction_ms",
    "loop_decode_stack_ms",
    "loop_decode_context_ms",
    "loop_decode_qkv_ms",
    "loop_decode_attention_out_ms",
    "loop_decode_shared_ffn_ms",
    "loop_decode_moe_ffn_ms",
    "loop_decode_combine_scale_ms",
    "loop_output_head_ms",
    "loop_update_ms",
    "loop_regenerate_ms",
    "loop_proposal_ms",
)

TRACKED_PHASE_METRICS = (
    "loop_decode_context_ms",
    "loop_decode_qkv_ms",
    "loop_decode_attention_out_ms",
    "loop_decode_shared_ffn_ms",
    "loop_decode_moe_ffn_ms",
    "loop_decode_combine_scale_ms",
    "loop_output_head_ms",
    "loop_update_ms",
    "loop_regenerate_ms",
    "loop_proposal_ms",
)

PHASE_CONFIDENCE_METRICS = (
    "loop_decode_context_ms",
    "loop_decode_qkv_ms",
    "loop_decode_attention_out_ms",
    "loop_decode_shared_ffn_ms",
    "loop_decode_moe_ffn_ms",
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


def range_value(rows: list[dict[str, str]], key: str) -> float:
    values = [as_float(row, key) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return max(values) - min(values) if values else float("nan")


def unique_value(rows: list[dict[str, str]], key: str) -> str:
    values = sorted({row.get(key, "") for row in rows if row.get(key, "") != ""})
    return ",".join(values) if values else "NA"


def format_ms(value: float) -> str:
    return f"{value:.3f}" if not math.isnan(value) else "NA"


def format_ratio(value: float) -> str:
    return f"{value:.4f}" if not math.isnan(value) else "NA"


def phase_delta(base: list[dict[str, str]], variant: list[dict[str, str]], metric: str) -> float:
    return median_value(variant, metric) - median_value(base, metric)


def dominant_phase(deltas: dict[str, float], reverse: bool) -> tuple[str, float]:
    candidates = {key: value for key, value in deltas.items() if not math.isnan(value)}
    if not candidates:
        return "NA", float("nan")
    if reverse:
        key = max(candidates, key=lambda item: candidates[item])
    else:
        key = min(candidates, key=lambda item: candidates[item])
    return key, candidates[key]


def range_over_delta(base: list[dict[str, str]], variant: list[dict[str, str]], metric: str) -> float:
    delta = abs(phase_delta(base, variant, metric))
    combined_range = range_value(base, metric) + range_value(variant, metric)
    if math.isnan(delta) or math.isnan(combined_range):
        return float("nan")
    if delta == 0.0:
        return float("inf") if combined_range > 0.0 else 0.0
    return combined_range / delta


def delta_confidence(range_delta_ratio: float) -> str:
    if math.isnan(range_delta_ratio) or math.isinf(range_delta_ratio):
        return "unstable"
    return "unstable" if range_delta_ratio > 1.0 else "range_bounded"


def promotion_decision(promotion_status: str, loop_delta: float, loop_confidence: str) -> str:
    if promotion_status != "quiet_gate_ok":
        return "blocked_by_host_noise"
    if loop_confidence != "range_bounded":
        return "blocked_by_range"
    if math.isnan(loop_delta):
        return "blocked_missing_delta"
    if loop_delta < 0.0:
        return "candidate_speedup"
    if loop_delta == 0.0:
        return "neutral"
    return "reject_regression"


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
            delta = variant_metric - base_metric
            summary[f"{metric}_base_ms"] = format_ms(base_metric)
            summary[f"{metric}_variant_ms"] = format_ms(variant_metric)
            summary[f"{metric}_delta_ms"] = format_ms(delta)
            summary[f"{metric}_speedup"] = format_ratio(1.0 / ratio) if ratio else "NA"
            summary[f"{metric}_base_range_ms"] = format_ms(range_value(base, metric))
            summary[f"{metric}_variant_range_ms"] = format_ms(range_value(variant, metric))
        tracked_deltas = {metric: phase_delta(base, variant, metric) for metric in TRACKED_PHASE_METRICS}
        tracked_delta = sum(tracked_deltas.values())
        loop_delta = median_value(variant, "loop_ms_median") - median_value(base, "loop_ms_median")
        positive_phase, positive_delta = dominant_phase(tracked_deltas, reverse=True)
        negative_phase, negative_delta = dominant_phase(tracked_deltas, reverse=False)
        summary["tracked_phase_delta_sum_ms"] = format_ms(tracked_delta)
        summary["untracked_delta_ms"] = format_ms(loop_delta - tracked_delta)
        summary["dominant_positive_phase"] = positive_phase
        summary["dominant_positive_delta_ms"] = format_ms(positive_delta)
        summary["dominant_negative_phase"] = negative_phase
        summary["dominant_negative_delta_ms"] = format_ms(negative_delta)
        loop_range_over_delta = range_over_delta(base, variant, "loop_ms_median")
        loop_confidence = delta_confidence(loop_range_over_delta)
        summary["loop_range_over_delta"] = format_ratio(loop_range_over_delta)
        summary["delta_confidence"] = loop_confidence
        summary["promotion_decision"] = promotion_decision(promotion, loop_delta, loop_confidence)
        for metric in PHASE_CONFIDENCE_METRICS:
            metric_range_over_delta = range_over_delta(base, variant, metric)
            summary[f"{metric}_range_over_delta"] = format_ratio(metric_range_over_delta)
            summary[f"{metric}_delta_confidence"] = delta_confidence(metric_range_over_delta)
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
    ]
    for metric in METRICS:
        fields.extend(
            [
                f"{metric}_base_ms",
                f"{metric}_variant_ms",
                f"{metric}_delta_ms",
                f"{metric}_speedup",
                f"{metric}_base_range_ms",
                f"{metric}_variant_range_ms",
            ]
        )
    fields.extend(
        [
            "tracked_phase_delta_sum_ms",
            "untracked_delta_ms",
            "dominant_positive_phase",
            "dominant_positive_delta_ms",
            "dominant_negative_phase",
            "dominant_negative_delta_ms",
            "loop_range_over_delta",
            "delta_confidence",
            "promotion_decision",
        ]
    )
    for metric in PHASE_CONFIDENCE_METRICS:
        fields.extend(
            [
                f"{metric}_range_over_delta",
                f"{metric}_delta_confidence",
            ]
        )
    fields.extend(
        [
            "max_process_cpu",
            "max_total_cpu",
            "max_process",
        ]
    )
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
