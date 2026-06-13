#!/usr/bin/env python3
"""Rank DiffusionGemma phase ABBA TSV rows into candidate optimization windows.

This is an attribution helper, not benchmark evidence. It consumes
scripts/diffusion_gemma_phase_abba.cr output and reports which measured phase
currently dominates the recomputed phase potential.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable


PHASE_METRICS = [
    "qkv_ms",
    "context_ms",
    "context_score_ms",
    "context_softmax_ms",
    "context_value_ms",
    "attention_out_ms",
    "shared_ffn_ms",
    "moe_ffn_ms",
    "moe_grouped_prep_ms",
    "moe_grouped_gate_up_ms",
    "moe_grouped_activation_ms",
    "moe_grouped_down_ms",
    "moe_grouped_scatter_combine_norm_ms",
    "combine_scale_ms",
]

ROUTE_FLAGS = [
    "shared_rows",
    "shared_resident",
    "moe_rows",
    "grouped_moe",
    "moe_router_batch",
    "moe_gpu_gather",
    "moe_gpu_prenorm",
    "moe_gpu_reduce",
    "attention_out_rows",
    "attention_residual_metal_rows",
    "attention_residual_context_buffer",
]


def median(values: Iterable[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return statistics.median(clean) if clean else float("nan")


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or "nan")
    except ValueError:
        return float("nan")


def fmt(value: float) -> str:
    if math.isnan(value):
        return "NA"
    if math.isinf(value):
        return "inf"
    return f"{value:.6f}"


def read_text(paths: list[Path]) -> str:
    if not paths:
        return sys.stdin.read()
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def read_rows(text: str) -> list[dict[str, str]]:
    data_lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    if not data_lines:
        return []
    return list(csv.DictReader(data_lines, delimiter="\t"))


def measured_samples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if row.get("kind") == "sample" and row.get("measured") == "true" and row.get("arm") in {"base", "variant"}
    ]


def metric_summary(samples: list[dict[str, str]], metric: str) -> dict[str, float]:
    by_arm = {
        arm: [as_float(row, metric) for row in samples if row.get("arm") == arm]
        for arm in ("base", "variant")
    }
    base_median = median(by_arm["base"])
    variant_median = median(by_arm["variant"])
    delta = base_median - variant_median
    speedup = base_median / variant_median if variant_median > 0 else float("nan")
    base_range = max(by_arm["base"]) - min(by_arm["base"]) if by_arm["base"] else float("nan")
    variant_range = max(by_arm["variant"]) - min(by_arm["variant"]) if by_arm["variant"] else float("nan")
    combined_range = base_range + variant_range if not math.isnan(base_range + variant_range) else float("nan")
    range_over_delta = combined_range / abs(delta) if abs(delta) > 0 else float("inf")
    return {
        "base_median": base_median,
        "variant_median": variant_median,
        "delta": delta,
        "speedup": speedup,
        "combined_range": combined_range,
        "range_over_delta": range_over_delta,
    }


def route_status(samples: list[dict[str, str]], flag: str, arm: str) -> str:
    values = {row.get(flag, "") for row in samples if row.get("arm") == arm}
    clean = sorted(value for value in values if value)
    return ",".join(clean) if clean else "NA"


def as_int(row: dict[str, str], key: str) -> int | None:
    try:
        return int(row.get(key, ""))
    except ValueError:
        return None


def sequence_position_medians(samples: list[dict[str, str]], metric: str) -> dict[int, float]:
    positions = sorted({idx for row in samples if (idx := as_int(row, "sequence_index")) is not None})
    return {
        idx: median(as_float(row, metric) for row in samples if as_int(row, "sequence_index") == idx)
        for idx in positions
    }


def arm_positions(samples: list[dict[str, str]], arm: str) -> str:
    positions = sorted({idx for row in samples if row.get("arm") == arm and (idx := as_int(row, "sequence_index")) is not None})
    return ",".join(str(idx) for idx in positions) if positions else "NA"


def print_tsv(summaries: dict[str, dict[str, float]], total_base: float) -> None:
    print("kind\tmetric\tbase_median_ms\tvariant_median_ms\tspeedup\tdelta_ms\tcombined_range_ms\trange_over_delta\tbase_pct")
    for metric, row in summaries.items():
        base_pct = row["base_median"] * 100.0 / total_base if total_base > 0 else float("nan")
        print("\t".join([
            "phase",
            metric,
            fmt(row["base_median"]),
            fmt(row["variant_median"]),
            fmt(row["speedup"]),
            fmt(row["delta"]),
            fmt(row["combined_range"]),
            fmt(row["range_over_delta"]),
            fmt(base_pct),
        ]))


def dominant_grouped_subphase(phase_rows: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]] | None:
    grouped = {
        metric: row for metric, row in phase_rows.items()
        if metric.startswith("moe_grouped_") and row["base_median"] > 0.0
    }
    if not grouped:
        return None
    metric = max(grouped, key=lambda key: grouped[key]["base_median"])
    return metric, grouped[metric]


def candidate_text(dominant_metric: str, route_changes: list[str], noisy_delta_count: int) -> list[str]:
    rows: list[str] = []
    if dominant_metric == "moe_ffn_ms":
        rows.append(
            "Ladder: MoE FFN dominates. Window=expert FFN rows; corridor=canvas/expert batch; legal move=retain/fuse grouped expert work only if total_ms descends under ABBA."
        )
    elif dominant_metric.startswith("moe_grouped_"):
        rows.append(
            "Ladder: grouped-MoE subphase dominates. Window=%s; corridor=expert batch body; legal move=fuse or transport this exact subphase without increasing route/scatter conflict."
            % dominant_metric
        )
    elif dominant_metric in {"qkv_ms", "context_ms", "attention_out_ms"}:
        rows.append(
            "Ladder: attention-side work dominates. Window=QKV/context/output span; corridor=resident attention buffer; legal move=fuse a broader GPU-resident attention corridor, not only the tail."
        )
    elif dominant_metric == "shared_ffn_ms":
        rows.append(
            "Ladder: shared FFN dominates. Window=shared dense FFN rows; corridor=canvas batch; legal move=batch/fuse shared gate-up-down while preserving exact residual boundaries."
        )
    elif dominant_metric == "combine_scale_ms":
        rows.append(
            "Spike: combine/scale dominates an otherwise tiny row. Window=post-FFN scalar tail; legal move=eliminate redundant scalar passes if total_ms proof is larger than noise."
        )
    if route_changes:
        rows.append(
            "Diamond: route flags differ (" + ", ".join(route_changes) + "). Check composition conflict before stacking another optimization."
        )
    if noisy_delta_count:
        rows.append(
            f"Adversary: {noisy_delta_count} phase deltas have range_over_delta>2. Treat those component wins as noisy unless total_ms also improves."
        )
    rows.append(
        "Dual frame: if phase attribution does not lower recomputed total_ms, fall back to exact baseline/profile-only mode before trying another local tweak."
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path, help="phase_abba TSV files; stdin is used when omitted")
    parser.add_argument("--top", type=int, default=8, help="number of phase rows to print")
    parser.add_argument("--tsv", action="store_true", help="emit machine-readable TSV only")
    args = parser.parse_args()

    samples = measured_samples(read_rows(read_text(args.paths)))
    if not samples:
        raise SystemExit("no measured phase sample rows found")

    summaries = {metric: metric_summary(samples, metric) for metric in ["total_ms", *PHASE_METRICS]}
    total_base = summaries["total_ms"]["base_median"]
    phase_rows = {
        metric: summaries[metric]
        for metric in PHASE_METRICS
        if not math.isnan(summaries[metric]["base_median"])
    }
    dominant_metric = max(phase_rows, key=lambda metric: phase_rows[metric]["base_median"]) if phase_rows else "none"
    dominant_base = phase_rows[dominant_metric]["base_median"] if dominant_metric != "none" else 0.0
    grouped_subphase = dominant_grouped_subphase(phase_rows)
    tied = sum(1 for row in phase_rows.values() if dominant_base > 0 and row["base_median"] >= dominant_base * 0.80)
    noisy_delta_count = sum(
        1 for row in phase_rows.values()
        if row["range_over_delta"] > 2.0 and abs(row["delta"]) > 0.0
    )
    route_changes = [
        flag for flag in ROUTE_FLAGS
        if route_status(samples, flag, "base") != route_status(samples, flag, "variant")
    ]

    if args.tsv:
        print_tsv(summaries, total_base)
        return 0

    print("DiffusionGemma phase atlas")
    print(
        "  Phi=(dominant_wait_bucket=%s, tied_dominant_routes=%d, conflict_or_sync_count=%d, remaining_work_ms=%s)"
        % (dominant_metric, tied, len(route_changes) + noisy_delta_count, fmt(total_base))
    )
    print(
        "  total_ms base=%s variant=%s speedup=%s delta_ms=%s range_over_delta=%s"
        % (
            fmt(summaries["total_ms"]["base_median"]),
            fmt(summaries["total_ms"]["variant_median"]),
            fmt(summaries["total_ms"]["speedup"]),
            fmt(summaries["total_ms"]["delta"]),
            fmt(summaries["total_ms"]["range_over_delta"]),
        )
    )
    if route_changes:
        print("  route_changes=" + ",".join(route_changes))
    if grouped_subphase:
        subphase_metric, subphase_row = grouped_subphase
        moe_base = summaries["moe_ffn_ms"]["base_median"]
        pct_moe = subphase_row["base_median"] * 100.0 / moe_base if moe_base > 0 else float("nan")
        print(
            "  grouped_moe_subphase=%s base=%s ms pct_of_moe=%s%%"
            % (subphase_metric, fmt(subphase_row["base_median"]), fmt(pct_moe))
        )

    position_medians = sequence_position_medians(samples, "total_ms")
    if position_medians:
        position_values = [value for value in position_medians.values() if not math.isnan(value)]
        position_span = max(position_values) - min(position_values) if position_values else float("nan")
        total_delta = abs(summaries["total_ms"]["delta"])
        print("\nSequence position bias")
        print("  total_ms_by_sequence_index=" + ", ".join(f"{idx}:{fmt(value)}" for idx, value in position_medians.items()))
        print("  base_positions=%s variant_positions=%s" % (arm_positions(samples, "base"), arm_positions(samples, "variant")))
        print("  position_span_ms=%s vs abs_total_delta_ms=%s" % (fmt(position_span), fmt(total_delta)))
        if position_span > total_delta:
            print("  warning=position_span_exceeds_total_delta")

    print("\nPhase buckets")
    for metric, row in sorted(phase_rows.items(), key=lambda item: (-item[1]["base_median"], item[0]))[:args.top]:
        base_pct = row["base_median"] * 100.0 / total_base if total_base > 0 else float("nan")
        print(
            "  %-22s base=%10s ms variant=%10s ms speedup=%8s delta=%10s pct_base=%6s%% rod=%s"
            % (
                metric,
                fmt(row["base_median"]),
                fmt(row["variant_median"]),
                fmt(row["speedup"]),
                fmt(row["delta"]),
                fmt(base_pct),
                fmt(row["range_over_delta"]),
            )
        )

    print("\nRoute flags")
    for flag in ROUTE_FLAGS:
        print(f"  {flag}: base={route_status(samples, flag, 'base')} variant={route_status(samples, flag, 'variant')}")

    print("\nLTP/WBA candidate windows")
    for row in candidate_text(dominant_metric, route_changes, noisy_delta_count):
        print("  " + row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
