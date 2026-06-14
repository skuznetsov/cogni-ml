#!/usr/bin/env python3
"""Rank DiffusionGemma prompt-cache ABBA rows into optimization windows.

This is an attribution helper, not benchmark evidence. It consumes
scripts/diffusion_gemma_prompt_cache_abba.cr TSV output and recomputes the
active prompt-cache/materialization potential before choosing another local
GPU transport branch.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable


PROMPT_CACHE_METRICS = [
    "projection_ms",
    "projection_norm_ms",
    "projection_matmul_ms",
    "projection_assemble_ms",
    "projection_copy_ms",
    "projection_head_norm_ms",
    "projection_q_norm_ms",
    "projection_k_norm_ms",
    "projection_v_norm_ms",
    "projection_rope_ms",
    "projection_rope_table_ms",
    "projection_rope_apply_ms",
    "projection_rope_q_apply_ms",
    "projection_rope_k_apply_ms",
    "materialize_ms",
    "materialize_context_ms",
    "materialize_attention_out_ms",
    "materialize_shared_ffn_ms",
    "materialize_moe_ffn_ms",
    "materialize_combine_scale_ms",
    "materialize_moe_grouped_prep_ms",
    "materialize_moe_grouped_gate_up_ms",
    "materialize_moe_grouped_activation_ms",
    "materialize_moe_grouped_down_ms",
    "materialize_moe_grouped_scatter_combine_norm_ms",
]

ROUTE_FLAGS = [
    "prompt_cache_policy",
    "materialize_final_rows",
    "materialize_batch_rows",
    "materialize_grouped_moe",
    "projection_backend",
    "fused_norm_rope",
]

GROUPED_ROUTE_SHAPE_COUNTERS = [
    "materialize_moe_grouped_active_experts",
    "materialize_moe_grouped_route_slots",
    "materialize_moe_grouped_max_expert_batch",
    "materialize_moe_grouped_over_threshold_experts",
]


def median(values: Iterable[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return statistics.median(clean) if clean else float("nan")


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or "nan")
    except ValueError:
        return float("nan")


def as_int(row: dict[str, str], key: str) -> int | None:
    try:
        return int(row.get(key, ""))
    except ValueError:
        return None


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
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    if not lines:
        return []
    return list(csv.DictReader(lines, delimiter="\t"))


def measured_samples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if row.get("kind") == "sample" and row.get("measured") == "true" and row.get("arm") in {"base", "variant"}
    ]


def metric_summary(samples: list[dict[str, str]], metric: str) -> dict[str, float]:
    values_by_arm = {
        arm: [as_float(row, metric) for row in samples if row.get("arm") == arm]
        for arm in ("base", "variant")
    }
    base_median = median(values_by_arm["base"])
    variant_median = median(values_by_arm["variant"])
    delta = base_median - variant_median
    speedup = base_median / variant_median if variant_median > 0 else float("nan")
    base_range = max(values_by_arm["base"]) - min(values_by_arm["base"]) if values_by_arm["base"] else float("nan")
    variant_range = max(values_by_arm["variant"]) - min(values_by_arm["variant"]) if values_by_arm["variant"] else float("nan")
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


def visible_median(row: dict[str, float]) -> float:
    finite = [value for value in (row["base_median"], row["variant_median"]) if not math.isnan(value)]
    return max(finite) if finite else float("nan")


def route_status(samples: list[dict[str, str]], flag: str, arm: str) -> str:
    values = {row.get(flag, "") for row in samples if row.get("arm") == arm}
    clean = sorted(value for value in values if value)
    return ",".join(clean) if clean else "NA"


def available_shape_counters(samples: list[dict[str, str]]) -> list[str]:
    return [
        counter for counter in GROUPED_ROUTE_SHAPE_COUNTERS
        if any(row.get(counter, "") for row in samples)
    ]


def sequence_position_medians(samples: list[dict[str, str]], metric: str) -> dict[int, float]:
    positions = sorted({idx for row in samples if (idx := as_int(row, "sequence_index")) is not None})
    return {
        idx: median(as_float(row, metric) for row in samples if as_int(row, "sequence_index") == idx)
        for idx in positions
    }


def arm_positions(samples: list[dict[str, str]], arm: str) -> str:
    positions = sorted({idx for row in samples if row.get("arm") == arm and (idx := as_int(row, "sequence_index")) is not None})
    return ",".join(str(idx) for idx in positions) if positions else "NA"


def grouped_subphase(phase_rows: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]] | None:
    grouped = {
        metric: row for metric, row in phase_rows.items()
        if metric.startswith("materialize_moe_grouped_") and visible_median(row) > 0.0
    }
    if not grouped:
        return None
    metric = max(grouped, key=lambda key: visible_median(grouped[key]))
    return metric, grouped[metric]


def candidate_text(dominant_metric: str, route_changes: list[str], noisy_count: int, regressions: list[str]) -> list[str]:
    rows: list[str] = []
    if dominant_metric == "materialize_ms":
        rows.append(
            "Ladder: prompt materialization dominates. Window=prompt rows after projection; corridor=context -> attention residual -> shared/MoE FFN -> final combine."
        )
    elif dominant_metric == "materialize_moe_ffn_ms":
        rows.append(
            "Ladder: prompt materialize MoE dominates. Window=prompt expert FFN rows; corridor=route maps plus gate/up/down; legal move must lower total prompt-cache Phi."
        )
    elif dominant_metric.startswith("materialize_moe_grouped_"):
        rows.append(
            "Ladder: grouped prompt-MoE subphase dominates. Window=%s; corridor=expert batch body; legal move=fuse/transport this subphase without increasing route conflicts."
            % dominant_metric
        )
    elif dominant_metric in {"materialize_context_ms", "materialize_attention_out_ms"}:
        rows.append(
            "Ladder: prompt attention materialization dominates. Window=prompt context/attention residual; corridor=resident attention rows, not MoE-only tuning."
        )
    elif dominant_metric == "materialize_shared_ffn_ms":
        rows.append(
            "Ladder: shared prompt FFN dominates. Window=shared dense FFN rows; corridor=batch/fuse shared gate-up-down while preserving exact prompt rows."
        )
    elif dominant_metric.startswith("projection_") or dominant_metric == "projection_ms":
        rows.append(
            "Ladder: prompt projection dominates. Window=QKV projection/norm/RoPE; corridor=projection Metal ownership and projection cache."
        )
    elif dominant_metric == "materialize_combine_scale_ms":
        rows.append(
            "Spike: prompt combine/scale dominates. Window=post-FFN scalar tail; legal move=eliminate redundant passes only if total_ms proof exceeds noise."
        )
    else:
        rows.append(
            "Measure: no clean prompt-cache phase dominates. Recompute with deeper or quieter ABBA before a local optimization."
        )
    if route_changes:
        rows.append(
            "Diamond: route flags differ (" + ", ".join(route_changes) + "). Check exactness/certificate boundary before stacking another route."
        )
    if regressions:
        rows.append(
            "Diamond: visible regressions (" + ", ".join(regressions[:3]) + "). Normalize the conflicting route before stacking another optimization."
        )
    if noisy_count:
        rows.append(
            f"Adversary: {noisy_count} phase deltas have range_over_delta>2. Treat component wins as noisy unless total_ms also descends."
        )
    rows.append(
        "Dual frame: if the next move does not lower recomputed prompt-cache Phi, keep it as infrastructure and fall back to exact/profile-only mode."
    )
    return rows


def print_tsv(summaries: dict[str, dict[str, float]], total_base: float) -> None:
    print("kind\tmetric\tbase_median_ms\tvariant_median_ms\tspeedup\tdelta_ms\tcombined_range_ms\trange_over_delta\tbase_pct")
    for metric, row in summaries.items():
        base_pct = row["base_median"] * 100.0 / total_base if total_base > 0 else float("nan")
        print("\t".join([
            "prompt_cache_phase",
            metric,
            fmt(row["base_median"]),
            fmt(row["variant_median"]),
            fmt(row["speedup"]),
            fmt(row["delta"]),
            fmt(row["combined_range"]),
            fmt(row["range_over_delta"]),
            fmt(base_pct),
        ]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path, help="prompt_cache_abba TSV files; stdin is used when omitted")
    parser.add_argument("--top", type=int, default=10, help="number of phase rows to print")
    parser.add_argument("--tsv", action="store_true", help="emit machine-readable TSV only")
    args = parser.parse_args()

    samples = measured_samples(read_rows(read_text(args.paths)))
    if not samples:
        raise SystemExit("no measured prompt-cache sample rows found")

    summaries = {metric: metric_summary(samples, metric) for metric in ["total_ms", *PROMPT_CACHE_METRICS]}
    total_base = summaries["total_ms"]["base_median"]
    phase_rows = {
        metric: summaries[metric]
        for metric in PROMPT_CACHE_METRICS
        if not math.isnan(summaries[metric]["base_median"])
    }
    shape_counters = available_shape_counters(samples)
    shape_summaries = {counter: metric_summary(samples, counter) for counter in shape_counters}
    dominant_metric = max(phase_rows, key=lambda metric: visible_median(phase_rows[metric])) if phase_rows else "none"
    dominant_base = visible_median(phase_rows[dominant_metric]) if dominant_metric != "none" else 0.0
    tied = sum(1 for row in phase_rows.values() if dominant_base > 0.0 and visible_median(row) >= dominant_base * 0.80)
    noisy_count = sum(
        1 for row in phase_rows.values()
        if row["range_over_delta"] > 2.0 and abs(row["delta"]) > 0.0
    )
    route_changes = [
        flag for flag in ROUTE_FLAGS
        if route_status(samples, flag, "base") != route_status(samples, flag, "variant")
    ]
    regressions = [
        metric for metric, row in sorted(
            phase_rows.items(),
            key=lambda item: (-visible_median(item[1]), item[0]),
        )
        if row["delta"] < 0.0 and total_base > 0.0 and visible_median(row) >= total_base * 0.05
    ]
    route_shape_pressure = 1 if shape_summaries.get("materialize_moe_grouped_over_threshold_experts", {}).get("variant_median", 0.0) > 0.0 else 0

    if args.tsv:
        print_tsv(summaries, total_base)
        return 0

    print("DiffusionGemma prompt-cache atlas")
    print(
        "  Phi=(dominant_wait_bucket=%s, tied_dominant_routes=%d, conflict_or_sync_count=%d, remaining_work_ms=%s)"
        % (dominant_metric, tied, len(route_changes) + noisy_count + len(regressions) + route_shape_pressure, fmt(total_base))
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
    if regressions:
        print("  visible_regressions=" + ",".join(regressions))
    if subphase := grouped_subphase(phase_rows):
        metric, row = subphase
        moe_base = summaries["materialize_moe_ffn_ms"]["base_median"]
        pct_moe = row["base_median"] * 100.0 / moe_base if moe_base > 0 else float("nan")
        print("  grouped_moe_subphase=%s base=%s ms pct_of_moe=%s%%" % (metric, fmt(row["base_median"]), fmt(pct_moe)))
    if shape_summaries:
        threshold = shape_summaries.get("materialize_moe_grouped_over_threshold_experts")
        if threshold and threshold["variant_median"] > 0.0:
            print("  grouped_route_pressure=over_threshold_experts variant_median=%s" % fmt(threshold["variant_median"]))

    positions = sequence_position_medians(samples, "total_ms")
    if positions:
        values = [value for value in positions.values() if not math.isnan(value)]
        span = max(values) - min(values) if values else float("nan")
        total_delta = abs(summaries["total_ms"]["delta"])
        print("\nSequence position bias")
        print("  total_ms_by_sequence_index=" + ", ".join(f"{idx}:{fmt(value)}" for idx, value in positions.items()))
        print("  base_positions=%s variant_positions=%s" % (arm_positions(samples, "base"), arm_positions(samples, "variant")))
        print("  position_span_ms=%s vs abs_total_delta_ms=%s" % (fmt(span), fmt(total_delta)))
        if span > total_delta:
            print("  warning=position_span_exceeds_total_delta")

    print("\nPrompt-cache buckets")
    for metric, row in sorted(phase_rows.items(), key=lambda item: (-visible_median(item[1]), item[0]))[:args.top]:
        base_pct = row["base_median"] * 100.0 / total_base if total_base > 0 else float("nan")
        visible_pct = visible_median(row) * 100.0 / total_base if total_base > 0 else float("nan")
        print(
            "  %-42s base=%10s ms variant=%10s ms speedup=%8s delta=%10s pct_base=%6s%% pct_visible=%6s%% rod=%s"
            % (
                metric,
                fmt(row["base_median"]),
                fmt(row["variant_median"]),
                fmt(row["speedup"]),
                fmt(row["delta"]),
                fmt(base_pct),
                fmt(visible_pct),
                fmt(row["range_over_delta"]),
            )
        )

    print("\nRoute flags")
    for flag in ROUTE_FLAGS:
        print(f"  {flag}: base={route_status(samples, flag, 'base')} variant={route_status(samples, flag, 'variant')}")

    if shape_summaries:
        print("\nGrouped route shape")
        for counter, row in shape_summaries.items():
            print(
                "  %-50s base=%10s variant=%10s delta=%10s rod=%s"
                % (
                    counter,
                    fmt(row["base_median"]),
                    fmt(row["variant_median"]),
                    fmt(row["delta"]),
                    fmt(row["range_over_delta"]),
                )
            )
        if shape_summaries.get("materialize_moe_grouped_over_threshold_experts", {}).get("variant_median", 0.0) > 0.0:
            print("  warning=expert_batches_cross_gemv_threshold")
        shape_drift = [
            counter for counter, row in shape_summaries.items()
            if row["delta"] != 0.0 and not math.isnan(row["delta"])
        ]
        if shape_drift:
            print("  warning=route_shape_differs_between_arms:" + ",".join(shape_drift))

    print("\nLTP/WBA candidate windows")
    for row in candidate_text(dominant_metric, route_changes, noisy_count, regressions):
        print("  " + row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
