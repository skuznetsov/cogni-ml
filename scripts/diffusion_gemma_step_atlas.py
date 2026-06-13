#!/usr/bin/env python3
"""Rank DiffusionGemma sparse denoise TSV rows into GPU ownership windows.

This is an attribution helper, not benchmark evidence. It accepts either a
prompt ABBA log directory with run_*.tsv files or one or more TSV files. The
reported byte counts are corridor estimates used to choose the next experiment;
they are not profiler counters.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Iterable


PHASE_METRICS = [
    "loop_decode_qkv_ms",
    "loop_decode_context_ms",
    "loop_decode_context_score_ms",
    "loop_decode_context_softmax_ms",
    "loop_decode_context_value_ms",
    "loop_decode_attention_out_ms",
    "loop_decode_shared_ffn_ms",
    "loop_decode_moe_ffn_ms",
    "loop_decode_combine_scale_ms",
    "loop_output_head_ms",
    "loop_update_ms",
    "loop_regenerate_ms",
    "loop_proposal_ms",
]

ROUTE_FLAGS = [
    "loop_context_backend",
    "loop_context_batch_rows",
    "loop_context_fixed_gqa2",
    "loop_attention_residual_metal_rows",
    "loop_attention_residual_context_buffer",
    "prompt_projection_backend",
    "prompt_projection_fused_norm_rope",
]


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or "nan")
    except ValueError:
        return float("nan")


def median(values: Iterable[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return statistics.median(clean) if clean else float("nan")


def fmt(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "NA"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as io:
        return list(csv.DictReader((line for line in io if line.strip() and not line.startswith("#")), delimiter="\t"))


def read_rows(inputs: list[Path]) -> list[dict[str, str]]:
    if not inputs:
        text = sys.stdin.read()
        return list(csv.DictReader((line for line in text.splitlines() if line.strip() and not line.startswith("#")), delimiter="\t"))

    rows: list[dict[str, str]] = []
    for path in inputs:
        if path.is_dir():
            for child in sorted(path.glob("run_*.tsv")):
                match = re.match(r"run_(\d+)_(base|variant)\.tsv", child.name)
                child_rows = read_tsv(child)
                for row in child_rows:
                    if match and not row.get("arm"):
                        row["arm"] = match.group(2)
                    row["_source"] = str(child)
                rows.extend(child_rows)
        else:
            child_rows = read_tsv(path)
            for row in child_rows:
                row["_source"] = str(path)
            rows.extend(child_rows)
    return rows


def measured_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if any(row.get("kind") == "sample" for row in rows):
        return [
            row for row in rows
            if row.get("kind") == "sample" and row.get("measured") in {"", "true"} and row.get("arm", "") in {"base", "variant", ""}
        ]
    return [row for row in rows if row.get("status", "ok") == "ok"]


def inferred_case(row: dict[str, str]) -> str:
    if row.get("case"):
        return row["case"]
    prompt_len = row.get("prompt_len", "?")
    canvas_len = row.get("canvas_len", "?")
    candidates = row.get("candidate_count", "?")
    return f"p{prompt_len}_c{canvas_len}_k{candidates}"


def inferred_arm(row: dict[str, str]) -> str:
    return row.get("arm") or row.get("_arm") or "single"


def route_status(rows: list[dict[str, str]], flag: str) -> str:
    values = sorted({row.get(flag, "") for row in rows if row.get(flag, "")})
    return ",".join(values) if values else "NA"


def median_field(rows: list[dict[str, str]], key: str) -> float:
    return median(as_float(row, key) for row in rows)


def first_available(rows: list[dict[str, str]], keys: list[str], fallback: float) -> float:
    for key in keys:
        value = median_field(rows, key)
        if not math.isnan(value):
            return value
    return fallback


def metric_summary(rows: list[dict[str, str]], metric: str) -> dict[str, float]:
    samples = [as_float(row, metric) for row in rows]
    med = median(samples)
    spread = max(samples) - min(samples) if samples and not any(math.isnan(v) for v in samples) else float("nan")
    return {"median": med, "range": spread}


def mib(bytes_count: float) -> float:
    return bytes_count / (1024.0 * 1024.0)


def kib(bytes_count: float) -> float:
    return bytes_count / 1024.0


def estimate_transport(rows: list[dict[str, str]], args: argparse.Namespace) -> dict[str, float]:
    canvas_len = first_available(rows, ["canvas_len"], 0.0)
    candidate_count = first_available(rows, ["candidate_count", "mean_candidate_tokens"], 1.0)
    steps_run = first_available(rows, ["steps_run"], 1.0)
    max_layers = first_available(rows, ["max_layers"], 1.0)
    prediction_count = first_available(rows, ["prediction_count"], canvas_len * steps_run)
    total_candidate_tokens = first_available(rows, ["total_candidate_tokens"], prediction_count * candidate_count)
    accepted_count = first_available(rows, ["accepted_count"], float("nan"))

    sparse_logits_bytes = total_candidate_tokens * 4.0
    sparse_sc_pair_bytes = total_candidate_tokens * 8.0
    full_vocab_logits_bytes = canvas_len * steps_run * args.vocab_size * 4.0
    route_slots = canvas_len * steps_run * max_layers * args.expert_used_count
    route_map_bytes = route_slots * args.route_entry_bytes
    compact_certificate_bytes = prediction_count * args.certificate_entry_bytes
    canvas_hidden_bytes = canvas_len * args.hidden_dim * 4.0
    route_gather_bytes = route_slots * args.hidden_dim * 4.0
    prompt_kv_rows = first_available(rows, ["prompt_len"], 0.0) + canvas_len
    prompt_kv_bytes = prompt_kv_rows * args.hidden_dim * 2.0 * 4.0 * max_layers

    return {
        "canvas_len": canvas_len,
        "candidate_count": candidate_count,
        "steps_run": steps_run,
        "max_layers": max_layers,
        "prediction_count": prediction_count,
        "accepted_count": accepted_count,
        "total_candidate_tokens": total_candidate_tokens,
        "sparse_logits_kib": kib(sparse_logits_bytes),
        "sparse_sc_pair_kib": kib(sparse_sc_pair_bytes),
        "full_vocab_logits_mib": mib(full_vocab_logits_bytes),
        "route_map_kib": kib(route_map_bytes),
        "compact_certificate_kib": kib(compact_certificate_bytes),
        "canvas_hidden_kib": kib(canvas_hidden_bytes),
        "route_gather_mib": mib(route_gather_bytes),
        "prompt_kv_est_mib": mib(prompt_kv_bytes),
    }


def dominant_phase(rows: list[dict[str, str]]) -> tuple[str, float, int, int]:
    summaries = {metric: metric_summary(rows, metric) for metric in PHASE_METRICS}
    non_empty = {
        metric: summary
        for metric, summary in summaries.items()
        if not math.isnan(summary["median"])
    }
    if not non_empty:
        return "none", 0.0, 0, 0
    metric = max(non_empty, key=lambda key: non_empty[key]["median"])
    dominant = non_empty[metric]["median"]
    tied = sum(1 for summary in non_empty.values() if dominant > 0 and summary["median"] >= dominant * 0.80)
    noisy = sum(
        1 for summary in non_empty.values()
        if summary["range"] > 0 and dominant > 0 and summary["range"] > dominant * 0.25
    )
    return metric, dominant, tied, noisy


def candidate_windows(dominant: str, transport: dict[str, float], route_changes: list[str]) -> list[str]:
    rows: list[str] = []
    if dominant == "loop_decode_moe_ffn_ms":
        rows.append("Ladder: MoE body dominates. Next legal move is GPU-owned route-map gather/reduce around resident grouped experts.")
    elif dominant in {"loop_decode_qkv_ms", "loop_decode_context_ms", "loop_decode_attention_out_ms"}:
        rows.append("Ladder: attention corridor dominates. Next legal move must carry QKV/context/attention residual together, not just the tail.")
    elif dominant == "loop_output_head_ms":
        rows.append("Spike: output head dominates. Next legal move is compact top-k/candidate logits on GPU, not full hidden/logit materialization.")
    elif dominant in {"loop_update_ms", "loop_regenerate_ms", "loop_proposal_ms"}:
        rows.append("Spike: control/update path dominates. Next legal move is compact GPU certificate plus resident candidate/self-conditioning metadata.")
    else:
        rows.append("Measure: no single phase dominates cleanly. Recompute with a deeper or quieter run before promoting another local optimization.")

    if transport["full_vocab_logits_mib"] > max(16.0, transport["sparse_logits_kib"] / 1024.0 * 100.0):
        rows.append("Spike: full-vocab EB/self-conditioning would be a large CPU-visible tensor; keep sparse candidate certificates as the product path.")
    if transport["route_gather_mib"] > 1.0:
        rows.append("Ladder: route gather/scatter bytes are large enough to justify a GPU map/reduce prototype if total_ms confirms the MoE window.")
    if route_changes:
        rows.append("Diamond: route flags differ (" + ",".join(route_changes) + "); compare arms only after checking route compatibility.")
    rows.append("Dual frame: if recomputed total loop Phi does not descend, revert to exact baseline/profile-only mode before stacking another route.")
    return rows


def summarize_group(rows: list[dict[str, str]], case: str, arm: str, args: argparse.Namespace) -> dict[str, object]:
    loop_ms = first_available(rows, ["loop_ms_median", "loop_ms", "total_ms"], float("nan"))
    prompt_cache_ms = first_available(rows, ["prompt_cache_ms"], float("nan"))
    dominant, dominant_ms, tied, noisy = dominant_phase(rows)
    transport = estimate_transport(rows, args)
    route_changes: list[str] = []
    return {
        "case": case,
        "arm": arm,
        "rows": len(rows),
        "loop_ms": loop_ms,
        "prompt_cache_ms": prompt_cache_ms,
        "dominant": dominant,
        "dominant_ms": dominant_ms,
        "tied": tied,
        "noisy": noisy,
        "route_status": {flag: route_status(rows, flag) for flag in ROUTE_FLAGS},
        "transport": transport,
        "windows": candidate_windows(dominant, transport, route_changes),
    }


def print_human(summaries: list[dict[str, object]]) -> None:
    print("DiffusionGemma step atlas")
    for summary in summaries:
        transport = summary["transport"]
        assert isinstance(transport, dict)
        print(
            "\ncase=%s arm=%s rows=%s Phi=(dominant_wait_bucket=%s,tied_routes=%s,boundary_conflicts=%s,remaining_work_ms=%s)"
            % (
                summary["case"],
                summary["arm"],
                summary["rows"],
                summary["dominant"],
                summary["tied"],
                summary["noisy"],
                fmt(float(summary["loop_ms"])),
            )
        )
        print(
            "  loop_ms=%s prompt_cache_ms=%s dominant_ms=%s"
            % (fmt(float(summary["loop_ms"])), fmt(float(summary["prompt_cache_ms"])), fmt(float(summary["dominant_ms"])))
        )
        print(
            "  rows: canvas=%s candidates=%s steps=%s max_layers=%s predictions=%s accepted=%s total_candidate_tokens=%s"
            % (
                fmt(float(transport["canvas_len"]), 0),
                fmt(float(transport["candidate_count"]), 0),
                fmt(float(transport["steps_run"]), 0),
                fmt(float(transport["max_layers"]), 0),
                fmt(float(transport["prediction_count"]), 0),
                fmt(float(transport["accepted_count"]), 0),
                fmt(float(transport["total_candidate_tokens"]), 0),
            )
        )
        print(
            "  transport_est: sparse_logits=%s KiB sparse_sc_pairs=%s KiB full_vocab_logits=%s MiB route_map=%s KiB compact_cert=%s KiB route_gather=%s MiB prompt_kv=%s MiB"
            % (
                fmt(float(transport["sparse_logits_kib"])),
                fmt(float(transport["sparse_sc_pair_kib"])),
                fmt(float(transport["full_vocab_logits_mib"])),
                fmt(float(transport["route_map_kib"])),
                fmt(float(transport["compact_certificate_kib"])),
                fmt(float(transport["route_gather_mib"])),
                fmt(float(transport["prompt_kv_est_mib"])),
            )
        )
        route_statuses = summary["route_status"]
        assert isinstance(route_statuses, dict)
        route_bits = [f"{flag}={value}" for flag, value in route_statuses.items() if value != "NA"]
        if route_bits:
            print("  routes: " + " ".join(route_bits))
        print("  candidate_windows:")
        for window in summary["windows"]:
            print("    - " + str(window))


def print_tsv(summaries: list[dict[str, object]]) -> None:
    fields = [
        "case", "arm", "rows", "loop_ms", "prompt_cache_ms", "dominant_wait_bucket",
        "dominant_ms", "tied_routes", "boundary_conflicts", "canvas_len",
        "candidate_count", "steps_run", "max_layers", "prediction_count",
        "accepted_count", "total_candidate_tokens", "sparse_logits_kib",
        "sparse_sc_pair_kib", "full_vocab_logits_mib", "route_map_kib",
        "compact_certificate_kib", "canvas_hidden_kib", "route_gather_mib",
        "prompt_kv_est_mib",
    ]
    print("\t".join(fields))
    for summary in summaries:
        transport = summary["transport"]
        assert isinstance(transport, dict)
        values = {
            "case": summary["case"],
            "arm": summary["arm"],
            "rows": summary["rows"],
            "loop_ms": summary["loop_ms"],
            "prompt_cache_ms": summary["prompt_cache_ms"],
            "dominant_wait_bucket": summary["dominant"],
            "dominant_ms": summary["dominant_ms"],
            "tied_routes": summary["tied"],
            "boundary_conflicts": summary["noisy"],
            **transport,
        }
        print("\t".join(str(values.get(field, "")) if not isinstance(values.get(field, ""), float) else fmt(float(values[field])) for field in fields))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path, help="ABBA directory, TSV files, or stdin when omitted")
    parser.add_argument("--vocab-size", type=int, default=262144, help="vocab size for full-vocab oracle byte estimates")
    parser.add_argument("--hidden-dim", type=int, default=2816, help="hidden dimension for row transport estimates")
    parser.add_argument("--expert-used-count", type=int, default=8, help="top-k routed experts per row")
    parser.add_argument("--route-entry-bytes", type=int, default=12, help="bytes per compact route-map entry estimate")
    parser.add_argument("--certificate-entry-bytes", type=int, default=16, help="bytes per compact per-row certificate estimate")
    parser.add_argument("--tsv", action="store_true", help="emit machine-readable TSV")
    args = parser.parse_args()

    rows = measured_rows(read_rows(args.inputs))
    if not rows:
        raise SystemExit("no usable DiffusionGemma TSV rows found")

    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((inferred_case(row), inferred_arm(row)), []).append(row)

    summaries = [
        summarize_group(group_rows, case, arm, args)
        for (case, arm), group_rows in sorted(groups.items())
    ]
    if args.tsv:
        print_tsv(summaries)
    else:
        print_human(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
