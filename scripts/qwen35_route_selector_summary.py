#!/usr/bin/env python3
"""Compare prompt route-selector rows against same-run pure pipeline rows."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path


def field(line: str, name: str, default: str = "") -> str:
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", line)
    return match.group(1) if match else default


def row_key(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (str(row["scope"]), str(row["name"]), str(row["gamma"]), str(row["split"]))


def parse_rows(paths: list[Path]) -> tuple[list[dict[str, object]], dict[tuple[str, str, str, str], dict[str, object]]]:
    selectors: list[dict[str, object]] = []
    baselines: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("self_spec_gpu_pipeline_route_selector "):
                row = {
                    "scope": field(line, "scope"),
                    "name": field(line, "name"),
                    "gamma": field(line, "gamma"),
                    "split": field(line, "draft_split", "default"),
                    "route": field(line, "route_selector_selected_route"),
                    "decision": field(line, "route_selector"),
                    "feature": field(line, "route_selector_feature"),
                    "value": float(field(line, "route_selector_value", "nan")),
                    "parity": field(line, "parity") == "true",
                    "overlap": float(field(line, "overlap_ms", "0")),
                    "rejections": int(field(line, "rejections", "0")),
                    "accepted": int(field(line, "accepted_draft_tokens", "0")),
                    "proposed": int(field(line, "proposed_tokens", "0")),
                }
                selectors.append(row)
            elif line.startswith("self_spec_gpu_pipeline layers=") or line.startswith("self_spec_gpu_pipeline_suite name="):
                # Treat only the ordinary lowrank row as the pure baseline.
                if " hybrid_route=" in line or " draft_no_ffn_layers=" in line or " draft_no_ffn=1" in line:
                    continue
                scope = "suite" if line.startswith("self_spec_gpu_pipeline_suite ") else "main"
                row = {
                    "scope": scope,
                    "name": field(line, "name", "main"),
                    "gamma": field(line, "gamma"),
                    "split": field(line, "draft_split", "default"),
                    "parity": field(line, "parity") == "true",
                    "overlap": float(field(line, "overlap_ms", "0")),
                    "rejections": int(field(line, "rejections", "0")),
                    "accepted": int(field(line, "accepted_draft_tokens", "0")),
                    "proposed": int(field(line, "proposed_tokens", "0")),
                }
                baselines[row_key(row)] = row
    return selectors, baselines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    selectors, baselines = parse_rows(args.logs)
    print(f"route_selector_summary selector_rows={len(selectors)} baselines={len(baselines)}")
    print("pair scope name gamma split decision route feature value selector_ms pure_ms delta% selector_rejects pure_rejects parity")
    deltas: list[float] = []
    selected_deltas: list[float] = []
    missing = 0
    parity_ok = 0
    for row in selectors:
        base = baselines.get(row_key(row))
        if base is None:
            missing += 1
            continue
        selector_ms = float(row["overlap"])
        pure_ms = float(base["overlap"])
        delta = (pure_ms - selector_ms) * 100.0 / pure_ms if pure_ms else 0.0
        deltas.append(delta)
        if row["decision"] == "select":
            selected_deltas.append(delta)
        if row["parity"] and base["parity"]:
            parity_ok += 1
        print(
            f"pair {row['scope']} {row['name']} {row['gamma']} {row['split']} "
            f"{row['decision']} {row['route']} {row['feature']} {row['value']:.6f} "
            f"{selector_ms:.3f} {pure_ms:.3f} {delta:.2f} "
            f"{row['rejections']} {base['rejections']} {row['parity'] and base['parity']}"
        )

    total_selector = sum(float(row["overlap"]) for row in selectors if row_key(row) in baselines)
    total_pure = sum(float(baselines[row_key(row)]["overlap"]) for row in selectors if row_key(row) in baselines)
    total_delta = (total_pure - total_selector) * 100.0 / total_pure if total_pure else 0.0
    median_delta = statistics.median(deltas) if deltas else 0.0
    selected_median_delta = statistics.median(selected_deltas) if selected_deltas else 0.0
    print(
        f"aggregate matched={len(deltas)} missing={missing} parity={parity_ok}/{len(deltas)} "
        f"selected={len(selected_deltas)} total_delta%={total_delta:.2f} "
        f"median_delta%={median_delta:.2f} selected_median_delta%={selected_median_delta:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
