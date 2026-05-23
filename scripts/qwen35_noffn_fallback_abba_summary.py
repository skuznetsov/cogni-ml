#!/usr/bin/env python3
"""Summarize no-FFN fallback ABBA probe rows."""

from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROW_PREFIX = "self_spec_gpu_pipeline_noffn_fallback_abba"


def field(line: str, name: str, default: str = "") -> str:
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", line)
    return match.group(1) if match else default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    rows = []
    for path in args.logs:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ROW_PREFIX not in line:
                continue
            rows.append({
                "scope": field(line, "scope"),
                "name": field(line, "name"),
                "mode": field(line, "mode"),
                "gamma": field(line, "gamma"),
                "split": field(line, "draft_split", "default"),
                "parity": field(line, "parity") == "true",
                "overlap": float(field(line, "overlap_ms", "0")),
                "plain": float(field(line, "plain_exact_ms", "0")),
                "serial": float(field(line, "serial_ms", "0")),
                "rejections": int(field(line, "rejections", "0")),
                "accepted": int(field(line, "accepted_draft_tokens", "0")),
                "proposed": int(field(line, "proposed_tokens", "0")),
            })

    by_mode = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)

    print(f"noffn_fallback_abba rows={len(rows)} modes={','.join(sorted(by_mode))}")
    for mode in sorted(by_mode):
        group = by_mode[mode]
        overlap = sum(row["overlap"] for row in group)
        plain = sum(row["plain"] for row in group)
        serial = sum(row["serial"] for row in group)
        median_overlap = statistics.median(row["overlap"] for row in group) if group else 0.0
        print(
            f"mode={mode} rows={len(group)} parity={sum(row['parity'] for row in group)}/{len(group)} "
            f"rejects={sum(row['rejections'] for row in group)} accepted={sum(row['accepted'] for row in group)} "
            f"proposed={sum(row['proposed'] for row in group)} overlap_total={overlap:.3f} "
            f"overlap_median={median_overlap:.3f} plain_speedup={plain / overlap if overlap else 0.0:.4f} "
            f"serial_speedup={serial / overlap if overlap else 0.0:.4f}"
        )

    if "off" in by_mode and "on" in by_mode:
        off = by_mode["off"]
        on = by_mode["on"]
        off_total = sum(row["overlap"] for row in off)
        on_total = sum(row["overlap"] for row in on)
        delta = (off_total - on_total) * 100.0 / off_total if off_total else 0.0
        print(f"delta off_to_on total_delta%={delta:.2f} off_total={off_total:.3f} on_total={on_total:.3f}")

        grouped = defaultdict(dict)
        for row in rows:
            key = (row["scope"], row["name"], row["gamma"], row["split"])
            grouped[key].setdefault(row["mode"], []).append(row)
        print("pair scope name gamma split off_median on_median delta% off_rejects on_rejects counter_changed")
        for key, modes in sorted(grouped.items()):
            if "off" not in modes or "on" not in modes:
                continue
            off_med = statistics.median(row["overlap"] for row in modes["off"])
            on_med = statistics.median(row["overlap"] for row in modes["on"])
            pair_delta = (off_med - on_med) * 100.0 / off_med if off_med else 0.0
            off_rej = sum(row["rejections"] for row in modes["off"])
            on_rej = sum(row["rejections"] for row in modes["on"])
            off_counts = (off_rej, sum(row["accepted"] for row in modes["off"]), sum(row["proposed"] for row in modes["off"]))
            on_counts = (on_rej, sum(row["accepted"] for row in modes["on"]), sum(row["proposed"] for row in modes["on"]))
            print(f"pair {key[0]} {key[1]} {key[2]} {key[3]} {off_med:.3f} {on_med:.3f} {pair_delta:.2f} {off_rej} {on_rej} {off_counts != on_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
