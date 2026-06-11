#!/usr/bin/env python3
"""Summarize DiffusionGemma ABBA host snapshots without model artifacts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def snapshot_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace") as io:
        for line in io:
            if "=" not in line:
                continue
            key, value = line.rstrip("\n").split("=", 1)
            fields[key] = value
    return fields


def as_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def format_float(value: float) -> str:
    return f"{value:.1f}" if not math.isnan(value) else "NA"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--load-threshold", type=float, default=30.0)
    parser.add_argument("--total-threshold", type=float, default=90.0)
    args = parser.parse_args()

    max_process_cpu = float("nan")
    max_total_cpu = float("nan")
    max_process = "NA"
    snapshots = 0
    for path in sorted(args.root.glob("host_snapshot_*.txt")):
        fields = snapshot_fields(path)
        process_cpu = as_float(fields.get("max_process_cpu", "nan"))
        total_cpu = as_float(fields.get("total_cpu", "nan"))
        snapshots += 1
        if math.isnan(max_process_cpu) or process_cpu > max_process_cpu:
            max_process_cpu = process_cpu
            max_process = fields.get("max_process", "NA")
        if math.isnan(max_total_cpu) or total_cpu > max_total_cpu:
            max_total_cpu = total_cpu

    quiet_candidate = (
        snapshots > 0
        and not math.isnan(max_process_cpu)
        and not math.isnan(max_total_cpu)
        and max_process_cpu < args.load_threshold
        and max_total_cpu < args.total_threshold
    )
    blocker = "none" if quiet_candidate else max_process
    print(f"quiet_candidate={'true' if quiet_candidate else 'false'}")
    print(f"snapshots={snapshots}")
    print(f"max_process_cpu={format_float(max_process_cpu)}")
    print(f"max_total_cpu={format_float(max_total_cpu)}")
    print(f"max_process={max_process}")
    print(f"blocker={blocker}")
    print(f"load_threshold={args.load_threshold:.1f}")
    print(f"total_threshold={args.total_threshold:.1f}")


if __name__ == "__main__":
    main()
