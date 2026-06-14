#!/usr/bin/env python3
"""Compare selected-vs-foreign fallback replay route-plan rows.

The selected replay plan may contain the full mixed suite while the foreign
replay plan usually contains only fallback windows. Compare only common
prompt:canvas windows so the decision is about the fallback corridor, not the
accepted fast windows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


WINDOW_KIND = "diffusion_gemma_mixed_route_plan_window_v1"


def die(message: str) -> None:
    raise SystemExit(message)


def as_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        die(f"window {window_key(row)} has invalid {key}")
        raise AssertionError from exc


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6f}"


def window_key(row: dict[str, Any]) -> str:
    try:
        return f"{int(row['prompt_token'])}:{int(row['canvas_token'])}"
    except (KeyError, TypeError, ValueError) as exc:
        die(f"route-plan window row has invalid prompt/canvas token: {row!r}")
        raise AssertionError from exc


def selected_cost(row: dict[str, Any]) -> float:
    return as_float(row, "mixed_variant_ms")


def observed_cost(row: dict[str, Any]) -> float:
    if "observed_variant_ms" in row:
        return as_float(row, "observed_variant_ms")
    return selected_cost(row)


def load_windows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        die(f"route plan not found: {path}")
    windows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                die(f"{path} invalid JSONL at line {lineno}: {exc}")
            if row.get("kind") != WINDOW_KIND:
                continue
            key = window_key(row)
            if key in windows:
                die(f"{path} contains duplicate window {key}")
            windows[key] = row
    if not windows:
        die(f"{path} contains no {WINDOW_KIND} rows")
    return windows


def compare(selected_path: Path, foreign_path: Path) -> list[dict[str, Any]]:
    selected = load_windows(selected_path)
    foreign = load_windows(foreign_path)
    common = sorted(set(selected) & set(foreign), key=lambda key: tuple(map(int, key.split(":"))))
    if not common:
        die("selected and foreign route plans have no common windows")

    rows: list[dict[str, Any]] = []
    for key in common:
        selected_row = selected[key]
        foreign_row = foreign[key]
        selected_ms = selected_cost(selected_row)
        foreign_ms = selected_cost(foreign_row)
        ratio = selected_ms / foreign_ms if foreign_ms > 0 else math.inf
        delta = selected_ms - foreign_ms
        if ratio > 1.0:
            winner = "foreign"
        elif ratio < 1.0:
            winner = "selected"
        else:
            winner = "tie"
        rows.append(
            {
                "window": key,
                "winner": winner,
                "foreign_vs_selected": ratio,
                "delta_ms": delta,
                "selected_ms": selected_ms,
                "foreign_ms": foreign_ms,
                "selected_route": selected_row.get("selected_route", ""),
                "foreign_route": foreign_row.get("selected_route", ""),
                "selected_status": selected_row.get("status", ""),
                "foreign_status": foreign_row.get("status", ""),
                "selected_reason": selected_row.get("reason", ""),
                "foreign_reason": foreign_row.get("reason", ""),
                "selected_observed_ms": observed_cost(selected_row),
                "foreign_observed_ms": observed_cost(foreign_row),
            }
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ms = sum(float(row["selected_ms"]) for row in rows)
    foreign_ms = sum(float(row["foreign_ms"]) for row in rows)
    ratio = selected_ms / foreign_ms if foreign_ms > 0 else math.inf
    if ratio > 1.0:
        winner = "foreign"
    elif ratio < 1.0:
        winner = "selected"
    else:
        winner = "tie"
    return {
        "window": "ALL_COMMON",
        "winner": winner,
        "foreign_vs_selected": ratio,
        "delta_ms": selected_ms - foreign_ms,
        "selected_ms": selected_ms,
        "foreign_ms": foreign_ms,
        "selected_route": "",
        "foreign_route": "",
        "selected_status": "",
        "foreign_status": "",
        "selected_reason": "",
        "foreign_reason": "",
        "selected_observed_ms": math.nan,
        "foreign_observed_ms": math.nan,
    }


FIELDS = [
    "window",
    "winner",
    "foreign_vs_selected",
    "delta_ms",
    "selected_ms",
    "foreign_ms",
    "selected_route",
    "foreign_route",
    "selected_status",
    "foreign_status",
    "selected_reason",
    "foreign_reason",
    "selected_observed_ms",
    "foreign_observed_ms",
]


def print_tsv(rows: list[dict[str, Any]]) -> None:
    print("\t".join(FIELDS))
    for row in rows:
        values: list[str] = []
        for field in FIELDS:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value))
        print("\t".join(values))


def print_text(rows: list[dict[str, Any]], selected_path: Path, foreign_path: Path) -> None:
    total = aggregate(rows)
    print("DiffusionGemma fallback replay compare")
    print(f"  selected_plan={selected_path}")
    print(f"  foreign_plan={foreign_path}")
    print(
        "  aggregate common_windows=%d winner=%s foreign_vs_selected=%s "
        "selected_ms=%s foreign_ms=%s delta_ms=%s"
        % (
            len(rows),
            total["winner"],
            fmt(float(total["foreign_vs_selected"])),
            fmt(float(total["selected_ms"])),
            fmt(float(total["foreign_ms"])),
            fmt(float(total["delta_ms"])),
        )
    )
    print("  windows:")
    for row in rows:
        print(
            "    %s winner=%s foreign_vs_selected=%s selected=%sms(%s/%s) "
            "foreign=%sms(%s/%s) delta_ms=%s"
            % (
                row["window"],
                row["winner"],
                fmt(float(row["foreign_vs_selected"])),
                fmt(float(row["selected_ms"])),
                row["selected_route"],
                row["selected_status"],
                fmt(float(row["foreign_ms"])),
                row["foreign_route"],
                row["foreign_status"],
                fmt(float(row["delta_ms"])),
            )
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-route-plan", required=True, type=Path)
    parser.add_argument("--foreign-route-plan", required=True, type=Path)
    parser.add_argument("--tsv", action="store_true")
    args = parser.parse_args(argv)

    rows = compare(args.selected_route_plan, args.foreign_route_plan)
    output_rows = [aggregate(rows)] + rows
    if args.tsv:
        print_tsv(output_rows)
    else:
        print_text(rows, args.selected_route_plan, args.foreign_route_plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
