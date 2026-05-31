#!/usr/bin/env python3
"""Compare no-validator self-draft route summaries and simple selectors."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def base_name(name: str) -> str:
    for suffix in ("_fence_prefill", "_think_closed", "_baseline"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def ffloat(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_summary(label: str, path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for row in csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"):
        row = dict(row)
        row["route"] = label
        row["base_name"] = base_name(row["name"])
        row["chain_exact_ratio"] = f"{ffloat(row.get('chain_ms', '0')) / ffloat(row.get('exact_ms', '1')):.8f}" if ffloat(row.get("exact_ms", "")) else "0"
        row["strong"] = str(int(row.get("status") in {"same_attractor_unchecked", "same_text_no_code_unchecked"} or (ffloat(row.get("lcs_ratio", "")) >= 0.75 and ffloat(row.get("word_ratio", "")) >= 0.65)))
        rows[row["base_name"]] = row
    return rows


def route_cost(row: dict[str, str], *, unsafe_penalty: float) -> float:
    ratio = ffloat(row["chain_exact_ratio"])
    if row.get("strong") != "1":
        ratio += unsafe_penalty
    return ratio


def summarize_choice(name: str, choices: list[dict[str, str]], *, unsafe_penalty: float) -> dict[str, str]:
    best = min(choices, key=lambda row: route_cost(row, unsafe_penalty=unsafe_penalty))
    oracle = min(choices, key=lambda row: ffloat(row["chain_exact_ratio"]))
    return {
        "name": name,
        "chosen_route": best["route"],
        "chosen_ratio": best["chain_exact_ratio"],
        "chosen_strong": best["strong"],
        "oracle_route": oracle["route"],
        "oracle_ratio": oracle["chain_exact_ratio"],
        "oracle_strong": oracle["strong"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route", action="append", required=True, help="LABEL=summary.tsv; repeat for each route")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--unsafe-penalty", type=float, default=10.0)
    args = ap.parse_args()

    routes: dict[str, dict[str, dict[str, str]]] = {}
    for spec in args.route:
        label, sep, path = spec.partition("=")
        if not sep:
            raise SystemExit(f"bad --route {spec!r}, expected LABEL=path")
        routes[label] = load_summary(label, Path(path))

    names = sorted(set.intersection(*(set(rows) for rows in routes.values())))
    route_labels = list(routes)
    rows_out: list[dict[str, str]] = []
    for name in names:
        choices = [routes[label][name] for label in route_labels]
        row = summarize_choice(name, choices, unsafe_penalty=args.unsafe_penalty)
        for choice in choices:
            prefix = choice["route"]
            row[f"{prefix}_ratio"] = choice["chain_exact_ratio"]
            row[f"{prefix}_strong"] = choice["strong"]
            row[f"{prefix}_agreement"] = choice.get("agreement_ratio", "")
            row[f"{prefix}_lcs"] = choice.get("lcs_ratio", "")
            row[f"{prefix}_word"] = choice.get("word_ratio", "")
            row[f"{prefix}_status"] = choice.get("status", "")
        rows_out.append(row)

    fields = [
        "name", "chosen_route", "chosen_ratio", "chosen_strong", "oracle_route", "oracle_ratio", "oracle_strong",
    ]
    for label in route_labels:
        fields.extend([f"{label}_ratio", f"{label}_strong", f"{label}_agreement", f"{label}_lcs", f"{label}_word", f"{label}_status"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as io:
        writer = csv.DictWriter(io, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)

    chosen_ratios = [ffloat(row["chosen_ratio"]) for row in rows_out]
    oracle_ratios = [ffloat(row["oracle_ratio"]) for row in rows_out]
    print(f"rows={len(rows_out)} out={args.out}")
    print(f"chosen median={statistics.median(chosen_ratios):.6f} mean={statistics.mean(chosen_ratios):.6f} strong={sum(row['chosen_strong']=='1' for row in rows_out)}/{len(rows_out)}")
    print(f"oracle median={statistics.median(oracle_ratios):.6f} mean={statistics.mean(oracle_ratios):.6f} strong={sum(row['oracle_strong']=='1' for row in rows_out)}/{len(rows_out)}")
    for label in route_labels:
        ratios = [ffloat(row[f"{label}_ratio"]) for row in rows_out]
        print(f"route={label} median={statistics.median(ratios):.6f} mean={statistics.mean(ratios):.6f} strong={sum(row[f'{label}_strong']=='1' for row in rows_out)}/{len(rows_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
