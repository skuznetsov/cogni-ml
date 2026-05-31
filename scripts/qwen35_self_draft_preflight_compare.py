#!/usr/bin/env python3
"""Compare short no-validator self-draft preflight rows with a longer reference.

The input files are summary.tsv files emitted by
scripts/qwen35_self_draft_code_attractor_suite.py. This helper evaluates a
simple threshold gate and reports per-prompt decisions plus a confusion matrix.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as io:
        return {row["name"]: row for row in csv.DictReader(io, delimiter="\t")}


def f64(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or "0")
    except ValueError:
        return 0.0


def gate(row: dict[str, str], *, min_lcs: float, min_word: float, min_agreement: float) -> bool:
    return (
        f64(row, "lcs_ratio") >= min_lcs
        and f64(row, "word_ratio") >= min_word
        and f64(row, "agreement_ratio") >= min_agreement
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preflight", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--min-lcs", type=float, default=0.75)
    ap.add_argument("--min-word", type=float, default=0.65)
    ap.add_argument("--min-agreement", type=float, default=0.0)
    args = ap.parse_args()

    preflight = load(args.preflight)
    reference = load(args.reference)
    tp = fp = tn = fn = 0
    print(
        "name\tpreflight_gate\treference_strong\tpreflight_status\treference_status\t"
        "pre_lcs\tpre_word\tpre_agree\tref_lcs\tref_word\tref_agree"
    )
    for name in sorted(reference):
        if name not in preflight:
            raise SystemExit(f"missing preflight row for {name}")
        p = preflight[name]
        r = reference[name]
        pred = gate(p, min_lcs=args.min_lcs, min_word=args.min_word, min_agreement=args.min_agreement)
        actual = r.get("status", "").startswith("same_attractor")
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
        print(
            f"{name}\t{int(pred)}\t{int(actual)}\t{p.get('status','')}\t{r.get('status','')}\t"
            f"{f64(p,'lcs_ratio'):.6f}\t{f64(p,'word_ratio'):.6f}\t{f64(p,'agreement_ratio'):.6f}\t"
            f"{f64(r,'lcs_ratio'):.6f}\t{f64(r,'word_ratio'):.6f}\t{f64(r,'agreement_ratio'):.6f}"
        )
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    print("")
    print("metric\tvalue")
    print(f"tp\t{tp}")
    print(f"fp\t{fp}")
    print(f"tn\t{tn}")
    print(f"fn\t{fn}")
    print(f"precision\t{precision:.6f}")
    print(f"recall\t{recall:.6f}")
    print(f"accuracy\t{accuracy:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
