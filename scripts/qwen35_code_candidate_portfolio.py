#!/usr/bin/env python3
"""Select a code candidate from draft/exact compile-check rows.

Input is compile_summary.tsv emitted by qwen35_code_block_compile_check.py. The
selector is intentionally conservative: compile success is the hard gate, then a
small deterministic preference order chooses among compiling candidates.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

KIND_PRIORITY = {
    "draft": 0,
    "exact": 1,
}


def f64(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or "0")
    except ValueError:
        return 0.0


def i64(row: dict[str, str], key: str) -> int:
    try:
        return int(row.get(key, "0") or "0")
    except ValueError:
        return 0


def candidate_score(row: dict[str, str]) -> tuple[int, float, float, int, int]:
    """Sort key for candidates; higher is better except kind priority."""
    return (
        1 if row.get("ok") == "1" else 0,
        f64(row, "lcs_ratio"),
        f64(row, "word_ratio"),
        i64(row, "code_chars"),
        -KIND_PRIORITY.get(row.get("kind", ""), 99),
    )


def select(rows: list[dict[str, str]]) -> tuple[str, dict[str, str] | None, str]:
    compiling = [row for row in rows if row.get("ok") == "1"]
    if compiling:
        best = max(compiling, key=candidate_score)
        if len(compiling) == 1:
            return ("compile_single", best, "one candidate passed compile")
        return ("compile_best", best, "multiple candidates passed compile; chose highest score")
    if rows:
        best = max(rows, key=candidate_score)
        return ("no_compile_fallback", best, "no candidate compiled; selected diagnostic fallback only")
    return ("missing", None, "no rows for prompt")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("compile_summary", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/qwen_code_candidate_portfolio"))
    ap.add_argument("--copy-selected", action="store_true", help="copy selected code files into out-dir/selected")
    args = ap.parse_args()

    with args.compile_summary.open(newline="", encoding="utf-8") as io:
        rows = list(csv.DictReader(io, delimiter="\t"))

    by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_name[row["name"]].append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = args.out_dir / "selected"
    if args.copy_selected:
        selected_dir.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict[str, str]] = []
    for name in sorted(by_name):
        decision, best, reason = select(by_name[name])
        draft = next((row for row in by_name[name] if row.get("kind") == "draft"), {})
        exact = next((row for row in by_name[name] if row.get("kind") == "exact"), {})
        selected_path = ""
        if best is not None:
            selected_path = best.get("code_path", "")
            if args.copy_selected and selected_path:
                dst = selected_dir / f"{name}.{best.get('kind', 'candidate')}.cr"
                shutil.copyfile(selected_path, dst)
                selected_path = str(dst)
        out_rows.append(
            {
                "name": name,
                "decision": decision,
                "selected_kind": best.get("kind", "") if best else "",
                "selected_ok": best.get("ok", "0") if best else "0",
                "selected_code_path": selected_path,
                "reason": reason,
                "draft_ok": draft.get("ok", ""),
                "exact_ok": exact.get("ok", ""),
                "draft_error": draft.get("error", ""),
                "exact_error": exact.get("error", ""),
                "source_status": best.get("source_status", "") if best else "",
                "lcs_ratio": best.get("lcs_ratio", "") if best else "",
                "word_ratio": best.get("word_ratio", "") if best else "",
            }
        )
        print(
            f"{name}\t{decision}\tselected={out_rows[-1]['selected_kind']}\t"
            f"draft_ok={out_rows[-1]['draft_ok']}\texact_ok={out_rows[-1]['exact_ok']}\t{reason}",
            flush=True,
        )

    out_path = args.out_dir / "portfolio_summary.tsv"
    fields = [
        "name",
        "decision",
        "selected_kind",
        "selected_ok",
        "selected_code_path",
        "reason",
        "draft_ok",
        "exact_ok",
        "draft_error",
        "exact_error",
        "source_status",
        "lcs_ratio",
        "word_ratio",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as io:
        writer = csv.DictWriter(io, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    compile_selected = sum(1 for row in out_rows if row["selected_ok"] == "1")
    print(f"summary={out_path}")
    print(f"compiled_selected={compile_selected}/{len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
