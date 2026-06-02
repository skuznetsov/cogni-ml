#!/usr/bin/env python3
"""Score unchecked Qwen self-draft route quality.

Inputs are the diagnostic artifacts produced by:

- qwen35_self_draft_code_attractor_suite.py
- qwen35_code_block_compile_check.py
- qwen35_code_candidate_repair.py

The scorer intentionally treats the draft output as the no-validator candidate.
Exact output is used only as a reference for agreement/timing and optional
baseline fields. This avoids promoting routes that merely match exact text while
producing unusable code.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path


STRONG_STATUSES = {
    "same_attractor_unchecked",
    "same_text_no_code_unchecked",
    "same_attractor_compile_ok",
}

DRIFT_STATUSES = {
    "drift_or_collapse",
    "topic_or_format_collapse",
}


@dataclass(frozen=True)
class RoutePaths:
    name: str
    summary: Path
    compile_summary: Path | None
    repair_summary: Path | None


def read_tsv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as io:
        return list(csv.DictReader(io, delimiter="\t"))


def ffloat(value: str | None) -> float:
    try:
        return float(value or "0")
    except ValueError:
        return 0.0


def fint(value: str | None) -> int:
    try:
        return int(value or "0")
    except ValueError:
        return 0


def has_think_leak(path: str | None) -> bool:
    if not path:
        return False
    p = Path(path)
    if not p.exists():
        return False
    text = p.read_text(encoding="utf-8", errors="replace").lower()
    return "<think>" in text or "</think>" in text


def draft_stub_like(row: dict[str, str], compile_row: dict[str, str] | None) -> bool:
    code_chars = fint(row.get("draft_code_chars"))
    if compile_row:
        substantive = compile_row.get("substantive_code")
        non_comment = fint(compile_row.get("non_comment_lines"))
        constructs = fint(compile_row.get("construct_count"))
        if substantive == "1":
            return False
        if code_chars > 0 and (non_comment <= 2 or constructs == 0):
            return True
    return code_chars > 0 and code_chars < 80


def route_from_arg(raw: str, root: Path | None) -> RoutePaths:
    if "=" in raw:
        name, base = raw.split("=", 1)
        base_path = Path(base)
    else:
        base_path = (root / raw) if root else Path(raw)
        name = base_path.name
    summary = base_path / "summary.tsv" if base_path.is_dir() else base_path
    if not summary.exists():
        raise SystemExit(f"missing summary for route {name}: {summary}")

    if root:
        compile_candidates = [
            root / f"{name}_compile_shape" / "compile_summary.tsv",
            root / f"{name}_compile" / "compile_summary.tsv",
        ]
        repair_candidates = [
            root / f"{name}_repair_shape" / "repair_summary.tsv",
            root / f"{name}_repair" / "repair_summary.tsv",
        ]
    else:
        parent = summary.parent.parent
        compile_candidates = [
            parent / f"{name}_compile_shape" / "compile_summary.tsv",
            parent / f"{name}_compile" / "compile_summary.tsv",
        ]
        repair_candidates = [
            parent / f"{name}_repair_shape" / "repair_summary.tsv",
            parent / f"{name}_repair" / "repair_summary.tsv",
        ]

    compile_summary = next((p for p in compile_candidates if p.exists()), None)
    repair_summary = next((p for p in repair_candidates if p.exists()), None)
    return RoutePaths(name, summary, compile_summary, repair_summary)


def score_row(
    route: str,
    row: dict[str, str],
    compile_by_key: dict[tuple[str, str], dict[str, str]],
    repair_by_key: dict[tuple[str, str], dict[str, str]],
) -> dict[str, object]:
    name = row["name"]
    compile_row = compile_by_key.get((name, "draft"))
    repair_row = repair_by_key.get((name, "draft"))

    status = row.get("status", "")
    lcs = ffloat(row.get("lcs_ratio"))
    word = ffloat(row.get("word_ratio"))
    agreement = ffloat(row.get("agreement_ratio"))
    chain_ms = ffloat(row.get("chain_ms"))
    exact_ms = ffloat(row.get("exact_ms"))
    speed_ratio = chain_ms / exact_ms if exact_ms > 0 else 0.0

    strong = status in STRONG_STATUSES or (lcs >= 0.75 and word >= 0.65)
    partial = status == "partial_attractor_shifted" or (lcs >= 0.50 and word >= 0.45)
    drift = status in DRIFT_STATUSES and not strong

    think_leak = has_think_leak(row.get("draft_text"))
    substantive = compile_row is not None and compile_row.get("substantive_code") == "1"
    # A tiny/comment-only snippet can compile and even survive "repair", but it
    # is not useful code. Gate compile/repair credit on the substantive-code
    # shape check so route scores do not reward placeholders like "# ...".
    compile_ok = substantive and compile_row is not None and compile_row.get("ok") == "1"
    repair_ok = substantive and repair_row is not None and repair_row.get("repaired_ok") == "1"
    stub = draft_stub_like(row, compile_row)

    score = 0.0
    score += 35.0 if strong else (15.0 if partial else 0.0)
    score += 20.0 if substantive else 0.0
    score += 15.0 if compile_ok else (10.0 if repair_ok else 0.0)
    score += 15.0 if 0.0 < speed_ratio < 0.97 else (8.0 if 0.0 < speed_ratio < 1.02 else 0.0)
    score -= 30.0 if drift else 0.0
    score -= 25.0 if think_leak else 0.0
    score -= 20.0 if stub else 0.0
    score = max(0.0, min(100.0, score))

    return {
        "route": route,
        "name": name,
        "score": f"{score:.3f}",
        "status": status,
        "strong": int(strong),
        "partial": int(partial),
        "drift": int(drift),
        "think_leak": int(think_leak),
        "stub": int(stub),
        "substantive_code": int(substantive),
        "compile_ok": int(compile_ok),
        "repair_ok": int(repair_ok),
        "agreement_ratio": f"{agreement:.6f}",
        "lcs_ratio": f"{lcs:.6f}",
        "word_ratio": f"{word:.6f}",
        "speed_ratio": f"{speed_ratio:.6f}",
        "chain_ms": f"{chain_ms:.3f}",
        "exact_ms": f"{exact_ms:.3f}",
        "draft_text": row.get("draft_text", ""),
    }


def summarize_route(route: str, rows: list[dict[str, object]]) -> dict[str, object]:
    scores = [ffloat(str(r["score"])) for r in rows]
    speed = [ffloat(str(r["speed_ratio"])) for r in rows if ffloat(str(r["speed_ratio"])) > 0]
    return {
        "route": route,
        "rows": len(rows),
        "score_mean": f"{statistics.mean(scores):.3f}" if scores else "0",
        "score_median": f"{statistics.median(scores):.3f}" if scores else "0",
        "speed_median": f"{statistics.median(speed):.6f}" if speed else "0",
        "strong": sum(int(r["strong"]) for r in rows),
        "partial": sum(int(r["partial"]) for r in rows),
        "drift": sum(int(r["drift"]) for r in rows),
        "think_leak": sum(int(r["think_leak"]) for r in rows),
        "stub": sum(int(r["stub"]) for r in rows),
        "substantive_code": sum(int(r["substantive_code"]) for r in rows),
        "compile_ok": sum(int(r["compile_ok"]) for r in rows),
        "repair_ok": sum(int(r["repair_ok"]) for r in rows),
    }


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as io:
        writer = csv.DictWriter(io, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=None, help="Experiment root containing route directories")
    ap.add_argument("--route", action="append", default=[], help="Route name under --root, path, or name=path")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/qwen_no_validator_quality_score"))
    args = ap.parse_args()

    if not args.route:
        if args.root is None:
            raise SystemExit("pass --root with route dirs or at least one --route")
        args.route = [
            p.name
            for p in sorted(args.root.iterdir())
            if p.is_dir() and (p / "summary.tsv").exists()
        ]
    if not args.route:
        raise SystemExit("no route summaries found")

    all_rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []
    for raw in args.route:
        paths = route_from_arg(raw, args.root)
        summary_rows = read_tsv(paths.summary)
        compile_rows = read_tsv(paths.compile_summary)
        repair_rows = read_tsv(paths.repair_summary)
        compile_by_key = {(row["name"], row["kind"]): row for row in compile_rows}
        repair_by_key = {(row["name"], row["kind"]): row for row in repair_rows}

        scored = [score_row(paths.name, row, compile_by_key, repair_by_key) for row in summary_rows]
        all_rows.extend(scored)
        route_rows.append(summarize_route(paths.name, scored))

    route_rows.sort(key=lambda r: ffloat(str(r["score_mean"])), reverse=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_prompt_path = args.out_dir / "quality_per_prompt.tsv"
    route_path = args.out_dir / "quality_by_route.tsv"
    write_tsv(
        per_prompt_path,
        all_rows,
        [
            "route",
            "name",
            "score",
            "status",
            "strong",
            "partial",
            "drift",
            "think_leak",
            "stub",
            "substantive_code",
            "compile_ok",
            "repair_ok",
            "agreement_ratio",
            "lcs_ratio",
            "word_ratio",
            "speed_ratio",
            "chain_ms",
            "exact_ms",
            "draft_text",
        ],
    )
    write_tsv(
        route_path,
        route_rows,
        [
            "route",
            "rows",
            "score_mean",
            "score_median",
            "speed_median",
            "strong",
            "partial",
            "drift",
            "think_leak",
            "stub",
            "substantive_code",
            "compile_ok",
            "repair_ok",
        ],
    )
    print(f"quality_by_route={route_path}")
    print(f"quality_per_prompt={per_prompt_path}")
    for row in route_rows:
        print(
            f"{row['route']}\tscore_mean={row['score_mean']}\tspeed_median={row['speed_median']}\t"
            f"strong={row['strong']}/{row['rows']}\tdrift={row['drift']}\t"
            f"think={row['think_leak']}\tstub={row['stub']}\t"
            f"substantive={row['substantive_code']}\tcompile={row['compile_ok']}\trepair={row['repair_ok']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
