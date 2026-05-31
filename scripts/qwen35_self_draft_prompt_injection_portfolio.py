#!/usr/bin/env python3
"""Run sequential prompt-injection portfolio probes for Qwen self-draft.

This is a thin orchestrator around qwen35_self_draft_code_attractor_suite.py.
It is intentionally sequential: one variant suite exits before the next starts.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterable


VARIANTS: dict[str, str] = {
    "baseline": "",
    "think_closed": "\n<think>\n\n</think>\n\n",
    "close_only": "\n</think>\n\n",
    "no_reason": "\nDo not write reasoning. Start immediately with Crystal code.\n",
    "fence_prefill": "\n```crystal\n",
    "final_fence_prefill": "\nFinal answer:\n```crystal\n",
}

CODEISH_PREFIXES = (
    "```",
    "#",
    "class ",
    "module ",
    "struct ",
    "enum ",
    "def ",
    "require ",
    "alias ",
    "record ",
)


def load_cases(path: Path, names: set[str] | None) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        obj = json.loads(raw)
        name = str(obj["name"])
        if names is not None and name not in names:
            continue
        cases.append({"name": name, "prompt": str(obj["prompt"])})
    return cases


def write_variant_prompts(cases: Iterable[dict[str, str]], variant: str, suffix: str, out: Path) -> None:
    with out.open("w", encoding="utf-8") as io:
        for case in cases:
            obj = {
                "name": f"{case['name']}_{variant}",
                "prompt": case["prompt"].rstrip() + suffix,
            }
            io.write(json.dumps(obj, ensure_ascii=False) + "\n")


def ffloat(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def generated_starts_codeish(path: str) -> bool:
    text = Path(path).read_text(encoding="utf-8", errors="replace").lstrip()
    return text.startswith(CODEISH_PREFIXES)


def summarize_variant(variant: str, summary: Path) -> dict[str, object]:
    rows = list(csv.DictReader(summary.open(encoding="utf-8"), delimiter="\t"))
    ratios = []
    for row in rows:
        exact_ms = ffloat(row.get("exact_ms", ""))
        chain_ms = ffloat(row.get("chain_ms", ""))
        if exact_ms > 0:
            ratios.append(chain_ms / exact_ms)

    def high_similarity_no_code(row: dict[str, str]) -> bool:
        has_code = int(row.get("draft_code_chars", "0") or 0) > 0 and int(row.get("exact_code_chars", "0") or 0) > 0
        return (not has_code) and ffloat(row.get("lcs_ratio", "")) >= 0.75 and ffloat(row.get("word_ratio", "")) >= 0.65

    def is_strong(row: dict[str, str]) -> bool:
        return row["status"] in {"same_attractor_unchecked", "same_text_no_code_unchecked", "same_attractor_compile_ok"} or high_similarity_no_code(row)

    def is_drift(row: dict[str, str]) -> bool:
        return row["status"] in {"drift_or_collapse", "topic_or_format_collapse"} and not high_similarity_no_code(row)

    strong = sum(is_strong(row) for row in rows)
    drift = sum(is_drift(row) for row in rows)
    both_code = sum(int(row.get("draft_code_chars", "0") or 0) > 0 and int(row.get("exact_code_chars", "0") or 0) > 0 for row in rows)
    draft_codeish = sum(generated_starts_codeish(row["draft_text"]) for row in rows if row.get("draft_text"))
    exact_codeish = sum(generated_starts_codeish(row["exact_text"]) for row in rows if row.get("exact_text"))
    return {
        "variant": variant,
        "rows": len(rows),
        "strong": strong,
        "partial": sum(row["status"] == "partial_attractor_shifted" for row in rows),
        "drift": drift,
        "both_code": both_code,
        "draft_codeish_start": draft_codeish,
        "exact_codeish_start": exact_codeish,
        "agreement_mean": f"{statistics.mean(ffloat(row['agreement_ratio']) for row in rows):.6f}" if rows else "0",
        "lcs_mean": f"{statistics.mean(ffloat(row['lcs_ratio']) for row in rows):.6f}" if rows else "0",
        "word_mean": f"{statistics.mean(ffloat(row['word_ratio']) for row in rows):.6f}" if rows else "0",
        "chain_exact_median": f"{statistics.median(ratios):.6f}" if ratios else "0",
        "summary": str(summary),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", type=Path, default=Path("scripts/qwen35_self_draft_code_attractor_suite.py"))
    ap.add_argument("--binary", type=Path, default=Path("/tmp/qwen35_probe_tail_salvage_handoff"))
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--prompts", type=Path, default=Path("examples/qwen_self_draft_code_prompts.jsonl"))
    ap.add_argument("--names", default="", help="Comma-separated base prompt names to include")
    ap.add_argument("--variants", default="close_only,no_reason,fence_prefill,final_fence_prefill")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-mem-mb", type=int, default=36000)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--layers", default="0,2")
    ap.add_argument("--skip-crystal-check", action="store_true")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {', '.join(unknown)}")
    names = {n.strip() for n in args.names.split(",") if n.strip()} or None
    cases = load_cases(args.prompts, names)
    if not cases:
        raise SystemExit("no prompt cases selected")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = args.out_dir / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []

    for variant in variants:
        variant_prompts = prompt_dir / f"{variant}.jsonl"
        variant_out = args.out_dir / variant
        write_variant_prompts(cases, variant, VARIANTS[variant], variant_prompts)
        cmd = [
            sys.executable,
            str(args.suite),
            f"--binary={args.binary}",
            f"--prompts={variant_prompts}",
            f"--out-dir={variant_out}",
            f"--timeout={args.timeout}",
            f"--max-mem-mb={args.max_mem_mb}",
            f"--steps={args.steps}",
            f"--tokens={args.tokens}",
            f"--rank={args.rank}",
            f"--layers={args.layers}",
        ]
        if args.model is not None:
            cmd.append(f"--model={args.model}")
        if args.skip_crystal_check:
            cmd.append("--skip-crystal-check")
        print(f"variant={variant} running {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, check=False).returncode
        summary = variant_out / "summary.tsv"
        if rc != 0 or not summary.exists():
            rows.append({"variant": variant, "rows": 0, "status": f"failed rc={rc}", "summary": str(summary)})
            continue
        row = summarize_variant(variant, summary)
        row["status"] = "ok"
        rows.append(row)
        print(
            f"variant={variant} strong={row['strong']}/{row['rows']} both_code={row['both_code']} "
            f"draft_codeish={row['draft_codeish_start']} drift={row['drift']} "
            f"lcs={row['lcs_mean']} word={row['word_mean']} chain_exact={row['chain_exact_median']}",
            flush=True,
        )

    summary_path = args.out_dir / "portfolio_summary.tsv"
    fields = [
        "variant", "status", "rows", "strong", "partial", "drift", "both_code",
        "draft_codeish_start", "exact_codeish_start", "agreement_mean", "lcs_mean",
        "word_mean", "chain_exact_median", "summary",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as io:
        writer = csv.DictWriter(io, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"portfolio_summary={summary_path}")
    return 0 if all(row.get("status") == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
