#!/usr/bin/env python3
"""Compile-check first code blocks from qwen self-draft suite outputs.

Input is a summary.tsv emitted by qwen35_self_draft_code_attractor_suite.py.
This avoids rerunning model inference when we only need validator/repair evidence.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path

CODE_RE = re.compile(r"```(?:crystal)?\n(.*?)(?:\n```|\Z)", re.S)
ERROR_RE = re.compile(r"Error: ([^|\n]+)")
CODEISH_RE = re.compile(r"^\s*(?:#|class\s|module\s|struct\s|enum\s|def\s|require\s|alias\s|record\s)")


def clean_implicit_candidate(text: str) -> str:
    for marker in ("\n<think>", "\n<|im_end|>", "\n<|im_start|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def best_code_candidate(candidates : list[str]) -> str:
    cleaned = [clean_implicit_candidate(candidate) for candidate in candidates]
    cleaned = [candidate for candidate in cleaned if candidate.strip()]
    if not cleaned:
        return ""
    codeish = [candidate for candidate in cleaned if CODEISH_RE.search(candidate)]
    pool = codeish or cleaned
    return max(pool, key=lambda candidate: len(candidate.strip()))


def first_code_block(text: str, *, implicit_open_fence: bool = False) -> str:
    if implicit_open_fence:
        prefix, sep, rest = text.partition("\n```")
        candidates = [prefix]
        candidates.extend(match.group(1) for match in CODE_RE.finditer(rest if sep else text))
        return best_code_candidate(candidates)
    match = CODE_RE.search(text)
    if match:
        return match.group(1)
    return ""


def run_to_file(cmd: list[str], path: Path, *, timeout: int | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as io:
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                timeout=timeout,
                text=True,
                stdout=io,
                stderr=subprocess.STDOUT,
            )
            return proc.returncode
        except subprocess.TimeoutExpired:
            io.write(f"\n[TIMEOUT] subprocess timeout after {timeout}s\n")
            return 124


def compact_error(text: str) -> str:
    one = text.strip().replace("\t", " ").replace("\n", " | ")
    match = ERROR_RE.search(one)
    if match:
        return match.group(1)[:240]
    if "[TIMEOUT]" in one:
        return "timeout"
    if not one:
        return ""
    return one[-240:]


def check_code(
    code: str,
    *,
    name: str,
    kind: str,
    out_dir: Path,
    run_safe: Path,
    crystal: str,
    timeout: int,
    max_mem_mb: int,
) -> tuple[bool, str, Path, Path]:
    code_path = out_dir / f"{name}.{kind}.first.cr"
    log_path = out_dir / f"{name}.{kind}.check.log"
    if not code.strip():
        code_path.write_text("", encoding="utf-8")
        log_path.write_text("no_code_block\n", encoding="utf-8")
        return (False, "no_code_block", code_path, log_path)
    code_path.write_text(code + "\n", encoding="utf-8")
    rc = run_to_file(
        [
            str(run_safe),
            crystal,
            str(timeout),
            str(max_mem_mb),
            "build",
            "--no-codegen",
            str(code_path),
        ],
        log_path,
        timeout=timeout + 30,
    )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    ok = rc == 0
    return (ok, "" if ok else compact_error(log_text), code_path, log_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/qwen_self_draft_compile_check"))
    ap.add_argument("--run-safe", type=Path, default=Path("scripts/run_safe.sh"))
    ap.add_argument("--crystal", default="crystal")
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--max-mem-mb", type=int, default=2500)
    ap.add_argument("--implicit-open-fence", action="store_true", help="Treat generated text as already inside an opening code fence when no fenced block is present")
    args = ap.parse_args()

    rows: list[dict[str, str]] = []
    with args.summary.open(newline="", encoding="utf-8") as io:
        source_rows = list(csv.DictReader(io, delimiter="\t"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for row in source_rows:
        name = row["name"]
        for kind in ("draft", "exact"):
            text_path = Path(row[f"{kind}_text"])
            text = text_path.read_text(encoding="utf-8", errors="replace")
            code = first_code_block(text, implicit_open_fence=args.implicit_open_fence)
            ok, error, code_path, log_path = check_code(
                code,
                name=name,
                kind=kind,
                out_dir=args.out_dir,
                run_safe=args.run_safe,
                crystal=args.crystal,
                timeout=args.timeout,
                max_mem_mb=args.max_mem_mb,
            )
            rows.append(
                {
                    "name": name,
                    "kind": kind,
                    "ok": str(int(ok)),
                    "code_chars": str(len(code)),
                    "error": error,
                    "code_path": str(code_path),
                    "log_path": str(log_path),
                    "text_path": str(text_path),
                    "source_status": row.get("status", ""),
                    "lcs_ratio": row.get("lcs_ratio", ""),
                    "word_ratio": row.get("word_ratio", ""),
                }
            )
            print(f"{name}\t{kind}\tok={int(ok)}\terror={error}", flush=True)

    out_path = args.out_dir / "compile_summary.tsv"
    fields = [
        "name",
        "kind",
        "ok",
        "code_chars",
        "error",
        "source_status",
        "lcs_ratio",
        "word_ratio",
        "code_path",
        "log_path",
        "text_path",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as io:
        writer = csv.DictWriter(io, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
