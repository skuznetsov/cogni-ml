#!/usr/bin/env python3
"""Run no-validator Qwen self-draft coding-attractor probes.

This suite is intentionally diagnostic, not a product decoder. It compares the
unchecked self-draft token chain against exact greedy text on coding prompts and
summarizes whether the draft remains in the same semantic/code attractor despite
low strict token agreement.
"""

from __future__ import annotations

import argparse
import ast
import csv
import difflib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptCase:
    name: str
    prompt: str


TEXT_RE = re.compile(r'draft_text=("(?:[^"\\]|\\.)*") exact_text=("(?:[^"\\]|\\.)*")')
FIELD_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)=([^\s]+)")
CODE_RE = re.compile(r"```(?:crystal)?\n(.*?)\n```", re.S)


def load_prompts(path: Path) -> list[PromptCase]:
    cases: list[PromptCase] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        obj = json.loads(raw)
        try:
            name = str(obj["name"])
            prompt = str(obj["prompt"])
        except KeyError as exc:
            raise ValueError(f"{path}:{lineno}: missing {exc.args[0]!r}") from exc
        cases.append(PromptCase(name=name, prompt=prompt))
    return cases


def run_to_file(cmd: list[str], path: Path, *, timeout: int | None = None) -> int:
    """Run a command with output redirected to a file.

    `scripts/run_safe.sh` starts a background watchdog that inherits stdout.
    Capturing stdout with PIPE can therefore wait for the watchdog's EOF even
    after the shell child has exited. File redirection avoids that pipe-lifetime
    trap and keeps this suite bounded by the explicit subprocess timeout.
    """
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


def parse_metrics(line: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in FIELD_RE.finditer(line.split(" draft_text=", 1)[0])}


def parse_probe_metrics(log_text: str) -> tuple[dict[str, str], str]:
    metrics: dict[str, str] = {}
    text_line = ""
    for raw_line in log_text.splitlines():
        if raw_line.startswith("self_draft_gpu_chain_updown "):
            metrics.update(parse_metrics(raw_line))
        elif raw_line.startswith("self_draft_gpu_chain "):
            metrics.update(parse_metrics(raw_line))
        elif raw_line.startswith("self_draft_gpu_chain_updown_text"):
            text_line = raw_line
            metrics.update(parse_metrics(raw_line))
        elif raw_line.startswith("self_draft_gpu_chain_text") and not text_line:
            text_line = raw_line
            metrics.update(parse_metrics(raw_line))
    return metrics, text_line


def ratio(value: str) -> float:
    if "/" not in value:
        return 0.0
    a, b = value.split("/", 1)
    denom = int(b)
    return int(a) / denom if denom else 0.0


def first_code_block(text: str) -> str:
    match = CODE_RE.search(text)
    if match:
        return match.group(1)
    marker = "```crystal\n"
    if marker in text:
        return text.split(marker, 1)[1]
    return ""


def crystal_check(code: str, *, out_path: Path, crystal: str, timeout: int, max_mem_mb: int, run_safe: Path) -> tuple[bool, str]:
    if not code.strip():
        return (False, "no_code_block")
    out_path.write_text(code + "\n", encoding="utf-8")
    check_log = out_path.with_suffix(out_path.suffix + ".check.log")
    returncode = run_to_file([
        str(run_safe),
        crystal,
        str(timeout),
        str(max_mem_mb),
        "build",
        "--no-codegen",
        str(out_path),
    ], check_log, timeout=timeout + 30)
    output = check_log.read_text(encoding="utf-8", errors="replace") if check_log.exists() else ""
    output = output.strip().replace("\t", " ").replace("\n", " | ")
    return (returncode == 0, output[-500:])


def classify(
    metrics: dict[str, str],
    draft: str,
    exact: str,
    draft_ok: bool,
    exact_ok: bool,
    *,
    compile_checked: bool,
) -> str:
    lcs = ratio(metrics.get("lcs_agreement", "0/0"))
    positional = ratio(metrics.get("agreement", "0/0"))
    has_code = bool(first_code_block(draft)) and bool(first_code_block(exact))
    word_ratio = difflib.SequenceMatcher(None, exact.split(), draft.split()).ratio()
    if not has_code:
        if lcs >= 0.75 and word_ratio >= 0.65:
            return "same_text_no_code_unchecked" if not compile_checked else "same_text_no_code"
        return "topic_or_format_collapse"
    if lcs >= 0.75 and word_ratio >= 0.65:
        if not compile_checked:
            return "same_attractor_unchecked"
        if draft_ok and exact_ok:
            return "same_attractor_compile_ok"
        return "same_attractor_code_invalid"
    if lcs >= 0.50 and positional < 0.35:
        return "partial_attractor_shifted"
    return "drift_or_collapse"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, default=Path("/tmp/qwen35_probe_tail_salvage_handoff"))
    ap.add_argument("--model", type=Path, default=None, help="Optional GGUF model path passed through to the probe")
    ap.add_argument("--prompts", type=Path, default=Path("examples/qwen_self_draft_code_prompts.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/qwen_self_draft_code_attractor_suite"))
    ap.add_argument("--run-safe", type=Path, default=Path("scripts/run_safe.sh"))
    ap.add_argument("--crystal", default="crystal")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-mem-mb", type=int, default=22000)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=768)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--layers", default="0,2")
    ap.add_argument("--exact-first", type=int, default=0, help="Emit N exact greedy tokens before the no-validator draft chain")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-crystal-check", action="store_true")
    ap.add_argument("--extra-probe-arg", action="append", default=[], help="Additional argument passed through to the probe; may be repeated")
    args = ap.parse_args()

    cases = load_prompts(args.prompts)
    if args.limit > 0:
        cases = cases[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for case in cases:
        started = time.time()
        log_path = args.out_dir / f"{case.name}.log"
        cmd = [
            str(args.run_safe),
            str(args.binary),
            str(args.timeout),
            str(args.max_mem_mb),
            "--prompt-as-prefix",
            f"--prompt-name={case.name}",
            f"--prompt={case.prompt}",
            f"--tokens={args.tokens}",
            "--calib-tokens=10000",
            f"--ranks={args.rank}",
            "--basis=pca",
            "--pca-iters=8",
            f"--simulate-logits-rank={args.rank}",
            f"--simulate-logits-layers={args.layers}",
            f"--simulate-self-draft-gpu-chain={args.steps}",
            "--simulate-self-draft-gpu-chain-text",
            "--simulate-self-draft-gpu-chain-top2",
        ]
        if args.exact_first > 0:
            cmd.append(f"--simulate-self-draft-gpu-chain-exact-first={args.exact_first}")
        cmd.extend(args.extra_probe_arg)
        if args.model is not None:
            cmd.insert(4, f"--model={args.model}")
        returncode = run_to_file(cmd, log_path, timeout=args.timeout + 45)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        metrics, line = parse_probe_metrics(log_text)
        if not line:
            rows.append({
                "name": case.name,
                "status": "probe_failed",
                "returncode": returncode,
                "log": str(log_path),
                "elapsed_s": f"{time.time() - started:.3f}",
            })
            continue
        text_match = TEXT_RE.search(line)
        if not text_match:
            rows.append({
                "name": case.name,
                "status": "parse_failed",
                "returncode": returncode,
                "log": str(log_path),
                "elapsed_s": f"{time.time() - started:.3f}",
            })
            continue
        draft = ast.literal_eval(text_match.group(1))
        exact = ast.literal_eval(text_match.group(2))
        draft_path = args.out_dir / f"{case.name}.draft.txt"
        exact_path = args.out_dir / f"{case.name}.exact.txt"
        draft_path.write_text(draft, encoding="utf-8")
        exact_path.write_text(exact, encoding="utf-8")
        draft_code = first_code_block(draft)
        exact_code = first_code_block(exact)
        if args.skip_crystal_check:
            draft_ok, draft_err = False, "skipped"
            exact_ok, exact_err = False, "skipped"
        else:
            draft_ok, draft_err = crystal_check(
                draft_code,
                out_path=args.out_dir / f"{case.name}.draft.first.cr",
                crystal=args.crystal,
                timeout=90,
                max_mem_mb=2500,
                run_safe=args.run_safe,
            )
            exact_ok, exact_err = crystal_check(
                exact_code,
                out_path=args.out_dir / f"{case.name}.exact.first.cr",
                crystal=args.crystal,
                timeout=90,
                max_mem_mb=2500,
                run_safe=args.run_safe,
            )
        words_draft = draft.split()
        words_exact = exact.split()
        row: dict[str, object] = {
            "name": case.name,
            "status": classify(
                metrics,
                draft,
                exact,
                draft_ok,
                exact_ok,
                compile_checked=not args.skip_crystal_check,
            ),
            "returncode": returncode,
            "agreement": metrics.get("agreement", ""),
            "agreement_ratio": f"{ratio(metrics.get('agreement', '0/0')):.6f}",
            "lcs_agreement": metrics.get("lcs_agreement", ""),
            "lcs_ratio": f"{ratio(metrics.get('lcs_agreement', '0/0')):.6f}",
            "first_mismatch": metrics.get("first_mismatch", ""),
            "top2_agreement": metrics.get("top2_agreement", ""),
            "top2_rescues": metrics.get("top2_rescues", ""),
            "salvage_accepted": metrics.get("salvage_accepted", ""),
            "salvage_corrections": metrics.get("salvage_corrections", ""),
            "salvage_dropped": metrics.get("salvage_dropped", ""),
            "submit_ms": metrics.get("submit_ms", ""),
            "wait_ms": metrics.get("wait_ms", ""),
            "chain_ms": metrics.get("chain_ms", ""),
            "exact_ms": metrics.get("exact_ms", ""),
            "word_ratio": f"{difflib.SequenceMatcher(None, words_exact, words_draft).ratio():.6f}",
            "draft_compile": "skipped" if args.skip_crystal_check else int(draft_ok),
            "exact_compile": "skipped" if args.skip_crystal_check else int(exact_ok),
            "draft_code_chars": len(draft_code),
            "exact_code_chars": len(exact_code),
            "draft_text": str(draft_path),
            "exact_text": str(exact_path),
            "log": str(log_path),
            "draft_error": draft_err,
            "exact_error": exact_err,
            "elapsed_s": f"{time.time() - started:.3f}",
        }
        rows.append(row)
        print(
            f"{case.name}\t{row['status']}\tagreement={row['agreement']}\t"
            f"lcs={row['lcs_agreement']}\tword={row['word_ratio']}\t"
            f"compile={row['draft_compile']}/{row['exact_compile']}\tlog={log_path}",
            flush=True,
        )

    summary_path = args.out_dir / "summary.tsv"
    fields = [
        "name",
        "status",
        "returncode",
        "agreement",
        "agreement_ratio",
        "lcs_agreement",
        "lcs_ratio",
        "first_mismatch",
        "top2_agreement",
        "top2_rescues",
        "salvage_accepted",
        "salvage_corrections",
        "salvage_dropped",
        "submit_ms",
        "wait_ms",
        "chain_ms",
        "exact_ms",
        "word_ratio",
        "draft_compile",
        "exact_compile",
        "draft_code_chars",
        "exact_code_chars",
        "elapsed_s",
        "draft_text",
        "exact_text",
        "log",
        "draft_error",
        "exact_error",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as io:
        writer = csv.DictWriter(io, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary={summary_path}")
    return 0 if all(int(row.get("returncode", 1)) == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
