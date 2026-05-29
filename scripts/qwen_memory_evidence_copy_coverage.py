#!/usr/bin/env python3
"""Measure whether memory-grounded generations are copyable from evidence text.

This is a cheap falsifier for an evidence-pointer draft route:

* prompt-suffix replay: what the existing n-gram route can already propose from
  the current prompt history;
* evidence-pointer coverage: whether generated ids appear as contiguous spans
  inside the explicit Evidence: text even when prompt-suffix replay cannot find
  them.

The script intentionally consumes existing probe logs and tokenizes only the
small fixture evidence text. It does not load model weights.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_MODEL = (
    Path.home()
    / ".cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
)
DEFAULT_TOKENIZER = Path.home() / "SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"


@dataclass(frozen=True)
class PromptCase:
    name: str
    prompt: str
    evidence: str


@dataclass(frozen=True)
class Generation:
    name: str
    source: str
    ids: tuple[int, ...]


def parse_fixture(path: Path) -> dict[str, PromptCase]:
    cases: dict[str, PromptCase] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "::" not in line:
            raise ValueError(f"bad fixture line without '::': {line!r}")
        name, prompt = line.split("::", 1)
        evidence = ""
        if prompt.startswith("Evidence: "):
            # Existing fixture uses "Evidence: ... Answer ..." framing.
            body = prompt[len("Evidence: ") :]
            marker = " Answer "
            if marker in body:
                evidence = body.split(marker, 1)[0].strip()
            else:
                evidence = body.strip()
        cases[name] = PromptCase(name=name, prompt=prompt, evidence=evidence)
    return cases


def parse_ids(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(",") if part)


def parse_logs(paths: Iterable[Path]) -> list[Generation]:
    generations: list[Generation] = []
    seen: set[tuple[str, str, tuple[int, ...]]] = set()
    # Prefer suite rows with names; ignore generic "main" rows unless explicitly
    # named by the probe.
    pattern = re.compile(r"\bname=([^\s]+).*?\bexact_ids=([0-9,]+)")
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            match = pattern.search(line)
            if not match:
                continue
            name = match.group(1)
            if name == "main":
                continue
            ids = parse_ids(match.group(2))
            key = (str(path), name, ids)
            if key in seen:
                continue
            seen.add(key)
            generations.append(Generation(name=name, source=str(path), ids=ids))
    return generations


def tokenize(text: str, *, model: Path, tokenizer: Path) -> tuple[int, ...]:
    if not text:
        return ()
    cmd = [
        str(tokenizer),
        "-m",
        str(model),
        "-p",
        text,
        "--ids",
        "--no-bos",
        "--log-disable",
    ]
    proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"tokenizer failed for {text[:40]!r}: {proc.stderr.strip()}")
    line = proc.stdout.strip()
    try:
        parsed = ast.literal_eval(line)
    except Exception as exc:  # pragma: no cover - diagnostic path
        raise RuntimeError(f"tokenizer output is not a Python list: {line!r}") from exc
    return tuple(int(x) for x in parsed)


def longest_prefix_in_source(generated: tuple[int, ...], source: tuple[int, ...]) -> int:
    if not generated or not source:
        return 0
    best = 0
    for start in range(len(source)):
        n = 0
        while start + n < len(source) and n < len(generated) and source[start + n] == generated[n]:
            n += 1
        if n > best:
            best = n
    return best


def longest_any_run_in_source(generated: tuple[int, ...], source: tuple[int, ...]) -> int:
    if not generated or not source:
        return 0
    best = 0
    for gen_start in range(len(generated)):
        for source_start in range(len(source)):
            n = 0
            while (
                gen_start + n < len(generated)
                and source_start + n < len(source)
                and generated[gen_start + n] == source[source_start + n]
            ):
                n += 1
            if n > best:
                best = n
    return best


def token_hit_rate(generated: tuple[int, ...], source: tuple[int, ...]) -> float:
    if not generated:
        return 0.0
    source_set = set(source)
    return sum(1 for token in generated if token in source_set) / len(generated)


def prompt_suffix_replay(prompt_ids: tuple[int, ...], gamma: int, min_ngram: int, max_ngram: int) -> tuple[int, int]:
    """Return (match_len, candidate_len) for current NgramDraft-style replay."""
    if not prompt_ids:
        return (0, 0)
    max_len = min(max_ngram, len(prompt_ids))
    for n in range(max_len, min_ngram - 1, -1):
        suffix_start = len(prompt_ids) - n
        if suffix_start <= 0:
            continue
        suffix = prompt_ids[suffix_start:]
        for start in range(suffix_start - 1, -1, -1):
            if tuple(prompt_ids[start : start + n]) == suffix and start + n < len(prompt_ids):
                return (n, min(gamma, len(prompt_ids) - (start + n)))
    return (0, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, default=Path("examples/qwen_memory_grounded_pairs.txt"))
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--min-ngram", type=int, default=6)
    ap.add_argument("--max-ngram", type=int, default=8)
    ap.add_argument("logs", type=Path, nargs="+")
    args = ap.parse_args()

    cases = parse_fixture(args.fixture)
    generations = parse_logs(args.logs)
    if not generations:
        print("no named exact_ids rows found", file=sys.stderr)
        return 2

    print(
        "name\tgen_tokens\tevidence_tokens\tprompt_ngram_match\tprompt_ngram_candidates\t"
        "evidence_prefix\tbest_evidence_run\tevidence_token_hit_pct\tlog"
    )
    for gen in generations:
        case = cases.get(gen.name)
        if not case:
            continue
        evidence_ids = tokenize(case.evidence, model=args.model, tokenizer=args.tokenizer)
        prompt_ids = tokenize(case.prompt, model=args.model, tokenizer=args.tokenizer)
        match_len, replay_len = prompt_suffix_replay(prompt_ids, args.gamma, args.min_ngram, args.max_ngram)
        prefix = longest_prefix_in_source(gen.ids, evidence_ids)
        best_run = longest_any_run_in_source(gen.ids, evidence_ids)
        hit_pct = token_hit_rate(gen.ids, evidence_ids) * 100.0
        print(
            f"{gen.name}\t{len(gen.ids)}\t{len(evidence_ids)}\t{match_len}\t{replay_len}\t"
            f"{prefix}\t{best_run}\t{hit_pct:.2f}\t{gen.source}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
