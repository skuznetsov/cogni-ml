#!/usr/bin/env python3
"""Extract a usable answer from llama.cpp DiffusionGemma prototype logs.

The PR runner is still chat-template rough and often emits channel/draft traces.
This helper keeps raw logs intact and surfaces a conservative answer candidate.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
LOG_PREFIX_RE = re.compile(r"^\d+\.\d+\.\d+\s+[IWE]\s*", re.M)
PROGRESS_RE = re.compile(r"\r?diffusion step:[^\n]*")


def normalize_text(raw: str) -> str:
    text = ANSI_RE.sub("", raw.replace("\r", "\n"))
    text = text.split("total time:", 1)[0]
    # Drop host/tool preamble. The generated text is after the final progress row.
    if "diffusion step:" in text:
        text = text.rsplit("diffusion step:", 1)[-1]
        nl = text.find("\n")
        if nl >= 0:
            text = text[nl + 1 :]
    text = LOG_PREFIX_RE.sub("", text)
    text = PROGRESS_RE.sub("", text)
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith(("cli=", "model=", "log=", "real ", "user ", "sys ")):
            continue
        if re.search(r"\b[IEW]\s+ggml_|llama_decode: failed|diffusion_generate_.*failed", stripped):
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def clean_candidate(candidate: str) -> str:
    candidate = candidate.strip().strip('"').strip()
    candidate = re.sub(r"<\|[^>]+>", "", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    candidate = candidate.replace(" .", ".").replace(" ,", ",")
    return candidate


def has_repeated_word_stutter(candidate: str) -> bool:
    words = re.findall(r"[A-Za-z']+", candidate.lower())
    return any(a == b for a, b in zip(words, words[1:]))


def score_candidate(candidate: str) -> tuple[int, int]:
    bad = sum(token in candidate.lower() for token in ("draft", "topic:", "constraint", "goal:", "thought", "user:", "<|", "*"))
    sentence_bonus = int(candidate.endswith((".", "!", "?")))
    length = len(candidate)
    length_score = min(length, 180) - max(0, length - 220)
    return (sentence_bonus - bad, length_score)


def extract_answer(raw: str) -> str:
    text = normalize_text(raw)
    candidates: list[str] = []

    # Prefer explicit quoted draft/final sentences, including the common
    # missing-closing-quote case at the end of a canvas.
    candidates.extend(m.group(1) for m in re.finditer(r'"([^"\n]{20,260})"?', text))

    for line in text.splitlines():
        stripped = line.strip().lstrip("*").strip()
        if not stripped:
            continue
        if stripped.startswith(("Topic:", "Constraint", "Goal:", "Privacy", "Speed", "Security", "Reliability", "Cost")):
            continue
        if "Draft" in stripped:
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                stripped = parts[1].strip()
        if len(stripped) >= 20:
            candidates.append(stripped)

    cleaned = [clean_candidate(c) for c in candidates]
    cleaned = [c for c in cleaned if len(c) >= 20 and not c.startswith(("/", ")", "."))]
    cleaned = [
        c
        for c in cleaned
        if not any(token in c.lower() for token in ("topic:", "constraint", "goal:", "thought", "user:"))
        and '"' not in c
        and c.count("*") == 0
        and not has_repeated_word_stutter(c)
    ]
    if not cleaned:
        return ""

    return max(cleaned, key=score_candidate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    raw = args.log.read_text(errors="replace")
    print(extract_answer(raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
