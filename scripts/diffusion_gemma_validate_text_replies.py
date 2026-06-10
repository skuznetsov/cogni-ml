#!/usr/bin/env python3
"""Validate compact DiffusionGemma resident TEXT replies."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


TRACE_MARKERS = ("<|channel>", "<|", "Draft", "Topic:", "Constraint", "Goal:", "thought")


def unescape_field(value: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(value):
        ch = value[i]
        if ch != "\\" or i + 1 >= len(value):
            out.append(ch)
            i += 1
            continue
        nxt = value[i + 1]
        if nxt == "n":
            out.append("\n")
        elif nxt == "r":
            out.append("\r")
        elif nxt == "t":
            out.append("\t")
        elif nxt == "\\":
            out.append("\\")
        else:
            out.append(nxt)
        i += 2
    return "".join(out)


def parse_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in line.split("\t")[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    return fields


def valid_sentence(text: str) -> bool:
    if len(text) < 20:
        return False
    if any(marker in text for marker in TRACE_MARKERS):
        return False
    if '"' in text or "*" in text:
        return False
    words = re.findall(r"[A-Za-z']+", text.lower())
    if any(a == b for a, b in zip(words, words[1:])):
        return False
    return text.rstrip().endswith((".", "!", "?"))


def validate(path: Path, max_reply_bytes: int, allow_no_clean: bool) -> tuple[int, list[str]]:
    rows = path.read_text(errors="replace").splitlines()
    errors: list[str] = []
    ok_rows = 0
    for idx, line in enumerate(rows, 1):
        encoded_len = len(line.encode())
        if encoded_len >= max_reply_bytes:
            errors.append(f"line {idx}: reply too large: {encoded_len} bytes")
        if allow_no_clean and line.startswith("TEXT_NO_CLEAN\t"):
            fields = parse_fields(line)
            for required in ("steps", "total_ms", "raw_bytes"):
                if required not in fields:
                    errors.append(f"line {idx}: missing {required}")
            continue
        if not line.startswith("TEXT_OK\t"):
            errors.append(f"line {idx}: not TEXT_OK")
            continue
        ok_rows += 1
        fields = parse_fields(line)
        for required in ("steps", "total_ms", "bytes", "text"):
            if required not in fields:
                errors.append(f"line {idx}: missing {required}")
        if "text" not in fields or "bytes" not in fields:
            continue
        text = unescape_field(fields["text"])
        try:
            declared_bytes = int(fields["bytes"])
        except ValueError:
            errors.append(f"line {idx}: invalid bytes={fields['bytes']!r}")
            continue
        actual_bytes = len(text.encode())
        if actual_bytes != declared_bytes:
            errors.append(f"line {idx}: bytes mismatch declared={declared_bytes} actual={actual_bytes}")
        if not valid_sentence(text):
            errors.append(f"line {idx}: text failed clean sentence checks: {text!r}")
    return ok_rows, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reply_log", type=Path)
    parser.add_argument("--max-reply-bytes", type=int, default=16 * 1024)
    parser.add_argument("--allow-no-clean", action="store_true")
    args = parser.parse_args()

    ok_rows, errors = validate(args.reply_log, args.max_reply_bytes, args.allow_no_clean)
    print(f"text_reply_validation rows_ok={ok_rows} errors={len(errors)} reply_log={args.reply_log}")
    for error in errors:
        print(f"text_reply_validation_error {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
