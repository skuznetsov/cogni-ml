#!/usr/bin/env python3
"""Build retrieval TSVs from source-code/document chunks.

Docs TSV format:
  doc_id<TAB>text

Queries TSV format:
  query_id<TAB>expected_doc_id<TAB>query_text

The generated queries are source-location/lead based. They are not human
relevance labels; they are a larger source-shaped falsifier for compressed
full-depth embedding/index experiments.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_EXTENSIONS = ".c,.h,.sql,.py,.md,.cr,.sh"


def clean_cell(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\t", " ")).strip()


def slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", text.lower()).strip("_")
    return out[:96] or "chunk"


def read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc


def interesting_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if not stripped:
            continue
        if stripped.startswith(("//", "--")) and len(stripped) < 24:
            continue
        if stripped.startswith("#") and not in_fence and len(stripped) < 16:
            continue
        out.append(stripped)
    return out


def make_chunks(
    root: Path,
    rel: Path,
    chunk_lines: int,
    stride_lines: int,
    min_chars: int,
    max_chars: int,
) -> list[tuple[str, int, int, str]]:
    path = root / rel
    lines = read_lines(path)
    chunks: list[tuple[str, int, int, str]] = []
    if not lines:
        return chunks
    chunk_idx = 0
    start = 0
    while start < len(lines):
        end = min(len(lines), start + chunk_lines)
        window = interesting_lines(lines[start:end])
        body = clean_cell(" ".join(window))
        if len(body) >= min_chars:
            text = clean_cell(f"File {rel.as_posix()} lines {start + 1}-{end}. {body}")[:max_chars]
            doc_id = f"{slug(rel.as_posix())}__l{start + 1:05d}_{end:05d}__{chunk_idx:04d}"
            chunks.append((doc_id, start + 1, end, text))
            chunk_idx += 1
        if end == len(lines):
            break
        start += stride_lines
    return chunks


def first_code_lead(text: str, max_chars: int) -> str:
    value = clean_cell(text)
    value = re.sub(r"^File .*? lines \d+-\d+\.\s*", "", value)
    return value[:max_chars].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--extensions", default=DEFAULT_EXTENSIONS)
    parser.add_argument("--max-docs", type=int, default=2000)
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--chunk-lines", type=int, default=32)
    parser.add_argument("--stride-lines", type=int, default=24)
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--max-chars", type=int, default=1600)
    parser.add_argument("--query-chars", type=int, default=260)
    parser.add_argument("--query-mode", choices=("path-lead", "lead-only", "symbol-lead"), default="path-lead")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extensions = {ext.strip() for ext in args.extensions.split(",") if ext.strip()}

    docs: list[tuple[str, str, int, int, str]] = []
    for path in sorted(root.rglob("*")):
        if len(docs) >= args.max_docs:
            break
        if not path.is_file() or path.suffix not in extensions:
            continue
        rel = path.relative_to(root)
        rel_parts = set(rel.parts)
        if ".git" in rel_parts or ".idea" in rel_parts or "build" in rel_parts:
            continue
        for doc_id, start, end, text in make_chunks(
            root,
            rel,
            args.chunk_lines,
            args.stride_lines,
            args.min_chars,
            args.max_chars,
        ):
            docs.append((doc_id, rel.as_posix(), start, end, text))
            if len(docs) >= args.max_docs:
                break

    if not docs:
        raise SystemExit("no source chunks produced docs")

    queries: list[tuple[str, str, str]] = []
    for doc_id, rel, start, end, text in docs:
        lead = first_code_lead(text, args.query_chars)
        if args.query_mode == "lead-only":
            query = lead
        elif args.query_mode == "symbol-lead":
            symbols = " ".join(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", lead)[:12])
            query = clean_cell(f"Find the source chunk involving {symbols}. {lead}")
        else:
            query = clean_cell(f"Find the source chunk in {rel} around lines {start}-{end}. {lead}")
        queries.append((f"q_{doc_id}", doc_id, query))
        if len(queries) >= args.max_queries:
            break

    docs_path = out_dir / "docs.tsv"
    queries_path = out_dir / "queries.tsv"
    docs_path.write_text("\n".join(f"{doc_id}\t{text}" for doc_id, _, _, _, text in docs) + "\n")
    queries_path.write_text("\n".join(f"{qid}\t{doc_id}\t{query}" for qid, doc_id, query in queries) + "\n")
    print(f"docs={len(docs)} queries={len(queries)}")
    print(f"docs_tsv={docs_path}")
    print(f"queries_tsv={queries_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
