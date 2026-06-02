#!/usr/bin/env python3
"""Build a small retrieval TSV corpus from Markdown sections.

Docs TSV format:
  doc_id<TAB>text

Queries TSV format:
  query_id<TAB>expected_doc_id<TAB>query_text

The generated queries are section-title/lead based. They are not a substitute
for a human relevance benchmark, but they are a useful real-document falsifier
for shallow embedding candidate recall.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^\s*```")


def clean_cell(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\t", " ")).strip()


def slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", text.lower()).strip("_")
    return out[:80] or "section"


def first_sentence(text: str, max_chars: int) -> str:
    text = clean_cell(text)
    if len(text) <= max_chars:
        return text
    match = re.search(r"(?<=[.!?])\s+", text[:max_chars])
    if match:
        return text[: match.end()].strip()
    return text[:max_chars].strip()


def markdown_sections(path: Path, max_chars: int, min_chars: int) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = path.stem.replace("-", " ").replace("_", " ")
    current_lines: list[str] = []
    in_fence = False

    for raw in path.read_text(errors="replace").splitlines():
        if FENCE_RE.match(raw):
            in_fence = not in_fence
            continue
        if not in_fence:
            heading = HEADING_RE.match(raw)
            if heading:
                if current_lines:
                    sections.append((current_title, current_lines))
                current_title = heading.group(2).strip()
                current_lines = []
                continue
        if raw.strip():
            current_lines.append(raw.strip())

    if current_lines:
        sections.append((current_title, current_lines))

    rows: list[tuple[str, str]] = []
    for title, lines in sections:
        body = clean_cell(" ".join(lines))
        if len(body) < min_chars:
            continue
        text = clean_cell(f"{title}. {body}")[:max_chars]
        rows.append((clean_cell(title), text))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Markdown root directory")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--glob", default="*.md", help="Markdown glob relative to root")
    parser.add_argument("--max-docs", type=int, default=128)
    parser.add_argument("--max-queries", type=int, default=64)
    parser.add_argument("--max-chars", type=int, default=1400)
    parser.add_argument("--query-chars", type=int, default=220)
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--query-mode", choices=("title-lead", "lead-only"), default="title-lead")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs: list[tuple[str, str, str]] = []
    for md in sorted(root.glob(args.glob)):
        if not md.is_file():
            continue
        rel = md.relative_to(root).as_posix()
        for idx, (title, text) in enumerate(markdown_sections(md, args.max_chars, args.min_chars)):
            doc_id = f"{slug(rel)}__{idx:03d}__{slug(title)}"
            docs.append((doc_id, title, text))
            if len(docs) >= args.max_docs:
                break
        if len(docs) >= args.max_docs:
            break

    if not docs:
        raise SystemExit("no markdown sections produced docs")

    queries: list[tuple[str, str, str]] = []
    for doc_id, title, text in docs:
        body = re.sub(rf"^{re.escape(title)}\.\s*", "", text)
        lead = first_sentence(body, args.query_chars)
        if args.query_mode == "lead-only":
            query = clean_cell(lead)
        else:
            query = clean_cell(f"Find the Markdown section about {title}. {lead}")
        queries.append((f"q_{doc_id}", doc_id, query))
        if len(queries) >= args.max_queries:
            break

    docs_path = out_dir / "docs.tsv"
    queries_path = out_dir / "queries.tsv"
    docs_path.write_text("\n".join(f"{doc_id}\t{text}" for doc_id, _, text in docs) + "\n")
    queries_path.write_text("\n".join(f"{qid}\t{doc_id}\t{query}" for qid, doc_id, query in queries) + "\n")

    print(f"docs={len(docs)} queries={len(queries)}")
    print(f"docs_tsv={docs_path}")
    print(f"queries_tsv={queries_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
