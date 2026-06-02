#!/usr/bin/env python3
"""Apply small deterministic Crystal repairs to compile-check candidates.

This is a validator experiment, not an LLM repair pass. It consumes
compile_summary.tsv from qwen35_code_block_compile_check.py, applies conservative
syntax-level fixes, and reruns `crystal build --no-codegen`.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path

ERROR_RE = re.compile(r"Error: ([^|\n]+)")


def run_to_file(cmd: list[str], path: Path, *, timeout: int | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as io:
        try:
            proc = subprocess.run(cmd, check=False, timeout=timeout, text=True, stdout=io, stderr=subprocess.STDOUT)
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
    return one[-240:]


def repair_end_identifiers(code: str) -> tuple[str, bool]:
    changed = False
    out: list[str] = []
    for line in code.splitlines():
        if line.strip() == "end":
            out.append(line)
            continue
        new_line = re.sub(r"\bend\b", "finish", line)
        changed = changed or new_line != line
        out.append(new_line)
    return "\n".join(out) + ("\n" if code.endswith("\n") else ""), changed


def repair_next_local(code: str) -> tuple[str, bool]:
    changed = False
    out: list[str] = []
    for line in code.splitlines():
        new_line = line
        if re.search(r"^\s*next\s*=", new_line):
            new_line = re.sub(r"\bnext\s*=", "next_node =", new_line, count=1)
        # Only rewrite the simple local variable forms commonly emitted here;
        # keep method names such as `next?` or arbitrary strings untouched.
        new_line = re.sub(r"=\s*next\b", "= next_node", new_line)
        new_line = re.sub(r"\bnext\.", "next_node.", new_line)
        changed = changed or new_line != line
        out.append(new_line)
    return "\n".join(out) + ("\n" if code.endswith("\n") else ""), changed


def repair_abstract_int_instance_vars(code: str) -> tuple[str, bool]:
    repaired = re.sub(r"(@[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*)Int\b", r"\1Int32", code)
    return repaired, repaired != code


def repair_common_crystal_syntax(code: str, error: str) -> tuple[str, list[str]]:
    repairs: list[str] = []
    new_code = code

    if "StandardError" in new_code:
        new_code = new_code.replace("StandardError", "Exception")
        repairs.append("standard_error_to_exception")

    if "cannot use 'end'" in error:
        new_code, changed = repair_end_identifiers(new_code)
        if changed:
            repairs.append("rename_end_identifier")

    if "unexpected token: \"=\"" in error or re.search(r"^\s*next\s*=", new_code, re.M):
        new_code, changed = repair_next_local(new_code)
        if changed:
            repairs.append("rename_next_local")

    if "can't use Int as the type of instance variable" in error:
        new_code, changed = repair_abstract_int_instance_vars(new_code)
        if changed:
            repairs.append("abstract_int_ivar_to_int32")

    if "invalid char escape sequence" in error or "'\\1'" in new_code:
        replaced = new_code.replace(".gsub(/\\\\(.)/, '\\1')", ".gsub(/\\\\(.)/) { |m| m[1].to_s }")
        if replaced != new_code:
            new_code = replaced
            repairs.append("fix_regex_backref_replacement")

    if "unexpected token: \"?\"" in error or "?\\n" in new_code or '?"' in new_code:
        replacements = {
            "?\\r\\n": '"\\r\\n"',
            "?\\n": "'\\n'",
            '?"': "'\"'",
        }
        before = new_code
        for old, new in replacements.items():
            new_code = new_code.replace(old, new)
        if new_code != before:
            repairs.append("fix_question_mark_char_literals")

    return new_code, repairs


def eof_likely_needs_end(error: str) -> bool:
    return "expecting identifier 'end', not 'EOF'" in error or (
        "unexpected token: EOF" in error
        and "end" in error
        and ("expecting" in error or "not 'EOF'" in error)
    )


def compile_code(code: str, *, path: Path, log_path: Path, run_safe: Path, crystal: str, timeout: int, max_mem_mb: int) -> tuple[bool, str]:
    path.write_text(code if code.endswith("\n") else code + "\n", encoding="utf-8")
    rc = run_to_file(
        [str(run_safe), crystal, str(timeout), str(max_mem_mb), "build", "--no-codegen", str(path)],
        log_path,
        timeout=timeout + 30,
    )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    return rc == 0, "" if rc == 0 else compact_error(log_text)


def repair_one(row: dict[str, str], *, out_dir: Path, run_safe: Path, crystal: str, timeout: int, max_mem_mb: int, max_append_end: int) -> dict[str, str]:
    name = row["name"]
    kind = row["kind"]
    src = Path(row["code_path"])
    original = src.read_text(encoding="utf-8", errors="replace")
    error = row.get("error", "")
    code, repairs = repair_common_crystal_syntax(original, error)
    candidate_path = out_dir / f"{name}.{kind}.repaired.cr"
    log_path = out_dir / f"{name}.{kind}.repair.check.log"

    ok, err = compile_code(code, path=candidate_path, log_path=log_path, run_safe=run_safe, crystal=crystal, timeout=timeout, max_mem_mb=max_mem_mb)
    appended = 0
    while not ok and eof_likely_needs_end(err) and appended < max_append_end:
        code = code.rstrip() + "\nend\n"
        appended += 1
        ok, err = compile_code(code, path=candidate_path, log_path=log_path, run_safe=run_safe, crystal=crystal, timeout=timeout, max_mem_mb=max_mem_mb)
    if appended:
        repairs.append(f"append_end_x{appended}")

    # Some repairs expose the next compiler frontier. For example appending
    # missing `end`s can reveal Crystal's abstract-Int instance-var error.
    followup_rounds = 0
    while not ok and followup_rounds < 3:
        followup_rounds += 1
        next_code, followup_repairs = repair_common_crystal_syntax(code, err)
        if next_code == code:
            break
        code = next_code
        repairs.extend(followup_repairs)
        ok, err = compile_code(code, path=candidate_path, log_path=log_path, run_safe=run_safe, crystal=crystal, timeout=timeout, max_mem_mb=max_mem_mb)

    return {
        "name": name,
        "kind": kind,
        "original_ok": row.get("ok", "0"),
        "repaired_ok": str(int(ok)),
        "original_error": error,
        "repaired_error": err,
        "repairs": ",".join(repairs),
        "code_path": str(candidate_path),
        "log_path": str(log_path),
        "source_status": row.get("source_status", ""),
        "lcs_ratio": row.get("lcs_ratio", ""),
        "word_ratio": row.get("word_ratio", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("compile_summary", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/qwen_code_candidate_repair"))
    ap.add_argument("--run-safe", type=Path, default=Path("scripts/run_safe.sh"))
    ap.add_argument("--crystal", default="crystal")
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--max-mem-mb", type=int, default=2500)
    ap.add_argument("--max-append-end", type=int, default=8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.compile_summary.open(newline="", encoding="utf-8"), delimiter="\t"))
    out_rows: list[dict[str, str]] = []
    for row in rows:
        repaired = repair_one(row, out_dir=args.out_dir, run_safe=args.run_safe, crystal=args.crystal, timeout=args.timeout, max_mem_mb=args.max_mem_mb, max_append_end=args.max_append_end)
        out_rows.append(repaired)
        print(
            f"{repaired['name']}\t{repaired['kind']}\t{repaired['original_ok']}->{repaired['repaired_ok']}\t"
            f"repairs={repaired['repairs']}\terror={repaired['repaired_error']}",
            flush=True,
        )

    out_path = args.out_dir / "repair_summary.tsv"
    fields = [
        "name",
        "kind",
        "original_ok",
        "repaired_ok",
        "original_error",
        "repaired_error",
        "repairs",
        "source_status",
        "lcs_ratio",
        "word_ratio",
        "code_path",
        "log_path",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as io:
        writer = csv.DictWriter(io, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)
    original_ok = sum(1 for row in out_rows if row["original_ok"] == "1")
    repaired_ok = sum(1 for row in out_rows if row["repaired_ok"] == "1")
    print(f"summary={out_path}")
    print(f"compiled_original={original_ok}/{len(out_rows)}")
    print(f"compiled_repaired={repaired_ok}/{len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
