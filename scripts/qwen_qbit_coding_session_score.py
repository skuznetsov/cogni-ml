#!/usr/bin/env python3
"""Score exact and resident-QBit coding continuations with external Crystal specs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROBE_SCHEMA = "qwen-qbit-quality-v1"


def extract_crystal_source(text: str) -> str:
    fences = list(
        re.finditer(r"```(?P<language>[^\n`]*)\n(?P<body>.*?)```", text, re.DOTALL)
    )
    crystal = [match for match in fences if match.group("language").strip().lower() in {"cr", "crystal"}]
    selected = crystal[0] if crystal else (fences[0] if len(fences) == 1 else None)
    if selected is not None:
        return selected.group("body").strip("\n") + "\n"
    return text.strip() + "\n"


def extract_quality_record(log_text: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for raw in log_text.splitlines():
        line = raw.strip()
        if not line.startswith("QBIT_QUALITY_JSON="):
            continue
        value = json.loads(line.split("=", 1)[1])
        if isinstance(value, dict) and value.get("schema") == PROBE_SCHEMA:
            records.append(value)
    if len(records) != 1:
        raise ValueError(f"expected exactly one {PROBE_SCHEMA} record, observed {len(records)}")
    record = records[0]
    if (
        record.get("execution_mode") != "resident_gpu"
        or record.get("resident_layers") != record.get("full_attention_layers")
        or record.get("resident_f32_owner_layers") != []
        or record.get("resident_cache_consistent") is not True
    ):
        raise ValueError("probe record does not prove sole resident GPU KV ownership")
    return record


def _path_below(root: Path, relative: Path, label: str) -> Path:
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes project: {relative}") from exc
    return resolved


def run_external_specs(
    *,
    project: Path,
    hidden_spec: Path,
    source_path: Path,
    generated_source: str,
    output_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    project = project.resolve()
    hidden_spec = hidden_spec.resolve()
    if not project.is_dir():
        raise ValueError(f"project is not a directory: {project}")
    if not hidden_spec.is_file():
        raise ValueError(f"hidden spec is not a file: {hidden_spec}")
    source = _path_below(project, source_path, "source path")
    if not source.is_file():
        raise ValueError(f"source path is not a file: {source}")
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")

    shutil.copytree(project, output_dir)
    generated_target = _path_below(output_dir, source_path, "generated source path")
    generated_target.write_text(generated_source, encoding="utf-8")
    external_target = output_dir / "spec" / "qbit_external_hidden_spec.cr"
    external_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hidden_spec, external_target)

    try:
        environment = os.environ.copy()
        environment["CRYSTAL_CACHE_DIR"] = str(output_dir / ".crystal-cache")
        completed = subprocess.run(
            ["crystal", "spec", "--no-color"],
            cwd=output_dir,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        exit_code = completed.returncode
        output = completed.stdout
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        output = (exc.stdout or "") + f"\n[TIMEOUT] crystal spec exceeded {timeout}s\n"

    (output_dir / "qbit_external_spec.log").write_text(output, encoding="utf-8")
    return {
        "external_pass": exit_code == 0,
        "exit_code": exit_code,
        "source_sha256": hashlib.sha256(generated_source.encode("utf-8")).hexdigest(),
        "source_bytes": len(generated_source.encode("utf-8")),
        "log": str(output_dir / "qbit_external_spec.log"),
        "output_tail": output[-2000:],
    }


def classify_verdict(*, exact: dict[str, Any], candidate: dict[str, Any]) -> str:
    if exact.get("external_pass") is not True:
        return "invalid_exact_baseline"
    if candidate.get("external_pass") is True:
        return "qbit_pass"
    return "qbit_regression"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-log", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hidden-spec", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Source path relative to project")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args(argv)

    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.output.exists():
        parser.error(f"--output already exists: {args.output}")

    try:
        record = extract_quality_record(args.probe_log.read_text(encoding="utf-8", errors="replace"))
        exact_source = extract_crystal_source(str(record.get("exact_text", "")))
        candidate_source = extract_crystal_source(str(record.get("candidate_text", "")))
        args.output.mkdir(parents=True)
        exact = run_external_specs(
            project=args.project,
            hidden_spec=args.hidden_spec,
            source_path=args.source,
            generated_source=exact_source,
            output_dir=args.output / "exact",
            timeout=args.timeout,
        )
        candidate = run_external_specs(
            project=args.project,
            hidden_spec=args.hidden_spec,
            source_path=args.source,
            generated_source=candidate_source,
            output_dir=args.output / "candidate",
            timeout=args.timeout,
        )
        verdict = classify_verdict(exact=exact, candidate=candidate)
        report = {
            "schema": "qwen-qbit-coding-session-v1",
            "verdict": verdict,
            "policy": record.get("policy"),
            "prompt_tokens": record.get("prompt_tokens"),
            "prefill_chunk_size": record.get("prefill_chunk_size"),
            "prefill_append_max_groups": record.get("prefill_append_max_groups"),
            "prefill_append_cooldown_ms": record.get("prefill_append_cooldown_ms"),
            "quality": {
                "top1_matches": record.get("retire_order_top1_matches"),
                "top1_count": record.get("retire_order_top1_count"),
                "ranked_top2_matches": record.get("teacher_top2_ranked_matches"),
                "ranked_top2_count": record.get("teacher_top2_ranked_count"),
                "top2_overlap": record.get("teacher_top2_set_overlap"),
                "top2_overlap_count": record.get("teacher_top2_set_overlap_count"),
                "exact_top1_covered": record.get("teacher_exact_top1_covered"),
                "exact_top1_covered_count": record.get("teacher_top2_steps"),
                "ecs_mean": record.get("teacher_token_ecs_mean"),
                "ecs_min": record.get("teacher_token_ecs_min"),
                "raw_bytes": record.get("prefix_raw_bytes"),
                "resident_bytes": record.get("prefix_payload_bytes"),
                "density": record.get("prefix_ratio"),
            },
            "timing_ms": {
                "exact_prefill": record.get("exact_prefill_ms"),
                "resident_prefill": record.get("resident_prefill_ms"),
                "exact_decode": record.get("exact_decode_ms"),
                "resident_decode": record.get("free_decode_ms"),
            },
            "exact": exact,
            "candidate": candidate,
        }
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            "QBIT_CODING_SESSION "
            f"verdict={verdict} exact_pass={exact['external_pass']} "
            f"candidate_pass={candidate['external_pass']} report={report_path}"
        )
        return 0 if verdict == "qbit_pass" else 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
