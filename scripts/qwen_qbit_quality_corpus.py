#!/usr/bin/env python3
"""Run and score a frozen Qwen adaptive-QBit quality corpus.

The task oracle is intentionally narrow and deterministic. It checks explicit
facts in both the exact F32 response and each compressed response; it is not a
general semantic model and it does not collapse the quality vector to one
score.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "examples" / "qwen_qbit_quality_heldout.jsonl"
DEFAULT_BINARY = Path("/private/tmp/qwen35_qbit_kv_head_quality_probe")
DEFAULT_RESIDENT_BINARY = Path("/private/tmp/qwen35_adaptive_resident_kv_quality_probe")
DEFAULT_OUT_DIR = Path("/private/tmp/qwen_qbit_quality_heldout")
DEFAULT_RESIDENT_OUT_DIR = Path("/private/tmp/qwen_qbit_resident_quality_heldout")
DEFAULT_RUN_SAFE = ROOT / "scripts" / "run_safe.sh"
DEFAULT_POLICY_MAPS = (
    "51:k0=bf16",
    "27=bf16,43=bf16,47=bf16,51=bf16",
)
DEFAULT_RESIDENT_MAPS = ("p4;27=bf16,43=bf16,47=bf16,51=bf16",)

PROBE_SCHEMA = "qwen-qbit-quality-v1"
REQUIRED_METRICS = (
    "retire_order_top1_matches",
    "retire_order_top1_count",
    "teacher_top2_ranked_matches",
    "teacher_top2_ranked_count",
    "teacher_top2_set_overlap",
    "teacher_top2_set_overlap_count",
    "teacher_exact_top1_covered",
    "teacher_top2_steps",
    "teacher_token_ecs_mean",
    "prefix_raw_bytes",
    "prefix_payload_bytes",
    "candidate_ids",
    "candidate_ended_with_eos",
)


def _matches(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, re.IGNORECASE | re.DOTALL) is not None
    except re.error as exc:
        raise ValueError(f"invalid semantic regex {pattern!r}: {exc}") from exc


def semantic_failures(text: str, rules: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for pattern in rules.get("required_all", []):
        if not _matches(str(pattern), text):
            failures.append(f"required_all did not match {pattern!r}")
    for group in rules.get("required_any", []):
        patterns = [str(pattern) for pattern in group]
        if not patterns or not any(_matches(pattern, text) for pattern in patterns):
            failures.append(f"required_any did not match any of {patterns!r}")
    for pattern in rules.get("forbidden", []):
        if _matches(str(pattern), text):
            failures.append(f"forbidden matched {pattern!r}")
    min_words = int(rules.get("min_words", 0))
    word_count = len(re.findall(r"\b[\w'-]+\b", text, re.UNICODE))
    if word_count < min_words:
        failures.append(f"min_words expected {min_words}, observed {word_count}")
    return failures


def extract_probe_records(log_text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in log_text.splitlines():
        line = raw.strip()
        if line.startswith("QBIT_QUALITY_JSON="):
            line = line.split("=", 1)[1]
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == PROBE_SCHEMA:
            records.append(value)
    return records


def extract_requested_generation(log_text: str) -> int:
    values = {int(match) for match in re.findall(r"\brequested_gen=(\d+)\b", log_text)}
    if len(values) != 1:
        raise ValueError(f"expected one requested_gen value, observed {sorted(values)}")
    return values.pop()


def _policy_result(
    record: dict[str, Any], rules: dict[str, Any], require_resident: bool = False
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_METRICS if name not in record]
    if missing:
        raise ValueError(f"policy {record.get('policy')!r} is missing metrics: {missing}")
    payload_bytes = int(record["prefix_payload_bytes"])
    if payload_bytes <= 0:
        raise ValueError(f"policy {record.get('policy')!r} has non-positive payload bytes")
    failures = semantic_failures(str(record["candidate_text"]), rules)
    if not bool(record["candidate_ended_with_eos"]):
        failures.append("candidate did not reach EOS")
    result = {
        "meaning_preserved": not failures,
        "semantic_failures": failures,
        "top1_matches": int(record["retire_order_top1_matches"]),
        "top1_count": int(record["retire_order_top1_count"]),
        "ranked_top2_matches": int(record["teacher_top2_ranked_matches"]),
        "ranked_top2_count": int(record["teacher_top2_ranked_count"]),
        "top2_overlap": int(record["teacher_top2_set_overlap"]),
        "top2_overlap_count": int(record["teacher_top2_set_overlap_count"]),
        "exact_top1_covered": int(record["teacher_exact_top1_covered"]),
        "exact_top1_covered_count": int(record["teacher_top2_steps"]),
        "ecs_mean": float(record["teacher_token_ecs_mean"]),
        "raw_bytes": int(record["prefix_raw_bytes"]),
        "payload_bytes": payload_bytes,
        "density": int(record["prefix_raw_bytes"]) / payload_bytes,
        "candidate_generated_tokens": len(record["candidate_ids"]),
        "candidate_ended_with_eos": bool(record["candidate_ended_with_eos"]),
        "candidate_text": str(record["candidate_text"]),
    }
    if require_resident:
        proof_fields = (
            "execution_mode",
            "full_attention_layers",
            "resident_layers",
            "resident_f32_owner_layers",
            "resident_cache_consistent",
        )
        missing_proof = [name for name in proof_fields if name not in record]
        if missing_proof:
            raise ValueError(
                f"policy {record.get('policy')!r} is missing resident GPU proof: "
                f"{missing_proof}"
            )
        raw_f32_owners = record["resident_f32_owner_layers"]
        raw_full_attention_layers = record["full_attention_layers"]
        raw_resident_layers = record["resident_layers"]
        if (
            type(raw_full_attention_layers) is not int
            or type(raw_resident_layers) is not int
            or not isinstance(raw_f32_owners, list)
            or any(type(layer) is not int for layer in raw_f32_owners)
        ):
            raise ValueError(
                f"policy {record.get('policy')!r} has invalid resident GPU proof"
            )
        f32_owners = list(raw_f32_owners)
        full_attention_layers = raw_full_attention_layers
        resident_layers = raw_resident_layers
        if (
            record["execution_mode"] != "resident_gpu"
            or full_attention_layers <= 0
            or resident_layers != full_attention_layers
            or f32_owners
            or record["resident_cache_consistent"] is not True
        ):
            raise ValueError(
                f"policy {record.get('policy')!r} has invalid resident GPU proof"
            )
        result.update(
            {
                "execution_mode": "resident_gpu",
                "full_attention_layers": full_attention_layers,
                "resident_layers": resident_layers,
                "resident_f32_owner_layers": f32_owners,
                "resident_cache_consistent": True,
            }
        )
    return result


def evaluate_case(
    case: dict[str, Any],
    records: list[dict[str, Any]],
    require_resident: bool = False,
) -> dict[str, Any]:
    name = str(case["name"])
    if not records:
        return {"name": name, "status": "invalid_probe", "failures": ["no probe records"]}

    exact_text = str(records[0].get("exact_text", ""))
    exact_ids = list(records[0].get("exact_ids", []))
    baseline_failures = semantic_failures(exact_text, dict(case["rules"]))
    if not bool(records[0].get("exact_ended_with_eos", False)):
        baseline_failures.append("exact baseline did not reach EOS")
    min_exact_tokens = int(case.get("min_exact_tokens", 0))
    if len(exact_ids) < min_exact_tokens:
        baseline_failures.append(
            f"min_exact_tokens expected {min_exact_tokens}, observed {len(exact_ids)}"
        )
    for record in records[1:]:
        if (
            record.get("exact_text") != exact_text
            or record.get("exact_ids") != exact_ids
            or record.get("exact_ended_with_eos") is not True
        ):
            baseline_failures.append("exact oracle changed between policies")
            break
    if baseline_failures:
        return {
            "name": name,
            "status": "invalid_baseline",
            "failures": baseline_failures,
            "exact_text": exact_text,
            "observed_generated_tokens": len(exact_ids),
        }

    policy_results: dict[str, Any] = {}
    for record in records:
        policy = str(record.get("policy", ""))
        if not policy:
            raise ValueError(f"case {name!r} contains a policy without a label")
        if policy in policy_results:
            raise ValueError(f"case {name!r} contains duplicate policy {policy!r}")
        policy_results[policy] = _policy_result(
            record, dict(case["rules"]), require_resident=require_resident
        )
    return {
        "name": name,
        "status": "valid",
        "exact_text": exact_text,
        "observed_generated_tokens": len(exact_ids),
        "policy_results": policy_results,
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, dict[str, Any]] = {}
    valid_cases = [result for result in results if result["status"] == "valid"]
    for result in valid_cases:
        for policy, row in result["policy_results"].items():
            target = aggregate.setdefault(
                policy,
                {
                    "meaning_passed": 0,
                    "meaning_cases": 0,
                    "top1_matches": 0,
                    "top1_count": 0,
                    "ranked_top2_matches": 0,
                    "ranked_top2_count": 0,
                    "top2_overlap": 0,
                    "top2_overlap_count": 0,
                    "exact_top1_covered": 0,
                    "exact_top1_covered_count": 0,
                    "ecs_weighted_sum": 0.0,
                    "raw_bytes": 0,
                    "payload_bytes": 0,
                },
            )
            target["meaning_cases"] += 1
            target["meaning_passed"] += int(row["meaning_preserved"])
            for field in (
                "top1_matches",
                "top1_count",
                "ranked_top2_matches",
                "ranked_top2_count",
                "top2_overlap",
                "top2_overlap_count",
                "exact_top1_covered",
                "exact_top1_covered_count",
                "raw_bytes",
                "payload_bytes",
            ):
                target[field] += int(row[field])
            target["ecs_weighted_sum"] += float(row["ecs_mean"]) * int(row["top1_count"])

    finalized: dict[str, Any] = {}
    for policy, row in aggregate.items():
        top1_count = row.pop("top1_count")
        payload_bytes = row["payload_bytes"]
        ecs_weighted_sum = row.pop("ecs_weighted_sum")
        finalized[policy] = {
            **row,
            "top1_count": top1_count,
            "ecs_weighted_mean": ecs_weighted_sum / top1_count if top1_count else None,
            "density": row["raw_bytes"] / payload_bytes if payload_bytes else None,
        }
    return finalized


def load_cases(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    names: set[str] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{lineno}: case must be an object")
        name = str(value.get("name", ""))
        if not re.fullmatch(r"[a-z0-9_-]+", name):
            raise ValueError(f"{path}:{lineno}: invalid case name {name!r}")
        if name in names:
            raise ValueError(f"{path}:{lineno}: duplicate case name {name!r}")
        if not isinstance(value.get("prompt"), str) or not value["prompt"].strip():
            raise ValueError(f"{path}:{lineno}: prompt must be non-empty")
        if not isinstance(value.get("rules"), dict):
            raise ValueError(f"{path}:{lineno}: rules must be an object")
        semantic_failures("qualification text", value["rules"])
        names.add(name)
        cases.append(value)
    if limit > 0:
        cases = cases[:limit]
    if not cases:
        raise ValueError(f"{path}: no cases selected")
    return cases


def select_cases(cases: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    if not names:
        return cases
    requested = set(names)
    known = {str(case["name"]) for case in cases}
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown cases: {sorted(unknown)}")
    return [case for case in cases if str(case["name"]) in requested]


def run_to_file(cmd: list[str], path: Path, timeout: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        try:
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout,
            )
            return completed.returncode
        except subprocess.TimeoutExpired:
            output.write(f"\n[TIMEOUT] corpus case exceeded {timeout}s\n")
            return 124


def build_probe_command(
    *,
    run_safe: Path,
    binary: Path,
    timeout: int,
    max_mem_mb: int,
    generation: int,
    retire_chunk: int,
    model: Path | None,
    policy_maps: tuple[str, ...],
    prompt: str,
    resident: bool,
) -> list[str]:
    command = [
        str(run_safe),
        str(binary),
        str(timeout),
        str(max_mem_mb),
        f"--gen={generation}",
    ]
    if resident:
        command.extend(f"--resident-map={policy_map}" for policy_map in policy_maps)
    else:
        command.extend((f"--retire-chunk={retire_chunk}", "--precisions=4"))
        command.extend(f"--adaptive-map={policy_map}" for policy_map in policy_maps)
    if model is not None:
        command.append(f"--model={model}")
    command.append(prompt)
    return command


def print_aggregate(aggregate: dict[str, Any]) -> None:
    print("QBIT_CORPUS_VECTOR", flush=True)
    for policy, row in aggregate.items():
        print(
            "  "
            f"policy={policy!r} "
            f"meaning={row['meaning_passed']}/{row['meaning_cases']} "
            f"top1={row['top1_matches']}/{row['top1_count']} "
            f"ranked_top2={row['ranked_top2_matches']}/{row['ranked_top2_count']} "
            f"top2_overlap={row['top2_overlap']}/{row['top2_overlap_count']} "
            f"exact_top1_covered={row['exact_top1_covered']}/{row['exact_top1_covered_count']} "
            f"ecs={row['ecs_weighted_mean']:.6f} "
            f"density={row['density']:.4f}x",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-safe", type=Path, default=DEFAULT_RUN_SAFE)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--max-mem-mb", type=int, default=28672)
    parser.add_argument("--gen", type=int, default=128)
    parser.add_argument("--retire-chunk", type=int, default=32)
    parser.add_argument("--case", action="append", default=[], help="Run one named case; may be repeated")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--resident",
        action="store_true",
        help="Require the real resident GPU KV probe and its ownership proof",
    )
    parser.add_argument(
        "--policy-map",
        action="append",
        default=[],
        help="Adaptive map; diagnostic and resident modes have separate defaults",
    )
    args = parser.parse_args()

    if args.timeout <= 0 or args.max_mem_mb <= 0 or args.gen < 2 or args.retire_chunk <= 0:
        parser.error("timeout/max-mem/gen/retire-chunk must be positive and gen must be at least 2")
    cases = select_cases(load_cases(args.cases), args.case)
    if args.limit > 0:
        cases = cases[: args.limit]
    policy_maps = tuple(args.policy_map) if args.policy_map else (
        DEFAULT_RESIDENT_MAPS if args.resident else DEFAULT_POLICY_MAPS
    )
    if args.resident:
        expected_policies = {f"resident[{value}]" for value in policy_maps}
        binary = args.binary or DEFAULT_RESIDENT_BINARY
        out_dir = args.out_dir or DEFAULT_RESIDENT_OUT_DIR
    else:
        expected_policies = {"p4", *(f"adaptive[{value}]" for value in policy_maps)}
        binary = args.binary or DEFAULT_BINARY
        out_dir = args.out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    infrastructure_failed = False
    for index, case in enumerate(cases, 1):
        name = str(case["name"])
        log_path = out_dir / f"{name}.log"
        print(f"QBIT_CORPUS_CASE {index}/{len(cases)} name={name} phase=start", flush=True)
        if not args.score_only:
            cmd = build_probe_command(
                run_safe=args.run_safe,
                binary=binary,
                timeout=args.timeout,
                max_mem_mb=args.max_mem_mb,
                generation=int(case.get("gen", args.gen)),
                retire_chunk=int(case.get("retire_chunk", args.retire_chunk)),
                model=args.model,
                policy_maps=policy_maps,
                prompt=str(case["prompt"]),
                resident=args.resident,
            )
            returncode = run_to_file(cmd, log_path, args.timeout + 90)
            if returncode != 0:
                infrastructure_failed = True
                result = {
                    "name": name,
                    "status": "runner_failed",
                    "returncode": returncode,
                    "log": str(log_path),
                }
                results.append(result)
                print(
                    f"QBIT_CORPUS_CASE {index}/{len(cases)} name={name} "
                    f"phase=done status=runner_failed returncode={returncode}",
                    flush=True,
                )
                continue
        if not log_path.is_file():
            infrastructure_failed = True
            results.append({"name": name, "status": "missing_log", "log": str(log_path)})
            print(
                f"QBIT_CORPUS_CASE {index}/{len(cases)} name={name} phase=done status=missing_log",
                flush=True,
            )
            continue

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        expected_generation = int(case.get("gen", args.gen))
        try:
            observed_generation = extract_requested_generation(log_text)
        except ValueError as exc:
            infrastructure_failed = True
            results.append(
                {
                    "name": name,
                    "status": "generation_unqualified",
                    "failure": str(exc),
                    "log": str(log_path),
                }
            )
            print(
                f"QBIT_CORPUS_CASE {index}/{len(cases)} name={name} "
                "phase=done status=generation_unqualified",
                flush=True,
            )
            continue
        if observed_generation != expected_generation:
            infrastructure_failed = True
            results.append(
                {
                    "name": name,
                    "status": "generation_mismatch",
                    "expected": expected_generation,
                    "observed": observed_generation,
                    "log": str(log_path),
                }
            )
            print(
                f"QBIT_CORPUS_CASE {index}/{len(cases)} name={name} "
                f"phase=done status=generation_mismatch expected={expected_generation} "
                f"observed={observed_generation}",
                flush=True,
            )
            continue
        records = extract_probe_records(log_text)
        observed_policies = {str(record.get("policy", "")) for record in records}
        if observed_policies != expected_policies:
            infrastructure_failed = True
            results.append(
                {
                    "name": name,
                    "status": "policy_mismatch",
                    "expected": sorted(expected_policies),
                    "observed": sorted(observed_policies),
                    "log": str(log_path),
                }
            )
            print(
                f"QBIT_CORPUS_CASE {index}/{len(cases)} name={name} "
                "phase=done status=policy_mismatch",
                flush=True,
            )
            continue
        result = evaluate_case(case, records, require_resident=args.resident)
        if result["status"] != "valid":
            infrastructure_failed = True
        results.append(result)
        if result["status"] == "valid":
            meanings = ",".join(
                f"{policy}:{'pass' if row['meaning_preserved'] else 'fail'}"
                for policy, row in result["policy_results"].items()
            )
            print(
                f"QBIT_CORPUS_CASE {index}/{len(cases)} name={name} phase=done "
                f"status=valid generated={result['observed_generated_tokens']} meaning={meanings}",
                flush=True,
            )
        else:
            print(
                f"QBIT_CORPUS_CASE {index}/{len(cases)} name={name} "
                f"phase=done status={result['status']}",
                flush=True,
            )

    aggregate = aggregate_results(results)
    report = {
        "schema": "qwen-qbit-quality-corpus-v1",
        "cases_file": str(args.cases),
        "default_generation_limit": args.gen,
        "execution_mode": "resident_gpu" if args.resident else "diagnostic_roundtrip",
        "case_generation_limits": {
            str(case["name"]): int(case.get("gen", args.gen)) for case in cases
        },
        "retire_chunk": None if args.resident else args.retire_chunk,
        "results": results,
        "aggregate": aggregate,
    }
    report_path = out_dir / "report.json"
    temporary_path = report_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, report_path)
    print_aggregate(aggregate)
    print(f"QBIT_CORPUS_REPORT path={report_path}", flush=True)
    return 2 if infrastructure_failed else 0


if __name__ == "__main__":
    sys.exit(main())
