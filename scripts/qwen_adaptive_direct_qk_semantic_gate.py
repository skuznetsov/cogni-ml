#!/usr/bin/env python3
"""Score the non-bit-exact adaptive direct-QK route on coding semantics.

This gate is deliberately separate from the probe's numerical gate. It does
not reinterpret a logit delta above 1e-4 as numerical equivalence. Instead it
requires identical self-fed token trajectories and text, exact ranked top-2
agreement at every observed step, ECS 1.0, passing external Crystal specs, and
coverage of both low and high long-context bands.

Security notice: external scoring executes generated Crystal source in copied
fixture directories. It is not a sandbox. Inspect new fixtures and generated
source before invoking the CLI on model output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


PROBE_SCHEMA = "qwen-adaptive-t8-decode-ab-v12"
MANIFEST_SCHEMA = "qwen-adaptive-direct-qk-semantic-suite-v1"
REPORT_SCHEMA = "qwen-adaptive-direct-qk-semantic-report-v1"
EXPECTED_COMPARISON = "p4_stage1_bundle"
EXPECTED_DEVICE = "Apple M2 Max"
EXPECTED_MODEL = "Qwen3.8-27B-Q4_K_M.gguf"
EXPECTED_RESIDENT_MAP = "p4;27=bf16,43=bf16,47=bf16,51=bf16"
MIN_CASES = 3
MIN_STEPS_PER_CASE = 32
MIN_TOTAL_STEPS = 96
MIN_REQUESTED_SAMPLES = 256
LOW_CONTEXT_MIN = 7_000
LOW_CONTEXT_MAX = 9_000
HIGH_CONTEXT_MIN = 14_336
PINNED_FIXTURES = {
    "lower_bound": "db979501712084f3ee1b2277f934ffca9ffc782331daa827a731369136a9d00d",
    "stable_unique": "7f0522b80154275d7fa26c2bc2e793d00eee806bfef4083eac68b5e0dc8e24f2",
    "merge_ranges": "3c473a101ea31698d259e19a5240add85a210563f535b176c2a1cb4a6b03a598",
}
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "spec" / "fixtures" / "qwen_flash_coding"

_NUMERIC_VIOLATION = re.compile(
    r"^(?:prefill_top1_logit_delta|"
    r"(?:prefill|warmup|sample_\d+)_(?:first_logit_delta|second_logit_delta|margin_delta))="
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$"
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _load_coding_score_module() -> Any:
    path = Path(__file__).with_name("qwen_qbit_coding_session_score.py")
    spec = importlib.util.spec_from_file_location("_qwen_qbit_coding_session_score", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CODING_SCORE = _load_coding_score_module()
extract_crystal_source = _CODING_SCORE.extract_crystal_source
run_external_specs = _CODING_SCORE.run_external_specs


def extract_probe_record(log_text: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for raw in log_text.splitlines():
        line = raw.strip()
        if not line.startswith("QBIT_T8_DECODE_JSON="):
            continue
        value = json.loads(line.split("=", 1)[1])
        if isinstance(value, dict) and value.get("schema") == PROBE_SCHEMA:
            records.append(value)
    if len(records) != 1:
        raise ValueError(f"expected exactly one {PROBE_SCHEMA} record, observed {len(records)}")
    return records[0]


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _case_violations(case: dict[str, Any]) -> tuple[list[str], list[str]]:
    task_id = case.get("task_id")
    label = task_id if isinstance(task_id, str) and task_id else "<missing-task>"
    record = case.get("record")
    baseline = case.get("baseline")
    candidate = case.get("candidate")
    structural: list[str] = []
    semantic: list[str] = []

    if not isinstance(task_id, str) or not task_id:
        structural.append(f"{label}: task_id must be a non-empty string")
    if not isinstance(record, dict):
        return structural + [f"{label}: record must be an object"], semantic
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        return structural + [f"{label}: external results must be objects"], semantic

    exact_fields = {
        "schema": PROBE_SCHEMA,
        "comparison": EXPECTED_COMPARISON,
        "state_source": "real_prompt_prefill",
        "semantic_quality_valid": True,
        "quality_top2": True,
        "quality_top2_production_prefill": True,
        "semantic_coding_quality": True,
        "quality_scope": "real_prefix_production_top1_then_free_run_top2",
        "quality_measurement_valid": True,
        "prefill_boundary_mode": "production_top1",
        "semantic_task_scored": False,
        "timing_gate_valid": False,
        "ecs_basis": "output.weight",
        "ecs_interpretation": "static_output_row_cosine_token_proxy",
        "ecs_equal_ids_short_circuit_to_one": True,
        "release_build": True,
        "device": EXPECTED_DEVICE,
        "model": EXPECTED_MODEL,
        "resident_map": EXPECTED_RESIDENT_MAP,
        "effective_splitk": True,
        "route_certificate_kind": "policy_eligibility",
        "baseline_splitk_chunk": 64,
        "candidate_splitk_chunk": 64,
        "baseline_stage2": "forced_on",
        "candidate_stage2": "forced_on",
        "baseline_direct_qk": False,
        "candidate_direct_qk": True,
        "baseline_v_contiguous": False,
        "candidate_v_contiguous": True,
        "p4_stage1_admission": "forced_off_vs_forced_on",
        "candidate_p4_t8_owners": 12,
        "candidate_bf16_t8_owners": 4,
        "candidate_p4_direct_qk_owners": 12,
        "candidate_p4_v_contiguous_owners": 12,
        "max_seq": 16_384,
        "prompt_repeats": 1,
        "pooled_scratch": True,
        "prefill_gc_guard": True,
        "quality_logit_tolerance": 0.0001,
    }
    for name, expected in exact_fields.items():
        if record.get(name) != expected:
            structural.append(
                f"{label}: {name} must be {expected!r}, observed {record.get(name)!r}"
            )

    prompt_hash = record.get("prompt_sha256")
    if not isinstance(prompt_hash, str) or _HEX64.fullmatch(prompt_hash) is None:
        structural.append(f"{label}: prompt_sha256 must be 64 lowercase hex characters")

    samples = record.get("samples")
    observed = record.get("observed_samples")
    if type(samples) is not int or samples < MIN_REQUESTED_SAMPLES:
        structural.append(
            f"{label}: samples must be at least {MIN_REQUESTED_SAMPLES}"
        )
    if type(observed) is not int or observed <= 0 or (type(samples) is int and observed > samples):
        structural.append(f"{label}: observed_samples must be in 1..samples")
    if record.get("baseline_eos") is not True or record.get("candidate_eos") is not True:
        structural.append(f"{label}: both trajectories must reach EOS")
    if record.get("aligned_eos") is not True:
        structural.append(f"{label}: trajectories did not reach aligned EOS")
    if record.get("coding_completion") != "aligned_eos":
        structural.append(f"{label}: coding_completion must be aligned_eos")
    baseline_eos_step = record.get("baseline_eos_step")
    candidate_eos_step = record.get("candidate_eos_step")
    if type(baseline_eos_step) is not int or baseline_eos_step < 0:
        structural.append(f"{label}: baseline_eos_step must be non-negative")
    if candidate_eos_step != baseline_eos_step:
        structural.append(f"{label}: candidate EOS step differs from baseline")
    completed = record.get("requested_samples_completed") is True
    termination = record.get("termination_reason")
    if completed:
        if termination != "requested_samples_completed":
            structural.append(f"{label}: completed samples have an invalid termination reason")
    elif type(observed) is int and termination != f"eos_before_sample_{observed}":
        structural.append(f"{label}: early EOS termination does not match observed_samples")

    prompt_tokens = record.get("prompt_tokens")
    max_seq = record.get("max_seq")
    if type(prompt_tokens) is not int or prompt_tokens <= 0:
        structural.append(f"{label}: prompt_tokens must be positive")
    if type(max_seq) is not int or max_seq <= 0:
        structural.append(f"{label}: max_seq must be positive")
    if (
        type(prompt_tokens) is int
        and type(max_seq) is int
        and type(samples) is int
        and prompt_tokens + samples + 2 > max_seq
    ):
        structural.append(f"{label}: prompt plus requested decode exceeds max_seq")
    if record.get("seeded_prefix_tokens") != prompt_tokens:
        structural.append(f"{label}: seeded prefix must equal the real prompt length")

    step_count = record.get("quality_step_count")
    quality_steps = record.get("quality_steps")
    if type(step_count) is not int or step_count < MIN_STEPS_PER_CASE:
        structural.append(
            f"{label}: quality_step_count must be at least {MIN_STEPS_PER_CASE}"
        )
        step_count = 0
    if not isinstance(quality_steps, list) or len(quality_steps) != step_count:
        structural.append(f"{label}: quality_steps must contain quality_step_count entries")
        quality_steps = []
    if type(observed) is int and step_count != observed + 1:
        structural.append(
            f"{label}: production-prefill quality_step_count must equal observed_samples + 1"
        )

    ranked_count = record.get("quality_ranked_top2_count")
    if ranked_count != step_count * 2:
        structural.append(f"{label}: ranked top-2 denominator is inconsistent")
    if record.get("quality_ranked_top2_matches") != ranked_count:
        semantic.append(f"{label}: ranked top-2 is not exact")
    if record.get("quality_min_set_overlap") != 2:
        semantic.append(f"{label}: top-2 set overlap fell below two")
    if record.get("quality_exact_top1_covered") != step_count:
        semantic.append(f"{label}: exact top-1 coverage is incomplete")
    if record.get("quality_exact_top2_covered") != step_count:
        semantic.append(f"{label}: exact top-2 coverage is incomplete")
    if record.get("quality_aligned_step_count") != step_count:
        semantic.append(f"{label}: not every quality step has paired logits")
    if record.get("quality_min_token_ecs") != 1.0:
        semantic.append(f"{label}: minimum token ECS is not 1.0")

    for index, step in enumerate(quality_steps):
        if not isinstance(step, dict):
            structural.append(f"{label}: quality step {index} is not an object")
            continue
        expected_phase = "warmup" if index == 0 else "sample"
        expected_index = -1 if index == 0 else index - 1
        if step.get("phase") != expected_phase or step.get("index") != expected_index:
            structural.append(f"{label}: quality step {index} has an invalid identity")
        for field in (
            "baseline_top1_logit",
            "baseline_top2_logit",
            "baseline_margin",
            "candidate_top1_logit",
            "candidate_top2_logit",
            "candidate_margin",
            "first_logit_delta",
            "second_logit_delta",
            "margin_delta",
            "token_ecs",
        ):
            if not _finite_number(step.get(field)):
                structural.append(
                    f"{label}: quality step {index} {field} must be finite"
                )
        if step.get("paired_logits_valid") is not True:
            semantic.append(f"{label}: quality step {index} lacks paired logits")
        if step.get("baseline_top1_id") != step.get("candidate_top1_id"):
            semantic.append(f"{label}: quality step {index} top-1 differs")
        if step.get("baseline_top2_id") != step.get("candidate_top2_id"):
            semantic.append(f"{label}: quality step {index} top-2 differs")
        if step.get("ranked_top2_matches") != 2:
            semantic.append(f"{label}: quality step {index} ranked top-2 is not exact")
        if step.get("top2_set_overlap") != 2:
            semantic.append(f"{label}: quality step {index} top-2 set overlap is not two")
        if step.get("exact_top1_covered") is not True:
            semantic.append(f"{label}: quality step {index} exact top-1 is not covered")
        if step.get("exact_top2_covered") is not True:
            semantic.append(f"{label}: quality step {index} exact top-2 is not covered")
        if step.get("token_ecs") != 1.0:
            semantic.append(f"{label}: quality step {index} token ECS is not 1.0")

    baseline_ids = record.get("baseline_output_ids")
    candidate_ids = record.get("candidate_output_ids")
    if not isinstance(baseline_ids, list) or not isinstance(candidate_ids, list):
        structural.append(f"{label}: output token sequences must be arrays")
    elif baseline_ids != candidate_ids:
        semantic.append(f"{label}: self-fed output token trajectories differ")
    else:
        if type(observed) is int and len(baseline_ids) != observed + 2:
            structural.append(f"{label}: output token count must equal observed_samples + 2")
        if record.get("free_common_prefix") != len(baseline_ids):
            semantic.append(f"{label}: free_common_prefix does not cover the trajectory")
        if type(baseline_eos_step) is int and baseline_eos_step != len(baseline_ids) - 1:
            structural.append(f"{label}: EOS must be the final emitted token")
    if record.get("first_divergence_step") is not None:
        semantic.append(f"{label}: first_divergence_step is present")
    if record.get("baseline_text") != record.get("candidate_text"):
        semantic.append(f"{label}: generated texts differ")

    violations = record.get("quality_violations")
    if not isinstance(violations, list) or not all(isinstance(item, str) for item in violations):
        structural.append(f"{label}: quality_violations must be an array of strings")
    else:
        for item in violations:
            if _NUMERIC_VIOLATION.fullmatch(item) is None:
                semantic.append(f"{label}: non-numeric quality violation: {item}")
        tolerance = record.get("quality_logit_tolerance")
        tolerance_value = float(tolerance) if _finite_number(tolerance) else 0.0001
        numeric_breach = (
            _finite_number(record.get("prefill_logit_delta"))
            and float(record["prefill_logit_delta"]) > tolerance_value
        ) or any(
            isinstance(step, dict)
            and any(
                _finite_number(step.get(field))
                and float(step[field]) > tolerance_value
                for field in ("first_logit_delta", "second_logit_delta", "margin_delta")
            )
            for step in quality_steps
        )
        strict_numeric = not numeric_breach
        if bool(violations) != numeric_breach:
            structural.append(f"{label}: numeric violations do not match measured deltas")
        if record.get("strict_numeric_gate_passed") is not strict_numeric:
            structural.append(f"{label}: strict_numeric_gate_passed is inconsistent")

    if record.get("semantic_trajectory_gate_passed") is not True:
        structural.append(f"{label}: producer semantic trajectory self-check failed")

    for name in (
        "prefill_logit_delta",
        "warm_logit_delta",
        "quality_max_second_logit_delta",
        "quality_max_margin_delta",
    ):
        if not _finite_number(record.get(name)):
            structural.append(f"{label}: {name} must be finite")

    if baseline.get("external_pass") is not True:
        structural.append(f"{label}: baseline external specs failed")
    if candidate.get("external_pass") is not True:
        semantic.append(f"{label}: candidate external specs failed")

    return structural, semantic


def evaluate_suite(cases: list[dict[str, Any]]) -> dict[str, Any]:
    structural: list[str] = []
    semantic: list[str] = []
    if len(cases) < MIN_CASES:
        structural.append(f"suite requires at least {MIN_CASES} cases")

    task_ids = [case.get("task_id") for case in cases]
    valid_task_ids = [item for item in task_ids if isinstance(item, str)]
    if len(set(valid_task_ids)) != len(valid_task_ids):
        structural.append("suite task_ids must be distinct")
    if set(valid_task_ids) != set(PINNED_FIXTURES):
        structural.append("suite must contain exactly the three pinned Crystal fixtures")

    prompt_hashes: list[Any] = []
    prompt_tokens: list[int] = []
    total_steps = 0
    numeric_diagnostics: list[dict[str, Any]] = []
    for case in cases:
        case_structural, case_semantic = _case_violations(case)
        structural.extend(case_structural)
        semantic.extend(case_semantic)
        record = case.get("record")
        if isinstance(record, dict):
            prompt_hashes.append(record.get("prompt_sha256"))
            if type(record.get("prompt_tokens")) is int:
                prompt_tokens.append(record["prompt_tokens"])
            if type(record.get("quality_step_count")) is int:
                total_steps += record["quality_step_count"]
            numeric_diagnostics.append(
                {
                    "task_id": case.get("task_id"),
                    "prefill_logit_delta": record.get("prefill_logit_delta"),
                    "warm_logit_delta": record.get("warm_logit_delta"),
                    "max_second_logit_delta": record.get("quality_max_second_logit_delta"),
                    "max_margin_delta": record.get("quality_max_margin_delta"),
                }
            )

    valid_prompt_hashes = [item for item in prompt_hashes if isinstance(item, str)]
    if len(set(valid_prompt_hashes)) != len(valid_prompt_hashes):
        structural.append("suite prompt_sha256 values must be distinct")
    if total_steps < MIN_TOTAL_STEPS:
        structural.append(f"suite requires at least {MIN_TOTAL_STEPS} quality steps")
    low_covered = any(LOW_CONTEXT_MIN <= tokens <= LOW_CONTEXT_MAX for tokens in prompt_tokens)
    high_covered = any(tokens >= HIGH_CONTEXT_MIN for tokens in prompt_tokens)
    if not low_covered:
        structural.append(
            f"suite lacks a context in {LOW_CONTEXT_MIN}..{LOW_CONTEXT_MAX} tokens"
        )
    if not high_covered:
        structural.append(f"suite lacks a context at or above {HIGH_CONTEXT_MIN} tokens")

    if structural:
        verdict = "invalid_evidence"
    elif semantic:
        verdict = "semantic_regression"
    else:
        verdict = "fixture_semantic_smoke_pass"
    strict_numeric_gate_passed = bool(cases) and all(
        isinstance(case.get("record"), dict)
        and case["record"].get("strict_numeric_gate_passed") is True
        for case in cases
    )
    return {
        "schema": REPORT_SCHEMA,
        "verdict": verdict,
        "claim": "three_pinned_crystal_fixtures_smoke_not_semantic_equivalence",
        "claim_scope": "three pinned Crystal fixtures; not broad semantic equivalence",
        "numerical_equivalence_claimed": False,
        "semantic_fixture_gate_passed": verdict == "fixture_semantic_smoke_pass",
        "strict_numeric_gate_passed": strict_numeric_gate_passed,
        "admission_eligible": False,
        "case_count": len(cases),
        "quality_step_count": total_steps,
        "context_coverage": {"low": low_covered, "high": high_covered},
        "structural_violations": structural,
        "semantic_violations": semantic,
        "violations": structural + semantic,
        "numeric_diagnostics": numeric_diagnostics,
        "cases": cases,
    }


def _resolve(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_manifest(manifest_path: Path, output_dir: Path, timeout: int) -> dict[str, Any]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    entries = manifest.get("cases")
    if not isinstance(entries, list):
        raise ValueError("manifest cases must be an array")
    base = manifest_path.resolve().parent

    prepared: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest case {index} must be an object")
        task_id = entry.get("task_id")
        if not isinstance(task_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", task_id):
            raise ValueError(f"manifest case {index} has an unsafe task_id")
        probe_log = _resolve(base, entry.get("probe_log"), f"{task_id}.probe_log")
        project = _resolve(base, entry.get("project"), f"{task_id}.project")
        hidden_spec = _resolve(base, entry.get("hidden_spec"), f"{task_id}.hidden_spec")
        prompt_file = _resolve(base, entry.get("prompt_file"), f"{task_id}.prompt_file")
        source_value = entry.get("source")
        if source_value != "src/answer.cr":
            raise ValueError(f"{task_id}.source must be src/answer.cr")
        source = Path(source_value)
        expected_project = (FIXTURE_ROOT / task_id).resolve()
        if task_id not in PINNED_FIXTURES or project != expected_project:
            raise ValueError(f"{task_id}.project must be the pinned repository fixture")
        if hidden_spec != expected_project / "check.cr":
            raise ValueError(f"{task_id}.hidden_spec must be the pinned check.cr")
        if _sha256(hidden_spec) != PINNED_FIXTURES[task_id]:
            raise ValueError(f"{task_id}.hidden_spec hash does not match the pinned oracle")
        canonical_prompt = (expected_project / "prompt.txt").read_bytes().rstrip()
        prompt_bytes = prompt_file.read_bytes()
        if not prompt_bytes.rstrip().endswith(canonical_prompt):
            raise ValueError(f"{task_id}.prompt_file must end with the pinned task prompt")

        record = extract_probe_record(probe_log.read_text(encoding="utf-8", errors="replace"))
        if record.get("prompt_sha256") != hashlib.sha256(prompt_bytes).hexdigest():
            raise ValueError(f"{task_id}.prompt_file hash does not match the probe record")
        prepared.append(
            {
                "task_id": task_id,
                "record": record,
                "project": project,
                "hidden_spec": hidden_spec,
                "source": source,
            }
        )

    preflight = evaluate_suite(
        [
            {
                "task_id": item["task_id"],
                "record": item["record"],
                "baseline": {"external_pass": True},
                "candidate": {"external_pass": True},
            }
            for item in prepared
        ]
    )
    if preflight["verdict"] != "fixture_semantic_smoke_pass":
        raise ValueError(
            "pre-execution validation failed: " + "; ".join(preflight["violations"])
        )
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    cases: list[dict[str, Any]] = []
    for item in prepared:
        task_id = item["task_id"]
        record = item["record"]
        task_output = output_dir / task_id
        task_output.mkdir()
        baseline_source = extract_crystal_source(str(record.get("baseline_text", "")))
        candidate_source = extract_crystal_source(str(record.get("candidate_text", "")))
        baseline = run_external_specs(
            project=item["project"],
            hidden_spec=item["hidden_spec"],
            source_path=item["source"],
            generated_source=baseline_source,
            output_dir=task_output / "baseline",
            timeout=timeout,
        )
        candidate = run_external_specs(
            project=item["project"],
            hidden_spec=item["hidden_spec"],
            source_path=item["source"],
            generated_source=candidate_source,
            output_dir=task_output / "candidate",
            timeout=timeout,
        )
        cases.append(
            {
                "task_id": task_id,
                "record": record,
                "baseline": baseline,
                "candidate": candidate,
            }
        )

    report = evaluate_suite(cases)
    report["fixture_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    report["external_execution_sandboxed"] = False
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        report = score_manifest(args.manifest, args.output, args.timeout)
        print(
            "DIRECT_QK_SEMANTIC_GATE "
            f"verdict={report['verdict']} cases={report['case_count']} "
            f"quality_steps={report['quality_step_count']} report={args.output / 'report.json'}"
        )
        return 0 if report["verdict"] == "fixture_semantic_smoke_pass" else 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
