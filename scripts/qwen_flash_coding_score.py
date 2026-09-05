#!/usr/bin/env python3
"""Score Flash coding output with external Crystal specs.

The probe log is JSONL, but also contains ordinary runner noise and diagnostic
JSON events. Exactly one ``config`` and one ``summary`` event are required.
The probe's state/teacher result is retained as a diagnostic; the external
Crystal tests are the quality oracle for this adapter.

Security notice: the reused external-spec runner executes generated Crystal
source inside a copied project. It is not a sandbox. Manually inspect the
generated source and test fixture before invoking this adapter.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QBIT_SCORE_SCRIPT = ROOT / "scripts" / "qwen_qbit_coding_session_score.py"
EXPECTED_FIXTURE = "chat_prompt_file_no_thinking"
_QBIT_SPEC = importlib.util.spec_from_file_location(
    "_qwen_qbit_coding_session_score", QBIT_SCORE_SCRIPT
)
if _QBIT_SPEC is None or _QBIT_SPEC.loader is None:
    raise ImportError(f"cannot load reusable scorer helpers from {QBIT_SCORE_SCRIPT}")
_QBIT_MODULE = importlib.util.module_from_spec(_QBIT_SPEC)
_QBIT_SPEC.loader.exec_module(_QBIT_MODULE)

# Keep source extraction and the external runner single-sourced in the existing
# scorer. In particular, do not create a second runner with different safety
# or project-copy semantics.
extract_crystal_source = _QBIT_MODULE.extract_crystal_source
run_external_specs = _QBIT_MODULE.run_external_specs


DIAGNOSTIC_FIELDS = (
    "passed",
    "state_passed",
    "teacher_passed",
    "free_match",
    "timing_is_diagnostic",
    "top1_matches",
    "top1_count",
    "candidate_free_count",
    "top2_ranked_matches",
    "top2_ranked_count",
    "exact_top1_covered",
    "token_ecs_min",
    "logit_cosine_min",
    "logit_max_abs",
    "baseline_prefix_ms",
    "candidate_prefix_ms",
    "baseline_append_ms",
    "candidate_append_ms",
    "append_ratio",
    "eos_stopping",
    "baseline_eos",
    "candidate_eos",
    "semantic_task_scored",
)


def _json_object_lines(log_text: str) -> list[tuple[int, dict[str, Any]]]:
    """Parse JSON object lines while allowing ordinary probe runner noise."""

    objects: list[tuple[int, dict[str, Any]]] = []
    for line_number, raw in enumerate(log_text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        # The Crystal probe emits plain self-test/status lines in addition to
        # JSONL. JSON object lines are still strict: a truncated object must
        # not silently turn into a missing-record error.
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON record on line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSON record on line {line_number} is not an object")
        objects.append((line_number, value))
    return objects


def _require_bool(record: dict[str, Any], name: str, label: str) -> bool:
    value = record.get(name)
    if type(value) is not bool:
        raise ValueError(f"{label}.{name} must be a boolean")
    return value


def _validate_probe_records(config: dict[str, Any], summary: dict[str, Any]) -> None:
    if config.get("fixture") != EXPECTED_FIXTURE:
        raise ValueError(
            f"config.fixture must be {EXPECTED_FIXTURE!r}, observed {config.get('fixture')!r}"
        )
    _require_bool(config, "control", "config")
    if _require_bool(summary, "eos_stopping", "summary") is not True:
        raise ValueError("summary.eos_stopping must be true")
    missing_eos = [
        name
        for name in ("baseline_eos", "candidate_eos")
        if name not in summary or summary.get(name) is not True
    ]
    if missing_eos:
        raise ValueError(f"summary requires true EOS markers: {missing_eos}")
    for name in ("baseline_text", "candidate_text"):
        if not isinstance(summary.get(name), str):
            raise ValueError(f"summary.{name} must be a string")


def extract_flash_records(log_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the unique config and summary records from a Flash probe log."""

    config_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    for _, record in _json_object_lines(log_text):
        event = record.get("event")
        if event == "config":
            config_records.append(record)
        elif event == "summary":
            summary_records.append(record)
    if len(config_records) != 1:
        raise ValueError(
            f"expected exactly one config record, observed {len(config_records)}"
        )
    if len(summary_records) != 1:
        raise ValueError(
            f"expected exactly one summary record, observed {len(summary_records)}"
        )
    config, summary = config_records[0], summary_records[0]
    _validate_probe_records(config, summary)
    return config, summary


def _diagnostics(summary: dict[str, Any]) -> dict[str, Any]:
    return {name: summary[name] for name in DIAGNOSTIC_FIELDS if name in summary}


def classify_verdict(
    *, baseline: dict[str, Any], candidate: dict[str, Any], control: bool = False
) -> str:
    """Classify external-test outcomes without promoting probe diagnostics."""

    if baseline.get("external_pass") is not True:
        return "invalid_flash_baseline"
    if candidate.get("external_pass") is True:
        return "flash_control_pass" if control else "flash_pass"
    return "flash_control_failure" if control else "flash_regression"


def _run_comparison(
    *,
    project: Path,
    hidden_spec: Path,
    source_path: Path,
    summary: dict[str, Any],
    output_dir: Path,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_source = extract_crystal_source(summary["baseline_text"])
    candidate_source = extract_crystal_source(summary["candidate_text"])
    baseline = run_external_specs(
        project=project,
        hidden_spec=hidden_spec,
        source_path=source_path,
        generated_source=baseline_source,
        output_dir=output_dir / "baseline",
        timeout=timeout,
    )
    candidate = run_external_specs(
        project=project,
        hidden_spec=hidden_spec,
        source_path=source_path,
        generated_source=candidate_source,
        output_dir=output_dir / "candidate",
        timeout=timeout,
    )
    return baseline, candidate


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
        config, summary = extract_flash_records(
            args.probe_log.read_text(encoding="utf-8", errors="replace")
        )
        args.output.mkdir(parents=True)
        baseline, candidate = _run_comparison(
            project=args.project,
            hidden_spec=args.hidden_spec,
            source_path=args.source,
            summary=summary,
            output_dir=args.output,
            timeout=args.timeout,
        )
        control = config["control"] is True
        verdict = classify_verdict(
            baseline=baseline, candidate=candidate, control=control
        )
        report = {
            "schema": "qwen-flash-coding-score-v1",
            "verdict": verdict,
            "comparison_mode": "control" if control else "flash_vs_baseline",
            "probe": {
                "fixture": config["fixture"],
                "eos_stopping": summary["eos_stopping"],
                "control": control,
                "baseline_eos": summary["baseline_eos"],
                "candidate_eos": summary["candidate_eos"],
            },
            "diagnostics": _diagnostics(summary),
            "baseline": baseline,
            "candidate": candidate,
        }
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            "FLASH_CODING "
            f"verdict={verdict} baseline_pass={baseline['external_pass']} "
            f"candidate_pass={candidate['external_pass']} control={control} report={report_path}"
        )
        # A control run is useful qualification evidence but is not a Flash vs
        # baseline comparison, so it remains non-promoting at the CLI boundary.
        return 0 if verdict == "flash_pass" else 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
