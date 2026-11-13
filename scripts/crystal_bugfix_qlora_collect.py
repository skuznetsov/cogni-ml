#!/usr/bin/env python3
"""Collect Crystal repair histories from generated answers on the training split."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

import crystal_bugfix_fragments as fragments
import crystal_bugfix_qlora_eval as evaluation


REPAIR_STAGES = {"compile_failed": "compile", "tests_failed": "tests"}
RUN_INTERRUPTED = {"infrastructure", "timeout", "generation_failure"}


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def replay_answer(case, answer, directory, runner, timeout):
    with tempfile.TemporaryDirectory(prefix="cogni-qlora-collect-") as scratch:
        replay = Path(scratch) / "replay"
        formatted, _ = evaluation.format_replacement(case, answer, replay)
        evaluation.atomic_json(directory / "replacement-format.json", formatted)
        if formatted["status"] != "passed":
            result = {
                "overall": "infrastructure" if formatted["status"] == "infrastructure"
                else "replacement_format_failed",
                "diagnostic": formatted["diagnostic"],
            }
        else:
            logs = directory / "runner-logs"
            logs.mkdir()
            command = [str(runner), str(replay), str(logs), case.get("backend", "cpu")]
            code, output, timed_out = evaluation.run_bounded(
                command, cwd=evaluation.ROOT, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                timeout=timeout,
            )
            evaluation.write_text(logs / "runner.log", output)
            result = evaluation.result_for_runner(case, code, timed_out, logs)
    evaluation.atomic_json(directory / "result.json", result)
    evaluation.write_text(directory / "diagnostic.txt", result.get("diagnostic", ""))
    return result


def collect_case(case, directory, generator, runner, args):
    case_id = case["id"]
    prompt, target = fragments.model_record(case)["messages"]
    initial = directory / "initial"
    request = {"id": f"{case_id}-initial", "messages": [prompt]}
    evaluation.atomic_json(initial / "request.json", request)
    report = {"id": case_id, "initial_result": "generation_failure"}
    try:
        raw, response = generator.request(request)
    except evaluation.GeneratorFailure as error:
        partial = getattr(error, "partial", b"") or generator.last_raw
        if partial:
            (initial / "response.partial").write_bytes(partial)
        evaluation.write_text(initial / "diagnostic.txt", str(error))
        return report, None

    (initial / "response.ndjson").write_bytes(raw + b"\n")
    answer = response["answer"]
    evaluation.write_text(initial / "answer.txt", answer)
    result = replay_answer(case, answer, initial, runner, args.runner_timeout)
    report["initial_result"] = result["overall"]
    stage = REPAIR_STAGES.get(result["overall"])
    if stage is None or args.stage not in ("all", stage):
        return report, None

    gold = directory / "gold"
    evaluation.write_text(gold / "answer.txt", target["content"])
    feedback = evaluation.repair_feedback(stage, result["diagnostic"])
    evaluation.write_text(gold / "feedback.txt", feedback)
    fixed = replay_answer(case, target["content"], gold, runner, args.runner_timeout)
    report["gold_result"] = fixed["overall"]
    if fixed["overall"] != "pass":
        return report, None
    return report, {
        "id": case_id, "family": case["family"], "source": case["source"], "split": "train",
        "stage": stage, "messages": [
            prompt, {"role": "assistant", "content": answer},
            {"role": "user", "content": feedback}, target,
        ],
    }


def run(args):
    binary = Path(evaluation.resolve_program(str(args.binary)))
    model = args.model.expanduser().resolve(strict=True)
    adapter = Path(args.adapter).expanduser().resolve(strict=True) if args.adapter != "-" else None
    runner = args.runner.expanduser().resolve(strict=True)
    if not model.is_file() or (adapter and not adapter.is_file()) or not os.access(runner, os.X_OK):
        raise ValueError("Need a model file, an adapter file or '-', and an executable replay runner")

    cases, _ = fragments.load_cases()
    train = {case["id"]: case for case in cases if case["split"] == "train"}
    selected = set(args.case_ids) if args.case_ids else set(train)
    unknown = selected - train.keys()
    if unknown:
        raise ValueError(f"Requested IDs are outside the training split: {', '.join(sorted(unknown))}")

    output = evaluation.create_output(args.output)
    command = [str(binary), "generate", str(model), str(adapter) if adapter else "-", "768", "4"]
    evaluation.atomic_json(output / "metadata.json", {
        "generator_command": command, "model_sha256": sha256(model),
        "adapter_sha256": sha256(adapter) if adapter else None, "generator_sha256": sha256(binary),
        "collector_sha256": sha256(Path(__file__)), "replay_runner": str(runner),
        "runner_sha256": sha256(runner), "stage_filter": args.stage,
        "train_case_ids": [case_id for case_id in train if case_id in selected],
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                 cwd=evaluation.ROOT, text=True).strip(),
    })
    cwd = output / "generator-cwd"
    cwd.mkdir()
    generator = evaluation.NDJSONGenerator(
        command, cwd, output / "generator.stderr.log", args.generation_timeout,
    )
    reports, candidates = [], []
    interrupted = False
    try:
        for case_id, case in train.items():
            if case_id not in selected:
                continue
            report, candidate = collect_case(case, output / "cases" / case_id, generator, runner, args)
            reports.append(report)
            if candidate is not None:
                candidates.append(candidate)
            print(f"{case_id}: {report['initial_result']} / {report.get('gold_result', 'skipped')}",
                  flush=True)
            interrupted = any(report.get(key) in RUN_INTERRUPTED
                              for key in ("initial_result", "gold_result"))
            if interrupted:
                break
    finally:
        generator_exit = generator.close()

    evaluation.write_text(output / "candidates.jsonl", "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in candidates
    ))
    summary = {
        "initial": dict(Counter(report["initial_result"] for report in reports)),
        "gold_replays": dict(Counter(report["gold_result"] for report in reports if "gold_result" in report)),
        "verified_candidates": len(candidates), "generator_exit_code": generator_exit,
    }
    evaluation.atomic_json(output / "report.json", {"cases": reports, "summary": summary})
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 2 if interrupted or generator_exit else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", required=True, help="adapter path or '-' for stock model weights")
    parser.add_argument("--stage", choices=("all", "compile", "tests"), default="all")
    parser.add_argument("--runner", type=Path, default=evaluation.DEFAULT_RUNNER)
    parser.add_argument("--output", type=Path, required=True, help="new directory under build/qlora")
    parser.add_argument("--case", dest="case_ids", action="append", help="select a train ID, repeatable")
    parser.add_argument("--generation-timeout", type=float, default=300)
    parser.add_argument("--runner-timeout", type=float, default=480)
    args = parser.parse_args()
    if min(args.generation_timeout, args.runner_timeout) <= 0:
        parser.error("timeouts must be positive")
    try:
        return run(args)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        print(f"collection failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
