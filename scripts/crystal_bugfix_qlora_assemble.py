#!/usr/bin/env python3
"""Append replay-verified repair histories to the expanded Crystal SFT split."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True

import crystal_bugfix_qlora_eval as evaluation


FAILURE = {"compile": "compile_failed", "tests": "tests_failed"}
# Matches Qwen35QLoRA::Block::MAX_TRAINING_SEQUENCE_LENGTH.
MAX_TRAINING_TOKENS = 2400


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_rows(paths, train):
    seen = set()
    for path in paths:
        for candidate in read_jsonl(path):
            case_id = candidate["id"]
            case = train.get(case_id)
            stage = candidate["stage"]
            if (case is None or candidate["split"] != "train" or
                    candidate["family"] != case["family"] or candidate["source"] != case["source"]):
                raise ValueError(f"{case_id}: candidate is outside the expanded training split")
            source = path.parent / "cases" / case_id
            initial, gold = source / "initial", source / "gold"
            failure = evaluation.result_for_runner(case, 0, False, initial / "runner-logs")
            success = evaluation.result_for_runner(case, 0, False, gold / "runner-logs")
            if (failure["overall"] != FAILURE[stage] or not failure["diagnostic"] or
                    success["overall"] != "pass"):
                raise ValueError(f"{case_id}: missing the expected failed and passing replays")

            prompt, target = evaluation.fragments.model_record(case)["messages"]
            request = json.loads((initial / "request.json").read_text(encoding="utf-8"))
            response = json.loads((initial / "response.ndjson").read_text(encoding="utf-8"))
            old = (initial / "answer.txt").read_text(encoding="utf-8")
            fixed = (gold / "answer.txt").read_text(encoding="utf-8")
            messages = [
                prompt,
                {"role": "assistant", "content": old},
                {"role": "user", "content": evaluation.repair_feedback(stage, failure["diagnostic"])},
                target,
            ]
            if (request["messages"] != [prompt] or request["id"] != response["id"] or
                    old != response["answer"] or old == fixed or fixed != target["content"] or
                    messages != candidate["messages"]):
                raise ValueError(f"{case_id}: candidate does not match the request, answer and replay")
            identity = (case_id, old)
            if identity not in seen:
                seen.add(identity)
                yield candidate, path


def run(args):
    cases, _ = evaluation.fragments.load_cases()
    train = {case["id"]: case for case in cases if case["split"] == "train"}
    train_file, valid_file = args.expanded / "train.jsonl", args.expanded / "valid.jsonl"
    originals = [evaluation.fragments.model_record(case) for case in train.values()]
    validation = [evaluation.fragments.model_record(case) for case in cases if case["split"] == "valid"]
    if read_jsonl(train_file) != originals or read_jsonl(valid_file) != validation:
        raise ValueError("Expanded export does not match the training and validation manifest")

    rows = list(verified_rows(args.candidates, train))
    probe = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row, _ in rows)
    token_counts = subprocess.run(
        [evaluation.resolve_program(str(args.token_counter)), str(args.model)],
        input=probe, text=True, encoding="utf-8", stdout=subprocess.PIPE, check=True,
    )
    counts = [json.loads(line) for line in token_counts.stdout.splitlines()]
    accepted, rejected, examples = [], [], []
    for index, ((row, source), count) in enumerate(zip(rows, counts, strict=True), 1):
        if (count["row"] != index or count["id"] != row["id"] or
                count["input_tokens"] != count["prefix_tokens"] + count["target_tokens"] - 1):
            raise ValueError("Token counter output does not match the candidate sequence")
        info = {
            "id": row["id"], "family": row["family"], "stage": row["stage"],
            "candidate_file": str(source.resolve()), "input_tokens": count["input_tokens"],
            "prefix_tokens": count["prefix_tokens"], "target_tokens": count["target_tokens"],
        }
        if count["input_tokens"] <= MAX_TRAINING_TOKENS:
            accepted.append(info)
            examples.append({"messages": row["messages"]})
        else:
            rejected.append(info)
    stage_counts = Counter(row["stage"] for row in accepted)
    if stage_counts.keys() != FAILURE.keys():
        raise ValueError("Need both compiler and test diagnostics within the training token limit")

    output = evaluation.create_output(args.output)
    train_output, valid_output = output / "train.jsonl", output / "valid.jsonl"
    shutil.copyfile(train_file, train_output)
    with train_output.open("r+b") as stream:
        stream.seek(-1, 2)
        if stream.read(1) != b"\n":
            stream.write(b"\n")
        for row in examples:
            stream.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    shutil.copyfile(valid_file, valid_output)
    summary = {
        "one_shot": len(originals), "repair": len(examples), "repair_stages": dict(stage_counts),
        "model_sha256": sha256(args.model), "candidate_sha256": {
            str(path.resolve()): sha256(path) for path in args.candidates
        },
        "train_sha256": sha256(train_output), "valid_sha256": sha256(valid_output),
        "accepted": accepted, "too_long": rejected,
    }
    evaluation.atomic_json(output / "assembly.json", summary)
    print(f"{output}: {len(originals)} one-shot + {len(examples)} repairs ({dict(stage_counts)})")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", type=Path, required=True, help="exported sft-expanded directory")
    parser.add_argument("--model", type=Path, required=True, help="training GGUF for token counting")
    parser.add_argument("--token-counter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory under build/qlora")
    parser.add_argument("candidates", nargs="+", type=Path, help="collector candidates.jsonl files")
    try:
        return run(parser.parse_args())
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        print(f"assembly failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
