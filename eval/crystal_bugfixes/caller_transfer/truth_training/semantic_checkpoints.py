"""Reconstruct the first 10 and 25 updates of both frozen full-SFT runs."""

import argparse
import json
from pathlib import Path
import subprocess
from statistics import mean

import fit
import semantic_diversity as semantic


OUT = semantic.OUT / "checkpoint-sweep-v1"
STEPS = (10, 25)
ROOTS = {"original": semantic.REFERENCE, "diverse": semantic.OUT}
ARCHIVE_PINS = {
    "protocol.json": "12e0901a6846b3c009f584db3161ea0b756cfe89965b78b57c4fb970b4ec71bc",
    "training/audit.json": "4bfef31ef2a3c753fb1b9dfe6788db9da7ef68a2dab383900e01ee55b9219fe7",
    "evaluation-protocol.json": "595c87f6de78e0abb03590080b8b46aa3f3366825d8af42f26f9804087cfec12",
    "results.json": "0526f81359f7c472ed38d4fa394c2de528fe52764550892e0a8442d2370d5e8f",
}
digest, read, keep = semantic.digest, semantic.read, semantic.keep


def freeze(out):
    for name, expected in ARCHIVE_PINS.items():
        assert digest(semantic.OUT / name) == expected, name
    archived = read(semantic.OUT / "protocol.json")
    for name, expected in archived["inputs"].items():
        assert digest(semantic.OUT / name) == expected, name
    for name, expected in archived["files"].items():
        assert digest(Path(name)) == expected, name

    adapters = {"parent": semantic.INITIAL, "original": semantic.verify_control(),
                "diverse": fit.fit(semantic.OUT, semantic.INITIAL, semantic.MODEL, semantic.TRAINER)}
    semantic.verify_training(semantic.OUT)
    audits = {name: read(root / "training/audit.json") for name, root in ROOTS.items()}
    labels = {name: [r["messages"][-1]["content"] for r in fit._rows(root / "train.jsonl")]
              for name, root in ROOTS.items()}
    assert labels["original"] == labels["diverse"]
    targets = [labels["original"][row - 1].split(",", 1)[0] for row in audits["original"]["visits"]]
    first_targets = {
        str(step): {label: targets[:step].count(label) for label in ("true", "false")}
        for step in (*STEPS, fit.UPDATES)
    }

    cases = read(semantic.OUT / "cases.json")
    assert len(cases) == 32 and len({row["pair"] for row in cases}) == 16
    assert sum(row["truth"][0] for row in cases) == 16
    requests = [dict(id="repair", messages=r["messages"], sampling=semantic.SAMPLING) for r in cases]
    request_pins = [semantic.experiment.transfer.fingerprint(req) for req in requests]
    evaluation = read(semantic.OUT / "evaluation-protocol.json")
    assert evaluation["protocol_sha256"] == ARCHIVE_PINS["protocol.json"]
    assert evaluation["training_audit"] == ARCHIVE_PINS["training/audit.json"]
    assert request_pins == evaluation["requests"]
    assert {name: digest(path) for name, path in adapters.items()} == evaluation["adapters"]

    protocol = dict(
        steps=[0, *STEPS, fit.UPDATES], continuation="independent exact trainer prefixes from tool-four",
        full_sft_config=fit.CONFIG, training_first_targets=first_targets,
        sampling=semantic.SAMPLING, context=semantic.CONTEXT,
        max_new=semantic.MAX_NEW, requests=request_pins, archived_comparison=ARCHIVE_PINS,
        audited_training={arm: digest(root / "training/audit.json") for arm, root in ROOTS.items()},
        files={str(Path(__file__).resolve()): digest(Path(__file__))},
        primary="complete sampled pairs and count of true leaders against false among 32 balanced prompts",
        first_token_ties="a tie is neither a true leader nor a correct first-token prediction",
    )
    keep(out / "protocol.json", protocol)
    return cases, requests, adapters, audits


def checkpoint(out, arm, steps, reference):
    attempts = out / "checkpoints" / arm / f"step-{steps:02}"
    attempts.mkdir(parents=True, exist_ok=True)
    previous = sorted(attempts.glob("attempt-*"))
    saved = [attempt for attempt in previous if (attempt / "adapter.bin").is_file()]
    if saved:
        attempt, = saved
    else:
        attempt = attempts / f"attempt-{len(previous) + 1:03}"
        attempt.mkdir()
    inputs = reference["inputs"]
    adapter = attempt / "adapter.bin"
    command = [semantic.TRAINER, "train", semantic.MODEL, inputs["train"]["path"],
               inputs["instrument"]["path"], adapter, str(steps), str(fit.SEED),
               str(fit.FULL_BLOCKS), semantic.INITIAL]
    receipt = dict(argv=[str(arg) for arg in command], protocol_sha256=digest(out / "protocol.json"))
    if saved:
        assert receipt == read(attempt / "command.json")
    else:
        fit._write_new(attempt / "command.json", receipt)
        print(f"{arm} step {steps}: replaying from tool-four", flush=True)
        with (attempt / "train.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    from semantic_checkpoint_audit import audit_prefix

    audit = audit_prefix(attempt, reference, steps)
    keep(attempt / "audit.json", audit)
    assert all(digest(Path(artifact["path"])) == artifact["sha256"]
               for artifact in reference["inputs"].values())
    print(f"{arm} step {steps}: archived visit and loss prefix matched", flush=True)
    return attempt, audit


def metrics(cases, observations):
    score = semantic.summarize(cases, observations["generations"], observations["readouts"])
    rows = score["rows"]
    summary = score["summary"]["all"]
    summary.update(
        true_leaders=sum(r["log_odds_true"] > 0 for r in rows),
        false_leaders=sum(r["log_odds_true"] < 0 for r in rows),
        ties=sum(r["log_odds_true"] == 0 for r in rows),
        false_first_correct=sum(r["first_correct"] and not r["truth"][0] for r in rows),
        mean_log_odds_true=mean(r["log_odds_true"] for r in rows),
        min_log_odds_true=min(r["log_odds_true"] for r in rows),
        max_log_odds_true=max(r["log_odds_true"] for r in rows),
    )
    return score


def archived_scores(cases):
    prior = read(semantic.OUT / "results.json")
    assert prior["protocol_sha256"] == ARCHIVE_PINS["evaluation-protocol.json"]
    results = {}
    for arm in ("parent", *ROOTS):
        path = semantic.OUT / f"{arm}-observations.json"
        observations = read(path)
        assert semantic.summarize(cases, observations["generations"], observations["readouts"]) == prior["arms"][arm]
        results[arm] = dict(observations_sha256=digest(path), score=metrics(cases, observations))
    return results


def evaluate(attempt, cases, requests, audit):
    observations_file = attempt / "observations.json"
    adapter = attempt / "adapter.bin"
    assert digest(adapter) == audit["adapter_sha256"]
    if observations_file.exists():
        observations = read(observations_file)
        assert observations["adapter_sha256"] == audit["adapter_sha256"]
    else:
        generations = semantic.run(cases, requests, [semantic.SAMPLER, "generate", semantic.MODEL,
                                                     adapter, str(semantic.MAX_NEW), "4"])
        readouts = semantic.run(cases, requests, [semantic.READOUT, "first_token", semantic.MODEL, adapter])
        observations = dict(adapter_sha256=audit["adapter_sha256"], generations=generations, readouts=readouts)
    score = metrics(cases, observations)
    keep(observations_file, observations)
    return dict(adapter_sha256=audit["adapter_sha256"], audit_sha256=digest(attempt / "audit.json"),
                observations_sha256=digest(observations_file), score=score)


def main(out):
    cases, requests, adapters, audits = freeze(out)
    archived = archived_scores(cases)
    scores = {arm: {"50": dict(archived[arm], adapter_sha256=digest(adapters[arm]))} for arm in ROOTS}
    for steps in STEPS:
        for arm in ROOTS:
            attempt, audit = checkpoint(out, arm, steps, audits[arm])
            result = evaluate(attempt, cases, requests, audit)
            scores[arm][str(steps)] = result
            print(arm, steps, json.dumps(result["score"]["summary"]["all"]), flush=True)
    audit_source = Path(__file__).with_name("semantic_checkpoint_audit.py")
    keep(out / "results.json", dict(protocol_sha256=digest(out / "protocol.json"),
                                    audit_source_sha256=digest(audit_source),
                                    tool_four=dict(archived["parent"], adapter_sha256=digest(adapters["parent"])),
                                    datasets=scores))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=OUT)
    main(cli.parse_args().output.resolve())
