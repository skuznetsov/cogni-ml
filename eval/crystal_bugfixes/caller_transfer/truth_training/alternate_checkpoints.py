"""Measure full-SFT checkpoints under an alternating first-literal schedule."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import subprocess

import fit
from fit_first_token import TOKEN_IDS
import semantic_checkpoints as prior
from semantic_checkpoint_audit import _factor_updates, _timing_free

semantic = prior.semantic
OUT = semantic.OUT / "alternating-order-v2"
SOURCE = Path(__file__).with_name("train_alternating_truth.cr")
CHECKPOINTS = (10, 25, 50)
PRIOR_RESULTS_SHA = "330cf10a1b0b7a9ec83533001c820c308efe2291d589b92146e36f9b68964b7d"
digest, read, keep = prior.digest, prior.read, prior.keep


def labels_for(reference):
    completions = [r["messages"][-1]["content"] for r in fit._rows(Path(reference["inputs"]["train"]["path"]))]
    allowed = {f"{a}, {b}" for a in TOKEN_IDS for b in TOKEN_IDS}
    assert all(completion in allowed for completion in completions)
    labels = [completion.split(", ")[0] for completion in completions]
    assert Counter(labels) == {"true": 16, "false": 16}
    return labels


def make_schedule(driver, labels, audits):
    size = len(labels)
    order = subprocess.run([driver, "order", str(size), str(fit.UPDATES), str(fit.SEED)],
                           capture_output=True, text=True, timeout=30, check=True)
    decks = json.loads(order.stdout)
    expected_length = (fit.UPDATES + size - 1) // size * size
    assert len(decks) == expected_length
    assert all(audit["visits"] == decks[:fit.UPDATES] for audit in audits.values())
    visits = []
    for start in range(0, len(decks), size):
        deck = decks[start:start + size]
        assert sorted(deck) == list(range(1, size + 1))
        positive = [row for row in deck if labels[row - 1] == "true"]
        negative = [row for row in deck if labels[row - 1] == "false"]
        for true_row, false_row in zip(positive, negative, strict=True):
            visits.extend((true_row, false_row))
    visits = visits[:fit.UPDATES]
    assert [labels[row - 1] for row in visits] == ["true", "false"] * (fit.UPDATES // 2)
    return decks, visits


def archive_events(reference):
    archive = Path(reference["inputs"]["train"]["path"]).parent / "training/train.log"
    assert digest(archive) == reference["train_log_sha256"]
    return fit._events(archive)


def check_log(log, reference, visits, labels, steps):
    events = fit._events(log)
    reference_events = archive_events(reference)
    start_index, = [i for i, event in enumerate(reference_events) if event["event"] == "start"]
    before = [_timing_free(event) for event in reference_events[:start_index + 1]]
    before[-1]["updates"] = len(visits)
    assert [_timing_free(event) for event in events[:start_index + 1]] == before
    expected = [("update", step) for step in range(1, len(visits) + 1)]
    for offset, step in enumerate(steps):
        expected.insert(step + offset, ("saved", step))
    transitions = events[start_index + 1:]
    assert [(e["event"], e["step"]) for e in transitions] == expected
    updates = [e for e in transitions if e["event"] == "update"]
    assert [u["row"] for u in updates] == visits
    assert [u["first_target_id"] for u in updates] == [TOKEN_IDS[labels[row - 1]] for row in visits]
    assert all(math.isfinite(u["loss"]) for u in updates)
    saved = [e for e in transitions if e["event"] == "saved"]
    assert all(0 <= s["seconds"] < math.inf for s in saved)
    assert math.isfinite(saved[-1]["validation_loss"])
    return updates, saved


def train(root, driver, reference, visits, labels, steps, protocol_sha):
    keep(root / "visits.json", visits)
    keep(root / "steps.json", list(steps))
    previous = sorted(root.glob("attempt-*"))
    complete = [p for p in previous if all((p / f"step-{s:02}/adapter.bin").is_file() for s in steps)]
    if complete:
        attempt, = complete
    else:
        attempt = root / f"attempt-{len(previous) + 1:03}"
        attempt.mkdir()
    log = attempt / "train.log"
    inputs = reference["inputs"]
    args = [driver, "train", semantic.MODEL, inputs["train"]["path"], inputs["instrument"]["path"],
            semantic.INITIAL, root / "visits.json", root / "steps.json",
            attempt, str(fit.SEED), str(fit.FULL_BLOCKS)]
    receipt = dict(argv=[str(arg) for arg in args], protocol_sha256=protocol_sha)
    if complete:
        assert read(attempt / "command.json") == receipt
    else:
        keep(attempt / "command.json", receipt)
        print(f"{root.name}: training {len(visits)} updates", flush=True)
        with log.open("x", encoding="utf-8") as output:
            subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, timeout=900, check=True)

    updates, saved = check_log(log, reference, visits, labels, steps)
    checker_path, checker = fit._checker(Path(fit.__file__).resolve().parents[4])
    assert str(checker_path.resolve()) == inputs["adapter_checker"]["path"]
    assert digest(checker_path) == inputs["adapter_checker"]["sha256"]
    for key in ("train", "instrument", "initial"):
        assert digest(Path(inputs[key]["path"])) == inputs[key]["sha256"]
    audited = {}
    for save in saved:
        step = save["step"]
        folder = attempt / f"step-{step:02}"
        adapter = folder / "adapter.bin"
        assert save["adapter"] == str(adapter)
        with open(os.devnull, "w") as sink, redirect_stdout(sink):
            adapter_sha = checker.check_adapter(adapter)
        factors = _factor_updates(semantic.INITIAL, adapter)
        target_count = sum(reference["target_tokens_by_train_row"][str(row)] for row in visits[:step])
        assert target_count == step * 4
        audit = dict(
            adapter_sha256=adapter_sha, train_log_sha256=digest(log), steps=step,
            first_targets=dict(Counter(labels[row - 1] for row in visits[:step])),
            target_tokens=target_count, updated_projections=len(factors), updated_factors=len(factors) * 2,
            seconds=save["seconds"],
        )
        keep(folder / "audit.json", audit)
        audited[step] = (folder, audit)
    return updates, audited


def qualify(out, driver, reference, labels, baseline, protocol_sha):
    steps = (10,)
    updates, saved = train(out / "unchanged-order-control", driver, reference,
                           reference["visits"][:10], labels, steps, protocol_sha)
    assert [u["loss"] for u in updates] == reference["losses"]["updates"][:10]
    archive, = (prior.OUT / "checkpoints/original/step-10").glob("attempt-*/observations.json")
    adapter_sha = saved[10][1]["adapter_sha256"]
    assert adapter_sha == baseline["adapter_sha256"] == digest(archive.with_name("adapter.bin"))
    print("Control: all 10 archived losses and adapter bytes matched", flush=True)
    return saved[10][1]


def main(out, driver):
    previous_path = prior.OUT / "results.json"
    assert digest(previous_path) == PRIOR_RESULTS_SHA
    cases, requests, _, audits = prior.freeze(prior.OUT)
    previous = read(previous_path)
    assert previous["protocol_sha256"] == digest(prior.OUT / "protocol.json")
    labels = labels_for(audits["original"])
    assert labels == labels_for(audits["diverse"])
    decks, visits = make_schedule(driver, labels, audits)
    keep(out / "protocol.json", dict(
        baseline_sha256=PRIOR_RESULTS_SHA, steps=list(CHECKPOINTS),
        training_config=fit.CONFIG, sampling=semantic.SAMPLING, context=semantic.CONTEXT,
        max_new=semantic.MAX_NEW, seed_decks=decks, visits=visits,
        targets_by_step=[labels[row - 1] for row in visits],
        requests=read(prior.OUT / "protocol.json")["requests"],
        metrics="true-leader count and mean true/false log-odds on 32 balanced prompts, and sampled complete pairs",
        archive_audits={arm: digest(root / "training/audit.json") for arm, root in prior.ROOTS.items()},
        files={str(p): digest(p) for p in (driver, SOURCE.resolve(), Path(__file__).resolve())},
    ))
    protocol_sha = digest(out / "protocol.json")
    control = qualify(out, driver, audits["original"], labels, previous["datasets"]["original"]["10"], protocol_sha)
    scores = {}
    for arm, reference in audits.items():
        assert digest(driver) == read(out / "protocol.json")["files"][str(driver)]
        _, checkpoints = train(out / arm, driver, reference, visits, labels, CHECKPOINTS, protocol_sha)
        scores[arm] = {}
        for step, (folder, audit) in checkpoints.items():
            entry = prior.evaluate(folder, cases, requests, audit)
            scores[arm][str(step)] = entry
            print(arm, step, json.dumps(entry["score"]["summary"]["all"]), flush=True)
    keep(out / "results.json", dict(
        protocol_sha256=protocol_sha, control=control, tool_four=previous["tool_four"],
        shuffled=previous["datasets"], alternating=scores,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=OUT)
    cli.add_argument("--driver", type=Path, default=OUT / "native-checkpoints")
    args = cli.parse_args()
    main(args.output.resolve(), args.driver.resolve())
