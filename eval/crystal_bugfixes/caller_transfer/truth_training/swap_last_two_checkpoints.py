"""Swap the last two visits of each frozen alternating full-SFT prefix."""

import argparse
from collections import Counter
import json
from pathlib import Path

import alternate_checkpoints as forward
import reverse_order_checkpoints as reverse

ARCHIVE_PINS, DRIVER, side = reverse.ARCHIVE_PINS, reverse.DRIVER, reverse.side

OUT = forward.semantic.OUT / "last-two-swap-v1"
digest, read, keep = forward.digest, forward.read, forward.keep


def swap_tail(visits, labels, step):
    original = visits[:step]
    assert len(original) == step >= 2
    swapped = original[:-2] + original[-2:][::-1]
    last_forward, last_swapped = (labels[rows[-1] - 1] for rows in (original, swapped))
    assert last_forward != last_swapped
    assert Counter(original) == Counter(swapped)
    return dict(visits=swapped, last_forward=last_forward, last_swapped=last_swapped)


def frozen_control(archive, reference, labels):
    control_root = forward.OUT / "unchanged-order-control"
    folder, = control_root.glob("attempt-*/step-10")
    assert digest(folder / "adapter.bin") == archive["control"]["adapter_sha256"]
    assert digest(folder.parent / "train.log") == archive["control"]["train_log_sha256"]
    result = forward.qualify(forward.OUT, DRIVER, reference, labels,
                             archive["shuffled"]["original"]["10"], ARCHIVE_PINS["protocol.json"])
    assert result == archive["control"]


def archived_updates(arm, reference, visits, labels, scores):
    folder, = (forward.OUT / arm).glob("attempt-*")
    log = folder / "train.log"
    log_sha = digest(log)
    for step in forward.CHECKPOINTS:
        path = folder / f"step-{step:02}" / "audit.json"
        assert digest(path) == scores[str(step)]["audit_sha256"]
        assert read(path)["train_log_sha256"] == log_sha
    updates, _ = forward.check_log(log, reference, visits, labels, forward.CHECKPOINTS)
    return updates


def compare(original, treatment, schedule, shared_updates):
    baseline = side(original, schedule["last_forward"])
    swapped = side(treatment, schedule["last_swapped"])
    delta = swapped["mean_log_odds_true"] - baseline["mean_log_odds_true"]
    sign = 1 if swapped["last_target"] == "true" else -1
    return dict(forward=baseline, swapped=swapped, mean_log_odds_delta=delta,
                mean_moves_toward_new_last=(sign * delta > 0), shared_updates=shared_updates)


def main(out):
    assert out != forward.OUT
    prior = forward.prior
    for file, expected in ARCHIVE_PINS.items():
        assert digest(forward.OUT / file) == expected, file
    archive = read(forward.OUT / "results.json")
    protocol = read(forward.OUT / "protocol.json")
    assert archive["protocol_sha256"] == ARCHIVE_PINS["protocol.json"]
    for file, expected in protocol["files"].items():
        assert digest(Path(file)) == expected, file
    cases, requests, _, audits = prior.freeze(prior.OUT)
    assert protocol["requests"] == read(prior.OUT / "protocol.json")["requests"]
    labels = forward.labels_for(audits["original"])
    assert labels == forward.labels_for(audits["diverse"])
    decks, visits = forward.make_schedule(DRIVER, labels, audits)
    assert decks == protocol["seed_decks"] and visits == protocol["visits"]
    schedules = {str(step): swap_tail(visits, labels, step) for step in forward.CHECKPOINTS}

    frozen_control(archive, audits["original"], labels)
    keep(out / "protocol.json", dict(
        archive_pins=ARCHIVE_PINS, training_config=protocol["training_config"],
        sampling=protocol["sampling"], context=protocol["context"], max_new=protocol["max_new"],
        requests=protocol["requests"], archive_audits=protocol["archive_audits"],
        forward_visits=visits, checkpoints=schedules,
        files={**protocol["files"], str(Path(__file__).resolve()): digest(Path(__file__)),
               str(Path(reverse.__file__).resolve()): digest(Path(reverse.__file__))},
    ))
    protocol_sha = digest(out / "protocol.json")
    scores, comparisons = {}, {}
    for arm, reference in audits.items():
        baseline = archive["alternating"][arm]
        old_updates = archived_updates(arm, reference, visits, labels, baseline)
        scores[arm], comparisons[arm] = {}, {}
        for step in forward.CHECKPOINTS:
            key = str(step)
            schedule = schedules[key]
            assert digest(DRIVER) == protocol["files"][str(DRIVER)]
            print(f"{arm} step {step}: checking the swapped prefix", flush=True)
            updates, checkpoints = forward.train(out / arm / f"target-{step:02}", DRIVER, reference,
                                                  schedule["visits"], labels, (step,), protocol_sha)
            assert [u["loss"] for u in updates[:-2]] == [u["loss"] for u in old_updates[:step - 2]]
            folder, audit = checkpoints[step]
            entry = prior.evaluate(folder, cases, requests, audit)
            scores[arm][key] = entry
            comparison = compare(baseline[key], entry, schedule, step - 2)
            comparisons[arm][key] = comparison
            print(arm, step, json.dumps(comparison), flush=True)
    keep(out / "results.json", dict(protocol_sha256=protocol_sha, tool_four=archive["tool_four"],
                                   swapped=scores, comparison=comparisons))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=OUT)
    main(cli.parse_args().output.resolve())
