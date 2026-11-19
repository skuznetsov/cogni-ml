"""Clear Adam's second moment once before each archived two-update branch."""

import argparse
import json
import math
from pathlib import Path

import first_moment_reset as first
import per_update_order as baseline

forward, prior, fit = baseline.forward, baseline.prior, baseline.fit
digest, read, keep = baseline.digest, baseline.read, baseline.keep
OUT = forward.semantic.OUT / "second-moment-reset-v1"
SOURCE = Path(__file__).with_name("train_second_moment_reset.cr")
PHASES = {"common": -2, "forward-1": -1, "forward-2": 0,
          "swapped-1": -1, "swapped-2": 0, "control-forward": 0}


def verify_training(attempt, events, archive, scores, visits, schedules, labels):
    setup = {"prefill", "prefill_recovery", "start"}
    assert [e for e in events if e["event"] in setup] == [e for e in archive if e["event"] in setup]
    updates = [e for e in events if e["event"] == "update"]
    old_updates = [e for e in archive if e["event"] == "update"]
    control = [e for e in old_updates if e["order"] == "forward"]
    assert [e["row"] for e in control] == visits
    assert [e for e in updates if e["order"] == "control"] == [
        {**e, "order": "control"} for e in control
    ]

    resets = [e for e in events if e["event"] == "reset_second"]
    assert [(e["target"], e["order"], e["step"]) for e in resets] == [
        (target, order, target - 2) for target in baseline.STEPS for order in first.ORDERS
    ]
    assert all(0 < e["active_second_tensors"] <= e["state_tensors"] == 56 for e in resets)
    saves = {(e["target"], e["phase"]): e for e in events if e["event"] == "saved"}
    assert len(saves) == len(PHASES) * len(baseline.STEPS)
    assert len(events) == len(saves) + len(resets) + len(updates) + sum(e["event"] in setup for e in events)
    assert len(updates) == len(visits) + 4 * len(baseline.STEPS)

    branches = {}
    for target in baseline.STEPS:
        ordinary = scores[str(target)]
        for phase, offset in PHASES.items():
            directory, sha = first.check_save(saves[target, phase], attempt, target, phase, target + offset)
            if phase == "common" or phase == "control-forward":
                archived_phase = "common" if phase == "common" else "forward-2"
                assert sha == ordinary[archived_phase]["adapter_sha256"]
            else:
                branches[target, phase] = (directory, sha)
        for order in first.ORDERS:
            planned = visits if order == "forward" else schedules[str(target)]["visits"]
            reference = control[target - 2:target] if order == "forward" else [
                e for e in old_updates if e["order"] == order and e["target"] == target
            ]
            trial = [e for e in updates if e["order"] == order and e["target"] == target]
            observed = baseline.update_signatures(trial)
            assert len(observed) == 2
            assert observed[0] == baseline.update_signatures(reference)[0]
            for offset, (depth, row, token, loss) in enumerate(observed, 1):
                expected_depth = target - 2 + offset
                assert (depth, row, token) == (
                    expected_depth, planned[expected_depth - 1], forward.TOKEN_IDS[labels[row - 1]]
                )
                assert math.isfinite(loss)
    return branches


def run(out, driver):
    control_protocol, control = baseline.archive_at(baseline.OUT, first.BASELINE_PINS)
    forward_protocol, _ = baseline.archive_at(forward.OUT, baseline.swap.ARCHIVE_PINS)
    swap_protocol, _ = baseline.archive_at(baseline.swap.OUT, baseline.SWAP_PINS)
    for name, expected in (forward_protocol["files"] | swap_protocol["files"] | control_protocol["files"]).items():
        assert digest(Path(name)) == expected, name
    cases, requests, _, audits = prior.freeze(prior.OUT)
    eval_fields = ("context", "sampling", "max_new", "requests")
    eval_metadata = {key: control_protocol[key] for key in eval_fields}
    frozen_eval = read(prior.OUT / "protocol.json")
    assert eval_metadata == {key: frozen_eval[key] for key in eval_fields}
    visits = control_protocol["forward_visits"]
    schedules = control_protocol["checkpoints"]
    labels = forward.labels_for(audits["original"])
    assert labels == forward.labels_for(audits["diverse"])
    _, replay = forward.make_schedule(baseline.swap.DRIVER, labels, audits)
    assert visits == replay
    assert schedules == {str(step): baseline.swap.swap_tail(visits, labels, step) for step in baseline.STEPS}
    for arm, entries in control["scores"].items():
        first.frozen_observations(arm, entries, cases, requests)

    keep(out / "protocol.json", dict(
        baseline=first.BASELINE_PINS, training_config=control_protocol["training_config"],
        intervention="zero second moment once at the shared prefix, preserving weights, first moment and step",
        forward_visits=visits, checkpoints=schedules, **eval_metadata,
        files={str(path.resolve()): digest(path) for path in (driver, SOURCE, Path(__file__), Path(first.__file__))},
    ))
    visits_path = out / "visits.json"
    keep(visits_path, visits)
    protocol_sha = digest(out / "protocol.json")

    runs = {}
    for arm, reference in audits.items():
        attempt, events = first.train(out / arm, driver, reference, visits_path, protocol_sha)
        archive_attempt, = (baseline.OUT / arm).glob("attempt-*")
        archive_log = archive_attempt / "train.log"
        assert digest(archive_log) == read(archive_attempt / "finished.json")["train_log_sha256"]
        branches = verify_training(
            attempt, events, fit._events(archive_log), control["scores"][arm], visits, schedules, labels
        )
        runs[arm] = digest(attempt / "train.log"), branches
        print(f"{arm}: matched 50 control updates and six reference adapters, checked six v resets", flush=True)

    scores, comparisons = {}, {}
    for arm, (log_sha, branches) in runs.items():
        scores[arm], comparisons[arm] = {}, {}
        for step in baseline.STEPS:
            reset = {}
            for order in first.ORDERS:
                for offset in (1, 2):
                    phase = f"{order}-{offset}"
                    print(f"Evaluating {arm} target {step} {phase}", flush=True)
                    directory, adapter_sha = branches[step, phase]
                    audit = dict(adapter_sha256=adapter_sha, train_log_sha256=log_sha, target=step,
                                 step=step - 2 + offset, phase=phase, protocol_sha256=protocol_sha)
                    keep(directory / "audit.json", audit)
                    reset[phase] = dict(path=str(directory), **prior.evaluate(directory, cases, requests, audit))
            key = str(step)
            scores[arm][key] = reset
            comparison = first.compare(control["scores"][arm][key], reset)
            comparisons[arm][key] = comparison
            print(arm, step, json.dumps(comparison), flush=True)
    keep(out / "results.json", dict(protocol_sha256=protocol_sha, scores=scores, comparisons=comparisons,
                                   tool_four=control["tool_four"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=OUT)
    out = cli.parse_args().output.resolve()
    run(out, out / "native-reset")
