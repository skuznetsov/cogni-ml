"""Clear both Adam moments once before each archived two-update branch."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import first_moment_reset as first
import per_update_order as baseline

forward, prior, fit = baseline.forward, baseline.prior, baseline.fit
digest, read, keep = baseline.digest, baseline.read, baseline.keep
OUT = forward.semantic.OUT / "both-moments-reset-v1"
SOURCE = Path(__file__).with_name("train_both_moments_reset.cr")
SINGLE_MOMENT_PINS = {
    "first-moment-reset-v1": {
        "protocol.json": "6e175594d348637641eadbfae9010886101b9c123ff27d32d0a7fc7e9b4131cf",
        "results.json": "88d6f705e50a9f691bd7ab6776cec40d1f8b4720d06756ac7251fe57e5a594f0",
    },
    "second-moment-reset-v1": {
        "protocol.json": "02f2ad06cf322930ddf6a7f49830c1368055131b0ff379193643f94edc4afaa3",
        "results.json": "dfa2b50b2123081ca7de31865bbef083deab13bae35bf92c7de897bc38477555",
    },
}


def check_branches(attempt, events, archive, scores, visits, schedules, labels):
    shared_events = []
    for event in events:
        assert event["event"] not in ("reset_first", "reset_second")
        if event["event"] == "reset_both":
            assert 0 < event["active_second_tensors"] <= event["state_tensors"] == 56
            # Reuse the chronology and control checks after validating the additional reset.
            event = dict(event, event="reset_first")
        shared_events.append(event)
    return first.check_branches(attempt, shared_events, archive, scores, visits, schedules, labels)


def evaluate(arm, target, phase, directory, adapter_sha, log_sha, protocol_sha, cases, requests):
    offset = int(phase[-1])
    audit = dict(adapter_sha256=adapter_sha, train_log_sha256=log_sha, target=target,
                 step=target - 2 + offset, phase=phase, protocol_sha256=protocol_sha)
    keep(directory / "audit.json", audit)
    entry = prior.evaluate(directory, cases, requests, audit)
    score = entry["score"]["summary"]["all"]
    print(f"{arm} target {target} {phase}: {score['mean_log_odds_true']:+.4f}, "
          f"{score['both_correct']}/16 pairs, {score['format_ok']}/32 formatted", flush=True)
    return dict(path=str(directory), **entry)


def run(out, driver):
    control_protocol, control = baseline.archive_at(baseline.OUT, first.BASELINE_PINS)
    forward_protocol, _ = baseline.archive_at(forward.OUT, baseline.swap.ARCHIVE_PINS)
    swap_protocol, _ = baseline.archive_at(baseline.swap.OUT, baseline.SWAP_PINS)
    for protocol in (forward_protocol, swap_protocol, control_protocol):
        for name, expected in protocol["files"].items():
            assert digest(Path(name)) == expected, name

    cases, requests, _, audits = prior.freeze(prior.OUT)
    eval_keys = ("context", "sampling", "max_new", "requests")
    eval_metadata = {key: control_protocol[key] for key in eval_keys}
    frozen_eval = read(prior.OUT / "protocol.json")
    assert eval_metadata == {key: frozen_eval[key] for key in eval_keys}
    visits = control_protocol["forward_visits"]
    schedules = control_protocol["checkpoints"]
    labels = forward.labels_for(audits["original"])
    assert labels == forward.labels_for(audits["diverse"])
    _, replay = forward.make_schedule(baseline.swap.DRIVER, labels, audits)
    assert visits == replay
    assert schedules == {str(step): baseline.swap.swap_tail(visits, labels, step) for step in baseline.STEPS}
    for arm, entries in control["scores"].items():
        first.frozen_observations(arm, entries, cases, requests)
    for name, pins in SINGLE_MOMENT_PINS.items():
        protocol, _ = baseline.archive_at(OUT.parent / name, pins)
        assert protocol["baseline"] == first.BASELINE_PINS
        for key in (*eval_keys, "training_config", "checkpoints", "forward_visits"):
            assert protocol[key] == control_protocol[key], (name, key)

    keep(out / "protocol.json", dict(
        baseline=first.BASELINE_PINS, single_moment_archives=SINGLE_MOMENT_PINS,
        training_config=control_protocol["training_config"],
        intervention="zero first and second moments once at the shared prefix, preserving weights and step",
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
        branches = check_branches(
            attempt, events, fit._events(archive_log), control["scores"][arm], visits, schedules, labels
        )
        runs[arm] = digest(attempt / "train.log"), branches
        print(f"{arm}: matched 50 control updates and six reference adapters, checked six m/v resets", flush=True)

    tasks = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for arm, (log_sha, branches) in runs.items():
            for (step, phase), (directory, adapter_sha) in branches.items():
                tasks[arm, step, phase] = pool.submit(
                    evaluate, arm, step, phase, directory, adapter_sha, log_sha, protocol_sha, cases, requests
                )
        scores, comparisons = {}, {}
        for arm in runs:
            scores[arm], comparisons[arm] = {}, {}
            for step in baseline.STEPS:
                key = str(step)
                reset = {f"{order}-{offset}": tasks[arm, step, f"{order}-{offset}"].result()
                         for order in first.ORDERS for offset in (1, 2)}
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
