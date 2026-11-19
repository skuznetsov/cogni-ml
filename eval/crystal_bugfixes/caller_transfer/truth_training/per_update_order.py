"""Measure the two branches of each archived last-two-visit swap."""

import argparse
import json
from pathlib import Path
import subprocess

import swap_last_two_checkpoints as swap

forward = swap.forward
prior, fit = forward.prior, forward.fit
digest, read, keep = forward.digest, forward.read, forward.keep
OUT = forward.semantic.OUT / "per-update-order-v1"
SOURCE = Path(__file__).with_name("train_order_branches.cr")
STEPS = forward.CHECKPOINTS
PHASE_OFFSETS = {"common": -2, "swapped-1": -1, "swapped-2": 0,
                 "forward-1": -1, "forward-2": 0}
SWAP_PINS = {
    "protocol.json": "c577405b42bf3db04e8f39c77168af1bc17b363e8ab13143a030873db6da8b81",
    "results.json": "406ebc4f52c403114f8168584759321a673492be07bc49d8008b1dd560427901",
}


def archive_at(path, pins):
    for name, expected in pins.items():
        assert digest(path / name) == expected, name
    results = read(path / "results.json")
    assert results["protocol_sha256"] == pins["protocol.json"]
    return read(path / "protocol.json"), results


def training_attempt(root, driver, reference, visits_path, protocol_sha):
    inputs = reference["inputs"]
    previous = sorted(root.glob("attempt-*"))
    complete = [p for p in previous if (p / "finished.json").is_file()]
    attempt, = complete if complete else [root / f"attempt-{len(previous) + 1:03}"]
    args = [driver, forward.semantic.MODEL, inputs["train"]["path"], inputs["instrument"]["path"],
            forward.semantic.INITIAL, visits_path, attempt]
    receipt = dict(argv=list(map(str, args)), protocol_sha256=protocol_sha)
    keep(attempt / "command.json", receipt)
    log = attempt / "train.log"
    if not complete:
        print(f"{root.name}: running 50 forward updates and 6 swapped updates", flush=True)
        with log.open("x", encoding="utf-8") as output:
            subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, timeout=1500, check=True)
    keep(attempt / "finished.json", dict(train_log_sha256=digest(log)))
    return attempt, fit._events(log)


def saved_branches(attempt, events):
    keys = [(s, phase, s + offset) for s in STEPS for phase, offset in PHASE_OFFSETS.items()]
    saves = [e for e in events if e["event"] == "saved"]
    assert [(e["target"], e["phase"], e["step"]) for e in saves] == keys
    branches = {}
    for step, phase, depth in keys:
        directory = attempt / f"target-{step}" / phase
        adapter = directory / "adapter.bin"
        assert saves[len(branches)]["adapter"] == str(adapter)
        branches[step, phase] = (depth, directory, digest(adapter))
    return branches


def update_signatures(events):
    return [(e["step"], e["row"], e["first_target_id"], e["loss"]) for e in events]


def matching_endpoints(arm, reference, visits, labels, schedules, archive, swapped, events, branches):
    def prefill(events):
        return [
            {k: v for k, v in e.items() if k != "seconds"}
            for e in events if e["event"] in ("prefill", "prefill_recovery", "start")
        ]

    baseline = archive["alternating"][arm]
    baseline_updates = swap.archived_updates(arm, reference, visits, labels, baseline)
    baseline_root, = (forward.OUT / arm).glob("attempt-*")
    assert prefill(events) == prefill(fit._events(baseline_root / "train.log"))
    updates = [e for e in events if e["event"] == "update"]
    assert len(updates) == len(visits) + 2 * len(STEPS)
    assert update_signatures(e for e in updates if e["order"] == "forward") == update_signatures(baseline_updates)
    endpoints = {}
    for step in STEPS:
        path, = (swap.OUT / arm / f"target-{step:02}").glob("attempt-*")
        folder = path / f"step-{step:02}"
        entry = swapped["swapped"][arm][str(step)]
        assert digest(folder / "audit.json") == entry["audit_sha256"]
        log = path / "train.log"
        assert digest(log) == read(folder / "audit.json")["train_log_sha256"]
        old_updates, _ = forward.check_log(log, reference, schedules[str(step)]["visits"], labels, (step,))
        assert update_signatures(old_updates[:-2]) == update_signatures(baseline_updates[:step - 2])
        assert update_signatures(e for e in updates if e["order"] == "swapped" and e["target"] == step) == (
            update_signatures(old_updates[-2:])
        )
        for phase, path, score in (("forward-2", baseline_root / f"step-{step:02}", baseline[str(step)]),
                                   ("swapped-2", folder, entry)):
            assert branches[step, phase][2] == score["adapter_sha256"] == digest(path / "adapter.bin")
            assert digest(path / "observations.json") == score["observations_sha256"]
            endpoints[step, phase] = path, score
    return endpoints


def summarize(phase_scores):
    summary = {phase: entry["score"]["summary"]["all"] for phase, entry in phase_scores.items()}
    keys = ("mean_log_odds_true", "true_leaders", "both_correct")
    levels = {phase: {key: values[key] for key in keys} for phase, values in summary.items()}
    order_gap = {
        str(offset): {key: levels[f"swapped-{offset}"][key] - levels[f"forward-{offset}"][key]
                      for key in keys}
        for offset in (1, 2)
    }
    return dict(levels=levels, swapped_minus_forward=order_gap)


def main(out, driver):
    protocol, archive = archive_at(forward.OUT, swap.ARCHIVE_PINS)
    swap_protocol, swapped = archive_at(swap.OUT, SWAP_PINS)
    for name, expected in (protocol["files"] | swap_protocol["files"]).items():
        assert digest(Path(name)) == expected, name
    cases, requests, _, audits = prior.freeze(prior.OUT)
    assert protocol["requests"] == swap_protocol["requests"] == read(prior.OUT / "protocol.json")["requests"]
    labels = forward.labels_for(audits["original"])
    assert labels == forward.labels_for(audits["diverse"])
    decks, visits = forward.make_schedule(swap.DRIVER, labels, audits)
    assert decks == protocol["seed_decks"] and visits == protocol["visits"] == swap_protocol["forward_visits"]
    schedules = {str(step): swap.swap_tail(visits, labels, step) for step in STEPS}
    assert schedules == swap_protocol["checkpoints"]
    keep(out / "protocol.json", dict(
        forward_archive=swap.ARCHIVE_PINS, swapped_archive=SWAP_PINS, forward_visits=visits,
        checkpoints=schedules, phases=PHASE_OFFSETS, archive_audits=protocol["archive_audits"],
        training_config=protocol["training_config"], sampling=protocol["sampling"],
        context=protocol["context"], max_new=protocol["max_new"], requests=protocol["requests"],
        files={str(path.resolve()): digest(path) for path in (driver, SOURCE, Path(__file__))},
    ))
    visits_path = out / "visits.json"
    keep(visits_path, visits)
    protocol_sha = digest(out / "protocol.json")
    runs = {}
    for arm, reference in audits.items():
        attempt, events = training_attempt(out / arm, driver, reference, visits_path, protocol_sha)
        branches = saved_branches(attempt, events)
        endpoints = matching_endpoints(arm, reference, visits, labels, schedules, archive, swapped, events, branches)
        assert len(endpoints) == 2 * len(STEPS)
        runs[arm] = digest(attempt / "train.log"), branches, endpoints
        print(f"{arm}: 6/6 endpoint adapters matched byte for byte", flush=True)

    scores, comparisons = {}, {}
    for arm, (log_sha, branches, endpoints) in runs.items():
        scores[arm], comparisons[arm] = {}, {}
        for step in STEPS:
            phase_scores = {}
            for phase in PHASE_OFFSETS:
                depth, directory, adapter_sha = branches[step, phase]
                if (step, phase) in endpoints:
                    path, archived = endpoints[step, phase]
                    entry = prior.evaluate(path, cases, requests, read(path / "audit.json"))
                    assert entry == archived
                else:
                    audit = dict(adapter_sha256=adapter_sha, train_log_sha256=log_sha, step=depth,
                                 target=step, phase=phase, protocol_sha256=protocol_sha)
                    keep(directory / "audit.json", audit)
                    entry = prior.evaluate(directory, cases, requests, audit)
                phase_scores[phase] = dict(path=str(directory), **entry)
            scores[arm][str(step)] = phase_scores
            comparison = summarize(phase_scores)
            comparisons[arm][str(step)] = comparison
            print(arm, step, json.dumps(comparison), flush=True)
    keep(out / "results.json", dict(protocol_sha256=protocol_sha, tool_four=archive["tool_four"],
                                   scores=scores, comparisons=comparisons))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=OUT)
    cli.add_argument("--driver", type=Path)
    args = cli.parse_args()
    out = args.output.resolve()
    main(out, (args.driver or out / "native-branches").resolve())
