"""Clear Adam's first moment once before each archived two-update branch."""

import argparse
import json
import math
from pathlib import Path
import subprocess

import per_update_order as baseline

forward, prior, fit = baseline.forward, baseline.prior, baseline.fit
digest, read, keep = baseline.digest, baseline.read, baseline.keep
OUT = forward.semantic.OUT / "first-moment-reset-v1"
SOURCE = Path(__file__).with_name("train_first_moment_reset.cr")
STEPS = baseline.STEPS
ORDERS = ("forward", "swapped")
BASELINE_PINS = {
    "protocol.json": "f9ffa5ab57ec680b8ca11ab20edf3dfcde962cc182e548d6749d7fca28085d42",
    "results.json": "0f280efa78d3cb8571956746783dce843659f8063a99094527e76def60d78321",
}


def frozen_observations(arm, scores, cases, requests):
    for key, phases in scores.items():
        step = f"step-{int(key):02}"
        for phase, entry in phases.items():
            directory = Path(entry["path"])
            assert digest(directory / "adapter.bin") == entry["adapter_sha256"]
            if phase == "forward-2":
                directory, = (forward.OUT / arm).glob(f"attempt-*/{step}")
            elif phase == "swapped-2":
                directory, = (baseline.swap.OUT / arm / f"target-{int(key):02}").glob(f"attempt-*/{step}")
            assert digest(directory / "observations.json") == entry["observations_sha256"]
            audit = read(directory / "audit.json")
            assert prior.evaluate(directory, cases, requests, audit) == {
                name: value for name, value in entry.items() if name != "path"
            }


def train(root, driver, reference, visits_path, protocol_sha):
    previous = sorted(root.glob("attempt-*"))
    complete = [attempt for attempt in previous if (attempt / "finished.json").is_file()]
    attempt, = complete if complete else [root / f"attempt-{len(previous) + 1:03}"]
    inputs = reference["inputs"]
    args = [driver, forward.semantic.MODEL, inputs["train"]["path"], inputs["instrument"]["path"],
            forward.semantic.INITIAL, visits_path, attempt]
    keep(attempt / "command.json", dict(argv=list(map(str, args)), protocol_sha256=protocol_sha))
    log = attempt / "train.log"
    if not complete:
        print(f"{root.name}: 50 control updates and 12 updates in reset branches", flush=True)
        with log.open("x", encoding="utf-8") as output:
            subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, timeout=1500, check=True)
    keep(attempt / "finished.json", dict(train_log_sha256=digest(log)))
    return attempt, fit._events(log)


def check_save(event, attempt, target, phase, depth):
    directory = attempt / f"target-{target}" / phase
    adapter = directory / "adapter.bin"
    assert event == dict(event="saved", target=target, phase=phase, step=depth, adapter=str(adapter))
    return directory, digest(adapter)


def check_branches(attempt, events, old_events, scores, visits, schedules, labels):
    setup = ("prefill", "prefill_recovery", "start")
    assert [e for e in events if e["event"] in setup] == [e for e in old_events if e["event"] in setup]
    control = [e for e in old_events if e["event"] == "update" and e["order"] == "forward"]
    assert [e["row"] for e in control] == visits
    records = iter(e for e in events if e["event"] not in setup)
    new_checkpoints = {}
    signature = baseline.update_signatures
    for archived_update in control:
        step = archived_update["step"]
        current = next(records)
        assert current == {**archived_update, "order": "control"}
        target = step + 2
        if target in STEPS:
            archive = scores[str(target)]
            _, common = check_save(next(records), attempt, target, "common", step)
            assert common == archive["common"]["adapter_sha256"]
            for order in ORDERS:
                reset = next(records)
                assert (reset["event"], reset["target"], reset["order"], reset["step"]) == (
                    "reset_first", target, order, step
                )
                assert 0 < reset["active_first_tensors"] <= reset["state_tensors"] == 56
                planned = visits if order == "forward" else schedules[str(target)]["visits"]
                old = control[step] if order == "forward" else next(
                    e for e in old_events if e["event"] == "update" and e["order"] == "swapped"
                    and e["target"] == target and e["step"] == step + 1
                )
                for offset in (1, 2):
                    update = next(records)
                    depth = step + offset
                    row = planned[depth - 1]
                    assert (update["event"], update["order"], update["target"], update["step"], update["row"]) == (
                        "update", order, target, depth, row
                    )
                    assert update["first_target_id"] == forward.TOKEN_IDS[labels[row - 1]]
                    assert math.isfinite(update["loss"])
                    if offset == 1:
                        assert signature([update]) == signature([old])
                    phase = f"{order}-{offset}"
                    new_checkpoints[target, phase] = check_save(next(records), attempt, target, phase, depth)
        elif step in STEPS:
            _, restored = check_save(next(records), attempt, step, "control-forward", step)
            assert restored == scores[str(step)]["forward-2"]["adapter_sha256"]
    assert next(records, None) is None
    assert len(new_checkpoints) == len(STEPS) * 4
    return new_checkpoints


def compare(reference, reset):
    ordinary = baseline.summarize(reference)
    cleared = baseline.summarize(dict(common=reference["common"], **reset))
    deltas = {
        phase: {key: value - ordinary["levels"][phase][key] for key, value in metrics.items()}
        for phase, metrics in cleared["levels"].items() if phase != "common"
    }
    return dict(ordinary=ordinary, reset=cleared, reset_minus_ordinary=deltas)


def main(out, driver):
    control_protocol, control = baseline.archive_at(baseline.OUT, BASELINE_PINS)
    forward_protocol, _ = baseline.archive_at(forward.OUT, baseline.swap.ARCHIVE_PINS)
    swap_protocol, _ = baseline.archive_at(baseline.swap.OUT, baseline.SWAP_PINS)
    for name, expected in (forward_protocol["files"] | swap_protocol["files"] | control_protocol["files"]).items():
        assert digest(Path(name)) == expected, name
    cases, requests, _, audits = prior.freeze(prior.OUT)
    eval_fields = ("context", "sampling", "max_new", "requests")
    eval_metadata = {key: control_protocol[key] for key in eval_fields}
    frozen_eval = read(prior.OUT / "protocol.json")
    assert eval_metadata == {key: frozen_eval[key] for key in eval_fields}
    labels = forward.labels_for(audits["original"])
    assert labels == forward.labels_for(audits["diverse"])
    _, visits = forward.make_schedule(baseline.swap.DRIVER, labels, audits)
    assert visits == control_protocol["forward_visits"]
    schedules = {str(step): baseline.swap.swap_tail(visits, labels, step) for step in STEPS}
    assert schedules == control_protocol["checkpoints"]
    for arm, scores in control["scores"].items():
        frozen_observations(arm, scores, cases, requests)

    keep(out / "protocol.json", dict(
        baseline=BASELINE_PINS, training_config=control_protocol["training_config"],
        intervention="zero first moment once at the shared prefix, preserving weights, second moment and step",
        forward_visits=visits, checkpoints=schedules, **eval_metadata,
        files={str(path.resolve()): digest(path) for path in (driver, SOURCE, Path(__file__))},
    ))
    visits_path = out / "visits.json"
    keep(visits_path, visits)
    protocol_sha = digest(out / "protocol.json")
    runs = {}
    for arm, reference in audits.items():
        attempt, events = train(out / arm, driver, reference, visits_path, protocol_sha)
        old_attempt, = (baseline.OUT / arm).glob("attempt-*")
        old_log = old_attempt / "train.log"
        assert digest(old_log) == read(old_attempt / "finished.json")["train_log_sha256"]
        baseline_scores = control["scores"][arm]
        branches = check_branches(attempt, events, fit._events(old_log), baseline_scores, visits, schedules, labels)
        runs[arm] = digest(attempt / "train.log"), branches
        print(f"{arm}: shared and restored adapters matched all six controls; checked each reset", flush=True)

    scores, comparisons = {}, {}
    for arm, (log_sha, branches) in runs.items():
        scores[arm], comparisons[arm] = {}, {}
        for step in STEPS:
            key = str(step)
            reset = {}
            for order in ORDERS:
                for offset in (1, 2):
                    phase = f"{order}-{offset}"
                    directory, adapter_sha = branches[step, phase]
                    audit = dict(adapter_sha256=adapter_sha, train_log_sha256=log_sha, target=step,
                                 step=step - 2 + offset, phase=phase, protocol_sha256=protocol_sha)
                    keep(directory / "audit.json", audit)
                    entry = prior.evaluate(directory, cases, requests, audit)
                    reset[phase] = dict(path=str(directory), **entry)
            scores[arm][key] = reset
            comparison = compare(control["scores"][arm][key], reset)
            comparisons[arm][key] = comparison
            print(arm, step, json.dumps(comparison), flush=True)
    keep(out / "results.json", dict(protocol_sha256=protocol_sha, scores=scores, comparisons=comparisons,
                                   tool_four=control["tool_four"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=OUT)
    args = cli.parse_args()
    out = args.output.resolve()
    main(out, out / "native-reset")
