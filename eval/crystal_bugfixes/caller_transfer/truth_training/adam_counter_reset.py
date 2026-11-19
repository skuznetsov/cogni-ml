"""Restart Adam once at each frozen two-update branch and compare with retained counters."""

import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
import math
from pathlib import Path
import subprocess
import sys

import both_moments_reset as retained

baseline, first, prior, fit = retained.baseline, retained.first, retained.prior, retained.fit
digest, read, keep = baseline.digest, baseline.read, baseline.keep
OUT = retained.OUT.with_name("adam-counter-reset-v1")
SOURCE = Path(__file__).with_name("train_adam_counter_reset.cr")
CONTROL_PINS = {
    "protocol.json": "d18102dba0032d4390c335cdbd6e36500b7eeb64f210e9f103007325598e7f00",
    "results.json": "2241478b47edc2f97bd706a9ce097d79203c5c06a1e6789ded84206c07e18e1a",
}
PHASES = tuple(f"{order}-{offset}" for order in first.ORDERS for offset in (1, 2))


def reference_plan(arm, control, scores, ordinary, cases, requests):
    attempt, = (retained.OUT / arm).glob("attempt-*/finished.json")
    attempt = attempt.parent
    log = attempt / "train.log"
    log_sha = digest(log)
    assert log_sha == read(attempt / "finished.json")["train_log_sha256"]
    events = fit._events(log)
    saves = {(e["target"], e["phase"]): e for e in events if e["event"] == "saved"}
    branches = []
    for target in baseline.STEPS:
        key = str(target)
        common_dir, common_sha = first.check_save(saves[target, "common"], attempt, target, "common", target - 2)
        assert common_sha == ordinary[key]["common"]["adapter_sha256"]
        for order in first.ORDERS:
            visits = control["forward_visits"] if order == "forward" else control["checkpoints"][key]["visits"]
            branches.append(dict(target=target, order=order, common=str(common_dir / "adapter.bin"),
                                 common_sha256=common_sha, rows=visits[target - 2:target]))
        assert set(scores[key]) == set(PHASES)
        for phase, entry in scores[key].items():
            step = target - 2 + int(phase[-1])
            directory, adapter_sha = first.check_save(saves[target, phase], attempt, target, phase, step)
            assert directory == Path(entry["path"]) and adapter_sha == entry["adapter_sha256"]
            assert digest(directory / "audit.json") == entry["audit_sha256"]
            assert digest(directory / "observations.json") == entry["observations_sha256"]
            audit = dict(adapter_sha256=adapter_sha, train_log_sha256=log_sha, target=target,
                         step=step, phase=phase, protocol_sha256=CONTROL_PINS["protocol.json"])
            assert read(directory / "audit.json") == audit
            assert prior.evaluate(directory, cases, requests, audit) == {
                key: value for key, value in entry.items() if key != "path"
            }
    return branches, events, log_sha


def train(root, driver, train_path, plan, protocol_sha):
    schedule = root / "branches.json"
    keep(schedule, plan)
    previous = sorted(root.glob("attempt-*"))
    complete = [attempt for attempt in previous if (attempt / "finished.json").is_file()]
    attempt, = complete if complete else [root / f"attempt-{len(previous) + 1:03}"]
    args = [driver, retained.forward.semantic.MODEL, train_path, schedule, attempt]
    keep(attempt / "command.json", dict(argv=list(map(str, args)), protocol_sha256=protocol_sha))
    log = attempt / "train.log"
    if not complete:
        print(f"{root.name}: training six two-update branches from archived factors", flush=True)
        with log.open("x", encoding="utf-8") as output:
            subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, timeout=900, check=True)
    return attempt, log


def check_branches(attempt, events, old_events, plan, labels, control):
    prefill = [event for event in old_events if event["event"] == "prefill"][:len(labels)]
    assert len(prefill) == len(labels) == 32 and events[:len(prefill)] == prefill
    config = control["training_config"]
    assert events[len(prefill)] == dict(event="start", model_sha256=fit.MODEL_SHA,
                                       seed=config["seed"], examples=len(prefill), blocks=config["layers"])
    updates = {(e["target"], e["order"], e["step"]): e for e in old_events if e["event"] == "update"}
    records = iter(events[len(prefill) + 1:])
    checkpoints = {}
    for branch in plan:
        target, order = branch["target"], branch["order"]
        assert next(records) == dict(event="reset_adam", target=target, order=order, step=target - 2,
                                     adam_step=0, common_sha256=branch["common_sha256"])
        for offset, row in enumerate(branch["rows"], 1):
            step = target - 2 + offset
            update = next(records)
            assert (update["event"], update["target"], update["order"], update["step"],
                    update["row"], update["adam_step"]) == ("update", target, order, step, row, offset)
            assert update["first_target_id"] == retained.forward.TOKEN_IDS[labels[row - 1]]
            assert math.isfinite(update["loss"])
            if offset == 1:
                assert baseline.update_signatures([update]) == baseline.update_signatures(
                    [updates[target, order, step]]
                )
            phase = f"{order}-{offset}"
            checkpoints[target, phase] = first.check_save(next(records), attempt, target, phase, step)
    assert next(records, None) is None
    assert len(checkpoints) == len(baseline.STEPS) * len(PHASES)
    return checkpoints


def factors(path):
    with path.open("rb") as stream:
        assert stream.readline() == b"COGNI-QWEN35-QLORA-2\n"
        assert stream.readline().decode().strip() == fit.MODEL_SHA
        assert stream.readline().decode().strip() == ",".join(map(str, fit.LAYERS))
        values = array("f")
        values.frombytes(stream.read())
    if sys.byteorder != "little":
        values.byteswap()
    assert len(values) == 1_277_952
    return values


def step_norms(common, entries):
    norms = {}
    initial = factors(common)
    for order in first.ORDERS:
        phases = (f"{order}-1", f"{order}-2")
        weights = [initial, *(factors(Path(entries[phase]["path"]) / "adapter.bin") for phase in phases)]
        for phase, (before, after) in zip(phases, pairwise(weights), strict=True):
            norms[phase] = math.sqrt(math.fsum((y - x) ** 2 for x, y in zip(before, after, strict=True)))
    assert all(math.isfinite(value) and value > 0 for value in norms.values())
    return norms


def run(out, driver):
    control, archive = baseline.archive_at(retained.OUT, CONTROL_PINS)
    assert control["baseline"] == first.BASELINE_PINS
    for name, expected in control["files"].items():
        assert digest(Path(name)) == expected, name
    ordinary_protocol, ordinary = baseline.archive_at(baseline.OUT, first.BASELINE_PINS)
    cases, requests, _, audits = prior.freeze(prior.OUT)
    frozen_eval = read(prior.OUT / "protocol.json")
    eval_keys = ("context", "sampling", "max_new", "requests")
    eval_metadata = {key: control[key] for key in eval_keys}
    assert eval_metadata == {key: frozen_eval[key] for key in eval_keys}
    for key in (*eval_keys, "training_config", "checkpoints", "forward_visits"):
        assert control[key] == ordinary_protocol[key], key
    labels = retained.forward.labels_for(audits["original"])
    assert labels == retained.forward.labels_for(audits["diverse"])
    schedules = {str(step): baseline.swap.swap_tail(control["forward_visits"], labels, step)
                 for step in baseline.STEPS}
    assert schedules == control["checkpoints"]
    plans = {arm: reference_plan(arm, control, archive["scores"][arm], ordinary["scores"][arm], cases, requests)
             for arm in audits}
    inputs = {arm: audit["inputs"]["train"] for arm, audit in audits.items()}
    for item in inputs.values():
        assert digest(Path(item["path"])) == item["sha256"]
    repo = SOURCE.resolve().parents[4]
    sources = (driver, SOURCE, Path(__file__), Path(retained.__file__),
               repo / "src/ml/optim/adam.cr", repo / "src/ml/gguf/qwen35_qlora.cr", repo / "build/bridge.o")
    protocol = dict(
        retained_counter=CONTROL_PINS, baseline=first.BASELINE_PINS,
        intervention="load the frozen common factors and create fresh Adam state once per branch",
        training_config=control["training_config"], train_inputs=inputs, model_sha256=fit.MODEL_SHA,
        archive_log_sha256={arm: log_sha for arm, (_, _, log_sha) in plans.items()},
        branches={arm: plan for arm, (plan, _, _) in plans.items()},
        forward_visits=control["forward_visits"], checkpoints=schedules, **eval_metadata,
        files={str(path.resolve()): digest(path) for path in sources},
    )
    keep(out / "protocol.json", protocol)
    protocol_sha = digest(out / "protocol.json")

    runs = {}
    for arm, (plan, old_events, _) in plans.items():
        attempt, log = train(out / arm, driver, inputs[arm]["path"], plan, protocol_sha)
        checkpoints = check_branches(attempt, fit._events(log), old_events, plan, labels, control)
        log_sha = digest(log)
        keep(attempt / "finished.json", dict(train_log_sha256=log_sha))
        runs[arm] = log_sha, checkpoints
        print(f"{arm}: six restarts checked, counters 0/1/2 and all first losses matched", flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {
            (arm, step, phase): pool.submit(
                retained.evaluate, arm, step, phase, directory, adapter_sha, log_sha, protocol_sha, cases, requests
            )
            for arm, (log_sha, checkpoints) in runs.items()
            for (step, phase), (directory, adapter_sha) in checkpoints.items()
        }
        scores, norms = {}, {}
        for arm in runs:
            scores[arm], norms[arm] = {}, {}
            for step in baseline.STEPS:
                key = str(step)
                updated = {phase: tasks[arm, step, phase].result() for phase in PHASES}
                scores[arm][key] = updated
                common = Path(ordinary["scores"][arm][key]["common"]["path"]) / "adapter.bin"
                norms[arm][key] = dict(retained_counter=step_norms(common, archive["scores"][arm][key]),
                                       reset_counter=step_norms(common, updated))
    keep(out / "results.json", dict(protocol_sha256=protocol_sha, scores=scores, step_norms=norms,
                                   tool_four=archive["tool_four"]))
    print("Compared 24 matched checkpoint pairs.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=OUT)
    output = cli.parse_args().output.resolve()
    run(output, output / "native-reset")
