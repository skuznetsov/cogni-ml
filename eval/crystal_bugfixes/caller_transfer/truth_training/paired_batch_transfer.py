"""Contrast true counterfactual batches with a label-balanced derangement."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import random
import subprocess
import time

import experiment
import fit
import paired_batch_fixtures
import probe


native, runner, transfer = experiment.native, experiment.runner, experiment.transfer
digest, keep, read = runner.digest, native.keep, native.read
REFERENCE = experiment.OUT
OUT = REFERENCE / "paired-batch-transfer-v2"
REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ORIGINAL_PROTOCOL = "b2c7d180def9c26c8b87aabcedc965a5b60e4c044e1bbc84f87cad02c885fb10"
FIRST_IDS = {"true, false": 1802, "false, true": 3721}
PAIR_SEED = 42
STEPS = 16
TARGETS = 128
ARMS = ("real", "shuffled")
SCORE_FIELDS = ("stage", "compiled", "target", "regressions", "passed", "total", "failed_specs")


def original():
    reference = REFERENCE / "protocol.json"
    assert digest(reference) == ORIGINAL_PROTOCOL
    archived = read(reference)
    for name in ("train.jsonl", "cases.json", "train-counts.json"):
        assert digest(REFERENCE / name) == archived["inputs"][name], name
    assert digest(runner.MODEL) == fit.MODEL_SHA
    assert digest(experiment.INITIAL) == fit.INITIAL_SHA
    assert digest(runner.SAMPLED_NATIVE) == archived["pins"]["sampler"]
    assert archived["sampling"] == transfer.SAMPLING and transfer.MAX_NEW == 128
    cases = [row for row in read(REFERENCE / "cases.json") if row["split"] == "train"]
    experiment.validate(cases)
    cases = {row["id"]: row for row in cases}
    counts = read(REFERENCE / "train-counts.json")
    rows = fit._rows(REFERENCE / "train.jsonl")
    pairs = {}
    for number, (row, count) in enumerate(zip(rows, counts, strict=True), 1):
        case = cases[row["id"]]
        target = ", ".join(map(experiment.literal, case["truth"]))
        assert row["messages"] == case["messages"] + [dict(role="assistant", content=target)]
        assert count["row"] == number and count["id"] == row["id"]
        assert count["target_tokens"] == 4 and count["input_tokens"] <= 4096
        count["first_target_id"] = FIRST_IDS[target]
        key = case["family"], case["public_zero"], tuple(case["order"])
        pair = pairs.setdefault(key, {})
        assert target not in pair
        pair[target] = number
    assert len(pairs) == STEPS and all(pair.keys() == FIRST_IDS.keys() for pair in pairs.values())
    assert len({row["id"] for row in rows}) == len(rows) == 32
    # Keep true anchors in original row order. Shuffle only their false partners.
    real = sorted([[pair["true, false"], pair["false, true"]] for pair in pairs.values()])
    false_rows = [batch[1] for batch in real]
    shuffled = false_rows.copy()
    rng = random.Random(PAIR_SEED)
    while any(a == b for a, b in zip(false_rows, shuffled, strict=True)):
        rng.shuffle(shuffled)
    shuffled = [[batch[0], partner] for batch, partner in zip(real, shuffled, strict=True)]
    return archived, counts, dict(real=real, shuffled=shuffled)


def case_id(row):
    return f"{row['family']}/{row['variant']}"


def observed(score):
    return {field: score.get(field) for field in SCORE_FIELDS}


def controls(out, rows):
    families = sorted({row["family"] for row in rows})
    assert len(rows) == 8 and len(families) == 4
    assert Counter(row["kind"] for row in rows) == {"flip": 4, "invariant": 4}
    results = {}
    for family in families:
        left, right = sorted((row for row in rows if row["family"] == family),
                             key=lambda row: row["variant"])
        assert (left["variant"], right["variant"]) == ("a", "b")
        assert left["template"] != right["template"]
        for field in ("kind", "issue", "fixture", "target_tests", "broken"):
            assert left[field] == right[field], (family, field)
        for row, peer in ((left, right), (right, left)):
            key = case_id(row)
            names = native.data.test_names(row["fixture"])
            assert len(set(names)) == len(names)
            assert set(row["target_tests"]) < set(names) and len(names) >= 3
            assert row["template"].count("<HOLE>") == 1
            patches = dict(teacher=row["teacher"], broken=row["broken"], opposite=peer["teacher"])
            if "regression_trap" in row:
                patches["regression_trap"] = row["regression_trap"]
            score = {name: observed(probe.score_patch(row, patch, out / "controls" / key / name))
                     for name, patch in patches.items()}
            assert score["teacher"]["stage"] == "pass", (key, score)
            assert score["broken"]["compiled"] and score["broken"]["target"] is False, (key, score)
            if row["kind"] == "flip":
                assert score["opposite"]["stage"] == "tests_failed", (key, score)
            else:
                assert score["opposite"]["stage"] == "pass", (key, score)
            if "regression_trap" in score:
                trap = score["regression_trap"]
                assert trap["compiled"] and trap["target"] and trap["regressions"] is False, (key, score)
            results[key] = score
    return results


def freeze(out, driver):
    archived, counts, plans = original()
    rows = paired_batch_fixtures.rows()
    keep(out / "cases.json", rows)
    keep(out / "controls.json", controls(out, rows))
    requests = {}
    for row in rows:
        key = case_id(row)
        user = f"{row['issue']}\n\n{probe.source(row['template'])}\n\n{probe.PATCH}"
        request, tokens = native.checked_request(probe.messages(user), "repair")
        requests[key] = dict(request=request, prefix_tokens=tokens)
    keep(out / "requests.json", requests)
    for arm, batches in plans.items():
        assert sorted(row for batch in batches for row in batch) == list(range(1, 33))
        assert all([counts[row - 1]["first_target_id"] for row in batch] == [1802, 3721]
                   for batch in batches)
        keep(out / f"{arm}-batches.json", batches)
    checker_path, checker = fit._checker(REPO)
    files = [
        Path(__file__), Path(paired_batch_fixtures.__file__), HERE / "train_paired_batch_sft.cr",
        driver, checker_path, Path(probe.__file__), Path(native.__file__), Path(transfer.__file__),
        Path(runner.__file__), Path(native.data.__file__), native.data.TOKEN_COUNTER,
        Path(transfer.score.__code__.co_filename), REPO / "src/ml/gguf/qwen35_qlora.cr",
        REPO / "src/ml/optim/adam.cr",
    ]
    inputs = ("cases.json", "controls.json", "requests.json",
              "real-batches.json", "shuffled-batches.json")
    keep(out / "protocol.json", dict(
        original_protocol=ORIGINAL_PROTOCOL,
        original_files={key: archived["inputs"][key] for key in ("train.jsonl", "train-counts.json", "cases.json")},
        model_sha256=fit.MODEL_SHA, initial_adapter_sha256=fit.INITIAL_SHA, sampler=archived["pins"]["sampler"],
        files={str(path.resolve()): digest(path) for path in files},
        inputs={name: digest(out / name) for name in inputs},
        stages="one native full-SFT mean-gradient Adam update per batch",
        optimizer=dict(name="Adam", learning_rate=0.0001, betas=[0.9, 0.999],
                       epsilon=1e-8, initial_step=0, reset=False),
        examples=32, updates=STEPS, target_tokens=TARGETS, targets_per_example=4,
        seed=fit.SEED, pair_seed=PAIR_SEED, rank=fit.RANK, layers=fit.LAYERS,
        projections=fit.CONFIG["projections"],
        evaluation=dict(context=4096, max_new=transfer.MAX_NEW, sampling=transfer.SAMPLING,
                        samples_per_variant=1, cases=list(requests), baseline="tool-four"),
        crystal=subprocess.check_output(["crystal", "--version"], text=True).splitlines()[0],
    ))
    print(f"Frozen {len(rows)} repair variants and both training schedules", flush=True)
    return rows, requests, counts, plans, checker


def train(out, driver, arm):
    root = out / "training" / arm
    root.mkdir(parents=True, exist_ok=True)
    launches = sorted(root.glob("attempt-*"))
    pending = [path for path in launches if not (path / "exit.json").is_file()]
    if pending:
        raise RuntimeError(f"Incomplete training attempt: {pending[0]}")
    successful = [path for path in launches if read(path / "exit.json")["returncode"] == 0]
    assert len(successful) <= 1, root
    attempt = successful[0] if successful else root / f"attempt-{len(launches) + 1:03}"
    batch_file = out / f"{arm}-batches.json"
    inputs = [driver, runner.MODEL, REFERENCE / "train.jsonl", experiment.INITIAL, batch_file]
    command = [str(path.resolve()) for path in (*inputs, attempt)]
    keep(attempt / "launch.json", dict(
        command=command, protocol=digest(out / "protocol.json"),
        inputs={str(path.resolve()): digest(path) for path in inputs},
    ))
    log = attempt / "train.log"
    if not successful:
        print(f"{arm}: training {STEPS} batches", flush=True)
        start = time.monotonic()
        returncode = "timeout"
        try:
            with log.open("x", encoding="utf-8") as stream:
                returncode = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                            timeout=1800, check=False).returncode
        finally:
            keep(attempt / "exit.json", dict(returncode=returncode, seconds=time.monotonic() - start))
        if returncode != 0:
            raise RuntimeError(f"{arm} native trainer returned {returncode}; see {log}")
    return attempt


def audit_training(attempt, arm, counts, batches, checker):
    log, adapter = attempt / "train.log", attempt / "adapter.bin"
    events = fit._events(log)
    kinds = [event["event"] for event in events]
    assert kinds == ["start"] + ["prefill"] * 32 + ["update"] * STEPS + ["saved"], (arm, kinds)
    start = events[0]
    for key, expected in dict(model_sha256=fit.MODEL_SHA, initial_adapter_sha256=fit.INITIAL_SHA,
                              seed=fit.SEED, blocks=fit.LAYERS, examples=32).items():
        assert start.get(key) == expected, (arm, key, start)
    fields = ("row", "first_target_id", "prefix_tokens", "target_tokens")
    for actual, expected in zip(events[1:33], counts, strict=True):
        assert all(actual.get(key) == expected[key] for key in fields), (arm, actual, expected)
    updates = events[33:-1]
    for step, (event, batch) in enumerate(zip(updates, batches, strict=True), 1):
        assert event["step"] == step and event["adam_after"] == step and event["rows"] == batch, (arm, event)
        assert event["first_target_ids"] == [1802, 3721] and event["target_tokens"] == [4, 4]
        assert len(event["losses"]) == 2 and all(map(math.isfinite, event["losses"]))
    assert sum(sum(event["target_tokens"]) for event in updates) == TARGETS
    assert events[-1] == dict(event="saved", adapter=str(adapter.resolve()),
                              updates=STEPS, target_tokens=TARGETS)
    with redirect_stdout(io.StringIO()):
        adapter_hash = checker.check_adapter(adapter)
    before, after = map(fit._load_factors, (experiment.INITIAL, adapter))
    offset, changed = 0, []
    for layer in fit.LAYERS:
        for name, width, height in fit.PROJECTIONS:
            mid, end = offset + fit.RANK * width, offset + fit.RANK * (width + height)
            a = sum(x != y for x, y in zip(before[offset:mid], after[offset:mid], strict=True))
            b = sum(x != y for x, y in zip(before[mid:end], after[mid:end], strict=True))
            assert a and b, (arm, layer, name)
            changed.append(dict(layer=layer, projection=name, a=a, b=b))
            offset = end
    assert offset == len(before) == len(after)
    assert adapter_hash != fit.INITIAL_SHA
    keep(attempt / "audit.json", dict(
        adapter_sha256=adapter_hash, train_log_sha256=digest(log), updates=STEPS,
        target_tokens=TARGETS, visited_rows=[row for batch in batches for row in batch],
        changes=changed,
    ))
    print(f"{arm}: verified {STEPS} Adam updates, {TARGETS} target tokens", flush=True)
    return adapter


def evaluate(out, arm, adapter, rows, requests):
    folder = out / "runs" / arm
    keep(folder / "pins.json", dict(adapter=digest(adapter), protocol=digest(out / "protocol.json")))
    command = [str(runner.SAMPLED_NATIVE), "generate", str(runner.MODEL), str(adapter),
               str(transfer.MAX_NEW), str(len(fit.LAYERS))]
    worker = recovery = None
    results = []
    try:
        for row in rows:
            key = case_id(row)
            loc = folder / key
            if worker is None and not (loc / "generation.json").exists():
                _, runtime, recovery = runner.load_runtime()
                worker = runner.ArmWorker(runtime.ev.evaluator, command, folder / "worker", folder / "worker.log", 240)
            response = native.sample(worker, recovery, requests[key]["request"], loc)
            answer = probe.normalize(response)
            score = observed(probe.score_patch(row, answer, loc / "checks"))
            if answer is None:
                score["stage"] = "length" if response["stop"] == "limit" else "format_failed"
            results.append(dict(case=key, family=row["family"], kind=row["kind"],
                                answer=answer, response=response["answer"], tokens=response["tokens"],
                                recovered=read(loc / "generation.json")["recovered"], **score))
            print(f"{arm} {key}: {score['stage']}", flush=True)
    finally:
        if worker is not None:
            assert worker.close() == 0
    keep(folder / "scores.json", results)
    return results


def summarize(rows):
    return {
        "variants": len(rows),
        "stages": dict(Counter(row["stage"] for row in rows)),
        "compiled": sum(row["compiled"] is True for row in rows),
        "target": sum(row["target"] is True for row in rows),
        "regressions": sum(row["regressions"] is True for row in rows),
        "full": sum(row["stage"] == "pass" for row in rows),
        "complete_families": [family for family in sorted({row["family"] for row in rows})
                              if all(row["stage"] == "pass" for row in rows if row["family"] == family)],
    }


def run(out, driver, preflight_only):
    rows, requests, counts, plans, checker = freeze(out, driver)
    if preflight_only:
        return
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {arm: pool.submit(train, out, driver, arm) for arm in ARMS}
        attempts = {arm: job.result() for arm, job in jobs.items()}
    children = {arm: audit_training(attempt, arm, counts, plans[arm], checker)
                for arm, attempt in attempts.items()}
    adapters = dict(tool_four=experiment.INITIAL, **children)
    adapter_hashes = {arm: digest(path) for arm, path in adapters.items()}
    assert len(set(adapter_hashes.values())) == 3
    scores = {arm: evaluate(out, arm, path, rows, requests) for arm, path in adapters.items()}
    summary = {arm: summarize(arm_rows) for arm, arm_rows in scores.items()}
    keep(out / "results.json", dict(protocol=digest(out / "protocol.json"),
                                    adapters=adapter_hashes, summary=summary, rows=scores))
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("driver", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.out.resolve(), args.driver.resolve(), args.preflight_only)
