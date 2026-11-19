"""Run and score native averaged-gradient full-SFT checkpoints."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import first_moment_reset as reset
import per_update_order as ordinary


forward, prior = ordinary.forward, ordinary.prior
semantic, fit = forward.semantic, ordinary.fit
digest, read, keep = ordinary.digest, ordinary.read, ordinary.keep
OUT = semantic.OUT / "mean-sft-step-v1"
SOURCE = Path(__file__).with_name("train_mean_sft_step.cr")
STEPS = ordinary.STEPS
ARCHIVE_PINS = reset.BASELINE_PINS
EVAL_WORKERS = 2


def _expect(condition, message):
    if not condition:
        raise ValueError(message)


def _phase_path(arm, target, phase, entry):
    """Resolve the immutable source checkpoint for an ordinary control phase."""
    if phase in ("common", "forward-1", "swapped-1"):
        return Path(entry["path"])
    if phase == "forward-2":
        path, = (forward.OUT / arm).glob(f"attempt-*/step-{target:02}")
        return path
    if phase == "swapped-2":
        path, = (ordinary.swap.OUT / arm / f"target-{target:02}").glob(
            f"attempt-*/step-{target:02}"
        )
        return path
    raise ValueError(f"unknown ordinary phase: {phase}")


def _verify_manifest(files):
    for name, expected in files.items():
        path = Path(name)
        _expect(path.is_file(), f"pinned file is missing: {path}")
        _expect(digest(path) == expected, f"pinned file changed: {path}")


def _verify_archived_control(arm, target, phase, entry, cases, baseline_log_sha):
    path = _phase_path(arm, target, phase, entry)
    adapter, audit_file, observations_file = (path / "adapter.bin", path / "audit.json",
                                               path / "observations.json")
    _expect(digest(adapter) == entry["adapter_sha256"], f"{arm}/{target}/{phase} adapter changed")
    _expect(digest(audit_file) == entry["audit_sha256"], f"{arm}/{target}/{phase} audit changed")
    _expect(digest(observations_file) == entry["observations_sha256"],
            f"{arm}/{target}/{phase} observations changed")
    audit = read(audit_file)
    _expect(audit["adapter_sha256"] == entry["adapter_sha256"], "control audit adapter mismatch")
    depth = target - 2 + {"common": 0, "forward-1": 1, "swapped-1": 1,
                          "forward-2": 2, "swapped-2": 2}[phase]
    if phase in ("common", "forward-1", "swapped-1"):
        _expect(audit["target"] == target and audit["phase"] == phase,
                "control audit phase mismatch")
        _expect(audit["step"] == depth, f"{arm}/{target}/{phase} has wrong optimizer depth")
    observations = read(observations_file)
    _expect(observations["adapter_sha256"] == entry["adapter_sha256"],
            "control observations adapter mismatch")
    _expect(prior.metrics(cases, observations) == entry["score"],
            f"{arm}/{target}/{phase} archived score does not recompute")
    if phase in ("common", "forward-1", "swapped-1"):
        _expect(audit["train_log_sha256"] == baseline_log_sha,
                f"{arm}/{target}/{phase} audit has wrong branch log")
    return dict(path=str(path), adapter_sha256=entry["adapter_sha256"],
                audit_sha256=entry["audit_sha256"],
                observations_sha256=entry["observations_sha256"], score=entry["score"])


def _prepare(out, driver):
    """Validate all frozen inputs and read archived controls without writing to them."""
    out.mkdir(parents=True, exist_ok=True)
    protocol, archive = ordinary.archive_at(ordinary.OUT, ARCHIVE_PINS)
    forward_protocol, _ = ordinary.archive_at(forward.OUT, ordinary.swap.ARCHIVE_PINS)
    swap_protocol, _ = ordinary.archive_at(ordinary.swap.OUT, ordinary.SWAP_PINS)
    _verify_manifest(protocol["files"])
    _verify_manifest(forward_protocol["files"] | swap_protocol["files"])

    for name, expected in semantic.CONTROL_PINS.items():
        _expect(digest(semantic.REFERENCE / name) == expected, f"semantic baseline changed: {name}")
    for name, expected in prior.ARCHIVE_PINS.items():
        _expect(digest(semantic.OUT / name) == expected, f"checkpoint archive changed: {name}")
    semantic_protocol = read(semantic.OUT / "protocol.json")
    _verify_manifest(semantic_protocol["files"])
    for name, expected in semantic_protocol["inputs"].items():
        _expect(digest(semantic.OUT / name) == expected, f"semantic input changed: {name}")
    checkpoint_protocol = read(prior.OUT / "protocol.json")

    visits_path = ordinary.OUT / "visits.json"
    visits = read(visits_path)
    _expect(visits == protocol["forward_visits"], "archived forward visits changed")
    _expect(len(visits) == fit.UPDATES == 50, "expected the archived 50-row visit sequence")
    cases = read(semantic.OUT / "cases.json")
    _expect(len(cases) == 32 and len({row["pair"] for row in cases}) == 16,
            "expected the archived 32-row, 16-pair holdout")
    _expect(sum(row["kind"] == "flip" for row in cases) == 16,
            "expected eight two-row flip pairs")
    requests = [dict(id="repair", messages=row["messages"], sampling=semantic.SAMPLING)
                for row in cases]
    request_pins = [semantic.experiment.transfer.fingerprint(request) for request in requests]
    _expect(request_pins == protocol["requests"] == checkpoint_protocol["requests"],
            "evaluation requests differ from the archived protocol")

    audits = {arm: read(root / "training/audit.json") for arm, root in prior.ROOTS.items()}
    _expect(set(audits) == {"original", "diverse"}, "unexpected training arms")
    _expect(audits["original"]["visits"] == audits["diverse"]["visits"],
            "the archived training arms do not share the same shuffled visits")
    _expect({arm: digest(prior.ROOTS[arm] / "training/audit.json") for arm in audits}
            == protocol["archive_audits"], "frozen training audits changed")
    labels = forward.labels_for(audits["original"])
    _expect(labels == forward.labels_for(audits["diverse"]), "training labels differ by arm")
    for arm, audit in audits.items():
        for item in audit["inputs"].values():
            _expect(digest(Path(item["path"])) == item["sha256"],
                    f"{arm} training input changed: {item['path']}")
        _expect(audit["config"] == fit.CONFIG, f"{arm} full-SFT configuration changed")
        _expect(len(audit["visits"]) == fit.UPDATES, f"{arm} archived visit count changed")
        _expect(audit["train_rows"] == 32 and audit["supervised_target_tokens_for_updates"] == 200,
                f"{arm} training scope changed")
        _expect(audit["updated_projections"] == 28 and audit["updated_factors"] == 56,
                f"{arm} archived full-SFT scope changed")
        _expect(not audit["holdout_supplied"], f"{arm} training used holdout rows")

    _expect(digest(semantic.MODEL) == fit.MODEL_SHA, "pinned Qwen model changed")
    _expect(digest(semantic.INITIAL) == fit.INITIAL_SHA, "pinned initial adapter changed")
    repo = Path(__file__).resolve().parents[4]
    checker_path, checker = fit._checker(repo)
    checker.check_adapter(semantic.INITIAL)

    baseline_controls, baseline_logs = {}, {}
    for arm in audits:
        attempt, = ordinary.OUT.joinpath(arm).glob("attempt-*")
        log = attempt / "train.log"
        finished = read(attempt / "finished.json")
        log_sha = digest(log)
        _expect(finished["train_log_sha256"] == log_sha, f"{arm} ordinary control log changed")
        baseline_logs[arm] = (log, log_sha, fit._events(log))
        baseline_controls[arm] = {}
        for target in STEPS:
            key = str(target)
            baseline_controls[arm][key] = {}
            for phase, entry in archive["scores"][arm][key].items():
                baseline_controls[arm][key][phase] = _verify_archived_control(
                    arm, target, phase, entry, cases, log_sha
                )

    orientations = {}
    for target in STEPS:
        rows = visits[target - 2:target]
        pair = [labels[row - 1] for row in rows]
        expected = ["false", "true"] if target == 25 else ["true", "false"]
        _expect(pair == expected, f"forward pair orientation changed at target {target}: {pair}")
        orientations[str(target)] = dict(
            forward="->".join(pair), swapped="->".join(reversed(pair)), rows=rows,
        )

    source_paths = [Path(__file__), SOURCE, Path(ordinary.__file__), Path(ordinary.SOURCE),
                    Path(forward.__file__), Path(ordinary.swap.__file__),
                    Path(prior.__file__), Path(fit.__file__), checker_path,
                    semantic.MODEL, semantic.INITIAL, semantic.SAMPLER, semantic.READOUT,
                    visits_path, semantic.OUT / "cases.json"]
    source_paths.extend(Path(item["path"]) for audit in audits.values()
                        for item in audit["inputs"].values())
    source_paths = sorted({path.resolve() for path in source_paths})
    _expect(Path(driver).is_file(), f"native driver is missing: {driver}")
    files = {str(path): digest(path) for path in (*source_paths, Path(driver).resolve())}
    crystal = shutil.which("crystal")
    _expect(crystal is not None, "Crystal compiler was not found")
    toolchain = dict(crystal_path=str(Path(crystal).resolve()), crystal_sha256=digest(crystal),
                     crystal_version=subprocess.check_output([crystal, "--version"], text=True).splitlines()[0])

    protocol_value = dict(
        archive_pins=ARCHIVE_PINS,
        forward_archive=ordinary.swap.ARCHIVE_PINS,
        swapped_archive=ordinary.SWAP_PINS,
        archived_visits=dict(path=str(visits_path.resolve()), sha256=digest(visits_path), visits=visits),
        archived_training_audits=protocol["archive_audits"],
        training_config=protocol["training_config"],
        evaluation=dict(cases_sha256=digest(semantic.OUT / "cases.json"), requests=request_pins,
                        sampling=protocol["sampling"], context=protocol["context"],
                        max_new=protocol["max_new"], rows=len(cases), pairs=16,
                        primary="complete sampled truth-table pairs"),
        pair_orientations=orientations,
        metric_definition=("For each flip pair, l_T and l_F are raw true-minus-false logits "
                           "on the true-required and false-required rows; b=(l_T+l_F)/2, "
                           "s=(l_T-l_F)/2; report means over eight pairs."),
        controls={arm: {target: {phase: {key: value[key] for key in
                                      ("adapter_sha256", "audit_sha256", "observations_sha256")}
                                 for phase, value in phases.items()}
                       for target, phases in targets.items()}
                  for arm, targets in baseline_controls.items()},
        model_sha256=fit.MODEL_SHA, initial_adapter_sha256=fit.INITIAL_SHA,
        lora_scope=dict(rank=fit.RANK, layers=fit.LAYERS,
                        projections=[name for name, _, _ in fit.PROJECTIONS],
                        factors=56, frozen_base="Qwen3.5-2B GGUF"),
        toolchain=toolchain,
        files=files,
    )
    keep(out / "protocol.json", protocol_value)
    return dict(protocol=protocol_value, protocol_sha256=digest(out / "protocol.json"),
                visits=visits, visits_path=visits_path, cases=cases, requests=requests,
                audits=audits, labels=labels, orientations=orientations,
                baseline_controls=baseline_controls, baseline_logs=baseline_logs)


def _launch_receipt(attempt, argv, cwd):
    launch_path = attempt / "launch.json"
    expected = dict(argv=list(map(str, argv)), cwd=str(cwd.resolve()),
                    inputs={str(Path(arg).resolve()): digest(arg) for arg in argv[:-1]})
    if launch_path.exists():
        _expect(read(launch_path) == expected, f"{attempt.name} launch receipt differs from protocol inputs")
    else:
        keep(launch_path, expected)
    return expected


def _complete_attempts(root):
    complete = []
    for attempt in sorted(root.glob("attempt-*")):
        exit_path = attempt / "exit.json"
        if not exit_path.exists():
            continue
        status = read(exit_path)
        _expect(status.get("returncode") == 0, f"{attempt} native training failed")
        _expect(isinstance(status.get("seconds"), (int, float)) and status["seconds"] >= 0,
                f"{attempt} has invalid elapsed time")
        _expect((attempt / "train.log").is_file(), f"{attempt} has no training log")
        _expect(all((attempt / f"target-{target}" / "mean" / "adapter.bin").is_file()
                    for target in STEPS), f"{attempt} is missing a mean checkpoint")
        complete.append(attempt)
    return complete


def _train_one(out, arm, driver, state):
    root = out / arm
    root.mkdir(parents=True, exist_ok=True)
    complete = _complete_attempts(root)
    if complete:
        _expect(len(complete) == 1, f"expected one completed attempt in {root}")
        attempt = complete[0]
    else:
        pending = [p for p in root.glob("attempt-*") if (p / "launch.json").exists()
                   and not (p / "exit.json").exists()]
        if pending:
            raise ValueError(f"native training is still running in {pending[0]}")
        previous = sorted(root.glob("attempt-*"))
        attempt = root / f"attempt-{len(previous) + 1:03}"
        attempt.mkdir(parents=True, exist_ok=True)

    argv = [str(Path(driver).resolve()), str(semantic.MODEL.resolve()),
            str(Path(state["audits"][arm]["inputs"]["train"]["path"]).resolve()),
            str(semantic.INITIAL.resolve()), str(state["visits_path"].resolve()), str(attempt.resolve())]
    launch = _launch_receipt(attempt, argv, Path(__file__).resolve().parents[4])
    exit_path, log_path = attempt / "exit.json", attempt / "train.log"
    if not exit_path.exists():
        _expect(not log_path.exists(), f"refusing to overwrite incomplete native log: {log_path}")
        print(f"{arm}: starting 48 forward updates and 3 mean-gradient checkpoints", flush=True)
        import time
        start = time.monotonic()
        with log_path.open("x", encoding="utf-8") as output:
            result = subprocess.run(argv, cwd=launch["cwd"], stdout=output, stderr=subprocess.STDOUT,
                                    timeout=1800, check=False)
        keep(exit_path, dict(returncode=result.returncode, seconds=time.monotonic() - start))
        _expect(result.returncode == 0, f"native trainer returned {result.returncode} for {arm}")

    _expect(read(exit_path).get("returncode") == 0, f"native training failed for {arm}")
    log_sha = digest(log_path)
    keep(attempt / "finished.json", dict(train_log_sha256=log_sha))
    events = fit._events(log_path)
    _validate_training_events(arm, events, state, attempt)
    return attempt, events, log_sha


def _validate_training_events(arm, events, state, attempt):
    expected_prefill = [event for event in state["baseline_logs"][arm][2]
                        if event["event"] == "prefill" and event.get("file") == "train.jsonl"]
    observed_prefill = [event for event in events if event["event"] == "prefill"]
    _expect(len(observed_prefill) == 32 and observed_prefill == expected_prefill,
            f"{arm} prefill records differ from the archived 32 training rows")
    starts = [event for event in events if event["event"] == "start"]
    _expect(starts == [dict(event="start", model_sha256=fit.MODEL_SHA, seed=42,
                            examples=32, blocks=fit.LAYERS)], f"{arm} start event mismatch")

    updates = [event for event in events if event["event"] == "update"]
    archived_updates = [event for event in state["baseline_logs"][arm][2]
                        if event["event"] == "update" and event.get("order") == "forward"][:48]
    _expect(len(updates) == 48 and [event["step"] for event in updates] == list(range(1, 49)),
            f"{arm} did not emit exactly the first 48 forward updates")
    _expect(ordinary.update_signatures(updates) == ordinary.update_signatures(archived_updates),
            f"{arm} forward update signatures differ from the ordinary archive")
    _expect(all(math.isfinite(event["loss"]) for event in updates), f"{arm} has nonfinite loss")

    mean_events = [event for event in events if event["event"] == "mean_step"]
    expected_targets = {10: (9, 8), 25: (24, 23), 50: (49, 48)}
    _expect(len(mean_events) == len(STEPS), f"{arm} must emit three mean-step events")
    trace = [(event["event"], event.get("step", event.get("target")))
             for event in events if event["event"] in ("update", "mean_step")]
    expected_trace = []
    for step in range(1, 49):
        expected_trace.append(("update", step))
        if step in (8, 23, 48):
            expected_trace.append(("mean_step", step + 1))
    _expect(trace == expected_trace, f"{arm} mean steps are not interleaved at the common prefixes")

    by_target = {event["target"]: event for event in mean_events}
    _expect(set(by_target) == set(STEPS), f"{arm} has wrong mean-step targets")
    for target in STEPS:
        event = by_target[target]
        mean_step, common_depth = expected_targets[target]
        expected_rows = state["visits"][mean_step - 1:mean_step + 1]
        controls = state["baseline_controls"][arm][str(target)]
        _expect(event.get("step") == mean_step and event.get("rows") == expected_rows,
                f"{arm} mean-step rows/depth mismatch at {target}")
        _expect(event.get("target_tokens") == [4, 4], f"{arm} target token count mismatch at {target}")
        baseline_events = state["baseline_logs"][arm][2]
        forward_first, = [item for item in baseline_events if item["event"] == "update"
                          and item.get("order") == "forward" and item["step"] == mean_step]
        swapped_first, = [item for item in baseline_events if item["event"] == "update"
                          and item.get("order") == "swapped" and item.get("target") == target
                          and item["step"] == mean_step]
        expected_losses = [forward_first["loss"], swapped_first["loss"]]
        _expect(event.get("losses") == expected_losses,
                f"{arm} mean-step single-example losses differ at {target}")
        for field, phase in (("common_sha256", "common"),
                             ("forward_first_sha256", "forward-1"),
                             ("swapped_first_sha256", "swapped-1")):
            _expect(event.get(field) == controls[phase]["adapter_sha256"],
                    f"{arm} {field} differs from archived {phase} at {target}")
        mean_adapter = attempt / f"target-{target}" / "mean" / "adapter.bin"
        _expect(mean_adapter.is_file() and digest(mean_adapter) == event.get("mean_sha256"),
                f"{arm} mean adapter hash mismatch at {target}")
        _expect(event.get("adam_before") == common_depth and event.get("adam_after") == mean_step,
                f"{arm} Adam update count mismatch at {target}")
        _expect(event.get("active_first") == 56 and event.get("active_second") == 56,
                f"{arm} mean-step Adam state is incomplete at {target}")
        _expect(event.get("reverse_equal") is True, f"{arm} mean is order-dependent at {target}")
        _expect(event["mean_sha256"] != controls["common"]["adapter_sha256"],
                f"{arm} mean step did not change the adapter at {target}")


def _mean_bias_signal(score):
    flips = {}
    for row in score["rows"]:
        if row["kind"] == "flip":
            flips.setdefault(row["pair"], []).append(row)
    components = []
    for pair, rows in sorted(flips.items()):
        _expect(len(rows) == 2, f"flip pair {pair} does not have two observations")
        true_row, = [row for row in rows if row["truth"][0] is True]
        false_row, = [row for row in rows if row["truth"][0] is False]
        l_t, l_f = true_row["log_odds_true"], false_row["log_odds_true"]
        components.append(dict(pair=pair, l_T=l_t, l_F=l_f, b=(l_t + l_f) / 2,
                               s=(l_t - l_f) / 2))
    _expect(len(components) == 8, "expected eight scored flip pairs")
    return dict(pairs=components,
                mean_b=sum(item["b"] for item in components) / len(components),
                mean_s=sum(item["s"] for item in components) / len(components))


def _evaluate_phase(arm, target, phase, directory, audit, cases, requests, checker):
    adapter = directory / "adapter.bin"
    _expect(digest(adapter) == audit["adapter_sha256"], f"{arm}/{target}/{phase} adapter changed")
    checker.check_adapter(adapter)
    entry = prior.evaluate(directory, cases, requests, audit)
    return dict(path=str(directory), **entry)


def _comparison(phases):
    metrics = {name: _mean_bias_signal(entry["score"]) | {
        "complete_pairs": entry["score"]["summary"]["all"]["both_correct"],
        "first_token_pairs": entry["score"]["summary"]["all"]["both_first_correct"],
        "flip_complete_pairs": entry["score"]["summary"]["flip"]["both_correct"],
        "all_rows_correct": entry["score"]["summary"]["all"]["correct"],
    } for name, entry in phases.items()}
    differences = {}
    for phase in ("forward-1", "swapped-1", "forward-2", "swapped-2"):
        differences[phase] = {
            key: metrics["mean"][key] - metrics[phase][key]
            for key in ("mean_b", "mean_s", "complete_pairs", "first_token_pairs", "flip_complete_pairs")
        }
    return dict(metrics=metrics, mean_minus_sequential=differences)


def main(out, driver, train_workers=2, preflight_only=False):
    state = _prepare(out, Path(driver).resolve())
    state.update(out=out, protocol_sha256=state["protocol_sha256"], events={}, log_sha={})
    if preflight_only:
        print("Preflight passed: archives and saved controls are pinned; native training was not started.", flush=True)
        return

    with ThreadPoolExecutor(max_workers=train_workers) as pool:
        futures = {pool.submit(_train_one, out, arm, driver, state): arm for arm in state["audits"]}
        trained = {}
        for future in as_completed(futures):
            arm = futures[future]
            attempt, events, log_sha = future.result()
            trained[arm] = (attempt, events, log_sha)
            print(f"{arm}: native events validated; train log sha256={log_sha}", flush=True)

    attempts = {arm: record[0] for arm, record in trained.items()}
    state["events"] = {arm: record[1] for arm, record in trained.items()}
    state["log_sha"] = {arm: record[2] for arm, record in trained.items()}
    repo = Path(__file__).resolve().parents[4]
    _, checker = fit._checker(repo)
    scores = {arm: {} for arm in trained}
    work = []
    for arm in trained:
        for target in STEPS:
            work.append((arm, target, "mean", attempts[arm] / f"target-{target}" / "mean"))
    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as pool:
        futures = {}
        for arm, target, phase, directory in work:
            event = next(event for event in state["events"][arm]
                         if event["event"] == "mean_step" and event["target"] == target)
            audit = dict(adapter_sha256=digest(directory / "adapter.bin"),
                         train_log_sha256=state["log_sha"][arm], step=event["step"],
                         target=target, phase="mean", protocol_sha256=state["protocol_sha256"])
            keep(directory / "audit.json", audit)
            futures[pool.submit(_evaluate_phase, arm, target, phase, directory, audit,
                                state["cases"], state["requests"], checker)] = (arm, target, phase)

        for future in as_completed(futures):
            arm, target, phase = futures[future]
            scores[arm].setdefault(str(target), {})[phase] = future.result()

    for arm in trained:
        attempt = attempts[arm]
        for target in STEPS:
            key = str(target)
            phases = dict(state["baseline_controls"][arm][key])
            phases.update(scores[arm][key])
            scores[arm][key] = phases
            scores[arm][key]["comparison"] = _comparison(phases)
            scores[arm][key]["orientation"] = state["orientations"][key]
            print(arm, target, json.dumps(scores[arm][key]["comparison"]["metrics"], sort_keys=True),
                  flush=True)

    keep(out / "results.json", dict(protocol_sha256=state["protocol_sha256"], scores=scores,
                                   training={arm: dict(attempt=str(attempts[arm]),
                                                       log_sha256=state["log_sha"][arm])
                                             for arm in trained}))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=OUT)
    cli.add_argument("--driver", type=Path)
    cli.add_argument("--train-workers", type=int, default=2)
    cli.add_argument("--preflight-only", action="store_true")
    args = cli.parse_args()
    _expect(args.train_workers >= 1, "--train-workers must be positive")
    output = args.output.resolve()
    driver = (args.driver or output / "native-mean").resolve()
    main(output, driver, args.train_workers, args.preflight_only)
