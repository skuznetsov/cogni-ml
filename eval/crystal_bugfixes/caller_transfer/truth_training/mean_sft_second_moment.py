"""Compare zeroing Adam's second moment with the matched averaged SFT controls."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mean_sft_first_moment as first


mean_step = first.mean_step
OUT = mean_step.semantic.OUT / "mean-sft-second-moment-v1"
SOURCE = Path(__file__).with_name("train_mean_sft_second_moment.cr")
FIRST_PINS = {
    "first-moment-protocol.json": "0b135f27a614467bc95950382a7f02adf0715ec1e1c35d68cce5af176cf22866",
    "results.json": "90a832677fa4cacb5c761f88c49795c502d49e2abddbcbce6684d3cbdb4fc7b4",
}
expect, digest, read, keep = first.expect, first.digest, first.read, first.keep


def archived_controls(state):
    carry_protocol, carry = mean_step.ordinary.archive_at(mean_step.OUT, first.CARRY_PINS)
    for name, expected in FIRST_PINS.items():
        expect(digest(first.OUT / name) == expected, f"first-moment archive changed: {name}")
    first_protocol = read(first.OUT / "first-moment-protocol.json")
    protocol_file = first.OUT / "protocol.json"
    expect(digest(protocol_file) == first_protocol["native_protocol_sha256"],
           "first-moment native protocol changed")
    first_results = read(first.OUT / "results.json")
    expect(first_results["protocol_sha256"] == FIRST_PINS["first-moment-protocol.json"]
           and first_protocol["carry_archive"] == first.CARRY_PINS,
           "first-moment run uses different controls")
    first_native_protocol = read(protocol_file)
    for field in ("evaluation", "training_config", "pair_orientations"):
        expect(state["protocol"][field] == carry_protocol[field] == first_native_protocol[field],
               f"archived protocols differ on {field}")

    controls = {}
    for arm in first.ARMS:
        controls[arm] = {}
        for target in mean_step.STEPS:
            key = str(target)
            carry_entry = carry["scores"][arm][key]["mean"]
            first_entry = first_results["scores"][arm][key]
            expect(first_entry["start"] == target - 2 and first_entry["carry"] == carry_entry,
                   f"{arm} start {target - 2} uses a different carry checkpoint")
            controls[arm][key] = {
                "carry": first.archived_checkpoint(carry_entry, state["cases"]),
                "zero_m": first.archived_checkpoint(first_entry["reset"], state["cases"]),
            }
    return controls


def check_reset_events(arm, events, controls):
    trace = [event for event in events
             if event["event"] in ("mean_second_moment_reset", "mean_step")]
    expect(len(trace) == 3 * len(mean_step.STEPS), f"{arm} reset trace is incomplete")
    for index, target in enumerate(mean_step.STEPS):
        start = target - 2
        expected = dict(event="mean_second_moment_reset", adam_before=start,
                        adam_after=start + 1, factors=56,
                        carry_sha256=controls[str(target)]["carry"]["adapter_sha256"])
        direct, reverse, checkpoint = trace[3 * index:3 * (index + 1)]
        expect(direct == reverse == expected, f"{arm} start {start} carry step mismatch")
        expect(checkpoint["event"] == "mean_step" and checkpoint["target"] == target
               and checkpoint["reverse_equal"], f"{arm} start {start} reset order check failed")


def main(out, driver):
    expect(out not in (mean_step.OUT.resolve(), first.OUT.resolve()),
           "choose a separate output for the second-moment experiment")
    state = mean_step._prepare(out, driver)
    controls = archived_controls(state)
    protocol_path = out / "second-moment-protocol.json"
    keep(protocol_path, dict(carry_archive=first.CARRY_PINS, first_moment_archive=FIRST_PINS,
                             intervention="zero Adam v before one averaged full-SFT update",
                             retained=["m", "per-parameter step counters"],
                             native_protocol_sha256=state["protocol_sha256"],
                             files={str(path.resolve()): digest(path)
                                    for path in (Path(__file__), SOURCE, Path(first.__file__), driver)}))
    protocol_sha = digest(protocol_path)

    def train(arm):
        attempt, events, log_sha = mean_step._train_one(out, arm, driver, state)
        check_reset_events(arm, events, controls[arm])
        print(f"{arm}: all three carry adapters matched byte for byte and reset states validated",
              flush=True)
        return attempt, log_sha

    with ThreadPoolExecutor(max_workers=2) as pool:
        trained = dict(zip(first.ARMS, pool.map(train, first.ARMS), strict=True))

    _, checker = mean_step.fit._checker(mean_step.forward.semantic.experiment.transfer.REPO)

    def evaluate(task):
        arm, target = task
        attempt, log_sha = trained[arm]
        directory = attempt / f"target-{target}" / "mean"
        audit = dict(adapter_sha256=digest(directory / "adapter.bin"),
                     train_log_sha256=log_sha, step=target - 1, target=target,
                     phase="zero-second-moment", protocol_sha256=protocol_sha)
        keep(directory / "audit.json", audit)
        reset = mean_step._evaluate_phase(arm, target, "zero-second-moment", directory, audit,
                                          state["cases"], state["requests"], checker)
        entries = dict(controls[arm][str(target)], zero_v=reset)
        scores = {mode: first.metrics(entry["score"]) for mode, entry in entries.items()}
        print(f"{arm} start {target - 2}: {scores}", flush=True)
        return dict(start=target - 2, checkpoints=entries, metrics=scores)

    tasks = [(arm, target) for arm in first.ARMS for target in mean_step.STEPS]
    results = {arm: {} for arm in first.ARMS}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for (arm, target), result in zip(tasks, pool.map(evaluate, tasks), strict=True):
            results[arm][str(target)] = result
    keep(out / "results.json", dict(protocol_sha256=protocol_sha, scores=results))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=OUT)
    cli.add_argument("--driver", type=Path)
    args = cli.parse_args()
    output = args.output.resolve()
    main(output, (args.driver or output / "native-mean").resolve())
