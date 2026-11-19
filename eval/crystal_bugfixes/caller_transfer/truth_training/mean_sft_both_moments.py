"""Compare clearing both Adam moments with the three matched averaged SFT controls."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path

import adam_counter_reset as counter_reset
import mean_sft_second_moment as second


first, mean_step = second.first, second.mean_step
factors = counter_reset.factors
OUT = mean_step.semantic.OUT / "mean-sft-both-moments-v1"
SOURCE = Path(__file__).with_name("train_mean_sft_both_moments.cr")
SECOND_PINS = {
    "second-moment-protocol.json": "08c01c5b606797e71a812191253f48c2a0f69fcbc6f392710c572f95d1817a8f",
    "results.json": "4ae96141d24c99b5eaf354e0ce54f400f603cb66caecf3bc992feb3f00974824",
}
EVENT = "mean_both_moments_reset"
PHASE = "zero-both-moments"
expect, digest, read, keep = second.expect, second.digest, second.read, second.keep


def archived_controls(state):
    controls = second.archived_controls(state)
    for name, expected in SECOND_PINS.items():
        expect(digest(second.OUT / name) == expected, f"second-moment archive changed: {name}")
    protocol = read(second.OUT / "second-moment-protocol.json")
    archive = read(second.OUT / "results.json")
    expect(archive["protocol_sha256"] == SECOND_PINS["second-moment-protocol.json"]
           and protocol["carry_archive"] == first.CARRY_PINS
           and protocol["first_moment_archive"] == second.FIRST_PINS,
           "second-moment run uses different controls")
    native_file = second.OUT / "protocol.json"
    expect(digest(native_file) == protocol["native_protocol_sha256"],
           "second-moment native protocol changed")
    native = read(native_file)
    for field in ("evaluation", "training_config", "pair_orientations", "archived_visits",
                  "model_sha256", "initial_adapter_sha256", "lora_scope"):
        expect(native[field] == state["protocol"][field], f"protocols differ on {field}")

    for arm in first.ARMS:
        for target in mean_step.STEPS:
            key = str(target)
            entry = archive["scores"][arm][key]
            checkpoints = entry["checkpoints"]
            expect(entry["start"] == target - 2
                   and all(checkpoints[mode] == item for mode, item in controls[arm][key].items()),
                   f"{arm} start {target - 2} uses different controls")
            controls[arm][key]["zero_v"] = first.archived_checkpoint(checkpoints["zero_v"], state["cases"])
            for mode, item in controls[arm][key].items():
                expect(digest(Path(item["path"]) / "audit.json") == item["audit_sha256"],
                       f"{arm} start {target - 2} {mode} audit changed")
    return controls


def check_reset_events(arm, events, controls):
    trace = [event for event in events if event["event"] in (EVENT, "mean_step")]
    expect(len(trace) == 3 * len(mean_step.STEPS), f"{arm} reset trace is incomplete")
    for index, target in enumerate(mean_step.STEPS):
        start = target - 2
        expected = dict(event=EVENT, adam_before=start, adam_after=start + 1,
                        factors=56, carry_sha256=controls[str(target)]["carry"]["adapter_sha256"])
        direct, reverse, checkpoint = trace[3 * index:3 * (index + 1)]
        expect(direct == reverse == expected, f"{arm} start {start} carry step mismatch")
        expect(checkpoint["event"] == "mean_step" and checkpoint["target"] == target
               and checkpoint["reverse_equal"], f"{arm} start {start} reset order check failed")


def step_norm(before, adapter):
    after = factors(adapter)
    value = math.sqrt(math.fsum((y - x) ** 2 for x, y in zip(before, after, strict=True)))
    expect(math.isfinite(value) and value > 0, f"invalid step norm for {adapter}")
    return value


def main(out, driver):
    expect(out not in (mean_step.OUT.resolve(), first.OUT.resolve(), second.OUT.resolve()),
           "choose a separate output for the both-moments experiment")
    state = mean_step._prepare(out, driver)
    controls = archived_controls(state)
    protocol_path = out / "both-moments-protocol.json"
    keep(protocol_path, dict(carry_archive=first.CARRY_PINS, first_moment_archive=second.FIRST_PINS,
                             second_moment_archive=SECOND_PINS,
                             intervention="zero Adam m and v before one averaged full-SFT update",
                             retained=["per-parameter step counters"],
                             norm="L2 of serialized FP32 factor changes from the matched common checkpoint",
                             native_protocol_sha256=state["protocol_sha256"],
                             files={str(path.resolve()): digest(path)
                                    for path in (Path(__file__), SOURCE, Path(second.__file__),
                                                 Path(counter_reset.__file__), driver)}))
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
        key = str(target)
        attempt, log_sha = trained[arm]
        directory = attempt / f"target-{target}" / "mean"
        audit = dict(adapter_sha256=digest(directory / "adapter.bin"),
                     train_log_sha256=log_sha, step=target - 1, target=target,
                     phase=PHASE, protocol_sha256=protocol_sha)
        keep(directory / "audit.json", audit)
        reset = mean_step._evaluate_phase(arm, target, PHASE, directory, audit,
                                          state["cases"], state["requests"], checker)
        entries = dict(controls[arm][key], zero_mv=reset)
        common = state["baseline_controls"][arm][key]["common"]
        before = factors(Path(common["path"]) / "adapter.bin")
        scores = {
            mode: dict(first.metrics(entry["score"]),
                       step_l2=step_norm(before, Path(entry["path"]) / "adapter.bin"))
            for mode, entry in entries.items()
        }
        print(f"{arm} start {target - 2}: {scores}", flush=True)
        return dict(start=target - 2, common_adapter_sha256=common["adapter_sha256"],
                    checkpoints=entries, metrics=scores)

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
