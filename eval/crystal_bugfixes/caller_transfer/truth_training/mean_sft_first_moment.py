"""Compare an averaged SFT update with and without Adam's first moment."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mean_sft_step as mean_step


OUT = mean_step.semantic.OUT / "mean-sft-first-moment-v1"
SOURCE = Path(__file__).with_name("train_mean_sft_first_moment.cr")
CARRY_PINS = {
    "protocol.json": "9540efcb14c9f183d26d0274ff1d42b53484aebdfb774f6ea0d8026a4e209d14",
    "results.json": "1bab01d4258800212242fe753b0af08ea9eaaca1f0d064b895d52e35c833fdde",
}
ARMS = ("original", "diverse")
expect, digest, read, keep = mean_step._expect, mean_step.digest, mean_step.read, mean_step.keep


def archived_checkpoint(entry, cases):
    directory = Path(entry["path"])
    adapter, observations = directory / "adapter.bin", directory / "observations.json"
    expect(digest(adapter) == entry["adapter_sha256"], f"{adapter} changed")
    expect(digest(observations) == entry["observations_sha256"], f"{observations} changed")
    raw = read(observations)
    expect(raw["adapter_sha256"] == entry["adapter_sha256"], "carry observations use the wrong adapter")
    score = mean_step.prior.metrics(cases, raw)
    expect(score == entry["score"], f"{observations} score changed")
    return entry


def check_reset_events(arm, events, controls):
    trace = [event for event in events
             if event["event"] in ("mean_first_moment_reset", "mean_step")]
    expect(len(trace) == 3 * len(mean_step.STEPS), f"{arm} reset trace is incomplete")
    for index, target in enumerate(mean_step.STEPS):
        start = target - 2
        carry = controls[str(target)]
        expected = dict(event="mean_first_moment_reset", adam_before=start,
                        adam_after=start + 1, factors=56, carry_sha256=carry["adapter_sha256"])
        direct, reverse, checkpoint = trace[3 * index:3 * (index + 1)]
        expect(direct == reverse == expected,
               f"{arm} start {start} does not reproduce the archived carry step before reset")
        expect(checkpoint["event"] == "mean_step" and checkpoint["target"] == target
               and checkpoint["reverse_equal"], f"{arm} start {start} reset order check failed")


def metrics(score):
    signal = mean_step._mean_bias_signal(score)
    return dict(b=signal["mean_b"], s=signal["mean_s"],
                complete_pairs=score["summary"]["all"]["both_correct"],
                flip_pairs=score["summary"]["flip"]["both_correct"])


def main(out, driver):
    expect(out != mean_step.OUT.resolve(), "choose a separate output for the first-moment experiment")
    carry_protocol, carry = mean_step.ordinary.archive_at(mean_step.OUT, CARRY_PINS)
    state = mean_step._prepare(out, driver)
    for field in ("evaluation", "training_config", "pair_orientations"):
        expect(state["protocol"][field] == carry_protocol[field], f"carry protocol differs on {field}")
    controls = {
        arm: {target: archived_checkpoint(phases["mean"], state["cases"])
              for target, phases in carry["scores"][arm].items()}
        for arm in ARMS
    }
    protocol_path = out / "first-moment-protocol.json"
    keep(protocol_path, dict(carry_archive=CARRY_PINS,
                             intervention="zero Adam m before one averaged full-SFT update",
                             retained=["v", "per-parameter step counters"],
                             native_protocol_sha256=state["protocol_sha256"],
                             files={str(path.resolve()): digest(path)
                                    for path in (Path(__file__), SOURCE, driver)}))
    protocol_sha = digest(protocol_path)

    def train(arm):
        attempt, events, log_sha = mean_step._train_one(out, arm, driver, state)
        check_reset_events(arm, events, controls[arm])
        print(f"{arm}: all three carry adapters matched byte for byte and reset states validated",
              flush=True)
        return attempt, log_sha

    with ThreadPoolExecutor(max_workers=2) as pool:
        trained = dict(zip(ARMS, pool.map(train, ARMS), strict=True))

    _, checker = mean_step.fit._checker(mean_step.forward.semantic.experiment.transfer.REPO)

    def evaluate(task):
        arm, target = task
        attempt, log_sha = trained[arm]
        directory = attempt / f"target-{target}" / "mean"
        audit = dict(adapter_sha256=digest(directory / "adapter.bin"),
                     train_log_sha256=log_sha, step=target - 1, target=target,
                     phase="zero-first-moment", protocol_sha256=protocol_sha)
        keep(directory / "audit.json", audit)
        reset = mean_step._evaluate_phase(arm, target, "zero-first-moment", directory, audit,
                                          state["cases"], state["requests"], checker)
        baseline = controls[arm][str(target)]
        before, after = metrics(baseline["score"]), metrics(reset["score"])
        print(f"{arm} start {target - 2}: {before} -> {after}", flush=True)
        return dict(start=target - 2, carry=baseline, reset=reset,
                    carry_metrics=before, reset_metrics=after,
                    delta={name: after[name] - before[name] for name in before})

    tasks = [(arm, target) for arm in ARMS for target in mean_step.STEPS]
    results = {arm: {} for arm in ARMS}
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
