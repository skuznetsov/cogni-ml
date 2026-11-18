"""Fit the two-choice first-token adapter on the sealed truth-table dataset."""

import argparse
from contextlib import redirect_stdout
import math
from pathlib import Path
from statistics import mean
import subprocess

import experiment as original
import fit

OUT = original.OUT / "first-token-loss-v1"
REPO = original.transfer.REPO
TOKEN_IDS = {"true": 1802, "false": 3721}
PINS = {
    "protocol.json": "b2c7d180def9c26c8b87aabcedc965a5b60e4c044e1bbc84f87cad02c885fb10",
    "train.jsonl": "63b892fef3c0c5558ee03da70a91963fb21fb04c1f2cfef6d6f108d9c8ea956f",
    "train-counts.json": "ad13979b19055d84d61318eb5148d90e5d36844611d5f30b773cca03133d0dc2",
    "training/audit.json": "9688ce78e0db627fb174d772f4ae296d593b1b986e071478c0a704355942c76d",
}
CONFIG = dict(fit.CONFIG, learning_rate=0.0001, choices=TOKEN_IDS,
              objective="negative log softmax of the two literal logits at the first answer token",
              context=4096, examples=32)


def audit(log, adapter, rows, counts, previous, checker):
    events = fit._events(log)
    example_count = len(rows)
    expected_events = ["prefill"] * example_count + ["start"] + ["update"] * fit.UPDATES + ["saved"]
    assert [e["event"] for e in events] == expected_events
    prefills = events[:example_count]
    start, *updates, saved = events[example_count:]
    for i, (prefill, row, count) in enumerate(zip(prefills, rows, counts, strict=True), 1):
        label = row["messages"][-1]["content"].split(",")[0]
        assert (prefill["row"], prefill["id"], prefill["label_id"]) == (i, row["id"], TOKEN_IDS[label])
        assert prefill["prefix_tokens"] == count["prefix_tokens"] and prefill["target_tokens"] == 1
    assert start["model_sha256"] == fit.MODEL_SHA and start["input_adapter"] == str(original.INITIAL)
    assert (start["seed"], start["updates"], start["examples"], start["blocks"], start["choices"]) == (
        fit.SEED, fit.UPDATES, example_count, fit.LAYERS, TOKEN_IDS)
    assert math.isclose(start["learning_rate"], CONFIG["learning_rate"], rel_tol=1e-6)
    assert [e["step"] for e in updates] == list(range(1, fit.UPDATES + 1))
    visits = [e["row"] for e in updates]
    assert visits == previous["visits"] and set(visits) == set(range(1, example_count + 1))
    assert all(math.isfinite(e["loss"]) and e["loss"] >= 0 for e in updates)
    assert saved["adapter"] == str(adapter) and math.isfinite(saved["seconds"])

    with redirect_stdout(None):
        adapter_sha = checker.check_adapter(adapter)
    before, after = fit._load_factors(original.INITIAL), fit._load_factors(adapter)
    offset = 0
    updated = 0
    for _ in fit.LAYERS:
        for _, width, height in fit.PROJECTIONS:
            for n in (width, height):
                end = offset + fit.RANK * n
                assert before[offset:end] != after[offset:end], f"unchanged factor at {offset}"
                offset = end
                updated += 1
    assert offset == len(before) == len(after)

    first_loss = {}
    repeated = []
    for e in updates:
        row, loss = e["row"], e["loss"]
        if row in first_loss:
            repeated.append((first_loss[row], loss))
        else:
            first_loss[row] = loss
    return dict(adapter_sha256=adapter_sha, train_log_sha256=fit._digest(log), visits=visits,
                supervised_target_tokens=len(updates), updated_factors=updated,
                repeated_rows=len(repeated), loss_reduced_on_repeat=sum(b < a for a, b in repeated),
                mean_first_visit_loss_on_repeated_rows=mean(a for a, _ in repeated),
                mean_second_visit_loss_on_repeated_rows=mean(b for _, b in repeated),
                elapsed_seconds=saved["seconds"])


def train(trainer, out):
    paths = {name: original.OUT / name for name in PINS}
    for name, path in paths.items():
        assert fit._digest(path) == PINS[name], f"archived training input changed: {path}"
    assert fit._digest(original.runner.MODEL) == fit.MODEL_SHA
    assert fit._digest(original.INITIAL) == fit.INITIAL_SHA
    rows = fit._rows(paths["train.jsonl"])
    assert len(rows) == len({r["id"] for r in rows}) == CONFIG["examples"]
    assert {r["messages"][-1]["content"] for r in rows} == {"true, false", "false, true"}
    checker_path, checker = fit._checker(REPO)
    inputs = [*paths.values(), original.runner.MODEL, original.INITIAL, trainer, checker_path,
              Path(__file__).resolve(), original.HERE / "fit.py", original.HERE / "train_first_token.cr",
              REPO / "src/ml/gguf/qwen35_qlora.cr", REPO / "build/bridge.o"]
    hashes = {str(p): fit._digest(p) for p in inputs}
    counts = fit._json(paths["train-counts.json"])
    previous = fit._json(paths["training/audit.json"])
    out.mkdir(parents=True)
    protocol = out / "fit-protocol.json"
    log, adapter = out / "train.log", out / "adapter.bin"
    command = [trainer, original.runner.MODEL, paths["train.jsonl"], original.INITIAL, adapter]
    fit._write_new(protocol, dict(config=CONFIG, files=hashes, command=[str(p) for p in command],
                                 holdout_supplied=False))
    print(f"Training inputs sealed in {protocol}.", flush=True)
    with log.open("x") as stderr:
        subprocess.run(command, stderr=stderr, check=True, cwd=REPO)
    result = audit(log, adapter, rows, counts, previous, checker)
    assert all(fit._digest(Path(path)) == sha for path, sha in hashes.items())
    fit._write_new(out / "training-audit.json", dict(result, protocol_sha256=fit._digest(protocol)))
    print(f"Verified 50 matched updates and {result['updated_factors']} changed factors.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("trainer", type=Path)
    cli.add_argument("--output", type=Path, default=OUT)
    args = cli.parse_args()
    train(args.trainer.resolve(strict=True), args.output.resolve())
