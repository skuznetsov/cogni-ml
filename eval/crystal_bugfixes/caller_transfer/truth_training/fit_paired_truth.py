"""Fit full SFT plus a paired first-token margin on the sealed truth dataset."""

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import math
from pathlib import Path
from statistics import mean
import subprocess

import experiment as original
import fit
from fit_first_token import PINS, REPO, TOKEN_IDS

OUT = original.OUT / "paired-loss-v1"
CONFIG = dict(fit.CONFIG, learning_rate=0.0001, choices=TOKEN_IDS, examples=32,
              context=4096, weight=0.5, minimum_margin=0.5,
              objective="full SFT on anchor + weight * hinge on the weaker correct-literal margin")


def pair_rows(rows):
    groups = defaultdict(dict)
    for index, row in enumerate(rows, 1):
        system, user, answer = [message["content"] for message in row["messages"]]
        assert user.count("return !") <= 1 and answer in {"true, false", "false, true"}
        inverted = "return !" in user
        signs = groups[system, user.replace("return !", "return ", 1)]
        assert inverted not in signs, f"duplicate training sign: {row['id']}"
        signs[inverted] = (index, answer)
    mates = [0] * len(rows)
    for signs in groups.values():
        direct, left = signs[False]
        inverted, right = signs[True]
        assert left != right
        mates[direct - 1], mates[inverted - 1] = inverted, direct
    return mates


def changed_factors(initial, adapter):
    before, after = fit._load_factors(initial), fit._load_factors(adapter)
    offset = updated = 0
    for layer in fit.LAYERS:
        for name, width, height in fit.PROJECTIONS:
            for n in (width, height):
                end = offset + fit.RANK * n
                assert before[offset:end] != after[offset:end], f"unchanged layer {layer} {name} factor"
                offset = end
                updated += 1
    assert offset == len(before) == len(after)
    return updated


def audit(log, adapter, rows, counts, previous, mates, checker):
    events = fit._events(log)
    n = len(rows)
    assert [e["event"] for e in events] == ["prefill"] * n + ["start"] + ["update"] * fit.UPDATES + ["saved"]
    for i, (prefill, row, count) in enumerate(zip(events[:n], rows, counts, strict=True), 1):
        label = row["messages"][-1]["content"].split(",")[0]
        assert (prefill["row"], prefill["id"], prefill["label_id"]) == (i, row["id"], TOKEN_IDS[label])
        assert prefill["prefix_tokens"] == count["prefix_tokens"]
        assert prefill["target_tokens"] == count["target_tokens"] == 4
    start, *updates, saved = events[n:]
    assert start["model_sha256"] == fit.MODEL_SHA and start["input_adapter"] == str(original.INITIAL)
    assert (start["seed"], start["updates"], start["examples"], start["blocks"], start["choices"]) == (
        fit.SEED, fit.UPDATES, n, fit.LAYERS, TOKEN_IDS)
    for key in ("learning_rate", "weight", "minimum_margin"):
        assert math.isclose(start[key], CONFIG[key], rel_tol=1e-6)
    steps = list(range(1, fit.UPDATES + 1))
    visits = [e["row"] for e in updates]
    assert [e["step"] for e in updates] == steps and visits == previous["visits"]
    assert set(visits) == set(range(1, n + 1))
    covered = {tuple(sorted((e["row"], e["mate_row"]))) for e in updates}
    assert len(covered) == n // 2
    for e in updates:
        assert e["mate_row"] == mates[e["row"] - 1]
        value = min(e["anchor_margin"], e["mate_margin"])
        assert all(math.isfinite(e[k]) for k in ("loss", "sft", "pair", "anchor_margin", "mate_margin"))
        assert e["sft"] >= 0
        assert math.isclose(e["pair"], max(0, CONFIG["minimum_margin"] - value), abs_tol=2e-6)
        assert math.isclose(e["loss"], e["sft"] + CONFIG["weight"] * e["pair"], rel_tol=1e-6)
    target_tokens = sum(counts[row - 1]["target_tokens"] for row in visits)
    assert target_tokens == 200
    assert saved["adapter"] == str(adapter) and math.isfinite(saved["seconds"])
    with redirect_stdout(None):
        adapter_sha = checker.check_adapter(adapter)
    active = [e for e in updates if e["pair"] > 0]
    return dict(adapter_sha256=adapter_sha, train_log_sha256=fit._digest(log), visits=visits,
                supervised_target_tokens=target_tokens, constraints=len(updates), unique_pairs=len(covered),
                active_hinges=len(active), active_mate_gradients=sum(e["mate_margin"] <= e["anchor_margin"]
                                                                  for e in active),
                mean_sft=mean(e["sft"] for e in updates), mean_hinge=mean(e["pair"] for e in updates),
                updated_factors=changed_factors(original.INITIAL, adapter), elapsed_seconds=saved["seconds"])


def train(trainer, out):
    paths = {name: original.OUT / name for name in PINS}
    for name, path in paths.items():
        assert fit._digest(path) == PINS[name], f"archived training input changed: {path}"
    assert fit._digest(original.runner.MODEL) == fit.MODEL_SHA
    assert fit._digest(original.INITIAL) == fit.INITIAL_SHA
    rows = fit._rows(paths["train.jsonl"])
    assert len(rows) == len({r["id"] for r in rows}) == CONFIG["examples"]
    mates = pair_rows(rows)
    counts = fit._json(paths["train-counts.json"])
    previous = fit._json(paths["training/audit.json"])
    checker_path, checker = fit._checker(REPO)
    out.mkdir(parents=True)
    mate_file = out / "mate-rows.json"
    fit._write_new(mate_file, mates)
    inputs = [*paths.values(), original.runner.MODEL, original.INITIAL, trainer, checker_path,
              mate_file, Path(__file__).resolve(), original.HERE / "fit.py",
              original.HERE / "fit_first_token.py", original.HERE / "train_paired_truth.cr",
              REPO / "src/ml/gguf/qwen35_qlora.cr", REPO / "build/bridge.o"]
    hashes = {str(p): fit._digest(p) for p in inputs}
    protocol = out / "fit-protocol.json"
    log, adapter = out / "train.log", out / "adapter.bin"
    command = [trainer, original.runner.MODEL, paths["train.jsonl"], mate_file, original.INITIAL, adapter]
    fit._write_new(protocol, dict(config=CONFIG, files=hashes, command=[str(p) for p in command],
                                 holdout_supplied=False))
    print(f"Training inputs sealed in {protocol}.", flush=True)
    with log.open("x") as stderr:
        subprocess.run(command, stderr=stderr, check=True, cwd=REPO)
    result = audit(log, adapter, rows, counts, previous, mates, checker)
    assert all(fit._digest(Path(path)) == sha for path, sha in hashes.items())
    fit._write_new(out / "training-audit.json", dict(result, protocol_sha256=fit._digest(protocol)))
    print(f"Verified 50 matched updates, 200 supervised targets and {result['updated_factors']} changed factors.",
          flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("trainer", type=Path)
    cli.add_argument("--output", type=Path, default=OUT)
    args = cli.parse_args()
    train(args.trainer.resolve(strict=True), args.output.resolve())
