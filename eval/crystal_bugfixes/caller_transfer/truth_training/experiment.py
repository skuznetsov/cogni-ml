"""Fit a native 2B adapter and measure joint caller-sign and input-order transfer."""

import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import random
import subprocess
import sys
from textwrap import indent

import curriculum
import fit as training

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import probe

native, transfer, runner = probe.native, probe.transfer, probe.runner
OUT = transfer.EXPERIMENT / "diagnostics/helper-truth-training-v1"
TRAINER = transfer.SCALE / "code-only-qlora-50/native-recovery"
INITIAL = runner.ORIGINAL_ADAPTER


def keep_text(path, content):
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Frozen input changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def literal(value):
    return str(value).lower()


def jsonl(path, rows):
    keep_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def validate(rows):
    families = {r["family"] for r in rows}
    assert len(rows) == 32 and len(families) == 4
    indexed = {(r["family"], r["public_zero"], r["order"][0], r["inverted"]): r for r in rows}
    for family, zero, first, inverted in product(families, (False, True), (0, 1), (False, True)):
        row = indexed[family, zero, first, inverted]
        assert row["order"] == [first, 1 - first]
        assert row["truth"] == [((zero if i == 0 else not zero) != inverted) for i in row["order"]]
        assert [m["role"] for m in row["messages"]] == ["system", "user"]
        assert row["source"].count("<HOLE>") == 1
        if inverted:
            direct = indexed[family, zero, first, False]
            for a, b in ((row["source"], direct["source"]),
                         (row["messages"][1]["content"], direct["messages"][1]["content"])):
                assert a.count("!") == b.count("!") + 1 and a.replace("return !", "return ", 1) == b
            assert row["messages"][0] == direct["messages"][0]
        other = indexed[family, zero, 1 - first, inverted]
        assert row["source"] == other["source"] and row["truth"] == other["truth"][::-1]
        user = row["messages"][1]["content"]
        swapped = other["messages"][1]["content"]
        old, new = str(row["order"]), str(other["order"])
        assert user.count(old) == 1 and user.replace(old, new) == swapped
        assert row["messages"][0] == other["messages"][0]


def crystal_controls(rows, folder):
    blocks, checks = [], []
    for idx, row in enumerate(rows):
        true_input = row["order"][row["truth"].index(True)]
        for wrong in (False, True):
            name = f"Control{idx}{'Opposite' if wrong else 'Teacher'}"
            body = row["source"].replace("<HOLE>", f'{row["helper_arg"]} == {true_input ^ wrong}')
            blocks.append(f"module {name}\n" + indent(body, "  ") + "\nend\n")
            for n in (0, 1):
                public = row["public_zero"] if n == 0 else not row["public_zero"]
                expected = literal(public != wrong)
                checks.append(f'raise "{name} at {n}" unless {name}::{row["call"]}({n}) == {expected}')
    sentinel = f"Passed {len(checks)} public-output checks"
    path = folder / "check.cr"
    keep_text(path, "\n".join(blocks + checks + [f'puts "{sentinel}"', ""]))
    proc = subprocess.run(["crystal", "run", str(path)], capture_output=True, text=True)
    if proc.returncode != 0 or proc.stdout.strip() != sentinel:
        raise RuntimeError(f"Crystal controls failed: {proc.stderr}\n{proc.stdout}")
    native.keep(folder / "receipt.json", dict(checks=len(checks), stdout=proc.stdout, sha256=runner.digest(path)))


def previous():
    assert runner.digest(probe.OUT / "fixtures.json") == "55458529e3955eee0c2f3c20213383f5603b77f57cc79a645206dbd349bd3ed0"
    assert runner.digest(probe.OUT / "results.json") == "5d388b7b8a1a513105fc2dae0d2dae98e0a813fb0a40b7661d68ed1e58cff092"
    rows = []
    for row in native.read(probe.OUT / "fixtures.json"):
        loc = probe.OUT / "runs/parent" / row["family"] / row["variant"] / "truth"
        req = native.read(loc / "request.json")
        assert transfer.fingerprint(req) == native.read(loc / "generation.json")["request_sha256"]
        assert req["sampling"] == transfer.SAMPLING and req["id"] == "repair"
        inverted = row["variant"] == "inverted"
        rows.append(dict(id=row["family"] + "-" + row["variant"], family=row["family"], inverted=inverted,
                         public_zero=row["truth"][0] != inverted, order=[0, 1], truth=row["truth"],
                         messages=req["messages"]))
    return rows


def freeze(out):
    splits = {split: curriculum.examples(split) for split in ("train", "holdout")}
    for rows in splits.values():
        validate(rows)
    train, holdout = splits.values()
    def identifiers(rows):
        return {name for row in rows for name in row["identifiers"]}
    assert identifiers(train).isdisjoint(identifiers(holdout))
    assert {r["family"] for r in train}.isdisjoint(r["family"] for r in holdout)
    splits["previous"] = previous()
    cases = [dict(r, split=split) for split, rows in splits.items() for r in rows]
    assert len({r["id"] for r in cases}) == len({transfer.fingerprint(r["messages"]) for r in cases}) == len(cases)
    train_rows = [dict(id=r["id"], messages=r["messages"] +
                       [dict(role="assistant", content=", ".join(map(literal, r["truth"])))]) for r in train]
    random.Random(42).shuffle(train_rows)
    jsonl(out / "train.jsonl", train_rows)
    jsonl(out / "instrument.jsonl", [dict(id="integer-control", messages=[
        dict(role="system", content="Follow the requested format."),
        dict(role="user", content="Return the integer zero."), dict(role="assistant", content="0")])])
    train_counts = native.data.token_count_rows(train_rows)
    assert all(0 < c["target_tokens"] and c["input_tokens"] <= 4096 for c in train_counts)
    native.keep(out / "train-counts.json", train_counts)
    counts = native.data.token_count_rows([dict(id=r["id"], messages=r["messages"] +
                    [dict(role="assistant", content="")]) for r in cases])
    for r, count in zip(cases, counts, strict=True):
        r["prefix_tokens"] = count["prefix_tokens"]
        assert r["prefix_tokens"] + transfer.MAX_NEW <= 4096
    native.keep(out / "cases.json", cases)
    pins = transfer.pins()
    assert runner.digest(INITIAL) == native.read(transfer.EXPERIMENT / "evaluation/runs/parent/pins.json")["adapter"]
    files = [Path(__file__), Path(curriculum.__file__), Path(training.__file__), Path(probe.__file__),
             Path(native.__file__), Path(transfer.__file__), Path(runner.__file__), Path(native.data.__file__),
             native.data.TOKEN_COUNTER, TRAINER, INITIAL]
    native.keep(out / "protocol.json", dict(inputs={k: runner.digest(out / k) for k in
                ("train.jsonl", "instrument.jsonl", "cases.json", "train-counts.json")}, files={str(p): runner.digest(p) for p in files},
                pins=pins, train_seed=42, rank=8, blocks=[11, 15, 19, 23], steps=50, sampling=transfer.SAMPLING,
                context=4096, max_new=transfer.MAX_NEW, attempts=1,
                primary="both caller signs correct within family, public polarity and requested order",
                secondary=["both signs in both orders", "individual responses", "previous caller-transfer prompts"],
                crystal=subprocess.check_output(["crystal", "--version"], text=True).splitlines()[0]))
    crystal_controls(train + holdout, out / "controls")
    return cases


def evaluate(out, arm, adapter, cases):
    _, runtime, recovery = runner.load_runtime()
    folder = out / "runs" / arm
    native.keep(folder / "pins.json", dict(adapter=runner.digest(adapter), protocol=runner.digest(out / "protocol.json")))
    worker = runner.ArmWorker(runtime.ev.evaluator, [str(runner.SAMPLED_NATIVE), "generate", str(runner.MODEL),
        str(adapter), str(transfer.MAX_NEW), "4"], folder / "worker", folder / "worker.log", 240)
    results = []
    try:
        for row in cases:
            req = dict(id="repair", messages=row["messages"], sampling=transfer.SAMPLING)
            response = native.sample(worker, recovery, req, folder / row["split"] / row["id"])
            values = probe.score_truth(probe.normalize(response))
            result = {k: row[k] for k in ("id", "family", "split", "public_zero", "inverted", "order", "truth", "prefix_tokens")}
            results.append(dict(result, arm=arm, raw=response["answer"], stop=response["stop"],
                                values=values, correct=(values == row["truth"])))
            if len(results) % 16 == 0:
                print(arm, len(results), "requests scored", flush=True)
    finally:
        assert worker.close() == 0
    native.keep(folder / "scores.json", results)
    return results


def summary(rows):
    by_pair, by_contract, by_order = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        key = (r["family"], r["public_zero"])
        by_pair[key + (tuple(r["order"]),)].append(r)
        by_contract[key].append(r)
        by_order[key + (r["inverted"],)].append(r)
    assert all(len(v) == 2 and {r["inverted"] for r in v} == {True, False} for v in by_pair.values())
    def responds_to_flip(group):
        a, b = group
        return a["values"] is not None and a["values"] in ([False, True], [True, False]) and b["values"] == a["values"][::-1]
    order_pairs = [v for v in by_order.values() if len(v) == 2]
    return dict(rows=len(rows), correct=sum(r["correct"] for r in rows),
                sign_flips=sum(responds_to_flip(v) for v in by_pair.values()),
                order_flips=sum(responds_to_flip(v) for v in order_pairs), order_total=len(order_pairs),
                format_ok=sum(r["values"] is not None for r in rows),
                direct=sum(r["correct"] for r in rows if not r["inverted"]),
                inverted=sum(r["correct"] for r in rows if r["inverted"]),
                pairs=sum(all(r["correct"] for r in v) for v in by_pair.values()), pair_total=len(by_pair),
                contracts=sum(all(r["correct"] for r in v) for v in by_contract.values()), contract_total=len(by_contract))


def main(out):
    cases = freeze(out)
    print("Protocol frozen; 256 executable controls passed.", flush=True)
    before = evaluate(out, "parent", INITIAL, cases)
    print("Parent scored. Starting native QLoRA.", flush=True)
    child = training.fit(out, INITIAL, runner.MODEL, TRAINER)
    after = evaluate(out, "trained", child, cases)
    scores = {arm: {split: summary([r for r in rows if r["split"] == split])
                    for split in ("train", "holdout", "previous")} for arm, rows in (("parent", before), ("trained", after))}
    native.keep(out / "results.json", dict(protocol=runner.digest(out / "protocol.json"),
                adapters=dict(parent=runner.digest(INITIAL), trained=runner.digest(child)), summary=scores, rows=before+after))
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    main(parser.parse_args().output.resolve())
