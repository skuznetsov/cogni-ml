"""Match full SFT on old and more varied executable Crystal caller tasks."""

import argparse
from collections import defaultdict, deque
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from statistics import mean
import subprocess
from textwrap import indent

import curriculum
import experiment
import fit
from compare_first_token_loss import run, validate_readout
from measure_first_token import SAMPLING
import semantic_holdout_families
import semantic_train_families


REFERENCE = experiment.OUT
OUT = REFERENCE / "semantic-diversity-v1"
READOUT = REFERENCE / "first-token-v1/readout"
MODEL = experiment.runner.MODEL
SAMPLER = experiment.runner.SAMPLED_NATIVE
INITIAL = experiment.INITIAL
TRAINER = experiment.TRAINER
CONTEXT, MAX_NEW = 4096, 128

CONTROL_PINS = {
    "protocol.json": "b2c7d180def9c26c8b87aabcedc965a5b60e4c044e1bbc84f87cad02c885fb10",
    "training/audit.json": "9688ce78e0db627fb174d772f4ae296d593b1b986e071478c0a704355942c76d",
    "training/adapter.bin": "7b20bf1e61d7134694ecc64b6d2c7a0e6026a640e0c8bf68b98527a3faac6716",
    "first-token-v1/readout": "484808a53fa6e50af58e29147a900ec82643106ef74cc65550dc90370c7cd284",
}
digest, read, keep = experiment.runner.digest, experiment.native.read, experiment.native.keep
literal = experiment.literal


def verify_control():
    for name, expected in CONTROL_PINS.items():
        if digest(REFERENCE / name) != expected:
            raise ValueError(f"Archived control changed: {name}")
    proto = read(REFERENCE / "protocol.json")
    assert digest(SAMPLER) == proto["pins"]["sampler"]
    assert digest(REFERENCE / "cases.json") == proto["inputs"]["cases.json"]
    # With the complete audit pinned, fit() revalidates the archived run and reuses
    # its adapter. Both arms therefore execute the same trainer and audit path.
    return fit.fit(REFERENCE, INITIAL, MODEL, TRAINER)


def source_and_call(family, caller_body):
    caller_path, helper_path = family["caller_path"], family["helper_path"]

    def method(path, name, argument, body):
        pad = " " * (2 * len(path))
        return [f"{pad}def self.{name}({argument} : Int32) : Bool",
                indent(body, pad + "  "), f"{pad}end"]

    caller = method(caller_path, family["caller_name"], family["caller_arg"], caller_body)
    helper = method(helper_path, family["helper_name"], family["helper_arg"], "<HOLE>")
    if caller_path == helper_path:
        source = curriculum._module_block(caller_path, [caller, helper])
    else:
        source = (curriculum._module_block(helper_path, [helper]) + "\n\n"
                  + curriculum._module_block(caller_path, [caller]))
    return source, f"{'::'.join(caller_path)}.{family['caller_name']}"


def examples(families, split):
    rows = []
    for index, family in enumerate(families):
        variants = family["variants"]
        assert len(variants) == 2 and variants[0][0] != variants[1][0]
        assert all(type(inverted) is bool for _, inverted in variants)
        kind = "flip" if variants[0][1] != variants[1][1] else "invariant"
        for side, (body, inverted) in enumerate(variants):
            source, call = source_and_call(family, body)
            assert source.count("<HOLE>") == 1
            for zero, first in product((False, True), (0, 1)):
                query_order = (first, 1 - first)
                contract_order = (1, 0) if (index + zero + (split == "holdout")) % 2 else (0, 1)
                truth = curriculum._truth_for_order(zero, inverted, query_order)
                pair = f"{family['family']}-z{int(zero)}-q{first}"
                messages = curriculum._messages(
                    family=family, source=source, public_zero=zero,
                    contract_order=contract_order, query_order=query_order)
                rows.append(dict(id=f"{pair}-{side}", pair=pair, side=side, kind=kind,
                                 split=split, family=family["family"], public_zero=zero,
                                 order=list(query_order), truth=truth, messages=messages,
                                 source=source, call=call, helper_arg=family["helper_arg"]))
    assert len(rows) == 32 and len({r["family"] for r in rows}) == 4
    assert {kind: sum(r["kind"] == kind for r in rows) for kind in ("flip", "invariant")} == {
        "flip": 16, "invariant": 16,
    }
    return rows


def check_crystal(rows, out):
    """Run both public inputs under all four possible helper truth tables."""
    blocks, assertions = [], []
    for index, row in enumerate(rows):
        for zero, one in product((False, True), repeat=2):
            name = f"Oracle{index}Z{int(zero)}O{int(one)}"
            arg = row["helper_arg"]
            body = row["source"].replace("<HOLE>", f"{arg} == 0 ? {literal(zero)} : {literal(one)}")
            blocks.append(f"module {name}\n{indent(body, '  ')}\nend")
            call = f"{name}::{row['call']}"
            observed = f"{{{call}(0), {call}(1)}}"
            public = f"{{{literal(row['public_zero'])}, {literal(not row['public_zero'])}}}"
            values = (zero, one)
            matches = literal([values[i] for i in row["order"]] == row["truth"])
            assertions.append(f'raise "{row["id"]} Z{int(zero)}O{int(one)}" '
                              f'unless ({observed} == {public}) == {matches}')
    sentinel = f"{len(assertions)} executable truth-table checks passed"
    path = out / "controls.cr"
    experiment.keep_text(path, "\n\n".join(blocks + assertions + [f'puts "{sentinel}"']) + "\n")
    result = subprocess.run(["crystal", "run", str(path)], capture_output=True, text=True, timeout=180)
    if result.returncode or result.stdout.strip() != sentinel:
        raise RuntimeError(f"Crystal oracle failed:\n{result.stdout}\n{result.stderr}")
    keep(out / "controls.json", dict(tables=len(assertions), public_calls=2 * len(assertions),
                                    stdout=result.stdout, source_sha256=digest(path)))
    return sentinel


def match_targets(rows, reference_rows):
    """Keep the old label at each row position, hence at each optimizer step."""
    pools = defaultdict(deque)
    for row in sorted(rows, key=lambda r: sha256(f"semantic-v1:{r['id']}".encode()).digest()):
        target = ", ".join(map(literal, row["truth"]))
        pools[target].append(dict(id=row["id"], messages=[
            *row["messages"], dict(role="assistant", content=target)]))
    ordered = [pools[r["messages"][-1]["content"]].popleft() for r in reference_rows]
    assert len(ordered) == len(rows) and all(not pool for pool in pools.values())
    return ordered


def freeze(out):
    train = examples(semantic_train_families.FAMILIES, "train")
    holdout = examples(semantic_holdout_families.FAMILIES, "holdout")
    old_cases = read(REFERENCE / "cases.json")
    old_sources = {r["source"] for r in old_cases if "source" in r}
    assert {r["source"] for r in train}.isdisjoint(r["source"] for r in holdout)
    assert old_sources.isdisjoint(r["source"] for r in train + holdout)
    all_cases = train + holdout + old_cases
    assert len({r["id"] for r in all_cases}) == len(all_cases)
    assert len({experiment.transfer.fingerprint(r["messages"]) for r in all_cases}) == len(all_cases)
    assert len({r["family"] for r in train + holdout}) == 8

    sentinel = check_crystal(train + holdout, out)
    ordered = match_targets(train, fit._rows(REFERENCE / "train.jsonl"))
    experiment.jsonl(out / "train.jsonl", ordered)
    experiment.keep_text(out / "instrument.jsonl", (REFERENCE / "instrument.jsonl").read_text())
    counter = experiment.native.data
    counts = counter.token_count_rows(ordered)
    old_counts = read(REFERENCE / "train-counts.json")
    assert [r["target_tokens"] for r in counts] == [r["target_tokens"] for r in old_counts] == [4] * 32
    assert all(r["input_tokens"] <= CONTEXT for r in counts)
    keep(out / "train-counts.json", counts)
    count_holdout = counter.token_count_rows([
        dict(id=r["id"], messages=[*r["messages"], dict(role="assistant", content="")]) for r in holdout
    ])
    for case, record in zip(holdout, count_holdout, strict=True):
        assert case["id"] == record["id"] and record["prefix_tokens"] + MAX_NEW <= CONTEXT
        case["prefix_tokens"] = record["prefix_tokens"]
    keep(out / "cases.json", holdout)
    files = [Path(module.__file__).resolve() for module in
             (curriculum, experiment, fit, semantic_holdout_families, semantic_train_families, counter)]
    files += [Path(__file__).resolve(), Path(run.__code__.co_filename).resolve(), counter.TOKEN_COUNTER,
              TRAINER, INITIAL, SAMPLER, READOUT]
    protocol = dict(
        inputs={name: digest(out / name) for name in (
            "train.jsonl", "train-counts.json", "instrument.jsonl", "cases.json", "controls.json", "controls.cr")},
        files={str(path): digest(path) for path in files},
        pins=dict(model=fit.MODEL_SHA), control=CONTROL_PINS,
        train_seed=fit.SEED, rank=fit.RANK, blocks=fit.LAYERS, steps=fit.UPDATES,
        target_labels_match_by_position=True, sampling=SAMPLING, context=CONTEXT, max_new=MAX_NEW,
        primary="both sampled answers correct within each held-out caller transformation pair",
        secondary=["individual answers", "pairs by transformation and family", "both first-token literal leaders",
                   "mean log odds of the correct first literal"],
        crystal=subprocess.check_output(["crystal", "--version"], text=True).splitlines()[0],
    )
    keep(out / "protocol.json", protocol)
    print(f"{sentinel}. Frozen 32 train and 32 holdout rows, four target tokens per train row.", flush=True)
    return holdout


def verify_training(out):
    current = read(out / "training/audit.json")
    control = read(REFERENCE / "training/audit.json")
    for key in ("config", "visits", "target_tokens_by_train_row", "instrument_target_tokens"):
        assert current[key] == control[key], key
    for key in ("initial", "model", "instrument", "trainer"):
        assert current["inputs"][key]["sha256"] == control["inputs"][key]["sha256"], key
    for audit in (control, current):
        assert audit["supervised_target_tokens_for_updates"] == 200 and audit["coverage_complete"]
        assert audit["updated_projections"] == 28 and audit["updated_factors"] == 56
        assert not audit["holdout_supplied"]
    assert current["adapter_sha256"] != control["adapter_sha256"]
    return current


def summarize(cases, generated, readouts):
    rows = []
    pairs = defaultdict(list)
    for case in cases:
        idx = case["id"]
        response, readout = generated[idx], readouts[idx]
        experiment.runner.verify_sampling({"sampling": SAMPLING}, response)
        validate_readout(case, readout)
        assert response["id"] == "repair" and 0 <= response["tokens"] <= MAX_NEW
        values = experiment.probe.score_truth(experiment.probe.normalize(response))
        for label in ("true", "false"):
            if readout["sampled_id"] == readout[label]["id"]:
                assert response["answer"].startswith(label), (idx, label, response)
        log_odds = readout["true"]["logit"] - readout["false"]["logit"]
        correct_log_odds = log_odds if case["truth"][0] else -log_odds
        fields = {key: case[key] for key in ("id", "pair", "side", "kind", "family", "truth", "order")}
        row = dict(fields, raw=response["answer"], stop=response["stop"], format_ok=values is not None,
                   correct=values == case["truth"], first_correct=correct_log_odds > 0,
                   log_odds_true=log_odds, correct_log_odds=correct_log_odds)
        rows.append(row)
        pairs[case["pair"]].append(row)

    assert len(pairs) == 16
    pair_scores = []
    for pair, observations in pairs.items():
        a, b = sorted(observations, key=lambda row: row["side"])
        assert [a["side"], b["side"]] == [0, 1] and a["kind"] == b["kind"]
        assert b["truth"] == (a["truth"][::-1] if a["kind"] == "flip" else a["truth"])
        shift = b["log_odds_true"] - a["log_odds_true"]
        directed = shift if b["truth"][0] else -shift
        pair_scores.append(dict(pair=pair, kind=a["kind"], family=a["family"],
                                correct=a["correct"] and b["correct"],
                                first_correct=a["first_correct"] and b["first_correct"],
                                log_odds_shift=shift,
                                directed_shift=directed if a["kind"] == "flip" else None))

    groups = ["all", "flip", "invariant", *sorted({r["family"] for r in rows})]
    summaries = {}
    for group in groups:
        selected = [r for r in rows if group in ("all", r["kind"], r["family"])]
        selected_pairs = [r for r in pair_scores if group in ("all", r["kind"], r["family"])]
        summaries[group] = dict(rows=len(selected), correct=sum(r["correct"] for r in selected),
                                format_ok=sum(r["format_ok"] for r in selected),
                                first_correct=sum(r["first_correct"] for r in selected),
                                pairs=len(selected_pairs), both_correct=sum(r["correct"] for r in selected_pairs),
                                both_first_correct=sum(r["first_correct"] for r in selected_pairs),
                                mean_correct_log_odds=mean(r["correct_log_odds"] for r in selected))
    flips = [p["directed_shift"] for p in pair_scores if p["kind"] == "flip"]
    inv = [abs(p["log_odds_shift"]) for p in pair_scores if p["kind"] == "invariant"]
    summaries["flip"]["shifted_toward_new_target"] = sum(shift > 0 for shift in flips)
    summaries["flip"]["mean_directed_shift"] = mean(flips)
    summaries["invariant"]["mean_absolute_shift"] = mean(inv)
    return dict(summary=summaries, rows=rows, pairs=pair_scores)


def evaluate(out, cases, adapters):
    requests = [dict(id="repair", messages=r["messages"], sampling=SAMPLING) for r in cases]
    metrics = {}
    protocol = read(out / "protocol.json")
    for name, expected in protocol["inputs"].items():
        assert digest(out / name) == expected, name
    for file, expected in protocol["files"].items():
        assert digest(Path(file)) == expected, file
    assert digest(MODEL) == protocol["pins"]["model"]
    pins = {arm: digest(path) for arm, path in adapters.items()}
    assert pins["diverse"] == read(out / "training/audit.json")["adapter_sha256"]
    keep(out / "evaluation-protocol.json", dict(
        protocol_sha256=digest(out / "protocol.json"), adapters=pins,
        training_audit=digest(out / "training/audit.json"),
        requests=[experiment.transfer.fingerprint(r) for r in requests]))
    for arm, adapter in adapters.items():
        generated = run(cases, requests, [SAMPLER, "generate", MODEL, adapter, str(MAX_NEW), "4"])
        readouts = run(cases, requests, [READOUT, "first_token", MODEL, adapter])
        keep(out / f"{arm}-observations.json", dict(generations=generated, readouts=readouts))
        metrics[arm] = summarize(cases, generated, readouts)
        print(arm, json.dumps(metrics[arm]["summary"]["all"]), flush=True)
    keep(out / "results.json", dict(protocol_sha256=digest(out / "evaluation-protocol.json"), arms=metrics))


def main(out):
    control = verify_control()
    cases = freeze(out)
    print("Starting or revalidating the matched native QLoRA full-SFT run.", flush=True)
    diverse = fit.fit(out, INITIAL, MODEL, TRAINER)
    audit = verify_training(out)
    print(f"Verified 50 updates and 200 target tokens in {audit['elapsed_seconds']:.1f}s. Evaluating.", flush=True)
    evaluate(out, cases, dict(parent=INITIAL, original=control, diverse=diverse))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=OUT)
    main(cli.parse_args().output.resolve())
