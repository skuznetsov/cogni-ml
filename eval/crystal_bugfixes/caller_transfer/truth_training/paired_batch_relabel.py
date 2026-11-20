"""Swap A and B labels while keeping both frozen candidate orders intact."""

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import subprocess

import paired_batch_choice as choice

prior, native, runner, transfer = choice.prior, choice.native, choice.runner, choice.transfer
digest, keep, read = choice.digest, choice.keep, choice.read
REFERENCE = choice.OUT
OUT = REFERENCE.with_name("paired-batch-choice-relabel-v1")
REFERENCE_PROTOCOL = "c99015e67db0894bdb9be9ecf7dd9bbf2ad5a44146fdff6f7bf0ae176525a93a"
REFERENCE_RESULTS = "669b8ea8159241b5c5a38310c1e43cb2df865e109c0dcae404be2b5afb52a1a6"
SWAP = {"A": "B", "B": "A"}


def relabel(order, patches):
    request = deepcopy(order["request"])
    user = request["messages"][-1]
    assert user["role"] == "user"
    prefix, *options = user["content"].rsplit("\n", 2)
    choices = {}
    for index, (label, option) in enumerate(zip(choice.LABELS, options, strict=True)):
        origin = order["choices"][label]
        assert option == f"{label}: {patches[origin]}"
        label_after = SWAP[label]
        options[index] = label_after + option[1:]
        choices[label_after] = origin
    user["content"] = "\n".join([prefix, *options])
    return dict(request=request, choices=choices)


def freeze(out):
    assert digest(REFERENCE / "protocol.json") == REFERENCE_PROTOCOL
    assert digest(REFERENCE / "results.json") == REFERENCE_RESULTS
    protocol, baseline = (read(REFERENCE / name) for name in ("protocol.json", "results.json"))
    assert baseline["protocol"] == REFERENCE_PROTOCOL
    assert digest(REFERENCE / "plan.json") == protocol["plan"]
    assert digest(REFERENCE / "controls.json") == baseline["controls"]
    assert digest(Path(choice.__file__)) == protocol["driver"]
    assert all(digest(path) == sha256 for path, sha256 in protocol["helpers"].items())
    assert protocol["context"] == 4096 and protocol["max_new"] == transfer.MAX_NEW == 128
    assert protocol["sampling"] == transfer.SAMPLING == dict(temperature=0.8, top_k=40, seed=101)
    crystal = subprocess.check_output(["crystal", "--version"], text=True).splitlines()[0]
    assert crystal == protocol["crystal"]

    adapters = {"tool_four": prior.experiment.INITIAL}
    for arm in prior.ARMS:
        adapters[arm] = choice.REFERENCE / "training" / arm / "attempt-001" / "adapter.bin"
    assert {arm: digest(path) for arm, path in adapters.items()} == protocol["adapters"]
    assert digest(runner.MODEL) == protocol["model"] and digest(runner.SAMPLED_NATIVE) == protocol["sampler"]

    transfer_protocol = choice.REFERENCE / "protocol.json"
    assert digest(transfer_protocol) == protocol["reference_protocol"]
    inputs = read(transfer_protocol)["inputs"]
    for name in ("cases.json", "controls.json"):
        assert digest(choice.REFERENCE / name) == inputs[name], name
    rows = {prior.case_id(row): row for row in read(choice.REFERENCE / "cases.json")}
    original = read(REFERENCE / "plan.json")
    assert list(rows) == list(original) and len(rows) == 8
    plans = {}
    for key, case in original.items():
        assert list(case["orders"]) == list(choice.LABELS)
        orders = {SWAP[label]: relabel(order, case["patches"]) for label, order in case["orders"].items()}
        plans[key] = dict(case, orders=orders)

    orders = [order for case in plans.values() for order in case["orders"].values()]
    assert len({transfer.fingerprint(order["request"]) for order in orders}) == len(orders) == 16
    prompts = [dict(id=str(index), messages=order["request"]["messages"] + [dict(role="assistant", content="")])
               for index, order in enumerate(orders)]
    for order, counted in zip(orders, native.data.token_count_rows(prompts), strict=True):
        count = counted["prefix_tokens"]
        assert count + transfer.MAX_NEW <= protocol["context"]
        order["prefix_tokens"] = count

    keep(out / "plan.json", plans)
    keep(out / "protocol.json", dict(
        protocol, reference_protocol=REFERENCE_PROTOCOL, reference_results=REFERENCE_RESULTS,
        reference_controls=baseline["controls"], plan=digest(out / "plan.json"),
        driver=digest(Path(__file__)), choice_driver=protocol["driver"], label_swap=SWAP,
    ))
    print("Frozen sixteen prompts with only the two option letters swapped", flush=True)
    return rows, original, plans, adapters, baseline


def compare(before, after, original, plans):
    lookup = {(row["case"], row["teacher_label"]): row for row in before}
    assert len(lookup) == len(before) == len(after) == 16
    pairs = []
    for new in after:
        key, label = new["case"], new["teacher_label"]
        old = lookup.pop((key, SWAP[label]))
        old_choices = original[key]["orders"][old["teacher_label"]]["choices"]
        new_choices = plans[key]["orders"][label]["choices"]
        old_position = list(old_choices).index(old["label"]) + 1 if old["label"] else None
        new_position = list(new_choices).index(new["label"]) + 1 if new["label"] else None
        assert old_choices[old["teacher_label"]] == new_choices[label] == "teacher"
        assert old_choices.get(old["label"]) == old["chosen"]
        assert new_choices.get(new["label"]) == new["chosen"]
        if old_position is None or new_position is None:
            followed = "invalid"
        elif old["label"] == new["label"]:
            assert old_position != new_position and old["chosen"] != new["chosen"]
            followed = "label"
        else:
            assert old_position == new_position and old["chosen"] == new["chosen"]
            followed = "line"
        pairs.append(dict(
            case=key, teacher_position=list(old_choices).index(old["teacher_label"]) + 1,
            before=dict(label=old["label"], position=old_position, stage=old["stage"]),
            after=dict(label=new["label"], position=new_position, stage=new["stage"]),
            followed=followed,
        ))
    assert not lookup
    return pairs


def run(out):
    rows, original, plans, adapters, baseline = freeze(out)
    controls = choice.check_candidates(out, rows, plans)
    assert digest(out / "controls.json") == baseline["controls"]
    results, pairs, summary = {}, {}, {}
    for arm, adapter in adapters.items():
        samples = choice.evaluate(out, arm, adapter, plans, controls)
        results[arm] = samples
        matches = compare(baseline["rows"][arm], samples, original, plans)
        pairs[arm] = matches
        summary[arm] = dict(
            choice.summarize(sorted(samples, key=lambda row: row["teacher_label"]), rows),
            followed=dict(Counter(pair["followed"] for pair in matches)),
            previously_B=dict(Counter(pair["followed"] for pair in matches if pair["before"]["label"] == "B")),
        )
    keep(out / "results.json", dict(
        protocol=digest(out / "protocol.json"), controls=digest(out / "controls.json"),
        summary=summary, pairs=pairs, rows=results,
    ))
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    run(parser.parse_args().out.resolve())
