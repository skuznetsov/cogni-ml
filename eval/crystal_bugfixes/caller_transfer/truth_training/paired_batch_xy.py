"""Measure choice under X/Y in both label assignments and both code orders."""

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

import paired_batch_choice as choice
import paired_batch_relabel as ab_swap

prior, native, transfer = choice.prior, choice.native, choice.transfer
digest, keep, read = choice.digest, choice.keep, choice.read
OUT = choice.OUT.with_name("paired-batch-choice-xy-v1")
RELABEL_PROTOCOL = "8b5ddd63b353b5a5a52502f462f92e62d9eddaca0d807850261a061ee18edaf3"
RELABEL_RESULTS = "b80d8ca3a7af51b0c804cb95f3679e469212981806ef03be8aec8a403fe8e56d"
RENAME = {"A": "X", "B": "Y"}
LAYOUTS = {"XY": "AB", "YX": "BA"}
OLD_RULE, NEW_RULE = "Reply with only A or B.", "Reply with only X or Y."


def rename_labels(case):
    orders = {}
    for teacher, order in case["orders"].items():
        request = deepcopy(order["request"])
        user = request["messages"][-1]
        assert user["role"] == "user"
        prefix, _, option_block = user["content"].rpartition("\n\n")
        options = option_block.split("\n")
        assert prefix.endswith(OLD_RULE)
        choices = {}
        renamed = []
        for option, (label, origin) in zip(options, order["choices"].items(), strict=True):
            assert option == f"{label}: {case['patches'][origin]}"
            renamed.append(RENAME[label] + option[1:])
            choices[RENAME[label]] = origin
        user["content"] = prefix.removesuffix(OLD_RULE) + NEW_RULE + "\n\n" + "\n".join(renamed)
        orders[RENAME[teacher]] = dict(request=request, choices=choices)
    return dict(case, orders=orders)


def freeze(out):
    ref = ab_swap.OUT
    assert digest(ref / "protocol.json") == RELABEL_PROTOCOL
    assert digest(ref / "results.json") == RELABEL_RESULTS
    protocol, swapped = (read(ref / name) for name in ("protocol.json", "results.json"))
    assert swapped["protocol"] == RELABEL_PROTOCOL
    assert digest(ref / "plan.json") == protocol["plan"]
    rows, original, reversed_labels, adapters, baseline = ab_swap.freeze(ref)
    reference = {"XY": baseline, "YX": swapped}
    assert swapped["controls"] == baseline["controls"]

    source = {"XY": original, "YX": reversed_labels}
    plans = {layout: {key: rename_labels(case) for key, case in cases.items()}
             for layout, cases in source.items()}
    orders = []
    for layout, cases in plans.items():
        for case in cases.values():
            assert list(case["orders"]) == list(layout)
            assert all(list(order["choices"]) == list(layout) for order in case["orders"].values())
            orders.extend(case["orders"].values())
    assert len({transfer.fingerprint(order["request"]) for order in orders}) == len(orders) == 32
    prompts = [dict(id=str(i), messages=order["request"]["messages"] + [dict(role="assistant", content="")])
               for i, order in enumerate(orders)]
    for order, counted in zip(orders, native.data.token_count_rows(prompts), strict=True):
        prefix = counted["prefix_tokens"]
        assert prefix + protocol["max_new"] <= protocol["context"]
        order["prefix_tokens"] = prefix
    keep(out / "plan.json", plans)
    keep(out / "protocol.json", dict(
        context=protocol["context"], max_new=protocol["max_new"], sampling=protocol["sampling"],
        model=protocol["model"], sampler=protocol["sampler"], adapters=protocol["adapters"],
        crystal=protocol["crystal"], controls=baseline["controls"],
        reference_ab=ab_swap.REFERENCE_RESULTS, reference_ba=RELABEL_RESULTS,
        reference_ab_protocol=ab_swap.REFERENCE_PROTOCOL, reference_ba_protocol=RELABEL_PROTOCOL,
        plan=digest(out / "plan.json"), driver=digest(Path(__file__)), rename=RENAME,
        parse="EOS and exactly X or Y after trimming surrounding whitespace",
    ))
    print("Frozen thirty-two X/Y prompts covering both code orders and both label assignments", flush=True)
    return rows, plans, adapters, reference


def align(arm, rows, raw, reference):
    matched = {}
    for layout, source_layout in LAYOUTS.items():
        old = {(row["case"], row["teacher_label"]): row for row in reference[layout]["rows"][arm]}
        new = {(row["case"], row["teacher_label"]): row for row in raw[layout]}
        assert len(old) == len(new) == 16
        matches = {}
        for key in rows:
            for position, (previous, current) in enumerate(zip(source_layout, layout), 1):
                before, after = old.pop((key, previous)), new.pop((key, current))
                label, former_label = after["label"], before["label"]
                slot = layout.index(label) + 1 if label else None
                former_slot = source_layout.index(former_label) + 1 if former_label else None
                matches[key, position] = dict(
                    after, layout=layout, teacher_position=position, position=slot,
                    ab_label=former_label, ab_position=former_slot, ab_stage=before["stage"],
                )
        assert not old and not new
        matched[layout] = matches

    result = []
    for coordinate, xy in matched["XY"].items():
        yx = matched["YX"][coordinate]
        if xy["label"] is None or yx["label"] is None:
            kept = "invalid"
        elif xy["label"] == yx["label"]:
            assert xy["position"] != yx["position"]
            kept = xy["label"]
        else:
            assert xy["position"] == yx["position"]
            kept = "first" if xy["position"] == 1 else "second"
        result.append(dict(case=xy["case"], teacher_position=xy["teacher_position"],
                           across_assignments=kept, XY=xy, YX=yx))
    return result


def summarize(matches):
    selections = [match[layout] for match in matches for layout in LAYOUTS]
    summary = prior.summarize(selections)
    summary.update(
        requests=summary.pop("variants"),
        labels=dict(Counter(row["label"] or "invalid" for row in selections)),
        positions={str(slot): sum(row["position"] == slot for row in selections) for slot in (1, 2, None)},
        by_layout={layout: prior.summarize([m[layout] for m in matches]) for layout in LAYOUTS},
        across_assignments=dict(Counter(m["across_assignments"] for m in matches)),
        same_ab_position=sum(row["position"] is not None and row["position"] == row["ab_position"]
                             for row in selections),
        valid_ab_pairs=sum(row["position"] is not None and row["ab_position"] is not None
                           for row in selections),
        full_ab=sum(row["ab_stage"] == "pass" for row in selections),
        variants_passing_all_four=[
            case for case in dict.fromkeys(row["case"] for row in selections)
            if all(row["stage"] == "pass" for row in selections if row["case"] == case)
        ],
    )
    return summary


def run(out):
    rows, plans, adapters, reference = freeze(out)
    controls = choice.check_candidates(out, rows, plans["XY"])
    assert digest(out / "controls.json") == reference["XY"]["controls"]
    matches, summary = {}, {}
    for arm, adapter in adapters.items():
        raw = {layout: choice.evaluate(out, f"{arm}-{layout}", adapter, plan, controls)
               for layout, plan in plans.items()}
        matches[arm] = align(arm, rows, raw, reference)
        summary[arm] = summarize(matches[arm])
    keep(out / "results.json", dict(
        protocol=digest(out / "protocol.json"), controls=digest(out / "controls.json"),
        summary=summary, matches=matches,
    ))
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    run(parser.parse_args().out.resolve())
