"""Reverse the order of X and Y in the answer instruction alone."""

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

import paired_batch_xy as xy

choice, prior, native, transfer = xy.choice, xy.prior, xy.native, xy.transfer
digest, keep, read = xy.digest, xy.keep, xy.read
OUT = xy.OUT.with_name("paired-batch-choice-xy-instruction-yx-v1")
REFERENCE_PROTOCOL = "b6ea8eefb7522c56ddc57a83badba0aaede04fab35042e2a9674dcd38d004b07"
REFERENCE_RESULTS = "816372b6c1fb4a2ec3603f4416ffb5e034a14b44acbcb8d3550bf5719c55a432"
INSTRUCTION = "Reply with only Y or X."


def reverse_instruction(order):
    revised = deepcopy(order)
    user = revised["request"]["messages"][-1]
    assert user["role"] == "user"
    prefix, separator, options = user["content"].rpartition("\n\n")
    assert prefix.endswith(xy.NEW_RULE)
    user["content"] = prefix.removesuffix(xy.NEW_RULE) + INSTRUCTION + separator + options
    return revised


def freeze(out):
    assert digest(xy.OUT / "protocol.json") == REFERENCE_PROTOCOL
    assert digest(xy.OUT / "results.json") == REFERENCE_RESULTS
    protocol, baseline = (read(xy.OUT / name) for name in ("protocol.json", "results.json"))
    assert baseline["protocol"] == REFERENCE_PROTOCOL
    assert digest(xy.OUT / "plan.json") == protocol["plan"]
    assert digest(xy.OUT / "controls.json") == baseline["controls"] == protocol["controls"]
    rows, original, adapters, _ = xy.freeze(xy.OUT)
    plans = {}
    orders = []
    for layout, cases in original.items():
        plans[layout] = {}
        for key, case in cases.items():
            changed = {label: reverse_instruction(order) for label, order in case["orders"].items()}
            plans[layout][key] = dict(case, orders=changed)
            orders.extend(changed.values())
    assert len(orders) == len({transfer.fingerprint(order["request"]) for order in orders}) == 32
    prompts = [dict(id=str(index), messages=order["request"]["messages"] + [dict(role="assistant", content="")])
               for index, order in enumerate(orders)]
    for order, counted in zip(orders, native.data.token_count_rows(prompts), strict=True):
        prefix = counted["prefix_tokens"]
        assert prefix + protocol["max_new"] <= protocol["context"]
        order["prefix_tokens"] = prefix
    keep(out / "plan.json", plans)
    keep(out / "protocol.json", dict(
        protocol, reference_protocol=REFERENCE_PROTOCOL, reference_results=REFERENCE_RESULTS,
        plan=digest(out / "plan.json"), driver=digest(Path(__file__)),
        instruction_before=xy.NEW_RULE, instruction_after=INSTRUCTION,
    ))
    print("Frozen thirty-two prompts with only the instruction's label order reversed", flush=True)
    return rows, plans, adapters, baseline


def align(before, raw):
    after = {layout: {(row["case"], row["teacher_label"]): row for row in samples}
             for layout, samples in raw.items()}
    assert len(before) == 16 and all(len(samples) == 16 for samples in after.values())
    result = []
    for match in before:
        key, position = match["case"], match["teacher_position"]
        sides = {}
        for layout, samples in after.items():
            teacher = layout[position - 1]
            old, new = match[layout], samples.pop((key, teacher))
            assert old["teacher_label"] == teacher
            label = new["label"]
            slot = layout.index(label) + 1 if label else None
            if label and old["label"]:
                assert (label == old["label"]) == (slot == old["position"])
            sides[layout] = dict(new, position=slot, before_label=old["label"],
                                 before_position=old["position"], before_stage=old["stage"])
        a, b = (sides[layout] for layout in xy.LAYOUTS)
        if a["label"] is None or b["label"] is None:
            kept = "invalid"
        elif a["label"] == b["label"]:
            assert a["position"] != b["position"]
            kept = a["label"]
        else:
            assert a["position"] == b["position"]
            kept = "first" if a["position"] == 1 else "second"
        result.append(dict(case=key, teacher_position=position,
                           before_across_assignments=match["across_assignments"],
                           across_assignments=kept, **sides))
    assert all(not samples for samples in after.values())
    return result


def summarize(matches):
    selections = [match[layout] for match in matches for layout in xy.LAYOUTS]
    summary = prior.summarize(selections)
    summary.update(
        requests=summary.pop("variants"),
        before_labels=dict(Counter(row["before_label"] or "invalid" for row in selections)),
        labels=dict(Counter(row["label"] or "invalid" for row in selections)),
        transitions=dict(Counter(f"{row['before_label']}->{row['label']}" for row in selections)),
        before_full=sum(row["before_stage"] == "pass" for row in selections),
        by_layout={layout: prior.summarize([m[layout] for m in matches]) for layout in xy.LAYOUTS},
        across_assignments=dict(Counter(m["across_assignments"] for m in matches)),
        assignment_transitions=dict(Counter(
            f"{m['before_across_assignments']}->{m['across_assignments']}" for m in matches
        )),
        variants_passing_all_four=[
            case for case in dict.fromkeys(row["case"] for row in selections)
            if all(row["stage"] == "pass" for row in selections if row["case"] == case)
        ],
    )
    return summary


def run(out):
    rows, plans, adapters, baseline = freeze(out)
    controls = choice.check_candidates(out, rows, plans["XY"])
    assert digest(out / "controls.json") == baseline["controls"]
    matches, summary = {}, {}
    for arm, adapter in adapters.items():
        raw = {layout: choice.evaluate(out, f"{arm}-{layout}", adapter, plan, controls)
               for layout, plan in plans.items()}
        matches[arm] = align(baseline["matches"][arm], raw)
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
