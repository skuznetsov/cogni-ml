"""Measure label bias and line preferences on both archived X/Y instructions."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
import io
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean
import subprocess

import paired_batch_instruction_order as instruction

xy, choice, prior = instruction.xy, instruction.choice, instruction.prior
runner, transfer = prior.runner, prior.transfer
digest, keep, read = instruction.digest, instruction.keep, instruction.read
OUT = xy.OUT.with_name("paired-batch-choice-logits-v1")
ARCHIVES = {
    "X or Y": (xy.OUT, instruction.REFERENCE_PROTOCOL, instruction.REFERENCE_RESULTS),
    "Y or X": (instruction.OUT,
               "c058ce13a07dc2aa455cc98ef4117d40967d923b566adf9d453647cd48489ad8",
               "92ded3258ecbea2a78b3dffceab8f4081c6e43c92f82bf0061554eaa19a919d6"),
}
FAMILIES = {
    "cursor_checkpoint_order": "flip",
    "zero_based_slot_limit": "flip",
    "outbox_pending_count": "invariant",
}


def archived_views(path, protocol_sha, results_sha, wording, rows, adapters):
    assert digest(path / "protocol.json") == protocol_sha
    assert digest(path / "results.json") == results_sha
    protocol, result = (read(path / name) for name in ("protocol.json", "results.json"))
    assert result["protocol"] == protocol_sha
    assert digest(path / "plan.json") == protocol["plan"]
    assert digest(path / "controls.json") == protocol["controls"] == result["controls"]
    assert protocol["adapters"] == adapters
    assert protocol["model"] == digest(runner.MODEL)
    assert protocol["sampler"] == digest(runner.SAMPLED_NATIVE)
    assert protocol["context"] == 4096 and protocol["max_new"] == 128
    assert protocol["sampling"] == transfer.SAMPLING
    plans = read(path / "plan.json")
    views = []
    for key, layout, teacher in product(rows, xy.LAYOUTS, "XY"):
        case = plans[layout][key]
        family = rows[key]["family"]
        canonical = plans[layout][f"{family}/a"]["patches"]["teacher"]
        order = case["orders"][teacher]
        request = order["request"]
        prompt, _, options = request["messages"][-1]["content"].rpartition("\n\n")
        assert prompt.endswith(f"Reply with only {wording}.")
        displayed = {label: case["patches"][origin] for label, origin in order["choices"].items()}
        assert list(displayed) == list(layout) and displayed[teacher] == rows[key]["teacher"]
        assert options == "\n".join(f"{label}: {line}" for label, line in displayed.items())
        canonical_label, = (label for label, line in displayed.items() if line == canonical)
        assert (canonical_label == teacher) == (rows[key]["variant"] == "a" or FAMILIES[family] == "invariant")
        assert request["sampling"] == protocol["sampling"]
        assert order["prefix_tokens"] + protocol["max_new"] <= protocol["context"]
        fingerprint = transfer.fingerprint(request)
        archived = {}
        for arm, adapter in adapters.items():
            folder = path / "runs" / f"{arm}-{layout}"
            assert read(folder / "pins.json") == dict(adapter=adapter, protocol=protocol_sha)
            run = folder / key / f"teacher-{teacher}"
            assert read(run / "request.json") == request
            generation = read(run / "generation.json")
            assert generation["request_sha256"] == fingerprint and not generation["recovered"]
            response = generation["response"]
            assert response["sampling"] == dict(mode="top_k", **protocol["sampling"])
            assert response["tokens"] == 1 and response["stop"] == "eos"
            letter = response["answer"].strip()
            assert letter in ("X", "Y")
            archived[arm] = letter
        views.append(dict(view=f"{wording}/{key}/{layout}/teacher-{teacher}", wording=wording,
                          case=key, layout=layout, teacher=teacher, canonical_label=canonical_label,
                          canonical_slot=layout.index(canonical_label) + 1, prefix_tokens=order["prefix_tokens"],
                          request=request, request_sha256=fingerprint, archived=archived))
    return protocol, plans, views


def freeze(out, binary, adapters):
    fixtures = prior.OUT / "cases.json"
    assert digest(prior.OUT / "protocol.json") == choice.REFERENCE_PROTOCOL
    assert read(prior.OUT / "protocol.json")["inputs"]["cases.json"] == digest(fixtures)
    rows = {prior.case_id(row): row for row in read(fixtures) if row["family"] in FAMILIES}
    assert set(rows) == {f"{family}/{variant}" for family, variant in product(FAMILIES, "ab")}
    assert all(FAMILIES[row["family"]] == row["kind"] for row in rows.values())
    adapter_hashes = {arm: digest(path) for arm, path in adapters.items()}
    protocols, plans, cohorts = {}, {}, {}
    for wording, archive in ARCHIVES.items():
        protocols[wording], plans[wording], cohorts[wording] = archived_views(*archive, wording, rows, adapter_hashes)
    before, after = ARCHIVES
    assert protocols[before]["controls"] == protocols[after]["controls"]
    for former, current in zip(cohorts[before], cohorts[after], strict=True):
        assert all(former[k] == current[k] for k in ("case", "layout", "teacher", "canonical_label", "prefix_tokens"))
        assert instruction.reverse_instruction({"request": former["request"]})["request"] == current["request"]
    views = cohorts[before] + cohorts[after]
    assert len(views) == len({v["request_sha256"] for v in views}) == 48
    for wording, family, layout, slot in product(ARCHIVES, FAMILIES, xy.LAYOUTS, (1, 2)):
        matched = [v["request"]["messages"][-1]["content"].rsplit("\n\n", 1)[1] for v in cohorts[wording]
                   if v["case"].startswith(f"{family}/") and v["layout"] == layout and v["canonical_slot"] == slot]
        assert len(matched) == 2 and matched[0] == matched[1]
    keep(out / "plan.json", views)
    reference = protocols[before]
    sources = (Path(__file__), prior.HERE / "choice_first_token.cr", prior.REPO / "build/bridge.o",
               prior.REPO / "src/ml/gguf/qwen35_qlora.cr",
               transfer.SCALE / "student-errors/eval_qwen35_sampler.cr")
    keep(out / "protocol.json", dict(
        archives={w: dict(protocol=p, results=r) for w, (_, p, r) in ARCHIVES.items()},
        binary=digest(binary), model=reference["model"], adapters=adapter_hashes, sampler=reference["sampler"],
        context=reference["context"], reserve=reference["max_new"], sampling=reference["sampling"],
        compiler=subprocess.check_output(["crystal", "--version"], text=True).strip(),
        sources={str(p.relative_to(prior.REPO)): digest(p) for p in sources},
        fixtures=digest(fixtures), archived_controls=reference["controls"], plan=digest(out / "plan.json"),
        families=FAMILIES, canonical="variant a's passing line, fixed across both variants",
        bias="mean raw X minus Y logit across the four balanced views",
        line_preference="mean X minus Y logit signed toward the canonical line, across the four views",
        criterion="positive then negative for flip pairs and positive twice for invariant pairs; exact ties abstain",
    ))
    return rows, plans[before]["XY"], views


def measure(binary, adapter, views, arm):
    payload = "".join(json.dumps(dict(v["request"], view=v["view"])) + "\n" for v in views)
    command = [str(binary), str(runner.MODEL), str(adapter)]
    proc = subprocess.run(command, input=payload, capture_output=True, check=True, text=True, cwd=prior.REPO)
    readouts = {}
    for view, raw in zip(views, proc.stdout.splitlines(), strict=True):
        row = json.loads(raw)
        assert row["view"] == view["view"] and row["prefix_tokens"] == view["prefix_tokens"]
        assert all(math.isfinite(row[name]) for name in ("x_logit", "y_logit"))
        label_ids = {"X": row["x_id"], "Y": row["y_id"]}
        assert row["x_id"] != row["y_id"]
        assert row["sampled_id"] == label_ids[view["archived"][arm]], (view["view"], arm, row)
        readouts[view["view"]] = row
    assert len({(r["x_id"], r["y_id"]) for r in readouts.values()}) == 1
    return readouts


def raw_margin(row):
    return row["x_logit"] - row["y_logit"]


def estimate(group, readouts, arm, controls):
    scores = {}
    for v in group:
        d = raw_margin(readouts[v["view"]])
        scores[v["layout"], v["canonical_slot"]] = d
    assert scores.keys() == set(product(xy.LAYOUTS, (1, 2)))
    signed_by_slot = {str(slot): (scores["XY", slot] - scores["YX", slot]) * (3 - 2 * slot) / 2
                      for slot in (1, 2)}
    s = mean(signed_by_slot.values())
    canonical_is_teacher, = {v["teacher"] == v["canonical_label"] for v in group}
    if s == 0:
        origin = None
    else:
        origin = "teacher" if (s > 0) == canonical_is_teacher else "foil"
    return dict(b=mean(scores.values()), s=s, s_by_slot=signed_by_slot,
                row_bias=mean(d if layout == "XY" else -d for (layout, _), d in scores.items()),
                canonical_is_teacher=canonical_is_teacher, chosen=origin, outcome=controls.get(origin),
                sampled_full=sum(v["archived"][arm] == v["teacher"] for v in group),
                argmax_full=sum(
                    raw_margin(readouts[v["view"]]) * (1 if v["teacher"] == "X" else -1) > 0 for v in group
                ))


def analyze(views, controls, readouts):
    summaries = {}
    before, after = ARCHIVES
    groups = {(w, case): [v for v in views if (v["wording"], v["case"]) == (w, case)]
              for w, case in product(ARCHIVES, controls)}
    for arm, raw in readouts.items():
        summaries[arm] = {}
        for wording in ARCHIVES:
            estimates = {case: estimate(groups[wording, case], raw, arm, controls[case]) for case in controls}
            outcomes = [r["outcome"] for r in estimates.values() if r["outcome"] is not None]
            families = {}
            for family, kind in FAMILIES.items():
                a, b = (estimates[f"{family}/{v}"] for v in "ab")
                families[family] = dict(kind=kind, s_a=a["s"], s_b=b["s"], delta_b_minus_a=b["s"] - a["s"],
                                        both_correct=(a["s"] > 0 and (b["s"] < 0 if kind == "flip" else b["s"] > 0)))
            summaries[arm][wording] = dict(
                mean_b=mean(r["b"] for r in estimates.values()),
                mean_row_bias=mean(r["row_bias"] for r in estimates.values()),
                sampled_full=sum(r["sampled_full"] for r in estimates.values()),
                argmax_full=sum(r["argmax_full"] for r in estimates.values()),
                compiled=sum(r["compiled"] for r in outcomes), targets=sum(r["target"] for r in outcomes),
                regressions=sum(r["regressions"] for r in outcomes), full=sum(r["stage"] == "pass" for r in outcomes),
                abstained=len(estimates) - len(outcomes), families=families, variants=estimates,
            )
        shifts = Counter()
        for case in controls:
            for old, new in zip(groups[before, case], groups[after, case], strict=True):
                delta = raw_margin(raw[new["view"]]) - raw_margin(raw[old["view"]])
                shifts["toward Y" if delta < 0 else "toward X" if delta > 0 else "tied"] += 1
        summaries[arm]["wording_shifts"] = dict(shifts)
    return summaries


def run(binary, out):
    adapters = {"tool_four": prior.experiment.INITIAL}
    adapters.update({arm: prior.OUT / "training" / arm / "attempt-001" / "adapter.bin" for arm in prior.ARMS})
    rows, plans, views = freeze(out, binary, adapters)
    print("Frozen 48 prompts and 144 archived draws", flush=True)
    with redirect_stdout(io.StringIO()):
        controls = choice.check_candidates(out, rows, plans)
    archived_controls = read(xy.OUT / "controls.json")
    assert controls == {case: archived_controls[case] for case in rows}
    print("Rebuilt and tested both candidates for all six variants", flush=True)
    readouts = {}
    for arm, adapter in adapters.items():
        raw = measure(binary, adapter, views, arm)
        repeated = measure(binary, adapter, views[::-1], arm)
        assert repeated == raw, f"{arm} logits depended on request order"
        keep(out / f"{arm}-logits.json", raw)
        readouts[arm] = raw
        print(f"{arm}: matched all 48 archived draws and reproduced logits with reversed requests", flush=True)
    summaries = analyze(views, controls, readouts)
    keep(out / "results.json", dict(
        protocol=digest(out / "protocol.json"), controls=digest(out / "controls.json"),
        readouts={arm: digest(out / f"{arm}-logits.json") for arm in readouts}, summary=summaries,
    ))
    for arm, by_wording in summaries.items():
        for wording in ARCHIVES:
            row = by_wording[wording]
            print(arm, wording, f'b={row["mean_b"]:+.4f}', f'full={row["full"]}/6',
                  {key: (round(r["s_a"], 4), round(r["s_b"], 4)) for key, r in row["families"].items()},
                  flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.binary.resolve(), args.out.resolve())
