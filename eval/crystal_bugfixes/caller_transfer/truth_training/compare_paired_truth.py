"""Compare paired-margin SFT with sealed controls on all 16 sign pairs."""

import argparse
import json
from pathlib import Path
from statistics import mean

import compare_first_token_loss as compare
import fit
import fit_first_token as choice
import fit_paired_truth as training

reference, prior = compare.reference, compare.prior
CHOICE_PINS = {
    "results.json": "bf47f72830d4c70e1388c20cacc9fea013cbaa4617f2e17c6418a8137e003750",
    "comparison-protocol.json": "56023bd4c68f1356e2b370b1b514bd4f580009be28118fceb20f62ba34184ab1",
    "choice-readouts.json": "b1cd0bcffed07e1e44ee00af9084be5c663d61acc65b01daf9280daafcc253c1",
}


def controls(cases, old):
    for directory, pins in ((reference.OUT, compare.READOUT_PINS), (choice.OUT, CHOICE_PINS)):
        for name, sha in pins.items():
            assert fit._digest(directory / name) == sha, f"archived control changed: {name}"
    readouts = {arm: fit._json(reference.OUT / f"{arm}-readouts.json") for arm in reference.ARMS}
    scores = {arm: {c["id"]: old[arm, c["id"]] for c in cases} for arm in reference.ARMS}
    archived = fit._json(choice.OUT / "results.json")
    assert archived["protocol_sha256"] == CHOICE_PINS["comparison-protocol.json"]
    scores["choice"] = archived["choice_scores"]
    readouts["choice"] = fit._json(choice.OUT / "choice-readouts.json")
    for case in cases:
        idx = case["id"]
        for arm in reference.ARMS:
            reference.verify_readout(readouts[arm][idx], case, scores[arm][idx])
        compare.validate_readout(case, readouts["choice"][idx])
        score = scores["choice"][idx]
        values = prior.probe.score_truth(prior.probe.normalize({"answer": score["raw"], "stop": score["stop"]}))
        assert score["values"] == values and score["correct"] == (values == case["truth"])
    return scores, readouts, archived["summary"]


def summarize(cases, scores, readouts, errors):
    metrics, pairs = compare.summarize(cases, scores, readouts, errors)
    for arm, summary in metrics.items():
        for target, group in summary["shifts"].items():
            selected = [
                row["arms"][arm] for row in pairs
                if target == "all" or (target == "prior_errors" and row["previous_error"]) or row["target"] == target
            ]
            group["inverted_leader_correct"] = sum(row["p_inverted"] > 0.5 for row in selected)
            group["both_leaders_correct"] = sum(row["both_argmax_correct"] for row in selected)
        summary["prior_errors_corrected"] = sum(scores[arm][idx]["correct"] for idx in errors)
        summary["new_errors"] = sum(
            scores["trained"][idx]["correct"] and not score["correct"] for idx, score in scores[arm].items())
    return metrics, pairs


def training_probe(binary, adapters, out):
    path = prior.OUT / "train.jsonl"
    assert fit._digest(path) == training.PINS["train.jsonl"]
    rows = fit._rows(path)
    mates = training.pair_rows(rows)
    requests = [
        {"id": "repair", "messages": row["messages"][:-1], "sampling": reference.SAMPLING} for row in rows
    ]
    counts = fit._json(prior.OUT / "train-counts.json")
    min_margin = training.CONFIG["minimum_margin"]
    report = {}
    for arm in (*reference.ARMS, "paired"):
        records = compare.run(rows, requests, [binary, "first_token", prior.runner.MODEL, adapters[arm]])
        margins = []
        for row, count in zip(rows, counts, strict=True):
            record = records[row["id"]]
            compare.validate_readout(count, record)
            label = row["messages"][-1]["content"].split(",")[0]
            margins.append(compare.conditional(record, label)[1])
        minima = [min(margins[i - 1], margins[j - 1]) for i, j in enumerate(mates, 1) if i < j]
        report[arm] = dict(mean_hinge=mean(max(0, min_margin - m) for m in minima),
                           both_leaders_correct=sum(m > 0 for m in minima),
                           margin_met=sum(m >= min_margin for m in minima),
                           margins={r["id"]: m for r, m in zip(rows, margins, strict=True)})
    fit._write_new(out / "training-readouts.json", report)


def main(out):
    cases, requests, old, errors, adapters = reference.archived_inputs()
    scores, readouts, original = controls(cases, old)
    archived_metrics, _ = compare.summarize(cases, scores, readouts, errors)
    assert archived_metrics == original
    archived_protocol = fit._json(choice.OUT / "comparison-protocol.json")
    assert archived_protocol["requests"] == [prior.transfer.fingerprint(r) for r in requests]
    assert archived_protocol["sampling"] == reference.SAMPLING and archived_protocol["context"] == 4096
    adapters["choice"] = choice.OUT / "adapter.bin"
    assert {arm: fit._digest(p) for arm, p in adapters.items()} == archived_protocol["adapters"]

    adapters["paired"] = out / "adapter.bin"
    audit_file = out / "training-audit.json"
    audit = fit._json(audit_file)
    assert audit["adapter_sha256"] == fit._digest(adapters["paired"])
    assert audit["train_log_sha256"] == fit._digest(out / "train.log")
    assert audit["protocol_sha256"] == fit._digest(out / "fit-protocol.json")
    full_audit = fit._json(prior.OUT / "training/audit.json")
    assert audit["visits"] == full_audit["visits"] and audit["supervised_target_tokens"] == 200
    assert (audit["constraints"], audit["unique_pairs"]) == (50, 16)

    binary = reference.OUT / "readout"
    protocol_path = out / "comparison-protocol.json"
    fit._write_new(protocol_path, dict(
        archived_readouts=compare.READOUT_PINS, archived_choice=CHOICE_PINS, archived_scores=reference.SCORES_SHA,
        adapters={arm: fit._digest(path) for arm, path in adapters.items()}, model=fit.MODEL_SHA,
        context=4096, max_new=128, sampling=reference.SAMPLING, training_audit=fit._digest(audit_file),
        sampler=fit._digest(prior.runner.SAMPLED_NATIVE), readout=fit._digest(binary),
        sources={p.name: fit._digest(p) for p in (Path(__file__), Path(compare.__file__), Path(reference.__file__))},
        case_order=[c["id"] for c in cases], requests=archived_protocol["requests"], previous_errors=errors,
        primary="both original sampled two-literal answers correct in a sign pair",
        diagnostic="shift for inverted target and correct literal leader on both signs",
        training_probe="weaker-target margin on all 16 original training pairs",
    ))
    command = [binary, "first_token", prior.runner.MODEL, adapters["paired"]]
    readouts["paired"] = compare.run(cases, requests, command)
    assert readouts["paired"] == compare.run(cases[::-1], requests[::-1], command)
    for case in cases:
        compare.validate_readout(case, readouts["paired"][case["id"]])
    fit._write_new(out / "paired-readouts.json", readouts["paired"])
    scores["paired"] = compare.score_choice(cases, requests, adapters["paired"], readouts["paired"], out, arm="paired")
    metrics, pairs = summarize(cases, scores, readouts, errors)
    fit._write_new(out / "results.json", dict(protocol_sha256=fit._digest(protocol_path),
                  summary=metrics, pairs=pairs, paired_scores=scores["paired"], reverse_readout_match=True))
    print(json.dumps(metrics, indent=2), flush=True)
    training_probe(binary, adapters, out)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=training.OUT)
    main(cli.parse_args().output.resolve(strict=True))
