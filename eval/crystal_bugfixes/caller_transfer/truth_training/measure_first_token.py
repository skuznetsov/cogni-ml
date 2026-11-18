"""Read the first-token distribution on the sealed caller-negation holdout."""

import argparse
from itertools import groupby
import json
import math
from pathlib import Path
from statistics import mean, median
import subprocess

import experiment as prior

HERE, PRIMARY = prior.HERE, prior.OUT
OUT = PRIMARY / "first-token-v1"
digest, read, keep = prior.runner.digest, prior.native.read, prior.native.keep
SCORES_SHA = "bf44b71bba64e13802ccc541fbc3d046fc0b2084809506d46b74bb7e6f9c3535"
SAMPLING = {"temperature": 0.8, "top_k": 40, "seed": 101}
METRICS = ("raw_probability", "sampling_probability")
ARMS = ("parent", "trained")


def archived_inputs():
    assert digest(PRIMARY / "results.json") == SCORES_SHA
    results = read(PRIMARY / "results.json")
    assert results["protocol"] == digest(PRIMARY / "protocol.json")
    protocol = read(PRIMARY / "protocol.json")
    assert protocol["inputs"]["cases.json"] == digest(PRIMARY / "cases.json")
    assert protocol["sampling"] == SAMPLING and protocol["max_new"] == 128 and protocol["context"] == 4096
    assert digest(prior.runner.MODEL) == protocol["pins"]["model"]
    assert digest(prior.runner.SAMPLED_NATIVE) == protocol["pins"]["sampler"]
    adapters = {"parent": prior.INITIAL, "trained": PRIMARY / "training/adapter.bin"}
    assert {arm: digest(path) for arm, path in adapters.items()} == results["adapters"]
    cases = [r for r in read(PRIMARY / "cases.json") if r["split"] == "holdout"]
    prior.validate(cases)
    old = {(r["arm"], r["id"]): r for r in results["rows"] if r["split"] == "holdout"}
    assert len(old) == 64
    requests = []
    for case in cases:
        req = {"id": "repair", "messages": case["messages"], "sampling": SAMPLING}
        for arm in ARMS:
            path = PRIMARY / "runs" / arm / "holdout" / case["id"]
            assert req == read(path / "request.json")
            gen = read(path / "generation.json")
            assert gen["request_sha256"] == prior.transfer.fingerprint(req) and not gen["recovered"]
            response = gen["response"]
            score = old[arm, case["id"]]
            assert response["answer"] == score["raw"] and response["stop"] == score["stop"] == "eos"
            values = prior.probe.score_truth(prior.probe.normalize(response))
            assert values == score["values"] and score["truth"] == case["truth"]
            assert score["correct"] == (values == case["truth"])
            assert score["raw"].startswith(prior.literal(values[0]))
        requests.append(req)
    errors = sorted(c["id"] for c in cases if not old["trained", c["id"]]["correct"])
    assert len(errors) == 10 and all(c["inverted"] for c in cases if c["id"] in errors)
    return cases, requests, old, errors, adapters


def verify_readout(result, case, original):
    assert result["id"] == "repair" and result["prefix_tokens"] == case["prefix_tokens"]
    assert result["prefix_tokens"] + 128 <= 4096
    positive, negative = result["true"], result["false"]
    assert positive["id"] != negative["id"]
    for token in (positive, negative):
        assert 0 <= token["id"] < result["vocab_size"]
        assert 1 <= token["rank"] <= result["vocab_size"] and math.isfinite(token["logit"])
        assert 0 < token["raw_probability"] <= 1
        assert 0 <= token["sampling_probability"] <= 1
        assert (token["sampling_probability"] > 0) == (token["rank"] <= SAMPLING["top_k"])
    for metric in METRICS:
        assert positive[metric] + negative[metric] <= 1 + 1e-9
    margin = positive["logit"] - negative["logit"]
    raw_ratio = math.log(positive["raw_probability"] / negative["raw_probability"])
    assert math.isclose(margin, raw_ratio, abs_tol=2e-5)
    if positive["sampling_probability"] and negative["sampling_probability"]:
        sampled_ratio = math.log(positive["sampling_probability"] / negative["sampling_probability"])
        assert math.isclose(margin / SAMPLING["temperature"], sampled_ratio, abs_tol=2e-5)
    expected = prior.literal(original["values"][0])
    assert result["sampled_id"] == result[expected]["id"], (case["id"], expected, result)


def measure(binary, adapter, cases, payload, old, arm, out):
    proc = subprocess.run([str(binary), "first_token", str(prior.runner.MODEL), str(adapter)],
                          input=payload, text=True, capture_output=True, cwd=prior.transfer.REPO)
    if proc.returncode:
        raise RuntimeError(f"{arm} readout failed: {proc.stderr}")
    results = [json.loads(line) for line in proc.stdout.splitlines()]
    assert len(results) == len(cases)
    for case, result in zip(cases, results, strict=True):
        verify_readout(result, case, old[arm, case["id"]])
    indexed = {case["id"]: result for case, result in zip(cases, results, strict=True)}
    assert len({(r["true"]["id"], r["false"]["id"], r["vocab_size"]) for r in results}) == 1
    keep(out / f"{arm}-readouts.json", indexed)
    return indexed


def check_historical_token(adapter, payload, cases, old, arm, out):
    cmd = [str(prior.runner.SAMPLED_NATIVE), "generate", str(prior.runner.MODEL), str(adapter), "1", "4"]
    run = subprocess.run(cmd, input=payload, capture_output=True, text=True, check=True, cwd=prior.transfer.REPO)
    responses = [json.loads(line) for line in run.stdout.splitlines()]
    records = {}
    for c, r in zip(cases, responses, strict=True):
        prior.runner.verify_sampling({"sampling": SAMPLING}, r)
        expected = prior.literal(old[arm, c["id"]]["values"][0])
        assert r["tokens"] == 1 and r["id"] == "repair" and r["answer"] == expected
        records[c["id"]] = r
    keep(out / f"{arm}-historical-first-token.json", records)


def pair_rows(cases, readouts, errors):
    def pair_key(case):
        return case["family"], case["public_zero"], tuple(case["order"])

    rows = []
    for (family, zero, order), variants in groupby(sorted(cases, key=pair_key), key=pair_key):
        direct, inverted = sorted(variants, key=lambda c: c["inverted"])
        assert not direct["inverted"] and inverted["inverted"] and direct["truth"] == inverted["truth"][::-1]
        target = prior.literal(inverted["truth"][0])
        row = dict(family=family, public_zero=zero, order=list(order), target=target, direct=direct["id"],
                   inverted=inverted["id"], previous_error=inverted["id"] in errors, arms={})
        for arm, outputs in readouts.items():
            a, b = outputs[direct["id"]], outputs[inverted["id"]]
            metrics = {}
            for metric in METRICS:
                probs = {side: {t: obs[t][metric] for t in ("true", "false")}
                         for side, obs in (("direct", a), ("inverted", b))}
                probs["delta_correct"] = probs["inverted"][target] - probs["direct"][target]
                metrics[metric] = probs
            row["arms"][arm] = metrics
        rows.append(row)
    assert len(rows) == 16 and sum(r["previous_error"] for r in rows) == 10
    return rows


def summarize(rows):
    answer = {}
    for arm in ARMS:
        scopes = {}
        for label, group in (("all", rows), ("ten_errors", [r for r in rows if r["previous_error"]])):
            metrics = {}
            for metric in METRICS:
                deltas = [r["arms"][arm][metric]["delta_correct"] for r in group]
                before = [r["arms"][arm][metric]["direct"][r["target"]] for r in group]
                after = [r["arms"][arm][metric]["inverted"][r["target"]] for r in group]
                metrics[metric] = dict(n=len(deltas), grew=sum(x > 0 for x in deltas),
                    fell=sum(x < 0 for x in deltas), tied=sum(x == 0 for x in deltas),
                    mean_direct=mean(before), mean_inverted=mean(after), mean_delta=mean(deltas),
                    median_delta=median(deltas), smallest_abs_delta=min(map(abs, deltas)))
            scopes[label] = metrics
        answer[arm] = scopes
    return answer


def main(binary, out):
    cases, requests, old, errors, adapters = archived_inputs()
    student = prior.transfer.SCALE / "student-errors"
    repo = prior.transfer.REPO
    sources = [Path(__file__), HERE / "first_token.cr", student / "eval_qwen35_sampler.cr", student / "eval_qwen35_native.cr",
               repo / "src/ml/gguf/qwen35_qlora.cr", repo / "build/bridge.o"]
    keep(out / "protocol.json", dict(previous_results=SCORES_SHA, previous_protocol=digest(PRIMARY / "protocol.json"),
        model=digest(prior.runner.MODEL), original_binary=digest(prior.runner.SAMPLED_NATIVE),
        probe_binary=digest(binary), crystal=subprocess.check_output(["crystal", "--version"], text=True).strip(),
        files={str(p.relative_to(repo)): digest(p) for p in sources},
        adapters={k: digest(p) for k, p in adapters.items()}, context=4096, reserve=128,
        sampling=SAMPLING, error_ids=errors, case_order=[c["id"] for c in cases],
        requests=[prior.transfer.fingerprint(r) for r in requests],
        primary="full-vocabulary softmax at temperature one",
        secondary="sampling probability at temperature 0.8 after top-40 filtering",
        comparison="P(inverted target | inverted caller) minus P(same target | direct caller)",
        replay="fresh model process with reversed request order for both arms"))
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    for arm in ARMS:
        check_historical_token(adapters[arm], payload, cases, old, arm, out)
    readouts = {arm: measure(binary, adapters[arm], cases, payload, old, arm, out) for arm in ARMS}
    rows = pair_rows(cases, readouts, errors)
    summary = summarize(rows)
    reverse_payload = "".join(json.dumps(r) + "\n" for r in reversed(requests))
    repeat = {arm: measure(binary, adapters[arm], cases[::-1], reverse_payload, old, arm, out / "repeat")
              for arm in ARMS}
    repeated_rows = pair_rows(cases, repeat, errors)
    status = {}
    for arm in ARMS:
        max_change = max(abs(readouts[arm][idx][t][m] - repeated[t][m])
                         for idx, repeated in repeat[arm].items() for t in ("true", "false") for m in METRICS)
        disagreements = []
        for a, b in zip(rows, repeated_rows, strict=True):
            for m in METRICS:
                delta = a["arms"][arm][m]["delta_correct"]
                delta_repeat = b["arms"][arm][m]["delta_correct"]
                if (delta > 0, delta < 0) != (delta_repeat > 0, delta_repeat < 0):
                    disagreements.append([a["inverted"], m])
        status[arm] = dict(max_probability_change=max_change, direction_disagreements=disagreements)
    keep(out / "results.json", dict(protocol=digest(out / "protocol.json"), rows=rows, summary=summary,
                                     repeat=status))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("binary", type=Path)
    cli.add_argument("--output", type=Path, default=OUT)
    args = cli.parse_args()
    main(args.binary.resolve(), args.output.resolve())
