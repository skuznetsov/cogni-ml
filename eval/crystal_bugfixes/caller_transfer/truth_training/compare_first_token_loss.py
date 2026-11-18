"""Compare binary loss and full SFT on the original caller-negation holdout."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean
import subprocess

import fit
import fit_first_token as training
import measure_first_token as reference

prior = reference.prior
ARMS = ("parent", "trained", "choice")
LITERALS = ("true", "false")
READOUT_PINS = {
    "protocol.json": "3e5a653ab9dedcb0c876aa66a29fd4e7bc67ddcb0fc2618cbf4a730a608f366c",
    "results.json": "998f968f3c0a01cc6bda8f75d2b1cb26a806b0839815cd5abbb0894d9ca26492",
    "readout": "484808a53fa6e50af58e29147a900ec82643106ef74cc65550dc90370c7cd284",
    "parent-readouts.json": "28bca7824ed203ef3b7d0132c4c16662e182eb97ff6b1edf5d0783db8328275c",
    "trained-readouts.json": "aa040da177ba866a7c76ca782bb8937566c7e436ee3400d43c91899e5edfc407",
}


def run(cases, requests, command):
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    run = subprocess.run(command, input=payload, capture_output=True, text=True, cwd=training.REPO)
    if run.returncode:
        raise RuntimeError(f"{command[1]} exited {run.returncode}: {run.stderr}")
    return {c["id"]: json.loads(line) for c, line in zip(cases, run.stdout.splitlines(), strict=True)}


def validate_readout(case, record):
    assert record["id"] == "repair" and record["prefix_tokens"] == case["prefix_tokens"]
    assert record["prefix_tokens"] + 128 <= 4096
    vocab = record["vocab_size"]
    assert 0 <= record["sampled_id"] < vocab and vocab == 248320
    for literal in LITERALS:
        token = record[literal]
        assert token["id"] == training.TOKEN_IDS[literal] and math.isfinite(token["logit"])
        assert 1 <= token["rank"] <= vocab and 0 < token["raw_probability"] <= 1
    t, f = (record[word] for word in LITERALS)
    assert math.isclose(t["logit"] - f["logit"], math.log(t["raw_probability"] / f["raw_probability"]),
                        rel_tol=0, abs_tol=2e-5)
    assert t["raw_probability"] + f["raw_probability"] <= 1 + 1e-9


def score_choice(cases, requests, adapter, readouts, out):
    command = [prior.runner.SAMPLED_NATIVE, "generate", prior.runner.MODEL, adapter]
    generated = run(cases, requests, [*command, "128", "4"])
    single = run(cases, requests, [*command, "1", "4"])
    scores = {}
    for case in cases:
        idx = case["id"]
        result, first = generated[idx], single[idx]
        for response in (result, first):
            prior.runner.verify_sampling({"sampling": reference.SAMPLING}, response)
            assert response["id"] == "repair"
        assert 0 <= first["tokens"] <= 1 and 0 <= result["tokens"] <= 128
        sampled_id = readouts[idx]["sampled_id"]
        literal = next((l for l in LITERALS if sampled_id == training.TOKEN_IDS[l]), None)
        if literal:
            assert first["tokens"] == 1 and first["answer"] == literal
            assert result["answer"].startswith(literal)
        else:
            assert first["answer"] not in LITERALS
        values = prior.probe.score_truth(prior.probe.normalize(result))
        scores[idx] = dict(values=values, correct=values == case["truth"], raw=result["answer"],
                           stop=result["stop"], tokens=result["tokens"])
    fit._write_new(out / "choice-generations.json", dict(full=generated, first_token=single))
    return scores


def conditional(record, target):
    other = "false" if target == "true" else "true"
    log_odds = record[target]["logit"] - record[other]["logit"]
    mass = record[target]["raw_probability"]
    return mass / (mass + record[other]["raw_probability"]), log_odds


def summarize(cases, cases_scores, readouts, errors):
    pairs = reference.pair_rows(cases, readouts, errors)
    for row in pairs:
        for arm in ARMS:
            direct, inverted = row["direct"], row["inverted"]
            target = row["target"]
            before, base = conditional(readouts[arm][direct], target)
            after, logged = conditional(readouts[arm][inverted], target)
            row["arms"][arm].update(
                p_direct=before, p_inverted=after, delta=after - before, delta_log_odds=logged - base,
                both_argmax_correct=base < 0 < logged,
                both_sampled_correct=all(cases_scores[arm][idx]["correct"] for idx in (direct, inverted)),
            )
    summaries = {}
    for arm in ARMS:
        summary = prior.summary([dict(c, **cases_scores[arm][c["id"]]) for c in cases])
        obs = [row["arms"][arm] for row in pairs]
        assert summary["pairs"] == sum(r["both_sampled_correct"] for r in obs)
        summary["first_choice_pairs"] = sum(r["both_argmax_correct"] for r in obs)
        summary["mean_boolean_mass"] = mean(
            sum(r[t]["raw_probability"] for t in LITERALS) for r in readouts[arm].values())
        summary["boolean_first_samples"] = sum(
            r["sampled_id"] in training.TOKEN_IDS.values() for r in readouts[arm].values())
        groups = {
            "all": obs,
            "prior_errors": [r["arms"][arm] for r in pairs if r["previous_error"]],
            **{t: [r["arms"][arm] for r in pairs if r["target"] == t] for t in LITERALS},
        }
        summary["shifts"] = {
            label: dict(n=len(values), grew=sum(r["delta"] > 0 for r in values),
                        mean_direct=mean(r["p_direct"] for r in values),
                        mean_inverted=mean(r["p_inverted"] for r in values),
                        mean_delta=mean(r["delta"] for r in values),
                        mean_delta_log_odds=mean(r["delta_log_odds"] for r in values),
                        mean_raw_delta=mean(r["raw_probability"]["delta_correct"] for r in values))
            for label, values in groups.items()
        }
        summaries[arm] = summary
    return summaries, pairs


def compare(out):
    archived = reference.OUT
    for name, sha in READOUT_PINS.items():
        assert fit._digest(archived / name) == sha, f"archived readout changed: {name}"
    cases, requests, old, errors, adapters = reference.archived_inputs()
    readouts = {arm: fit._json(archived / f"{arm}-readouts.json") for arm in reference.ARMS}
    for case in cases:
        for arm in reference.ARMS:
            reference.verify_readout(readouts[arm][case["id"]], case, old[arm, case["id"]])
    adapters["choice"] = out / "adapter.bin"
    audit_file = out / "training-audit.json"
    audit = fit._json(audit_file)
    assert audit["adapter_sha256"] == fit._digest(adapters["choice"])
    assert audit["protocol_sha256"] == fit._digest(out / "fit-protocol.json")
    assert audit["train_log_sha256"] == fit._digest(out / "train.log")
    assert len(audit["visits"]) == audit["supervised_target_tokens"] == fit.UPDATES

    protocol_file = out / "comparison-protocol.json"
    protocol = dict(archived_readouts=READOUT_PINS, archived_scores=reference.SCORES_SHA,
                    context=4096, max_new=128, sampling=reference.SAMPLING,
                    adapters={arm: fit._digest(path) for arm, path in adapters.items()},
                    training_audit=fit._digest(audit_file),
                    readout=str(archived / "readout"),
                    model=fit._digest(prior.runner.MODEL),
                    sampler=fit._digest(prior.runner.SAMPLED_NATIVE),
                    script=fit._digest(Path(__file__)),
                    requests=[prior.transfer.fingerprint(r) for r in requests],
                    previous_errors=errors,
                    primary="both original sampled two-literal answers correct in each sign pair",
                    diagnostic="change in log odds for the inverted target at the first answer slot")
    fit._write_new(protocol_file, protocol)
    command = [archived / "readout", "first_token", prior.runner.MODEL, adapters["choice"]]
    readouts["choice"] = run(cases, requests, command)
    repeated = run(cases[::-1], requests[::-1], command)
    assert readouts["choice"] == repeated
    for case in cases:
        validate_readout(case, readouts["choice"][case["id"]])
    fit._write_new(out / "choice-readouts.json", readouts["choice"])
    scores = {arm: {c["id"]: old[arm, c["id"]] for c in cases} for arm in reference.ARMS}
    scores["choice"] = score_choice(cases, requests, adapters["choice"], readouts["choice"], out)
    metrics, pairs = summarize(cases, scores, readouts, errors)
    expected = fit._json(reference.PRIMARY / "results.json")["summary"]
    for arm in reference.ARMS:
        for key, value in expected[arm]["holdout"].items():
            assert metrics[arm][key] == value
    fit._write_new(out / "results.json", dict(protocol_sha256=fit._digest(protocol_file),
                  summary=metrics, pairs=pairs, choice_scores=scores["choice"], reverse_readout_match=True))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=training.OUT)
    args = cli.parse_args()
    compare(args.output.resolve(strict=True))
