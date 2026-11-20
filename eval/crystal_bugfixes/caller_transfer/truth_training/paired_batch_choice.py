"""Compare sampled A/B repair selection on the frozen paired-batch transfer cases."""

import argparse
from collections import Counter
from pathlib import Path
import subprocess

import paired_batch_transfer as prior
import probe

native, runner, transfer = prior.native, prior.runner, prior.transfer
digest, keep, read = prior.digest, prior.keep, prior.read
REFERENCE = prior.OUT
OUT = REFERENCE.with_name("paired-batch-choice-v1")
REFERENCE_PROTOCOL = "1948146f79fd0ef030076af17afef4b3e2b9e4fa2748be4495ca902dda4bc610"
REFERENCE_RESULTS = "4af529809a9d739981ad48efb83bac57f3847b2ac357ad6ec1a17f13a252c060"
LABELS = ("A", "B")
CHOICE = "Choose the line that fills <HOLE> and meets the requirement. Reply with only A or B."


def freeze(out):
    assert digest(REFERENCE / "protocol.json") == REFERENCE_PROTOCOL
    assert digest(REFERENCE / "results.json") == REFERENCE_RESULTS
    protocol, baseline = (read(REFERENCE / name) for name in ("protocol.json", "results.json"))
    assert baseline["protocol"] == REFERENCE_PROTOCOL
    for name in ("cases.json", "controls.json"):
        assert digest(REFERENCE / name) == protocol["inputs"][name], name

    adapters = {"tool_four": prior.experiment.INITIAL}
    for arm in prior.ARMS:
        adapters[arm] = REFERENCE / "training" / arm / "attempt-001" / "adapter.bin"
    assert {arm: digest(path) for arm, path in adapters.items()} == baseline["adapters"]
    assert digest(runner.MODEL) == protocol["model_sha256"]
    assert digest(runner.SAMPLED_NATIVE) == protocol["sampler"]
    settings = protocol["evaluation"]
    assert settings["context"] == 4096 and settings["max_new"] == transfer.MAX_NEW == 128
    assert settings["sampling"] == transfer.SAMPLING

    rows = {prior.case_id(row): row for row in read(REFERENCE / "cases.json")}
    assert list(rows) == settings["cases"] and len(rows) == 8
    plans = {}
    for key, row in rows.items():
        if row["kind"] == "flip":
            other = "b" if row["variant"] == "a" else "a"
            foil_name = "opposite"
            foil = rows[f"{row['family']}/{other}"]["teacher"]
        else:
            foil_name, foil = "regression_trap", row["regression_trap"]
        assert foil != row["teacher"] and row["template"].count("<HOLE>") == 1, key
        patches = {"teacher": row["teacher"], "foil": foil}
        plans[key] = dict(patches=patches, foil_source=foil_name, orders={})
        for teacher_label, origins in zip(LABELS, (("teacher", "foil"), ("foil", "teacher")), strict=True):
            choices = dict(zip(LABELS, origins, strict=True))
            options = "\n".join(f"{label}: {patches[origin]}" for label, origin in choices.items())
            user = f"{row['issue']}\n\n{probe.source(row['template'])}\n\n{CHOICE}\n\n{options}"
            request = dict(id="repair", messages=probe.messages(user), sampling=transfer.SAMPLING)
            plans[key]["orders"][teacher_label] = dict(choices=choices, request=request)

    requests = [order for plan in plans.values() for order in plan["orders"].values()]
    assert len(requests) == len({transfer.fingerprint(order["request"]) for order in requests}) == 16
    count_rows = [
        dict(id=str(index), messages=order["request"]["messages"] + [dict(role="assistant", content="")])
        for index, order in enumerate(requests)
    ]
    for order, counted in zip(requests, native.data.token_count_rows(count_rows), strict=True):
        prefix = counted["prefix_tokens"]
        assert prefix + transfer.MAX_NEW <= settings["context"]
        order["prefix_tokens"] = prefix
    keep(out / "plan.json", plans)

    helpers = [
        Path(probe.__file__), Path(native.__file__), Path(transfer.__file__),
        Path(runner.__file__), Path(native.data.__file__), native.data.TOKEN_COUNTER,
        Path(transfer.score.__code__.co_filename),
    ]
    hashes = {str(path.resolve()): digest(path) for path in helpers}
    assert all(protocol["files"][path] == sha256 for path, sha256 in hashes.items())
    keep(out / "protocol.json", dict(
        reference_protocol=REFERENCE_PROTOCOL, reference_results=REFERENCE_RESULTS,
        adapters=baseline["adapters"], model=protocol["model_sha256"], sampler=protocol["sampler"],
        plan=digest(out / "plan.json"), driver=digest(Path(__file__)), helpers=hashes,
        context=settings["context"], max_new=transfer.MAX_NEW, sampling=transfer.SAMPLING,
        parse="EOS and exactly A or B after trimming surrounding whitespace",
        crystal=subprocess.check_output(["crystal", "--version"], text=True).splitlines()[0],
    ))
    return rows, plans, adapters, baseline


def check_candidates(out, rows, plans):
    archived = read(REFERENCE / "controls.json")
    scores = {}
    for key, row in rows.items():
        plan = plans[key]
        scores[key] = {}
        for origin, patch in plan["patches"].items():
            result = prior.observed(probe.score_patch(row, patch, out / "controls" / key / origin))
            old = archived[key]["teacher" if origin == "teacher" else plan["foil_source"]]
            assert result == old, (key, origin, result)
            assert result["compiled"] and (result["stage"] == "pass") == (origin == "teacher")
            scores[key][origin] = result
    keep(out / "controls.json", scores)
    print("Verified sixteen candidate lines, exactly one full-suite pass in each pair", flush=True)
    return scores


def evaluate(out, arm, adapter, plans, controls):
    folder = out / "runs" / arm
    keep(folder / "pins.json", dict(adapter=digest(adapter), protocol=digest(out / "protocol.json")))
    command = [str(runner.SAMPLED_NATIVE), "generate", str(runner.MODEL), str(adapter),
               str(transfer.MAX_NEW), str(len(prior.fit.LAYERS))]
    worker = recovery = None
    results = []
    try:
        for key, plan in plans.items():
            for teacher_label, order in plan["orders"].items():
                loc = folder / key / f"teacher-{teacher_label}"
                if worker is None and not (loc / "generation.json").exists():
                    _, runtime, recovery = runner.load_runtime()
                    worker = runner.ArmWorker(
                        runtime.ev.evaluator, command, folder / "worker", folder / "worker.log", 240
                    )
                response = native.sample(worker, recovery, order["request"], loc)
                answer = response["answer"].strip()
                origin = order["choices"].get(answer) if response["stop"] == "eos" else None
                if origin is None:
                    score = prior.observed({})
                    score["stage"] = "length" if response["stop"] == "limit" else "format_failed"
                else:
                    score = controls[key][origin]
                results.append(dict(
                    case=key, family=key.split("/")[0], teacher_label=teacher_label,
                    response=response["answer"], stop=response["stop"], tokens=response["tokens"],
                    label=answer if origin else None, chosen=origin, **score,
                    recovered=read(loc / "generation.json")["recovered"],
                ))
                print(f"{arm} {key} teacher={teacher_label}: {answer!r} {score['stage']}", flush=True)
    finally:
        if worker is not None:
            assert worker.close() == 0
    return results


def summarize(rows, cases):
    summary = prior.summarize(rows)
    per_case = {key: [row for row in rows if row["case"] == key] for key in cases}
    assert all([row["teacher_label"] for row in pair] == list(LABELS) for pair in per_case.values())
    summary.update(
        requests=summary.pop("variants"),
        labels=dict(Counter(row["label"] or "invalid" for row in rows)),
        full_by_teacher_label={
            label: sum(row["stage"] == "pass" and row["teacher_label"] == label for row in rows)
            for label in LABELS
        },
        variants_by_orders_passed=dict(Counter(
            str(sum(row["stage"] == "pass" for row in pair)) for pair in per_case.values()
        )),
    )
    return summary


def run(out, preflight_only):
    rows, plans, adapters, baseline = freeze(out)
    controls = check_candidates(out, rows, plans)
    if preflight_only:
        return
    results = {arm: evaluate(out, arm, path, plans, controls) for arm, path in adapters.items()}
    summary = {arm: summarize(samples, rows) for arm, samples in results.items()}
    keep(out / "results.json", dict(
        protocol=digest(out / "protocol.json"), controls=digest(out / "controls.json"),
        free_generation=baseline["summary"], summary=summary, rows=results,
    ))
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.out.resolve(), args.preflight_only)
