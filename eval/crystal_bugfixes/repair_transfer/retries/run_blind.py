"""Pair neutral retry instructions with the archived diagnostic conversations."""

import argparse
from itertools import product
from pathlib import Path

import run_retries as retries

transfer, runner = retries.transfer, retries.runner
read, write = retries.read, retries.write
REFERENCE = retries.OUT
OUT = REFERENCE.with_name("fix-transfer-blind-retries-v1")
PROTOCOL_SHA = "0b096aa5ed9bece180b3e2fdcd336291eaaa26d8a6ce2eb824abc7ef0b2670ba"
RESULTS_SHA = "41da7ee40dfd3696298bd140614ae434d5d78a4a3269e9d148a1c2fe2d0a8453"
INSTRUCTION = "Revise the same hole. Return only the replacement line."


def freeze(out):
    assert runner.digest(REFERENCE / "protocol.json") == PROTOCOL_SHA
    assert runner.digest(REFERENCE / "results.json") == RESULTS_SHA
    reference = read(REFERENCE / "protocol.json")
    assert runner.digest(Path(retries.__file__)) == reference["runner"]
    assert runner.digest(Path(runner.__file__)) == reference["native_helpers"]
    assert runner.digest(Path(retries.data.__file__)) == reference["replay_helpers"]
    assert runner.digest(retries.data.TOKEN_COUNTER) == reference["tokens"]
    assert transfer.pins() == reference["pins"]
    assert runner.digest(retries.PARENT / "seal.json") == reference["parent_seal"]
    assert runner.digest(retries.PARENT / "results.json") == reference["parent_results"]
    seal = read(retries.PARENT / "seal.json")
    for name, digest in seal["code"].items():
        assert runner.digest(transfer.HERE / name) == digest
    assert seal["sampling"] == reference["sampling"] == transfer.SAMPLING
    assert reference["max_new"] == transfer.MAX_NEW and reference["budget"] == 4096
    retries.keep(out / "protocol.json", dict(
        reference_protocol_sha256=PROTOCOL_SHA, reference_results_sha256=RESULTS_SHA,
        blind_runner_sha256=runner.digest(Path(__file__)), pins=reference["pins"],
        sampling=transfer.SAMPLING, budget=4096, max_new=transfer.MAX_NEW,
        additional_retries=2, instruction=INSTRUCTION, stop="two blind samples per branch",
        history="retain initial request, append this branch's previous raw reply and the fixed instruction",
    ))
    reference_results = read(REFERENCE / "results.json")
    assert reference_results["protocol_sha256"] == PROTOCOL_SHA
    histories = {(r["arm"], r["context"], r["case"]): r["attempts"]
                 for r in reference_results["rows"]}
    assert set(histories) == set(product(transfer.ARMS, transfer.ARMS, retries.CASES))
    return seal["entries"], histories


def run(out):
    entries, histories = freeze(out)
    cases = {r["id"]: r["case"] for r in retries.data.read_jsonl(transfer.DATA / "prepared.jsonl")}
    _, native, recovery = runner.load_runtime()
    rows = []
    for arm in transfer.ARMS:
        dest = out / "runs" / arm
        adapter = transfer.EXPERIMENT / "train" / arm / "adapter.bin"
        worker = runner.ArmWorker(native.ev.evaluator, [str(runner.SAMPLED_NATIVE), "generate",
            str(runner.MODEL), str(adapter), str(transfer.MAX_NEW), "4"],
            dest / "worker", dest / "worker.log", 240)
        try:
            for case_id, ctx in product(retries.CASES, transfer.ARMS):
                frozen = retries.PARENT / "fixtures/rewrite" / case_id
                design = read(frozen / "design.json")
                req = read(frozen / (ctx + ".json"))
                entry = entries[case_id]["rewrite"]
                assert transfer.fingerprint(design) == entry["design"]
                assert transfer.fingerprint(req) == entry["requests"][ctx]
                messages = req["messages"]
                baseline = histories[arm, ctx, case_id][0]
                lane = dest / ctx / case_id
                # Both continuations start at the same model reply and measured outcome.
                retries.keep(lane / "attempt-0/result.json", baseline)
                attempts = [baseline]
                for number in (1, 2):
                    messages = messages + [dict(role="assistant", content=attempts[-1]["raw"]),
                                           dict(role="user", content=INSTRUCTION)]
                    request, tokens = retries.checked_request(messages, case_id)
                    loc = lane / f"attempt-{number}"
                    response = retries.sample(worker, recovery, request, loc)
                    result = retries.evaluate(cases[case_id], design, response, loc)
                    result["prefix_tokens"] = tokens
                    write(loc / "result.json", result)
                    attempts.append(result)
                    full = result["full"]
                    print(arm, ctx, case_id, number, full["stage"], full["passed"], full["total"], flush=True)
                rows.append(dict(arm=arm, context=ctx, case=case_id, attempts=attempts))
        finally:
            assert worker.close() == 0
    write(out / "results.json", dict(protocol_sha256=runner.digest(out / "protocol.json"), rows=rows))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=OUT)
    run(cli.parse_args().output.resolve())
