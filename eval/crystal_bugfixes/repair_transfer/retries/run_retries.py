"""Continue the twelve frozen rewrite conversations with public Crystal feedback."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib import util
from itertools import product
from pathlib import Path
import sys

transfer_dir = Path(__file__).resolve().parents[1]
# The native runtime imports a different experiment module named "run".
sys.path.insert(0, str(transfer_dir))
spec = util.spec_from_file_location("repair_transfer_source", transfer_dir / "run.py")
transfer = util.module_from_spec(spec)
spec.loader.exec_module(transfer)
sys.path.remove(str(transfer_dir))

PARENT = transfer.DEFAULT_OUTPUT
OUT = PARENT.with_name("fix-transfer-retries-v1")
CASES = ("autograd-mean-gradient-element-count", "qwen-image21-conditioning-mask-binary",
         "trace-declined-sentinel")
SEAL = "732ac608e6f54c361d4a8410745c4ed6418a001e65651b3064825aa8340e8d84"
BASELINE = "89b3f50ddd483aa6026c675f07aa0bd84b2bb90815ae2e3b1ddfdb6a51015d7f"
read, write = transfer.read, transfer.write
data, runner = transfer.data, transfer.runner


def keep(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Existing evidence differs at {path}. Use a new output directory.")
    else:
        write(path, value)


def feedback(public):
    if public["stage"] in ("compile_failed", "tests_failed"):
        return data.failure_message(dict(candidate_stage=public["stage"],
                                         candidate_diagnostic=public["diagnostic"]))
    if public["stage"] in ("format_failed", "length"):
        return public["diagnostic"] + "\n\nRevise the same hole. Return only the replacement line."
    raise ValueError(f"No retry feedback for {public['stage']}")


def checked_request(messages, case_id):
    req = dict(id=case_id, messages=messages, sampling=transfer.SAMPLING)
    count_row = dict(id=case_id, messages=messages + [dict(role="assistant", content="")])
    count = data.token_count_rows([count_row])[0]["prefix_tokens"]
    if count + transfer.MAX_NEW > 4096:
        raise ValueError(f"{case_id}: {count} prefix tokens + {transfer.MAX_NEW} exceeds 4K")
    return req, count


def evaluate(case, design, response, loc):
    raw = response["answer"]
    answer = data.normalize_answer(raw) if response["stop"] == "eos" else None
    if answer is None:
        limited = response["stop"] != "eos"
        result = dict(stage="length" if limited else "format_failed", compiled=None,
                      target=None, regressions=None, passed=None, total=None,
                      diagnostic=("The response reached the output limit." if limited else
                                  "The response did not contain one insertable line for the hole."))
        return dict(raw=raw, answer=None, public=result, full=result)
    return dict(raw=raw, answer=answer,
                public=transfer.score(data, case, design, answer, hidden=False, dest=loc / "checks-public"),
                full=transfer.score(data, case, design, answer, hidden=True, dest=loc / "checks-full"))


def sample(worker, recovery, req, loc):
    artifact = loc / "generation.json"
    key = transfer.fingerprint(req)
    keep(loc / "request.json", req)
    if not artifact.exists():
        _, response, retry = runner.generate_with_recovery(worker, req, loc, recovery)
        write(artifact, dict(request_sha256=key, response=response, recovered=bool(retry)))
    gen = read(artifact)
    assert gen["request_sha256"] == key
    response = gen["response"]
    runner.verify_sampling(req, response)
    assert response["id"] == req["id"] and 0 <= response["tokens"] <= transfer.MAX_NEW
    return response


def setup(out):
    assert runner.digest(PARENT / "seal.json") == SEAL
    assert runner.digest(PARENT / "results.json") == BASELINE
    old_seal = read(PARENT / "seal.json")
    assert transfer.pins() == old_seal["files"]
    assert all(runner.digest(transfer.HERE / f) == v for f, v in old_seal["code"].items())
    assert old_seal["max_total_tokens"] == 4096 and old_seal["max_new_tokens"] == transfer.MAX_NEW
    assert old_seal["sampling"] == transfer.SAMPLING
    keep(out / "protocol.json", dict(parent_seal=SEAL, parent_results=BASELINE, cases=list(CASES),
         pins=old_seal["files"], budget=4096, max_new=transfer.MAX_NEW, sampling=transfer.SAMPLING,
         additional_retries=2, stop="first public pass; full-suite success reported separately",
         feedback="append previous raw reply and measured public diagnostic, preserve full dialogue",
         tokens=runner.digest(data.TOKEN_COUNTER), runner=runner.digest(Path(__file__)),
         replay_helpers=runner.digest(Path(data.__file__)), native_helpers=runner.digest(Path(runner.__file__))))
    cases = {r["id"]: r["case"] for r in data.read_jsonl(transfer.DATA / "prepared.jsonl")}
    archived = {(r["arm"], r["context"], r["case"]): r for r in read(PARENT / "results.json")["rows"]
                if r["mode"] == "rewrite" and r["case"] in CASES}
    assert len(archived) == 12

    def prepare(case_id):
        frozen = PARENT / "fixtures/rewrite" / case_id
        case, design = cases[case_id], read(frozen / "design.json")
        entry = old_seal["entries"][case_id]["rewrite"]
        assert transfer.fingerprint(design) == entry["design"]
        control = out / "controls" / case_id
        teacher = transfer.score(data, case, design, design["teacher"], hidden=True, dest=control / "teacher")
        broken = transfer.score(data, case, design, design["baseline"], hidden=True, dest=control / "broken")
        assert teacher["stage"] == "pass" and broken["compiled"] and broken["target"] is False
        write(control / "results.json", dict(teacher=teacher, broken=broken))
        for arm, ctx in product(transfer.ARMS, repeat=2):
            old = archived[arm, ctx, case_id]
            initial_req = read(frozen / (ctx + ".json"))
            generation = read(PARENT / "runs/rewrite" / arm / ctx / case_id / "generation.json")
            assert transfer.fingerprint(initial_req) == generation["request_sha256"] == entry["requests"][ctx]
            assert initial_req["sampling"] == transfer.SAMPLING
            runner.verify_sampling(initial_req, generation["response"])
            assert old["raw"] == generation["response"]["answer"]
            assert old["answer"] == data.normalize_answer(old["raw"]) and old["stage"] != "pass"
            loc = out / "runs" / arm / ctx / case_id / "attempt-0"
            public = transfer.score(data, case, design, old["answer"], hidden=False, dest=loc / "checks-public")
            assert public["stage"] == old["stage"] and public["stage"] in ("compile_failed", "tests_failed")
            full = {k: old[k] for k in ("stage", "compiled", "target", "regressions", "passed", "total", "diagnostic")}
            keep(loc / "result.json", dict(raw=old["raw"], answer=old["answer"], public=public, full=full))
        return case_id, (case, design)

    with ThreadPoolExecutor(max_workers=2) as pool:
        return dict(pool.map(prepare, CASES))


def run(out):
    fixtures = setup(out)
    _, native, recovery = runner.load_runtime()
    rows = []
    for arm in transfer.ARMS:
        dest = out / "runs" / arm
        adapter = transfer.EXPERIMENT / "train" / arm / "adapter.bin"
        worker = runner.ArmWorker(native.ev.evaluator, [str(runner.SAMPLED_NATIVE), "generate",
                                  str(runner.MODEL), str(adapter), str(transfer.MAX_NEW), "4"],
                                  dest / "worker", dest / "worker.log", 240)
        try:
            for case_id, ctx in product(CASES, transfer.ARMS):
                case, design = fixtures[case_id]
                messages = read(PARENT / "fixtures/rewrite" / case_id / (ctx + ".json"))["messages"]
                lane = dest / ctx / case_id
                previous = read(lane / "attempt-0/result.json")
                attempts = [previous]
                for number in (1, 2):
                    if previous["public"]["stage"] == "pass":
                        break
                    messages = messages + [dict(role="assistant", content=previous["raw"]),
                                           dict(role="user", content=feedback(previous["public"]))]
                    req, tokens = checked_request(messages, case_id)
                    loc = lane / f"attempt-{number}"
                    response = sample(worker, recovery, req, loc)
                    previous = dict(prefix_tokens=tokens, **evaluate(case, design, response, loc))
                    write(loc / "result.json", previous)
                    attempts.append(previous)
                    full = previous["full"]
                    print(arm, ctx, case_id, number, full["stage"], full["passed"], full["total"], flush=True)
                rows.append(dict(arm=arm, context=ctx, case=case_id, attempts=attempts))
        finally:
            assert worker.close() == 0
    assert len(rows) == 12
    write(out / "results.json", dict(protocol_sha256=runner.digest(out / "protocol.json"), rows=rows))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=OUT)
    run(cli.parse_args().output.resolve())
