"""Freeze and score two transformations of reproduced, seen Qwen 2B repairs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import sys

from grad_mask import build as grad_mask
from profile_trace import build as profile_trace
from scoring import score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCALE = REPO / "build/qlora/scale-2b"
EXPERIMENT = SCALE / "own-vs-peer-v1"
DATA = EXPERIMENT / "data"
DEFAULT_OUTPUT = EXPERIMENT / "diagnostics/fix-transfer-v1"
ARMS = ("own", "peer")
MODES = ("rename", "rewrite", "original")
BUILDERS = {
    "autograd-mean-gradient-element-count": grad_mask,
    "metal-encode-profile-total-stages": profile_trace,
    "qwen-image21-conditioning-mask-binary": grad_mask,
    "trace-declined-sentinel": profile_trace,
}
SAMPLING = {"temperature": 0.8, "top_k": 40, "seed": 101}
MAX_NEW = 128


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


data = load("transfer_data", DATA / "generate_pairs.py")
runner = load("transfer_runner", SCALE / "code-only-qlora-50/eval/runner.py")
read = runner.read_json
write = runner.atomic_json


def fingerprint(obj):
    return sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def pins():
    manifest = read(DATA / "manifest.json")
    paths = {key: DATA / key for key in manifest["data_files"]
             if key in ("own.jsonl", "peer.jsonl", "prepared.jsonl")}
    expected = {key: manifest["data_files"][key] for key in paths}
    native = read(EXPERIMENT / "evaluation/runs/parent/pins.json")
    expected.update(model=native["model"], sampler=native["sampler"])
    paths.update(model=runner.MODEL, sampler=runner.SAMPLED_NATIVE)
    expected.update(read(EXPERIMENT / "train/matched.json")["adapters"])
    paths.update({arm: EXPERIMENT / "train" / arm / "adapter.bin" for arm in ARMS})
    actual = {key: runner.digest(path) for key, path in paths.items()}
    assert actual == expected
    return actual


def original(row, source, fixture, drafts):
    case = row["case"]
    return dict(template=data.apply_replace(source, case["slot"], "<HOLE>"),
                excerpt=data.prompt_excerpt(source, case["slot"]), baseline=case["slot"]["old"],
                drafts=drafts, fixture=fixture, guards="", teacher=case["teacher"],
                description=case["problem"], mechanism="Unchanged archived source and specs.")


def alpha_diagnostic(archived, current, renames, stage):
    """Retain the historical prompt's path/trace noise while validating the live failure."""
    def rename(match):
        return renames.get(match.group(), match.group())

    expected = re.sub(r"[A-Za-z_][A-Za-z_0-9]*", rename, archived)
    if stage == "compile_failed":
        # The compiler recalculates the column and width of its caret after a rename.
        signature = r"(?m)^Error: .+$"
        assert re.findall(signature, expected) == re.findall(signature, current) != []
        return current

    # Compare spec names, triggering assertions, exception type and numeric results.
    def signature(text):
        text = re.sub(r"trace_id=(?:[0-9]+|<[^>]+>)", "trace_id=ID", text)
        text = re.sub(r"host_elapsed_ms=(?:[0-9.]+|<[^>]+>)", "host_elapsed_ms=MS", text)
        return [s.strip() for s in text.splitlines() if re.match(
            r"^\s*(?:\d+\) |Failure/Error:|Expected:|got:|to include:|.*\(ArgumentError\)$)", s)]

    assert signature(expected) == signature(current) != [], (expected, current)
    return expected


def freeze_case(out, case_id, row, pair):
    case = row["case"]
    blob = data.git_blob(case["parent"], case["source"])
    source = data.seeded_source(case, blob)
    fixture = (DATA / row["case_path"]).with_name(case["fixture"]).read_text()
    for value, key in ((source, "seeded_source_sha256"), (fixture, "fixture_sha256")):
        assert data.digest_bytes(value.encode()) == row[key]
    drafts = {arm: pair[arm + "_draft"] for arm in ARMS}
    reference = original(row, source, fixture, drafts)
    entries = {}
    variants = {}
    for mode in MODES:
        dest = out / "fixtures" / mode / case_id
        design = reference if mode == "original" else BUILDERS[case_id](
            case_id, mode, row, source, fixture, drafts.copy())
        assert design["template"].count("<HOLE>") == design["excerpt"].count("<HOLE>") == 1
        assert design["excerpt"] in design["template"]
        assert set(design["drafts"]) == set(ARMS)
        if mode != "original":
            assert design["teacher"] != reference["teacher"]
            assert design["guards"].strip()
        variants[mode] = design
        if mode == "rewrite":
            assert design["excerpt"] != variants["rename"]["excerpt"]
        if mode == "rename":
            assert design["description"] == reference["description"]
        assert set(data.test_names(fixture)) == set(data.test_names(design["fixture"]))
        write(dest / "design.json", design)
        baseline = score(data, case, design, design["baseline"], hidden=False, dest=dest / "baseline")
        teacher = score(data, case, design, design["teacher"], hidden=True, dest=dest / "teacher")
        assert not baseline.get("infrastructure_error") and not teacher.get("infrastructure_error")
        assert baseline["compiled"] and baseline["target"] is False, (case_id, mode, baseline)
        assert teacher["stage"] == "pass" and teacher["target"] and teacher["regressions"], (case_id, mode, teacher)
        controls = {"baseline": baseline, "teacher": teacher}
        if mode == "rewrite":
            for name, other in (("historical_teacher", reference), ("renamed_teacher", variants["rename"])):
                controls[name] = score(data, case, design, other["teacher"], hidden=True, dest=dest / name)
                assert not controls[name].get("infrastructure_error")
                assert controls[name]["stage"] != "pass", (case_id, name, "teacher still works")
        requests = {}
        for ctx in ARMS:
            draft = design["drafts"][ctx]
            replay = score(data, case, design, draft, hidden=False, dest=dest / ("draft-" + ctx))
            assert not replay.get("infrastructure_error")
            assert replay["stage"] in ("compile_failed", "tests_failed"), (case_id, mode, ctx, replay)
            controls["draft_" + ctx] = replay
            base_diag, draft_diag = baseline["diagnostic"], replay["diagnostic"]
            if mode == "rename":
                history = read(EXPERIMENT / "diagnostics/seen-repair" / ctx / ctx / case_id / "request.json")["messages"]
                old_base = history[1]["content"].split("Failure on the original source:\n", 1)[1].split("\n\nFile:", 1)[0]
                old_draft = history[3]["content"].split(" reported:\n", 1)[1].split("\n\nRevise", 1)[0]
                base_diag = alpha_diagnostic(old_base, base_diag, design["renames"], baseline["stage"])
                draft_diag = alpha_diagnostic(old_draft, draft_diag, design["renames"], replay["stage"])
            user = (f"{design['description']}\n\nFailure on the original source:\n{base_diag}\n\n"
                    f"File: {case['source']}\n\n```crystal\n{design['excerpt']}\n```\n\n"
                    "Replace <HOLE> with the complete code for that line. Return only that one line now.")
            feedback = data.failure_message({"candidate_stage": replay["stage"], "candidate_diagnostic": draft_diag})
            messages = [dict(role="system", content=data.SYSTEM), dict(role="user", content=user),
                        dict(role="assistant", content=draft), dict(role="user", content=feedback)]
            requests[ctx] = {"id": case_id, "messages": messages, "sampling": SAMPLING}
            write(dest / (ctx + ".json"), requests[ctx])
        write(dest / "controls.json", controls)
        entries[mode] = dict(design=fingerprint(design), controls=fingerprint(controls),
                             requests={ctx: fingerprint(requests[ctx]) for ctx in ARMS})
        print(mode, case_id, "controls passed", flush=True)
    return case_id, entries


def freeze(out):
    if (out / "seal.json").exists():
        raise RuntimeError("This experiment is already frozen. Choose a fresh output for a new freeze.")
    code = {p.name: runner.digest(p) for p in HERE.glob("*.py")}
    files = pins()
    seen = read(EXPERIMENT / "diagnostics/seen-repair/results.json")["cases"]
    for arm in ARMS:
        exact = {r["case_id"] for r in seen if r["model"] == r["context"] == arm and r["stage"] == "pass" and r["exact"]}
        assert exact == set(BUILDERS)
    rows = {r["id"]: r for r in data.read_jsonl(DATA / "prepared.jsonl")}
    pairs = {r["case_id"]: r for r in read(DATA / "manifest.json")["pair_rows"]}
    with ThreadPoolExecutor(max_workers=2) as executor:
        entries = dict(executor.map(lambda k: freeze_case(out, k, rows[k], pairs[k]), BUILDERS))
    counts_input = []
    for k in BUILDERS:
        for mode in MODES:
            for ctx in ARMS:
                req = read(out / "fixtures" / mode / k / (ctx + ".json"))
                counts_input.append({"id": f"{mode}/{k}/{ctx}", "messages": req["messages"] + [dict(role="assistant", content="")]})
    counts = {r["id"]: r["prefix_tokens"] for r in data.token_count_rows(counts_input)}
    assert len(counts) == 24 and max(counts.values()) + MAX_NEW <= 4096
    assert code == {p.name: runner.digest(p) for p in HERE.glob("*.py")}
    write(out / "seal.json", dict(files=files, code=code, entries=entries, prefix_tokens=counts,
                                  sampling=SAMPLING, max_new_tokens=MAX_NEW, max_total_tokens=4096))
    print("Frozen. Prefix range:", min(counts.values()), max(counts.values()), flush=True)


def generate(out):
    seal = read(out / "seal.json")
    assert pins() == seal["files"]
    assert {p.name: runner.digest(p) for p in HERE.glob("*.py")} == seal["code"]
    cases = {r["id"]: r["case"] for r in data.read_jsonl(DATA / "prepared.jsonl")}
    _, native, recovery = runner.load_runtime()
    results = []
    for mode in MODES:
        for arm in ARMS:
            dest = out / "runs" / mode / arm
            adapter = EXPERIMENT / "train" / arm / "adapter.bin"
            worker = runner.ArmWorker(native.ev.evaluator, [str(runner.SAMPLED_NATIVE), "generate",
                str(runner.MODEL), str(adapter), str(MAX_NEW), "4"], dest / "worker", dest / "worker.log", 240)
            try:
                for k in BUILDERS:
                    frozen = out / "fixtures" / mode / k
                    design = read(frozen / "design.json")
                    entry = seal["entries"][k][mode]
                    assert fingerprint(design) == entry["design"]
                    assert fingerprint(read(frozen / "controls.json")) == entry["controls"]
                    for ctx in ((arm,) if mode == "original" else ARMS):
                        loc = dest / ctx / k
                        loc.mkdir(parents=True, exist_ok=True)
                        req = read(frozen / (ctx + ".json"))
                        req_hash = fingerprint(req)
                        assert req_hash == entry["requests"][ctx] and req["sampling"] == SAMPLING
                        assert seal["prefix_tokens"][f"{mode}/{k}/{ctx}"] + MAX_NEW <= 4096
                        artifact = loc / "generation.json"
                        if not artifact.exists():
                            _, response, recovered = runner.generate_with_recovery(worker, req, loc, recovery)
                            write(artifact, dict(request_sha256=req_hash, response=response, recovered=bool(recovered)))
                        gen = read(artifact)
                        assert gen["request_sha256"] == req_hash
                        response = gen["response"]
                        runner.verify_sampling(req, response)
                        assert response["id"] == k and response["tokens"] <= MAX_NEW
                        answer = data.normalize_answer(response["answer"]) if response["stop"] == "eos" else None
                        if answer is None:
                            result = dict(stage="format_failed" if response["stop"] == "eos" else "length",
                                          compiled=None, target=None, regressions=None, passed=None, total=None)
                        else:
                            result = score(data, cases[k], design, answer, hidden=True, dest=loc / "checks")
                        assert not result.get("infrastructure_error"), (mode, arm, ctx, k, result)
                        results.append(dict(mode=mode, arm=arm, context=ctx, case=k, answer=answer,
                                            raw=response["answer"], exact=answer == design["teacher"],
                                            recovered=gen["recovered"], **result))
                        write(loc / "result.json", results[-1])
                        print(mode, arm, ctx, k, result["stage"], flush=True)
            finally:
                assert worker.close() == 0
    assert len({(r["mode"], r["arm"], r["context"], r["case"]) for r in results}) == len(results) == 40
    write(out / "results.json", dict(seal_sha256=runner.digest(out / "seal.json"), rows=results))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("freeze", "generate"))
    cli.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = cli.parse_args()
    {"freeze": freeze, "generate": generate}[args.action](args.output.resolve())
