"""Train native binary choice on historical Crystal pairs and replay the frozen holdout."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
from itertools import product
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from textwrap import indent

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "truth_training"))
import paired_batch_choice_logits as archive

prior, runner, transfer = archive.prior, archive.runner, archive.transfer
fit, probe, data = prior.fit, archive.choice.probe, prior.native.data
digest, keep, read = archive.digest, archive.keep, archive.read
OUT = archive.OUT.with_name("real-choice-pairs-v1")
INITIAL = prior.experiment.INITIAL
MODEL = runner.MODEL

TRAIN_SEED = 42
UPDATES = 50
TRAIN_WORDING = "X or Y"
LAYOUTS = tuple(product(("XY", "YX"), (0, 1)))
FIXTURES = [HERE / name for name in ("ngram.json", "numerics.json", "runtime.json")]
REFERENCE_PROTOCOL = "720322df68beec24d013772a22605189abf5747d6884290a5c63eceb2f45a11f"
REFERENCE_RESULTS = "b0fdbdb56900a6127de67f6543e6303d925fae7c59e23fc8de60f23989c8f040"
CRITERION = ("At least one of Cursor and Slot gains both correct signs under both wordings "
             "and both Outbox variants retain their correct signs and full-suite passes.")


def git_source(revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=prior.REPO, text=True)


def load_pairs():
    pairs = sorted([pair for path in FIXTURES for pair in read(path)], key=lambda p: p["id"])
    ids = {pair["id"] for pair in pairs}
    assert len(ids) == len(pairs) and pairs
    for pair in pairs:
        key, lines, variants = pair["id"], pair["candidates"], pair["variants"]
        assert re.fullmatch(r"[a-z0-9][a-z0-9_-]+", key) and key not in archive.FAMILIES
        assert len(lines) == len(variants) == 2 and lines[0] != lines[1]
        assert all(line.strip() == line and line and "\n" not in line for line in lines)
        assert variants[0]["template"] != variants[1]["template"]
        for row in variants:
            source = row["source"]
            assert re.fullmatch(r"[0-9a-f]{40}", source["revision"])
            assert row["template"].count("<HOLE>") == 1 and row["issue"].strip()
            assert "<HOLE>" not in row.get("setup", "")
            assert all(row[group].strip() for group in ("target", "regressions"))
            assert source["anchor"] and git_source(source["revision"], source["path"]).count(source["anchor"]) == 1, key
    return pairs


def check_candidate(loc, row, line, compile_env):
    loc.mkdir(parents=True)
    path = loc / "case.cr"
    body = row.get("setup", "") + "\n" + row["template"].replace("<HOLE>", line)
    checks = ['case ARGV[0]?']
    for group in ("target", "regressions"):
        checks += [f'when "{group}"', indent(row[group].rstrip(), "  "), f'  puts "Passed {group}"']
    checks += ['else', '  raise "Unknown test group"', 'end', '']
    path.write_text(body.rstrip() + "\n\n" + "\n".join(checks), encoding="utf-8")
    build = subprocess.run(["crystal", "build", path.name, "-o", "candidate"], cwd=loc, env=compile_env,
                           capture_output=True, text=True, timeout=90)
    (loc / "compile.log").write_text(build.stdout + build.stderr)
    assert build.returncode == 0, f"candidate did not compile: {loc}"
    result = dict(compiled=True)
    for group in ("target", "regressions"):
        command = [str(loc / "candidate"), group]
        test = subprocess.run(command, cwd=loc, capture_output=True, text=True, timeout=15)
        (loc / f"{group}.log").write_text(test.stdout + test.stderr)
        assert (test.returncode == 0 and test.stdout.strip().endswith(f"Passed {group}")) or (
            test.returncode == 1 and "Unhandled exception:" in test.stderr
        ), (loc, group, test.returncode)
        result[group] = test.returncode == 0
    result["full"] = result["target"] and result["regressions"]
    return result


def verify_train_pairs(pairs, out):
    jobs = [(pair, variant, candidate) for pair in pairs for variant, candidate in product(range(2), repeat=2)]
    crystal_path = subprocess.check_output(["crystal", "env", "CRYSTAL_PATH"], text=True).strip()
    compile_env = dict(os.environ, CRYSTAL_PATH=os.pathsep.join((str(prior.REPO), crystal_path)))

    def check(job):
        pair, variant, candidate = job
        key = f"{pair['id']}/{variant}/{candidate}"
        score = check_candidate(out / "train-controls" / key, pair["variants"][variant],
                                pair["candidates"][candidate], compile_env)
        assert score["full"] == (variant == candidate), (key, score)
        return key, score

    with ThreadPoolExecutor(max_workers=4) as executor:
        controls = dict(executor.map(check, jobs))
    keep(out / "train-controls.json", controls)
    print(f"Verified {len(pairs)} real-source pairs, {len(controls)} compiling candidates, one passing line per variant",
          flush=True)


def training_plan(pairs, out):
    rows, schedule = {}, []
    rng = random.Random(TRAIN_SEED)
    offset = {pair["id"]: i % len(LAYOUTS) for i, pair in enumerate(pairs)}
    while len(schedule) < UPDATES:
        order = pairs.copy()
        rng.shuffle(order)
        for pair in order[:UPDATES - len(schedule)]:
            key = pair["id"]
            layout, slot = LAYOUTS[offset[key] % len(LAYOUTS)]
            offset[key] += 1
            label_for = [layout[slot], layout[1 - slot]]
            options = "\n".join(f"{label}: {pair['candidates'][label_for.index(label)]}" for label in layout)
            ids = []
            for variant, row in enumerate(pair["variants"]):
                case_id = f"{key}/{layout}/{slot}/{variant}"
                rule = "Choose the line that fills <HOLE> and meets the requirement."
                prompt = (f"{row['issue']}\n\n{probe.source(row['template'])}\n\n"
                          f"{rule} Reply with only {TRAIN_WORDING}.\n\n{options}")
                messages = probe.messages(prompt) + [dict(role="assistant", content=label_for[variant])]
                entry = dict(id=case_id, pair=key, variant=variant, messages=messages)
                assert rows.setdefault(case_id, entry) == entry
                ids.append(case_id)
            assert len(set(label_for)) == 2
            schedule.append(ids)
    ordered = list(rows.values())
    indices = {row["id"]: i for i, row in enumerate(ordered)}
    assert len({transfer.fingerprint(row["messages"][:-1]) for row in ordered}) == len(ordered)
    counts = data.token_count_rows(ordered)
    for row, count in zip(ordered, counts, strict=True):
        assert count["target_tokens"] == 2 and count["prefix_tokens"] + transfer.MAX_NEW <= 4096
        row["prefix_tokens"] = count["prefix_tokens"]
    prior.experiment.jsonl(out / "train.jsonl", ordered)
    keep(out / "schedule.json", [[indices[i] for i in batch] for batch in schedule])
    return ordered


def load_holdout():
    reference = archive.OUT
    assert digest(reference / "protocol.json") == REFERENCE_PROTOCOL
    assert digest(reference / "results.json") == REFERENCE_RESULTS
    protocol, results = (read(reference / name) for name in ("protocol.json", "results.json"))
    assert protocol["model"] == fit.MODEL_SHA and protocol["adapters"]["tool_four"] == fit.INITIAL_SHA
    assert protocol["context"] == 4096 and protocol["reserve"] == transfer.MAX_NEW == 128
    assert protocol["sampling"] == transfer.SAMPLING
    assert results["protocol"] == REFERENCE_PROTOCOL
    assert digest(reference / "plan.json") == protocol["plan"]
    assert digest(reference / "controls.json") == results["controls"]
    assert digest(reference / "tool_four-logits.json") == results["readouts"]["tool_four"]
    views = read(reference / "plan.json")
    cases_path = prior.OUT / "cases.json"
    assert digest(cases_path) == protocol["fixtures"]
    xy_protocol = read(archive.xy.OUT / "protocol.json")
    assert digest(archive.xy.OUT / "protocol.json") == archive.ARCHIVES[TRAIN_WORDING][1]
    assert digest(archive.xy.OUT / "plan.json") == xy_protocol["plan"]
    plans = read(archive.xy.OUT / "plan.json")["XY"]
    cases = {prior.case_id(row): row for row in read(cases_path) if row["family"] in archive.FAMILIES}
    assert len(views) == 48 and set(cases) == {view["case"] for view in views}
    assert all(v["request"]["sampling"] == transfer.SAMPLING for v in views)
    return views, cases, plans


def freeze(out, trainer, readout):
    assert digest(MODEL) == fit.MODEL_SHA and digest(INITIAL) == fit.INITIAL_SHA
    out.mkdir(parents=True)
    pairs = load_pairs()
    views, cases, plans = load_holdout()
    rows = training_plan(pairs, out)
    heldout_prompts = {transfer.fingerprint(v["request"]["messages"]) for v in views}
    assert heldout_prompts.isdisjoint(transfer.fingerprint(row["messages"][:-1]) for row in rows)
    assert {r["template"] for p in pairs for r in p["variants"]}.isdisjoint(row["template"] for row in cases.values())
    verify_train_pairs(pairs, out)
    with redirect_stdout(io.StringIO()):
        controls = archive.choice.check_candidates(out, cases, plans)
    assert controls == read(archive.OUT / "controls.json")
    print(f"Rebuilt and tested both candidates for each of the {len(cases)} holdout variants", flush=True)
    checker, _ = fit._checker(prior.REPO)
    files = [Path(__file__), HERE / "train_choice_pairs.cr", *FIXTURES, trainer, readout, checker,
             prior.HERE / "choice_first_token.cr", prior.REPO / "build/bridge.o",
             prior.REPO / "src/ml/gguf/qwen35_qlora.cr", data.TOKEN_COUNTER,
             prior.REPO / "src/ml/gguf/reader.cr", prior.REPO / "src/ml/gguf/dequant.cr",
             transfer.SCALE / "student-errors/eval_qwen35_sampler.cr"]
    keep(out / "protocol.json", dict(
        files={str(path): digest(path) for path in files},
        inputs={name: digest(out / name) for name in ("train.jsonl", "schedule.json", "train-controls.json", "controls.json")},
        model=fit.MODEL_SHA, initial=fit.INITIAL_SHA, training_pairs=[p["id"] for p in pairs],
        holdout=dict(protocol=REFERENCE_PROTOCOL, results=REFERENCE_RESULTS, families=archive.FAMILIES),
        context=4096, reserve=transfer.MAX_NEW, sampling=transfer.SAMPLING, train_wording=TRAIN_WORDING,
        rank=fit.RANK, blocks=fit.LAYERS, updates=UPDATES, train_seed=TRAIN_SEED, learning_rate=1e-4,
        loss="mean two-choice first-token CE on both sides, one Adam update using both gradients",
        metric="sign of the mean canonical-line logit margin over all four balanced presentations",
        criterion=CRITERION,
        crystal=subprocess.check_output(["crystal", "--version"], text=True).strip(),
    ))
    return views, rows, controls


def verify_inputs(out):
    protocol = read(out / "protocol.json")
    for name, expected in protocol["files"].items():
        assert digest(Path(name)) == expected, name
    for name, expected in protocol["inputs"].items():
        assert digest(out / name) == expected, name
    assert digest(MODEL) == protocol["model"] and digest(INITIAL) == protocol["initial"]
    return protocol


def train(out, trainer, rows):
    adapter = out / "adapter.bin"
    log = out / "train.log"
    command = [trainer, MODEL, INITIAL, out / "train.jsonl", out / "schedule.json", adapter]
    print(f"Training {UPDATES} paired updates from tool-four; live log {log}", flush=True)
    with log.open("x") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=prior.REPO, check=True)

    audit_training(out, rows)
    return adapter


def audit_training(out, rows):
    protocol = verify_inputs(out)
    adapter, log = out / "adapter.bin", out / "train.log"
    events = fit._events(log)
    start, end = events[0], events[-1]
    assert start["event"] == "start" and end["event"] == "saved"
    assert start["model_sha256"] == protocol["model"] and start["initial_sha256"] == protocol["initial"]
    for field in ("train_seed", "rank", "blocks", "updates", "learning_rate"):
        assert start[field] == protocol[field]
    assert start["projections"] == len(fit.LAYERS) * len(fit.PROJECTIONS)
    updates = [e for e in events if e["event"] == "update"]
    prefills = [e for e in events if e["event"] == "prefill"]
    by_id = {row["id"]: row for row in rows}
    assert len(prefills) == len(rows) and {e["id"] for e in prefills} == by_id.keys()
    labels = start["labels"]
    for e in prefills:
        row = by_id[e["id"]]
        assert e["prefix_tokens"] == row["prefix_tokens"]
        assert e["target_id"] == labels[row["messages"][-1]["content"]]
    schedule = read(out / "schedule.json")
    assert len(updates) == len(schedule) == end["updates"] == UPDATES
    for step, (e, batch) in enumerate(zip(updates, schedule, strict=True), 1):
        a, b = [rows[i] for i in batch]
        expected = [labels[row["messages"][-1]["content"]] for row in (a, b)]
        assert e["step"] == step and e["rows"] == batch and e["target_ids"] == expected
        assert set(expected) == set(labels.values()) and a["pair"] == b["pair"] and (a["variant"], b["variant"]) == (0, 1)
        assert all(math.isfinite(e[k]) and e[k] >= 0 for k in ("loss_a", "loss_b", "loss"))
    assert end["target_tokens"] == 2 * UPDATES and end["train_rows"] == len(rows)

    checker_path, checker = fit._checker(prior.REPO)
    with redirect_stdout(io.StringIO()):
        assert checker.check_adapter(INITIAL) == fit.INITIAL_SHA
        sha = checker.check_adapter(adapter)
    before, after = map(fit._load_factors, (INITIAL, adapter))
    start_index, changes = 0, []
    for block, (projection, width, height) in product(fit.LAYERS, fit.PROJECTIONS):
        end_index = start_index + fit.RANK * (width + height)
        split = start_index + fit.RANK * width
        changed_a = sum(a != b for a, b in zip(before[start_index:split], after[start_index:split], strict=True))
        changed_b = sum(a != b for a, b in zip(before[split:end_index], after[split:end_index], strict=True))
        assert changed_a > 0 and changed_b > 0, (block, projection)
        changes.append(dict(block=block, projection=projection, changed_a=changed_a, changed_b=changed_b))
        start_index = end_index
    assert start_index == len(before) == len(after)
    keep(out / "training-audit.json", dict(
        protocol=digest(out / "protocol.json"), log=digest(log), adapter=sha, checker=digest(checker_path),
        updates=len(updates), train_rows=len(rows), target_tokens=end["target_tokens"], seconds=end["seconds"],
        visits=dict(Counter(rows[step[0]]["pair"] for step in schedule)), factors=changes,
    ))


def measure(binary, adapter, arm, views, out):
    payload = "".join(json.dumps(dict(v["request"], view=v["view"])) + "\n" for v in views)
    with (out / f"{arm}-readout.log").open("x") as log:
        proc = subprocess.run([binary, MODEL, adapter], input=payload, stdout=subprocess.PIPE,
                              stderr=log, text=True, check=True, cwd=prior.REPO)
    raw = {}
    for view, line in zip(views, proc.stdout.splitlines(), strict=True):
        obs = json.loads(line)
        assert obs["view"] == view["view"] and obs["prefix_tokens"] == view["prefix_tokens"]
        assert all(math.isfinite(obs[k]) for k in ("x_logit", "y_logit"))
        raw[view["view"]] = obs
    assert len(raw) == len(views)
    return raw


def letter(obs, sampled):
    if sampled:
        return {obs["x_id"]: "X", obs["y_id"]: "Y"}.get(obs["sampled_id"])
    margin = archive.raw_margin(obs)
    return "X" if margin > 0 else "Y" if margin < 0 else None


def evaluate(out, binary, adapter, views, rows, controls):
    first = {}
    for a, b in read(out / "schedule.json"):
        first.setdefault(rows[a]["pair"], (rows[a], rows[b]))
    train_probe = [
        dict(view="train/" + row["id"], prefix_tokens=row["prefix_tokens"],
             request=dict(id="repair", messages=row["messages"][:-1], sampling=transfer.SAMPLING))
        for batch in first.values() for row in batch
    ]
    aggregates, decisions, fitted = {}, {}, {}
    reference = read(archive.OUT / "tool_four-logits.json")
    assert digest(adapter) == read(out / "training-audit.json")["adapter"]
    training_log = fit._events(out / "train.log")
    labels = training_log[0]["labels"]
    for arm, path in (("tool_four", INITIAL), ("paired", adapter)):
        raw = measure(binary, path, arm, views + train_probe, out)
        assert {(o["x_id"], o["y_id"]) for o in raw.values()} == {(labels["X"], labels["Y"])}
        if arm == "tool_four":
            assert all(raw[key] == value for key, value in reference.items())
        keep(out / f"{arm}-logits.json", raw)
        annotated = [dict(v, archived={arm: letter(raw[v["view"]], sampled=True)}) for v in views]
        aggregates[arm] = archive.analyze(annotated, controls, {arm: raw})[arm]
        decisions[arm] = []
        for v in views:
            obs = raw[v["view"]]
            result = {k: v[k] for k in ("view", "wording", "case", "layout", "teacher", "canonical_slot")}
            for mode in ("argmax", "sampled"):
                chosen = letter(obs, sampled=mode == "sampled")
                origin = None if chosen is None else "teacher" if chosen == v["teacher"] else "foil"
                result[mode] = dict(label=chosen, origin=origin, outcome=controls[v["case"]].get(origin))
            decisions[arm].append(result)

        fitted[arm] = {}
        for key, pair in first.items():
            margins = [archive.raw_margin(raw["train/" + row["id"]])
                       * (1 if row["messages"][-1]["content"] == "X" else -1) for row in pair]
            fitted[arm][key] = dict(margins=margins, both_correct=all(m > 0 for m in margins))
        for wording in archive.ARCHIVES:
            score = aggregates[arm][wording]
            print(f"{arm} {wording}: full={score['full']}/6, "
                  f"sampled={score['sampled_full']}/24, argmax={score['argmax_full']}/24", flush=True)

    flip_families = [name for name, kind in archive.FAMILIES.items() if kind == "flip"]
    gains = [name for name in flip_families if all(
        aggregates["paired"][w]["families"][name]["both_correct"] and
        not aggregates["tool_four"][w]["families"][name]["both_correct"] for w in archive.ARCHIVES)]
    preserves = all(aggregates["paired"][w]["families"]["outbox_pending_count"]["both_correct"] for w in archive.ARCHIVES)
    keep(out / "results.json", dict(
        protocol=digest(out / "protocol.json"), training=digest(out / "training-audit.json"),
        logits={arm: digest(out / f"{arm}-logits.json") for arm in aggregates},
        holdout=aggregates, decisions=decisions, training_probe=fitted,
        transfer=dict(criterion=CRITERION, flip_gains=gains, outbox_preserved=preserves, reached=bool(gains) and preserves),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trainer", type=Path)
    parser.add_argument("readout", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out, trainer, readout = args.out.resolve(), args.trainer.resolve(), args.readout.resolve()
    views, rows, controls = freeze(out, trainer, readout)
    adapter = train(out, trainer, rows)
    evaluate(out, readout, adapter, views, rows, controls)


if __name__ == "__main__":
    main()
