"""Measure flag reasoning, selection of a method to inspect, and executable repair."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess
import sys
from textwrap import indent

import fixtures

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "repair_transfer/retries"))
import run_retries as native

transfer, runner = native.transfer, native.runner
OUT = transfer.EXPERIMENT / "diagnostics/caller-transfer-v1"
SYSTEM = "You are maintaining a Crystal codebase. Follow the requested answer format."
PATCH = "Fill <HOLE> in the helper. Return only the replacement line of Crystal code."
TRUTH = ("For the corrected helper, what Boolean must it return for argument 0 and then for "
         "argument 1 to meet the public requirement? Return only two Boolean literals in that "
         "order, separated by a comma.")
ACTION = ("You can inspect one method before making your edit, or proceed immediately. "
          "Reply only OPEN followed by its exact method name, or PROCEED.")


def context(row):
    return f"{row['issue']}\n\nHelper to repair:\n```crystal\n{row['helper']}\n```"


def messages(user):
    return [dict(role="system", content=SYSTEM), dict(role="user", content=user)]


def source(body):
    return f"Method source:\n```crystal\n{body}\n```"


def selection_prompt(row):
    signatures = [body.strip().splitlines()[0].strip() for body in row["methods"].values()]
    return messages(context(row) + "\n\nFile symbol index:\n" + "\n".join(signatures) + "\n\n" + ACTION)


def normalize(response):
    if response["stop"] == "eos":
        return native.data.normalize_answer(response["answer"])
    return None


def score_truth(answer):
    match = re.fullmatch(r"(true|false)\s*,\s*(true|false)", (answer or "").lower())
    return [word == "true" for word in match.groups()] if match else None


def score_patch(row, answer, loc):
    if answer is None:
        return dict(stage="format_or_length", compiled=None, target=None, regressions=None, passed=None)
    case = dict(source="spec/subject.cr", target_tests=row["target_tests"])
    return transfer.score(native.data, case, row, answer, hidden=True, dest=loc)


def controls(out, row):
    dest = out / "controls" / row["family"] / row["variant"]
    scores = {key: score_patch(row, row[key], dest / key) for key in ("teacher", "opposite", "broken")}
    assert scores["teacher"]["stage"] == "pass"
    for key in ("opposite", "broken"):
        assert scores[key]["compiled"] and scores[key]["target"] is False, (dest, key, scores[key])
    assert set(row["target_tests"]) <= set(scores["opposite"]["failed_specs"])
    native.write(dest / "results.json", scores)


def validate_pairs(rows):
    families = {r["family"] for r in rows}
    assert len(families) == 2 and len(rows) == 4
    correct_positions = set()
    for family in families:
        a, b = sorted([r for r in rows if r["family"] == family], key=lambda r: r["variant"])
        assert (a["variant"], b["variant"]) == ("direct", "inverted")
        for key in ("helper", "caller", "fixture", "issue", "broken", "target_tests"):
            assert a[key] == b[key], (family, key)
        assert a["truth"] in ([False, True], [True, False]) and b["truth"] == a["truth"][::-1]
        assert a["teacher"] == b["opposite"] and b["teacher"] == a["opposite"]
        assert list(a["methods"]) == list(b["methods"])
        assert [k for k in a["methods"] if a["methods"][k] != b["methods"][k]] == [a["caller"]]
        assert a["caller"] not in a["issue"]
        body = indent(a["methods"][a["caller"]], "  ")
        flipped = indent(b["methods"][a["caller"]], "  ")
        assert a["template"].count(body) == 1
        assert flipped.count("!") == body.count("!") + 1 and flipped.replace("!", "", 1) == body
        assert a["template"].replace(body, flipped) == b["template"]
        assert selection_prompt(a) == selection_prompt(b)
        correct_positions.add(list(a["methods"]).index(a["caller"]))
    assert len(correct_positions) == len(families)
    assert len({tuple(r["truth"]) for r in rows if r["variant"] == "direct"}) == 2


def freeze(out, rows, adapters):
    pins = transfer.pins()
    parent = runner.digest(adapters["parent"])
    assert parent == native.read(transfer.EXPERIMENT / "evaluation/runs/parent/pins.json")["adapter"]
    pins["parent"] = parent
    for key in ("own", "peer"):
        audit = native.read(transfer.EXPERIMENT / "train" / key / "audit.json")
        assert audit["inputs"][str(adapters["parent"])] == parent
        assert audit["adapter_sha256"] == pins[key]
    validate_pairs(rows)
    files = [Path(__file__), Path(fixtures.__file__), Path(native.__file__), Path(transfer.__file__),
             transfer.HERE / "scoring.py", Path(runner.__file__), Path(native.data.__file__),
             native.data.TOKEN_COUNTER]
    protocol = dict(pins=pins, files={str(p): runner.digest(p) for p in files},
                    fixtures=transfer.fingerprint(rows), sampling=transfer.SAMPLING,
                    context=4096, max_new=transfer.MAX_NEW,
                    lanes=["truth", "shown_repair", "table_repair", "action", "search_repair"],
                    action_budget=1, repair_attempts=1, arms=list(adapters),
                    crystal=subprocess.check_output(["crystal", "--version"], text=True).splitlines()[0])
    native.keep(out / "protocol.json", protocol)
    native.keep(out / "fixtures.json", rows)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda r: controls(out, r), rows))


def evaluate(row, worker, recovery, dest):
    def generate(lane, dialogue):
        # The serving ID is constant across variants. Only the message bodies reach the model.
        request, count = native.checked_request(dialogue, "repair")
        response = native.sample(worker, recovery, request, dest / lane)
        return dict(raw=response["answer"], parsed=normalize(response), prefix_tokens=count,
                    stop=response["stop"])

    def repair(lane, dialogue):
        reply = generate(lane, dialogue)
        reply["score"] = score_patch(row, reply["parsed"], dest / lane / "checks")
        return reply

    shown = context(row) + "\n\n" + source(row["methods"][row["caller"]])
    truth = generate("truth", messages(shown + "\n\n" + TRUTH))
    truth["values"] = score_truth(truth["parsed"])
    truth["correct"] = truth["values"] == row["truth"]
    shown_repair = repair("shown_repair", messages(shown + "\n\n" + PATCH))
    a, b = [str(v).lower() for v in row["truth"]]
    table = f"Required helper result at argument 0: {a}. At argument 1: {b}."
    table_repair = repair("table_repair", messages(context(row) + "\n\n" + table + "\n\n" + PATCH))
    dialogue = selection_prompt(row)
    action = generate("action", dialogue)
    choice = action["parsed"] or ""
    symbol = choice.removeprefix("OPEN ").strip()
    opened = symbol if choice.startswith("OPEN ") and symbol in row["methods"] else None
    action.update(opened=opened, valid=opened is not None or choice == "PROCEED",
                  correct=opened == row["caller"])
    returned = source(row["methods"][opened]) if opened else "No method source was opened."
    followup = dialogue + [dict(role="assistant", content=action["raw"]),
                           dict(role="user", content=returned + "\n\n" + PATCH)]
    search_repair = repair("search_repair", followup)
    return dict(truth=truth, shown_repair=shown_repair, table_repair=table_repair,
                action=action, search_repair=search_repair)


def run(out):
    adapters = {"parent": runner.ORIGINAL_ADAPTER,
                **{arm: transfer.EXPERIMENT / "train" / arm / "adapter.bin" for arm in transfer.ARMS}}
    rows = fixtures.rows()
    freeze(out, rows, adapters)
    print("Fixture pairs, executable controls and file hashes verified.", flush=True)
    _, runtime, recovery = runner.load_runtime()
    results = []
    for arm, adapter in adapters.items():
        dest = out / "runs" / arm
        command = [str(runner.SAMPLED_NATIVE), "generate", str(runner.MODEL), str(adapter),
                   str(transfer.MAX_NEW), "4"]
        worker = runner.ArmWorker(runtime.ev.evaluator, command, dest / "worker", dest / "worker.log", 240)
        try:
            for row in rows:
                loc = dest / row["family"] / row["variant"]
                result = evaluate(row, worker, recovery, loc)
                result.update(arm=arm, family=row["family"], variant=row["variant"])
                native.write(loc / "result.json", result)
                results.append(result)
                print(arm, row["family"], row["variant"], result["truth"]["correct"],
                      result["action"]["opened"], result["search_repair"]["score"]["stage"], flush=True)
        finally:
            assert worker.close() == 0
    native.write(out / "results.json", dict(protocol_sha256=runner.digest(out / "protocol.json"), rows=results))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=OUT)
    run(cli.parse_args().output.resolve())
