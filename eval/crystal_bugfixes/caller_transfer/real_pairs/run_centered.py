"""Compare centered and ordinary paired choice CE on the same native QLoRA schedule."""

import argparse
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
from shutil import copyfile
from statistics import mean
import subprocess

import run as paired

archive = paired.archive
digest, keep, read = paired.digest, paired.keep, paired.read
REFERENCE = paired.OUT.with_name("real-choice-pairs-v1-attempt2")
OUT = paired.OUT.with_name("real-choice-pairs-centered-v1")
REFERENCE_PROTOCOL = "f2f30b03bab310b965f6d0d4051bb9e13ba9604556202cc039aad4f0274caa93"
REFERENCE_RESULTS = "b2531921256f6b8caf4d90c2b1fe3404be69051b79d05bdf9a4a9b7cf85f5fc0"
CENTER_WEIGHT = 0.25


def freeze(out, trainer, binary):
    assert digest(REFERENCE / "protocol.json") == REFERENCE_PROTOCOL
    assert digest(REFERENCE / "results.json") == REFERENCE_RESULTS
    protocol, results = (read(REFERENCE / name) for name in ("protocol.json", "results.json"))
    assert results["protocol"] == REFERENCE_PROTOCOL
    assert digest(REFERENCE / "training-audit.json") == results["training"]
    audit = read(REFERENCE / "training-audit.json")
    assert audit["protocol"] == REFERENCE_PROTOCOL
    assert digest(REFERENCE / "adapter.bin") == audit["adapter"]
    assert digest(REFERENCE / "paired-logits.json") == results["logits"]["paired"]
    assert digest(paired.MODEL) == protocol["model"] == paired.fit.MODEL_SHA
    assert digest(paired.INITIAL) == protocol["initial"] == paired.fit.INITIAL_SHA
    assert len(protocol["training_pairs"]) == 9
    assert subprocess.check_output(["crystal", "--version"], text=True).strip() == protocol["crystal"]

    shared = {path: sha for path, sha in protocol["files"].items()
              if Path(path).name not in {"train", "run.py", "train_choice_pairs.cr"}}
    for path, sha in shared.items():
        assert digest(Path(path)) == sha, path
    assert digest(binary) == next(sha for path, sha in shared.items() if Path(path).name == "readout")
    views, cases, plans = paired.load_holdout()
    out.mkdir(parents=True)
    for name, sha in protocol["inputs"].items():
        assert digest(REFERENCE / name) == sha, name
        if name != "controls.json":
            copyfile(REFERENCE / name, out / name)

    with redirect_stdout(io.StringIO()):
        controls = archive.choice.check_candidates(out, cases, plans)
    assert controls == read(REFERENCE / "controls.json")
    assert digest(out / "controls.json") == protocol["inputs"]["controls.json"]
    here = Path(__file__).resolve().parent
    files = [trainer, binary, here / "run.py", here / "train_choice_pairs.cr",
             here / "choice_pair_center.cr", Path(__file__).resolve(), REFERENCE / "adapter.bin",
             Path(archive.__file__).resolve(), archive.OUT / "plan.json"]
    keep(out / "protocol.json", dict(
        protocol, files={**shared, **{str(path): digest(path) for path in files}},
        reference=dict(protocol=REFERENCE_PROTOCOL, results=REFERENCE_RESULTS, adapter=audit["adapter"]),
        loss="mean two-choice CE + center_weight * mean(logit(X) - logit(Y))^2 within each pair",
        center_weight=CENTER_WEIGHT, criterion=paired.CRITERION,
    ))
    print(f"Froze nine pairs and the exact 50-batch schedule; rebuilt all {2 * len(cases)} holdout candidates", flush=True)
    return views, controls, results


def train(out, trainer, rows):
    paired.verify_inputs(out)
    adapter, log = out / "adapter.bin", out / "train.log"
    command = [trainer, paired.MODEL, paired.INITIAL, out / "train.jsonl", out / "schedule.json",
               adapter, str(CENTER_WEIGHT)]
    print(f"Native training with center weight {CENTER_WEIGHT}; live log {log}", flush=True)
    with log.open("x") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=paired.prior.REPO, check=True)

    paired.audit_training(out, rows)
    events = paired.fit._events(log)
    assert events[0]["center_weight"] == CENTER_WEIGHT
    penalties = []
    for event in (e for e in events if e["event"] == "update"):
        center = event["center"]
        b = mean((center["log_odds_a"], center["log_odds_b"]))
        assert math.isclose(center["mean_log_odds"], b, rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(center["scale"], CENTER_WEIGHT * center["mean_log_odds"], rel_tol=1e-6, abs_tol=1e-7)
        expected = CENTER_WEIGHT * center["mean_log_odds"] ** 2
        assert math.isclose(center["penalty"], expected, abs_tol=1e-12)
        ce_loss = mean((event["loss_a"], event["loss_b"]))
        assert math.isclose(event["ce_loss"], ce_loss, abs_tol=1e-12)
        assert math.isclose(event["loss"], ce_loss + expected, abs_tol=1e-12)
        penalties.append(center["penalty"])
    assert len(penalties) == paired.UPDATES and any(p > 0 for p in penalties)
    print(f"Audited all {len(penalties)} centered losses; mean penalty {mean(penalties):.6f}", flush=True)
    return adapter


def train_scores(rows, schedule, raw):
    presentations, first = {}, {}
    for left, right in schedule:
        a, b = rows[left], rows[right]
        name, variant = a["id"].rsplit("/", 1)
        assert variant == "0" and b["id"] == name + "/1" and a["pair"] == b["pair"]
        if name in presentations:
            continue
        assert a["messages"][-1]["content"] != b["messages"][-1]["content"]
        x_first = a["messages"][-1]["content"] == "X"
        s_a, s_b = (archive.raw_margin(raw["train/" + row["id"]]) * (1 if x_first else -1)
                    for row in (a, b))
        presentations[name] = dict(s_a=s_a, s_b=s_b, center=mean((s_a, s_b)),
                                   separation=(s_a - s_b) / 2, both_correct=s_a > 0 > s_b)
        first.setdefault(a["pair"], name)
    assert len(presentations) == 36 and len(first) == 9
    return dict(presentations=presentations, first_presentation=first)


def evaluate(out, binary, adapter, views, controls, reference, rows):
    assert digest(adapter) == read(out / "training-audit.json")["adapter"]
    expected = read(REFERENCE / "paired-logits.json")
    first = expected[views[0]["view"]]
    label_ids = (first["x_id"], first["y_id"])
    train_views = [dict(view="train/" + row["id"], prefix_tokens=row["prefix_tokens"],
                        request=dict(id="repair", messages=row["messages"][:-1], sampling=paired.transfer.SAMPLING))
                   for row in rows]
    schedule = read(out / "schedule.json")
    comparisons, fitted = {}, {}
    for arm, path in (("paired", REFERENCE / "adapter.bin"), ("centered", adapter)):
        raw = paired.measure(binary, path, arm, views + train_views, out)
        assert {(o["x_id"], o["y_id"]) for o in raw.values()} == {label_ids}
        if arm == "paired":
            assert all(raw[key] == value for key, value in expected.items())
        keep(out / f"{arm}-logits.json", raw)
        annotated = [dict(v, archived={arm: paired.letter(raw[v["view"]], sampled=True)}) for v in views]
        comparisons[arm] = archive.analyze(annotated, controls, {arm: raw})[arm]
        if arm == "paired":
            assert comparisons[arm] == reference["holdout"]["paired"]
        fitted[arm] = train_scores(rows, schedule, raw)
        for wording in archive.ARCHIVES:
            score = comparisons[arm][wording]
            print(f"{arm} {wording}: full={score['full']}/6, "
                  f"sampled={score['sampled_full']}/24, argmax={score['argmax_full']}/24", flush=True)

    def correct(arm, wording, family):
        return comparisons[arm][wording]["families"][family]["both_correct"]

    gains = [family for family, kind in archive.FAMILIES.items() if kind == "flip"
             and all(correct("centered", w, family) for w in archive.ARCHIVES)
             and not all(correct("paired", w, family) for w in archive.ARCHIVES)]
    preserves = all(correct("centered", w, "outbox_pending_count") for w in archive.ARCHIVES)
    keep(out / "results.json", dict(
        protocol=digest(out / "protocol.json"), training=digest(out / "training-audit.json"),
        logits={arm: digest(out / f"{arm}-logits.json") for arm in comparisons},
        holdout=comparisons, training_probe=fitted,
        transfer=dict(criterion=paired.CRITERION, flip_gains=gains, outbox_preserved=preserves,
                      reached=bool(gains) and preserves),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trainer", type=Path)
    parser.add_argument("readout", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out, trainer, binary = args.out.resolve(), args.trainer.resolve(), args.readout.resolve()
    views, controls, reference = freeze(out, trainer, binary)
    rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    adapter = train(out, trainer, rows)
    evaluate(out, binary, adapter, views, controls, reference, rows)


if __name__ == "__main__":
    main()
