"""Compare pair center and hinge penalties on the same native QLoRA schedule."""

import argparse
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
from shutil import copyfile
from statistics import mean
from struct import pack, unpack
import subprocess

import run as paired

archive = paired.archive
digest, keep, read = paired.digest, paired.keep, paired.read
OUT = paired.OUT.with_name("real-choice-pairs-centered-v1")
HINGE_OUT = paired.OUT.with_name("real-choice-pairs-centered-hinge-v1")
DIRECTED_OUT = paired.OUT.with_name("real-choice-pairs-directed-hinge-v1")
DIRECTED_CENTER_ONE_OUT = DIRECTED_OUT.with_name("real-choice-pairs-directed-hinge-center-1-v1")
DIRECTED_HINGE_TWO_OUT = DIRECTED_OUT.with_name("real-choice-pairs-directed-hinge-2-center-1-v1")
DIRECTED_CENTER_FOUR_OUT = DIRECTED_OUT.with_name("real-choice-pairs-directed-hinge-2-center-4-v1")
DIRECTED_HINGE_EIGHT_OUT = DIRECTED_OUT.with_name("real-choice-pairs-directed-hinge-8-center-4-v1")
DIRECTED_CENTER_SIXTEEN_OUT = DIRECTED_OUT.with_name("real-choice-pairs-directed-hinge-8-center-16-v1")
DIRECTED_HINGE_THIRTY_TWO_OUT = DIRECTED_OUT.with_name("real-choice-pairs-directed-hinge-32-center-16-v1")
DIRECTED_CENTER_SIXTY_FOUR_OUT = DIRECTED_OUT.with_name("real-choice-pairs-directed-hinge-32-center-64-v1")
DIRECTED_HINGE_ONE_TWENTY_EIGHT_OUT = DIRECTED_OUT.with_name("real-choice-pairs-directed-hinge-128-center-64-v1")
DIRECTED_CENTER_256_OUT = DIRECTED_OUT.with_name("real-choice-pairs-directed-hinge-128-center-256-v1")
DIRECTED_HINGE_512_OUT = DIRECTED_OUT.with_name("real-choice-pairs-directed-hinge-512-center-256-v1")
CANDIDATES = {
    "paired": ("centered", OUT),
    "centered": ("centered_hinge", HINGE_OUT),
    "centered_hinge": ("directed_hinge", DIRECTED_OUT),
    "directed_hinge": ("directed_hinge_center_1", DIRECTED_CENTER_ONE_OUT),
    "directed_hinge_center_1": ("directed_hinge_2_center_1", DIRECTED_HINGE_TWO_OUT),
    "directed_hinge_2_center_1": ("directed_hinge_2_center_4", DIRECTED_CENTER_FOUR_OUT),
    "directed_hinge_2_center_4": ("directed_hinge_8_center_4", DIRECTED_HINGE_EIGHT_OUT),
    "directed_hinge_8_center_4": ("directed_hinge_8_center_16", DIRECTED_CENTER_SIXTEEN_OUT),
    "directed_hinge_8_center_16": ("directed_hinge_32_center_16", DIRECTED_HINGE_THIRTY_TWO_OUT),
    "directed_hinge_32_center_16": ("directed_hinge_32_center_64", DIRECTED_CENTER_SIXTY_FOUR_OUT),
    "directed_hinge_32_center_64": ("directed_hinge_128_center_64", DIRECTED_HINGE_ONE_TWENTY_EIGHT_OUT),
    "directed_hinge_128_center_64": ("directed_hinge_128_center_256", DIRECTED_CENTER_256_OUT),
    "directed_hinge_128_center_256": ("directed_hinge_512_center_256", DIRECTED_HINGE_512_OUT),
}
REFERENCES = {
    "paired": (
        paired.OUT.with_name("real-choice-pairs-v1-attempt2"),
        "f2f30b03bab310b965f6d0d4051bb9e13ba9604556202cc039aad4f0274caa93",
        "b2531921256f6b8caf4d90c2b1fe3404be69051b79d05bdf9a4a9b7cf85f5fc0",
    ),
    "centered": (
        OUT.with_name("real-choice-pairs-centered-v1-attempt2"),
        "2a183b65c62f8063738fa74ee11f0483ed666f18a95dea3434615aacffc6c2ce",
        "8b2be0f9419c3c8111713e8363354787a58ae439919e5fb8f4da7539af718cae",
    ),
    "centered_hinge": (
        HINGE_OUT,
        "2e34ac6455a12f79f1a9423294fd38cc9eba68b0b6d6af6f3fe8a99ac59ce6f9",
        "5b1c2649517d8a7234ce1d365fec047799a429ddb6c7149ab779d4ac43e14b05",
    ),
    "directed_hinge": (
        DIRECTED_OUT,
        "79d688b3e4ff52f989e6256394a5804e879f4dad6cadeddeae8c393eff8ce89e",
        "0aba166d10b79deb6d56f72255706e39c74d37641dc5222c2b53ef506d326764",
    ),
    "directed_hinge_center_1": (
        DIRECTED_CENTER_ONE_OUT,
        "c6064a99a5792cc3377c5f54ac2e31c96810865726876bb4efc3d6284de14135",
        "aef557d5dbbba1a67188a49d1106fa1e61f20db88fe9b82dc08efe0d2ce795ae",
    ),
    "directed_hinge_2_center_1": (
        DIRECTED_HINGE_TWO_OUT,
        "ac9bcca88cd0419225191a72f71f8259b04dfcdde6fb527af1b4049f208f7682",
        "d4de912b7e7129d7a4b0770b016111e0d9bdd8e75bb405312dbd6cad64fd7d28",
    ),
    "directed_hinge_2_center_4": (
        DIRECTED_CENTER_FOUR_OUT,
        "cdd0692f0585875944cf2dbc7c3d043a3f43afa6390c2cb3c01c14f7c07cc9d8",
        "c224df3a630b0cbb80edcc182a40b110e7d11507987edb7f97b6b0c93140105d",
    ),
    "directed_hinge_8_center_4": (
        DIRECTED_HINGE_EIGHT_OUT,
        "4c145e841f6715224c75d9cd69feec0136f7281b6d403b9840cb080368efa98a",
        "3ade8941cb8ca0902e0d5a7eb1c67439e3f5cd8b4f83f529d4e4a8967316fbd2",
    ),
    "directed_hinge_8_center_16": (
        DIRECTED_CENTER_SIXTEEN_OUT,
        "0ccb2c27d73e90e33c250679f20099274146d251daeb254e79cfb0e545b5b00f",
        "8f30fe426ca9c1149d4a06dad824f45f1391023ba91d9b2d43bc0826d3ef8b5c",
    ),
    "directed_hinge_32_center_16": (
        DIRECTED_HINGE_THIRTY_TWO_OUT,
        "833caf584cf6eef2cb816942ad089a13eb7efec95cea4c8b96ddfd2a38f728c8",
        "1f16fa9c182b444fefd2d0c63c7e068f772565d881b0dc0eaff58f274b88f343",
    ),
    "directed_hinge_32_center_64": (
        DIRECTED_CENTER_SIXTY_FOUR_OUT,
        "90c6eb36a38231fc421173af9e4266657cc35ef3a7677ec80670bee4aa146aaa",
        "b2fad49732eb5852676592d68ee8a5be7a640d911a1971bb822d193abe9408c7",
    ),
    "directed_hinge_128_center_64": (
        DIRECTED_HINGE_ONE_TWENTY_EIGHT_OUT,
        "2885a5241a79cc2a64032015698bca6e8bf7c09baf081bd9663f55fdda19cb18",
        "38d1019cc50a0c3751bc475494183a43c24eb14c55a1c3b9c951bd402364968b",
    ),
    "directed_hinge_128_center_256": (
        DIRECTED_CENTER_256_OUT,
        "7227c8d029b12fc604ee9d27313d98ee907ef827165da2620bf4fc4c475d0a66",
        "792b8cff26648bfa6847bad3b5a38f66237d69d9a39c9aeeefcf393260f4b58b",
    ),
}
CENTER_WEIGHT = 0.25
HINGE_WEIGHT = 0.5
MINIMUM_MARGIN = 0.5
DIRECTED_WEIGHTS = {
    "directed_hinge": (CENTER_WEIGHT, HINGE_WEIGHT),
    "directed_hinge_center_1": (1.0, HINGE_WEIGHT),
    "directed_hinge_2_center_1": (1.0, 2.0),
    "directed_hinge_2_center_4": (4.0, 2.0),
    "directed_hinge_8_center_4": (4.0, 8.0),
    "directed_hinge_8_center_16": (16.0, 8.0),
    "directed_hinge_32_center_16": (16.0, 32.0),
    "directed_hinge_32_center_64": (64.0, 32.0),
    "directed_hinge_128_center_64": (64.0, 128.0),
    "directed_hinge_128_center_256": (256.0, 128.0),
    "directed_hinge_512_center_256": (256.0, 512.0),
}


def freeze(out, trainer, binary, baseline):
    reference, protocol_sha, results_sha = REFERENCES[baseline]
    assert digest(reference / "protocol.json") == protocol_sha
    assert digest(reference / "results.json") == results_sha
    protocol, results = (read(reference / name) for name in ("protocol.json", "results.json"))
    assert results["protocol"] == protocol_sha
    assert digest(reference / "training-audit.json") == results["training"]
    audit = read(reference / "training-audit.json")
    assert audit["protocol"] == protocol_sha
    logits = reference / f"{baseline}-logits.json"
    assert digest(reference / "adapter.bin") == audit["adapter"]
    assert digest(logits) == results["logits"][baseline]
    assert digest(paired.MODEL) == protocol["model"] == paired.fit.MODEL_SHA
    assert digest(paired.INITIAL) == protocol["initial"] == paired.fit.INITIAL_SHA
    assert len(protocol["training_pairs"]) == 9
    assert subprocess.check_output(["crystal", "--version"], text=True).strip() == protocol["crystal"]

    loss = "mean two-choice CE + center_weight * mean(logit(X) - logit(Y))^2 within each pair"
    center_weight = CENTER_WEIGHT
    hinge_weight = 0.0
    hinge_mode = "minimum"
    reference_files = [reference / "adapter.bin", logits]
    if baseline != "paired":
        reference_center, reference_hinge = DIRECTED_WEIGHTS.get(baseline, (CENTER_WEIGHT, HINGE_WEIGHT))
        assert protocol["center_weight"] == reference_center
        assert digest(reference / "train.log") == audit["log"]
        reference_files.append(reference / "train.log")
        hinge_weight = HINGE_WEIGHT
        if baseline == "centered_hinge" or baseline in DIRECTED_WEIGHTS:
            assert protocol["hinge_weight"] == reference_hinge and protocol["minimum_margin"] == MINIMUM_MARGIN
            hinge_mode = "difference"
            loss += " + hinge_weight * max(0, minimum_margin - (correct_margin_a + correct_margin_b) / 2)"
        else:
            loss += " + hinge_weight * max(0, minimum_margin - min(correct_margin_a, correct_margin_b))"
    if baseline in DIRECTED_WEIGHTS:
        assert protocol["hinge_mode"] == hinge_mode
        assert protocol["files"][str(trainer)] == digest(trainer)
        candidate, _ = CANDIDATES[baseline]
        center_weight, hinge_weight = DIRECTED_WEIGHTS[candidate]

    replaced = {"train", "run.py", "train_choice_pairs.cr", "choice_pair_hinge.cr", Path(__file__).name}
    shared = {path: sha for path, sha in protocol["files"].items()
              if Path(path).name not in replaced}
    for path, sha in shared.items():
        assert digest(Path(path)) == sha, path
    assert digest(binary) == next(sha for path, sha in shared.items() if Path(path).name == "readout")
    views, cases, plans = paired.load_holdout()
    out.mkdir(parents=True)
    for name, sha in protocol["inputs"].items():
        assert digest(reference / name) == sha, name
        if name != "controls.json":
            copyfile(reference / name, out / name)

    with redirect_stdout(io.StringIO()):
        controls = archive.choice.check_candidates(out, cases, plans)
    assert controls == read(reference / "controls.json")
    assert digest(out / "controls.json") == protocol["inputs"]["controls.json"]
    here = Path(__file__).resolve().parent
    files = [trainer, binary, here / "run.py", here / "train_choice_pairs.cr",
             here / "choice_pair_center.cr", here / "choice_pair_hinge.cr", Path(__file__).resolve(),
             Path(archive.__file__).resolve(), archive.OUT / "plan.json", *reference_files]
    keep(out / "protocol.json", dict(
        protocol, files={**shared, **{str(path): digest(path) for path in files}},
        reference=dict(arm=baseline, protocol=protocol_sha, results=results_sha, adapter=audit["adapter"]),
        loss=loss, center_weight=center_weight, hinge_weight=hinge_weight, minimum_margin=MINIMUM_MARGIN,
        hinge_mode=hinge_mode, criterion=paired.CRITERION,
    ))
    print(f"Froze nine pairs and the exact 50-batch schedule; rebuilt all {2 * len(cases)} holdout candidates", flush=True)
    return views, controls, results


def float32(value):
    return unpack("<f", pack("<f", value))[0]


def train(out, trainer, rows):
    protocol = paired.verify_inputs(out)
    center_weight = protocol["center_weight"]
    hinge_weight, minimum_margin = protocol["hinge_weight"], protocol["minimum_margin"]
    hinge_mode = protocol["hinge_mode"]
    adapter, log = out / "adapter.bin", out / "train.log"
    command = [trainer, paired.MODEL, paired.INITIAL, out / "train.jsonl", out / "schedule.json",
               adapter, str(center_weight)]
    if hinge_weight:
        command += [str(hinge_weight), str(minimum_margin)]
        if hinge_mode == "difference":
            command.append(hinge_mode)
    print(f"Native training with center weight {center_weight}, {hinge_mode} hinge weight {hinge_weight}; live log {log}",
          flush=True)
    with log.open("x") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=paired.prior.REPO, check=True)

    paired.audit_training(out, rows)
    events = paired.fit._events(log)
    start = events[0]
    assert start["center_weight"] == center_weight
    updates = [event for event in events if event["event"] == "update"]
    if hinge_weight:
        assert start["hinge_weight"] == hinge_weight and start["minimum_margin"] == minimum_margin
        assert start.get("hinge_mode", "minimum") == hinge_mode
        control_events = paired.fit._events(REFERENCES[protocol["reference"]["arm"]][0] / "train.log")
        control_start = control_events[0]
        first = next(event for event in control_events if event["event"] == "update")
        for key in ("rows", "target_ids", "loss_a", "loss_b", "ce_loss"):
            assert updates[0][key] == first[key], key
        if center_weight == control_start["center_weight"]:
            assert updates[0]["center"] == first["center"]
        else:
            center, reference_center = updates[0]["center"], first["center"]
            for key in ("log_odds_a", "log_odds_b", "mean_log_odds"):
                assert center[key] == reference_center[key], key
            ratio = center_weight / control_start["center_weight"]
            assert center["penalty"] == ratio * reference_center["penalty"]
            assert math.isclose(center["scale"], ratio * reference_center["scale"], rel_tol=1e-6)
        if control_start.get("hinge_weight", 0.0) and control_start.get("hinge_mode", "minimum") == hinge_mode:
            hinge, reference_hinge = updates[0]["hinge"], first["hinge"]
            ratio = hinge_weight / control_start["hinge_weight"]
            for key in ("margin_a", "margin_b"):
                assert hinge[key] == reference_hinge[key], key
            for key in ("penalty", "scale_a", "scale_b"):
                assert hinge[key] == ratio * reference_hinge[key], key
    penalties, hinge_penalties = [], []
    for event in updates:
        center = event["center"]
        b = mean((center["log_odds_a"], center["log_odds_b"]))
        assert math.isclose(center["mean_log_odds"], b, rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(center["scale"], center_weight * center["mean_log_odds"], rel_tol=1e-6, abs_tol=1e-7)
        expected = center_weight * center["mean_log_odds"] ** 2
        assert math.isclose(center["penalty"], expected, abs_tol=1e-12)
        ce_loss = mean((event["loss_a"], event["loss_b"]))
        assert math.isclose(event["ce_loss"], ce_loss, abs_tol=1e-12)
        hinge_penalty = 0.0
        if hinge_weight:
            hinge = event["hinge"]
            sign = 1 if event["target_ids"][0] == start["labels"]["X"] else -1
            margin_a, margin_b = sign * center["log_odds_a"], -sign * center["log_odds_b"]
            assert (hinge["margin_a"], hinge["margin_b"]) == (margin_a, margin_b)
            a, b = float32(margin_a), float32(margin_b)
            if hinge_mode == "difference":
                score = float32((a + b) / 2)
                fraction_a = 0.5
            else:
                score = min(a, b)
                fraction_a = 0.5 if a == b else float(a < b)
            gap = max(0.0, float32(minimum_margin - score))
            assert hinge["penalty"] == hinge_weight * gap
            assert hinge["scale_a"] == (-sign * hinge_weight * fraction_a if gap else 0.0)
            assert hinge["scale_b"] == (sign * hinge_weight * (1 - fraction_a) if gap else 0.0)
            hinge_penalty = hinge["penalty"]
            hinge_penalties.append(hinge_penalty)
        else:
            assert event.get("hinge") is None
        assert math.isclose(event["loss"], ce_loss + expected + hinge_penalty, rel_tol=0.0, abs_tol=1e-12)
        penalties.append(center["penalty"])
    assert len(penalties) == paired.UPDATES and any(p > 0 for p in penalties)
    print(f"Audited all {len(penalties)} centered losses; mean penalty {mean(penalties):.6f}", flush=True)
    if hinge_weight:
        active = sum(p > 0 for p in hinge_penalties)
        print(f"Audited all paired hinges; active {active}/{len(updates)}, "
              f"mean penalty {mean(hinge_penalties):.6f}", flush=True)
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


def evaluate(out, binary, adapter, views, controls, reference, rows, baseline):
    assert digest(adapter) == read(out / "training-audit.json")["adapter"]
    reference_path = REFERENCES[baseline][0]
    candidate, _ = CANDIDATES[baseline]
    expected = read(reference_path / f"{baseline}-logits.json")
    first = expected[views[0]["view"]]
    label_ids = (first["x_id"], first["y_id"])
    train_views = [dict(view="train/" + row["id"], prefix_tokens=row["prefix_tokens"],
                        request=dict(id="repair", messages=row["messages"][:-1], sampling=paired.transfer.SAMPLING))
                   for row in rows]
    schedule = read(out / "schedule.json")
    comparisons, fitted = {}, {}
    for arm, path in ((baseline, reference_path / "adapter.bin"), (candidate, adapter)):
        raw = paired.measure(binary, path, arm, views + train_views, out)
        assert {(o["x_id"], o["y_id"]) for o in raw.values()} == {label_ids}
        if arm == baseline:
            assert all(raw[key] == value for key, value in expected.items())
        keep(out / f"{arm}-logits.json", raw)
        annotated = [dict(v, archived={arm: paired.letter(raw[v["view"]], sampled=True)}) for v in views]
        comparisons[arm] = archive.analyze(annotated, controls, {arm: raw})[arm]
        if arm == baseline:
            assert comparisons[arm] == reference["holdout"][baseline]
        fitted[arm] = train_scores(rows, schedule, raw)
        if arm == baseline and baseline != "paired":
            assert raw.keys() == expected.keys()
            assert fitted[arm] == reference["training_probe"][baseline]
        for wording in archive.ARCHIVES:
            score = comparisons[arm][wording]
            print(f"{arm} {wording}: full={score['full']}/6, "
                  f"sampled={score['sampled_full']}/24, argmax={score['argmax_full']}/24", flush=True)

    def correct(arm, wording, family):
        return comparisons[arm][wording]["families"][family]["both_correct"]

    gains = [family for family, kind in archive.FAMILIES.items() if kind == "flip"
             and all(correct(candidate, w, family) for w in archive.ARCHIVES)
             and not all(correct(baseline, w, family) for w in archive.ARCHIVES)]
    preserves = all(correct(candidate, w, "outbox_pending_count") for w in archive.ARCHIVES)
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
    parser.add_argument("--out", type=Path)
    objectives = parser.add_mutually_exclusive_group()
    objectives.add_argument("--hinge", dest="baseline", action="store_const", const="centered",
                            help="add a paired minimum-margin hinge and compare against centered CE")
    objectives.add_argument("--directed-hinge", dest="baseline", action="store_const", const="centered_hinge",
                            help="replace the minimum-margin hinge with a hinge on the directed difference")
    objectives.add_argument("--directed-hinge-center-1", dest="baseline", action="store_const",
                            const="directed_hinge", help="raise the directed-hinge center weight from 0.25 to 1.0")
    objectives.add_argument("--directed-hinge-2-center-1", dest="baseline", action="store_const",
                            const="directed_hinge_center_1", help="raise directed hinge weight to 2.0 at center weight 1.0")
    objectives.add_argument("--directed-hinge-2-center-4", dest="baseline", action="store_const",
                            const="directed_hinge_2_center_1", help="raise center weight to 4.0 at directed hinge weight 2.0")
    objectives.add_argument("--directed-hinge-8-center-4", dest="baseline", action="store_const",
                            const="directed_hinge_2_center_4", help="raise directed hinge weight to 8.0 at center weight 4.0")
    objectives.add_argument("--directed-hinge-8-center-16", dest="baseline", action="store_const",
                            const="directed_hinge_8_center_4", help="raise center weight to 16.0 at directed hinge weight 8.0")
    objectives.add_argument("--directed-hinge-32-center-16", dest="baseline", action="store_const",
                            const="directed_hinge_8_center_16", help="raise directed hinge weight to 32.0 at center weight 16.0")
    objectives.add_argument("--directed-hinge-32-center-64", dest="baseline", action="store_const",
                            const="directed_hinge_32_center_16", help="raise center weight to 64.0 at directed hinge weight 32.0")
    objectives.add_argument("--directed-hinge-128-center-64", dest="baseline", action="store_const",
                            const="directed_hinge_32_center_64", help="raise directed hinge weight to 128.0 at center weight 64.0")
    objectives.add_argument("--directed-hinge-128-center-256", dest="baseline", action="store_const",
                            const="directed_hinge_128_center_64", help="raise center weight to 256.0 at directed hinge weight 128.0")
    objectives.add_argument("--directed-hinge-512-center-256", dest="baseline", action="store_const",
                            const="directed_hinge_128_center_256", help="raise directed hinge weight to 512.0 at center weight 256.0")
    parser.set_defaults(baseline="paired")
    args = parser.parse_args()
    baseline = args.baseline
    _, default_out = CANDIDATES[baseline]
    out = (args.out or default_out).resolve()
    trainer, binary = args.trainer.resolve(), args.readout.resolve()
    views, controls, reference = freeze(out, trainer, binary, baseline)
    rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    adapter = train(out, trainer, rows)
    evaluate(out, binary, adapter, views, controls, reference, rows, baseline)


if __name__ == "__main__":
    main()
