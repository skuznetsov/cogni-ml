"""Offline ARC pilot: prepare -> native export -> parity -> train/evaluate."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import torch

from data import PROGRAMS, infer_target, load_tasks, make_prompt, matching_programs, split_tasks, train_hash
from tail import configure_mode, from_gguf


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def prepare(args):
    tasks = load_tasks(args.data_dir)
    split = split_tasks(tasks, args.seed, args.holdout_fraction)
    targets = {i: infer_target(t) for i, t in tasks.items()}
    selected = {}
    for name, cap in [("train", args.max_train), ("holdout", args.max_holdout)]:
        eligible = [i for i in split[name] if targets[i] is not None]
        # Sampling uses task hashes and seed, never test labels or difficulty.
        random.Random(args.seed).shuffle(eligible)
        selected[name] = eligible[:cap]
        if not selected[name]:
            raise ValueError(f"no unambiguous DSL-covered tasks in {name} split")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    corpus = {"data_dir": str(Path(args.data_dir).resolve()), "seed": args.seed,
              "holdout_fraction": args.holdout_fraction, "split": split,
              "selected": selected, "task_count": len(tasks),
              "unique_target_count": sum(x is not None for x in targets.values()),
              "unsupported_count": sum(not matching_programs(t) for t in tasks.values()),
              "ambiguous_count": sum(len(matching_programs(t)) > 1 for t in tasks.values()),
              "programs": [p.name for p in PROGRAMS],
              "train_hashes": {i: train_hash(t) for i, t in tasks.items()},
              "source_sha256": {i: digest(Path(args.data_dir) / f"{i}.json") for i in tasks},
              "optimization_reads_test_outputs": False}
    write_new(out / "corpus.json", corpus)
    # Test answers remain in the source corpus, not requests or optimizer data.
    rows = []
    for name, ids in selected.items():
        for task_id in ids:
            rows.append({"task_id": task_id, "split": name, "target": targets[task_id],
                         "prompt": make_prompt(tasks[task_id])})
    write_new(out / "examples.json", rows)
    with (out / "requests.jsonl").open("x") as f:
        for row in rows:
            f.write(json.dumps({"task_id": row["task_id"], "prompt": row["prompt"]}) + "\n")
    print(json.dumps({k: corpus[k] for k in ["task_count", "unique_target_count", "unsupported_count",
                                            "ambiguous_count", "selected"]}, indent=2))


def read_features(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    header = rows[0]
    if header.get("kind") != "header":
        raise ValueError("missing native feature header")
    ids = [r.get("task_id") for r in rows[1:] if r.get("kind") == "task"]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate feature task IDs")
    return header, rows[1:]


def check_parity(model, rows):
    checks = []
    with torch.no_grad():
        for row in rows:
            if row["kind"] not in {"task", "parity"}:
                continue
            x, ref = torch.tensor(row["inputs"]), torch.tensor(row["reference"])
            actual = model.block(x)
            if not torch.isfinite(actual).all():
                raise ValueError("nonfinite forward output")
            torch.testing.assert_close(actual, ref, atol=3e-4, rtol=3e-4)
            checks.append({"task_id": row.get("task_id"), "tokens": len(x),
                           "max_abs_error": float((actual - ref).abs().max())})
    if not checks:
        raise ValueError("no native parity rows")
    return checks


def parity(args):
    header, rows = read_features(args.features)
    if header.get("model_sha256") != digest(args.model):
        raise ValueError("features exported from different model bytes")
    model = from_gguf(args.model, header["label_ids"])
    print(json.dumps(check_parity(model, rows), indent=2))


def scores(model, examples, features, tasks):
    label_correct = grids_correct = grids_total = task_correct = 0
    predictions = {}
    model.eval()
    with torch.no_grad():
        for e in examples:
            choice = int(model(features[e["task_id"]]).argmax())
            label_correct += choice == e["target"]
            task = tasks[e["task_id"]]
            grids = [PROGRAMS[choice].apply(t["input"]) for t in task["test"]]
            matches = [g == t["output"] for g, t in zip(grids, task["test"]) if "output" in t]
            grids_correct += sum(matches)
            grids_total += len(matches)
            task_correct += bool(matches) and len(matches) == len(grids) and all(matches)
            predictions[e["task_id"]] = {"program": PROGRAMS[choice].name, "grids": grids}
    return {"tasks": len(examples), "label_correct": label_correct,
            "exact_grid_correct": grids_correct, "labelled_test_grids": grids_total,
            "all_test_exact_tasks": task_correct, "predictions": predictions}


def train(args):
    out = Path(args.out)
    if (out / "results.json").exists() or any((out / f"{m}.pt").exists() for m in ["head", "last-block"]):
        raise ValueError("existing run results/checkpoints; use a fresh output directory")
    corpus = json.loads((out / "corpus.json").read_text())
    examples = json.loads((out / "examples.json").read_text())
    tasks = load_tasks(corpus["data_dir"])
    if set(tasks) != set(corpus["source_sha256"]) or any(
            digest(Path(corpus["data_dir"]) / f"{i}.json") != h
            for i, h in corpus["source_sha256"].items()):
        raise ValueError("ARC corpus changed after preparation")
    for e in examples:
        if e["task_id"] not in corpus["selected"][e["split"]] or e["target"] != infer_target(tasks[e["task_id"]]):
            raise ValueError("example split/target differs from train-only preparation")
        if e["prompt"] != make_prompt(tasks[e["task_id"]]):
            raise ValueError("prompt differs from train-only preparation")
    expected_requests = [{"task_id": e["task_id"], "prompt": e["prompt"]} for e in examples]
    saved_requests = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines() if line.strip()]
    if saved_requests != expected_requests:
        raise ValueError("export requests differ from train-only preparation")
    expected_split = split_tasks(tasks, corpus["seed"], corpus["holdout_fraction"])
    if corpus["split"] != expected_split:
        raise ValueError("task partition differs from the fixed seed/content split")
    header, rows = read_features(args.features)
    if Path(header["model"]).resolve() != Path(args.model).resolve():
        raise ValueError("features exported from a different model path")
    model_hash = digest(args.model)
    if header.get("model_sha256") != model_hash:
        raise ValueError("features exported from different model bytes")
    if header.get("requests_sha256") != digest(out / "requests.jsonl"):
        raise ValueError("features exported from different requests")
    allowed = {e["task_id"] for e in examples}
    covered = [r["task_id"] for r in rows if r["kind"] in {"task", "skipped"}]
    if set(covered) != allowed or len(covered) != len(allowed):
        raise ValueError("incomplete/duplicate feature export: expected every selected task")
    if any(r["kind"] == "task" and r["task_id"] not in allowed for r in rows):
        raise ValueError("features contain an unexpected task")
    features = {r["task_id"]: torch.tensor(r["inputs"]) for r in rows if r["kind"] == "task"}
    used = [e for e in examples if e["task_id"] in features]
    partitions = {s: [e for e in used if e["split"] == s] for s in ["train", "holdout"]}
    if not all(partitions.values()):
        raise ValueError("no usable train/holdout features; token cap may exclude all tasks")
    torch.manual_seed(corpus["seed"])
    model = from_gguf(args.model, header["label_ids"])
    parity_result = check_parity(model, rows)
    initial = {n: v.clone() for n, v in model.state_dict().items()}
    result = {"corpus": corpus, "model_sha256": model_hash,
              "features_sha256": digest(args.features), "config": asdict(model.cfg),
              "parity": parity_result, "excluded_feature_tasks": sorted(allowed - set(features)),
              "epochs": args.epochs, "lr": args.lr, "torch_version": torch.__version__,
              "scope": "restricted DSL pilot on real ARC training split; not public/hidden eval",
              "modes": {}}
    # This grammar is solvable by exhaustive train-only search. Keep that
    # comparator visible: better label accuracy need not improve an ARC solver.
    result["dsl_search"] = {}
    for split, es in partitions.items():
        matches = []
        exact_tasks = 0
        for e in es:
            task = tasks[e["task_id"]]
            program = PROGRAMS[infer_target(task)]
            task_matches = [program.apply(t["input"]) == t["output"]
                            for t in task["test"] if "output" in t]
            matches.extend(task_matches)
            exact_tasks += bool(task_matches) and len(task_matches) == len(task["test"]) and all(task_matches)
        result["dsl_search"][split] = {"tasks": len(es), "exact_grid_correct": sum(matches),
                                       "labelled_test_grids": len(matches), "all_test_exact_tasks": exact_tasks}
    for mode in ["baseline", "head", "last-block"]:
        model.load_state_dict(initial)
        configure_mode(model, mode)
        started = time.monotonic()
        loss_history = []
        if mode != "baseline":
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
            for epoch in range(args.epochs):
                model.train()
                order = partitions["train"][:]
                random.Random(corpus["seed"] + epoch).shuffle(order)
                losses = []
                for e in order:
                    if time.monotonic() - started > args.max_seconds:
                        raise TimeoutError(f"{mode} exceeded time budget; no quality conclusion")
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(features[e["task_id"]])
                    loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([e["target"]]))
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                        raise ValueError("nonfinite gradient")
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                loss_history.append(sum(losses) / len(losses))
        changed = [n for n, v in model.state_dict().items() if not torch.equal(v, initial[n])]
        if any(not (n == "head.weight" if mode == "head" else n.startswith("block.")
                    if mode == "last-block" else False) for n in changed):
            raise ValueError(f"frozen weight changed in {mode}")
        if mode != "baseline" and not changed:
            raise ValueError(f"no weights changed in {mode}")
        result["modes"][mode] = {"changed_weights": changed, "loss_history": loss_history,
                                 "elapsed_seconds": time.monotonic() - started,
                                 "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                                 **{s: scores(model, partitions[s], features, tasks) for s in partitions}}
        if mode != "baseline":
            torch.save({"mode": mode, "config": asdict(model.cfg), "label_ids": header["label_ids"],
                        "model_sha256": result["model_sha256"],
                        "state_dict": {n: v for n, v in model.state_dict().items() if n in changed}},
                       out / f"{mode}.pt")
        print(mode, {s: {k: v for k, v in result["modes"][mode][s].items() if k != "predictions"}
                     for s in partitions}, flush=True)
    result["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result["source_files_sha256"] = {p.name: digest(p) for p in Path(__file__).parent.iterdir()
                                      if p.suffix in {".py", ".cr"}}
    write_new(out / "results.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-fraction", type=float, default=.25)
    p.add_argument("--max-train", type=int, default=8)
    p.add_argument("--max-holdout", type=int, default=8)
    for action in ["parity", "train"]:
        p = sub.add_parser(action)
        p.add_argument("--model", required=True)
        p.add_argument("--features", required=True)
        if action == "train":
            p.add_argument("--out", required=True)
            p.add_argument("--epochs", type=int, default=3)
            p.add_argument("--lr", type=float, default=1e-4)
            p.add_argument("--max-seconds", type=float, default=120)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action == "prepare" and min(args.max_train, args.max_holdout) <= 0:
        parser.error("task caps must be positive")
    if args.action == "train" and (args.epochs <= 0 or args.lr <= 0 or args.max_seconds <= 0):
        parser.error("training controls must be positive")
    globals()[args.action](args)


if __name__ == "__main__":
    main()
