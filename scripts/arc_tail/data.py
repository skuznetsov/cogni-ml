"""Real ARC task selection; test answers never enter supervision or prompts."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class Program:
    name: str
    apply: object


def rotate90(g):
    return [list(row) for row in zip(*g[::-1])]


def crop(g):
    cells = [(r, c) for r, row in enumerate(g) for c, x in enumerate(row) if x != 0]
    if not cells:
        return [row[:] for row in g]
    r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
    c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
    return [row[c0:c1 + 1] for row in g[r0:r1 + 1]]


PROGRAMS = [
    Program("identity", lambda g: [r[:] for r in g]),
    Program("rotate90", rotate90),
    Program("rotate180", lambda g: [r[::-1] for r in g[::-1]]),
    Program("rotate270", lambda g: rotate90(rotate90(rotate90(g)))),
    Program("flip_horizontal", lambda g: [r[::-1] for r in g]),
    Program("flip_vertical", lambda g: [r[:] for r in g[::-1]]),
    Program("transpose", lambda g: [list(r) for r in zip(*g)]),
    Program("crop_nonzero", crop),
]
LABELS = list("ABCDEFGH")


def valid_grid(g):
    return (isinstance(g, list) and bool(g) and isinstance(g[0], list) and bool(g[0])
            and all(isinstance(r, list) and len(r) == len(g[0])
                    and all(type(x) is int and 0 <= x <= 9 for x in r) for r in g))


def validate_task(task):
    if not task.get("train") or not task.get("test"):
        raise ValueError("ARC task requires train and test examples")
    grids = [g for e in task["train"] for g in (e["input"], e["output"])]
    grids += [e["input"] for e in task["test"]]
    if not all(valid_grid(g) for g in grids):
        raise ValueError("invalid ARC input/train-output grid")


def matching_programs(task):
    return [i for i, p in enumerate(PROGRAMS)
            if all(p.apply(e["input"]) == e["output"] for e in task["train"])]


def infer_target(task):
    matches = matching_programs(task)
    return matches[0] if len(matches) == 1 else None


def train_hash(task):
    # Ignore task filename and all test data: a renamed duplicate cannot cross splits.
    examples = sorted(json.dumps(e, sort_keys=True, separators=(",", ":"))
                      for e in task["train"])
    return hashlib.sha256(json.dumps(examples).encode()).hexdigest()


def split_tasks(tasks, seed, holdout_fraction):
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout fraction must be between zero and one")
    seen = set()
    split = {"train": [], "holdout": []}
    for task_id, task in sorted(tasks.items()):
        key = train_hash(task)
        if key in seen:
            raise ValueError(f"duplicate training content in task {task_id}")
        seen.add(key)
        value = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16) / 2**64
        split["holdout" if value < holdout_fraction else "train"].append(task_id)
    return split


def make_prompt(task):
    def grid(g):
        return "/".join("".join(map(str, r)) for r in g)
    lines = ["Choose the grid transformation. Reply with one letter."]
    lines += [f"{a}={p.name}" for a, p in zip(LABELS, PROGRAMS)]
    lines += [f"{grid(e['input'])} -> {grid(e['output'])}" for e in task["train"]]
    return "\n".join(lines) + "\nAnswer:"


def load_tasks(data_dir):
    p = Path(data_dir).resolve()
    # Eval-labelled paths are not admitted as a training corpus.
    if any(x.lower() in {"evaluation", "eval", "test"} for x in p.parts):
        raise ValueError("public evaluation/test directories cannot be used for training")
    files = sorted(p.glob("*.json"))
    if not files:
        raise ValueError(f"no ARC task JSONs in {p}")
    tasks = {f.stem: json.loads(f.read_text()) for f in files}
    for t in tasks.values():
        validate_task(t)
    return tasks
