#!/usr/bin/env python3
"""Export and replay the historical Crystal bugfix pilot."""

import argparse
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval" / "crystal_bugfixes"
SPLITS = ("train", "valid", "test")
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", re.M)
RESULT = re.compile(r"^(\d+) examples?, (\d+) failures?, (\d+) errors?, (\d+) pending$", re.M)
SPEC_COMMAND = ["crystal", "spec", "-Dcpu_only", "--no-color", "spec/pilot_regression_spec.cr"]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def blob(revision, path):
    return git("show", f"{revision}:{path}").decode("utf-8")


def load_cases():
    manifest = json.loads((DATASET / "cases.json").read_text(encoding="utf-8"))
    anchor = manifest["history"]
    cases = manifest["cases"]
    owners = {}
    ids = set()
    for case in cases:
        if case["id"] in ids or case["split"] not in SPLITS:
            raise ValueError(f"Duplicate ID or unknown split: {case['id']}")
        ids.add(case["id"])
        for field in ("source", "spec", "family"):
            key = field, case[field]
            if owners.setdefault(key, case["split"]) != case["split"]:
                raise ValueError(f"{field} crosses splits: {case[field]}")
        git("merge-base", "--is-ancestor", case["fix"], anchor)
    if {case["split"] for case in cases} != set(SPLITS):
        raise ValueError("Every split needs at least one case")
    return cases


def historical_patch(case):
    fix, source = case["fix"], case["source"]
    patch = git("diff", "--no-ext-diff", "--no-textconv", "--no-color",
                "--no-renames", "--src-prefix=a/", "--dst-prefix=b/",
                "--diff-algorithm=myers", "--no-indent-heuristic",
                "--unified=3", f"{fix}^", fix, "--", source)
    patch = patch.decode("utf-8")
    if not HUNK.search(patch):
        raise ValueError(f"No historical patch for {case['id']}")
    return "".join(line for line in patch.splitlines(keepends=True)
                   if not line.startswith("index "))


def visible_ranges(case, patch, line_count):
    ranges = [(start, end) for start, end in case.get("context", [])]
    for hunk in HUNK.finditer(patch):
        start = int(hunk[1])
        count = int(hunk[2] or 1)
        ranges.append((max(1, start - 7), min(line_count, start + count + 6)))
    merged = []
    for start, end in sorted(ranges):
        if not 1 <= start <= end <= line_count:
            raise ValueError(f"Invalid source range for {case['id']}: {start}-{end}")
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def model_record(case):
    patch = historical_patch(case)
    source = case["source"]
    lines = blob(f"{case['fix']}^", source).splitlines(keepends=True)
    excerpts = []
    for start, end in visible_ranges(case, patch, len(lines)):
        code = "".join(lines[start - 1:end])
        excerpts.append(f"{source}, lines {start}-{end}\n```crystal\n{code}```")
    prompt = (
        f"{case['problem']}\n\n"
        f"Change only {source}. Preserve behavior outside the requested fix. "
        "Reply with a unified diff with a/ and b/ paths and context lines.\n\n"
        + "\n\n".join(excerpts)
    )
    messages = [{"role": "user", "content": prompt}]
    if case["split"] == "test":
        return {"id": case["id"], "messages": messages}
    messages.append({"role": "assistant", "content": patch})
    return {"messages": messages}


def export(cases, output):
    records = {split: [] for split in SPLITS}
    for case in cases:
        records[case["split"]].append(model_record(case))
    output.mkdir(parents=True, exist_ok=False)
    training = output / "sft"
    evaluation = output / "heldout"
    training.mkdir()
    evaluation.mkdir()
    for split, rows in records.items():
        path = (evaluation if split == "test" else training) / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        print(f"{path}: {len(rows)}")


def regression_spec(case):
    if fixture := case.get("fixture"):
        return (DATASET / fixture).read_text(encoding="utf-8")
    original = blob(case["fix"], case["spec"])
    if "spec_ranges" not in case:
        return original
    lines = original.splitlines(keepends=True)
    ranges = case["spec_ranges"]
    if any(not 1 <= start <= end <= len(lines) for start, end in ranges):
        raise ValueError(f"Invalid regression spec range for {case['id']}")
    return "".join("".join(lines[start - 1:end]) for start, end in ranges)


def materialize(case, output):
    output.mkdir(parents=True, exist_ok=False)
    archive = git("archive", f"{case['fix']}^", "src", "spec")
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        tree.extractall(output, filter="data")
    (output / "spec" / "pilot_regression_spec.cr").write_text(
        regression_spec(case), encoding="utf-8",
    )


def apply_candidate(case, output, answer):
    source = case["source"]
    patch = answer.replace("\r\n", "\n").strip("\n")
    if patch.startswith("```diff\n") and patch.endswith("\n```"):
        patch = patch.removeprefix("```diff\n").removesuffix("\n```")
    if not patch.startswith((f"diff --git a/{source} b/{source}\n", f"--- a/{source}\n")):
        raise ValueError(f"Expected a unified diff for {source}")
    if len(patch.encode("utf-8")) > 65_536:
        raise ValueError("Patch exceeds 64 KiB")

    # Disable repository discovery even if the replay directory is under build/.
    env = {"PATH": os.environ["PATH"], "GIT_CEILING_DIRECTORIES": str(output.parent)}

    def apply(*args):
        return subprocess.run(["git", "apply", "--recount", *args], cwd=output,
                              env=env, input=(patch + "\n").encode("utf-8"),
                              check=True, capture_output=True).stdout

    stats = apply("--numstat", "-z").split(b"\0")
    if len(stats) != 2 or stats[0].split(b"\t", 2)[-1] != source.encode("utf-8"):
        raise ValueError(f"The patch must modify only {source}")
    if apply("--summary"):
        raise ValueError("File creation, deletion, renames and mode changes are outside this task")
    apply("--check")
    apply()


def run_spec(output, command=SPEC_COMMAND):
    scratch = output / ".pilot"
    scratch.mkdir(exist_ok=True)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "CRYSTAL_CACHE_DIR": str(scratch / "cache"),
    }
    return subprocess.run(command, cwd=output, env=env, timeout=120,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def verify(case):
    with tempfile.TemporaryDirectory(prefix="crystal-bugfix-") as scratch:
        output = Path(scratch).resolve() / "replay"
        materialize(case, output)
        before = run_spec(output)
        expected_before = (case["examples"], case["red_count"], case.get("red_errors", 0), 0)
        expect_result(case, "before", before, 1, expected_before)
        if case["failure"] not in before.stdout:
            raise ValueError(f"{case['id']}: unexpected regression failure\n{before.stdout}")

        apply_candidate(case, output, historical_patch(case))
        fixed = (output / case["source"]).read_text(encoding="utf-8")
        if fixed != blob(case["fix"], case["source"]):
            raise ValueError(f"{case['id']}: replay differs from historical fix")
        after = run_spec(output)
        expect_result(case, "after", after, 0, (case["examples"], 0, 0, 0))
        failed_before = expected_before[1] + expected_before[2]
        print(f"{case['id']} ({case['split']}): {failed_before} failed before, "
              f"{case['examples']} passed after", flush=True)


def expect_result(case, phase, process, expected_code, expected_counts):
    result = RESULT.search(process.stdout)
    if (process.returncode != expected_code or not result
            or tuple(map(int, result.groups())) != expected_counts):
        raise ValueError(f"{case['id']} {phase}: unexpected spec result\n{process.stdout}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("export", help="write SFT records and prompt-only held-out records")
    exporter.add_argument("output", type=Path)
    verifier = commands.add_parser("verify", help="replay historical red/green Crystal specs")
    verifier.add_argument("ids", nargs="*")
    prepare = commands.add_parser("prepare", help="apply an answer and print the test command")
    prepare.add_argument("id")
    prepare.add_argument("answer", type=Path)
    prepare.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = load_cases()
    by_id = {case["id"]: case for case in cases}

    if args.command == "export":
        export(cases, args.output)
    elif args.command == "verify":
        for case in [by_id[key] for key in args.ids] if args.ids else cases:
            verify(case)
    else:
        case = by_id[args.id]
        answer = args.answer.read_text(encoding="utf-8")
        output = args.output.resolve()
        materialize(case, output)
        apply_candidate(case, output, answer)
        print(f"Replay directory: {output}")
        print("Run only in an isolated environment for untrusted patches:")
        print(shlex.join(SPEC_COMMAND))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace").strip() if exc.stderr else str(exc)
        raise SystemExit(detail) from exc
    except (KeyError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(str(exc)) from exc
