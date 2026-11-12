#!/usr/bin/env python3
"""Export and replay fragment replacements for the expanded Crystal dataset."""

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

import crystal_bugfix_dataset as pilot


MANIFEST = pilot.DATASET / "fragments.json"
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.M)
FENCE = re.compile(
    r"^[ \t]*```(?:crystal|cr)?[ \t]*\n(.*?)^[ \t]*```[ \t]*(?:\n|\Z)",
    re.M | re.S,
)


def load_cases():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    original = pilot.load_cases()
    by_id = {case["id"]: case for case in original}
    retained = []
    for key, options in manifest["retain_train"].items():
        case = by_id[key]
        if case["split"] != "train" or options.keys() - {"lines", "fragmented"}:
            raise ValueError(f"Only fragment formatting can change for retained training case {key}")
        retained.append(dict(case, **options))
    cases = retained + manifest["cases"]
    owners = {}
    fixes = set()
    ids = set()
    for case in original + manifest["cases"]:
        key, fix, split = case["id"], case["fix"], case["split"]
        if key in ids or fix in fixes or split not in pilot.SPLITS:
            raise ValueError(f"Duplicate ID or fix, or unknown split: {key}")
        fixes.add(fix)
        ids.add(key)
        for field in ("family", "source", "spec"):
            value = case[field]
            if owners.setdefault((field, value), split) != split:
                raise ValueError(f"{field} crosses splits: {value}")
        pilot.git("merge-base", "--is-ancestor", fix, manifest["history"])
    train_ids = {case["id"] for case in cases if case["split"] == "train"}
    small_train = manifest["small_train"]
    if len(small_train) != 5 or set(small_train) != {case["id"] for case in retained}:
        raise ValueError("The small cohort must contain five distinct training cases")
    if not 20 <= len(train_ids) <= 25:
        raise ValueError("The expanded cohort must contain 20–25 training cases")
    if {case["split"] for case in cases} != set(pilot.SPLITS):
        raise ValueError("The expanded dataset needs cases in each split")
    return cases, set(small_train)


def fragments(case):
    fix, source = case["fix"], case["source"]
    before = pilot.blob(f"{fix}^", source).splitlines(keepends=True)
    after = pilot.blob(fix, source)
    if case.get("fragmented"):
        fixed = after.splitlines(keepends=True)
        regions = []
        for hunk in HUNK.finditer(pilot.historical_patch(case)):
            first, start = int(hunk[1]), int(hunk[3])
            old_count, new_count = int(hunk[2] or 1), int(hunk[4] or 1)
            last = first + old_count - 1
            regions.append((first, last, "".join(before[first - 1:last]),
                            "".join(fixed[start - 1:start + new_count - 1])))
        if splice(before, regions, [new for _, _, _, new in regions]) != after:
            raise ValueError(f"{case['id']}: fragmented reference differs from historical fix")
        return before, regions
    first, last = case["lines"]
    if not 1 <= first <= last <= len(before):
        raise ValueError(f"{case['id']}: invalid editable lines")
    prefix = "".join(before[:first - 1])
    old = "".join(before[first - 1:last])
    suffix = "".join(before[last:])
    if reference := case.get("reference_lines"):
        fixed_lines = after.splitlines(keepends=True)
        start, end = reference
        if not 1 <= start <= end <= len(fixed_lines):
            raise ValueError(f"{case['id']}: invalid historical reference lines")
        new = "".join(fixed_lines[start - 1:end])
        if old == new or not new.strip():
            raise ValueError(f"{case['id']}: invalid historical reference replacement")
        return before, [(first, last, old, new)]
    if not after.startswith(prefix) or not after.endswith(suffix):
        raise ValueError(f"{case['id']}: editable fragment omits historical changes")
    end = len(after) - len(suffix)
    new = after[len(prefix):end]
    if old == new or not new.strip() or end < len(prefix):
        raise ValueError(f"{case['id']}: invalid reference replacement")
    return before, [(first, last, old, new)]


def splice(before, regions, replacements):
    parts = []
    offset = 0
    for (first, last, _, _), code in zip(regions, replacements, strict=True):
        if not offset < first <= last <= len(before):
            raise ValueError("Editable regions must be disjoint nonempty line ranges")
        parts.extend(("".join(before[offset:first - 1]), code))
        offset = last
    parts.append("".join(before[offset:]))
    return "".join(parts)


def fenced(code):
    return f"```crystal\n{code}```"


def reference_answer(regions):
    return "\n\n".join(fenced(new) for _, _, _, new in regions)


def model_record(case):
    before, regions = fragments(case)
    context = ""
    if len(regions) == 1:
        first, last, _, _ = regions[0]
        for label, lines in (("preceding", before[max(0, first - 7):first - 1]),
                             ("following", before[last:last + 6])):
            if lines:
                context += f"Read-only {label} lines:\n{fenced(''.join(lines))}\n\n"
    editable = "\n\n".join(f"Editable lines {first}-{last}:\n{fenced(old)}"
                           for first, last, old, _ in regions)
    response = ("one Crystal code block" if len(regions) == 1 else
                f"{len(regions)} Crystal code blocks in the listed order")
    prompt = (
        f"{case['problem']}\n\n"
        f"Replace only the editable lines of {case['source']}. "
        "Return each entire replacement, including unchanged lines inside it. "
        f"Keep indentation and reply with {response}.\n\n"
        f"{context}{editable}"
    )
    messages = [{"role": "user", "content": prompt}]
    if case["split"] == "test":
        return {"id": case["id"], "messages": messages}
    messages.append({"role": "assistant", "content": reference_answer(regions)})
    return {"messages": messages}


def export(cases, small_train, output):
    rows = {case["id"]: model_record(case) for case in cases}
    output.mkdir(parents=True, exist_ok=False)
    datasets = {
        "sft-5/train.jsonl": [rows[case["id"]] for case in cases if case["id"] in small_train],
        "sft-expanded/train.jsonl": [rows[case["id"]] for case in cases if case["split"] == "train"],
        "heldout/test.jsonl": [rows[case["id"]] for case in cases if case["split"] == "test"],
    }
    valid = [rows[case["id"]] for case in cases if case["split"] == "valid"]
    for folder in ("sft-5", "sft-expanded"):
        datasets[f"{folder}/valid.jsonl"] = valid
    for name, records in datasets.items():
        path = output / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                        encoding="utf-8")
        print(f"{path}: {len(records)}", flush=True)


def replacements(answer, count):
    answer = answer.replace("\r\n", "\n").strip("\n")
    if len(answer.encode("utf-8")) > 65_536:
        raise ValueError("Replacement exceeds 64 KiB")
    matches = list(FENCE.finditer(answer))
    if count == 1 and not matches and "```" not in answer:
        codes = [answer + "\n"]
    elif len(matches) == count and not FENCE.sub("", answer).strip():
        codes = [match[1] for match in matches]
    else:
        raise ValueError(f"Expected {count} Crystal code block(s)")
    if any(not code.strip() or "```" in code
           or code.lstrip().startswith(("diff --git", "diff -u", "--- a/"))
           for code in codes):
        raise ValueError("Expected nonempty Crystal replacements")
    return codes


def prepare(case, answer, output):
    before, regions = fragments(case)
    code = splice(before, regions, replacements(answer, len(regions)))
    pilot.materialize(case, output)
    (output / case["source"]).write_text(code, encoding="utf-8")


def spec_command(case, replay):
    backend = case.get("backend", "cpu")
    if backend == "cpu":
        return pilot.SPEC_COMMAND
    if backend != "metal":
        raise ValueError(f"{case['id']}: unsupported replay backend {backend}")
    bridge = replay / ".replay-bridge.o"
    subprocess.run(
        ["xcrun", "clang++", "-c", "src/ml/metal/bridge.mm", "-o", str(bridge),
         "-std=c++17", "-fobjc-arc", "-fPIC"],
        cwd=replay, capture_output=True, check=True, timeout=120,
    )
    flags = f"{shlex.quote(str(bridge))} -framework Metal -framework Foundation -lc++"
    return ["crystal", "spec", "--no-color", f"--link-flags={flags}",
            "spec/pilot_regression_spec.cr"]


def verify(case):
    with tempfile.TemporaryDirectory(prefix="crystal-fragment-") as scratch:
        replay = Path(scratch).resolve() / "replay"
        baseline, regions = fragments(case)
        pilot.materialize(case, replay)
        command = spec_command(case, replay)
        before = pilot.run_spec(replay, command)
        if case.get("red_compile"):
            if before.returncode != 1 or pilot.RESULT.search(before.stdout):
                raise ValueError(f"{case['id']}: expected a compilation failure\n{before.stdout}")
        else:
            counts = (case["examples"], case["red_count"], case.get("red_errors", 0), 0)
            pilot.expect_result(case, "before", before, 1, counts)
        if case["failure"] not in before.stdout:
            raise ValueError(f"{case['id']}: unexpected regression failure\n{before.stdout}")

        source = replay / case["source"]
        # Replay the actual exported answer through the same parser as model responses.
        answer = reference_answer(regions)
        code = splice(baseline, regions, replacements(answer, len(regions)))
        source.write_text(code, encoding="utf-8")
        after = pilot.run_spec(replay, command)
        pilot.expect_result(case, "after", after, 0, (case["examples"], 0, 0, 0))
        fixed_source = pilot.blob(case["fix"], case["source"])
        if case.get("reference_lines"):
            source.write_text(fixed_source, encoding="utf-8")
            historical = pilot.run_spec(replay, command)
            pilot.expect_result(case, "historical", historical, 0, (case["examples"], 0, 0, 0))
        elif source.read_text(encoding="utf-8") != fixed_source:
            raise ValueError(f"{case['id']}: replay differs from historical fix")
        red = "compile" if case.get("red_compile") else "test"
        print(f"{case['id']} ({case['split']}): {red} failure before, "
              f"{case['examples']} passed after", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("output", type=Path)
    verifier = commands.add_parser("verify")
    verifier.add_argument("ids", nargs="*")
    preparer = commands.add_parser("prepare")
    preparer.add_argument("id")
    preparer.add_argument("answer", type=Path)
    preparer.add_argument("output", type=Path)
    args = parser.parse_args()
    cases, small_train = load_cases()
    by_id = {case["id"]: case for case in cases}
    if args.command == "export":
        export(cases, small_train, args.output)
    elif args.command == "verify":
        for case in [by_id[key] for key in args.ids] if args.ids else cases:
            verify(case)
    else:
        case = by_id[args.id]
        output = args.output.resolve()
        prepare(case, args.answer.read_text(encoding="utf-8"), output)
        print("Run in an isolated environment for untrusted replacements:")
        print(shlex.join(spec_command(case, output)))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace").strip() if exc.stderr else str(exc)
        raise SystemExit(detail) from exc
    except (KeyError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(str(exc)) from exc
