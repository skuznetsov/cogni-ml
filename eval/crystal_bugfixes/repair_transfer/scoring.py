"""Compile and score a one-line Crystal repair against an isolated spec suite."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile


_SUMMARY = re.compile(
    r"(?m)^\s*(\d+) examples?, (\d+) failures?, (\d+) errors?"
    r"(?:, (\d+) pending)?\s*$"
)
_FAILED = re.compile(r"(?m)^\s+\d+\)\s+([^\n]+)")


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _clean(data, output: str, roots: tuple[Path, ...]) -> str:
    for root in roots:
        for candidate in (str(root), os.path.realpath(root)):
            output = output.replace(candidate, "fixture")
    output = re.sub(r"\btrace_id=\d+", "trace_id=<trace>", output)
    output = re.sub(r"\b(?:host_)?elapsed(?:_ms)?=[0-9.]+", "elapsed=<time>", output)
    output = re.sub(r"\b0x[0-9a-fA-F]{6,}\b", "<address>", output)
    return data.clean_diagnostic(output)


def _result(stage, compiled, target, regressions, passed, total, diagnostic, **extra):
    return {
        "stage": stage,
        "compiled": compiled,
        "target": target,
        "regressions": regressions,
        "passed": passed,
        "total": total,
        "diagnostic": diagnostic,
        **extra,
    }


def _write(dest: Path, name: str, output: str, status: int | str) -> None:
    (dest / name).write_text(output, encoding="utf-8")
    (dest / f"{name[:-4]}.status").write_text(f"{status}\n", encoding="utf-8")


def _failure_names(raw: str, expected: list[str]) -> list[str] | None:
    names = _FAILED.findall(raw)
    mapped = []
    for failure in names:
        matches = [name for name in expected if failure == name or failure.endswith(" " + name)]
        if len(matches) != 1:
            return None
        mapped.append(matches[0])
    return mapped


def score(data, case: dict, design: dict, answer: str, *, hidden: bool, dest: Path) -> dict:
    """Replay `answer` in `design.template`; hidden checks are included only when requested."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _write(dest, "compile.log", "", "not_run")
    _write(dest, "test.log", "", "not_run")
    template, fixture = design["template"], design["fixture"]
    if not isinstance(template, str) or template.count("<HOLE>") != 1:
        raise ValueError("template must contain exactly one literal <HOLE>")
    if not isinstance(fixture, str) or not isinstance(answer, str) or not answer or "\n" in answer or "\r" in answer or answer != answer.strip():
        raise ValueError("fixture and normalized one-line answer are required")
    source = Path(case["source"])
    if source.is_absolute() or ".." in source.parts:
        raise ValueError("case source must be a safe repository-relative path")
    patched = template.replace("<HOLE>", answer, 1)
    if "<HOLE>" in patched:
        raise ValueError("answer cannot contain the literal <HOLE> token")

    if hidden and design.get("guards"):
        guards = design["guards"]
        if not isinstance(guards, str):
            raise ValueError("hidden guards must be spec source text")
        fixture = fixture.rstrip() + "\n\n" + guards.strip() + "\n"
    names = data.test_names(fixture)
    needles = case["target_tests"]
    targets = [name for name in names if any(needle in name for needle in needles)]
    regressions = [name for name in names if name not in targets]
    if not targets or not regressions:
        raise ValueError("fixture must name at least one target and one regression example")

    support = case.get("support_files", [])
    if not isinstance(support, list) or not all(isinstance(path, str) for path in support):
        raise ValueError("support_files must be pinned repository-relative paths")

    (dest / "answer.txt").write_text(answer + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="own-peer-v1-") as temporary:
        root = Path(temporary) / "replay"
        source_file = root / source
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text(patched, encoding="utf-8")
        for path in support:
            support_path = Path(path)
            if support_path.is_absolute() or ".." in support_path.parts:
                raise ValueError(f"unsafe support source path: {path!r}")
            target = root / support_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(data.git_blob(case["parent"], path), encoding="utf-8")
        spec = root / "spec" / "pilot_regression_spec.cr"
        spec.parent.mkdir(parents=True, exist_ok=True)
        spec.write_text(fixture, encoding="utf-8")
        binary = root / "spec-runner"

        try:
            build = subprocess.run(
                ["crystal", "build", "--no-color", "-Dcpu_only", str(spec.relative_to(root)), "-o", str(binary.relative_to(root))],
                cwd=root, capture_output=True, text=True, timeout=120, check=False,
            )
            build_log, build_code = build.stdout + build.stderr, build.returncode
        except subprocess.TimeoutExpired as error:
            _write(dest, "compile.log", _text(error.stdout) + _text(error.stderr) + "\nCOMPILE TIMEOUT\n", "timeout")
            raise
        _write(dest, "compile.log", build_log, build_code)
        if build_code != 0 or not binary.is_file():
            if build_code == 0:
                raise RuntimeError("Crystal exited successfully without producing the spec runner")
            diagnostic = _clean(data, build_log, (root, Path(temporary)))
            return _result(
                "compile_failed", False, None, None, None, len(names), diagnostic,
                compile_exit_code=build_code, test_exit_code=None,
                compile_log="compile.log", test_log="test.log",
            )

        try:
            run = subprocess.run(
                ["./spec-runner", "--no-color"], cwd=root, capture_output=True,
                text=True, timeout=60, check=False,
            )
            test_log, test_code = run.stdout + run.stderr, run.returncode
        except subprocess.TimeoutExpired as error:
            _write(dest, "test.log", _text(error.stdout) + _text(error.stderr) + "\nREPLAY TIMEOUT\n", "timeout")
            raise
        _write(dest, "test.log", test_log, test_code)

    match = _SUMMARY.search(test_log)
    failed = _failure_names(test_log, names)
    if not match:
        raise RuntimeError("Crystal spec run did not produce a complete summary; see test.log")

    observed, failures, errors, pending = (int(value or 0) for value in match.groups())
    if observed != len(names):
        raise RuntimeError(f"Crystal ran {observed} examples; fixture declares {len(names)}; see test.log")
    if failed is None or len(failed) != failures + errors:
        raise RuntimeError("Crystal failure names did not map one-to-one to fixture examples; see test.log")
    passed = max(0, observed - failures - errors - pending)
    target_failed = any(name in targets for name in failed)
    regression_failed = any(name in regressions for name in failed)
    target_result = False if target_failed else (None if pending else True)
    regression_result = False if regression_failed else (None if pending else True)
    if test_code != 0 and failures == 0 and errors == 0 and pending == 0:
        raise RuntimeError("Crystal runner exited nonzero after reporting no failed examples; see test.log")
    passed_all = test_code == 0 and failures == 0 and errors == 0 and pending == 0
    diagnostic = "" if passed_all else _clean(data, test_log, (root, Path(temporary)))
    if not passed_all and not diagnostic:
        diagnostic = f"{failures} failures, {errors} errors, {pending} pending in {observed} examples; refusing to pass."
    return _result(
        "pass" if passed_all else "tests_failed", True, target_result, regression_result,
        passed, len(names), diagnostic,
        compile_exit_code=0, test_exit_code=test_code, failures=failures,
        errors=errors, pending=pending, observed=observed, failed_specs=failed or [],
        compile_log="compile.log", test_log="test.log",
    )
