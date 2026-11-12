#!/usr/bin/env python3
"""Evaluate a native QLoRA generator on the six held-out Crystal fragments."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "build" / "qlora"
SCRIPTS = ROOT / "scripts"
DEFAULT_RUNNER = SCRIPTS / "crystal_bugfix_qlora_replay.py"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import crystal_bugfix_fragments as fragments  # noqa: E402
import crystal_bugfix_dataset as pilot  # noqa: E402


MAX_ANSWER_BYTES = 65_536
MAX_REQUEST_BYTES = 512 * 1024
MAX_RESPONSE_LINE_BYTES = 128 * 1024
FEEDBACK_BYTES = 2_048
TIMEOUT_CODES = {124}
INFRASTRUCTURE_CODES = {126, 127, 137}


class GeneratorFailure(Exception):
    """The generator process violated its request/response contract."""


class GeneratorTimeout(GeneratorFailure):
    """The generator did not answer within the configured bound."""

    def __init__(self, message, partial=b""):
        super().__init__(message)
        self.partial = partial


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def clean_text(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value.replace("\x00", "�")


def bounded_diagnostic(value, limit=FEEDBACK_BYTES):
    """Return a bounded, printable diagnostic suitable for a repair prompt."""
    value = clean_text(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = value.splitlines()
    if len(lines) > 48:
        value = "\n".join(lines[-48:])
    encoded = value.encode("utf-8")
    if len(encoded) > limit:
        encoded = encoded[-limit:]
        value = encoded.decode("utf-8", errors="replace")
        value = "[earlier diagnostic lines omitted]\n" + value
    return value.strip()


def resolve_program(value):
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    found = shutil.which(value)
    if not found:
        raise ValueError(f"Executable not found: {value}")
    return str(Path(found).resolve())


def create_output(path):
    output = path.expanduser().resolve()
    output_root = OUTPUT_ROOT.resolve()
    if output == output_root or output_root not in output.parents:
        raise ValueError(f"Output must be a new child of {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    return output


def generator_environment():
    allowed = (
        "PATH", "HOME", "TMPDIR", "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    )
    return {key: os.environ[key] for key in allowed if key in os.environ}


class NDJSONGenerator:
    """One persistent native generator process speaking one JSON line at a time."""

    def __init__(self, command, cwd, stderr_path, timeout):
        self.timeout = timeout
        self.buffer = bytearray()
        self.last_raw = b""
        self.stderr_file = stderr_path.open("wb")
        try:
            self.proc = subprocess.Popen(
                command,
                cwd=cwd,
                env=generator_environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr_file,
                bufsize=0,
                start_new_session=True,
            )
        except BaseException:
            self.stderr_file.close()
            raise

    def _readline(self, deadline):
        fd = self.proc.stdout.fileno()
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GeneratorTimeout("No complete NDJSON response before timeout",
                                       bytes(self.buffer))
            readable, _, _ = select.select([fd], [], [], remaining)
            if not readable:
                raise GeneratorTimeout("No complete NDJSON response before timeout",
                                       bytes(self.buffer))
            chunk = os.read(fd, 8192)
            if not chunk:
                raise GeneratorFailure(
                    f"Generator stdout closed before a response (exit={self.proc.poll()})"
                )
            self.buffer.extend(chunk)
            if len(self.buffer) > MAX_RESPONSE_LINE_BYTES and b"\n" not in self.buffer:
                raise GeneratorFailure("Generator response line exceeds 128 KiB")
        line, _, remainder = self.buffer.partition(b"\n")
        self.buffer = bytearray(remainder)
        if len(line) > MAX_RESPONSE_LINE_BYTES:
            raise GeneratorFailure("Generator response line exceeds 128 KiB")
        return line

    def request(self, request):
        if self.proc.poll() is not None:
            raise GeneratorFailure(f"Generator exited with status {self.proc.returncode}")
        self.last_raw = b""
        payload = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_REQUEST_BYTES:
            raise GeneratorFailure("Request exceeds the 512 KiB protocol bound")
        deadline = time.monotonic() + self.timeout
        write_fd = self.proc.stdin.fileno()
        was_blocking = os.get_blocking(write_fd)
        os.set_blocking(write_fd, False)
        try:
            remaining_payload = memoryview(payload + b"\n")
            while remaining_payload:
                if self.proc.poll() is not None:
                    raise GeneratorFailure(f"Generator exited with status {self.proc.returncode}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GeneratorTimeout("Timed out writing request to generator", bytes(self.buffer))
                _, writable, _ = select.select([], [write_fd], [], remaining)
                if not writable:
                    raise GeneratorTimeout("Timed out writing request to generator", bytes(self.buffer))
                try:
                    written = os.write(write_fd, remaining_payload)
                except BlockingIOError:
                    continue
                remaining_payload = remaining_payload[written:]
        except GeneratorFailure:
            raise
        except (BrokenPipeError, OSError) as error:
            raise GeneratorFailure(f"Could not write generator request: {error}") from error
        finally:
            os.set_blocking(write_fd, was_blocking)
        raw = self._readline(deadline)
        self.last_raw = raw
        try:
            response = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GeneratorFailure(f"Generator returned invalid JSON: {error}") from error
        if not isinstance(response, dict):
            raise GeneratorFailure("Generator response must be a JSON object")
        if response.get("id") != request["id"]:
            raise GeneratorFailure(
                f"Generator response id mismatch: expected {request['id']!r}"
            )
        answer = response.get("answer")
        tokens = response.get("tokens")
        stop = response.get("stop")
        if not isinstance(answer, str):
            raise GeneratorFailure("Generator response answer must be a string")
        try:
            answer_bytes = answer.encode("utf-8")
        except UnicodeEncodeError as error:
            raise GeneratorFailure("Generator answer contains invalid Unicode") from error
        if len(answer_bytes) > MAX_ANSWER_BYTES:
            raise GeneratorFailure("Generator answer exceeds 64 KiB")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            raise GeneratorFailure("Generator response tokens must be a nonnegative integer")
        if stop not in ("eos", "limit"):
            raise GeneratorFailure("Generator response stop must be 'eos' or 'limit'")
        return raw, response

    def close(self):
        if self.proc.poll() is not None:
            result = self.proc.wait()
            self.stderr_file.close()
            return result
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            result = self.proc.wait(timeout=5)
            self.stderr_file.close()
            return result
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            result = self.proc.wait(timeout=5)
            self.stderr_file.close()
            return result
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            result = self.proc.wait(timeout=5)
            self.stderr_file.close()
            return result


def run_bounded(command, cwd, env, timeout):
    """Run a local read-only tool, terminating its process group on timeout."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
        return process.returncode, clean_text(output), False
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            output, _ = process.communicate()
        partial = clean_text(error.output or b"")
        if output:
            partial = clean_text(output)
        return process.returncode, partial, True


def stage_result(status, diagnostic="", **extra):
    result = {"status": status}
    if diagnostic:
        result["diagnostic"] = diagnostic
    result.update(extra)
    return result


def format_replacement(case, answer, replay):
    """Validate response framing and apply the replacement through fragments.prepare."""
    try:
        _, regions = fragments.fragments(case)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        return stage_result("infrastructure", str(error)), None
    try:
        fragments.replacements(answer, len(regions))
    except ValueError as error:
        return stage_result("failed", str(error)), None

    try:
        fragments.prepare(case, answer, replay)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        return stage_result("infrastructure", str(error)), None
    return stage_result("passed"), replay / case["source"]


def read_status(path):
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if not value.isdecimal():
        return None
    return int(value)


def result_for_runner(case, returncode, runner_timed_out, logs):
    """Read the disposable runner's separate isolation/compile/test results."""
    runner_output = logs / "runner.log"
    runner_text = runner_output.read_text(encoding="utf-8", errors="replace") if runner_output.exists() else ""
    if runner_timed_out:
        return {
            "overall": "timeout",
            "compile": stage_result("not_run"),
            "tests": stage_result("not_run"),
            "diagnostic": bounded_diagnostic(runner_text or "Disposable runner timed out"),
            "infrastructure": True,
        }
    if returncode != 0:
        return {
            "overall": "infrastructure",
            "compile": stage_result("not_run"),
            "tests": stage_result("not_run"),
            "diagnostic": bounded_diagnostic(runner_text or f"Runner exited with {returncode}"),
            "infrastructure": True,
        }

    isolation = read_status(logs / "isolation.status")
    compile_status = read_status(logs / "compile.status")
    test_status = read_status(logs / "test.status")
    if isolation != 0 or compile_status is None:
        detail = runner_text
        if (logs / "isolation.log").exists():
            detail += "\n" + (logs / "isolation.log").read_text(encoding="utf-8", errors="replace")
        return {
            "overall": "infrastructure",
            "compile": stage_result("not_run"),
            "tests": stage_result("not_run"),
            "diagnostic": bounded_diagnostic(detail or "Isolation or compile status is missing"),
            "infrastructure": True,
        }

    compile_log = (logs / "compile.log").read_text(encoding="utf-8", errors="replace") \
        if (logs / "compile.log").exists() else ""
    if compile_status in TIMEOUT_CODES:
        return {
            "overall": "timeout",
            "compile": stage_result("timeout", bounded_diagnostic(compile_log), exit_code=compile_status),
            "tests": stage_result("not_run"),
            "diagnostic": bounded_diagnostic(compile_log or "Compilation timed out"),
            "infrastructure": False,
        }
    if compile_status in INFRASTRUCTURE_CODES:
        return {
            "overall": "infrastructure",
            "compile": stage_result("infrastructure", bounded_diagnostic(compile_log),
                                     exit_code=compile_status),
            "tests": stage_result("not_run"),
            "diagnostic": bounded_diagnostic(compile_log or
                                              f"Compilation ended with infrastructure status {compile_status}"),
            "infrastructure": True,
        }
    if compile_status != 0:
        return {
            "overall": "compile_failed",
            "compile": stage_result("failed", bounded_diagnostic(compile_log), exit_code=compile_status),
            "tests": stage_result("not_run"),
            "diagnostic": bounded_diagnostic(compile_log or f"Compilation exited with {compile_status}"),
            "infrastructure": False,
        }
    if test_status is None:
        return {
            "overall": "infrastructure",
            "compile": stage_result("passed"),
            "tests": stage_result("not_run"),
            "diagnostic": "Compilation succeeded but the runner produced no test status",
            "infrastructure": True,
        }

    test_log = (logs / "test.log").read_text(encoding="utf-8", errors="replace") \
        if (logs / "test.log").exists() else ""
    if test_status in TIMEOUT_CODES:
        return {
            "overall": "timeout",
            "compile": stage_result("passed"),
            "tests": stage_result("timeout", bounded_diagnostic(test_log), exit_code=test_status),
            "diagnostic": bounded_diagnostic(test_log or "Regression tests timed out"),
            "infrastructure": False,
        }
    if test_status in INFRASTRUCTURE_CODES:
        return {
            "overall": "infrastructure",
            "compile": stage_result("passed"),
            "tests": stage_result("infrastructure", bounded_diagnostic(test_log),
                                   exit_code=test_status),
            "diagnostic": bounded_diagnostic(test_log or
                                              f"Tests ended with infrastructure status {test_status}"),
            "infrastructure": True,
        }

    results = list(pilot.RESULT.finditer(test_log))
    if len(results) != 1:
        return {
            "overall": "tests_failed",
            "compile": stage_result("passed"),
            "tests": stage_result("failed", exit_code=test_status),
            "diagnostic": bounded_diagnostic(
                test_log or "The compiled regression executable produced no test result"
            ),
            "infrastructure": False,
        }
    counts = tuple(map(int, results[0].groups()))
    expected = (case["examples"], 0, 0, 0)
    if test_status != 0 or counts != expected:
        return {
            "overall": "tests_failed",
            "compile": stage_result("passed"),
            "tests": stage_result("failed", bounded_diagnostic(test_log),
                                  exit_code=test_status, counts=counts, expected=expected),
            "diagnostic": bounded_diagnostic(test_log or f"Expected {expected}, got {counts}"),
            "infrastructure": False,
        }
    return {
        "overall": "pass",
        "compile": stage_result("passed"),
        "tests": stage_result("passed", counts=counts),
        "diagnostic": "",
        "infrastructure": False,
    }


def repair_feedback(stage, diagnostic):
    detail = bounded_diagnostic(diagnostic)
    message = (
        f"The previous replacement failed at the {stage} stage. Return the complete "
        "replacement code block(s) again, in the requested order. Do not return a diff."
    )
    if detail:
        message += f"\n\nBounded tool diagnostic:\n{detail}"
    return message


def render_markdown(report):
    lines = [
        f"# Crystal QLoRA held-out evaluation: {report['arm']}",
        "",
        f"Started: {report['started_at']}  ",
        f"Repairs allowed: {report['repairs']}  ",
        f"Model: {report['model']}  ",
        f"Adapter: {report['adapter']}",
        "",
        "| Case | Attempts | Replacement / format | Compile | Tests | Result |",
        "|---|---:|---|---|---|---|",
    ]
    for case in report["cases"]:
        attempts = case["attempts"]
        final = attempts[-1] if attempts else {}
        replacement = final.get("replacement_format", {}).get("status", "not_run")
        compile_status = final.get("compile", {}).get("status", "not_run")
        test_status = final.get("tests", {}).get("status", "not_run")
        lines.append(
            f"| {case['id']} | {len(attempts)} | {replacement} | {compile_status} | "
            f"{test_status} | {case['result']} |"
        )
    lines.extend(("", "## Summary", ""))
    for result, count in sorted(report["summary"].items()):
        lines.append(f"- {result}: {count}")
    lines.append("")
    if report.get("infrastructure_diagnostic"):
        lines.extend(("## Infrastructure diagnostic", "", report["infrastructure_diagnostic"], ""))
    return "\n".join(lines)


def save_report(output, report):
    atomic_json(output / "report.json", report)
    write_text(output / "report.md", render_markdown(report))


def run(args):
    binary = resolve_program(args.binary)
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"Model file does not exist: {model}")
    adapter = args.adapter
    if adapter != "-":
        adapter_path = Path(adapter).expanduser().resolve()
        if not adapter_path.is_file():
            raise ValueError(f"Adapter file does not exist: {adapter_path}")
        adapter = str(adapter_path)
    runner = Path(args.replay_runner).expanduser().resolve()
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise ValueError(f"Disposable replay runner is not executable: {runner}")

    cases, _ = fragments.load_cases()
    heldout = [case for case in cases if case["split"] == "test"]
    if len(heldout) != 6:
        raise ValueError(f"Expected six held-out fragment cases, found {len(heldout)}")
    records = {case["id"]: fragments.model_record(case) for case in heldout}
    for case in heldout:
        messages = records[case["id"]]["messages"]
        if not messages or any(message.get("role") != "user" for message in messages):
            raise ValueError(f"{case['id']}: held-out generator prompt must contain user messages only")

    output_arg = args.output or OUTPUT_ROOT / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    output = create_output(output_arg)
    generator_dir = output / "generator-cwd"
    generator_dir.mkdir()
    stderr_path = output / "generator.stderr.log"
    report = {
        "schema_version": 1,
        "arm": args.arm,
        "model": str(model),
        "adapter": adapter,
        "generator_binary": binary,
        "replay_runner": str(runner),
        "started_at": utc_now(),
        "repairs": args.repairs,
        "max_new_tokens": args.max_new_tokens,
        "generation_timeout_seconds": args.generation_timeout,
        "runner_timeout_seconds": args.runner_timeout,
        "replacement_format_stage": "response framing and fragments.prepare splice",
        "cases": [],
        "summary": {},
    }
    for case in heldout:
        case_dir = output / "cases" / case["id"]
        case_dir.mkdir(parents=True)
        write_text(case_dir / "prompt.json",
                   json.dumps(records[case["id"]], indent=2, ensure_ascii=False) + "\n")
        report["cases"].append({"id": case["id"], "result": "not_run", "attempts": []})
    save_report(output, report)

    command = [binary, "generate", str(model), adapter, str(args.max_new_tokens)]
    generator = None
    candidate_scratch = set()
    stop_for_infrastructure = False
    try:
        try:
            generator = NDJSONGenerator(command, generator_dir, stderr_path,
                                        args.generation_timeout)
        except OSError as error:
            report["infrastructure_diagnostic"] = str(error)
            for case_report in report["cases"]:
                case_report["result"] = "not_run_infrastructure"
            report["finished_at"] = utc_now()
            report["artifact_directory"] = str(output)
            report["summary"] = {"not_run_infrastructure": len(report["cases"])}
            save_report(output, report)
            print(f"Report: {output / 'report.md'}", flush=True)
            return 2
        for case, case_report in zip(heldout, report["cases"], strict=True):
            base_messages = [dict(message) for message in records[case["id"]]["messages"]]
            case_dir = output / "cases" / case["id"]
            case_result = "not_run"
            for attempt_number in range(1, args.repairs + 2):
                attempt_dir = case_dir / f"attempt-{attempt_number:02d}"
                attempt_dir.mkdir()
                request_id = f"{case['id']}-attempt-{attempt_number}"
                request = {"id": request_id, "messages": base_messages}
                request_path = attempt_dir / "request.json"
                atomic_json(request_path, request)
                attempt = {
                    "number": attempt_number,
                    "request_id": request_id,
                    "replacement_format": stage_result("not_run"),
                    "compile": stage_result("not_run"),
                    "tests": stage_result("not_run"),
                    "result": "infrastructure",
                }
                case_report["attempts"].append(attempt)
                try:
                    raw, response = generator.request(request)
                except GeneratorTimeout as error:
                    if error.partial:
                        write_text(attempt_dir / "response.partial", clean_text(error.partial))
                    attempt["result"] = "timeout"
                    attempt["diagnostic"] = str(error)
                    case_result = "timeout"
                    stop_for_infrastructure = True
                    report["infrastructure_diagnostic"] = str(error)
                    save_report(output, report)
                    break
                except GeneratorFailure as error:
                    if generator.last_raw:
                        write_text(attempt_dir / "response.invalid.ndjson",
                                   clean_text(generator.last_raw))
                    attempt["result"] = "infrastructure"
                    attempt["diagnostic"] = str(error)
                    case_result = "infrastructure"
                    report["infrastructure_diagnostic"] = str(error)
                    stop_for_infrastructure = True
                    save_report(output, report)
                    break

                write_text(attempt_dir / "response.ndjson", raw.decode("utf-8", errors="replace") + "\n")
                atomic_json(attempt_dir / "response.json", response)
                write_text(attempt_dir / "answer.txt", response["answer"])
                attempt["tokens"] = response["tokens"]
                attempt["stop"] = response["stop"]

                replay_root = Path(tempfile.mkdtemp(prefix="cogni-qlora-replay-"))
                candidate_scratch.add(replay_root)
                replay = replay_root / "replay"
                replacement, source = format_replacement(case, response["answer"], replay)
                attempt["replacement_format"] = replacement
                if replacement["status"] != "passed":
                    attempt["result"] = replacement["status"] if replacement["status"] in (
                        "timeout", "infrastructure"
                    ) else "replacement_format_failed"
                    attempt["diagnostic"] = replacement.get("diagnostic", "Replacement rejected")
                    case_result = attempt["result"]
                    shutil.rmtree(replay_root, ignore_errors=True)
                    candidate_scratch.discard(replay_root)
                    if replacement["status"] == "infrastructure":
                        report["infrastructure_diagnostic"] = attempt["diagnostic"]
                        stop_for_infrastructure = True
                    save_report(output, report)
                    if stop_for_infrastructure:
                        break
                    if attempt_number <= args.repairs:
                        base_messages.extend([
                            {"role": "assistant", "content": response["answer"]},
                            {"role": "user", "content": repair_feedback(
                                "replacement/format", attempt["diagnostic"]
                            )},
                        ])
                        continue
                    break

                logs = attempt_dir / "runner-logs"
                logs.mkdir()
                runner_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
                try:
                    runner_code, runner_output, runner_timed_out = run_bounded(
                        [str(runner), str(replay), str(logs), case.get("backend", "cpu")],
                        cwd=ROOT,
                        env=runner_env,
                        timeout=args.runner_timeout,
                    )
                except OSError as error:
                    runner_code, runner_output, runner_timed_out = 127, str(error), False
                write_text(logs / "runner.log", runner_output)
                result = result_for_runner(case, runner_code, runner_timed_out, logs)
                attempt["compile"] = result["compile"]
                attempt["tests"] = result["tests"]
                attempt["result"] = result["overall"]
                if result["diagnostic"]:
                    attempt["diagnostic"] = result["diagnostic"]
                case_result = result["overall"]
                if result["infrastructure"]:
                    report["infrastructure_diagnostic"] = result["diagnostic"]
                    stop_for_infrastructure = True
                save_report(output, report)

                # Keep the logs and prompt/response artifacts, but discard the
                # materialized regression fixture tree before the next request.
                shutil.rmtree(replay_root, ignore_errors=True)
                candidate_scratch.discard(replay_root)
                if result["overall"] == "pass" or stop_for_infrastructure:
                    break
                if attempt_number <= args.repairs:
                    stage = "compile" if result["overall"] == "compile_failed" else "tests"
                    if result["overall"] == "timeout":
                        stage = "timeout"
                    base_messages.extend([
                        {"role": "assistant", "content": response["answer"]},
                        {"role": "user", "content": repair_feedback(stage, result["diagnostic"])},
                    ])
                    continue
                break

            case_report["result"] = case_result
            save_report(output, report)
            print(f"{case['id']}: {case_result} after {len(case_report['attempts'])} attempt(s)", flush=True)
            if stop_for_infrastructure:
                break

        generator_returncode = generator.close()
        generator = None
        report["generator_exit_code"] = generator_returncode
    finally:
        if generator is not None:
            report["generator_exit_code"] = generator.close()
        for path in candidate_scratch:
            shutil.rmtree(path, ignore_errors=True)

    for case_report in report["cases"]:
        if case_report["result"] == "not_run" and not case_report["attempts"]:
            if stop_for_infrastructure:
                case_report["result"] = "not_run_infrastructure"
    report["finished_at"] = utc_now()
    report["summary"] = {}
    for case_report in report["cases"]:
        result = case_report["result"]
        report["summary"][result] = report["summary"].get(result, 0) + 1
    report["artifact_directory"] = str(output)
    save_report(output, report)
    print(f"Report: {output / 'report.md'}", flush=True)
    if report["summary"].get("infrastructure", 0) or report["summary"].get("not_run_infrastructure", 0):
        return 2
    if report["summary"].get("timeout", 0) and any(
        case["result"] == "not_run" for case in report["cases"]
    ):
        return 2
    if any(case["result"] != "pass" for case in report["cases"]):
        return 1
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, help="native-binary executable")
    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument("--adapter", default="-", help="adapter path or '-' for base weights")
    parser.add_argument("--arm", required=True, help="short label recorded in the report")
    parser.add_argument("--output", type=Path, help=f"new directory beneath {OUTPUT_ROOT}")
    parser.add_argument("--replay-runner", default=str(DEFAULT_RUNNER),
                        help="guarded disposable replay runner")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--repairs", type=int, default=2, help="bounded retries, 0 through 2")
    parser.add_argument("--generation-timeout", type=float, default=300)
    parser.add_argument("--runner-timeout", type=float, default=480)
    args = parser.parse_args(argv)
    if not 0 <= args.repairs <= 2:
        parser.error("--repairs must be between 0 and 2")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if min(args.generation_timeout, args.runner_timeout) <= 0:
        parser.error("all timeouts must be positive")
    return args


def main(argv=None):
    try:
        return run(parse_args(argv))
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        print(f"evaluation setup failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
